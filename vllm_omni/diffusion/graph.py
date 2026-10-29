# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-graph replay of fixed-shape diffusion forwards (CUDA graphs, XPU graphs).

``GraphedForward`` wraps a keyword-only forward. Calls are keyed by the
*structure* of their kwargs (tensor shapes/dtypes and every non-tensor value);
the first call of a key captures a graph on static copies of the tensors and
later calls copy the live tensors into those buffers and replay. Values that
only change per call are therefore ordinary inputs and never force a
re-capture; only a new *shape* does.

Capture and replay go through ``current_omni_platform.capture_graph`` /
``replay_graph``, so the device specifics live in one place per platform (CUDA
shares vLLM's graph pool; XPU gives every graph a private pool). Graph capture
forbids host synchronization inside the captured region. On CUDA the very first
capture is validated with ``torch.cuda.set_sync_debug_mode("error")``, so a
synchronizing operator fails loudly on every rank at the same program point
instead of leaving peers waiting in a collective. Any failure disables the
wrapper for the rest of the process (agreed across ``group`` when one is given)
and the caller falls back to eager execution.

``build_denoise_graph_runner`` is the opt-in for diffusion denoise steps
(``diffusion_use_graph``): it refuses configurations in which a step reads
state the graph cannot see, so a replay could silently diverge from eager.
"""

from __future__ import annotations

import dataclasses
import traceback
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_PRIMITIVES = (int, float, bool, str, bytes, type(None))


def kwargs_signature(value: Any) -> Any:
    """A hashable description of ``value`` in which tensors count by shape and dtype only."""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), value.device.type)
    if isinstance(value, dict):
        return ("dict", tuple((str(key), kwargs_signature(value[key])) for key in sorted(value, key=str)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(kwargs_signature(item) for item in value))
    if isinstance(value, _PRIMITIVES):
        return ("value", type(value).__name__, value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple((field.name, kwargs_signature(getattr(value, field.name))) for field in dataclasses.fields(value)),
        )
    return ("repr", type(value).__qualname__, repr(value))


def _holds_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_holds_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_tensor(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(_holds_tensor(getattr(value, field.name)) for field in dataclasses.fields(value))
    return False


def clone_static(value: Any) -> Any:
    """Deep-copy the tensors of a kwargs tree into fresh contiguous buffers.

    Tensors are reached through dicts, lists and tuples only. Any other object
    (a layout dataclass, a device, a dtype) is passed through unchanged and
    must not hold a tensor: it would keep its capture-time address without
    being refreshed on replay, so such objects are rejected.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict):
        return {key: clone_static(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_static(item) for item in value)
    if isinstance(value, list):
        return [clone_static(item) for item in value]
    if isinstance(value, (*_PRIMITIVES, torch.device, torch.dtype, torch.Size)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and not _holds_tensor(value):
        return value
    raise TypeError(
        "graph kwargs may hold tensors inside dicts, lists and tuples, primitives, torch.device/dtype/Size and "
        f"tensor-free dataclasses only, got {type(value)!r}"
    )


def copy_into(static: Any, current: Any) -> int:
    """Copy the tensors of ``current`` into the matching buffers of ``static``.

    Returns the number of tensors copied. The trees must share one signature.
    """
    if isinstance(static, torch.Tensor):
        if not isinstance(current, torch.Tensor) or static.shape != current.shape or static.dtype != current.dtype:
            raise ValueError("graph input differs in shape or dtype from the captured buffer")
        if static.data_ptr() != current.data_ptr() or static.stride() != current.stride():
            static.copy_(current)
        return 1
    if isinstance(static, dict):
        if not isinstance(current, dict) or static.keys() != current.keys():
            raise ValueError("graph input keys differ from the captured kwargs")
        return sum(copy_into(static[key], current[key]) for key in static)
    if isinstance(static, (list, tuple)):
        if not isinstance(current, (list, tuple)) or len(static) != len(current):
            raise ValueError("graph input sequence differs from the captured kwargs")
        return sum(copy_into(s, c) for s, c in zip(static, current, strict=True))
    return 0


@dataclass
class _Entry:
    graph: Any
    static_kwargs: dict[str, Any]
    outputs: Any
    # Objects whose tensors the captured kernels read by address although
    # they are not among the kwargs (for example an embedding plan installed
    # through a context variable). The entry owns a reference so the memory
    # cannot be freed and reused while the graph is resident.
    keep_alive: tuple[Any, ...] = ()
    # Pinned entries (captured up front for a configured shape range) are
    # never evicted by later, unpinned shapes.
    pinned: bool = False


@contextmanager
def _sync_debug_mode(mode: str) -> Iterator[None]:
    """Make host synchronization raise. CUDA only; other platforms run the probe as-is."""
    if not current_omni_platform.is_cuda_alike():
        yield
        return
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(mode)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@contextmanager
def _side_stream(device: torch.device) -> Iterator[None]:
    """Run the warm-up forwards on a side stream, ordered after the current work."""
    current = torch.accelerator.current_stream(device)
    side = torch.Stream(device=device)
    side.wait_stream(current)
    with side:
        yield
    current.wait_stream(side)


class GraphedForward:
    """Capture-and-replay wrapper around a keyword-only forward."""

    def __init__(
        self,
        module: Callable[..., Any],
        *,
        device: torch.device,
        max_entries: int = 16,
        warmup_iters: int = 2,
        group: torch.distributed.ProcessGroup | None = None,
        name: str = "graph",
        enabled: bool = True,
        capture_context: Callable[[], AbstractContextManager] | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.module = module
        self.device = torch.device(device)
        self.max_entries = int(max_entries)
        self.warmup_iters = int(warmup_iters)
        self.group = group
        self.name = name
        self.enabled = bool(enabled) and self.device.type == current_omni_platform.device_type
        # Entered around the probe, warm-up and capture forwards, e.g. to bypass a
        # host-keyed cache that would otherwise synchronize under capture.
        self.capture_context = capture_context or nullcontext
        self._entries: OrderedDict[Any, _Entry] = OrderedDict()
        self._probe_pending = True
        self._pinned_overflow_warned = False
        self.captures = 0
        self.replays = 0
        self.evictions = 0
        self.eager_calls = 0

    # -- public --------------------------------------------------------------

    def __call__(
        self, *, variant: Any = None, keep_alive: tuple[Any, ...] = (), pin: bool = False, **kwargs: Any
    ) -> Any:
        """Run ``module(**kwargs)``, by replay when a graph of this shape exists.

        ``variant`` distinguishes module states the kwargs do not show (for
        example whether the LoRA hooks are enabled, or which embedding plan is
        installed); ``keep_alive`` holds every object outside ``kwargs`` whose
        tensors the forward reads, so a resident graph keeps them allocated;
        ``pin`` exempts a newly captured graph from LRU eviction. Returned
        tensors are clones, so the caller may hold them across the next call.
        """
        if not self.enabled or current_omni_platform.is_graph_capturing():
            # Inside someone else's capture the outer graph records these kernels anyway.
            self.eager_calls += 1
            return self.module(**kwargs)
        key = (kwargs_signature(variant), kwargs_signature(kwargs))
        entry = self._entries.get(key)
        if entry is None:
            entry = self._capture(key, kwargs, keep_alive, variant, pin=pin)
            if entry is None:
                self.eager_calls += 1
                return self.module(**kwargs)
        else:
            entry.pinned = entry.pinned or pin
            copy_into(entry.static_kwargs, kwargs)
            self._entries.move_to_end(key)
        current_omni_platform.replay_graph(entry.graph, lambda: None)
        self.replays += 1
        return _clone_outputs(entry.outputs)

    @property
    def num_graphs(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        return {
            "graphs": len(self._entries),
            "pinned": sum(1 for entry in self._entries.values() if entry.pinned),
            "captures": self.captures,
            "replays": self.replays,
            "evictions": self.evictions,
            "eager_calls": self.eager_calls,
        }

    def clear(self) -> None:
        """Release captured graphs before the buffers they read off-argument are freed."""
        self._entries.clear()

    def disable(self, reason: str) -> None:
        if self.enabled:
            logger.warning("%s device graphs disabled: %s", self.name, reason)
        self.enabled = False
        self.clear()

    # -- capture -------------------------------------------------------------

    def _capture(
        self,
        key: Any,
        kwargs: dict[str, Any],
        keep_alive: tuple[Any, ...] = (),
        variant: Any = None,
        *,
        pin: bool = False,
    ) -> _Entry | None:
        failure: str | None = None
        static: dict[str, Any] = {}
        graph: Any = None
        captured: list[Any] = []
        try:
            static = clone_static(kwargs)
            with self.capture_context():
                if self._probe_pending:
                    # A synchronizing operator raises here on every rank at
                    # the same point, before any capture starts.
                    with _sync_debug_mode("error"):
                        self.module(**static)
                    self._probe_pending = False
                # The first capture warms lazy kernel/library state for the
                # whole process; later shapes need a single warm forward.
                warmups = self.warmup_iters if self.captures == 0 else min(self.warmup_iters, 1)
                if warmups:
                    with _side_stream(self.device):
                        for _ in range(warmups):
                            self.module(**static)
                graph = current_omni_platform.capture_graph(lambda: captured.append(self.module(**static)))
            if graph is None:
                failure = f"{type(current_omni_platform).__name__} cannot capture device graphs"
        except Exception as exc:  # noqa: BLE001 - any capture failure means eager from now on
            failure = f"{type(exc).__name__}: {exc}"
            logger.warning("%s device graph capture failed:\n%s", self.name, traceback.format_exc())
        ok = self._agree(failure is None)
        if not ok:
            del graph, captured, static
            self.disable(failure or "a peer rank failed to capture")
            return None
        if len(self._entries) >= self.max_entries:
            victim = next((k for k, e in self._entries.items() if not e.pinned), None)
            if victim is None:
                if not self._pinned_overflow_warned:
                    self._pinned_overflow_warned = True
                    logger.warning(
                        "%s: all %d resident device graphs are pinned; growing past max_entries",
                        self.name,
                        len(self._entries),
                    )
            else:
                del self._entries[victim]
                self.evictions += 1
        entry = _Entry(graph=graph, static_kwargs=static, outputs=captured[0], keep_alive=tuple(keep_alive), pinned=pin)
        self._entries[key] = entry
        self.captures += 1
        logger.info("%s device graph captured for %s (%d resident)", self.name, variant, len(self._entries))
        return entry

    def _agree(self, ok: bool) -> bool:
        group = self.group
        if group is None or torch.distributed.get_world_size(group) <= 1:
            return ok
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=self.device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
        return bool(int(flag.item()))


def _clone_outputs(outputs: Any) -> Any:
    if isinstance(outputs, torch.Tensor):
        return outputs.clone()
    if isinstance(outputs, tuple):
        return tuple(_clone_outputs(item) for item in outputs)
    if isinstance(outputs, list):
        return [_clone_outputs(item) for item in outputs]
    if isinstance(outputs, dict):
        return {key: _clone_outputs(item) for key, item in outputs.items()}
    return outputs


# -- denoise steps -------------------------------------------------------------

# Runners built for denoise steps, so state owned outside the pipeline (the active
# LoRA adapter) can drop the graphs that froze it.
_DENOISE_RUNNERS: weakref.WeakSet[GraphedForward] = weakref.WeakSet()


def denoise_graph_refusal(od_config: Any) -> str | None:
    """Why a captured denoise step could diverge from eager under ``od_config``, or None.

    A replay re-runs the kernels the capture step chose. Anything a step reads
    outside its arguments -- the forward context's step index or timestep, a
    per-request KV layout, weights moved by hooks -- is frozen at capture time.
    """
    # Lazy: the offloader package imports diffusion.data, which imports the LoRA
    # manager, which imports this module.
    from vllm_omni.diffusion.offloader.config import offload_enabled

    kv_mode = DiffusionKVCacheMode(getattr(od_config, "diffusion_kv_mode", None) or DiffusionKVCacheMode.DENSE_LEGACY)
    cache_backend = getattr(od_config, "cache_backend", None) or "none"
    parallel_config = getattr(od_config, "parallel_config", None)
    attention_config = getattr(od_config, "diffusion_attention_config", None)
    attention_specs = [getattr(attention_config, "default", None), *getattr(attention_config, "per_role", {}).values()]
    if getattr(od_config, "enforce_eager", False):
        return "enforce_eager=True"
    if kv_mode is not DiffusionKVCacheMode.DENSE_LEGACY:
        return f"diffusion_kv_mode={kv_mode.value!r} reads per-request paged KV state outside the step arguments"
    if cache_backend != "none":
        return f"cache_backend={cache_backend!r} decides per step whether the transformer runs at all"
    if offload_enabled(od_config):
        return "CPU offload moves the weights that a captured graph reads"
    if getattr(parallel_config, "pipeline_parallel_size", 1) > 1:
        return "pipeline parallelism hands intermediate tensors between stages"
    if getattr(od_config, "diffusion_kv_cache_skip_step_indices", None):
        return "diffusion_kv_cache_skip_steps picks the attention KV dtype from the step index"
    if any(getattr(getattr(spec, "skip_softmax", None), "disabled_until_timestep", 0) for spec in attention_specs):
        return "skip-softmax disabled_until_timestep picks dense or sparse attention from the timestep"
    return None


def build_denoise_graph_runner(
    od_config: Any,
    module: Callable[..., Any],
    name: str,
    *,
    device: torch.device | None = None,
    refusal: str | None = None,
    max_entries: int = 2,
) -> GraphedForward | None:
    """Return a denoise-step runner if ``od_config`` opts in and nothing makes a step unsafe to replay.

    ``refusal`` lets the pipeline add its own reason to stay eager. ``device``
    defaults to this rank's accelerator (pipelines build runners before their
    weights are on it).
    """
    requested = bool(getattr(od_config, "diffusion_use_graph", False))
    # Stated either way: a silent None is indistinguishable from "engaged and did not pay".
    logger.info("%s: diffusion_use_graph=%s", name, requested)
    if not requested:
        return None
    refusal = refusal or denoise_graph_refusal(od_config)
    if refusal is not None:
        logger.warning("%s: diffusion_use_graph ignored because %s", name, refusal)
        return None
    if device is None:
        from vllm_omni.diffusion.distributed.utils import get_local_device

        device = get_local_device()
    runner = GraphedForward(module, device=device, max_entries=max_entries, warmup_iters=1, name=name)
    if not runner.enabled:
        logger.warning("%s: diffusion_use_graph ignored because %s has no device graphs", name, device)
        return None
    _DENOISE_RUNNERS.add(runner)
    return runner


def invalidate_denoise_graphs(reason: str) -> None:
    """Drop every captured denoise graph in this process; later steps recapture.

    For state a graph froze that lives outside the step arguments and outside
    the pipeline, such as which LoRA adapter is active: its Python branches ran
    once at capture time, so a replay keeps the capture-time adapter.
    """
    for runner in list(_DENOISE_RUNNERS):
        if runner.num_graphs:
            logger.debug("%s: dropping %d device graph(s) (%s)", runner.name, runner.num_graphs, reason)
        runner.clear()


def warn_if_denoise_graph_unsupported(od_config: Any, pipeline: Any) -> None:
    """Say so when ``diffusion_use_graph`` is set on a pipeline that never builds a runner.

    Pipelines opt in by assigning ``self._denoise_graph_runner`` (``None`` when refused, which
    :func:`build_denoise_graph_runner` already reports); anything else would ignore the flag silently.
    """
    if getattr(od_config, "diffusion_use_graph", False) and not hasattr(pipeline, "_denoise_graph_runner"):
        logger.warning(
            "diffusion_use_graph=True is ignored: %s does not support denoise graphs; running eager.",
            type(pipeline).__name__,
        )


__all__ = [
    "GraphedForward",
    "build_denoise_graph_runner",
    "clone_static",
    "copy_into",
    "denoise_graph_refusal",
    "invalidate_denoise_graphs",
    "kwargs_signature",
    "warn_if_denoise_graph_unsupported",
]
