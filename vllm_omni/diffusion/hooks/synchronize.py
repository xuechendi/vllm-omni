# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronize hook for profiling accuracy on asynchronous backends (XPU, etc).

When profiling with torch.profiler, asynchronous dispatch (common on XPU/SYCL)
means kernel timings overlap across transformer blocks, making per-block
attribution inaccurate. This hook inserts a device synchronize barrier after
each transformer block's forward pass so that profiler traces show clean,
non-overlapping regions per block.

Each block and its direct ``nn.Module`` children are wrapped with
``torch.profiler.record_function`` events so that the trace shows both
block-level and sub-module-level regions (e.g.
``WanTransformerBlock_0``, ``WanTransformerBlock_0.attn1``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook

logger = init_logger(__name__)

_HOOK_NAME = "profiler_synchronize"
_CHILD_HOOK_NAME = "profiler_synchronize_child"


def _format_input_shapes(args: tuple) -> str:
    """Build a compact shape suffix from the first few tensor arguments."""
    shapes = []
    for a in args:
        if isinstance(a, torch.Tensor):
            shapes.append(str(tuple(a.shape)))
        if len(shapes) >= 3:
            break
    if not shapes:
        return ""
    return " [" + ", ".join(shapes) + "]"


class SynchronizeHook(ModelHook):
    """Wraps a block forward with a profiler event and synchronizes after it."""

    def __init__(self, synchronize_fn, event_name: str):
        self._synchronize_fn = synchronize_fn
        self._event_name = event_name
        self._record_fn: torch.profiler.record_function | None = None

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        label = self._event_name + _format_input_shapes(args)
        self._record_fn = torch.profiler.record_function(label)
        self._record_fn.__enter__()
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        with torch.profiler.record_function(f"{self._event_name}::synchronize"):
            self._synchronize_fn()
        if self._record_fn is not None:
            self._record_fn.__exit__(None, None, None)
            self._record_fn = None
        return output


class ChildSynchronizeHook(ModelHook):
    """Wraps a child module's forward with a profiler event and sync."""

    def __init__(self, synchronize_fn, event_name: str):
        self._synchronize_fn = synchronize_fn
        self._event_name = event_name
        self._record_fn: torch.profiler.record_function | None = None

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        label = self._event_name + _format_input_shapes(args)
        self._record_fn = torch.profiler.record_function(label)
        self._record_fn.__enter__()
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        with torch.profiler.record_function(f"{self._event_name}::synchronize"):
            self._synchronize_fn()
        if self._record_fn is not None:
            self._record_fn.__exit__(None, None, None)
            self._record_fn = None
        return output


def apply_profiler_synchronize_hooks(pipeline: nn.Module) -> int:
    """Apply synchronize hooks to all transformer blocks and their children.

    Uses ``_repeated_blocks`` (declared on every transformer model) to
    auto-discover block class names, then attaches a
    :class:`SynchronizeHook` to each matching block and a
    :class:`ChildSynchronizeHook` to each direct ``nn.Module`` child
    within that block.

    This produces a two-level trace hierarchy, e.g.::

        WanTransformerBlock_0
        ├── WanTransformerBlock_0.norm1
        ├── WanTransformerBlock_0.attn1
        ├── WanTransformerBlock_0.attn2
        ├── WanTransformerBlock_0.norm2
        ├── WanTransformerBlock_0.ffn
        ├── WanTransformerBlock_0.norm3
        └── WanTransformerBlock_0::synchronize

    Args:
        pipeline: The top-level pipeline model (e.g. Wan22I2VPipeline).

    Returns:
        Number of hooks applied (blocks + children).
    """
    from vllm_omni.platforms import current_omni_platform

    synchronize_fn = current_omni_platform.synchronize

    transformer_attrs = ["transformer", "transformer_2", "dit", "unet"]
    count = 0

    for attr in transformer_attrs:
        transformer = getattr(pipeline, attr, None)
        if transformer is None:
            continue

        repeated_blocks = getattr(transformer, "_repeated_blocks", None)
        if not repeated_blocks:
            continue

        block_index = 0
        for submod in transformer.modules():
            if submod.__class__.__name__ in repeated_blocks:
                block_name = f"{submod.__class__.__name__}_{block_index}"

                # Hook the block itself
                registry = HookRegistry.get_or_create(submod)
                registry.register_hook(_HOOK_NAME, SynchronizeHook(synchronize_fn, block_name))
                count += 1

                # Hook direct nn.Module children (attn1, attn2, ffn, norms, etc.)
                for child_name, child in submod.named_children():
                    child_event = f"{block_name}.{child_name}"
                    child_registry = HookRegistry.get_or_create(child)
                    child_registry.register_hook(
                        _CHILD_HOOK_NAME,
                        ChildSynchronizeHook(synchronize_fn, child_event),
                    )
                    count += 1

                block_index += 1

    return count


def remove_profiler_synchronize_hooks(pipeline: nn.Module) -> int:
    """Remove synchronize hooks from all transformer blocks and children.

    Returns:
        Number of hooks removed.
    """
    transformer_attrs = ["transformer", "transformer_2", "dit", "unet"]
    count = 0

    for attr in transformer_attrs:
        transformer = getattr(pipeline, attr, None)
        if transformer is None:
            continue

        repeated_blocks = getattr(transformer, "_repeated_blocks", None)
        if not repeated_blocks:
            continue

        for submod in transformer.modules():
            if submod.__class__.__name__ in repeated_blocks:
                # Remove block hook
                registry: HookRegistry | None = getattr(submod, "_hook_registry", None)
                if registry is not None and registry.get_hook(_HOOK_NAME) is not None:
                    registry.remove_hook(_HOOK_NAME)
                    count += 1

                # Remove child hooks
                for _child_name, child in submod.named_children():
                    child_registry: HookRegistry | None = getattr(child, "_hook_registry", None)
                    if child_registry is not None and child_registry.get_hook(_CHILD_HOOK_NAME) is not None:
                        child_registry.remove_hook(_CHILD_HOOK_NAME)
                        count += 1

    return count
