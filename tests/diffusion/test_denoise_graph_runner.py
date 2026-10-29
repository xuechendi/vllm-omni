# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.graph as graph_mod
import vllm_omni.diffusion.profiler.diffusion_pipeline_profiler as profiler_mod
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.graph import GraphedForward, build_denoise_graph_runner, invalidate_denoise_graphs

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0
        self.replay_fn = None

    def replay(self) -> None:
        self.replays += 1
        self.replay_fn()


class _FakePlatform:
    """Stands in for ``current_omni_platform`` graph capture on CPU.

    A call made under capture is recorded against the exact tensors it received,
    and its output is blanked, because a real capture records kernels without
    running them. Replay recomputes into that same output tensor.
    """

    device_type = "cpu"

    def __init__(self) -> None:
        self.capturing: _FakeGraph | None = None
        self.outer_capture = False
        self.graphs: list[_FakeGraph] = []

    def capture_graph(self, capture, generators=()):
        del generators
        graph = _FakeGraph()
        self.graphs.append(graph)
        self.capturing = graph
        try:
            capture()
        finally:
            self.capturing = None
        return graph

    @staticmethod
    def replay_graph(graph, eager):
        graph.replay()

    def is_graph_capturing(self) -> bool:
        return self.outer_capture or self.capturing is not None

    @staticmethod
    def is_cuda_alike() -> bool:
        return False

    def wrap(self, fn):
        def body(**kwargs):
            out = fn(**kwargs)
            graph = self.capturing
            if graph is not None:
                graph.replay_fn = lambda: out.copy_(fn(**kwargs))
                out.fill_(float("nan"))
            return out

        return body


@pytest.fixture
def fake_graphs(monkeypatch):
    fake = _FakePlatform()
    monkeypatch.setattr(graph_mod, "current_omni_platform", fake)
    monkeypatch.setattr(graph_mod, "_side_stream", lambda device: nullcontext())
    return fake


def _affine(x: torch.Tensor, mode: str = "gen_image") -> torch.Tensor:
    return x * 2 + (1 if mode == "gen_image" else -1)


def _runner(fake, fn=_affine, **kwargs) -> GraphedForward:
    kwargs.setdefault("warmup_iters", 1)
    return GraphedForward(fake.wrap(fn), device=torch.device("cpu"), name="t", **kwargs)


def test_replay_reads_fresh_inputs_and_returns_clones(fake_graphs):
    calls = []

    def fn(**kwargs):
        calls.append(kwargs)
        return _affine(**kwargs)

    runner = _runner(fake_graphs, fn)
    first = runner(x=torch.tensor([1.0, 2.0]), mode="gen_image")
    torch.testing.assert_close(first, torch.tensor([3.0, 5.0]))
    calls_after_capture = len(calls)

    second = runner(x=torch.tensor([10.0, 20.0]), mode="gen_image")

    torch.testing.assert_close(second, torch.tensor([21.0, 41.0]))
    # Clones: CFG branches replaying the same graph cannot overwrite each other.
    torch.testing.assert_close(first, torch.tensor([3.0, 5.0]))
    assert second.data_ptr() != first.data_ptr()
    assert len(calls) == calls_after_capture + 1  # the replay, not an eager call
    assert runner.stats() == {"graphs": 1, "pinned": 0, "captures": 1, "replays": 2, "evictions": 0, "eager_calls": 0}


def test_nested_and_list_outputs_are_refreshed(fake_graphs):
    def fn(xs):
        return [xs[0] + xs[1], xs[0] - xs[1]]

    def wrapped(**kwargs):
        outs = fn(**kwargs)
        graph = fake_graphs.capturing
        if graph is not None:
            graph.replay_fn = lambda: [o.copy_(n) for o, n in zip(outs, fn(**kwargs))]
        return outs

    runner = GraphedForward(wrapped, device=torch.device("cpu"), warmup_iters=0)
    runner(xs=[torch.ones(2), torch.ones(2)])

    out = runner(xs=[torch.full((2,), 3.0), torch.full((2,), 4.0)])

    torch.testing.assert_close(out[0], torch.full((2,), 7.0))
    torch.testing.assert_close(out[1], torch.full((2,), -1.0))
    assert runner.replays == 2 and runner.captures == 1


def test_new_shape_flag_or_variant_captures_separately_with_lru_eviction(fake_graphs):
    runner = _runner(fake_graphs, max_entries=2)
    runner(x=torch.zeros(2))
    runner(x=torch.zeros(2), mode="gen_text")
    runner(x=torch.zeros(2), variant="lora_on")
    out = runner(x=torch.ones(3))

    torch.testing.assert_close(out, torch.full((3,), 3.0))
    assert runner.stats()["captures"] == 4 and runner.num_graphs == 2 and runner.stats()["evictions"] == 2


def test_outer_capture_runs_eager(fake_graphs):
    runner = _runner(fake_graphs)
    fake_graphs.outer_capture = True

    torch.testing.assert_close(runner(x=torch.ones(2)), torch.full((2,), 3.0))
    assert runner.enabled and runner.eager_calls == 1 and runner.num_graphs == 0


def test_capture_failure_disables_and_releases_earlier_graphs(fake_graphs, monkeypatch):
    runner = _runner(fake_graphs)
    runner(x=torch.ones(2))
    assert runner.num_graphs == 1

    def failing_capture(capture, generators=()):
        raise RuntimeError("UR_RESULT_ERROR_UNSUPPORTED_FEATURE")

    monkeypatch.setattr(fake_graphs, "capture_graph", failing_capture)
    out = runner(x=torch.ones(3))

    torch.testing.assert_close(out, torch.full((3,), 3.0))
    assert not runner.enabled and runner.num_graphs == 0


def test_platform_without_graphs_disables_runner(fake_graphs, monkeypatch):
    monkeypatch.setattr(fake_graphs, "capture_graph", lambda capture, generators=(): None)
    runner = _runner(fake_graphs)

    torch.testing.assert_close(runner(x=torch.ones(2)), torch.full((2,), 3.0))
    assert not runner.enabled


def test_device_off_the_platform_never_captures(fake_graphs):
    runner = GraphedForward(_affine, device=torch.device("meta"))
    assert not runner.enabled


def test_profiler_skips_synchronize_during_capture(monkeypatch):
    for capturing, expected_syncs in ((False, 1), (True, 0)):
        syncs = []
        monkeypatch.setattr(
            profiler_mod,
            "current_omni_platform",
            SimpleNamespace(
                is_available=lambda: True,
                is_graph_capturing=lambda c=capturing: c,
                synchronize=lambda: syncs.append(1),
            ),
        )
        profiler_mod._sync_if_useful()
        assert len(syncs) == expected_syncs


def _od_config(**overrides):
    fields = dict(
        diffusion_use_graph=True,
        enforce_eager=False,
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        cache_backend="none",
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _build(fake, od_config, **kwargs):
    return build_denoise_graph_runner(od_config, fake.wrap(_affine), "T", device=torch.device("cpu"), **kwargs)


def test_build_runner_when_requested_and_safe(fake_graphs):
    runner = _build(fake_graphs, _od_config())
    assert isinstance(runner, GraphedForward)
    assert runner.name == "T" and runner.max_entries == 2 and runner.warmup_iters == 1


_GATED = AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax={"disabled_until_timestep": 0.5}))


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"diffusion_use_graph": False}, None),
        ({"enforce_eager": True}, None),
        ({"diffusion_kv_mode": DiffusionKVCacheMode.PAGED_SCHEDULER}, None),
        ({"diffusion_kv_mode": "paged_scheduler"}, None),
        ({"cache_backend": "cache_dit"}, None),
        ({"enable_cpu_offload": True}, None),
        ({"enable_layerwise_offload": True}, None),
        ({"enable_distributed_layerwise_offload": True}, None),
        ({"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}}, None),
        ({"parallel_config": SimpleNamespace(pipeline_parallel_size=2)}, None),
        ({"diffusion_kv_cache_skip_step_indices": {0, 3}}, None),
        ({"diffusion_attention_config": _GATED}, None),
        ({}, "pipeline-specific reason"),
    ],
)
def test_build_runner_stays_eager_when_unsafe(fake_graphs, overrides, refusal):
    assert _build(fake_graphs, _od_config(**overrides), refusal=refusal) is None


def test_build_runner_defaults_to_the_local_device(fake_graphs, monkeypatch):
    import vllm_omni.diffusion.distributed.utils as dist_utils

    monkeypatch.setattr(dist_utils, "get_local_device", lambda: torch.device("cpu"))
    runner = build_denoise_graph_runner(_od_config(), _affine, "T")
    assert runner is not None and runner.device == torch.device("cpu")


def test_invalidate_denoise_graphs_drops_only_denoise_runners(fake_graphs):
    runners = [_build(fake_graphs, _od_config()) for _ in range(2)]
    other = _runner(fake_graphs)  # e.g. a model-owned graph that keys its own LoRA state
    for runner in (*runners, other):
        runner(x=torch.ones(2))

    invalidate_denoise_graphs("adapter changed")

    for runner in runners:
        assert runner.enabled and runner.num_graphs == 0
        torch.testing.assert_close(runner(x=torch.zeros(2)), torch.full((2,), 1.0))
        assert runner.captures == 2
    assert other.num_graphs == 1


@pytest.mark.parametrize(
    ("use_graph", "has_runner", "warns"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_warn_if_denoise_graph_unsupported(monkeypatch, use_graph, has_runner, warns):
    warnings = []
    monkeypatch.setattr(graph_mod.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    pipeline = SimpleNamespace(_denoise_graph_runner=None) if has_runner else SimpleNamespace()

    graph_mod.warn_if_denoise_graph_unsupported(SimpleNamespace(diffusion_use_graph=use_graph), pipeline)

    assert bool(warnings) is warns
