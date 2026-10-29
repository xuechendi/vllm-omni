# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DenoiseGraphRunner against real device graphs (CUDA graphs / XPU graphs)."""

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.graph import GraphedForward
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _body() -> tuple[nn.Module, torch.device]:
    device = torch.device(current_omni_platform.device_type, 0)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)).to(device, torch.bfloat16)
    return model, device


@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@torch.inference_mode()
def test_replay_matches_eager_across_eviction_and_invalidation():
    model, device = _body()

    def fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x) * t

    runner = GraphedForward(fn, device=device, name="device", max_entries=2, warmup_iters=1)
    # Three shapes through a two-graph LRU, then back to the first: every later
    # capture happens after an earlier graph was freed. XPU needs a private pool
    # per graph for that (a shared pool asserts in the caching allocator).
    for rows in (64, 96, 128, 64, 64):
        x = torch.randn(rows, 256, device=device, dtype=torch.bfloat16)
        t = torch.rand(1, device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(runner(x=x, t=t), fn(x, t), rtol=0, atol=0)
    runner.clear()
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    t = torch.rand(1, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(runner(x=x, t=t), fn(x, t), rtol=0, atol=0)

    assert runner.enabled
    assert runner.stats() == {"graphs": 1, "pinned": 0, "captures": 5, "replays": 6, "evictions": 2, "eager_calls": 0}
