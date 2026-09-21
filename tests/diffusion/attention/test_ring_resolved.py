# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the per-rank [RING_RESOLVED] self-evidence line."""

import pytest
import torch.distributed

from vllm_omni.diffusion.attention.backends import ring_flash_attn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _fresh_dedup():
    saved = set(ring_flash_attn._RING_RESOLVED_SEEN)
    ring_flash_attn._RING_RESOLVED_SEEN.clear()
    yield
    ring_flash_attn._RING_RESOLVED_SEEN.clear()
    ring_flash_attn._RING_RESOLVED_SEEN.update(saved)


def _fake_group(monkeypatch, rank, world):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world)


def test_ring_resolved_prints_values_once(monkeypatch, capsys):
    _fake_group(monkeypatch, 1, 2)
    ring_flash_attn._ring_resolved_selfproof(object(), "bf16")
    ring_flash_attn._ring_resolved_selfproof(object(), "bf16")
    err = capsys.readouterr().err
    assert err.count("[RING_RESOLVED]") == 1, err
    assert "rank1" in err and "precision=bf16" in err and "world=2" in err


def test_ring_resolved_distinct_precisions_each_print(monkeypatch, capsys):
    _fake_group(monkeypatch, 0, 2)
    ring_flash_attn._ring_resolved_selfproof(object(), "bf16")
    ring_flash_attn._ring_resolved_selfproof(object(), "fp8_pertensor")
    err = capsys.readouterr().err
    assert err.count("[RING_RESOLVED]") == 2, err
    assert "precision=fp8_pertensor" in err


def test_ring_resolved_unknown_group_says_unknown(monkeypatch, capsys):
    def _boom(group=None):
        raise RuntimeError("not initialised")

    monkeypatch.setattr(torch.distributed, "get_rank", _boom)
    monkeypatch.setattr(torch.distributed, "get_world_size", _boom)
    ring_flash_attn._ring_resolved_selfproof(object(), "bf16")
    err = capsys.readouterr().err
    assert "rankUNKNOWN" in err and "world=UNKNOWN" in err
