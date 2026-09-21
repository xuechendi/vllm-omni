# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The XPU ring entry points must honor, or loudly reject, the model's packed
valid prefix on every entry -- not only on the selector-preconverted one.

* ``ring_flash_attn_forward`` documents a global ``valid_kv_length``; the
  SYCL/TLA branch used to accept it and silently drop it, so the kernel
  attended to the packed suffix padding.
* ``select_xpu_ring_attn_kwargs`` for KLOX_SAGE_V3_HYBRID guarded only the
  sequence-parallel padding; a packed valid prefix (which the kernel has no
  valid_kv_lens argument to honor) slipped through the same way.
"""

import types
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.backends import ring_flash_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.ring import ring_attn_utils
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass(frozen=True)
class _FakeSpGroup:
    ring_world_size: int = 2
    ring_group: object = None


class _XpuProxy:
    """A CPU tensor that claims device.type == 'xpu' for the device gate."""

    def __init__(self, t: torch.Tensor) -> None:
        object.__setattr__(self, "_t", t)

    @property
    def device(self):
        return types.SimpleNamespace(type="xpu")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_t"), name)


@pytest.fixture
def sycl_kernel_spy(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        return torch.zeros(tuple(q.shape)), torch.zeros(q.shape[0], q.shape[2], q.shape[1])

    monkeypatch.setattr(ring_flash_attn, "HAS_SYCL_TLA_RING_ATTN", True)
    monkeypatch.setattr(ring_flash_attn, "sycl_tla_ring_attn_forward", fake_kernel)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    return calls


def _local_qkv(block_len=4):
    t = torch.randn(1, block_len, 2, 8)
    return _XpuProxy(t), _XpuProxy(t), _XpuProxy(t)


def test_sycl_direct_call_converts_the_global_prefix(sycl_kernel_spy):
    q, k, v = _local_qkv()

    ring_flash_attn.ring_flash_attn_forward(
        object(),
        q,
        k,
        v,
        softmax_scale=0.25,
        causal=False,
        attn_type=AttnType.SYCL_TLA,
        valid_kv_length=7,
    )

    assert sycl_kernel_spy[0]["valid_kv_lens"] == [4, 3]


def test_sycl_direct_call_accepts_a_consistent_pair(sycl_kernel_spy):
    q, k, v = _local_qkv()

    ring_flash_attn.ring_flash_attn_forward(
        object(),
        q,
        k,
        v,
        softmax_scale=0.25,
        causal=False,
        attn_type=AttnType.SYCL_TLA,
        valid_kv_lens=[4, 3],
        valid_kv_length=7,
    )

    assert sycl_kernel_spy[0]["valid_kv_lens"] == [4, 3]


def test_sycl_direct_call_rejects_an_inconsistent_pair(sycl_kernel_spy):
    q, k, v = _local_qkv()

    with pytest.raises(ValueError, match="disagree"):
        ring_flash_attn.ring_flash_attn_forward(
            object(),
            q,
            k,
            v,
            softmax_scale=0.25,
            causal=False,
            attn_type=AttnType.SYCL_TLA,
            valid_kv_lens=[4, 4],
            valid_kv_length=7,
        )

    assert not sycl_kernel_spy


def test_klox_selector_rejects_a_packed_suffix_padding(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_RING_ATTN_TYPE", "klox_sage_v3_hybrid")
    metadata = AttentionMetadata(extra={"valid_kv_length": 7})
    key = torch.randn(1, 4, 2, 8)

    with pytest.raises(ValueError, match="valid_kv_length"):
        ring_attn_utils.select_xpu_ring_attn_kwargs(key, metadata, _FakeSpGroup())


def test_klox_selector_accepts_a_full_valid_length(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_RING_ATTN_TYPE", "klox_sage_v3_hybrid")
    metadata = AttentionMetadata(extra={"valid_kv_length": 8})
    key = torch.randn(1, 4, 2, 8)

    kwargs = ring_attn_utils.select_xpu_ring_attn_kwargs(key, metadata, _FakeSpGroup())

    assert kwargs is not None
    assert kwargs["attn_type"] == AttnType.KLOX_SAGE_V3_HYBRID
