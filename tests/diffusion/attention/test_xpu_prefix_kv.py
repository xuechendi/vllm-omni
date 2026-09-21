# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the XPU prefix-KV slicing switch (``H3_XPU_PREFIX_KV``).

``flash_attn.py`` reads the switch and the platform once at import, so every
test reloads the module under an explicit fake platform and env instead of
mutating the already-imported copy. The teardown reloads once more with the
real platform and a clean env so later tests see the module exactly as a
fresh interpreter would.
"""

import importlib
import os

import pytest
import torch

import vllm_omni.diffusion.attention.backends.flash_attn as fa_module
import vllm_omni.platforms as omni_platforms

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_ENV = "H3_XPU_PREFIX_KV"


class _FakePlatform:
    def __init__(self, name: str) -> None:
        self.name = name

    def is_cuda(self) -> bool:
        return self.name == "cuda"

    def is_xpu(self) -> bool:
        return self.name == "xpu"

    def is_npu(self) -> bool:
        return self.name == "npu"

    def is_rocm(self) -> bool:
        return self.name == "rocm"

    def is_musa(self) -> bool:
        return self.name == "musa"


class VarlenSpy:
    """Captures the tensors/kwargs handed to flash_attn_varlen_func."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, q=None, k=None, v=None, **kwargs):
        self.calls.append({"q": q, "k": k, "v": v, **kwargs})
        return torch.zeros_like(q)


@pytest.fixture
def load_flash_attn():
    """Reload flash_attn.py under a chosen platform/env; restore afterwards."""
    saved_platform = omni_platforms.current_omni_platform
    saved_env = os.environ.get(_ENV)

    def _load(platform: str = "xpu", env: str | None = None):
        if env is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = env
        omni_platforms.current_omni_platform = _FakePlatform(platform)
        return importlib.reload(fa_module)

    yield _load

    omni_platforms.current_omni_platform = saved_platform
    if saved_env is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = saved_env
    importlib.reload(fa_module)


@pytest.fixture
def varlen_spy(monkeypatch):
    from vllm_omni.diffusion.attention.backends.utils import fa as fa_utils

    spy = VarlenSpy()
    monkeypatch.setattr(fa_utils, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", spy, raising=False)
    return spy


def _impl(mod, *, causal=False, role="self"):
    return mod.FlashAttentionImpl(
        num_heads=4,
        head_size=8,
        softmax_scale=1.0,
        causal=causal,
        role=role,
    )


def _meta(mod, *, valid_kv_length=None, attn_mask=None):
    extra = {} if valid_kv_length is None else {"valid_kv_length": valid_kv_length}
    return mod.AttentionMetadata(attn_mask=attn_mask, extra=extra)


def _qkv(batch=1, seq=64):
    return (
        torch.randn(batch, seq, 4, 8),
        torch.randn(batch, seq, 4, 8),
        torch.randn(batch, seq, 4, 8),
    )


def test_xpu_slices_kv_prefix(load_flash_attn, varlen_spy):
    """K/V trimmed to the valid prefix, Q kept whole, cu_seqlens split."""
    mod = load_flash_attn("xpu")
    impl = _impl(mod)

    q, k, v = _qkv()
    out = impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=50))

    assert len(varlen_spy.calls) == 1
    c = varlen_spy.calls[0]
    assert c["k"].shape[0] == 50, c["k"].shape
    assert c["v"].shape[0] == 50, c["v"].shape
    assert c["q"].shape[0] == 64, c["q"].shape
    assert c["cu_seqlens_k"].tolist() == [0, 50]
    assert c["cu_seqlens_q"].tolist() == [0, 64]
    assert c["max_seqlen_k"] == 50
    assert c["max_seqlen_q"] == 64
    assert tuple(out.shape) == (1, 64, 4, 8)


def test_xpu_backend_advertises_prefix_kv_slicing(load_flash_attn):
    """The model side reads this as a plain attribute, so it must be a bool."""
    mod = load_flash_attn("xpu")
    flag = mod.FlashAttentionBackend.supports_prefix_kv_slicing
    assert flag is True
    assert isinstance(flag, bool)


def test_xpu_env_off_keeps_mask_path(load_flash_attn):
    """H3_XPU_PREFIX_KV=0 -> flag False and the masked path still runs."""
    mod = load_flash_attn("xpu", env="0")
    assert mod.FlashAttentionBackend.supports_prefix_kv_slicing is False

    impl = _impl(mod)
    seen = {}

    def fake_masked(query, key, value, attention_mask, *, fp8_kv=False):
        seen["k_len"] = key.size(1)
        return torch.zeros_like(query)

    impl._forward_varlen_masked = fake_masked
    mask = torch.arange(64)[None] < 50
    q, k, v = _qkv()
    impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=50, attn_mask=mask))

    assert seen["k_len"] == 64, "K must NOT be sliced when the switch is off"


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_cuda_npu_unchanged(load_flash_attn, platform):
    """Other platforms keep the old capability answers exactly."""
    mod = load_flash_attn(platform)
    assert mod.FlashAttentionBackend.supports_prefix_kv_slicing is False
    assert mod.FlashAttentionBackend.supports_packed_mask_free() is True
    assert mod.FlashAttentionBackend.supports_multi_doc_packed_varlen() is (platform == "cuda")


def test_no_valid_kv_length_is_dense_unchanged(load_flash_attn, varlen_spy):
    """A model that never sets valid_kv_length is byte-for-byte unaffected."""
    mod = load_flash_attn("xpu")
    impl = _impl(mod)

    q, k, v = _qkv()
    impl.forward_xpu(q, k, v, _meta(mod))

    c = varlen_spy.calls[0]
    assert c["k"].shape[0] == 64, "no valid_kv_length -> no slicing"
    assert c["cu_seqlens_k"].tolist() == [0, 64]


def test_cross_attention_never_sliced(load_flash_attn, varlen_spy):
    """Cross-attn K/V is not a [real, pad] prefix of Q; slicing it would be wrong."""
    mod = load_flash_attn("xpu")
    impl = _impl(mod, role="cross")
    q, k, v = _qkv()
    impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=50))
    assert varlen_spy.calls[0]["k"].shape[0] == 64


def test_valid_kv_length_equal_is_noop(load_flash_attn, varlen_spy):
    """used == k_len means there is no padding to drop."""
    mod = load_flash_attn("xpu")
    impl = _impl(mod)
    q, k, v = _qkv()
    impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=64))
    assert varlen_spy.calls[0]["k"].shape[0] == 64


def test_slice_is_a_view_not_a_copy(load_flash_attn, varlen_spy):
    """The whole point: no index_select / scatter, so the slice must share storage."""
    mod = load_flash_attn("xpu")
    impl = _impl(mod)
    q, k, v = _qkv()
    impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=50))
    assert varlen_spy.calls[0]["k"].data_ptr() == k.data_ptr(), "K/V slicing must be zero-copy"


def test_numeric_equivalence_against_masked_path(load_flash_attn, monkeypatch):
    """Slicing K/V must give the same valid rows as masking them out.

    Uses a real softmax attention as the stand-in kernel so the comparison is
    about the *plan*, not about flash-attention internals.
    """
    mod = load_flash_attn("xpu")
    from vllm_omni.diffusion.attention.backends.utils import fa as fa_utils

    def ref_varlen(q=None, k=None, v=None, **kw):
        att = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2)) * kw["softmax_scale"]
        return (att.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1)

    monkeypatch.setattr(fa_utils, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", ref_varlen, raising=False)
    impl = _impl(mod)
    torch.manual_seed(0)
    q, k, v = _qkv()

    sliced = impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=50))

    att = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) * impl.softmax_scale
    att[..., 50:] = float("-inf")
    masked = (att.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)

    assert torch.allclose(sliced[:, :50], masked[:, :50], atol=1e-5), (sliced - masked).abs().max()


# --------------------------------------------------------------------------
# Fail-closed contract: the model skipped its padding mask because the backend
# advertised prefix slicing, so a valid_kv_length the helper cannot honor must
# raise (like main's cuDNN backend), not silently attend over padding rows.
# --------------------------------------------------------------------------


def test_bool_valid_kv_length_raises(load_flash_attn, varlen_spy):
    mod = load_flash_attn("xpu")
    impl = _impl(mod)
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=True))


def test_non_int_valid_kv_length_raises(load_flash_attn, varlen_spy):
    mod = load_flash_attn("xpu")
    impl = _impl(mod)
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=torch.tensor(50)))


def test_batch_gt_one_with_padding_raises(load_flash_attn, varlen_spy):
    mod = load_flash_attn("xpu")
    impl = _impl(mod)
    q, k, v = _qkv(batch=2)
    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=50))


def test_kv_length_mismatch_with_padding_raises(load_flash_attn, varlen_spy):
    mod = load_flash_attn("xpu")
    impl = _impl(mod)
    q = torch.randn(1, 64, 4, 8)
    k = torch.randn(1, 64, 4, 8)
    v = torch.randn(1, 60, 4, 8)
    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=50))


@pytest.mark.parametrize("bad", [0, -1, 65, 10_000])
def test_out_of_range_valid_kv_length_raises(load_flash_attn, varlen_spy, bad):
    """Zero, negative and beyond-K values are contract violations, not quiet
    no-slices — the model already skipped its padding mask on the strength of
    this field (main's cuDNN range: 0 < valid <= k_len)."""
    mod = load_flash_attn("xpu")
    impl = _impl(mod)
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_xpu(q, k, v, _meta(mod, valid_kv_length=bad))
