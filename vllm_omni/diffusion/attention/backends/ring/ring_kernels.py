# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import math

import torch

from .ring_globals import (
    HAS_AITER,
    HAS_FA3,
    HAS_FA4,
    HAS_FLASH_ATTN,
    HAS_FLASHINFER,
    HAS_KLOX_SAGE_V3,
    HAS_SYCL_TLA_RING_ATTN,
    fa3_fwd_func,
    fa4_attn_func,
)

if HAS_SYCL_TLA_RING_ATTN:
    from ring_attn_interface_ipc_pull import ring_attn_ipc_forward

_scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention
_scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention

try:
    import torch_musa  # noqa: F401


    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_attention_flash_musa
    _scaled_dot_product_efficient_attention = None
except ModuleNotFoundError:
    pass

if HAS_AITER:
    from aiter import flash_attn_func as flash_attn_func_aiter

if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward

if HAS_FLASHINFER:
    from flashinfer.prefill import single_prefill_with_kv_cache

_LOG2_E = math.log2(math.e)


def sycl_tla_ring_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
    valid_kv_lens=None,
    precision="bf16",
    scale_q=None,
    scale_k=None,
    scale_v=None,
):
    """Run the SYCL/TLA IPC ring kernel on Intel XPU."""
    if not HAS_SYCL_TLA_RING_ATTN:
        raise RuntimeError("SYCL/TLA IPC ring attention is not available")

    return ring_attn_ipc_forward(
        process_group=process_group,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        joint_tensor_key=joint_tensor_key,
        joint_tensor_value=joint_tensor_value,
        joint_strategy=joint_strategy,
        return_softmax_lse=True,
        valid_kv_lens=valid_kv_lens,
        precision=precision,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )


def klox_sage_v3_hybrid_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    """deepklox SageAttention V3 Hybrid (MXFP4 E2M1 QK^T + MXFP8 E4M3 P/V) as a
    Ring Attention block kernel, on Intel CRI (Xe3P).

    q/k/v are (bs, seqlen, nhead, headdim), i.e. the kernel's NHD layout, which
    it reads without a staging transpose.

    Checked here: the arguments this signature accepts but the kernel cannot see
    (it takes no dropout/window/softcap/alibi/scale), plus Sq == Skv, which has a
    ring-specific cause worth naming. Everything the kernel does see -- device,
    dtype, head_dim, MHA, B=1 -- it validates itself with a clear message. Under
    pure ring on Wan2.2 all of them hold (D=128, MHA, non-causal, equal shards,
    no joint K/V: cross-attention opts out of sequence parallelism).
    """
    del return_softmax  # the kernel returns (out, lse) only, never the probabilities
    if not HAS_KLOX_SAGE_V3:
        raise ImportError(
            "deepklox SageAttention V3 Hybrid is not available. It needs a deepklox built on "
            "CRI with DEEPKLOX_ENABLED_KERNELS=sageattn, XPU_BUILD_WITH_XE3P=1 and "
            "DEEPKLOX_SAGEATTN_V3_HYBRID=1."
        )
    from deepklox.sageattn_interface import sageattn_v3_hybrid

    if causal:
        raise ValueError("sageattn_v3_hybrid is non-causal; ring attention with causal=True cannot use it")
    if dropout_p:
        raise ValueError("sageattn_v3_hybrid does not support dropout")
    if window_size not in ((-1, -1), None):
        raise ValueError(f"sageattn_v3_hybrid does not support a sliding window, got {window_size}")
    if softcap:
        raise ValueError("sageattn_v3_hybrid does not support softcap")
    if alibi_slopes is not None:
        raise ValueError("sageattn_v3_hybrid does not support alibi slopes")
    if q.dim() != 4:
        raise ValueError(f"sageattn_v3_hybrid needs rank-4 (bs, seqlen, nhead, headdim) inputs, got {tuple(q.shape)}")

    _bs, s_q, _h_q, d = q.shape
    if k.shape[1] != s_q:
        raise ValueError(
            f"sageattn_v3_hybrid requires Sq == Skv, got Sq={s_q} Skv={k.shape[1]}. Under ring this means "
            "either a joint K/V prefix or unequal shards (sequence-parallel padding); neither is supported."
        )
    # The scale is hard-wired inside the kernel, so a caller-supplied one that
    # differs would be silently dropped.
    if softmax_scale is not None and abs(softmax_scale - d**-0.5) > 1e-6:
        raise ValueError(f"sageattn_v3_hybrid hard-wires scale=1/sqrt({d}); got softmax_scale={softmax_scale}")

    # B=1 only. Batched CFG (bs=2) is a loop rather than an error: the kernel is
    # per-sequence anyway, so this costs nothing but a second launch.
    outs, lses = [], []
    for i in range(q.shape[0]):
        out_i, lse_i = sageattn_v3_hybrid(
            q[i : i + 1].contiguous(),
            k[i : i + 1].contiguous(),
            v[i : i + 1].contiguous(),
            tensor_layout="NHD",
            return_lse=True,
        )
        outs.append(out_i)
        lses.append(lse_i)
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    lse = lses[0] if len(lses) == 1 else torch.cat(lses, dim=0)

    # The kernel documents its LSE in the log2 domain; update_out_and_lse merges
    # with sigmoid/logsigmoid of LSE differences, i.e. natural log. Getting this
    # conversion wrong does not crash, it just mis-weights every ring step.
    return out, lse.float() / _LOG2_E


def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
    op_type="efficient",
):
    assert op_type in ["flash", "efficient"], f"Invalid op_type: {op_type}"
    """
    q shape (bs, seqlen, nhead, hs)
    k shape (bs, seqlen, nhead, hs)
    v shape (bs, seqlen, nhead, hs)
    """
    if op_type == "flash" and q.dtype == torch.float32:
        raise ValueError("The explicitly requested Flash ring kernel does not support float32.")

    # Volta (V100, sm70) does not have native bf16 support for the efficient
    # SDPA kernel. When running Ring attention in bf16 on such GPUs, the
    # efficient kernel can fail with "cutlassF: no kernel found to launch".
    #
    # We keep the overall model in bf16 for numerical stability, but cast
    # the attention inputs to fp16 for the SDPA call. The output is cast
    # back to bf16 to match the caller's expected dtype.
    orig_dtype = q.dtype
    if q.is_cuda and q.dtype == torch.bfloat16 and op_type == "efficient":
        major_cc, _minor_cc = torch.cuda.get_device_capability(q.device)
        if major_cc < 8:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # GQA/MQA support. Ring attention only ever *communicates* K/V with H_kv
    # heads -- that is precisely the benefit of GQA -- so expanding here does
    # not change communication volume. The expansion is needed because the aten
    # SDPA ops below are required for the LSE output that ring accumulation
    # consumes, and they do not accept `enable_gqa`.
    if k.shape[1] != q.shape[1]:
        num_heads, num_kv_heads = q.shape[1], k.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA/MQA.")
        n_rep = num_heads // num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    if op_type == "flash":
        out, lse = _scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    elif op_type == "efficient":
        out, lse = _scaled_dot_product_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    else:
        raise ValueError(f"Invalid op_type: {op_type}")

    out = out.transpose(1, 2)
    # Keep LSE in fp32 for numerical stability when accumulating across ring
    # steps (update_out_and_lse uses sigmoid/logsigmoid on LSE diffs).
    # Casting LSE down to fp16/bf16 can introduce NaNs on some GPUs/shapes.
    # PyTorch SDPA returns LSE as (B, H, S).
    lse = lse.to(torch.float32)

    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)

    return out, lse


def flash_attn_forward(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < "2.6.3":
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    # FA2 softmax_lse is (B, H, S), optionally padded on S.
    return block_out, block_lse


def fa3_forward(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax):
    """FA3 forward pass for inference.

    FA3 supports Ampere, Ada, and Hopper GPUs. Dropout is ignored since FA3 is inference-only.
    Uses low-level API (_flash_attn_forward) which always returns softmax_lse,
    required for Ring Attention's correct accumulation.
    """
    assert HAS_FA3, "FA3 is not available"
    assert fa3_fwd_func is not None, "FA3 low-level API (fa3_fwd_func) not available"

    # Low-level API always returns (out, softmax_lse, S_dmask, rng_state)
    out, softmax_lse, *_ = fa3_fwd_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0] if window_size else -1,
        window_size_right=window_size[1] if window_size else -1,
        softcap=softcap if softcap else 0.0,
    )

    # FA3 softmax_lse is (B, H, S).
    return out, softmax_lse


# Legacy alias for backward compatibility
flash_attn3_func_forward = fa3_forward


def flash_attn4_func_forward(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    """FA4 forward pass returning the LSE required by Ring Attention."""
    del dropout_p, alibi_slopes, return_softmax
    assert HAS_FA4, "FA4 is not available"
    assert fa4_attn_func is not None, "FA4 high-level API is not available"

    left, right = window_size if window_size else (-1, -1)
    out, softmax_lse = fa4_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(
            None if left == -1 else left,
            None if right == -1 else right,
        ),
        softcap=softcap or 0.0,
        return_lse=True,
    )
    # FA4 CuTe softmax_lse is (B, H, S).
    return out, softmax_lse


def flash_attn_forward_aiter(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_AITER, "Aiter is not available"
    block_out, block_lse = flash_attn_func_aiter(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_lse=True,
    )

    # Aiter softmax_lse is (B, H, S), matching FA2.
    return block_out, block_lse


def flashinfer_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert HAS_FLASHINFER, "FlashInfer is not available"
    if q.ndim == 4:
        if q.shape[0] > 1:
            raise ValueError("batch size > 1 is not supported")
        out, lse = single_prefill_with_kv_cache(
            q[0],
            k[0],
            v[0],
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        # FlashInfer unbatched LSE is (H, S). Canonical ring layout is (B, H, S).
        out, lse = out.unsqueeze(0), lse.unsqueeze(0)
    elif q.ndim == 3:
        out, lse = single_prefill_with_kv_cache(
            q,
            k,
            v,
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        # Unbatched (H, S) matches canonical heads-major layout without a batch dim.
    else:
        raise ValueError(f"Invalid input shape: {q.shape}")
    lse = lse / _LOG2_E
    return out, lse
