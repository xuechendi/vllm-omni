# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online per-tensor FP8 quantization for the FlashAttention varlen path.

``flash_attn_varlen_func`` accepts FP8 Q/K/V alongside ``{q,k,v}_descale``
dequantization factors; see the signature this mirrors in
``vllm/vllm_flash_attn/flash_attn_interface.py``. DiT activations carry no
calibrated scales, so each scale is derived online from the tensor's own amax:
one scalar per tensor (per-tensor granularity), broadcast to the
``(num_sequences, num_kv_heads)`` layout the kernel indexes -- the same shape
vLLM's own FA backend builds with ``layer._k_scale.expand(descale_shape)``.

There are two FP8 modes in the Xe kernel, and which one we get is a property of
the installed kernel build, not something the caller can query:

* **K/V only** -- Q stays native. The kernel dequantizes K and V per element
  inside the mainloop (``dequantize(tSrK, scale_k)``), so both DPAS GEMMs still
  run at native width and FP8 buys only memory traffic. This is the sole mode
  the ``xe_2`` kernel implements.
* **Full FP8** -- Q is FP8 too, so Q*K runs natively in FP8; the q/k descales
  fold into the softmax scale, the softmax output P is packed to e4m3 so P*V is
  also native FP8, and scale_v is applied once after the K loop. Implemented
  only in the ``xe_3`` kernel, which ``attn_interface.cpp`` selects for
  ``is_xe3p_arch()`` (CRI and NVL-P).

So we ask for full FP8 and fall back permanently if the installed stack cannot
deliver it -- see ``fp8_varlen_attn``. Two distinct failure modes have to be
detected, because only one of them raises:

1. Older kernel wheels reject the argument outright with
   ``NotImplementedError("FA2 does not support q_descale")``.
2. If the required kernel policy was not compiled, the kernel wrapper swallows
   its own "not compiled" error and silently substitutes a pure-PyTorch
   reference attention. That returns correct numbers but is orders of magnitude
   slower, so it must not be used every step.

Case 2 is detected by ownership of the output buffer: we hand the kernel an
``out`` tensor, and only the real kernel writes into it. The reference fallback
allocates its own, so a differing ``data_ptr()`` means the kernel did not run.
Supplying ``out`` is required anyway -- ``xpu_ops.flash_attn_varlen_func``
otherwise allocates ``torch.empty(q.shape, dtype=q.dtype)``, which is FP8 once Q
is quantized, and the C++ op's own default for an FP8 query is fp16. Neither is
the native dtype the pipeline expects back.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

SYCL_TLA_FP8_LABEL = "fp8_sycl_tla"
SYCL_TLA_MXFP8_LABEL = "mxfp8_sycl_tla"

_SYCL_TLA_HEAD_DIM = 128
_MX_GROUP_SIZE = 32
_UE8M0_BIAS = 127
_MIN_AMAX = 1e-6

# None until the first FP8 attention call resolves it; False latches permanently
# so a stack without full-FP8 support pays the probe exactly once per process.
_full_fp8_supported: bool | None = None

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# The scale is computed in fp32 but has to be applied in the tensor's own dtype
# to keep the division a single pass. Rounding it to bf16 can round *down*, which
# would push the amax element just above _FP8_MAX and cast to inf. One bf16 ULP
# of headroom makes ``tensor / scale <= _FP8_MAX`` hold by construction, so no
# clamp pass is needed (a clamp costs ~25% more than the division itself).
_FP8_SCALE_HEADROOM = 1.0 + 2.0**-8


@torch._dynamo.disable
@lru_cache(maxsize=1)
def _import_sycl_tla_fmha(label: str):
    try:
        import sycl_tla_fmha
    except ImportError as exc:
        raise ImportError(
            f"kv_cache_dtype='{label}' requires the sycl_tla_fmha extension built from "
            "cutlass-sycl examples/06_bmg_flash_attention. Put its build directory "
            "on PYTHONPATH."
        ) from exc
    return sycl_tla_fmha


@torch._dynamo.disable
@lru_cache(maxsize=1)
def _load_sycl_tla_fp8_func():
    module = _import_sycl_tla_fmha(SYCL_TLA_FP8_LABEL)
    return module.prefill_fp8_e4m3_bshd


@torch._dynamo.disable
@lru_cache(maxsize=1)
def _load_sycl_tla_mxfp8_func():
    module = _import_sycl_tla_fmha(SYCL_TLA_MXFP8_LABEL)
    if not hasattr(module, "prefill_mxfp8_e4m3_bshd"):
        raise ImportError(
            f"kv_cache_dtype='{SYCL_TLA_MXFP8_LABEL}' requires a sycl_tla_fmha build "
            f"that exports prefill_mxfp8_e4m3_bshd; rebuild the extension from {module.__file__}"
        )
    return module.prefill_mxfp8_e4m3_bshd


def _abs_amax(tensor: torch.Tensor) -> torch.Tensor:
    return torch.maximum(tensor.amax().abs(), tensor.amin().abs()).float()


def _scale_to_fp8(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        tensor.to(torch.float32, copy=True)
        .mul_(scale)
        .clamp_(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
    )


def _validate_sycl_tla_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, label: str) -> int:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"kv_cache_dtype='{label}' requires BSHD Q/K/V tensors")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(f"kv_cache_dtype='{label}' requires matching Q/K/V batch sizes")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError(f"kv_cache_dtype='{label}' requires matching K/V shapes")
    head_dim = query.shape[-1]
    if head_dim != _SYCL_TLA_HEAD_DIM or key.shape[-1] != head_dim or value.shape[-1] != head_dim:
        raise ValueError(
            f"kv_cache_dtype='{label}' only supports Q/K/V head_dim {_SYCL_TLA_HEAD_DIM}"
        )
    return head_dim


def sycl_tla_fp8_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Run the dense BSHD SYCL-TLA FP8 E4M3 attention kernel."""
    head_dim = _validate_sycl_tla_inputs(query, key, value, SYCL_TLA_FP8_LABEL)
    prefill = _load_sycl_tla_fp8_func()
    alpha = (softmax_scale if softmax_scale is not None else head_dim**-0.5) * math.sqrt(head_dim)

    q_amax = _abs_amax(query).clamp_(min=_MIN_AMAX)
    k_amax = _abs_amax(key).clamp_(min=_MIN_AMAX)
    v_amax = _abs_amax(value).clamp_(min=_MIN_AMAX)
    q_scale = torch.sqrt(alpha * k_amax / q_amax)
    v_scale = _FP8_MAX / v_amax

    query_fp8 = _scale_to_fp8(query, q_scale)
    key_fp8 = _scale_to_fp8(key, alpha / q_scale)
    value_fp8 = _scale_to_fp8(value, v_scale)

    torch.xpu.synchronize()
    output = prefill(q=query_fp8, k=key_fp8, v=value_fp8, is_causal=causal)
    return (output / v_scale).to(query.dtype)


def _mx_exponent(blocked: torch.Tensor, reduce_dim: int) -> torch.Tensor:
    amax = torch.maximum(blocked.amax(reduce_dim).abs(), blocked.amin(reduce_dim).abs()).float()
    amax = amax.clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax / _FP8_MAX))
    return exponent.clamp(-_UE8M0_BIAS, _UE8M0_BIAS)


def _mx_apply(blocked: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    return (
        blocked.to(torch.float32, copy=True)
        .div_(torch.exp2(exponent))
        .clamp_(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
    )


@torch._dynamo.disable
def _mx_quantize(tensor: torch.Tensor, group_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """MX-quantize a BSHD tensor in 32-element groups."""
    group_size = _MX_GROUP_SIZE
    batch, seq_len, num_heads, head_dim = tensor.shape

    if group_dim == 3:
        if head_dim % group_size:
            raise ValueError(f"head dim {head_dim} must be a multiple of {group_size}")
        blocked = tensor.reshape(batch, seq_len, num_heads, head_dim // group_size, group_size)
        exponent = _mx_exponent(blocked, -1)
        values = _mx_apply(blocked, exponent.unsqueeze(-1)).reshape_as(tensor)
        return values, (exponent + _UE8M0_BIAS).to(torch.uint8)

    if group_dim != 1:
        raise ValueError(f"unsupported group_dim {group_dim}")

    full_len = (seq_len // group_size) * group_size
    values = torch.empty(tensor.shape, dtype=torch.float8_e4m3fn, device=tensor.device)
    exponent = None
    if full_len:
        blocked = tensor[:, :full_len].reshape(batch, full_len // group_size, group_size, num_heads, head_dim)
        exponent = _mx_exponent(blocked, 2)
        values[:, :full_len] = _mx_apply(blocked, exponent.unsqueeze(2)).reshape(
            batch, full_len, num_heads, head_dim
        )
    if full_len < seq_len:
        tail = tensor[:, full_len:].unsqueeze(1)
        tail_exponent = _mx_exponent(tail, 2)
        values[:, full_len:] = _mx_apply(tail, tail_exponent.unsqueeze(2)).reshape(
            batch, seq_len - full_len, num_heads, head_dim
        )
        exponent = tail_exponent if exponent is None else torch.cat((exponent, tail_exponent), dim=1)
    assert exponent is not None
    return values, (exponent + _UE8M0_BIAS).to(torch.uint8)


def sycl_tla_mxfp8_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Run the dense BSHD SYCL-TLA block-scaled MXFP8 attention kernel."""
    head_dim = _validate_sycl_tla_inputs(query, key, value, SYCL_TLA_MXFP8_LABEL)
    prefill = _load_sycl_tla_mxfp8_func()
    alpha = (softmax_scale if softmax_scale is not None else head_dim**-0.5) * math.sqrt(head_dim)
    scaled_query = query if alpha == 1.0 else query * alpha

    query_fp8, query_exp = _mx_quantize(scaled_query, 3)
    key_fp8, key_exp = _mx_quantize(key, 3)
    value_fp8, value_exp = _mx_quantize(value, 1)

    torch.xpu.synchronize()
    output = prefill(
        q=query_fp8,
        k=key_fp8,
        v=value_fp8,
        scale_q=query_exp.permute(0, 2, 3, 1).contiguous(),
        scale_k=key_exp.permute(0, 2, 3, 1).contiguous(),
        scale_v=value_exp.permute(0, 2, 1, 3).contiguous(),
        is_causal=causal,
    )
    return output.to(query.dtype)


def _quantize_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``tensor`` to FP8 E4M3 under a single scale for the whole tensor.

    Deliberately *not* ``vllm._custom_ops.scaled_fp8_quant``. That op has a fused
    CUDA kernel but no XPU one, and its fallback runs at roughly a tenth of the
    memory bandwidth this device sustains: measured on Crescent Island at the
    Wan2.2-A14B self-attention shape (10560 x 40 x 128, 103 MB bf16) it takes
    12.8 ms per tensor, versus 1.13 ms here, for a bit-identical scale. Three
    tensors per attention call made quantization cost 38 ms against an 11 ms
    bf16 attention, which turned FP8 into a net 1.6x *slowdown* end to end.

    Returns the FP8 tensor and a ``(1,)`` fp32 scale, matching the descale layout
    ``flash_attn_varlen_func`` expects after ``.expand()``.
    """
    amax = tensor.abs().amax().float()
    # An all-zero tensor has no meaningful scale; 1.0 keeps the division finite
    # and quantizes to all zeros, which is the right answer. Guarding on amax
    # rather than clamping the scale to a floor matters: a floor large enough to
    # be safe here would flush a legitimately small tensor entirely to zero.
    scale = torch.where(amax > 0, amax * (_FP8_SCALE_HEADROOM / _FP8_MAX), torch.ones_like(amax))
    quantized = tensor.div(scale.to(tensor.dtype)).to(torch.float8_e4m3fn)
    return quantized, scale.reshape(1)


def quantize_kv_per_tensor(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Quantize varlen K/V to per-tensor FP8 and build the matching descales.

    ``key`` and ``value`` are ``(total_kv_tokens, num_kv_heads, head_dim)``, the
    layout ``flash_attn_varlen_func`` consumes. Returns the FP8 K/V plus the
    ``k_descale``/``v_descale`` kwargs to forward to the kernel.
    """
    key_fp8, k_scale = _quantize_per_tensor(key)
    value_fp8, v_scale = _quantize_per_tensor(value)
    descale_shape = (num_sequences, key.shape[-2])
    return (
        key_fp8,
        value_fp8,
        {
            "k_descale": k_scale.expand(descale_shape),
            "v_descale": v_scale.expand(descale_shape),
        },
    )


def _rejected_fp8_query(exc: BaseException) -> bool:
    """True when the kernel refused an FP8 query rather than failing for real."""
    message = str(exc)
    if isinstance(exc, NotImplementedError):
        return "q_descale" in message
    return "not compiled" in message


def _wrote_into(result: Any, out: torch.Tensor) -> bool:
    """True when the kernel filled our ``out`` buffer instead of allocating one."""
    tensor = result[0] if isinstance(result, tuple) else result
    if not isinstance(tensor, torch.Tensor) or out.numel() == 0:
        return True
    return tensor.data_ptr() == out.data_ptr()


def fp8_varlen_attn(
    varlen_func: Callable[..., Any],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_sequences: int,
    **kernel_kwargs: Any,
) -> Any:
    """Run ``varlen_func`` with per-tensor FP8 inputs, preferring the full-FP8 path.

    Q/K/V are ``(total_tokens, num_heads, head_dim)``. Quantizes K/V always, and
    Q as well while the installed kernel supports it; on the first call that
    proves it does not, downgrades to K/V-only for the rest of the process.
    """
    global _full_fp8_supported

    key_fp8, value_fp8, descales = quantize_kv_per_tensor(key, value, num_sequences=num_sequences)

    if _full_fp8_supported is not False:
        query_fp8, q_scale = _quantize_per_tensor(query)
        out = torch.empty(
            (*query.shape[:-1], value.shape[-1]),
            dtype=query.dtype,
            device=query.device,
        )
        try:
            result = varlen_func(
                q=query_fp8,
                k=key_fp8,
                v=value_fp8,
                out=out,
                q_descale=q_scale.expand((num_sequences, key.shape[-2])),
                **descales,
                **kernel_kwargs,
            )
        except (NotImplementedError, RuntimeError) as exc:
            if not _rejected_fp8_query(exc):
                raise
            _full_fp8_supported = False
            logger.warning(
                "FP8 attention: this kernel build rejected an FP8 query (%s); "
                "falling back to FP8 K/V with a native-dtype query for the rest "
                "of this process. Native FP8 Q*K needs an xe3p device (CRI or "
                "NVL-P) and a kernel wheel whose flash_attn_varlen_func accepts "
                "q_descale.",
                exc,
            )
        else:
            if _wrote_into(result, out):
                # Log the winning mode too, not just the downgrades: otherwise a
                # silent log is ambiguous between "full FP8" and "FP8 never ran".
                # Only on the None -> True transition -- this runs on every
                # attention call, i.e. once per DiT block per denoise step.
                if _full_fp8_supported is None:
                    logger.info(
                        "FP8 attention: using full FP8 (Q, K and V quantized), so "
                        "Q*K and P*V both run natively in FP8."
                    )
                _full_fp8_supported = True
                return result
            # The wrapper silently substituted its PyTorch reference attention,
            # which is correct but far too slow to keep calling. Use this
            # result (fixing the dtype it chose for an FP8 query) and drop to
            # K/V-only, which the real kernel does compile.
            _full_fp8_supported = False
            logger.warning(
                "FP8 attention: the full-FP8 kernel policy is not compiled in "
                "this wheel, so the kernel wrapper fell back to reference "
                "attention. Dropping to FP8 K/V with a native-dtype query."
            )
            tensor = result[0] if isinstance(result, tuple) else result
            return tensor.to(query.dtype) if tensor.dtype != query.dtype else tensor

    return varlen_func(
        q=query,
        k=key_fp8,
        v=value_fp8,
        **descales,
        **kernel_kwargs,
    )
