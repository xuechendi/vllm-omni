# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""XPU ring attention: which block kernel, and the arguments only it takes.

`RingParallelAttention.run_attention` asks `select_xpu_ring_attn_kwargs` once, on
XPU, and hands the result straight to `ring_flash_attn_func`. Nothing else in
that method changes: the CUDA/ROCm selection (FA4/FA3/AITER) and the SDPA
fallback stay where they were.

Selectable with ``VLLM_OMNI_RING_ATTN_TYPE``:

* ``sycl_tla`` -- SYCL/TLA IPC ring attention. Also the default when the
  extension is built, which is what this path did before.
* ``klox_sage_v3_hybrid`` -- deepklox SageAttention V3 Hybrid on CRI/Xe3P
  (MXFP4 E2M1 QK^T + MXFP8 E4M3 P/V).

Any other value returns None, i.e. "not an XPU kernel, use the generic
selection". On XPU that lands on the SDPA ring: a ring block kernel has to
return an LSE to merge partial outputs, and the XPU flash-attn op that
``utils/fa.py`` binds (``xpu_ops.flash_attn_varlen_func``) returns none, so
``flash_attn`` cannot be a ring kernel here however it is requested.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.ring.ring_globals import HAS_SYCL_TLA_RING_ATTN
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.backends.ring_flash_attn import _RING_ATTN_TYPE_ENV
from vllm_omni.diffusion.attention.backends.utils.fa_fp8 import SYCL_TLA_FP8_LABEL

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator

logger = init_logger(__name__)

# The block kernels this module dispatches. Everything else falls through.
_XPU_RING_ATTN_TYPES = (AttnType.SYCL_TLA, AttnType.KLOX_SAGE_V3_HYBRID)


def resolve_ring_attn_type() -> AttnType | None:
    """``VLLM_OMNI_RING_ATTN_TYPE`` as an AttnType, or None if it is unset.

    Read here rather than in ``ring_flash_attn_func`` because the ring strategy
    always passes an explicit ``attn_type``, so the env var that function
    consults never applies on the serve path. An env var also needs no plumbing
    through the config chain and reaches the *spawned* diffusion workers, which
    inherit os.environ.

    An unknown name raises (``AttnType.from_string``) instead of silently
    autoselecting a different kernel. This runs per attention call, not in
    ``Attention.__init__``, where a ``try/except Exception`` would turn a typo
    into a silent fallback to non-ring attention.
    """
    requested = os.environ.get(_RING_ATTN_TYPE_ENV)
    if requested is None or not requested.strip():
        return None
    return AttnType.from_string(requested.strip().lower())


def _valid_kv_lens_from_padding(orig_seq_len: int, block_len: int, ring_world_size: int) -> list[int]:
    valid_kv_lens = [
        max(0, min(orig_seq_len - ring_rank * block_len, block_len)) for ring_rank in range(ring_world_size)
    ]
    if any(valid_kv_len == 0 for valid_kv_len in valid_kv_lens):
        raise ValueError(
            "SYCL/TLA ring attention does not support an empty local K/V shard; "
            "sequence length must provide at least one token per ring rank."
        )
    return valid_kv_lens


def select_xpu_ring_attn_kwargs(
    key: torch.Tensor,
    attn_metadata: AttentionMetadata | None,
    sp_group: SequenceParallelGroupCoordinator,
) -> dict[str, Any] | None:
    """``ring_flash_attn_func`` kwargs for an XPU block kernel, else None.

    None means "no XPU kernel applies", so the caller keeps its generic
    selection. The returned dict always carries ``attn_type`` and the joint
    tensors, plus the arguments only the chosen kernel accepts.
    """
    attn_type = resolve_ring_attn_type()
    if attn_type is None:
        # Same default as before: SYCL/TLA whenever its extension is built.
        attn_type = AttnType.SYCL_TLA if HAS_SYCL_TLA_RING_ATTN else None
    if attn_type not in _XPU_RING_ATTN_TYPES:
        return None

    joint_key, joint_value = None, None
    joint_strategy = "front"
    precision = None
    if attn_metadata is not None:
        joint_key = attn_metadata.joint_key
        joint_value = attn_metadata.joint_value
        if attn_metadata.joint_strategy is not None:
            joint_strategy = attn_metadata.joint_strategy
        if attn_metadata.extra.get("kv_cache_dtype") == SYCL_TLA_FP8_LABEL:
            precision = "fp8_pertensor"

    kwargs: dict[str, Any] = {
        "attn_type": attn_type,
        "joint_tensor_key": joint_key,
        "joint_tensor_value": joint_value,
        "joint_strategy": joint_strategy,
    }

    from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

    padding_size = 0
    orig_seq_len = None
    if is_forward_context_available():
        forward_ctx = get_forward_context()
        orig_seq_len = forward_ctx.sp_original_seq_len
        padding_size = forward_ctx.sp_padding_size or 0

    if attn_type == AttnType.SYCL_TLA:
        valid_kv_lens = None
        if orig_seq_len is not None and padding_size > 0:
            valid_kv_lens = _valid_kv_lens_from_padding(orig_seq_len, key.shape[1], sp_group.ring_world_size)
        kwargs["sycl_tla_precision"] = precision
        kwargs["valid_kv_lens"] = valid_kv_lens
        return kwargs

    # KLOX_SAGE_V3_HYBRID. Padded shards carry no mask through the ring loop and
    # this kernel takes no valid_kv_lens, so the padding would be attended to as
    # if it were content. Fail instead of returning wrong pixels.
    if padding_size:
        raise ValueError(
            f"{_RING_ATTN_TYPE_ENV}={attn_type.value} cannot handle sequence-parallel padding "
            f"(sp_padding_size={padding_size}); it has no way to mask the padded K/V. "
            "Use a sequence length divisible by the ring degree."
        )
    logger.info_once("Ring attention block kernel: deepklox SageAttention V3 Hybrid (MXFP4 QK^T + MXFP8 E4M3 PV)")
    return kwargs
