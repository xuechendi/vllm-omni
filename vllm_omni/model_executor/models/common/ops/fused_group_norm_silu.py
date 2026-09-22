# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: N803

"""Fused GroupNorm + SiLU operator.

This operator fuses GroupNorm followed by SiLU activation into a single kernel,
reducing memory traffic and kernel launch overhead. The implementation uses
Triton for CUDA/ROCm compatibility, and falls back to native PyTorch ops when
Triton is unavailable (NPU, CPU, ...), so callers never need a platform check.

Measured against eager ``F.silu(F.group_norm(...))`` on one L20X, bf16, 32
groups: 1.1-1.5x at the DiT ResBlock's activation sizes, where both paths are
dominated by launch overhead, and 2.2-2.9x at the VAE's decode-resolution
activations, where the saved memory traffic is what pays.
"""

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=6),
    ],
    key=["spatial_size", "C"],
)
@triton.jit
def _group_norm_silu_kernel(
    # Input/Output pointers
    x_ptr,
    out_ptr,
    # Normalization parameters
    weight_ptr,
    bias_ptr,
    # Shape info; x is contiguous (N, C, spatial_size)
    C,
    spatial_size,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Fused GroupNorm + SiLU kernel.
    Computes SiLU(GroupNorm(x)) in one kernel, avoiding intermediate tensors.
    Mean and variance use a parallel Welford reduction (block-level register
    reduction merged with the Chan formula), which avoids the catastrophic
    cancellation of E[x^2] - E[x]^2 on large-offset inputs. x is read once for
    the statistics and once for normalization (2 reads total).
    One program handles each (batch, group) pair. Channels are processed serially,
    while spatial positions are vectorized for diffusion workloads.
    Moments are accumulated in fp32 to match PyTorch numerics.
    """
    pid = tl.program_id(0)

    group_size = C // num_groups
    n_idx = pid // num_groups
    g_idx = pid % num_groups

    # === Pass 1: Welford reduction over the whole group (fp32) ===
    # Each block is reduced in registers (block mean, then centered M2), so the
    # loaded values are reused instead of re-reading memory; blocks are merged
    # with the Chan et al. parallel-Welford formula. This keeps both the mean and
    # the variance accurate for inputs like ``10000 +- 0.1``, where the naive
    # ``E[x^2] - E[x]^2`` cancels catastrophically, and reads x only once for the
    # statistics (2 reads total including normalize).
    n_total = tl.zeros([1], dtype=tl.float32)
    mean_total = tl.zeros([1], dtype=tl.float32)
    m2_total = tl.zeros([1], dtype=tl.float32)

    for c_offset in range(group_size):
        c_idx = g_idx * group_size + c_offset
        base = n_idx * C * spatial_size + c_idx * spatial_size

        for s_start in tl.range(0, spatial_size, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < spatial_size

            x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
            x_val = x_val.to(tl.float32)

            # block-level reduction in registers (no extra memory traffic)
            n = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
            bsum = tl.sum(x_val, axis=0)
            bmean = bsum / n
            bm2 = tl.sum(tl.where(mask, (x_val - bmean) * (x_val - bmean), 0.0), axis=0)

            # Chan et al. merge into the running (n, mean, m2)
            delta = bmean - mean_total
            new_n = n_total + n
            mean_total = mean_total + delta * (n / new_n)
            m2_total = m2_total + bm2 + delta * delta * (n_total * n / new_n)
            n_total = new_n

    mean = mean_total
    var = m2_total / n_total
    rstd = 1.0 / tl.sqrt(var + eps)

    # === Pass 2: normalize, apply affine, and SiLU ===
    for c_offset in range(group_size):
        c_idx = g_idx * group_size + c_offset
        base = n_idx * C * spatial_size + c_idx * spatial_size

        weight_val = tl.load(weight_ptr + c_idx).to(tl.float32)
        bias_val = tl.load(bias_ptr + c_idx).to(tl.float32)

        for s_start in tl.range(0, spatial_size, BLOCK_SIZE, num_stages=num_stages):
            offsets = s_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < spatial_size

            x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
            x_val = x_val.to(tl.float32)

            # Normalize and apply affine
            norm_val = (x_val - mean) * rstd * weight_val + bias_val

            # Apply SiLU: x * sigmoid(x)
            out_val = norm_val * tl.sigmoid(norm_val)

            # ``tl.store`` casts to the output pointer's dtype, which the
            # caller picked to match eager GroupNorm's autocast behaviour.
            tl.store(out_ptr + base + offsets, out_val, mask=mask)


# Spatial rows per program in the channels-last kernels, as an element count: the tile is
# (NHWC_TILE / BLOCK_C) x BLOCK_C. Swept on PVC at the VAE decoder's four norm shapes; 32768
# was the best of {2048 ... 65536} everywhere, by 1.4x over 8192 at 128ch, and 65536 falls off
# a cliff (2.7x worse) once the fp32 tile stops fitting in registers.
NHWC_TILE = 32768


def _nhwc_tiling(C: int) -> tuple[int, int]:
    """BLOCK_C, BLOCK_S for the channels-last kernels.

    BLOCK_C covers all channels at once, which is the whole point of the NHWC form: a
    group's channels are innermost, so reducing one group in isolation would load
    ``C/num_groups`` contiguous elements per spatial position (8 bytes of every 256-byte
    line at 128ch) -- exactly the uncoalesced access that makes aten's channels-last
    group_norm 6.6x slower than its NCHW path. Taking every channel and accumulating all
    groups' moments side by side keeps every load dense.
    """
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = max(1, NHWC_TILE // BLOCK_C)
    return BLOCK_C, BLOCK_S


@triton.jit
def _gn_silu_nhwc_stats_kernel(
    x_ptr,
    mean_ptr,
    m2_ptr,
    spatial_size,
    C,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    num_blocks: tl.constexpr,
):
    """Per-channel (mean, M2) of one BLOCK_S x C tile of x, which is (N, S, C).

    Per *channel*, not per group: a group's moments are assembled afterwards by merging its
    channels, which is exact because Chan's parallel-Welford merge is associative over any
    partition of the samples. One tile per program -- the sweep found no gain from looping
    over several, and a flat grid keeps every program's access a single dense block.
    """
    n_idx = tl.program_id(0)
    blk = tl.program_id(1)

    offs_s = blk * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = tl.arange(0, BLOCK_C)
    s_mask = offs_s < spatial_size
    c_mask = offs_c < C

    ptrs = x_ptr + n_idx * spatial_size * C + offs_s[:, None] * C + offs_c[None, :]
    x_val = tl.load(ptrs, mask=s_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

    cnt = tl.sum(s_mask.to(tl.float32), axis=0)
    mean = tl.sum(x_val, axis=0) / cnt
    centred = x_val - mean[None, :]
    m2 = tl.sum(tl.where(s_mask[:, None], centred * centred, 0.0), axis=0)

    out_off = (n_idx * num_blocks + blk) * C + offs_c
    tl.store(mean_ptr + out_off, mean, mask=c_mask)
    tl.store(m2_ptr + out_off, m2, mask=c_mask)


@triton.jit
def _gn_silu_nhwc_merge_kernel(
    mean_ptr,
    m2_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    spatial_size,
    C,
    num_blocks,
    eps,
    BLOCK_S: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per (batch, group): merge that group's partials into a mean/rstd, then
    write the per-channel scale/shift the apply pass consumes.

    Doing this in Triton rather than a dozen small torch reductions is what makes the small
    shapes worth it -- at 1024ch@2x128^2 the whole op is 0.2 GB, so ~15 extra launches were
    costing more than the normalisation itself.

    The per-block sample counts are recomputed here instead of being carried out of the
    stats kernel: they are ``min(BLOCK_S, S - k*BLOCK_S)`` by construction, so passing them
    through memory would only add traffic.
    """
    n_idx = tl.program_id(0)
    g_idx = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_G)
    g_mask = offs_g < GROUP_SIZE
    c_idx = g_idx * GROUP_SIZE + offs_g

    n_run = tl.zeros([1], dtype=tl.float32)
    mean_run = tl.zeros([1], dtype=tl.float32)
    m2_run = tl.zeros([1], dtype=tl.float32)

    for k0 in range(0, num_blocks, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < num_blocks
        # rows the stats kernel actually reduced for block k, zero where masked out so the
        # whole tile can join the merge unconditionally
        rows = tl.minimum(BLOCK_S, spatial_size - offs_k * BLOCK_S)
        cnt = tl.where(k_mask, tl.maximum(rows, 0), 0).to(tl.float32)

        off = n_idx * num_blocks * C + offs_k[:, None] * C + c_idx[None, :]
        tile_mask = k_mask[:, None] & g_mask[None, :]
        m = tl.load(mean_ptr + off, mask=tile_mask, other=0.0)
        v = tl.load(m2_ptr + off, mask=tile_mask, other=0.0)
        w = tl.where(tile_mask, cnt[:, None], 0.0)

        tile_n = tl.sum(tl.sum(w, axis=1), axis=0)
        tile_mean = tl.sum(tl.sum(w * m, axis=1), axis=0) / tile_n
        dev = m - tile_mean
        tile_m2 = tl.sum(tl.sum(v + w * dev * dev, axis=1), axis=0)

        delta = tile_mean - mean_run
        new_n = n_run + tile_n
        mean_run = mean_run + delta * (tile_n / new_n)
        m2_run = m2_run + tile_m2 + delta * delta * (n_run * tile_n / new_n)
        n_run = new_n

    rstd = 1.0 / tl.sqrt(m2_run / n_run + eps)
    w_val = tl.load(weight_ptr + c_idx, mask=g_mask, other=0.0).to(tl.float32)
    b_val = tl.load(bias_ptr + c_idx, mask=g_mask, other=0.0).to(tl.float32)
    scale = rstd * w_val
    tl.store(scale_ptr + n_idx * C + c_idx, scale, mask=g_mask)
    tl.store(shift_ptr + n_idx * C + c_idx, b_val - mean_run * scale, mask=g_mask)


@triton.jit
def _gn_silu_nhwc_apply_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    shift_ptr,
    spatial_size,
    C,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """out = silu(x * scale + shift): normalisation and affine arrive already folded into one
    per-channel scale/shift pair, so this pass is a plain fused multiply-add."""
    n_idx = tl.program_id(0)
    blk = tl.program_id(1)

    offs_s = blk * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    mask = (offs_s < spatial_size)[:, None] & c_mask[None, :]

    scale = tl.load(scale_ptr + n_idx * C + offs_c, mask=c_mask, other=0.0)
    shift = tl.load(shift_ptr + n_idx * C + offs_c, mask=c_mask, other=0.0)

    off = n_idx * spatial_size * C + offs_s[:, None] * C + offs_c[None, :]
    x_val = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    y = x_val * scale[None, :] + shift[None, :]
    tl.store(out_ptr + off, y * tl.sigmoid(y), mask=mask)


def _fused_group_norm_silu_nhwc(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """The channels-last path: same function, channels-last in and channels-last out.

    Keeping the layout is the point. A channels-last decoder can then chain convs and norms
    without ever materialising an NCHW copy, which is the only way a channels-last VAE
    decode pays at all: the conversion back off the 128ch activation costs 78 ms on its own,
    against a ~18 ms conv win per level.
    """
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    group_size = C // num_groups

    # (N, C, *spatial) channels-last -> (N, S, C) contiguous. A pure view: the bytes of a
    # channels-last tensor already are (N, S, C).
    x_flat = x.permute(0, *range(2, x.ndim), 1).reshape(N, -1, C)
    spatial_size = x_flat.size(1)

    BLOCK_C, BLOCK_S = _nhwc_tiling(C)
    num_blocks = triton.cdiv(spatial_size, BLOCK_S)
    opts = {"device": x.device, "dtype": torch.float32}
    mean = torch.empty((N, num_blocks, C), **opts)
    m2 = torch.empty((N, num_blocks, C), **opts)
    scale = torch.empty((N, C), **opts)
    shift = torch.empty((N, C), **opts)

    _gn_silu_nhwc_stats_kernel[(N, num_blocks)](
        x_flat, mean, m2, spatial_size, C,
        BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, num_blocks=num_blocks, num_warps=16,
    )
    _gn_silu_nhwc_merge_kernel[(N, num_groups)](
        mean, m2, weight, bias, scale, shift, spatial_size, C, num_blocks, eps,
        BLOCK_S=BLOCK_S, GROUP_SIZE=group_size,
        BLOCK_G=triton.next_power_of_2(group_size),
        BLOCK_K=min(1024, triton.next_power_of_2(num_blocks)), num_warps=4,
    )

    out_dtype = x.dtype
    if torch.is_autocast_enabled(x.device.type):
        out_dtype = torch.float32
    out_flat = torch.empty_like(x_flat, dtype=out_dtype)
    _gn_silu_nhwc_apply_kernel[(N, num_blocks)](
        x_flat, out_flat, scale, shift, spatial_size, C,
        BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, num_warps=16,
    )

    # back to (N, C, *spatial) as a view, in the input's memory format
    return out_flat.reshape(N, *spatial, C).permute(0, x.ndim - 1, *range(1, x.ndim - 1))


def _is_channels_last(x: torch.Tensor) -> bool:
    if x.ndim == 4:
        return x.is_contiguous(memory_format=torch.channels_last)
    if x.ndim == 5:
        return x.is_contiguous(memory_format=torch.channels_last_3d)
    return False


def fused_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused GroupNorm + SiLU activation.
    Computes SiLU(GroupNorm(x, num_groups, weight, bias, eps)) in a single Triton kernel,
    avoiding intermediate tensors and reducing memory traffic and launch overhead.

    - x: (N, C, *spatial), any spatial rank.
    - weight, bias: per-channel parameters, shape (C,).
    - num_groups: GroupNorm group count, default 32.
    - eps: numerical stability epsilon, default 1e-6.
    Uses fp32 accumulation for numeric alignment with PyTorch.
    Returns the same shape and eager F.group_norm-compatible dtype.
    Spatial dimensions are flattened during computation and restored afterward;
    this is exact since GroupNorm reduces across each channel group and all spatial positions.
    Non-contiguous inputs are materialized before launch.
    """
    # Fallback if Triton not available (NPU, CPU, ...)
    if not HAS_TRITON:
        return F.silu(F.group_norm(x, num_groups, weight, bias, eps))

    # Validate inputs
    assert x.ndim >= 3, f"Expected at least 3D input (N, C, *spatial), got {x.ndim}D"
    assert x.size(1) % num_groups == 0, f"Channels {x.size(1)} must be divisible by num_groups {num_groups}"
    assert weight.ndim == 1 and weight.size(0) == x.size(1), (
        f"Weight shape {weight.shape} doesn't match channels {x.size(1)}"
    )
    assert bias.ndim == 1 and bias.size(0) == x.size(1), f"Bias shape {bias.shape} doesn't match channels {x.size(1)}"

    # A channels-last input gets the channels-last kernel, which returns a channels-last
    # result. Falling through to the NCHW path would work, but the ``contiguous()`` below
    # would copy the activation in and the caller's next conv would copy it back out --
    # 88 ms per norm at the VAE decoder's 128ch shape, which is the whole budget.
    if _is_channels_last(x):
        return _fused_group_norm_silu_nhwc(x, weight, bias, num_groups, eps)

    # Collapse arbitrary spatial ranks into a single axis so one kernel serves
    # both the 2D DiT blocks and the 3D VAE blocks.
    #
    # ``contiguous()`` is not just defensive. HunyuanImage3's UNetUp feeds this
    # op straight out of ``rearrange(x, "b (h w) c -> b c h w")``, i.e. a
    # permuted view whose *channel* stride is 1. Indexing that layout directly
    # from the kernel makes every warp's spatial load stride by C elements, and
    # the resulting uncoalesced traffic turned the op into 0.44x of eager at
    # (1, 4096, 64, 64). One coalesced pre-pass costs far less than that, so
    # normalize the layout here and let the kernel assume a dense block.
    orig_shape = x.shape
    B, C = orig_shape[0], orig_shape[1]
    x_flat = x.contiguous().reshape(B, C, -1)
    spatial_size = x_flat.size(2)

    # Allocate output with the dtype eager GroupNorm would return, so that the
    # fused path stays a drop-in replacement inside autocast regions.
    out_dtype = x_flat.dtype
    if torch.is_autocast_enabled(x_flat.device.type):
        out_dtype = torch.float32
    out_flat = torch.empty_like(x_flat, dtype=out_dtype)

    # Only B*num_groups programs are launched, which is well under the SM count
    # for typical diffusion batches. A memory-bound kernel can still saturate
    # HBM from few CTAs, but only with enough loads in flight, so widen the CTA
    # for the large activations instead of leaving it at the 4-warp default.

    # One program per (batch, group) pair.
    grid = (B * num_groups,)

    _group_norm_silu_kernel[grid](
        x_flat,
        out_flat,
        weight,
        bias,
        C,
        spatial_size,
        num_groups=num_groups,
        eps=eps,
    )

    return out_flat.reshape(orig_shape)
