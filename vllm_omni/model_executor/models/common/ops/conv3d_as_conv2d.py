# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""conv3d rewritten as a sum of batched conv2d, for channels-last activations.

A stride-1 conv3d whose depth padding is ``kD // 2`` is algebraically kD conv2d plus kD-1
shifted adds, with the frames riding in the batch dimension::

    out[:, :, t] = sum_j conv2d(x[:, :, t + j - kD//2], weight[:, :, j])

Identical FLOPs, and on an NDHWC buffer both the reshape into per-frame NHWC images and the
reshape back are pure views. It is worth doing because the 3D kernel is the slow one: every
conv in HunyuanImage-3.0's VAE decoder is 3x3x3 over a temporal extent of only **4** (the
image path encodes one frame and ``ffactor_temporal`` expands it), which leaves oneDNN's
implicit-GEMM almost nothing to block along depth. Measured on one PVC, bf16, against the
300 TFLOPs peak the kernel bench scores conv against:

    shape                       conv3d NCDHW      3x conv2d NHWC
    128ch  @4x1024^2      38.13 ms  32% peak    20.49 ms  60% peak   1.86x
    256ch  @4x512^2       34.08 ms  36% peak    17.77 ms  70% peak   1.92x
    512ch  @4x256^2       27.40 ms  45% peak    15.64 ms  79% peak   1.75x
    256->512 @4x512^2     64.11 ms  39% peak    40.06 ms  62% peak   1.60x
    512->1024 @4x256^2    51.86 ms  48% peak    30.68 ms  81% peak   1.69x

The layout is not optional: the same decomposition in NCHW is a *loss* (0.37x at 128ch, where
one NCHW tap costs 32.68 ms against NHWC's 5.01), so this only pays inside a decoder that
carries channels-last throughout.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_CL3 = torch.channels_last_3d


def _as_frames(x: Tensor) -> Tensor:
    """(N, C, D, H, W) channels_last_3d -> (N*D, C, H, W) channels_last. A pure view."""
    n, c, d, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(n * d, h, w, c).permute(0, 3, 1, 2)


def _as_volume(x: Tensor, n: int, d: int) -> Tensor:
    """(N*D, C, H, W) channels_last -> (N, C, D, H, W) channels_last_3d. A pure view."""
    _, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(n, d, h, w, c).permute(0, 4, 1, 2, 3)


def conv3d_as_conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (1, 1, 1),
) -> Tensor:
    """conv3d(x, weight, bias) as kD batched conv2d over the frame axis.

    ``x`` must be channels_last_3d and ``weight`` channels_last_3d; the result is
    channels_last_3d. Exact up to accumulation order -- the frames that depth padding would
    have zeroed are simply dropped from each shifted tap's range.
    """
    n, d = x.shape[0], x.shape[2]
    kd = weight.shape[2]
    pad_d = padding[0]
    spatial = (padding[1], padding[2])
    frames = _as_frames(x)

    # weight[:, :, j] of an ODHWI weight is already an OHWI-contiguous 2D kernel, so the taps
    # cost no copy and no extra memory. The centre tap needs no shift, so it seeds the
    # accumulator and carries the bias; the rest are added in place over their valid range,
    # which for a channels-last volume is a dense slice.
    out = _as_volume(F.conv2d(frames, weight[:, :, pad_d], bias, stride[1:], spatial), n, d)
    for j in range(kd):
        off = j - pad_d
        if off == 0:
            continue
        lo, hi = max(0, -off), d - max(0, off)
        if hi <= lo:
            continue
        tap = _as_volume(F.conv2d(frames, weight[:, :, j], None, stride[1:], spatial), n, d)
        out[:, :, lo:hi] += tap[:, :, lo + off : hi + off]
    return out


class Conv3dChannelsLast(nn.Conv3d):
    """``nn.Conv3d`` that takes the conv2d decomposition when it sees a channels-last input.

    A drop-in replacement: same parameters, same shapes, same ``state_dict``, and an NCDHW
    input falls straight through to ``nn.Conv3d.forward``, so a decoder that has not opted
    into channels-last is bit-identical to before. The weight is flipped to channels_last_3d
    on the first channels-last call, which is also what makes the per-tap 2D kernels free
    views rather than copies.
    """

    def _supports_taps(self) -> bool:
        return (
            self.groups == 1
            and all(s == 1 for s in self.stride)
            and all(dl == 1 for dl in self.dilation)
            and self.padding_mode == "zeros"
            # the tap ranges below assume the depth padding keeps D unchanged
            and all(p == k // 2 for p, k in zip(self.padding, self.kernel_size, strict=True))
        )

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_contiguous(memory_format=_CL3) or not self._supports_taps():
            return super().forward(x)
        if not self.weight.is_contiguous(memory_format=_CL3):
            self.weight.data = self.weight.data.contiguous(memory_format=_CL3)
        return conv3d_as_conv2d(x, self.weight, self.bias, self.stride, self.padding)


def to_channels_last_3d(x: Tensor) -> Tensor:
    """``x`` as a channels_last_3d tensor, via an explicit permute.

    ``Tensor.to(memory_format=channels_last_3d)`` and this are within noise of each other in
    the cheap direction (10.5 vs 10.9 ms on 1 GB), but the explicit form is the one that
    stays a no-op when the input is already channels-last.
    """
    if x.is_contiguous(memory_format=_CL3):
        return x
    return x.to(memory_format=_CL3)


def pixel_shuffle_channels_last(x: Tensor, r1: int, r2: int, r3: int) -> Tensor:
    """``rearrange(x, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)")`` for NDHWC input.

    einops on a channels-last tensor falls off a cliff -- 343 ms at (1, 512, 4, 1024, 1024)
    against 19.7 ms for the NCDHW original -- because it reasons about the logical strides and
    ends up with a 2-byte innermost run. Written as an explicit 7-D permute the innermost run
    is C elements (256 B at 128ch) and the copy runs at 449-762 GB/s, i.e. at or below the
    NCDHW cost with none of the layout conversion.
    """
    n, cc, d, h, w = x.shape
    c = cc // (r1 * r2 * r3)
    v = x.permute(0, 2, 3, 4, 1).reshape(n, d, h, w, r1, r2, r3, c)
    v = v.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(n, d * r1, h * r2, w * r3, c)
    return v.contiguous().permute(0, 4, 1, 2, 3)
