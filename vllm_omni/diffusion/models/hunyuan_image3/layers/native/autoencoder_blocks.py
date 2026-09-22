# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ResnetBlock for the HunyuanImage3 autoencoder — default fallback implementation.

Uses native PyTorch ``nn.GroupNorm`` + ``F.silu``. Hardware-specific optimized
versions live in subdirectories (e.g., nvidia/) and are selected via __init__.py
dispatch.

The fast path here is selected by the *activation layout*, not by platform: if the block is
handed a ``channels_last_3d`` tensor it uses the channels-last fused GroupNorm+SiLU kernel and
the conv2d decomposition of its conv3d, and returns channels-last. An NCDHW input runs exactly
the code it always did. ``Decoder.forward`` is what decides, once, which it will be.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.common.ops import Conv3dChannelsLast, fused_group_norm_silu

_CL3 = torch.channels_last_3d


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv3dChannelsLast(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv3dChannelsLast(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3dChannelsLast(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def _norm_silu(self, norm: nn.GroupNorm, h):
        # aten's channels-last group_norm is 6.6x slower than its NCHW path at these shapes, so
        # a channels-last activation has to go through the fused kernel rather than eager.
        if h.is_contiguous(memory_format=_CL3):
            return fused_group_norm_silu(h, norm.weight, norm.bias, norm.num_groups, norm.eps)
        return F.silu(norm(h))

    def forward(self, x):
        h = self._norm_silu(self.norm1, x)
        h = self.conv1(h)

        h = self._norm_silu(self.norm2, h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


__all__ = ["ResnetBlock"]
