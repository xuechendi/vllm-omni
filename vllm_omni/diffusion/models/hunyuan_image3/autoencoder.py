# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from torch import Tensor, nn

# ResnetBlock is platform-dispatched in the layers package __init__: CUDA gets a
# fused GroupNorm+SiLU kernel, every other backend gets the plain PyTorch block.
from vllm_omni.diffusion.models.hunyuan_image3.layers import ResnetBlock
from vllm_omni.model_executor.models.common.ops import (
    Conv3dChannelsLast,
    fused_group_norm_silu,
    pixel_shuffle_channels_last,
    to_channels_last_3d,
)
from vllm_omni.platforms import current_omni_platform

CL3 = torch.channels_last_3d

# Run the *decoder's* upsampling stack on channels-last (NDHWC) activations.
#
# Every conv in the decoder is 3x3x3 over a temporal extent of 4, a shape oneDNN's 3D kernel only
# reaches 32-48% of peak on; rewriting it as 3 batched conv2d is 1.60-1.92x, but only in
# channels-last, and the conversion back off the 128ch activation costs 78 ms, so the layout has
# to be carried across the whole stack rather than wrapped around individual convs. Everything
# that layout touches then dispatches on it (``ResnetBlock``, ``Conv3dChannelsLast``,
# ``fused_group_norm_silu``), so this flag is the single switch, and with it off the decoder runs
# exactly the ops it ran before.
#
# Measured on one PVC, bf16, 1024x1024 (z = [1, 32, 1, 64, 64]): **1138 -> 598 ms, 1.90x**, with
# every upsampling level between 1.53x and 3.4x. Both bf16 paths sit the same distance from an
# fp32 reference (1.16e-02 vs 1.14e-02 rel L2), i.e. the rewrite costs time, not accuracy --
# ``tests/hy-image3/test_vae_decode_channels_last.py`` is that check. A second host reproduces it
# at 1177 -> 600 ms (1.96x), and in the AR+DiT server the profiler's ``vae.decode`` goes
# 1.398 -> 0.751 s (1.86x, or ~1.89x once the ~4% cross-instance drift that the untouched AR and
# DiT metrics show in the same pair of runs is taken out), i.e. 12% off the request's wall time.
#
# The switch is per decoder instance (``Decoder.decode_channels_last``), not per process, so a
# caller can hold two decoders with different layouts and a test needs no monkeypatching. Three
# places set it, in precedence order:
#
#   1. ``od_config.vae_decode_channels_last`` -- deploy YAML / engine arg, plumbed by the
#      pipeline. Left unset (``None``) it means "use the default", not "off".
#   2. ``HY3_VAE_DECODE_CHANNELS_LAST=0|1`` -- process-wide default, for A/B runs against an
#      otherwise identical deployment.
#   3. the platform: on for XPU, where the numbers above come from. CUDA has its own ResnetBlock
#      with plain ``nn.Conv3d``, so it would pick up the channels-last norms without the conv
#      rewrite they are supposed to pay for; it needs its own measurement before being enabled.
DECODE_CHANNELS_LAST_ENV = "HY3_VAE_DECODE_CHANNELS_LAST"


def resolve_decode_channels_last(configured: bool | None = None) -> bool:
    """Whether a decoder should run its upsampling stack channels-last.

    ``configured`` is an explicit choice from a config object, or ``None`` to fall back to
    ``HY3_VAE_DECODE_CHANNELS_LAST`` and then to the platform default.
    """
    if configured is not None:
        return bool(configured)
    env = os.environ.get(DECODE_CHANNELS_LAST_ENV)
    if env is not None:
        return env.strip() == "1"
    return current_omni_platform.is_xpu()


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: torch.Generator | None = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor
    posterior: DiagonalGaussianDistribution | None = None


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


class AttnBlock(nn.Module):
    """Attention with torch sdpa implementation."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # The memory concern is already handled via VAE tiling,
        # so there's no need to chunk here — calling nn.Conv3d directly is fine.
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class DownsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = nn.Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h + shortcut


class UpsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3dChannelsLast(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if h.is_contiguous(memory_format=CL3):
            return self._shuffle_channels_last(x, h, r1)
        h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut

    def _shuffle_channels_last(self, x: Tensor, h: Tensor, r1: int) -> Tensor:
        """The NDHWC form of the branch above, with two rewrites that the layout makes free.

        The shuffle is a permutation, so ``shuffle(h) + shuffle(shortcut) == shuffle(h + shortcut)``
        and only one shuffle is needed. And the shortcut's ``repeat_interleave`` never has to be
        materialised: interleaving by ``repeats`` along the channel axis is exactly splitting
        NDHWC's innermost axis into ``(C, repeats)``, so the add broadcasts into a view of h.
        """
        c = x.shape[1]
        hv = h.permute(0, 2, 3, 4, 1).unflatten(-1, (c, self.repeats))
        hv.add_(x.permute(0, 2, 3, 4, 1).unsqueeze(-1))
        return pixel_shuffle_channels_last(h, r1, 2, 2)


class Encoder(nn.Module):
    """
    The encoder network of AutoencoderKLConv3D.
    """

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(
                    self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing
                )
            if hasattr(self.down[i_level], "downsample"):
                h = forward_with_checkpointing(self.down[i_level].downsample, h, use_checkpointing=use_checkpointing)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    """
    The decoder network of AutoencoderKLConv3D.
    """

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
        decode_channels_last: bool | None = None,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        # Plain attribute, so whoever built this decoder through ``from_config`` can still flip it
        # afterwards; see ``AutoencoderKLConv3D.decode_channels_last``.
        self.decode_channels_last = resolve_decode_channels_last(decode_channels_last)

        # z to block_in
        block_in = block_out_channels[0]
        self.conv_in = Conv3dChannelsLast(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3dChannelsLast(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z: Tensor) -> Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # Hand the upsampling stack a channels-last activation and every block below follows it:
        # the conversion is paid here, at the smallest activation in the decoder, rather than
        # around each conv. The middle stays NCDHW because AttnBlock's sdpa wants NCDHW anyway
        # and it runs at 1/256th of the last level's size.
        if self.decode_channels_last and not use_checkpointing:
            h = to_channels_last_3d(h)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)

        # end
        if h.is_contiguous(memory_format=CL3):
            h = fused_group_norm_silu(h, self.norm_out.weight, self.norm_out.bias, 32, self.norm_out.eps)
            # Only the 3-channel RGB result goes back to NCDHW. That direction is the expensive
            # one (78 ms on the 128ch activation), but here it is 25 MB, and callers -- tiling,
            # blending, `decode`'s frame slice -- all assume the standard layout.
            return self.conv_out(h).contiguous()

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

    @torch.no_grad()
    def warmup(self, latent_hw: int = 8) -> None:
        """One throwaway decode, so that the first real one is not the one paying for setup.

        The channels-last path costs a process ~3 s the first time it runs, most of it in the
        very first Triton launch (2.5 s on its own, independent of the on-disk kernel cache:
        3.80 s with 399 entries already there against 4.17 s with none). On top of that, any of
        these kernels the host has never compiled has to be compiled now -- once per host, not
        once per server -- and unwarmed all of it landed on a user's first request: that decode
        was 11.2 s against a 0.60 s steady state on a host with the kernels cached, and 23.8 s
        on one seeing them for the first time.

        None of it is per-shape, so a decode at a 128px latent (a 25 MB activation, ~44 ms once
        warm) pays it: after this, the production 1024x1024 decode's first call is 1.49 s and
        settles at 0.60 s, against 1.93 s and 1.14 s for NCDHW. The residual ~0.9 s on that
        first call is oneDNN building conv2d primitives for the real shapes.
        """
        if not self.decode_channels_last:
            return
        ref = self.conv_in.weight
        # zeros, not randn: warmup must not touch the global RNG that seeded generation. A zero
        # variance is harmless here -- GroupNorm's eps keeps rstd finite and nothing is checked.
        z = torch.zeros(1, self.z_channels, 1, latent_hw, latent_hw, device=ref.device, dtype=ref.dtype)
        self(z)


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    """
    Autoencoder model with KL-regularized latent space based on 3D convolutions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float | None = None,
        shift_factor: float | None = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        only_encoder: bool = False,  # only build encoder for saving memory
        only_decoder: bool = False,  # only build decoder for saving memory
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        # build model
        if not only_decoder:
            self.encoder = Encoder(
                in_channels=in_channels,
                z_channels=latent_channels,
                block_out_channels=block_out_channels,
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                downsample_match_channel=downsample_match_channel,
            )
        if not only_encoder:
            self.decoder = Decoder(
                z_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=list(reversed(block_out_channels)),
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                upsample_match_channel=upsample_match_channel,
            )

        # slicing and tiling related
        self.use_slicing = False
        self.slicing_bsz = 1
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False
        self.use_tiling_during_training = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

        # use torch.compile for faster encode speed
        self.use_compile = False

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        """spatial tailing for frames"""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))  # 256 * (1 - 0.25) = 192
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)  # 8 * 0.25 = 2
        row_limit = self.tile_latent_min_size - blend_extent  # 8 - 2 = 6

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        """temporal tailing for frames"""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))  # 64 * (1 - 0.25) = 48
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)  # 8 * 0.25 = 2
        t_limit = self.tile_latent_min_tsize - blend_extent  # 8 - 2 = 6

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        moments = torch.cat(result_row, dim=-3)
        return moments

    def spatial_tiled_decode(self, z: torch.Tensor):
        """spatial tailing for frames"""
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)  # 256 * 0.25 = 64
        row_limit = self.tile_sample_min_size - blend_extent  # 256 - 64 = 192

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: torch.Tensor):
        """temporal tailing for frames"""
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)  # 64 * 0.25 = 16
        t_limit = self.tile_sample_min_tsize - blend_extent  # 64 - 16 = 48
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        dec = torch.cat(result_row, dim=-3)
        return dec

    def encode(self, x: Tensor, return_dict: bool = True):
        """
        Encodes the input by passing through the encoder network.
        Support slicing and tiling for memory efficiency.
        """

        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (
                x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.spatial_tiled_encode(x)

            if self.use_compile:

                @torch.compile
                def encoder(x):
                    return self.encoder(x)

                return encoder(x)
            return self.encoder(x)

        if len(x.shape) != 5:  # (B, C, T, H, W)
            x = x[:, :, None]
        assert len(x.shape) == 5  # (B, C, T, H, W)
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        else:
            assert x.shape[2] != self.ffactor_temporal and x.shape[2] % self.ffactor_temporal == 0

        if self.use_slicing and x.shape[0] > 1:
            if self.slicing_bsz == 1:
                encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            else:
                sections = [self.slicing_bsz] * (x.shape[0] // self.slicing_bsz)
                if x.shape[0] % self.slicing_bsz != 0:
                    sections.append(x.shape[0] % self.slicing_bsz)
                encoded_slices = [_encode(x_slice) for x_slice in x.split(sections)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @property
    def decode_channels_last(self) -> bool:
        """Whether ``decode`` runs the upsampling stack on channels-last activations."""
        return getattr(self, "decoder", None) is not None and self.decoder.decode_channels_last

    @decode_channels_last.setter
    def decode_channels_last(self, enabled: bool | None) -> None:
        """Select the decoder's layout. ``None`` restores the default (env, then platform).

        Set from ``od_config.vae_decode_channels_last`` at load time, next to the other VAE
        knobs, and safe to flip afterwards: nothing is cached across calls except the weights'
        own layout, which each conv fixes up on its first channels-last call.
        """
        if getattr(self, "decoder", None) is not None:
            self.decoder.decode_channels_last = resolve_decode_channels_last(enabled)

    def warmup_decode(self) -> None:
        """Pay the decoder's one-time setup cost at load time. A no-op unless channels-last is on."""
        if getattr(self, "decoder", None) is not None:
            self.decoder.warmup()

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):
        """
        Decodes the input by passing through the decoder network.
        Support slicing and tiling for memory efficiency.
        """

        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if z.shape[-3] == 1:
            decoded = decoded[:, :, -1:]

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
