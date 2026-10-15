# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from contextlib import nullcontext
from types import MethodType
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanDecoder3d,
    WanEncoder3d,
    WanResidualBlock,
    WanRMS_norm,
    unpatchify,
)
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    can_encode_frames,
    decode_frames,
    encode_frames,
    is_encoder_installed,
    is_installed,
)
from vllm_omni.diffusion.models.interface import DecodedChunkConsumer
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


# torch.compile the RMS norms of the encoder and decoder, each fused with the SiLU that follows
# it; on by default, set to 0 for the eager norms. Ring4 720p/61f: encode 3.22 -> 2.58 s, decode
# 3.01 -> 2.83 s, for ~45 s of one-time compilation on the first request.
COMPILE_ENV = "VLLM_OMNI_WAN_VAE_COMPILE"


def _compile_enabled() -> bool:
    return os.environ.get(COMPILE_ENV, "1") == "1"


def _rms_norm_act(x: torch.Tensor, weight: torch.Tensor, silu: bool) -> torch.Tensor:
    """``WanRMS_norm`` (channel-first, no bias) optionally followed by SiLU, as one kernel.

    Eager, the norm is five passes over the activation, and the SiLU after
    it is a sixth; compiled, it is one read and one write. ``weight`` is ``gamma * scale``
    precomputed in fp32, so the whole chain stays in fp32 registers and rounds to the
    activation dtype once -- closer to the fp32 reference than either eager path.
    """
    x32 = x.float()
    out = x32 / torch.linalg.vector_norm(x32, dim=1, keepdim=True).clamp_min(1e-12) * weight
    if silu:
        out = torch.nn.functional.silu(out)
    return out.to(x.dtype)


_compiled_rms_norm_act = None
# A 720p decode on a height shard sees dozens of distinct (shape, silu) keys, encode as many again,
# past dynamo's default of 8 recompiles. The higher limit is applied only around the VAE's own call:
# raising it globally let the DiT attention (which guards on a call counter) recompile on every
# step, ring8 diffuse 7.8 -> 35 s.
_VAE_RECOMPILE_LIMIT = 128
_LIMIT_NAME = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"


def _get_compiled_rms_norm_act():
    # One compiled function shared by every norm, compiled static: the dynamic-shape kernel dynamo
    # switches to after the first shape change measured 1.2-1.6x on the 720p shard shapes, the
    # per-shape static kernels 2.4-3.1x.
    global _compiled_rms_norm_act
    if _compiled_rms_norm_act is None:
        _compiled_rms_norm_act = torch.compile(_rms_norm_act, dynamic=False)
    return _compiled_rms_norm_act


def _compiled_rms_norm_forward(self: WanRMS_norm, x: torch.Tensor) -> torch.Tensor:
    fn = _get_compiled_rms_norm_act()
    with torch._dynamo.config.patch(**{_LIMIT_NAME: _VAE_RECOMPILE_LIMIT}):
        return fn(x, self._vllm_omni_norm_weight, self._vllm_omni_norm_silu)


def _compilable_norm(norm: Any) -> bool:
    return isinstance(norm, WanRMS_norm) and norm.channel_first and not torch.is_tensor(norm.bias) and not norm.bias


def _bind_compiled_norm(norm: WanRMS_norm, *, silu: bool) -> None:
    norm._vllm_omni_norm_weight = norm.gamma.detach().float() * norm.scale
    norm._vllm_omni_norm_silu = silu
    norm._vllm_omni_compiled = True
    norm.forward = MethodType(_compiled_rms_norm_forward, norm)


class OmniAutoencoderKLWan(AutoencoderKLWan):
    _vae_compile_installed = False

    def _install_vae_compile(self) -> None:
        """Swap every RMS norm of the encoder and decoder for the compiled ``_rms_norm_act``.

        A norm that a SiLU follows -- both norms of every residual block, and each head's
        ``norm_out`` -- absorbs it, and the owner's ``nonlinearity`` becomes an identity; that
        is the only place those modules apply it. The rest (the attention block's norm) are
        compiled without it. Convolutions stay eager.

        A component that ``wan_vae_fastpath`` is installed on is left alone: its block forwards
        call the norms through ``_norm_act`` rather than ``norm.forward``, so the SiLU moved into
        a compiled norm would be dropped. Its own fused norm kernel covers the same ground.
        """
        if self._vae_compile_installed:
            return
        self._vae_compile_installed = True
        if not _compile_enabled():
            return

        fused = plain = 0
        roots = [
            m
            for m, fast_path in (
                (getattr(self, "encoder", None), is_encoder_installed(self)),
                (getattr(self, "decoder", None), is_installed(self)),
            )
            if m is not None and not fast_path
        ]
        for root in roots:
            for module in root.modules():
                if isinstance(module, WanResidualBlock):
                    norms = [module.norm1, module.norm2]
                elif isinstance(module, (WanEncoder3d, WanDecoder3d)):
                    norms = [module.norm_out]
                else:
                    continue
                if not isinstance(module.nonlinearity, torch.nn.SiLU) or not all(map(_compilable_norm, norms)):
                    continue
                for norm in norms:
                    _bind_compiled_norm(norm, silu=True)
                    fused += 1
                module.nonlinearity = torch.nn.Identity()
            for module in root.modules():
                if _compilable_norm(module) and not getattr(module, "_vllm_omni_compiled", False):
                    _bind_compiled_norm(module, silu=False)
                    plain += 1
        logger.info("Compiled Wan VAE RMS norms: %d fused with SiLU, %d plain (encoder + decoder).", fused, plain)

    def _execution_context(self):
        try:
            first_param = next(self.parameters())
        except StopIteration:
            return nullcontext()

        dtype = first_param.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()

        return current_omni_platform.create_autocast_context(
            device_type=first_param.device.type,
            dtype=dtype,
            enabled=True,
        )

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        self._install_vae_compile()
        with self._execution_context():
            return super().encode(x, return_dict=return_dict)

    def _encode(self, x: torch.Tensor):
        # Keep the parent's tiling dispatch and patchified tile coordinates.
        # Only the untiled, supported inference schedule uses preallocation.
        if not is_encoder_installed(self):
            return super()._encode(x)
        if can_encode_frames(self, x):
            tiled = self.use_tiling and (
                x.shape[-2] > self.tile_sample_min_height or x.shape[-1] > self.tile_sample_min_width
            )
            if not tiled:
                return encode_frames(self, x)
        try:
            return super()._encode(x)
        finally:
            self.clear_cache()

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        """Decode a Wan latent using the Diffusers-compatible full-tensor API."""
        self._install_vae_compile()
        with self._execution_context():
            return super().decode(z, return_dict=return_dict)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        """Decode one latent batch, preserving the parent Diffusers contract."""
        _, _, _, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if getattr(self, "use_tiling", False) and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        return self._decode_temporal(z, return_dict=return_dict)

    # Chunks are published straight from the decoder, clamped to the
    # checkpoint's output range.
    chunk_value_range: ClassVar[tuple[float, float]] = (-1.0, 1.0)

    @apply_forward_hook
    def decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Decode ``z`` while synchronously delivering temporal chunks.

        The ordinary :meth:`decode` contract remains the Diffusers contract.
        Chunk callbacks deliberately reject a sliced batch: Diffusers slices
        each sample before decoding, while a streaming consumer needs a stable
        batch ownership contract. Callers can disable ``use_slicing`` or
        invoke this capability once per sample.

        Every rank participating in distributed VAE execution must invoke this
        method. Spatial-sharded execution invokes ``on_chunk`` only on the
        output-owning rank. Chunks have shape ``(B, C, T, H, W)`` and
        values clamped to ``[-1, 1]``. After a callback failure, remaining
        chunks are decoded and discarded before the exception is re-raised.
        """
        if not callable(on_chunk):
            raise TypeError("on_chunk must be callable")
        if getattr(self, "use_slicing", False) and z.shape[0] > 1:
            raise ValueError(
                "Wan chunked decode does not support use_slicing for batch size > 1; "
                "disable VAE slicing or decode one sample at a time"
            )

        with self._execution_context():
            self._decode_with_chunks(z, on_chunk=on_chunk)

    def _decode_temporal(
        self,
        z: torch.Tensor,
        *,
        on_chunk: DecodedChunkConsumer | None = None,
        return_dict: bool = True,
    ):
        """Decode the non-tiled temporal path, optionally streaming chunks."""
        if on_chunk is None and is_installed(self) and not torch.is_grad_enabled():
            try:
                out = decode_frames(self, z)
            finally:
                self.clear_cache()
            if not return_dict:
                return (out,)
            return DecoderOutput(sample=out)

        self.clear_cache()
        callback_error: BaseException | None = None
        try:
            x = self.post_quant_conv(z)
            decoded_chunks: list[torch.Tensor] = []
            for i in range(z.shape[2]):
                self._conv_idx = [0]
                if i == 0:
                    chunk = self.decoder(
                        x[:, :, i : i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        first_chunk=True,
                    )
                else:
                    chunk = self.decoder(
                        x[:, :, i : i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                    )

                if on_chunk is None:
                    decoded_chunks.append(chunk)
                    continue

                if callback_error is None:
                    try:
                        if self.config.patch_size is not None:
                            chunk = unpatchify(chunk, patch_size=self.config.patch_size)
                        on_chunk(torch.clamp(chunk, min=-1.0, max=1.0))
                    except BaseException as exc:
                        # A previously installed spatial-shard decoder still
                        # runs collectives, even for latents below the tiling
                        # threshold. Drain before surfacing consumer failures.
                        callback_error = exc
                del chunk

            if callback_error is not None:
                raise callback_error
            if on_chunk is not None:
                return None

            out = torch.cat(decoded_chunks, dim=2)
            if self.config.patch_size is not None:
                out = unpatchify(out, patch_size=self.config.patch_size)
            out = torch.clamp(out, min=-1.0, max=1.0)
            if not return_dict:
                return (out,)
            return DecoderOutput(sample=out)
        finally:
            self.clear_cache()

    @staticmethod
    def _discard_decoded_chunk(_chunk: torch.Tensor) -> None:
        """Consume a non-owner rank's chunk without retaining its tensor."""

    def _decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Run the temporal streaming primitive without API or hook handling."""
        distributed_enabled = False
        if hasattr(self, "is_distributed_enabled"):
            distributed_enabled = self.is_distributed_enabled()

        if getattr(self, "use_tiling", False):
            tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
            tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
            exceeds_tile = z.shape[-1] > tile_latent_min_width or z.shape[-2] > tile_latent_min_height
            if exceeds_tile:
                if distributed_enabled:
                    self._tiled_decode_with_chunks(z, on_chunk=on_chunk)
                    return
                raise ValueError("Wan chunk callbacks are unsupported for spatial tiling")

        callback = on_chunk
        if distributed_enabled and dist.get_rank(group=self.distributed_executor.group) != 0:
            callback = self._discard_decoded_chunk
        self._decode_temporal(z, on_chunk=callback)


class DistributedAutoencoderKLWan(OmniAutoencoderKLWan, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def retain_stage_components(self, *, encode: bool) -> None:
        """Prune the unused Wan half before device placement/offload discovery.

        Diffusers 0.40 caches encoder/decoder convolution counts separately;
        both encode and decode call clear_cache, but neither executes the other
        half. Keep the public attributes and zero only the removed cache count.
        This reduces resident weights, NOT from_pretrained's peak host memory:
        the ordinary checkpoint loader still loads both halves first.
        """
        removed = "decoder" if encode else "encoder"
        projection = "post_quant_conv" if encode else "quant_conv"
        if not isinstance(getattr(self, "_cached_conv_counts", None), dict):
            raise RuntimeError("Wan stage VAE pruning requires Diffusers' cached encoder/decoder conv counts.")
        setattr(self, removed, None)
        setattr(self, projection, None)
        self._cached_conv_counts[removed] = 0
        self.clear_cache()

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            blend_width = self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width

        tiletask_list: list[TileTask] = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                time_list = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )
        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_sample_stride_height": tile_sample_stride_height,
            "tile_sample_stride_width": tile_sample_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space."""
        self.clear_cache()
        time = []
        with self._execution_context():
            for k in range(len(task.tensor)):
                self._conv_idx = [0]
                tile = self.post_quant_conv(task.tensor[k])
                decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=(k == 0))
                time.append(decoded)
        result = torch.cat(time, dim=2)
        return result

    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = x.shape
        encode_spatial_compression_ratio = self.spatial_compression_ratio
        # Scale tile parameters for patchified coordinate system
        tile_sample_min_height = self.tile_sample_min_height
        tile_sample_min_width = self.tile_sample_min_width
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            assert encode_spatial_compression_ratio % self.config.patch_size == 0
            encode_spatial_compression_ratio = self.spatial_compression_ratio // self.config.patch_size
            # When input is patchified, scale tile parameters accordingly
            tile_sample_min_height = tile_sample_min_height // self.config.patch_size
            tile_sample_min_width = tile_sample_min_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size

        latent_height = height // encode_spatial_compression_ratio
        latent_width = width // encode_spatial_compression_ratio

        tile_latent_min_height = tile_sample_min_height // encode_spatial_compression_ratio
        tile_latent_min_width = tile_sample_min_width // encode_spatial_compression_ratio
        tile_latent_stride_height = tile_sample_stride_height // encode_spatial_compression_ratio
        tile_latent_stride_width = tile_sample_stride_width // encode_spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tiletask_list: list[TileTask] = []
        temporal_compression = self.config.scale_factor_temporal
        for i in range(0, height, tile_sample_stride_height):
            for j in range(0, width, tile_sample_stride_width):
                time_list = []
                frame_range = 1 + (num_frames - 1) // temporal_compression
                for k in range(frame_range):
                    if k == 0:
                        tile = x[:, :, :1, i : i + tile_sample_min_height, j : j + tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + temporal_compression * (k - 1) : 1 + temporal_compression * k,
                            i : i + tile_sample_min_height,
                            j : j + tile_sample_min_width,
                        ]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_sample_stride_height, j // tile_sample_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={
                "latent_height": latent_height,
                "latent_width": latent_width,
                "blend_height": blend_height,
                "blend_width": blend_width,
                "tile_latent_stride_height": tile_latent_stride_height,
                "tile_latent_stride_width": tile_latent_stride_width,
            },
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        """Encode a single sample tile into latent space."""
        self.clear_cache()
        try:
            time = []
            for tile in task.tensor:
                self._enc_conv_idx = [0]
                encoded = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                encoded = self.quant_conv(encoded)
                time.append(encoded)
            return torch.cat(time, dim=2)
        finally:
            self.clear_cache()

    def encode_tile_merge(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        """Merge encoded tiles into a full latent tensor."""
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_latent_stride_height"],
                        : grid_spec.tile_spec["tile_latent_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["latent_height"], : grid_spec.tile_spec["latent_width"]
        ]
        return enc

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full image."""
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        self.clear_cache()
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_sample_stride_height"],
                        : grid_spec.tile_spec["tile_sample_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
        ]

        if self.config.patch_size is not None:
            dec = unpatchify(dec, patch_size=self.config.patch_size)

        dec = torch.clamp(dec, min=-1.0, max=1.0)
        return dec

    def _spatial_shard_decode_split_dim(self) -> str | None:
        mode = self.distributed_executor.parallel_mode
        if mode == "spatial_shard_width":
            return "width"
        if mode == "spatial_shard_height":
            return "height"
        return None

    def _spatial_shard_decode_enabled(self, z: torch.Tensor) -> bool:
        split_dim = self._spatial_shard_decode_split_dim()
        if split_dim is None:
            return False
        if z.ndim != 5:
            logger.warning("Wan VAE spatial sharded decode expects 5D latent input; falling back to tiled decode.")
            return False
        if not self.is_distributed_enabled():
            return False

        group = self.distributed_executor.group
        world_size = dist.get_world_size(group=group)
        requested_size = int(self.distributed_executor.parallel_size)
        if requested_size != world_size:
            logger.warning(
                "Wan VAE spatial sharded decode currently requires vae_patch_parallel_size "
                "to match the DIT group size; falling back to tiled decode. "
                "requested=%s dit_group=%s split_dim=%s",
                requested_size,
                world_size,
                split_dim,
            )
            return False
        return True

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)

        spatial_shard_enabled = self._spatial_shard_decode_enabled(z)
        if spatial_shard_enabled:
            split_dim = self._spatial_shard_decode_split_dim()
            assert split_dim is not None
            logger.debug("Decode running with Wan VAE spatial_shard_%s mode", split_dim)
            return wan_spatial_shard.spatial_shard_decode(
                self,
                z,
                group=self.distributed_executor.group,
                return_dict=return_dict,
                split_dim=split_dim,
            )

        logger.debug("Decode running with distributed executor")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=False,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)

    def _tiled_decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Decode tiled latents and deliver temporal chunks to ``on_chunk``."""
        if not self.is_distributed_enabled():
            raise ValueError("Wan chunk callbacks are unsupported for spatial tiling")

        spatial_shard_enabled = self._spatial_shard_decode_enabled(z)
        if not spatial_shard_enabled:
            raise ValueError("Wan chunk callbacks require spatial-shard decode when tiling is enabled")

        split_dim = self._spatial_shard_decode_split_dim()
        assert split_dim is not None
        logger.debug("Decode running with Wan VAE spatial_shard_%s mode", split_dim)
        wan_spatial_shard.spatial_shard_decode(
            self,
            z,
            group=self.distributed_executor.group,
            return_dict=True,
            split_dim=split_dim,
            on_chunk=on_chunk,
        )

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using distributed VAE executor.

        Note: x is already patchified by parent's _encode() before calling this method.
        """
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)

        logger.debug("Encode running with distributed executor")
        self.clear_cache()
        result = self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
            broadcast_result=True,
        )
        self.clear_cache()
        return result
