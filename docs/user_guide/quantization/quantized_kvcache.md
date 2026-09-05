# Quantized KV Cache

## Overview

In DiT-based image and video generation, Flash Attention can take a large share
of denoising time, especially for high-resolution or long-frame workloads.
vLLM-Omni supports online FP8 quantization for eligible diffusion Flash
Attention (FA) to reduce FA latency while keeping model weights in their
original dtype.

This feature is configured through `diffusion_kv_cache_dtype` on
`OmniDiffusionConfig` (CLI: `--diffusion-kv-cache-dtype`). It is intentionally
**not** the same as vLLM's `--kv-cache-dtype`, which controls autoregressive
language-model KV cache storage and defaults to `"auto"`. Diffusion FA
quantization uses the dedicated diffusion flags so omni serve does not inherit
that default.

In vLLM-Omni diffusion pipelines, this is a runtime FA path: attention
activations are dynamically quantized before the attention operator. Which
tensors are quantized, and at what granularity, is backend-specific — the NPU
backend quantizes Q, K, and V with block scales, while the XPU backend uses
per-tensor scales and covers either Q/K/V or only K/V depending on the device and
kernel build (see [Hardware Support](#hardware-support)).
It does not quantize model weights and is separate from [FP8 W8A8](fp8.md),
[Int8 W8A8](int8.md), or pre-quantized checkpoint formats.

Despite the name, there is no persistent KV cache here: diffusion recomputes
Q/K/V every forward pass, so the flag quantizes activations rather than cache
storage. It is named by analogy to the autoregressive flag only.

If `diffusion_kv_cache_dtype` is not set, behavior is unchanged and attention
runs in the native dtype.

## Hardware Support

| Device | FP8 FA |
|--------|--------|
| Ascend NPU | ✅ |
| Intel XPU | ✅ |
| NVIDIA GPU | ❌ |
| AMD ROCm | ❌ |

Legend: `✅` supported, `❌` unsupported.

FP8 FA is implemented for the NPU and XPU Flash Attention backends. Other
backends do not support `diffusion_kv_cache_dtype="fp8"` for diffusion attention
and fall back to native dtype execution.

On NPU, `FLASH_ATTN` quantizes **Q, K, and V** to FP8 E4M3 with **block** scales
(block size 128 for Q, 256 for K/V), after a Hadamard/QuaRot rotation of Q and K,
and feeds `dequant_scale_{query,key,value}` to
`npu_fused_infer_attention_score_v2`.

On XPU, `FLASH_ATTN` uses **per-tensor** FP8 E4M3 with `{q,k,v}_descale` passed to
`flash_attn_varlen_func`. Scales are derived online from each tensor's own amax,
so no calibration step is required. Which tensors get quantized is decided at
runtime, because it depends on the device and the installed kernel wheel:

- **Full FP8** (Q, K and V): Q\*K and P\*V both run natively in FP8 — the q/k
  descales fold into the softmax scale and the softmax output is packed to e4m3.
  This is implemented only in the `xe_3` kernel, which the kernel library
  dispatches for `is_xe3p_arch()` (CRI and NVL-P), and it additionally needs a
  kernel wheel whose `flash_attn_varlen_func` accepts `q_descale`.
- **K/V only** (Q stays native): the fallback. The kernel dequantizes K and V
  per element inside the mainloop, so the GEMMs still run at native width and
  FP8 saves memory traffic rather than math. This is the only mode the `xe_2`
  kernel implements.

vLLM-Omni requests full FP8 and downgrades permanently on the first call that
proves the stack cannot serve it, logging which mode is in use. Expect a much
smaller speedup in the K/V-only mode; measure against the unquantized baseline
rather than assuming a gain.

Ring attention is rejected for any FP8 FA configuration because ring kernels do
not propagate descale factors; use Ulysses SP instead.

## Model Type Support

### Diffusion Model

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| Wan2.2 | Eligible DiT full-attention FA on Ascend NPU | Tested | Compare quality and latency against a BF16 baseline before production use |
| Wan2.2 | DiT self-attention FA on Intel XPU | Not tested | Cross-attention opts out via `Attention(disable_kv_quant=True)` — its text-encoder sequences are short, so FP8 buys no speedup and costs quality |
| Other diffusion models | Eligible DiT full-attention FA on Ascend NPU or Intel XPU | Not tested | You can try `diffusion_kv_cache_dtype="fp8"`; tune `diffusion_kv_cache_skip_steps` and `diffusion_kv_cache_skip_layers` when higher precision is needed |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

Not tested for FP8 FA. Treat any use as experimental unless a model-specific
guide documents support.

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

Not tested. If the diffusion stage uses the same NPU Flash Attention backend,
`diffusion_kv_cache_dtype` may apply in theory; validate quality and latency for
each stage and model.

## Configuration

Offline diffusion example:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
    --model <your-wan2.2-model> \
    --prompt "A cat sitting on a surfboard at the beach" \
    --height 1280 \
    --width 720 \
    --num-frames 61 \
    --num-inference-steps 4 \
    --ulysses-degree 4 \
    --vae-patch-parallel-size 4 \
    --diffusion-kv-cache-dtype fp8 \
    --diffusion-kv-cache-skip-steps "0,1" \
    --diffusion-kv-cache-skip-layers "0-2"
```

Online serving:

```bash
vllm serve <your-model> --omni --diffusion-kv-cache-dtype fp8
```

Deploy config:

```yaml
stages:
  - stage_id: 0
    diffusion_kv_cache_dtype: "fp8"
    diffusion_kv_cache_skip_steps: "0,1"
    diffusion_kv_cache_skip_layers: "0-2"
```

The `model_stage` and diffusion execution type belong to the registered
`PipelineConfig`; the deploy YAML only carries runtime overrides.

The legacy keyword aliases `kv_cache_dtype`, `kv_cache_skip_steps`, and
`kv_cache_skip_layers` remain accepted when constructing
`OmniDiffusionConfig` directly. They are not deploy YAML fields; prefer the
`diffusion_*` names for new code.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diffusion_kv_cache_dtype` | str \| None | `None` | Set to `"fp8"` to enable dynamic FP8 FA on supported attention backends |
| `diffusion_kv_cache_skip_steps` | str \| None | `None` | Denoising step selector to keep in native dtype, for example `"0,1,4-6"` |
| `diffusion_kv_cache_skip_layers` | str \| None | `None` | Transformer layer selector to keep in native dtype, for example `"0-2,10"` |

Selectors use comma-separated integers and inclusive ranges. Listed steps or
layers skip FP8 FA; all other eligible full-attention forwards use the FP8 path.

## Validation and Notes

1. Compare generated images or videos against a BF16 baseline with the same
   seed, prompt, resolution, frame count, and denoising steps.
2. Use `diffusion_kv_cache_skip_steps` for denoising steps where quality is more
   sensitive.
3. Use `diffusion_kv_cache_skip_layers` for transformer layers that show visible quality
   regressions.
4. Report both latency and quality results when enabling this option for a new
   model. For image or video models, include visual comparison and quantitative
   metrics when available, such as PSNR or SSIM.
