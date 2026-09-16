# SYCL-TLA FP8 Attention on XPU

SYCL-TLA provides dense FP8 FlashAttention kernels for Intel XPU diffusion
models. vLLM-Omni exposes two modes through the `FLASH_ATTN` backend:

| `--diffusion-kv-cache-dtype` | Quantization |
| --- | --- |
| `fp8_sycl_tla` | Per-tensor E4M3 Q/K/V |
| `mxfp8_sycl_tla` | Block-scaled E4M3 Q/K/V with UE8M0 scales over groups of 32 elements |

These values select attention quantization. They are independent of
`--quantization`, which controls transformer linear layers.

## Requirements

- An Intel XPU supported by the SYCL-TLA build target.
- Intel oneAPI `icx` and `icpx` compilers.
- Ninja and CMake.
- vLLM-Omni configured to use the `FLASH_ATTN` diffusion attention backend.
- Q, K, and V head dimension `128`.

The SYCL-TLA path requires dense, unpadded attention inputs.

## Build SYCL-TLA

Use the [`v0.1.0_next_pyfmha` branch of
`mengker33/libraries.ai.cutlass.internal`](https://github.com/mengker33/libraries.ai.cutlass.internal/tree/v0.1.0_next_pyfmha).

From the repository root, configure and build the Python FMHA extension:

```bash
export IGC_ExtraOCLOptions="-cl-intel-512-GRF-per-thread"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/build/deps/oneMKL/lib/"

mkdir -p build
cd build

CC=icx CXX=icpx cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_cri \
  -DCUTLASS_SYCL_RUNNING_CI=ON \
  -DSYCL_TLA_BUILD_PYTHON_FMHA=ON

ninja sycl_tla_fmha -j 8
```

Make the resulting `sycl_tla_fmha` Python extension importable. From the
SYCL-TLA repository root:

```bash
export PYTHONPATH="$PWD/build/examples/06_bmg_flash_attention:$PYTHONPATH"
```

Confirm that the extension and both entry points are available:

```bash
python - <<'PY'
import sycl_tla_fmha

assert hasattr(sycl_tla_fmha, "prefill_fp8_e4m3_bshd")
assert hasattr(sycl_tla_fmha, "prefill_mxfp8_e4m3_bshd")
print(sycl_tla_fmha.__file__)
PY
```

## Enable FP8 Attention

The shared text-to-video example forwards `--diffusion-kv-cache-dtype` to the
diffusion attention configuration. The following Wan2.2 command enables the
per-tensor SYCL-TLA FP8 FlashAttention kernel:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model /workspace/hf_models/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 \
  --width 1280 \
  --num-frames 81 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 5.0 \
  --fps 16 \
  --diffusion-kv-cache-dtype fp8_sycl_tla \
  --quantization mxfp8 \
  --cache-backend cache_dit \
  --enable-diffusion-pipeline-profiler
```

Here, `--diffusion-kv-cache-dtype fp8_sycl_tla` selects the attention kernel,
while `--quantization mxfp8` separately enables W8A8 MXFP8 transformer linear
layers.

> **Quality note:** For better output quality, consider adding
> `--diffusion-kv-cache-skip-layers "33,34,38,39"`. This keeps those attention
> layers on the BF16 fallback path.

## Enable MXFP8 Attention

Use the same command with the block-scaled attention mode:

```bash
--diffusion-kv-cache-dtype mxfp8_sycl_tla
```

The extension must export `prefill_mxfp8_e4m3_bshd`; otherwise vLLM-Omni
raises an import error describing the missing entry point.

## Enable SYCL-TLA Ring Attention

Ring attention requires the [`fp8_sycl_tla_ring_ce_read` branch of
`mengker33/libraries.ai.cutlass.internal`](https://github.com/mengker33/libraries.ai.cutlass.internal/tree/fp8_sycl_tla_ring_ce_read),
which is different from the branch used by the FP8 attention build above.
Configure the repository with the same CMake options shown in
[Build SYCL-TLA](#build-sycl-tla), then build both targets from the `build`
directory:

```bash
ninja sycl_tla_fmha sycl_tla_ipc_p2p -j 8
```

From the SYCL-TLA repository root, add the source and build directories to
`PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/examples/06_bmg_flash_attention:$PWD/build/examples/06_bmg_flash_attention:$PYTHONPATH"
```

Test the ring IPC kernel on two XPU ranks:

```bash
torchrun \
  --standalone \
  --nproc-per-node=2 \
  examples/06_bmg_flash_attention/test_ring_attn_fp8_ipc.py
```

After adding the SYCL-TLA source and build directories to `PYTHONPATH` as shown
above, run the vLLM-Omni offline text-to-video example from the vLLM-Omni
repository root:

```bash
python3 examples/offline_inference/text_to_video/text_to_video.py \
  --model /workspace/hf_models/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 \
  --width 1280 \
  --num-frames 81 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 5.0 \
  --fps 16 \
  --quantization mxfp8 \
  --diffusion-kv-cache-dtype fp8_sycl_tla \
  --ulysses-degree 1 \
  --ring-degree 2 \
  --ring-attn-type sycl_tla \
  --cache-backend cache_dit \
  --enable-diffusion-pipeline-profiler \
  --output ring2_cachedit.mp4
```

This command uses per-tensor FP8 Q/K/V for ring attention. Omit
`--diffusion-kv-cache-dtype fp8_sycl_tla` to use BF16 SYCL-TLA ring attention
instead.

## Troubleshooting

- `No module named 'sycl_tla_fmha'`: add the directory containing the built
  module to `PYTHONPATH`.
- Ring IPC module import errors: build the `sycl_tla_ipc_p2p` target and add
  both `examples/06_bmg_flash_attention` directories shown above to
  `PYTHONPATH`.
- Missing `prefill_mxfp8_e4m3_bshd`: rebuild with
  `-DSYCL_TLA_BUILD_PYTHON_FMHA=ON` from the specified branch.
- KV quantization is disabled at startup: ensure the diffusion backend resolves
  to `FLASH_ATTN` on XPU and check the warning for an unsupported dtype.
- Head-dimension error: the current SYCL-TLA integration only supports
  `head_dim=128`.
