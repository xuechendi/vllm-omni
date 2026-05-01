# T2V Profiling Command Lines

This file tracks profiling runs for text-to-video generation with different configurations.

## Date: 2026-05-01

### TP=4 Profiling (4 steps, NO cache_dit, GPUs 4-7)

**Command:**
```bash
cd /workspace/vllm-omni/examples/offline_inference/text_to_video/ && \
ulimit -n 1048576 && \
ZE_AFFINITY_MASK=4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
VLLM_TORCH_PROFILER_DIR=perf/t2v_tp4_4steps_nocache_profile_gpu4567 \
python text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 4 \
  --boundary-ratio 0.875 --flow-shift 5.0 --fps 16 \
  --tensor-parallel-size 4 \
  --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --enable-diffusion-pipeline-profiler \
  --output t2v_720p_tp4_4steps_nocache_profile_gpu4567.mp4 \
  > wan-t2v_tp4_4steps_nocache_profile_gpu4567.log 2>&1
```

**Results:**
- Total time: 256.09 seconds (4:16) for 4 steps (~64s per step)
- Video saved: t2v_720p_tp4_4steps_nocache_profile_gpu4567.mp4
- Traces: `perf/t2v_tp4_4steps_nocache_profile_gpu4567/`
  - Directory 1 (20260501-190313): trace_rank0.json.gz (74MB), trace_rank3.json.gz (74MB)
  - Directory 2 (20260501-190314): trace_rank1.json.gz (74MB), trace_rank2.json.gz (74MB)
- All 4 ranks exported successfully (split across 2 directories due to 1-second timing difference)
- Cache-dit: DISABLED
- Status: ✅ **Completed successfully**

---

### TP=8 Profiling (10 steps with cache_dit)

**Command:**
```bash
cd /workspace/vllm-omni/examples/offline_inference/text_to_video/ && \
ulimit -n 1048576 && \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
VLLM_TORCH_PROFILER_DIR=perf/t2v_tp8_10steps_profile \
python text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 10 \
  --boundary-ratio 0.875 --flow-shift 5.0 --fps 16 \
  --tensor-parallel-size 8 \
  --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --enable-diffusion-pipeline-profiler \
  --output t2v_720p_tp8_10steps_profile.mp4 \
  > wan-t2v_tp8_10steps_profile.log 2>&1
```

**Results:**
- Traces: `perf/t2v_tp8_10steps_profile/20260501-170327_stage_0_diffusion_1777655007/`
- Diffusion time: ~284 seconds
- All 8 ranks exported successfully
- Cache-dit: ENABLED
- guidance_scale: 4.0 (default)

---

### TP=4, CFG=2 Profiling (10 steps, NO cache_dit)

**Command:**
```bash
cd /workspace/vllm-omni/examples/offline_inference/text_to_video/ && \
ulimit -n 1048576 && \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
VLLM_TORCH_PROFILER_DIR=perf/t2v_tp4_cfg2_10steps_profile \
python text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 10 \
  --boundary-ratio 0.875 --flow-shift 5.0 --fps 16 \
  --tensor-parallel-size 4 \
  --cfg-parallel-size 2 \
  --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --enable-diffusion-pipeline-profiler \
  --output t2v_720p_tp4_cfg2_10steps_profile.mp4 \
  > wan-t2v_tp4_cfg2_10steps_profile.log 2>&1
```

**Results:**
- Traces: `perf/t2v_tp4_cfg2_10steps_profile/20260501-173059_stage_0_diffusion_1777656659/`
- Diffusion time: ~257 seconds
- Total generation time: 313.8 seconds
- All 8 ranks exported successfully
- Cache-dit: DISABLED (memory constraints)
- guidance_scale: 4.0 (default)
- CFG parallel: Ranks 0-3 (positive), Ranks 4-7 (negative)

---

## Performance Test Runs (40 steps)

### TP=8 Performance (40 steps with cache_dit)

**Command:**
```bash
cd /workspace/vllm-omni/examples/offline_inference/text_to_video/ && \
ulimit -n 1048576 && \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
python text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 --flow-shift 5.0 --fps 16 \
  --tensor-parallel-size 8 \
  --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --cache-backend cache_dit \
  --enable-diffusion-pipeline-profiler \
  --output t2v_720p_tp8_cpu_offload.mp4 \
  > wan-t2v_720p_tp8_cpu_offload.log 2>&1
```

**Results:**
- Total time: 595.04 seconds (9m 55s)
- Diffusion: ~592 seconds
- VAE decode: ~40 seconds
- Average per step: 14.9s

---

### TP=4, CFG=2 Performance (40 steps with cache_dit)

**Command:**
```bash
cd /workspace/vllm-omni/examples/offline_inference/text_to_video/ && \
ulimit -n 1048576 && \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
python text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 --flow-shift 5.0 --fps 16 \
  --tensor-parallel-size 4 \
  --cfg-parallel-size 2 \
  --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --cache-backend cache_dit \
  --enable-diffusion-pipeline-profiler \
  --output t2v_720p_tp4_cfg2_cpu_offload.mp4 \
  > wan-t2v_720p_tp4_cfg2_cpu_offload.log 2>&1
```

**Results:**
- Total time: 589.35 seconds (9m 49s)
- Diffusion: ~587 seconds
- VAE decode: ~41 seconds
- Average per step: 14.7s
- **5.7 seconds faster than TP=8** (~1% speedup)

---

## Key Configuration Notes

### Required Environment Variables:
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` - Prevents DEVICE_LOST errors
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` - Required for multiprocessing
- `VLLM_TORCH_PROFILER_DIR=<path>` - Enables torch profiler and sets output directory

### Common Settings (all runs):
- Model: Wan2.2-T2V-A14B-Diffusers
- Resolution: 720x1280, 81 frames
- boundary_ratio: 0.875 (high-noise/low-noise split)
- flow_shift: 5.0
- CPU offload: ENABLED
- VAE slicing + tiling: ENABLED
- guidance_scale: 4.0 (default, enables CFG)
- Attention backend: TORCH_SDPA (default)

### GPU Allocation:
- **TP=8**: All 8 GPUs (0-7) do sequential CFG (positive then negative)
- **TP=4, CFG=2**: 8 GPUs total
  - Ranks 0-3 (GPUs 0-3): Positive CFG branch with TP=4
  - Ranks 4-7 (GPUs 4-7): Negative CFG branch with TP=4

### Profiler Configuration (added to text_to_video.py):
```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": profiler_dir,
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_memory": True,
    "torch_profiler_with_stack": False,
    "active_iterations": args.num_inference_steps,
    "warmup_iterations": 0,
}
```

### Trace Files:
- Each rank generates a compressed JSON trace (~70-76MB)
- Also generates Excel ops summary files
- Directory format: `perf/<run_name>/<timestamp>_stage_0_diffusion_<id>/`

---

## Performance Comparison

| Configuration | Steps | Total Time | Diffusion | Avg/Step | Cache-DiT | Notes |
|--------------|-------|-----------|-----------|----------|-----------|-------|
| TP=8 | 40 | 595.04s | ~592s | 14.9s | ✅ | Sequential CFG |
| TP=4, CFG=2 | 40 | 589.35s | ~587s | 14.7s | ✅ | **Parallel CFG** |
| TP=8 | 10 | - | ~284s | 28.4s | ✅ | With profiling |
| TP=4, CFG=2 | 10 | 313.8s | ~257s | 25.7s | ❌ | With profiling, no cache |

**Key Finding:** CFG parallel (TP=4, CFG=2) is faster than sequential CFG (TP=8) despite using same 8 GPUs, demonstrating the benefit of distributing CFG branches across separate GPU groups.
