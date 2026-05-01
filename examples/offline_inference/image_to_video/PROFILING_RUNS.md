# I2V Profiling Command Lines

This file tracks profiling runs for image-to-video generation with different configurations.

## Date: 2026-05-01

### I2V TP=4 Profiling (4 steps, GPUs 0-3)

**Command:**
```bash
cd /workspace/vllm-omni/examples/offline_inference/image_to_video/ && \
ulimit -n 1048576 && \
ZE_AFFINITY_MASK=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
VLLM_TORCH_PROFILER_DIR=perf/i2v_tp4_4steps_profile_gpu0123 \
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying in the wind" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 4 \
  --tensor-parallel-size 4 \
  --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --enable-diffusion-pipeline-profiler \
  --output i2v_720p_tp4_4steps_profile.mp4 \
  > wan-i2v_tp4_4steps_profile_gpu0123.log 2>&1
```

**First Run Results (without profiler_config - NO TRACES):**
- Total time: 272.2 seconds for 4 steps (~68s per step)
- Video saved: i2v_720p_tp4_4steps_profile.mp4 (2.2MB)
- Issue: Profiler traces NOT captured (profiler_config was missing in image_to_video.py)
- Fix Applied: Added dict-based profiler_config to image_to_video.py

**Second Run Results (2026-05-01):**
- Total time: 292.75 seconds (4:52) for 4 steps (~73s per step)
- Video saved: i2v_720p_tp4_4steps_profile.mp4
- Traces: `perf/i2v_tp4_4steps_profile_gpu0123/20260501-190035_stage_0_diffusion_1777662035/`
  - trace_rank0.json.gz (123MB)
  - trace_rank1.json.gz (123MB)
  - trace_rank2.json.gz (123MB)
  - trace_rank3.json.gz (123MB)
- All 4 ranks exported successfully
- Status: ✅ **Profiling traces captured successfully with fixed profiler_config**

---

## Key Configuration Notes

### Required Environment Variables:
- `ZE_AFFINITY_MASK=0,1,2,3` - Use GPUs 0-3
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` - Prevents DEVICE_LOST errors
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` - Required for multiprocessing
- `VLLM_TORCH_PROFILER_DIR=<path>` - Enables torch profiler and sets output directory

### Common Settings:
- Model: Wan2.2-I2V-14B
- Resolution: 720x1280, 81 frames
- CPU offload: ENABLED
- VAE slicing + tiling: ENABLED
- Attention backend: TORCH_SDPA (default)
- Compiled mode: torch.compile ENABLED (default)

### GPU Allocation:
- **TP=4**: GPUs 0-3 with tensor parallelism

### Profiler Configuration (added to image_to_video.py):
```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": profile_dir,
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_memory": True,
    "torch_profiler_with_stack": False,
    "active_iterations": args.num_inference_steps,
    "warmup_iterations": 0,
}
```

### Trace Files:
- Each rank generates a compressed JSON trace (~167MB)
- Directory format: `perf/<run_name>/<timestamp>_stage_0_diffusion_<id>/`

---

## Code Changes

### 2026-05-01: Added profiler_config support
Fixed image_to_video.py to properly pass profiler_config dict to Omni() constructor, enabling torch profiler trace capture. Previously, the script only checked for VLLM_TORCH_PROFILER_DIR environment variable but never created the profiler_config parameter.
