# S2V Profiling Command Lines

This file tracks profiling runs for speech-to-video generation with different configurations.

## Date: 2026-05-01

### S2V TP=4 Profiling (4 steps, GPUs 4-7)

**Command:**
```bash
cd /workspace/vllm-omni/examples/offline_inference/speech_to_video/ && \
ulimit -n 1048576 && \
ZE_AFFINITY_MASK=4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SYCL_UR_USE_LEVEL_ZERO_V2=0 \
python speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" \
  --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 4 \
  --fps 16 \
  --enable-cpu-offload \
  --tensor-parallel-size 4 \
  --vae-use-slicing --vae-use-tiling \
  --enable-diffusion-pipeline-profiler \
  --profile-dir perf/s2v_tp4_4steps_profile_gpu4567 \
  --profile-record-shapes \
  --profile-with-memory \
  --output s2v_720p_tp4_4steps_profile_gpu4567.mp4 \
  > wan-s2v_tp4_4steps_profile_gpu4567.log 2>&1
```

**Results:**
- Total generation time: 339.49 seconds for 4 steps (~85s per step)
- Traces: `perf/s2v_tp4_4steps_profile_gpu4567/20260501-183507_stage_0_diffusion_1777660507/`
- All 4 ranks exported successfully (167MB each compressed)
- Diffusion time: ~224 seconds (4 steps)
- VAE encode: ~28 seconds per clip
- VAE decode: ~49 seconds
- Peak memory: 26.96 GB reserved, 23.31 GB allocated, 3.65 GB pool overhead (13.6%)
- Status: ✅ **Profiling traces captured successfully**
- Issue: Video output failed with OmniRequestOutput unwrapping error (output handling bug, doesn't affect profiling data)

---

## Key Configuration Notes

### Required Environment Variables:
- `ZE_AFFINITY_MASK=4,5,6,7` - Use GPUs 4-7
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` - Prevents DEVICE_LOST errors
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` - Required for multiprocessing

### Common Settings:
- Model: Wan2.2-S2V-14B
- Resolution: 720x1280, 81 frames
- flow_shift: 3.0 (default for S2V)
- guidance_scale: 4.5 (default for S2V)
- CPU offload: ENABLED
- VAE slicing + tiling: ENABLED
- Attention backend: TORCH_SDPA (default)
- Compiled mode: torch.compile ENABLED (default)

### GPU Allocation:
- **TP=4**: GPUs 4-7 with tensor parallelism

### Profiler Configuration (CLI args converted to dict):
Uses `--profile-dir`, `--profile-record-shapes`, `--profile-with-memory` CLI arguments which are converted to:
```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": profile_dir,
    "torch_profiler_record_shapes": args.profile_record_shapes,
    "torch_profiler_with_stack": args.profile_with_stack,
    "torch_profiler_with_memory": args.profile_with_memory,
    "active_iterations": args.num_inference_steps,
    "warmup_iterations": 0,
}
```

### Trace Files:
- Each rank generates a compressed JSON trace (~167MB)
- Directory format: `perf/<run_name>/<timestamp>_stage_0_diffusion_<id>/`

---

## Code Changes

### 2026-05-01: Converted to dict-based profiler_config
Changed speech_to_video.py from class-based ProfilerConfig to dict-based profiler_config to match the pattern used in text_to_video.py and image_to_video.py. This ensures consistent profiler configuration across all video generation scripts.

### Known Issues

**OmniRequestOutput unwrapping error:**
```
ValueError: Expected OmniRequestOutput, got <class 'NoneType'>
```
This error occurs during output unwrapping in speech_to_video.py line 340. The generation and profiling complete successfully, but the final video file is not saved. This is a bug in the output handling code and does not affect the profiling trace capture.
