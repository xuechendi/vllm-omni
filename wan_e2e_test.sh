#!/bin/bash
# WAN 2.2 End-to-End Test Script
# Tests T2V, I2V, and S2V models at 720p with TP=4 and CPU offloading

set -e  # Exit on error

# Environment setup
export SYCL_UR_USE_LEVEL_ZERO_V2=0
ulimit -n 1048576

echo "=========================================="
echo "WAN 2.2 End-to-End Test Suite"
echo "Resolution: 720x1280, Frames: 81, TP: 4"
echo "=========================================="

# ============================================================================
# Test 1: Text-to-Video (T2V)
# ============================================================================
echo ""
echo "=========================================="
echo "Test 1: Text-to-Video (T2V)"
echo "=========================================="

cd /workspace/vllm-omni/examples/offline_inference/text_to_video/

VLLM_WORKER_MULTIPROC_METHOD=spawn \
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
python /workspace/vllm-omni/examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 --flow-shift 5.0 --fps 16 \
  --tensor-parallel-size 4 --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --cache-backend cache_dit \
  --enable-diffusion-pipeline-profiler \
  --output t2v_720p_tp4_cpu_offload.mp4 2>&1 | tee wan-t2v_720p.log

echo "T2V test completed. Output: t2v_720p_tp4_cpu_offload.mp4"

# ============================================================================
# Test 2: Image-to-Video (I2V)
# ============================================================================
echo ""
echo "=========================================="
echo "Test 2: Image-to-Video (I2V)"
echo "=========================================="

cd /workspace/vllm-omni/examples/offline_inference/image_to_video/

# Download reference image if not exists
if [ ! -f "cherry_blossom.jpg" ]; then
    echo "Downloading cherry blossom reference image..."
    wget -q https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
fi

VLLM_WORKER_MULTIPROC_METHOD=spawn \
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 --boundary-ratio 0.875 \
  --flow-shift 5.0 --fps 16 \
  --enable-cpu-offload \
  --tensor-parallel-size 4 --cache-backend cache_dit \
  --vae-use-slicing --vae-use-tiling \
  --enable-diffusion-pipeline-profiler \
  --output i2v_720p_output_tp4_cpu_offloading.mp4 2>&1 | tee wan-i2v_720p_tp4.log

echo "I2V test completed. Output: i2v_720p_output_tp4_cpu_offloading.mp4"

# ============================================================================
# Test 3: Speech-to-Video (S2V)
# ============================================================================
echo ""
echo "=========================================="
echo "Test 3: Speech-to-Video (S2V)"
echo "=========================================="

cd /workspace/vllm-omni/examples/offline_inference/speech_to_video/

# Download reference image and audio if not exists
if [ ! -f "Five Hundred Miles.png" ]; then
    echo "Downloading reference image..."
    wget -q -O "Five Hundred Miles.png" "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.png"
fi

if [ ! -f "Five Hundred Miles.MP3" ]; then
    echo "Downloading audio file..."
    wget -q -O "Five Hundred Miles.MP3" "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.MP3"
fi

VLLM_WORKER_MULTIPROC_METHOD=spawn \
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
python -u speech_to_video.py \
  --model Wan-AI/Wan2.2-S2V-14B \
  --image "Five Hundred Miles.png" --audio "Five Hundred Miles.MP3" \
  --prompt "A person singing" \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 \
  --tensor-parallel-size 4 --enable-cpu-offload \
  --vae-use-slicing --vae-use-tiling \
  --enable-diffusion-pipeline-profiler \
  --output s2v_720p_tp4.mp4 2>&1 | tee wan-s2v_720p_tp4.log

echo "S2V test completed. Output: s2v_720p_tp4.mp4"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo "Outputs:"
echo "  T2V: /workspace/vllm-omni/examples/offline_inference/text_to_video/t2v_720p_tp4_cpu_offload.mp4"
echo "  I2V: /workspace/vllm-omni/examples/offline_inference/image_to_video/i2v_720p_output_tp4_cpu_offloading.mp4"
echo "  S2V: /workspace/vllm-omni/examples/offline_inference/speech_to_video/s2v_720p_tp4.mp4"
echo ""
echo "Logs:"
echo "  T2V: /workspace/vllm-omni/examples/offline_inference/text_to_video/wan-t2v_720p.log"
echo "  I2V: /workspace/vllm-omni/examples/offline_inference/image_to_video/wan-i2v_720p_tp4.log"
echo "  S2V: /workspace/vllm-omni/examples/offline_inference/speech_to_video/wan-s2v_720p_tp4.log"
echo "=========================================="
