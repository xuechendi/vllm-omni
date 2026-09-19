#!/bin/bash
# HunyuanImage-3.0-Instruct-Distil, online MXFP8 weights, image-to-text,
# AR-only on a single XPU card with TP=1. Self-contained.
#   bash hy_image3_i2t_mxfp8weight.sh
#   ZE_AFFINITY_MASK=2 IMAGE=/workspace/foo.png bash hy_image3_i2t_mxfp8weight.sh
#
# Offline rather than serve+curl, unlike the t2i script. HunyuanImage3 ships no
# chat template (tokenizer_config.json has no chat_template key), and i2t needs
# the model's own token layout plus task-specific stop tokens -- built by
# build_prompt_tokens(task="i2t") / resolve_stop_token_ids() and passed as
# prompt_token_ids. The x_to_text example already does exactly that via
# vllm_omni/model_extras/hunyuan_image3.py, so /v1/chat/completions would need a
# hand-written template to reach the same prompt.
MODEL=${MODEL:-/mnt/disk1/hf_home/hub/HunyuanImage-3.0-Instruct-Distil/}
HERE=$(cd "$(dirname "$0")" && pwd)
DEPLOY=${DEPLOY:-$HERE/hunyuan_image3_ar_mxfp8_tp1.yaml}
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
# Default input is the t2i script's output, so the two tests chain: generate an
# astronaut in a greenhouse, then ask the AR stage what it sees.
IMAGE=${IMAGE:-/workspace/hy_image3_logs/mxfp8_torch_sdpa.png}
PROMPT=${PROMPT:-"Describe the content of the picture."}
MAX_TOKENS=${MAX_TOKENS:-256}
# /workspace is the host mount, so these outlive the container.
OUT_DIR=${OUT_DIR:-/workspace/hy_image3_logs}; mkdir -p "$OUT_DIR"
LOG=$OUT_DIR/mxfp8_i2t.log; OUT=$OUT_DIR/mxfp8_i2t.txt

if [ ! -f "$IMAGE" ]; then
    echo "input image not found: $IMAGE (run hy_image3_t2i_1024_mxfp8weight.sh first, or pass IMAGE=)"
    exit 1
fi

# See the t2i script for the mask -> card mapping on this node. TP=1 means only
# the masked card is touched, and the deploy YAML's devices: "0" resolves through
# this mask.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Same opt-in as the t2i script, and it matters just as much here: the AR stage
# loads the same 80B checkpoint through its own load_weights, and copying
# straight out of the safetensors mapping runs at ~71 MB/s on this node.
export VLLM_OMNI_STAGE_WEIGHTS=${STAGE_WEIGHTS:-1}
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}

# The SigLIP2 tower runs under TORCH_SDPA no matter what this is set to: its
# head_dim is 72 (1152/16), giving a 144-byte q stride that the Xe2 2D block
# loads reject, so vllm_omni/model_executor/models/hunyuan_image3/siglip2.py
# retargets that one module. Without that retarget the vision encoder fails on
# XPU and i2t cannot run at all -- t2i never notices because it has no image input.
echo "model=$MODEL"
echo "deploy=$DEPLOY"
echo "image=$IMAGE"
time python "$REPO/examples/offline_inference/x_to_text/x_to_text.py" \
    --model "$MODEL" \
    --image "$IMAGE" \
    --prompt "$PROMPT" \
    --deploy-config "$DEPLOY" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0.0 \
    --trust-remote-code \
    --output "$OUT" > "$LOG" 2>&1
RC=$?

if [ $RC -eq 0 ] && [ -s "$OUT" ]; then
    echo "I2T OK: $OUT ($(wc -c < "$OUT") bytes)"
    cat "$OUT"
else
    echo "I2T FAILED (rc=$RC) -- tail of $LOG:"
    tail -30 "$LOG"
    exit 1
fi
