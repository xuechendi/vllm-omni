#!/bin/bash
# HunyuanImage-3.0-Instruct-Distil it2i (image + edit instruction -> image) on two
# XPU cards, AR stage 0 + DiT stage 1, TP=1 each, online MXFP8. Self-contained.
#
#   POST /v1/images/edits with an image upload: the AR *reads* the image through its
#   SigLIP2 tower, reasons about the instruction, recaptions, and the DiT draws
#   conditioned on that same image (forwarded by ar2diffusion as
#   multi_modal_data["image"]).
#
#   bash hy_image3_it2i_mxfp8weight.sh
#   IMAGE=/workspace/some.png bash hy_image3_it2i_mxfp8weight.sh
#   ZE_AFFINITY_MASK=2,3 bash hy_image3_it2i_mxfp8weight.sh
#
# Why this is a separate script and not a case in hy_image3_ar_dit_mxfp8weight.sh:
# that test's deploy asks for `attention_backend: FLASH_ATTN` on stage 0, which on XPU
# is honoured only with the warning that the prefix-LM bidirectional mask cannot be
# applied -- "image/video inputs will produce incorrect results; only use this for
# text-only workloads" (vllm/platforms/xpu.py:167-185). The backend is picked once per
# model at engine init, before any request arrives, so one server cannot serve t2i on
# flash attention and it2i correctly. The split therefore has to be at deploy level,
# and this script makes it by deriving its deploy from that file with that one key
# dropped -- one source of truth for everything else (devices, MXFP8, connectors,
# sampling params), so the two tests stay comparable.
#
# What that leaves on stage 0 is Triton attention plus the XPU graph capture
# (compilation_config: FULL_DECODE_ONLY, mode NONE), which is the config the arm
# harness calls `graph`. NOTE, honestly: that combination was never measured. The 4x
# TPOT win recorded in the main deploy YAML was measured as FLASH_ATTN + graph
# together, and the graph-only arm was cancelled before it ran. Attention is 2.0% of AR
# device time, so it should carry over nearly intact, but if this run misbehaves the
# first thing to try is VLLM_XPU_ENABLE_XPU_GRAPH=0, which takes stage 0 back to the
# fully validated eager Triton path (expect it2i around 30s again rather than whatever
# the graph gives).
#
# it2i is also what makes the SigLIP2 XPU fixes in
# model_executor/models/hunyuan_image3/siglip2.py load-bearing: stage 0 builds that
# tower under the deploy's `quantization: mxfp8`, so without the block-scaled guard
# there stage 0 dies at init. t2i never notices -- with no image input the tower is
# never called -- so this script is the one that regression-tests it.
#
# Measured previously (eager Triton stage 0, compiled DiT): e2e 28-33s, stage_0_gen_ms
# 15.4s against stage_1_gen_ms 13.5s -- so unlike t2i the AR is only ~53% of the
# request, because the edit CoT is short (130-170 output tokens) while the DiT does the
# same 8 steps plus the image encode. TTFT is 1.9-3.1s rather than t2i's ~300ms, and
# that gap is the SigLIP2 image encode.
PORT=${PORT:-8010}
BACKEND=${BACKEND:-FLASH_ATTN}
MODEL=${MODEL:-/mnt/disk1/hf_home/hub/HunyuanImage-3.0-Instruct-Distil/}
# 8 steps is what the Distil model card recommends; guidance 1.0 disables CFG, which
# the CFG-distilled checkpoint expects (config.json: cfg_distilled=true). Both pinned
# equal to the t2i test so the timings compare.
STEPS=${STEPS:-8}
GUIDANCE=${GUIDANCE:-1.0}

# ---- it2i knobs ----
EDIT_PROMPT=${EDIT_PROMPT:-"Replace the orchids with sunflowers and change the light to a warm sunset."}
# recaption, not think_recaption: the edit instruction is already concrete, so the
# <think> block buys little here and it is the AR decode length that dominates cost.
EDIT_BOT_TASK=${EDIT_BOT_TASK:-recaption}
SYS_TYPE=${SYS_TYPE:-en_unified}
# No size field on the request: for /v1/images/edits the output geometry follows the
# input image, and the AR's <img_ratio_*> choice is read back from the CoT
# (_extract_ratio_index, stage_input_processors/hunyuan_image3.py:73).
#
# The image to edit. Empty means "chain onto the t2i test's recaption output", which is
# the intended way to run these two back to back; failing that, the DiT-only test's.
IMAGE=${IMAGE:-}

# /workspace is the host mount, so these outlive the container.
OUT_DIR=${OUT_DIR:-/workspace/hy_image3_logs}; mkdir -p "$OUT_DIR"
TAG=it2i_mxfp8_$(echo "$BACKEND" | tr 'A-Z' 'a-z')
LOG=$OUT_DIR/$TAG.log

# The deploy: generated from the t2i one, minus stage 0's attention_backend (see the
# header). Regenerated on every run so it cannot drift from the file it is derived
# from; pass DEPLOY= to use a hand-written one instead and the derivation is skipped.
BASE_DEPLOY=${BASE_DEPLOY:-$(cd "$(dirname "$0")" && pwd)/hunyuan_image3_ar_dit_mxfp8_tp1.yaml}
DEPLOY=${DEPLOY:-$OUT_DIR/$TAG.deploy.yaml}
if [ "$DEPLOY" != "$BASE_DEPLOY" ] && [ -z "${DEPLOY_PREBUILT:-}" ]; then
    [ -s "$BASE_DEPLOY" ] || { echo "no base deploy at $BASE_DEPLOY"; exit 2; }
    # Commented out rather than deleted so the generated file still shows what was
    # removed and why, for anyone reading it next to a server log.
    sed 's|^\([[:space:]]*\)attention_backend:\(.*\)$|\1# attention_backend:\2  <-- removed by hy_image3_it2i_mxfp8weight.sh: FLASH_ATTN cannot apply the prefix-LM bidirectional mask, so it silently corrupts image input. Stage 0 falls back to Triton (xpu.py:181).|' \
        "$BASE_DEPLOY" > "$DEPLOY" || exit 2
    # Both directions of the assertion: the key must be gone from the generated file,
    # and it must have been there to remove. A silent no-op here would mean serving
    # it2i on flash attention, i.e. a plausible-looking wrong image.
    if grep -qE '^[[:space:]]*attention_backend:' "$DEPLOY"; then
        echo "refusing to run: attention_backend survived in $DEPLOY"; exit 2
    fi
    if ! grep -qE '^[[:space:]]*#[[:space:]]*attention_backend:' "$DEPLOY"; then
        echo "note: $BASE_DEPLOY had no attention_backend to remove -- stage 0 was"
        echo "already on the platform default. Continuing."
    fi
fi
echo "deploy: $DEPLOY (from $BASE_DEPLOY)"

# Two cards, one per stage; the deploy's devices "0"/"1" are logical indices resolved
# through this mask. Mask 0 is this node's Gen5-x4 degraded link, so 2,3 is the pair to
# benchmark on when it is free. The box is shared -- check per-card free memory first
# (torch.xpu.mem_get_info; there is no xpu-smi in this container), each stage wants
# ~110 GiB at gpu_memory_utilization 0.9 and init fails outright without it.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export NO_PROXY=0.0.0.0,127.0.0.1,localhost
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Reaches the DiT only (diffusion/data.py:2021). Stage 0's backend is vLLM's XPU
# selector's business and is exactly what this script does not override; the SigLIP2
# tower is always TORCH_SDPA (head_dim 72 is not 64-byte aligned, so the retarget in
# siglip2.py is forced). XeFMHA is ~10% of DiT device time on it2i against 6% on t2i --
# the image conditioning lengthens the attention.
export DIFFUSION_ATTENTION_BACKEND=$BACKEND

# The gate for the deploy's stage-0 compilation_config: without it, xpu.py:303 pins
# cudagraph_mode back to NONE and logs "XPU Graph is disabled by environment variable".
# Set to 0 to fall back to the fully validated eager path (see the header note -- graph
# + Triton is the unmeasured combination).
export VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1}

# Pinned-host staging of the checkpoint tensors: a host->device copy straight out of
# the safetensors mapping cannot be DMA'd by level-zero and runs at ~71 MB/s, ~4.6 GB/s
# via a pinned buffer. Off by default in the loaders, and it pays twice here, once per
# stage. STAGE_WEIGHTS=0 to measure the unpatched path.
export VLLM_OMNI_STAGE_WEIGHTS=${STAGE_WEIGHTS:-1}
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}

# Resolve the input image before spending ten minutes on start-up.
# The t2i test's recaption output first (TAG ar_dit_mxfp8_<backend>), then the
# DiT-only test's (TAG mxfp8_<backend>, hy_image3_t2i_1024_mxfp8weight.sh).
BL=$(echo "$BACKEND" | tr 'A-Z' 'a-z')
SRC=$IMAGE
if [ -z "$SRC" ]; then
    for cand in "$OUT_DIR/ar_dit_mxfp8_$BL.recaption.png" "$OUT_DIR/mxfp8_$BL.png"; do
        [ -s "$cand" ] && { SRC=$cand; break; }
    done
fi
# An explicit IMAGE= that does not exist is an error, not something to fall back from.
if [ ! -s "$SRC" ]; then
    echo "no input image${SRC:+ at $SRC}: pass IMAGE=, or run"
    echo "hy_image3_ar_dit_mxfp8weight.sh first -- its recaption output is what this"
    echo "chains onto by default."
    exit 2
fi
echo "editing $SRC ($(stat -c %s "$SRC") bytes)"

# setsid: own process group, so the teardown names this run and only this run. This
# container's PID namespace is shared with other users' runs and the worker process
# titles are not unique to a run, so pkill on the title is not an option -- it has
# SIGKILLed concurrent runs before.
setsid vllm-omni serve $MODEL --omni --port $PORT --host 0.0.0.0 \
    --deploy-config "$DEPLOY" --trust-remote-code \
    --enable-diffusion-pipeline-profiler \
    --enable-ar-profiler \
    --stage-init-timeout 7200 --init-timeout 7200 > $LOG 2>&1 &
SRV=$!
PGID=$(ps -o pgid= -p $SRV 2>/dev/null | tr -d ' ')
teardown() {
    if [ -n "$PGID" ]; then kill -TERM -"$PGID" 2>/dev/null; else kill -TERM $SRV 2>/dev/null; fi
    sleep 8
    if [ -n "$PGID" ]; then kill -KILL -"$PGID" 2>/dev/null; else kill -KILL $SRV 2>/dev/null; fi
}
until curl -s -o /dev/null --max-time 5 http://127.0.0.1:$PORT/health; do
    kill -0 $SRV || { tail -30 $LOG; exit 1; }; sleep 10
done
grep -aiE 'mxfp8|quantization|siglip|stage 0|stage 1|connector' $LOG | tail -8

# The correctness gate this script exists for. If stage 0 came up on flash attention
# the image conditioning is silently wrong, so refuse to report a result.
if grep -aq 'Using Flash Attention on XPU for a multimodal' $LOG; then
    echo "ABORT: stage 0 is on FLASH_ATTN, which cannot apply the prefix-LM mask --"
    echo "image input would be silently corrupted. Check $DEPLOY."
    teardown; exit 2
fi
grep -aoE 'Using Triton backend|Falling back to Triton[^|]*' $LOG | tail -2

OUT=$OUT_DIR/$TAG.png
COT=$OUT_DIR/$TAG.cot.txt
JSON=$OUT.json
echo "== edit (it2i) =="
# Multipart, and b64_json because /v1/images/edits rejects anything else (the `file`
# shortcut is generations-only).
time curl --max-time 3600 -sS -X POST http://127.0.0.1:$PORT/v1/images/edits -o "$JSON" \
    -F "image=@$SRC" \
    -F "prompt=$EDIT_PROMPT" \
    -F "num_inference_steps=$STEPS" \
    -F "guidance_scale=$GUIDANCE" \
    -F "seed=1234" \
    -F "bot_task=$EDIT_BOT_TASK" \
    -F "sys_type=$SYS_TYPE" \
    -F "response_format=b64_json"

RC=0
python - "$JSON" "$OUT" "$COT" <<'PY' || RC=1
import base64, json, sys
payload = json.load(open(sys.argv[1]))
data = payload.get("data") or []
if not data or not data[0].get("b64_json"):
    print("NO IMAGE -- server said:", json.dumps(payload)[:400]); sys.exit(1)
png = base64.b64decode(data[0]["b64_json"])
open(sys.argv[2], "wb").write(png)
cot = payload.get("cot_output") or ""
open(sys.argv[3], "w").write(cot)
ok = png.startswith(bytes.fromhex("89504e470d0a1a0a"))
print(f"PNG {'OK' if ok else 'BAD MAGIC'}: {sys.argv[2]} ({len(png)} bytes), size={payload.get('size')}")
# Non-empty and about the edit is the real check here: an it2i CoT that never mentions
# the instruction, or is byte-identical to the t2i one, means the image never reached
# the AR. Eyeball it; it is short (130-170 tokens).
print("CoT:", cot[:400] if cot else "(empty -- stage 0 returned no text)")
sys.exit(0 if ok else 1)
PY

teardown
# Metric names differ between pipelines, so match the profiler prefix rather than a
# fixed metric pair. stage_1 should be close to t2i's; stage_0 carries the SigLIP2
# encode in its TTFT.
grep -aoE '\[DiffusionPipelineProfiler\] [A-Za-z0-9_.]+ took [0-9.]+s' $LOG | tail -8
grep -aoE 'Capturing CUDA graphs \([a-z, A-Z_]+\)|Graph capturing finished[^|]*' $LOG | tail -2
grep -aoE 'ttft_ms[^0-9]*[0-9.]+|tpot_ms[^0-9]*[0-9.]+|num_tokens_out[^0-9]*[0-9,]+' $LOG | tail -6
exit $RC
