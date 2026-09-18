#!/bin/bash
# HunyuanImage-3.0-Instruct-Distil, online MXFP8 weights, 1024x1024 / 8 steps,
# DiT-only on a single XPU card with TP=1. Self-contained.
#   bash hy_image3_t2i_1024_mxfp8weight.sh                        # card2, 8 steps
#   ZE_AFFINITY_MASK=0 STEPS=50 bash hy_image3_t2i_1024_mxfp8weight.sh
#
# BACKEND is TORCH_SDPA and not FLASH_ATTN (which is what the Wan scripts use)
# because HunyuanImage3's DiT passes a per-head attention mask and the diffusion
# flash-attn backend refuses anything but a 2D one:
#   vllm_omni/diffusion/attention/backends/flash_attn.py:170
#     assert attention_mask.ndim == 2, "attention_mask must be 2D, ..."
# so a request under FLASH_ATTN fails with that message after the server is up.
BACKEND=${BACKEND:-TORCH_SDPA}
PORT=${PORT:-8006}
MODEL=${MODEL:-/mnt/disk1/hf_home/hub/HunyuanImage-3.0-Instruct-Distil/}
DEPLOY=${DEPLOY:-$(cd "$(dirname "$0")" && pwd)/hunyuan_image3_dit_mxfp8_tp1.yaml}
# 8 steps is what the Distil model card recommends; the undistilled
# HunyuanImage-3.0-Instruct wants 50 and is the reason the deploy default is 50.
STEPS=${STEPS:-8}
SIZE=${SIZE:-1024x1024}
# /workspace is the host mount, so these outlive the container.
OUT_DIR=${OUT_DIR:-/workspace/hy_image3_logs}; mkdir -p "$OUT_DIR"
TAG=mxfp8_$(echo "$BACKEND" | tr 'A-Z' 'a-z')
LOG=$OUT_DIR/$TAG.log; OUT=$OUT_DIR/$TAG.png

# Mask 1 is /dev/dri/card2 in this node's level-zero order (0..3 -> card1..card4,
# per the ze_peer device map in ~/ze_peer_debug.sh); card1 is the Gen5-x4
# degraded link, so it is not the one to benchmark on. TP=1 means only the single
# masked card is ever touched, and the deploy YAML's devices: "0" resolves
# through this mask.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-1} NO_PROXY=0.0.0.0,127.0.0.1,localhost
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DIFFUSION_ATTENTION_BACKEND=$BACKEND

# Opt into pinned-host staging of the checkpoint tensors (see load_weights in
# hunyuan_image3_transformer.py). A host->device copy straight out of the
# safetensors mapping cannot be DMA'd by level-zero and runs at ~71 MB/s; via a
# pinned buffer it is ~4.6 GB/s, which is what turns a ~70 min load into ~2 min
# on this node. Off by default in the loader, so it has to be asked for here.
# STAGE_WEIGHTS=0 to measure the unpatched path (then raise nothing: the 7200s
# deadlines below already cover it).
export VLLM_OMNI_STAGE_WEIGHTS=${STAGE_WEIGHTS:-1}

# Optional and, as it turned out, not the fix: reading the 157 GiB sequentially
# into page cache does make every loader read a cache hit, but the load was never
# disk-bound -- with the checkpoint fully cached the copies still crawled, and
# touching the pages from the CPU first made it *worse*. The cost was the
# mmap-source H2D copy, which VLLM_OMNI_STAGE_WEIGHTS above addresses. Left here
# because it costs ~11s and rules out cold cache as a variable. PREWARM=0 to skip.
PREWARM=${PREWARM:-1}
PREWARM_SH=${PREWARM_SH:-/workspace/prewarm_checkpoint.sh}
if [ "$PREWARM" = "1" ]; then
    if [ -x "$PREWARM_SH" ]; then
        JOBS=${JOBS:-4} "$PREWARM_SH" "$MODEL" || echo "pre-warm failed, continuing"
    else
        echo "pre-warm skipped: $PREWARM_SH not executable"
    fi
fi

# Deadlines are sized for the STAGE_WEIGHTS=0 path, so the script still finishes
# if staging is turned off. Weight load is the long pole and it is NOT disk-bound:
# with the checkpoint
# fully page-cached (0 MB/s of block-device reads, measured) the layerwise online
# MXFP8 path still needs 110-150s per shard. A shard is one of the 32 layers, so
# each one is ~192 per-expert expert_data.copy_() calls (64 experts x w1/w3/w2)
# driven serially from a single thread -- ~49 MB/s effective, against 3.3 GB/s
# for the same 27 MB copy measured standalone, i.e. per-copy overhead, not
# bandwidth. Budget ~75 min, hence 7200s deadlines: at 3600 the stage init dies
# just short of finishing. Quantization and TP live in the deploy YAML, not here
# -- a stage value always beats --quantization.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
#
# setsid: give the server and its workers their own process group, so the teardown at the
# bottom can name this run and only this run. See the comment there.
setsid vllm-omni serve $MODEL --omni --port $PORT --host 0.0.0.0 \
    --deploy-config "$DEPLOY" --trust-remote-code \
    --enable-diffusion-pipeline-profiler \
    --stage-init-timeout 7200 --init-timeout 7200 > $LOG 2>&1 &
SRV=$!
PGID=$(ps -o pgid= -p $SRV 2>/dev/null | tr -d ' ')
until curl -s -o /dev/null --max-time 5 http://127.0.0.1:$PORT/health; do
    kill -0 $SRV || { tail -20 $LOG; exit 1; }; sleep 10
done
grep -aiE 'mxfp8|quantization' $LOG | tail -3

# guidance_scale=1.0 disables CFG, which the CFG-distilled checkpoint expects
# (config.json: cfg_distilled=true) and which halves the work per step -- omitting
# it would silently pick the pipeline's 5.0 default and run an uncond pass too.
# bot_task=vanilla is required without an AR stage: there is nothing to produce
# CoT/recaption text, so the DiT must take the prompt directly.
# response_format=file makes the body the PNG itself, so -o writes the image.
time curl --max-time 1800 -sS -X POST http://127.0.0.1:$PORT/v1/images/generations \
    -H 'Content-Type: application/json' -o $OUT -d '{
    "prompt": "A cinematic portrait of an astronaut resting in a sunlit greenhouse, orchids in the foreground.",
    "size": "'"$SIZE"'",
    "num_inference_steps": '"$STEPS"',
    "guidance_scale": 1.0,
    "seed": 1234,
    "bot_task": "vanilla",
    "use_system_prompt": "en_vanilla",
    "response_format": "file"
}'

# On an error the body is JSON, not a PNG, and -o would leave it in place looking
# like a generated image. Check the magic bytes instead of trusting the name.
if [ "$(head -c8 "$OUT" | od -An -tx1 | tr -d ' \n')" = "89504e470d0a1a0a" ]; then
    echo "PNG OK: $OUT ($(stat -c%s "$OUT") bytes)"
else
    echo "NOT A PNG -- server said:"; head -c 400 "$OUT"; echo
fi

# Stop this run's workers and nobody else's. This used to be
#     pkill -TERM -f vLLM-Omni::DiffusionWorker
# which matches on process title, not parentage. The title prefix is hardcoded
# ("vLLM-Omni", vllm_omni/diffusion/worker/diffusion_worker.py:168), so the pattern names
# every vLLM-Omni worker in the PID namespace -- and this container is shared, with other
# runs live in that same namespace. This line was SIGKILLing a concurrent 2-card Wan ring
# run twice: once mid weight-load (exit 137, no traceback) and once part-way through
# denoising (curl "(52) Empty reply from server"), both of which read like bugs in the
# victim and were not. Kill the process group setsid gave us instead.
if [ -n "$PGID" ]; then kill -TERM -"$PGID" 2>/dev/null; else kill -TERM $SRV 2>/dev/null; fi
sleep 8
if [ -n "$PGID" ]; then kill -KILL -"$PGID" 2>/dev/null; else kill -KILL $SRV 2>/dev/null; fi
# Metric names differ between pipelines, so match the profiler prefix rather
# than the Wan scripts' fixed Pipeline.diffuse / Pipeline.forward pair.
grep -aoE '\[DiffusionPipelineProfiler\] [A-Za-z0-9_.]+ took [0-9.]+s' $LOG | tail -5
