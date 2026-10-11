#!/bin/bash
# HunyuanImage-3.0-Instruct-Distil, online MXFP8 weights, AR (stage 0) + DiT
# (stage 1) on two XPU cards with TP=1 each. t2i only. Self-contained.
#
#   t2i   prompt -> AR thinks/recaptions -> DiT draws   (POST /v1/images/generations)
#
#   bash hy_image3_ar_dit_mxfp8weight.sh                 # both t2i cases
#   CASES=recaption bash hy_image3_ar_dit_mxfp8weight.sh
#   ZE_AFFINITY_MASK=2,3 STEPS=50 bash hy_image3_ar_dit_mxfp8weight.sh
#   VAE_CHANNELS_LAST=0 bash hy_image3_ar_dit_mxfp8weight.sh   # A/B the NCHW VAE decode
#
# Four cards, TP=2 per stage, via the sibling deploy -- DEPLOY is the only thing
# that changes the topology, and ZE_AFFINITY_MASK must list all four cards
# because the yaml's `devices` are LOGICAL indices mapped through it (stage 0
# takes physical 0,1 and stage 1 takes 2,3):
#
#   ZE_AFFINITY_MASK=0,1,2,3 DEPLOY=./hunyuan_image3_ar_dit_mxfp8_tp2.yaml \
#     bash hy_image3_ar_dit_mxfp8weight.sh
#
# DEPLOY is handed to --deploy-config verbatim, so it resolves against YOUR cwd,
# not the script's -- the ./ above assumes you are in this directory, as the
# examples at the top do. Only the default is absolute (it is derived from $0).
# Pass a full path when running from anywhere else.
#
# Measured that way 2026-09-28, vllm-omni 0.30.0_rebase (07ba68abe) on
# 10.239.11.197 cards 0-3: both t2i cases produced 1024x1024 PNGs (3,147,861
# bytes each), model.forward 0.3739 s, final_layer.forward 0.0044 s, vae.decode
# 0.752-0.757 s, pipeline forward 3.970-3.975 s; recaption 1m33s wall including
# start-up, think_recaption 16.4 s on the already-warm server.
#
# TP=2 is a SCALING DATAPOINT, not a speedup: see the tp2 yaml's header for why
# (the AR decode is host-bound on op launches, not memory-bound, so sharding adds
# a per-layer allreduce to a step that was already launch-dominated). What it does
# buy is headroom -- ~41 GiB per card instead of ~82.5 GiB. Compare against the
# tp1 numbers below, and state which deploy a number came from.
#
# it2i -- image + edit instruction, where the AR *looks at the image* through its
# SigLIP2 tower before recaptioning -- is the other direction these two stages can
# serve and it is the reason to run AR+DiT at all (the DiT-only deployment has no
# vision tower and no CoT, the AR-only one can read an image but cannot draw). It is
# a separate test, hy_image3_it2i_mxfp8weight.sh, not a case here: the deploy this
# script uses asks for FLASH_ATTN on stage 0, which is correct only for text-only AR
# input, and the attention backend is chosen once per server at init. So the two
# flows cannot share a server, and `edit` is rejected below rather than silently
# producing a wrong image.
#
# Expect roughly twice the DiT-only start-up: each stage loads its own full copy
# of the 80B checkpoint (~82.5 GiB after MXFP8), in parallel but competing for
# host bandwidth.
# Both cases served by one server -- start-up dominates the cost of this test (each
# stage loads its own ~82.5 GiB copy of the checkpoint), so running them in one
# process is worth far more than isolating them:
#   recaption        bot_task recaption       (AR rewrites, then draws)
#   think_recaption  bot_task think_recaption (adds the <think> block)
#   long             bot_task think_recaption on $LONG_PROMPT -- the long-CoT case,
#                    ~1150 AR tokens instead of a few hundred. Not in the default
#                    list: it roughly doubles the run. Ask for it with
#                    CASES="recaption think_recaption long".
# Measured with the current deploy (stage 0 flash attention + XPU graph, stage 1
# compiled), think_recaption: e2e 27.5-29.0s, stage_0 19.6-21.1s, stage_1 7.7s,
# TPOT 26.1-26.9 ms/token at 740-817 output tokens. The same server hour with stage 0
# eager and on Triton attention gave e2e 89-97s and TPOT 104-109 ms, so treat any
# think_recaption above ~40s as a sign the graph did not take -- grep the log for
# "Capturing CUDA graphs (decode, FULL)".
CASES=${CASES:-"recaption think_recaption"}
BACKEND=${BACKEND:-FLASH_ATTN}
PORT=${PORT:-8007}
MODEL=${MODEL:-/mnt/disk1/hf_home/hub/HunyuanImage-3.0-Instruct-Distil/}
DEPLOY=${DEPLOY:-$(cd "$(dirname "$0")" && pwd)/hunyuan_image3_ar_dit_mxfp8_tp1.yaml}
# 8 steps is what the Distil model card recommends; the undistilled
# HunyuanImage-3.0-Instruct wants 50.
STEPS=${STEPS:-8}
# guidance_scale 1.0 disables CFG, which the CFG-distilled checkpoint expects
# (config.json: cfg_distilled=true) and which halves the work per step. The
# checkpoint's generation_config.json says 2.5, but vllm-omni does not read it,
# and 1.0 is what the DiT-only run in this directory was validated with -- keep
# them equal so the tests stay comparable.
GUIDANCE=${GUIDANCE:-1.0}

# ---- t2i knobs ----
PROMPT=${PROMPT:-"A cinematic portrait of an astronaut resting in a sunlit greenhouse, orchids in the foreground."}
# Pinned to match the DiT-only test. SIZE= (empty) instead hands the choice to
# the AR: resolve_stop_token_ids() then stops it on <img_ratio_*> and
# _extract_ratio_index() (stage_input_processors/hunyuan_image3.py:73) reads the
# ratio back. Either way the DiT reuses the KV prefix up to the CoT end
# (_get_kv_reuse_len in hunyuan_image3_transformer.py).
SIZE=${SIZE:-1024x1024}
# ---- the `long` case: drive the AR toward a multi-thousand-token CoT ----
# The other t2i prompts produce 290 (recaption) / 734 (think_recaption) output tokens,
# which never approaches the deploy's max_tokens and so never exercises the long-CoT
# path: the KV growth, the per-token cost at depth, and the truncation edge. Length is
# not directly settable for /v1/images/* (max_tokens is deploy-only and there is no
# min_tokens knob on the request), so it has to be *earned* from the prompt -- many
# distinct subjects, each with its own spatial relation, material and lighting
# constraint, plus an explicit instruction to account for every element. Under
# bot_task think_recaption the AR then writes a per-element analysis before the
# recaption, and that is where the tokens go.
#
# Measured, so nobody has to assume: this prompt earns num_tokens_out 1133 and 1162 on two
# runs -- 5717 characters of CoT -- against 303/315 for the plain recaption case and
# 828/766 for think_recaption on the short prompt. So it is ~4x the short cases and ~5k
# *characters*, not 5k tokens; prompting alone does not get this model to 5k tokens for a
# single image request, and there is no min_tokens knob on /v1/images/* to force it. The
# per-token cost is unchanged (2m13s for 1162 tokens is the same ~115ms TPOT), so what
# this case actually exercises is KV growth and the long-prefix DiT attention, not a
# different decode regime. Check num_tokens_out in the server log rather than assuming.
LONG_PROMPT=${LONG_PROMPT:-"A vast derelict orbital greenhouse seen in one continuous wide shot, and every detail matters: seven tiers of hydroponic racks curving away to the left, each tier planted with a different species -- pale orchids, blood-red amaranth, blue-green kelp in glass columns, wheat gone to seed, carnivorous pitcher plants, moss over broken soil trays, and a single grafted apple tree at the far end; three astronauts in mismatched suits, one kneeling to splice an irrigation line, one floating near the apex reading a clipboard of handwritten charts, one asleep against a rusted bulkhead with a helmet cradled in both arms; the Earth's terminator line crossing the panelled roof so the left third of the scene is lit by hard blue daylight, the middle by warm sodium work lamps on tripods, and the right third only by the green glow of the kelp columns; frost on the inside of the lower panes, condensation beading on the copper pipes, a slow drift of leaf debris and water droplets in zero gravity; lens: 24mm, deep focus, faint anamorphic flare where the daylight clips the frame edge. Account for every element, every light source and every material, and keep the three lighting zones physically consistent with each other."}
# With an AR stage present the CoT/recaption is generated, so bot_task must NOT
# be `vanilla` -- that value exists precisely because the DiT-only deployment has
# nothing to produce the CoT and must take the prompt directly.
#
# bot_task for the two t2i cases comes from the case name itself (see the loop at
# the bottom). What it actually changes, with SYS_TYPE pinned to en_unified below:
# only the AR's opening token (_BOT_TASK_PRESETS, prompt_utils.py:67-73). The
# unified system prompt keys its mode off exactly that -- <recaption> rewrites the
# prompt straight away, <think> writes an analysis block first and then the same
# recaption -- so think_recaption is the recaption case plus a few thousand extra AR
# decode tokens in front of identical DiT work. That is the comparison these two
# cases are for.
# (Also worth knowing: with SYS_TYPE pinned to en_unified, bot_task `think` and
# `think_recaption` are the same request. They differ only in the preset sys_type,
# which this pinning overrides.)
SYS_TYPE=${SYS_TYPE:-en_unified}

# /workspace is the host mount, so these outlive the container.
OUT_DIR=${OUT_DIR:-/workspace/hy_image3_logs}; mkdir -p "$OUT_DIR"
TAG=ar_dit_mxfp8_$(echo "$BACKEND" | tr 'A-Z' 'a-z')
LOG=$OUT_DIR/$TAG.log

# Two cards, one per stage. The deploy YAML says devices "0" and "1", which are
# logical indices resolved through this mask, so the mask picks the physical
# pair. Mask N is /dev/dri/card(N+1) in this node's level-zero order (see the
# ze_peer device map in ~/ze_peer_debug.sh); mask 0 is the Gen5-x4 degraded link,
# so 2,3 is the pair to benchmark on when it is free. This box is shared -- check
# per-card free memory before starting, each stage wants ~110 GiB at
# gpu_memory_utilization 0.9 and the init fails outright if it is not there.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export NO_PROXY=0.0.0.0,127.0.0.1,localhost
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Both backends work for the DiT. FLASH_ATTN goes through the piecewise split in
# vllm_omni/diffusion/attention/utils/piecewise_attn.py, which is what lets the XPU
# varlen kernel serve HunyuanImage3's mixed causal/full 4D mask; it matched a 4D-mask
# SDPA reference to 5e-4 at DiT shapes on this box. BACKEND=TORCH_SDPA for the
# reference path.
#
# This only reaches the DiT (diffusion/data.py:2021), and the DiT does take it: the
# selector logs FLASH_ATTN and the trace is full of XeFMHAFwdKernel. It does NOT reach
# the AR: stage 0's backend comes from vLLM's XPU selector, and the deploy YAML's
# `attention_backend: FLASH_ATTN` is what steers that one (the selector would otherwise
# fall back to Triton for this prefix-LM model, xpu.py:181). The SigLIP2 tower is always
# TORCH_SDPA -- head_dim 72 is not 64-byte aligned, so the retarget in siglip2.py is
# forced, not chosen. Three backends in one server, one per stage, set three ways.
export DIFFUSION_ATTENTION_BACKEND=$BACKEND

# The gate for the deploy's stage-0 compilation_config: without this,
# vllm/platforms/xpu.py:303 pins cudagraph_mode back to NONE and logs "XPU Graph is
# disabled by environment variable", so the YAML's FULL_DECODE_ONLY silently does
# nothing and TPOT goes back to ~105 ms/token. supports_xpu_graph() needs torch
# >= 2.11.0.dev (torch_utils.py:1018); this container has 2.14.0a0, and the capture is
# single-GPU only, which is what stage 0 is (TP=1, one card).
export VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1}

# Opt into pinned-host staging of the checkpoint tensors (see load_weights in
# hunyuan_image3_transformer.py and hunyuan_image3.py). A host->device copy
# straight out of the safetensors mapping cannot be DMA'd by level-zero and runs
# at ~71 MB/s; via a pinned buffer it is ~4.6 GB/s. Off by default in the
# loaders, so it has to be asked for here -- and it pays twice in this test,
# once per stage. STAGE_WEIGHTS=0 to measure the unpatched path.
export VLLM_OMNI_STAGE_WEIGHTS=${STAGE_WEIGHTS:-1}
# Sized for the STAGE_WEIGHTS=0 path so the run still finishes with staging off:
# the online MXFP8 loader needs 110-150s per shard there, and a shard is one of
# the 32 layers (~192 per-expert copies each, driven serially). Two stages of
# that is well past the 3600s default.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}

# VAE decode layout. Stage 1's decoder runs channels-last (NDHWC) by default on
# XPU -- ~1.9x on vae.decode, already folded into the stage_1 ~7.7s above -- so
# leaving this unset is the fast path and needs nothing set. The knob is the
# HY3_VAE_DECODE_CHANNELS_LAST env flag that resolve_decode_channels_last() reads
# (autoencoder.py): =1 pins the fast NDHWC path on regardless of platform, =0
# A/Bs the old NCHW decode, unset keeps the platform default. Only export it when
# VAE_CHANNELS_LAST is set, so an unset run leaves the env untouched.
VAE_CHANNELS_LAST=${VAE_CHANNELS_LAST:-}
case "$VAE_CHANNELS_LAST" in
    1|true|on)   export HY3_VAE_DECODE_CHANNELS_LAST=1 ;;
    0|false|off) export HY3_VAE_DECODE_CHANNELS_LAST=0 ;;
    "")          ;;  # unset -> platform default (NDHWC on XPU)
    *) echo "VAE_CHANNELS_LAST='$VAE_CHANNELS_LAST' unrecognized (want 1/0)"; exit 2 ;;
esac

# Validate before spending ten minutes on start-up.
for c in $CASES; do
    case "$c" in
        recaption|think_recaption|long) ;;
        # Not "unknown": it used to be a case here, and running it against this deploy
        # would return a plausible-looking wrong image rather than fail.
        edit) echo "the it2i 'edit' case moved to hy_image3_it2i_mxfp8weight.sh -- this"
              echo "deploy sets stage 0 to FLASH_ATTN, which cannot apply the prefix-LM"
              echo "bidirectional mask and so silently corrupts image input"; exit 2 ;;
        *) echo "unknown case '$c' (want recaption, think_recaption, long)"; exit 2 ;;
    esac
done

# setsid: give the server and its workers their own process group, so the
# teardown at the bottom names this run and only this run. This container's PID
# namespace is shared with other users' runs and the worker process titles are
# not unique to a run, so pkill on the title is not an option -- it has SIGKILLed
# concurrent runs before.
#
# --enable-ar-profiler is the AR counterpart of the flag above (there is no
# --enable-ar-pipeline-profiler). It is pure stage_durations bookkeeping -- it makes
# omni_base.py:606 add ar_stage_<id> in seconds -- and it is nearly redundant: the merge
# 14 lines below that already emits stage_<id>_gen_ms for every stage unconditionally, so
# the response carries the AR's generate time with or without it. Measured on this test:
# stage_0_gen_ms ~30-36k against stage_1_gen_ms ~13k, i.e. the AR is ~3/4 of the request.
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

# The endpoint answers with JSON: the PNG as data[0].b64_json and the AR's CoT as
# cot_output. b64_json rather than the generations-only `response_format: file`
# shortcut, so this stays comparable with the it2i test (/v1/images/edits accepts
# nothing else).
#
# cot_output used to come back null here: the multi-stage /v1/images/generations
# handler destructured the CoT out and threw it away (api_server.py:1961-1966) while
# the edits handler passed it through. That is fixed, so an empty CoT on a
# think_recaption run now means something actually went wrong in stage 0 -- check
# num_tokens_out in the server log, which should show a few hundred tokens.
decode() {  # decode <json> <png> <cot>
    python - "$1" "$2" "$3" <<'PY'
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
print("CoT:", cot[:300] if cot else "(empty -- stage 0 returned no text)")
sys.exit(0 if ok else 1)
PY
}

RC=0
for CASE in $CASES; do
    REQ_PROMPT=$PROMPT
    case "$CASE" in
        # Same bot_task as think_recaption, different prompt: the length comes from
        # what is being described, not from the tag.
        long) BOT_TASK=${LONG_BOT_TASK:-think_recaption}; REQ_PROMPT=$LONG_PROMPT ;;
        *)    BOT_TASK=$CASE ;;   # the case name *is* the bot_task
    esac
    OUT=$OUT_DIR/$TAG.$CASE.png
    COT=$OUT_DIR/$TAG.$CASE.cot.txt
    JSON=$OUT.json
    echo "== $CASE (t2i) =="
    SIZE_FIELD=""; [ -n "$SIZE" ] && SIZE_FIELD='"size": "'"$SIZE"'",'
    time curl --max-time 3600 -sS -X POST http://127.0.0.1:$PORT/v1/images/generations \
        -H 'Content-Type: application/json' -o "$JSON" -d '{
        "prompt": "'"$REQ_PROMPT"'",
        '"$SIZE_FIELD"'
        "num_inference_steps": '"$STEPS"',
        "guidance_scale": '"$GUIDANCE"',
        "seed": 1234,
        "bot_task": "'"$BOT_TASK"'",
        "use_system_prompt": "'"$SYS_TYPE"'",
        "response_format": "b64_json"
    }'
    decode "$JSON" "$OUT" "$COT" || RC=1
done

teardown
# Metric names differ between pipelines, so match the profiler prefix rather
# than a fixed metric pair.
grep -aoE '\[DiffusionPipelineProfiler\] [A-Za-z0-9_.]+ took [0-9.]+s' $LOG | tail -8
# The AR side of the same question, and the two lines to check when a run looks slow:
# the capture line proves the graph took, and tpot_ms is what it bought.
grep -aoE 'Capturing CUDA graphs \([a-z, A-Z_]+\)|Graph capturing finished[^|]*' $LOG | tail -2
grep -aoE 'tpot_ms[^0-9]*[0-9.]+|num_tokens_out[^0-9]*[0-9,]+' $LOG | tail -6
exit $RC
