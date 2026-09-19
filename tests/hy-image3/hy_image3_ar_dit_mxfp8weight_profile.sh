#!/bin/bash
# torch.profiler capture of the two-card AR+DiT MXFP8 run.
#
# Second run of hy_image3_ar_dit_mxfp8weight.sh: same model, same stages, same
# 8-step / guidance 1.0 workload, but the deploy config carries a per-stage
# profiler_config and the request is sent twice -- the first one absorbs the
# one-off warm-up (the first denoise step is ~2x the steady-state one) so the
# captured trace is steady state only.
#
# Same cases and the same request shapes as that script (DiT on FLASH_ATTN and compiled,
# stage 0 on FLASH_ATTN with XPU graph capture), so the two runs' end-to-end times compare
# case by case. Stage 0's max_tokens is 8192 in both deploy files, which is headroom
# rather than a target -- the two default cases emit 290-817 tokens -- and the note there
# explains why it has to be generous: truncation is silent.
#
# The warm-up request is doing two jobs now that the DiT is really compiled: the first
# request pays a ~3min inductor compile of the decoder block (it was ~50s eager), and
# only the steady-state request after it is captured.
#
#   bash hy_image3_ar_dit_mxfp8weight_profile.sh                       # both t2i cases
#   CASES=recaption bash hy_image3_ar_dit_mxfp8weight_profile.sh       # one case
#   PROFILE_STAGES='[1]' bash hy_image3_ar_dit_mxfp8weight_profile.sh  # DiT only
#
# Per case: one warm-up request (the first denoise step is ~2x the steady-state one),
# then start_profile -> the same request again -> stop_profile, so every capture is
# steady state. One server for both, for the same reason as the non-profiling script:
# start-up dominates.
#
# Only the DiT is profiled (PROFILE_STAGES='[1]'), not because the AR is uninteresting --
# it owns ~3/4 of the wall clock -- but because capturing stage 0 kills the server on this
# stack. See the note on PROFILE_STAGES below and stage 0's commented-out profiler_config
# in the deploy YAML; the AR's timing comes from stage_0_gen_ms instead.
#
# t2i only, like the non-profiling driver and for the same reason: this deploy sets stage
# 0 to FLASH_ATTN, which cannot apply the prefix-LM bidirectional mask, so an it2i
# request through it would return a plausible-looking wrong image. There is no profiling
# driver for it2i; to trace that flow, run hy_image3_it2i_mxfp8weight.sh's deploy with
# the prof file's profiler_config blocks pasted in.
# `long` (the multi-thousand-token CoT case, see LONG_PROMPT below) is not in the default
# list because it roughly doubles the run.
CASES=${CASES:-"recaption think_recaption"}
BACKEND=${BACKEND:-FLASH_ATTN}
# 8007 is the non-profiling run's port; separate so both can be up at once.
PORT=${PORT:-8008}
MODEL=${MODEL:-/mnt/disk1/hf_home/hub/HunyuanImage-3.0-Instruct-Distil/}
DEPLOY=${DEPLOY:-$(cd "$(dirname "$0")" && pwd)/hunyuan_image3_ar_dit_mxfp8_tp1_prof.yaml}
STEPS=${STEPS:-8}
GUIDANCE=${GUIDANCE:-1.0}
PROMPT=${PROMPT:-"A cinematic portrait of an astronaut resting in a sunlit greenhouse, orchids in the foreground."}
# The `long` case's prompt. Kept byte-identical to the non-profiling driver's default
# so the two runs' CoT lengths are comparable; the rationale for its shape is documented
# there. Measured here: num_tokens_out 1133 (warm-up) and 1162 (profiled), 5717 chars of
# CoT, 1m46s / 2m13s end to end, against 303/315 for recaption. Its DiT trace is the
# interesting artifact -- same 8 steps but a much longer text prefix, so
# pipeline.forward is 5.65s against recaption's 5.40s.
LONG_PROMPT=${LONG_PROMPT:-"A vast derelict orbital greenhouse seen in one continuous wide shot, and every detail matters: seven tiers of hydroponic racks curving away to the left, each tier planted with a different species -- pale orchids, blood-red amaranth, blue-green kelp in glass columns, wheat gone to seed, carnivorous pitcher plants, moss over broken soil trays, and a single grafted apple tree at the far end; three astronauts in mismatched suits, one kneeling to splice an irrigation line, one floating near the apex reading a clipboard of handwritten charts, one asleep against a rusted bulkhead with a helmet cradled in both arms; the Earth's terminator line crossing the panelled roof so the left third of the scene is lit by hard blue daylight, the middle by warm sodium work lamps on tripods, and the right third only by the green glow of the kelp columns; frost on the inside of the lower panes, condensation beading on the copper pipes, a slow drift of leaf debris and water droplets in zero gravity; lens: 24mm, deep focus, faint anamorphic flare where the daylight clips the frame edge. Account for every element, every light source and every material, and keep the three lighting zones physically consistent with each other."}
SIZE=${SIZE:-1024x1024}
# bot_task for the two t2i cases is the case name itself, set in the loop below.
# With SYS_TYPE pinned to en_unified it changes only the AR's opening token
# (_BOT_TASK_PRESETS, prompt_utils.py:67-73): <recaption> rewrites the prompt
# immediately, <think> writes an analysis block first and then the same recaption. So
# think_recaption is the recaption case plus a few thousand extra AR decode tokens in
# front of identical DiT work -- which is why the stage-1 traces of the two should
# come out near-identical and the stage-0 cost is the whole difference.
SYS_TYPE=${SYS_TYPE:-en_unified}
# JSON list of stage ids, or null for "every stage that declares a
# profiler_config". This is the /start_profile + /stop_profile request body
# (ProfileRequest, api_server.py:253).
#
# Stage 1 only, and it has to be: '[0,1]' takes the server down. Measured on the
# recaption case (303 output tokens) -- the stage-0 capture itself works and writes a
# valid 237 MB gzipped trace, but the profiler_stop that follows aborts the AR worker
# process outright:
#   terminate called after throwing an instance of 'std::runtime_error'
#     Kineto Profiler on XPU got error from function
#     libkineto::XpuptiActivityApi::clearActivities()
#     XpuptiActivityApi.cpp:148. Error code: 2 (PTI_ERROR_NOT_IMPLEMENTED).
# i.e. the XPU PTI plugin has no clearActivities, Kineto raises from a destructor path
# and nothing catches it. Consequences, all observed: /stop_profile never answers (the
# curl below hits its 600s max-time), the stage-1 stop queued behind it never runs so
# the DiT trace is lost too, and the API server dies leaving an orphaned DiffusionWorker
# holding its card. So there is no "profile the AR as well" option here until that
# plugin gains clearActivities -- the AR's cost has to come from the metrics instead
# (stage_0_gen_ms in the response, TPOT x num_tokens_out in the server log), which is
# also why --enable-ar-profiler is passed below.
PROFILE_STAGES=${PROFILE_STAGES:-'[1]'}
# Kept as a separate knob for the `long` case even though it now matches the default:
# that case is where a stage-0 capture would be worst (thousands of decode steps x 32
# layers) if the crash above is ever fixed and the default goes back to '[0,1]'.
LONG_PROFILE_STAGES=${LONG_PROFILE_STAGES:-'[1]'}
# Seconds to wait after /stop_profile. The export (trace + xlsx) runs in the
# worker's on_trace_ready and is not awaited by the endpoint, so tearing the
# server down too early loses the artifacts. 120 is ample for a stage-1 capture (8
# denoise steps); it was raised to 240 while stage 0 was in the capture set, which is
# no longer a supported configuration.
EXPORT_WAIT=${EXPORT_WAIT:-120}
OUT_DIR=${OUT_DIR:-/workspace/hy_image3_logs/prof}; mkdir -p "$OUT_DIR"
TAG=ar_dit_mxfp8_prof
LOG=$OUT_DIR/$TAG.log
# Which trace session belongs to which case: the profiler names its session dirs
# stage{id}_rank{rank} under a timestamp and knows nothing about cases, so the
# mapping has to be recorded here as the run goes.
MAP=$OUT_DIR/$TAG.trace_map.txt

# Validate before spending ten minutes on start-up.
for c in $CASES; do
    case "$c" in
        recaption|think_recaption|long) ;;
        # Not "unknown": it used to be a case here, and running it against this deploy
        # would trace a plausible-looking wrong image rather than fail.
        edit) echo "the it2i 'edit' case moved to hy_image3_it2i_mxfp8weight.sh -- this"
              echo "deploy sets stage 0 to FLASH_ATTN, which cannot apply the prefix-LM"
              echo "bidirectional mask and so silently corrupts image input"; exit 2 ;;
        *) echo "unknown case '$c' (want recaption, think_recaption, long)"; exit 2 ;;
    esac
done

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export NO_PROXY=0.0.0.0,127.0.0.1,localhost
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DIFFUSION_ATTENTION_BACKEND=$BACKEND
export VLLM_OMNI_STAGE_WEIGHTS=${STAGE_WEIGHTS:-1}
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
# The gate for the prof deploy's stage-0 compilation_config: without it, xpu.py:303 pins
# cudagraph_mode back to NONE and the timings here stop matching the non-profiling run
# (TPOT ~105 ms/token instead of ~26). Kept on so the two runs stay comparable, even
# though stage 0 is not the stage being traced -- and note that a stage-0 trace taken
# with it on is not comparable to the eager ones in hy_image3_logs/prof_eager_1837,
# because a replayed decode step no longer shows its ~2,329 launches.
export VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1}

# Compute-runtime debug keys for the profiled run. NEOReadDebugKeys=1 is the gate:
# without it the driver ignores every other debug key silently, so the two must be
# set together. ExperimentalCopyThroughLock=0 turns off the copy-through-lock path
# for host<->device transfers, which is what keeps them out of the profile as
# lock-serialised CPU time. Only set here, not in the non-profiling run -- these
# change driver behaviour, so timings taken with them are not comparable to
# timings taken without.
export NEOReadDebugKeys=1
export ExperimentalCopyThroughLock=0

# torch_profiler_dir in the deploy YAML is "./profile_traces", relative, so it
# resolves against the server process's cwd; run from OUT_DIR so the traces land
# beside the log. Every other path here is already absolute.
TRACE_DIR=$OUT_DIR/profile_traces
cd "$OUT_DIR" || exit 1

# --profiler-config is passed because it was asked for and it is a valid serve
# flag (vLLM EngineArgs.profiler_config, parsed as JSON), but note it does NOT
# reach the stages on the omni multi-stage path -- the deploy YAML's per-stage
# profiler_config is what does the work. Details in the YAML header.
PROFILER_CONFIG=${PROFILER_CONFIG:-'{"profiler":"torch","torch_profiler_dir":"./profile_traces","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true}'}

# --enable-ar-profiler is the AR counterpart of --enable-diffusion-pipeline-profiler
# (there is no --enable-ar-pipeline-profiler). Unlike --profiler-config it does reach the
# stages -- it is in the CLI -> stage injection loop, async_omni_engine.py:1372-1381 --
# but it is only stage_durations bookkeeping, nothing to do with torch.profiler: it makes
# omni_base.py:606 add ar_stage_<id> (seconds) for the non-current stage's events. The
# merge right below that one (omni_base.py:620) already publishes stage_<id>_gen_ms for
# every stage unconditionally, so the AR's cost is in the response either way and this
# flag just adds a duplicate key. Kept on for completeness; PROFILE_STAGES is what
# actually gets AR kernels into a trace.
setsid vllm-omni serve $MODEL --omni --port $PORT --host 0.0.0.0 \
    --deploy-config "$DEPLOY" --trust-remote-code \
    --enable-diffusion-pipeline-profiler \
    --enable-ar-profiler \
    --profiler-config "$PROFILER_CONFIG" \
    --stage-init-timeout 7200 --init-timeout 7200 > $LOG 2>&1 &
SRV=$!
PGID=$(ps -o pgid= -p $SRV 2>/dev/null | tr -d ' ')
teardown() {
    if [ -n "$PGID" ]; then kill -TERM -"$PGID" 2>/dev/null; else kill -TERM $SRV 2>/dev/null; fi
    sleep 8
    if [ -n "$PGID" ]; then kill -KILL -"$PGID" 2>/dev/null; else kill -KILL $SRV 2>/dev/null; fi
}
until curl -s -o /dev/null --max-time 5 http://127.0.0.1:$PORT/health; do
    kill -0 $SRV || { tail -40 $LOG; exit 1; }; sleep 10
done
# If this line is absent the endpoints below were never registered, which means no
# stage's engine_args carried a profiler_config.
grep -aiE 'profil|Traces will be saved' $LOG | tail -5

send_request() {  # send_request <out-json>
    SIZE_FIELD=""; [ -n "$SIZE" ] && SIZE_FIELD='"size": "'"$SIZE"'",'
    curl --max-time 3600 -sS -X POST http://127.0.0.1:$PORT/v1/images/generations \
        -H 'Content-Type: application/json' -o "$1" -d '{
        "prompt": "'"$REQ_PROMPT"'",
        '"$SIZE_FIELD"'
        "num_inference_steps": '"$STEPS"',
        "guidance_scale": '"$GUIDANCE"',
        "seed": 1234,
        "bot_task": "'"$BOT_TASK"'",
        "use_system_prompt": "'"$SYS_TYPE"'",
        "response_format": "b64_json"
    }'
}

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
: > "$MAP"; : > "$MAP".seen
for CASE in $CASES; do
    REQ_PROMPT=$PROMPT
    STAGES=$PROFILE_STAGES
    case "$CASE" in
        # Same bot_task as think_recaption, different prompt: the length comes from
        # what is being described, not from the tag.
        long) BOT_TASK=${LONG_BOT_TASK:-think_recaption}; REQ_PROMPT=$LONG_PROMPT
              STAGES=$LONG_PROFILE_STAGES ;;
        *)    BOT_TASK=$CASE ;;   # the case name *is* the bot_task
    esac
    OUT=$OUT_DIR/$TAG.$CASE.png
    COT=$OUT_DIR/$TAG.$CASE.cot.txt

    echo "== $CASE (t2i): warm-up request, not profiled =="
    time send_request "$OUT_DIR/$TAG.$CASE.warmup.json"

    echo "== $CASE: start_profile (stages=$STAGES) =="
    curl -sS --max-time 300 -X POST http://127.0.0.1:$PORT/start_profile \
        -H 'Content-Type: application/json' -d '{"stages": '"$STAGES"'}'; echo

    echo "== $CASE: profiled request =="
    time send_request "$OUT.json"

    echo "== $CASE: stop_profile =="
    curl -sS --max-time 600 -X POST http://127.0.0.1:$PORT/stop_profile \
        -H 'Content-Type: application/json' -d '{"stages": '"$STAGES"'}'; echo
    # The export (trace + xlsx) runs in the worker's on_trace_ready and is not
    # awaited by the endpoint, so the next case's start_profile -- or the teardown
    # after the last one -- must not race it.
    sleep "$EXPORT_WAIT"

    decode "$OUT.json" "$OUT" "$COT" || RC=1
    # Sessions written since the previous case, so the timestamps in the trace dir
    # can be attributed afterwards.
    for d in $(find "$TRACE_DIR" -mindepth 1 -maxdepth 2 -type d 2>/dev/null | sort); do
        grep -qxF "$d" "$MAP".seen 2>/dev/null && continue
        echo "$d" >> "$MAP".seen
        echo "$CASE  $d" >> "$MAP"
    done
done

grep -aoE '\[DiffusionPipelineProfiler\] [A-Za-z0-9_.]+ took [0-9.]+s' $LOG | tail -24

teardown
# One timestamped session dir per worker per capture, named stage<id>_rank<rank>: the
# gzipped chrome trace plus the op tables (summary + by_shape, since record_shapes is
# on).
echo "== artifacts under $TRACE_DIR =="
find "$TRACE_DIR" \( -name '*.json.gz' -o -name '*.pt.trace.json*' -o -name '*.xlsx' -o -name '*.txt' \) \
    -printf '%10s  %p\n' 2>/dev/null | sort -k2
echo "== case -> trace session =="
cat "$MAP"
exit $RC
