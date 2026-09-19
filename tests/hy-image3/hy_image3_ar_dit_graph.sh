#!/bin/bash
# HunyuanImage-3.0-Instruct-Distil, AR (stage 0) + DiT (stage 1), two cards, TP=1 each, online
# MXFP8, t2i. Serves one arm of the denoise-graph A/B (`diffusion_use_graph`,
# vllm_omni/diffusion/graph.py): start the server, send N requests, optionally under unitrace.
# It measures nothing itself -- it produces a server log, one response JSON + PNG per request
# and (PROFILE=1) a unitrace timeline, and the last section prints what to run on them.
#
#   GRAPH=1 bash hy_image3_ar_dit_graph.sh              # graph on,  wall clock
#   GRAPH=0 bash hy_image3_ar_dit_graph.sh              # graph off, wall clock -> the A/B
#   GRAPH=1 PROFILE=1 bash hy_image3_ar_dit_graph.sh    # graph on, under unitrace
#
# One arm per run, and one run at a time unless you give each its own PORT and ZE_AFFINITY_MASK.
# Two profiled runs at once now get distinct unitrace sessions (see SESSION), but they still share
# the host, which is one more reason a traced wall clock is not a number to quote.
#
# Stage 0's XPU graph is ON in both arms (VLLM_XPU_ENABLE_XPU_GRAPH=1): without it the AR
# decodes at ~94 ms/token instead of ~24.5 and swamps everything stage 1 does. The GRAPH=0 arm
# is NOT eager either -- the DiT is regionally torch.compile'd either way ("Regional compilation
# applied to 32 module(s)"), so what changes is only whether that same compiled body is replayed
# from a graph or submitted kernel by kernel ("launched").
#
# DO NOT COMPARE A PROFILE=1 WALL CLOCK WITH A PROFILE=0 ONE, OR ACROSS ARMS UNDER TRACE.
# Tracing hooks every host-side kernel launch, so it taxes the launched arm (28,970 host calls in
# one traced denoise loop) far more than the replaying one (13,636, of which 13,541 are step 0
# plus the capture). Measured: traced, graph-on looked 17% faster per denoise loop; untraced it is
# 0.3-0.5 s per request SLOWER (stage_1 7.52/7.55 s against 6.99/7.44 s), because every request
# re-captured. Quote wall clock from PROFILE=0; take structure from PROFILE=1. Device *busy* time
# is the part of a trace that agrees either way (3702.59 vs 3703.44 ms per 8-step loop).
#
# WHY NOT --enable-diffusion-pipeline-profiler, in EITHER mode: its targets (patch_embed,
# final_layer, model.layers[0]) sit inside the denoise body and each one does a full device
# synchronize. A host sync under a Level Zero graph capture is not a stall but an error
# (UR_RESULT_ERROR_UNSUPPORTED_FEATURE), so every capture failed with it on; and replayed steps
# never run those wrappers, so it also confounds the A/B. unitrace replaces it.
#
# COST: two cards for ~20-30 min (each stage loads its own ~82.5 GiB MXFP8 copy of the 80B
# backbone). The box is shared, so this checks free memory before launching, and its teardown
# kills only its own process group.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

GRAPH=${GRAPH:-1}                # diffusion_use_graph for stage 1: the whole experiment
PROFILE=${PROFILE:-0}            # 1 = run the server under unitrace and trace the last request
MODEL=${MODEL:-/mnt/disk1/hf_home/hub/HunyuanImage-3.0-Instruct-Distil/}
BASE_DEPLOY=${BASE_DEPLOY:-$HERE/hunyuan_image3_ar_dit_mxfp8_tp1.yaml}
PORT=${PORT:-8011}
STEPS=${STEPS:-8}
SIZE=${SIZE:-1024x1024}
# 1.0, i.e. NO classifier-free guidance, same as hy_image3_ar_dit_mxfp8weight.sh. This is not a
# style choice: cfg_factor is 1 + int(guidance_scale > 1.0), so anything above 1.0 doubles the
# denoise batch (pipeline_hunyuan_image3.py, latent_model_input = cat([latents] * cfg_factor)) and
# costs ~1.8x per step -- measured 1573 ms/step at 7.5 against 874 ms at 1.0, and it changes the
# image as well. The distilled checkpoint is meant to be run guidance-free; the log line
# "runs without classifier-free guidance" is how to confirm a run was.
GUIDANCE=${GUIDANCE:-1.0}
BOT_TASK=${BOT_TASK:-recaption}
SYS_TYPE=${SYS_TYPE:-en_unified}
# Byte-identical to hy_image3_ar_dit_mxfp8weight.sh's default, as are SIZE, STEPS and GUIDANCE:
# every one of those changes the denoise cost, so they have to match for these numbers to be
# comparable with everything else measured on this deploy.
PROMPT=${PROMPT:-"A cinematic portrait of an astronaut resting in a sunlit greenhouse, orchids in the foreground."}
# Rep 1 pays the DiT's ~3 min inductor compile and, with the graph on, the first capture, so it
# is never quoted. PROFILE=1 traces only the last rep.
REPS=${REPS:-3}
NEED_FREE_GIB=${NEED_FREE_GIB:-110}

OUT_DIR=${OUT_DIR:-/workspace/hy_image3_logs}; mkdir -p "$OUT_DIR"   # /workspace is host-mounted
TAG=${TAG:-ar_dit}$([ "$PROFILE" = 1 ] && echo _unitrace).graph$GRAPH
LOG=$OUT_DIR/$TAG.log
DEPLOY=$OUT_DIR/$TAG.deploy.yaml
RESULT_DIR=$OUT_DIR/$TAG.unitrace
UNITRACE=${UNITRACE:-/opt/gfx-deps/profiling_tools/bin/unitrace}
# Alphanumeric only: the dynamic-control flags reject the rest. The pid is in it because the name
# is the ONLY thing --resume/--stop address, and it is global: running both arms at once under one
# shared name made each run's --stop end the other's collection, so the arm that was still going
# traced nothing ("No event of interest is logged for process N (python)", 5 result dirs instead
# of 95). Measured on 2026-09-19 with two concurrent PROFILE=1 runs.
SESSION=${SESSION:-hy3graph${GRAPH}p$$}
EVENT_BUFFER=${EVENT_BUFFER:-2000000}   # unitrace's default is unlimited: GiB of host memory
EXIT_WAIT_S=${EXIT_WAIT_S:-420}  # how long teardown waits for a timeline to be written

# Two cards, one per stage. The deploy says devices "0" and "1", logical indices resolved through
# this mask. Mask 0 is this node's Gen5-x4 degraded card, kept as the default only because every
# recorded number used it.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export NO_PROXY=0.0.0.0,127.0.0.1,localhost
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Stage 1's attention (stage 0's is the deploy's attention_backend). This is also why the deploy
# is t2i only: FLASH_ATTN cannot apply the prefix-LM bidirectional mask an it2i request needs,
# and would return a plausible-looking wrong image rather than fail.
export DIFFUSION_ATTENTION_BACKEND=${DIFFUSION_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1}   # stage 0's graph, both arms
# Pinned-host staging of the checkpoint: a H2D copy straight out of the safetensors mapping
# cannot be DMA'd (~71 MB/s vs ~4.6 GB/s), and this run pays it twice. Start-up only.
export VLLM_OMNI_STAGE_WEIGHTS=${STAGE_WEIGHTS:-1}
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}

echo "=== $TAG: diffusion_use_graph=$GRAPH, profiler=$PROFILE, $REPS reps ==="

# The committed deploy plus stage 1's denoise-graph key, in a run-local copy kept next to the
# results as the deploy actually used. Appended after "- stage_id: 1" (which appears exactly
# once) rather than replaced, because the base file does not carry the key at all.
[ "$(grep -c '^  - stage_id: 1$' "$BASE_DEPLOY")" = 1 ] || { echo "no unique stage_id: 1 anchor in $BASE_DEPLOY"; exit 1; }
sed "/^  - stage_id: 1\$/a\\    diffusion_use_graph: $([ "$GRAPH" = 1 ] && echo true || echo false)" \
    "$BASE_DEPLOY" > "$DEPLOY"
grep -A1 '^  - stage_id: 1$' "$DEPLOY"

# A server already on PORT answers /health, so the wait below would "come ready" instantly and
# every request would go to SOMEONE ELSE'S server. Measured, and it silently invalidated a whole
# run: ours died with OSError: [Errno 98] Address already in use while a foreign server served all
# three requests (stage_1 11.3 s from a deploy that was not this one) and unitrace, having no
# child to follow, wrote "No event of interest is logged" and no timeline at all.
if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    exec 3>&-
    echo "port $PORT is already in use -- another server (possibly another user's) holds it."
    echo "Pick a free one: PORT=8012 GRAPH=$GRAPH PROFILE=$PROFILE bash $0"
    exit 1
fi

# xpu-smi does not exist in this container; torch.xpu.mem_get_info does, and it reports the
# masked cards only. Launching while the other arm still holds its memory fails mid-weight-load,
# ten minutes in.
python3 -c "
import sys, torch
free = [torch.xpu.mem_get_info(i)[0] / 2**30 for i in range(torch.xpu.device_count())]
print('free GiB per card of mask $ZE_AFFINITY_MASK:', [round(g, 1) for g in free])
sys.exit(0 if all(g >= $NEED_FREE_GIB for g in free) else 1)
" || { echo "< $NEED_FREE_GIB GiB free: the cards are busy, not launching"; exit 1; }

# setsid gives the server and its workers their own process group, so teardown names this run and
# only this run. This container's PID namespace is shared with other users' runs and the worker
# process titles are not unique, so pkill on a title is not an option -- it has SIGKILLed
# concurrent runs before. It is also what makes `kill -STOP -$PGID` in teardown safe.
#
# --enable-ar-profiler is stage_durations bookkeeping, not a sampling profiler, and synchronizes
# nothing (unlike the diffusion pipeline profiler, which is why that one is absent).
SERVE=(vllm-omni serve "$MODEL" --omni --port "$PORT" --host 0.0.0.0 --deploy-config "$DEPLOY"
       --trust-remote-code --enable-ar-profiler --stage-init-timeout 7200 --init-timeout 7200)
if [ "$PROFILE" = 1 ]; then
    mkdir -p "$RESULT_DIR"
    # --chrome-kernel-logging   per-kernel rows: what makes a replayed step measurable at all
    # --chrome-device-logging   the DEVICE track, i.e. execution as opposed to submission
    # --start-paused --session  dynamic control; without it the trace is all weight loading
    # --follow-child-process 1  default, but the workers are spawned children
    # --teardown-on-signal 15   a SIGTERM teardown still flushes the timeline
    # --pid --tid               the API server and the two workers must be separable
    UNITRACE_ARGS=${UNITRACE_ARGS:-"--chrome-kernel-logging --chrome-device-logging \
      --chrome-event-buffer-size $EVENT_BUFFER --pid --tid --verbose \
      --follow-child-process 1 --teardown-on-signal 15 \
      --start-paused --session $SESSION --result-dir $RESULT_DIR"}
    # shellcheck disable=SC2086
    setsid $UNITRACE $UNITRACE_ARGS "${SERVE[@]}" > "$LOG" 2>&1 &
else
    setsid "${SERVE[@]}" > "$LOG" 2>&1 &
fi
SRV=$!
PGID=$(ps -o pgid= -p $SRV 2>/dev/null | tr -d ' ')
echo "server pid $SRV (pgid ${PGID:-?}), log $LOG"
[ "$PROFILE" = 1 ] && echo "unitrace session $SESSION (this is the name --resume/--stop take)"

teardown() {
    if [ "$PROFILE" != 1 ]; then
        kill -TERM ${PGID:+-}"${PGID:-$SRV}" 2>/dev/null; sleep 8
        kill -KILL ${PGID:+-}"${PGID:-$SRV}" 2>/dev/null
        return 0
    fi
    # Each traced process writes its whole timeline when it EXITS (--stop only ends collection),
    # and 15 MB of that takes over a minute -- so the flush is the fiddly part. SIGTERM to the
    # group is not enough: vLLM's frontend tears its own engine-core children down and SIGKILLs
    # the stragglers, which truncated the AR's file at a page-aligned 483,328 bytes twice while
    # the DiffusionWorker (not on that leash) wrote fine. So freeze the group first, so the
    # frontend's shutdown code cannot run at all, then flush each traced worker by hand.
    $UNITRACE --stop "$SESSION" 2>/dev/null; sleep 5
    [ -n "${PGID:-}" ] || { kill -TERM $SRV; sleep "$EXIT_WAIT_S"; kill -KILL $SRV 2>/dev/null; return 0; }
    kill -STOP -"$PGID" 2>/dev/null
    # unitrace names each dir <comm>.<pid>, so the workers are python.<pid> today but that is a
    # process title, not a promise. Take every live pid in our own group EXCEPT the frontend: it
    # is the one whose shutdown SIGKILLs the workers, so it has to stay frozen until they are done.
    local pid prev cur alive d p pids=""
    for d in "$RESULT_DIR"/*.[0-9]*; do
        pid=${d##*.}
        case ${d##*/} in ldconfig.*|uname.*|unitrace.*) continue;; esac
        [ -d "$d" ] && [ "$pid" != "$SRV" ] && kill -0 "$pid" 2>/dev/null || continue
        [ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$PGID" ] || continue
        pids="$pids $pid"; kill -CONT "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
    done
    if [ -z "$pids" ]; then   # no traced process left to flush: nothing to wait for
        echo "  WARNING: no live traced worker under $RESULT_DIR to flush -- there will be no"
        echo "  timeline. Usually this means the server never really started under unitrace:"
        echo "  grep -a 'Error\\|Traceback\\|No event of interest' $LOG"
        kill -CONT -"$PGID" 2>/dev/null; kill -KILL -"$PGID" 2>/dev/null; return 0
    fi
    echo "  flushing traced workers:$pids (up to $EXIT_WAIT_S s)"
    prev=-1
    local st
    for _ in $(seq 1 $((EXIT_WAIT_S / 10))); do
        cur=$(find "$RESULT_DIR" -name 'chrome_trace.json' -printf '%s+' 2>/dev/null)0; cur=$((cur))
        # A worker that has exited here becomes a ZOMBIE, because the parent that would reap it is
        # the frontend we just froze -- and `kill -0` on a zombie succeeds. Read the state out of
        # /proc instead, or this loop never breaks early and every profiled run waits out
        # EXIT_WAIT_S. (Field 3 of stat, after the comm field, which can itself contain spaces.)
        alive=0
        for p in $pids; do
            st=$(cat /proc/"$p"/stat 2>/dev/null) || continue
            [ -n "$st" ] || continue
            st=${st##*\) }; st=${st%% *}
            [ "$st" = Z ] || alive=$((alive + 1))
        done
        echo "    bytes $cur, $alive worker(s) alive"
        [ "$cur" -gt 0 ] && [ "$cur" = "$prev" ] && [ "$alive" = 0 ] && break
        prev=$cur; sleep 10
    done
    kill -CONT -"$PGID" 2>/dev/null   # now the frontend may shut down; nothing left to lose
    kill -TERM -"$PGID" 2>/dev/null; sleep 20; kill -KILL -"$PGID" 2>/dev/null
}
trap teardown EXIT

until curl -s -o /dev/null --max-time 5 http://127.0.0.1:"$PORT"/health; do
    kill -0 $SRV 2>/dev/null || { echo "server died:"; tail -60 "$LOG"; exit 1; }
    sleep 10
done
# ...and prove the thing that answered is ours: under unitrace $SRV is the unitrace process that
# exec'd into the server, so compare process GROUPS, not pids. Belt and braces with the port check
# above, because being wrong here costs a whole run and is invisible in the results.
APID=$(grep -aom1 'APIServer pid=[0-9]*' "$LOG" | sed 's/.*pid=//')
if [ -z "$APID" ] || [ "$(ps -o pgid= -p "$APID" 2>/dev/null | tr -d ' ')" != "${PGID:-}" ]; then
    echo "/health answered but the server is not ours (APIServer pid=${APID:-none}, our pgid ${PGID:-?}):"
    tail -40 "$LOG"; exit 1
fi
# Both of these must appear or the run is measuring something else: "Regional compilation applied
# to 32 module(s)" in BOTH arms, and stage 0's "Capturing CUDA graphs (decode, FULL)".
grep -aiE "XPU Graph is|Capturing CUDA graphs|Regional compilation|diffusion_use_graph" "$LOG" | tail -6

request() {  # request <label>: one generation, saved as <label>.json + <label>.png
    local stem=$OUT_DIR/$TAG.$1 t0 t1
    t0=$(date +%s%3N)
    curl -s --max-time 3600 -H 'Content-Type: application/json' -o "$stem.json" \
        -d "{\"prompt\": \"$PROMPT\", \"size\": \"$SIZE\", \"num_inference_steps\": $STEPS,
             \"guidance_scale\": $GUIDANCE, \"seed\": 1234, \"bot_task\": \"$BOT_TASK\",
             \"use_system_prompt\": \"$SYS_TYPE\", \"response_format\": \"b64_json\"}" \
        http://127.0.0.1:"$PORT"/v1/images/generations
    t1=$(date +%s%3N)
    # The image comes back as base64 inside the response JSON. Cut it out with grep/sed rather
    # than a json parser so nothing here needs python: the pattern tolerates either spacing,
    # because a compact JSONResponse and a default json.dumps differ by the space after the colon.
    grep -o '"b64_json"[[:space:]]*:[[:space:]]*"[^"]*"' "$stem.json" \
        | sed 's/.*"b64_json"[[:space:]]*:[[:space:]]*"//; s/"$//' | base64 -d > "$stem.png"
    echo "$1: e2e $((t1 - t0)) ms | $(grep -o '"stage_[01]_gen_ms"[[:space:]]*:[[:space:]]*[0-9.]*' "$stem.json" | tr '\n' ' ')|" \
         "PNG $(stat -c %s "$stem.png" 2>/dev/null || echo 0) bytes"
    [ -s "$stem.png" ] || { echo "  NO IMAGE: $(head -c 300 "$stem.json")"; return 1; }
}

# PROFILE=1 traces only the last rep: --start-paused keeps the earlier ones (and the weight
# loading, which is most of the process's life) out of the timeline.
RC=0
for i in $(seq 1 "$REPS"); do
    if [ "$PROFILE" = 1 ] && [ "$i" = "$REPS" ]; then
        $UNITRACE --resume "$SESSION" || { echo "resume failed"; exit 1; }
        request "traced$i" || RC=1
        $UNITRACE --stop "$SESSION"
    else
        request "rep$i" || RC=1
    fi
done

# Teardown here, not at exit: each traced process writes its timeline when it exits, so nothing
# can be looked at while the server is up (waiting after --stop watches 0-byte files -- 300 s of
# zeros, measured). It also frees the cards for the other arm.
echo "== tearing down =="
teardown; trap - EXIT

# A raw result dir is not viewable: one directory per traced process (95 of them in a measured
# run, because the DiT JIT-compiles and every g++/as/cc1plus child is followed too), of which 2
# have content; a worker killed with the group leaves unterminated JSON that Perfetto rejects
# outright; and most records are flow arrows, which is most of the bytes. make_perfetto.py, next
# to this script, reduces the dir to one .json.gz per stage. Run it here rather than leaving it
# as a suggestion, because it is cheap and because the raw dirs are what get copied off the host
# otherwise -- 506 MB where 17.7 MB says the same thing.
if [ "$PROFILE" = 1 ]; then
    echo "== making the timelines viewable =="
    python3 "$HERE/make_perfetto.py" "$RESULT_DIR" \
        || echo "  conversion failed; the raw $RESULT_DIR/*/chrome_trace.json are still there"
fi

echo
echo "=== what this run left behind ==="
echo "  $LOG"
echo "  $OUT_DIR/$TAG.rep*.{json,png}" $([ "$PROFILE" = 1 ] && echo "$OUT_DIR/$TAG.traced*.{json,png}")
[ "$PROFILE" = 1 ] && echo "  $RESULT_DIR/perfetto_<stage>_${TAG}_<python.PID>.json.gz  (drag into Perfetto)" \
                   && echo "  $RESULT_DIR/<python.PID>/chrome_trace.json  (the raw timelines it came from)"
cat <<TXT

=== reading it ===
Wall clock, per request (quote PROFILE=0 only, and only reps 2+):
  grep -ho '"stage_[01]_gen_ms"[[:space:]]*:[[:space:]]*[0-9.]*' $OUT_DIR/$TAG.rep*.json
  # then the same for the other arm's directory of rows, i.e. run this script with GRAPH=$((1 - GRAPH))

Was the run guidance-free, i.e. one denoise batch per step rather than two (one line per request;
0 means GUIDANCE > 1 and every per-step number is ~1.8x what the A/B should quote):
  grep -ac 'runs without classifier-free guidance' $LOG

Did the graph engage, and how often did it pay for itself:
  grep -ac 'captured denoise graph' $LOG        # captures: 1 per shape key, ~683 ms of recording each
  grep -ac 'dropping .* captured denoise' $LOG  # invalidations
  grep -ac 'diffusion_use_graph ignored' $LOG   # refused (kv-cache skip steps or a cache backend)
  grep -ac 'disabling denoise graphs' $LOG      # permanently disabled after a capture failure
A capture needs ~9-16 replayed steps to break even and a request supplies $((STEPS - 1)), so a
capture count equal to the request count means the feature cannot have paid.

The images: a stale captured buffer returns a VALID PNG of the right byte size that is washed out
or entirely black, and nothing in the log or the metrics notices. Healthy is std 60-80:
  python3 -c "import numpy,PIL.Image,sys;a=numpy.asarray(PIL.Image.open(sys.argv[1]));print(a.mean(),a.std())" $OUT_DIR/$TAG.rep2.png
TXT
[ "$PROFILE" = 1 ] && cat <<TXT
The perfetto_*.json.gz written above are what to drag into https://ui.perfetto.dev (copy them off
the host first). To redo that by hand, or for an older result dir:
  python3 $HERE/make_perfetto.py $RESULT_DIR [--only dit] [--keep-flows]
Measure on the RAW files, not on those: the reduced copy is for eyeballing, and Perfetto's own
totals do not distinguish the two tracks below.
The DiT worker is the process whose kernels include conv/gen_conv (the AR has none). DEVICE<...>
is kernel execution, HOST<...> the Level Zero calls that submitted it -- never union the two, or a
launched step counts 6083 "kernels" instead of 1982 and its device busy comes out above the truth.
A replay is where DEVICE is continuous and HOST nearly empty (19 calls for a whole step against
4101); a launched step has one HOST call per kernel with visible gaps between them.
TXT
exit "$RC"
