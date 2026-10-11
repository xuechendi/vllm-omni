#!/bin/bash
# MiniMax-H3 joint video+audio on Intel XPU (CRI D93): t2va, fl2va or ref2va, on
# 1 card with distributed layerwise offload or on 2/4/8 cards with the SYCL-TLA
# ring. Self-contained: it starts its own server, sends one /v1/videos/sync
# request, checks the returned MP4 really has a video AND an audio stream, tears
# the server down, then prints the per-switch self-evidence lines.
#
#   bash minimax_h3_va_xpu.sh                                # t2va, 1 card, DLO resident 50
#   CARDS=4 bash minimax_h3_va_xpu.sh                        # 4-card sycl_tla ring
#   TASK=fl2va REF_IMAGE=/workspace/h3_in/first.png bash minimax_h3_va_xpu.sh
#   TASK=ref2va CARDS=4 REF_IMAGE=/workspace/h3_in/cat.png bash minimax_h3_va_xpu.sh
#   STEPS=4 WARM_STEPS=0 REPS=1 bash minimax_h3_va_xpu.sh    # functional smoke, NOT a perf number
#   KEEP_SERVER=1 bash minimax_h3_va_xpu.sh                  # leave the server up for more curls
#   CACHE=1 bash minimax_h3_va_xpu.sh                        # cache_dit denoise loop
#   CACHE=1 QUANT=mxfp8 ATTN=SAGE_ATTN_3 bash minimax_h3_va_xpu.sh          # 1 card
#   CACHE=1 QUANT=mxfp8 KV_DTYPE=fp8_sycl_tla CARDS=4 bash minimax_h3_va_xpu.sh
#
# The defaults ARE the reference measurement profile, so a bare invocation is
# directly comparable to the recorded numbers below: 1248x768, 141 frames,
# seed 42, 50 steps, prompts/canon_t2va.txt, one 4-step warm request then two
# timed ones. Nothing about that profile is implicit -- every part of it is a
# variable below -- but it is the default precisely so that reproducing a cell
# does not depend on remembering a shell line. Change any of WIDTH / HEIGHT /
# FRAMES / PROMPT_FILE and the result is no longer comparable: all four feed the
# DiT sequence length, and a 1.59x difference in frame count alone moves the
# per-step cost by 1.6-2.5x depending on how attention-bound the shape is.
#
# Recorded t2va cells at that profile, all on hq-cri-d93-a, cards 0-3 for the
# 4-card rows, reported as the server's own x-inference-time-s:
#
#   cards  stack                                    timed          peak MB
#   1      h3port/cri:b98ebcc2-mkstack              889.96         76846
#   4      h3port/cri:b98ebcc2-mkstack              212.07/212.12  92934
#   4      vllm-cri-omni (tree 6ed0e9d3, 2026-09-21) 222.61/222.49 92934
#
# The 4.8% between the last two rows is a stack difference, not run-to-run
# spread: within each row the two reps agree to 0.06-0.5%, and peak memory is
# bit-identical, so both rows provably did the same work. The stack deltas are
# torch 2.15.0a0+git42a5034 vs 2.14.0a0+gitd35bb04, vllm-xpu-kernels
# dev499+g2c0e2b80e vs dev522+g2ed1db606, triton-xpu 3.7.2 vs triton 3.8.0, and a
# patched tree vs 6ed0e9d3. Driver and kernel were identical, so those are out.
#
# Recorded QUANTIZED cells, 2026-09-28, vllm-omni 0.30.0_rebase (07ba68abe) on
# 10.239.11.197 cards 0-3. These are a DIFFERENT profile from the table above --
# they all carry CACHE=1 -- so they are comparable to each other and NOT to the
# rows above. Reading 212.07 -> 56.39 as an mxfp8 win would be wrong: nearly all
# of that gap is cache_dit skipping denoise steps. The mxfp8 claim is the 69.97
# -> 56.39 row pair, which holds CACHE=1 and the ring fixed and varies only the
# linear method.
#
#   cards  linear  kv            cache  warm    timed          peak MB
#   4      bf16    fp8_sycl_tla  1      92.21   69.97/69.68    90076
#   4      mxfp8   fp8_sycl_tla  1      41.71   56.39/55.60    53522
#
# So at 50 steps mxfp8 is 1.24x faster than bf16 for 41% less peak memory
# (90076 -> 53522 MB), reps agreeing to 1.4%, both arms emitting MP4s with a
# video AND an audio stream. Run them as:
#
#   CACHE=1 KV_DTYPE=fp8_sycl_tla CARDS=4 bash minimax_h3_va_xpu.sh              # bf16 row
#   CACHE=1 QUANT=mxfp8 KV_DTYPE=fp8_sycl_tla CARDS=4 bash minimax_h3_va_xpu.sh  # mxfp8 row
#
# If the mxfp8 row ever dies with `RuntimeError: scale_q must be finite and
# positive` (raised natively from sycl_tla_fmha.so), that is NOT a missing-scale
# bug in vllm-omni: with bf16 q/k/v the wheel is required to receive no scales
# and derives them itself as amax(|t|).clamp_min(tiny)/448. The clamp makes a
# zero scale impossible, so it means either a NaN/Inf reached it (blame upstream
# numerics) or the tensor was all zeros and the scale underflowed (blame weight
# loading). ring_kernels.py re-raises with per-tensor nan/inf/zero/amax counts to
# say which of the two it is; quote that line rather than bisecting.
#
# Four settings that belong to the container, not to this script, and that were
# part of every number above -- a run in a container without them is not
# comparable, so the preflight below warns rather than lets it pass silently:
# --shm-size 32g, --ipc=host, --ulimit memlock=-1, --ulimit stack=67108864, plus
# --privileged and the /dev/dri devices. Run this under a login shell
# (bash -lc ...) so the venv and the oneAPI environment are active; a plain
# `bash -c` leaves vllm_omni unimportable.
#
# The two launch shapes are the ones validated on 8x CRI D93 and written down in
# docs/user_guide/diffusion/models/minimax_h3_xpu_perf.md:
#
#   CARDS=1   DLO with --dlo-resident-layers 50, i.e. all 50 DiT blocks resident
#             (DLO still owns the weights, which is why H3_GEMM_RELAYOUT declines
#             them -- see the relayout note below), ring degree 1, attention =
#             vllm-xpu-kernels bf16 FA. DLO is skipped automatically when QUANT is
#             set, since quantized weights do not need it and DLO owning them
#             changes a second variable; see the DLO= note at that branch.
#   CARDS=N   TP1 / Ulysses 1 / ring = N, text encoder tp = N, VAE patch
#             parallel = N with tiled decode, no DLO.
#
# Task -> checkpoint partition, which is why TASK picks the model path as well as
# extra_params.task: t2va and fl2va are both served by the FL2VA DiT, ref2va by
# the Ref2VA one (recipes/MiniMaxAI/MiniMax-H3.md). --task-type downloads/loads
# only the selected partition (~134 GiB of bf16 safetensors each), so a t2va and a
# ref2va run are two different servers, not two requests to one server. Do not run
# both at once unless the host has RAM for two copies.
#
# Do NOT pass --enforce-eager (it is in the CUDA consumer-GPU recipes, not this
# one). The default is --cache-backend none, the uncached reference path, which is
# what makes the numbers comparable to the 50-step cells above; CACHE=1 switches
# to cache_dit and is a different measurement, not a faster version of the same
# one -- it skips block recomputation, so it trades quality for speed and has to
# be validated on output, not only timed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}

TASK=${TASK:-t2va}
CARDS=${CARDS:-1}
PORT=${PORT:-8098}
STEPS=${STEPS:-50}
# 42 is the seed every recorded cell above used. The recipe's own examples use
# 1101 (t2va) / 2101 (fl2va) / 3100 (ref2va); seed changes the content, not the
# cost, so it is pinned here for content comparability only. Note that the output
# is not bit-reproducible even at a fixed seed -- two reps of the same request
# produced different MP4 checksums -- so compare times and stage timings, never
# output hashes.
SEED=${SEED:-42}
FPS=${FPS:-24}
FLOW_SHIFT=${FLOW_SHIFT:-12}
AUDIO_FLOW_SHIFT=${AUDIO_FLOW_SHIFT:-3.0}
DURATION=${DURATION:-}
# Clip length can be given two ways and they are NOT interchangeable in a
# comparison: DURATION goes in extra_params and the pipeline derives the frame
# count from it and fps, while FRAMES sets num_frames directly. Set FRAMES to
# pin the exact frame count (and then duration is not sent at all) -- that is
# what a cross-host or cross-stack A/B needs, because frame count drives the DiT
# sequence length and therefore the per-step cost quadratically in attention.
# Default 141 (= 5.875 s at 24 fps), the frame count of every recorded cell.
FRAMES=${FRAMES:-141}
# The prompt is part of the measured shape: it sets the text-token count that
# every step cross-attends to. PROMPT_FILE sends a prompt file verbatim
# (curl -F name=<file), which is the only way to reproduce a long structured
# reference prompt without shell quoting mangling it -- and the reason the
# reference prompt is checked in next to this script rather than referred to by
# an absolute path in somebody's home directory. prompts/canon_t2va.txt is the
# 2327-byte structured prompt (md5 10785de138103c44034b0e08b7f5f116, 508 Qwen
# tokens) used for the recorded t2va cells; it is only the default for t2va,
# since the other two tasks carry their own prompt with their reference input.
PROMPT_FILE=${PROMPT_FILE:-}
if [ -z "$PROMPT_FILE" ] && [ "${TASK:-t2va}" = t2va ] && [ -s "$HERE/prompts/canon_t2va.txt" ]; then
    PROMPT_FILE=$HERE/prompts/canon_t2va.txt
fi
# Warm-then-timed, the discipline a steady-state number requires. A first
# request on a fresh server pays torch.compile of the VAE decoder blocks, the
# recompiles, and allocator growth; including that in a "50-step time" inflates
# it and the inflation does not scale with card count, so it corrupts scaling
# ratios too. WARM_STEPS>0 sends one throwaway request at that step count first
# (4 is enough to trigger every compile and costs a fraction of a 50-step run),
# then REPS timed requests at STEPS. REPS>1 is how run-to-run spread becomes
# visible instead of assumed.
# Defaults 2 and 4: two timed reps is the minimum that makes spread visible, and
# 4 warm steps triggered every compile in the recorded runs (the warm request cost
# 76-79 s against a 212-222 s timed one, so it is cheap insurance).
REPS=${REPS:-2}
WARM_STEPS=${WARM_STEPS:-4}
# CACHE=1 switches the denoise loop from the uncached reference path to the
# cache_dit backend, which skips recomputing transformer blocks on steps whose
# residual moves less than residual_diff_threshold. It is a speed/quality trade,
# not a free win, so it is off by default and its number is NOT comparable to the
# recorded cells -- the reference line at the bottom is suppressed when it is on.
# The config below is the one the reference harness uses: 1 front block always
# computed, no back blocks, the first 4 steps never cached (they set the residual
# baseline), at most 4 consecutive cached steps, TaylorSeer extrapolation off.
CACHE=${CACHE:-0}
case "$CACHE" in 0|1) ;; *) echo "[h3] CACHE must be 0 or 1, got '$CACHE'"; exit 2 ;; esac
CACHE_CFG=${CACHE_CFG:-'{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.24,"max_continuous_cached_steps":4,"enable_taylorseer":false,"taylorseer_order":1,"scm_steps_mask_policy":null,"scm_steps_policy":"dynamic"}'}
# The JSON is passed as one argv element, so a single quote inside it would break
# the quoting rather than be rejected by the JSON parser -- refuse it up front.
case "$CACHE_CFG" in *"'"*) echo "[h3] CACHE_CFG must not contain a single quote"; exit 2 ;; esac

# ---- precision knobs --------------------------------------------------------
# Three INDEPENDENT axes, kept as three variables rather than one "fp8 preset",
# because they quantize different tensors, are served by different kernels, and
# can each be wrong on its own. All are empty by default: every recorded cell
# above is bf16 weights, bf16 activations and a bf16 ring.
#
#   QUANT     W8A8 linear quantization of the DiT, via
#             --diffusion-quantization-config. A bare name expands to
#             '{"method":"<name>"}', or pass full JSON. On XPU only the ONLINE
#             (dynamic) path exists: mxfp8_config.py raises NotImplementedError
#             for a pre-quantized checkpoint, so weights are cast at load from the
#             bf16 safetensors and activation scales come from a per-call amax.
#             That means QUANT=mxfp8 needs no second checkpoint -- and also that
#             it cannot be compared against an offline-quantized number from a
#             CUDA or NPU run. It binds the DiT's vLLM LinearBase layers
#             (Column/Row/QKV/MergedColumn); the 6 plain nn.Linear stay bf16.
#   KV_DTYPE  attention K/V transport, via --diffusion-kv-cache-dtype:
#             fp8 | fp8_sycl_tla | mxfp8_sycl_tla. On the N-card ring this is the
#             only thing that makes the circulated K/V blocks fp8 --
#             ring_attn_utils.py turns fp8_sycl_tla into precision="fp8_pertensor"
#             for the SYCL/TLA block kernel, and leaves it bf16 otherwise.
#             VLLM_OMNI_RING_ATTN_PRECISION does NOT do this; it is inert here.
#   ATTN      --diffusion-attention-backend, e.g. SAGE_ATTN_3 for the deepklox
#             SageAttention V3 Hybrid kernel (MXFP4 E2M1 QK^T + MXFP8 E4M3 P/V).
#             1-card axis: with CARDS>1 attention goes through the ring, whose
#             block kernel is picked by VLLM_OMNI_RING_ATTN_TYPE
#             (sycl_tla | klox_sage_v3_hybrid), not by this flag. The sage3
#             backend also has a flash fallback for calls the kernel declines
#             (GQA, non-128 head size, contract rejections), so "SAGE_ATTN_3 was
#             selected" is not "every call ran the sage kernel" -- the call counts
#             printed at server exit are what distinguishes them.
QUANT=${QUANT:-}
KV_DTYPE=${KV_DTYPE:-}
ATTN=${ATTN:-}
QUANT_CFG=
if [ -n "$QUANT" ]; then
    case "$QUANT" in
        "{"*) QUANT_CFG=$QUANT ;;
        *) QUANT_CFG="{\"method\":\"$QUANT\"}" ;;
    esac
    case "$QUANT_CFG" in *"'"*) echo "[h3] QUANT must not contain a single quote"; exit 2 ;; esac
fi
# Reject an unknown K/V label here rather than at the first attention call, which
# is minutes of weight load away.
case "$KV_DTYPE" in
    ''|fp8|fp8_sycl_tla|mxfp8_sycl_tla) ;;
    *) echo "[h3] KV_DTYPE must be empty, fp8, fp8_sycl_tla or mxfp8_sycl_tla, got '$KV_DTYPE'"; exit 2 ;;
esac
# Where the checkpoint is depends on which side of the container mount you are on,
# so try the known roots in order and let MODEL_ROOT= override. The download lives
# at /mnt/disk2/models/MiniMax-H3 on this host; /mnt/disk2/models is NOT mounted
# into the vllm-omni-chendi container (only /mnt/disk2/user_home/chendi, as
# /workspace), so inside that container the checkpoint has to be reachable under
# /workspace or bind-mounted at /models -- otherwise this script cannot see it no
# matter what MODEL_ROOT says.
#
# On hq-cri-d93-a the checkpoint is at /mnt/disk1/heyuan/hf_models/MiniMax-H3 and
# /mnt/disk1/hf_models/MiniMax-H3 is a SYMLINK to it. That symlink is a trap
# inside a container: if the bind mount covers /mnt/disk1/hf_models but not
# /mnt/disk1/heyuan, the entry lists in `ls` and every access to it returns
# ENOENT, because the link target is not in the mount namespace. Hence the -d
# test below (which follows the link and so rejects a dangling one) and hence the
# real target being on the candidate list ahead of the symlink.
MODEL_ROOT=${MODEL_ROOT:-}
MODEL_CANDIDATES=(/models/MiniMax-H3 /workspace/models/MiniMax-H3 /mnt/disk2/models/MiniMax-H3
                  /mnt/disk1/heyuan/hf_models/MiniMax-H3 /mnt/disk1/hf_models/MiniMax-H3)
if [ -z "$MODEL_ROOT" ]; then
    for _root in "${MODEL_CANDIDATES[@]}"; do
        [ -d "$_root" ] && { MODEL_ROOT=$_root; break; }
    done
    # Nothing matched: say that, rather than reporting the last candidate as if it
    # had been chosen -- naming a path from a different host is how this turns into
    # "the checkpoint is corrupt" instead of "the mount is missing".
    if [ -z "$MODEL_ROOT" ]; then
        echo "[h3] no MiniMax-H3 checkpoint found. Tried, in order:"
        for _root in "${MODEL_CANDIDATES[@]}"; do
            if [ -L "$_root" ] && [ ! -d "$_root" ]; then
                echo "[h3]   $_root -> $(readlink "$_root") (symlink whose target is not in this mount namespace)"
            else
                echo "[h3]   $_root ($( [ -e "$_root" ] && echo "exists but is not a directory" || echo absent ))"
            fi
        done
        echo "[h3] Set MODEL_ROOT=/path/to/MiniMax-H3, or bind-mount the checkpoint into the"
        echo "[h3] container (the reference runs used -v <host path>:/models:ro, giving"
        echo "[h3] /models/MiniMax-H3). A mount cannot be added to a running container."
        exit 2
    fi
fi

# ---- task -> partition, DiT task id, request shape ---------------------------
# Output contract (recipe "Official input matrix and limits"): 4-15 s at 24 fps,
# stereo 32 kHz audio, 32-pixel canvas multiple. t2va needs one NAMED ratio
# (21:9, 16:9, 4:3, 1:1, 3:4, 9:16); fl2va always follows the first input image's
# ratio and ignores a generic aspect_ratio, so this script does not send width or
# height for it; ref2va defaults to 16:9 with `adaptive` as the accepted alias.
case "$TASK" in
    t2va)
        PART=FL2VA; TASK_TYPE=fl2va
        PROMPT=${PROMPT:-"In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts."}
        DURATION=${DURATION:-8.7}
        # 1248x768 is the recorded shape: 78x48 latent patches x 36 latent frames
        # = 134,784 DiT tokens. 1344x768 is also a valid 16:9 canvas and an easy
        # thing to reach for, but it is 1.08x the tokens and not comparable.
        # Bound as variables, not inlined, so the echo below and the reference-cell
        # gate at the bottom both report the shape that was actually sent.
        WIDTH=${WIDTH:-1248}; HEIGHT=${HEIGHT:-768}
        SHAPE_FIELDS=(-F "width=$WIDTH" -F "height=$HEIGHT" -F "aspect_ratio=${ASPECT:-16:9}")
        ;;
    fl2va)
        PART=FL2VA; TASK_TYPE=fl2va
        PROMPT=${PROMPT:-"A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience."}
        DURATION=${DURATION:-8.7}
        # No width/height: fl2va follows the first input image's ratio. (The
        # reference harness does send them for fl2va too; they are dropped here
        # because the recipe says the image wins, so sending them would only
        # obscure which one the pipeline honoured.)
        SHAPE_FIELDS=()
        ;;
    ref2va)
        PART=Ref2VA; TASK_TYPE=ref2va
        PROMPT=${PROMPT:-"A white cat sits on a beige couch and slowly looks toward the camera."}
        DURATION=${DURATION:-8.0}
        # short_edge controls the 768-pixel canvas and must be 768.
        SHAPE_FIELDS=(-F "aspect_ratio=${ASPECT:-adaptive}" -F "short_edge=768")
        ;;
    *) echo "[h3] TASK must be t2va, fl2va or ref2va, got '$TASK'"; exit 2 ;;
esac
MODEL=${MODEL:-$MODEL_ROOT/$PART}

case "$CARDS" in
    1) MASK=${MASK:-0} ;;
    2) MASK=${MASK:-0,1} ;;
    4) MASK=${MASK:-0,1,2,3} ;;
    8) MASK=${MASK:-0,1,2,3,4,5,6,7} ;;
    *) echo "[h3] CARDS must be 1, 2, 4 or 8, got '$CARDS'"; exit 2 ;;
esac

# fl2va and ref2va need a real input image, and the checkpoint partitions do not
# ship one -- the accuracy test pulls its inputs from the model repo's assets/
# (tests/e2e/accuracy/minimax_h3/test_minimax_h3_i2va_ref2va_similarity.py).
# Check BEFORE starting the server: a missing file found after the fact costs a
# full ~134 GiB weight load. Reference images accept JPG/JPEG/PNG/WEBP/HEIC/HEIF
# up to 30 MiB; pass a video with REF_IMAGE=... and REF_MIME=video/mp4 for the
# ref2va video forms.
REF_IMAGE=${REF_IMAGE:-}
REF_MIME=${REF_MIME:-image/png}
if [ "$TASK" != t2va ]; then
    if [ -z "$REF_IMAGE" ] || [ ! -f "$REF_IMAGE" ]; then
        echo "[h3] TASK=$TASK needs a reference input: REF_IMAGE=/path/to/first_frame.png (got '${REF_IMAGE:-unset}')"
        echo "[h3] MiniMax-H3's own assets are in the HF repo under assets/ (i2va.mp4, r2va.mp4)."
        exit 2
    fi
fi
if [ ! -d "$MODEL" ]; then
    echo "[h3] checkpoint partition not found: $MODEL (set MODEL_ROOT= or MODEL=)"
    exit 2
fi
# Each partition is ~134 GiB of bf16 safetensors, so a snapshot download is long
# enough that starting a server against a half-finished one is easy to do by
# accident -- and it fails minutes later, inside weight load, with an error that
# points at the loader rather than at the missing bytes. hf_hub writes partial
# blobs as *.incomplete under .cache/huggingface/download and only renames them
# on success, so their presence is a positive "still downloading" signal.
# The pipeline needs <part>/** plus, for Ref2VA, transformer/**.
if [ "${SKIP_CHECKPOINT_CHECK:-0}" != 1 ]; then
    INCOMPLETE=$(find "$MODEL_ROOT/.cache/huggingface/download/$PART" "$MODEL/.cache" \
                      -name '*.incomplete' 2>/dev/null | wc -l)
    if [ "$INCOMPLETE" -gt 0 ]; then
        echo "[h3] $PART is still downloading: $INCOMPLETE *.incomplete blob(s) under $MODEL_ROOT"
        echo "[h3] wait for the snapshot to finish, or SKIP_CHECKPOINT_CHECK=1 to try anyway"
        exit 2
    fi
    if [ ! -d "$MODEL/transformer" ] || [ ! -f "$MODEL/model_index.json" ]; then
        echo "[h3] $PART looks incomplete: need $MODEL/model_index.json and $MODEL/transformer/"
        echo "[h3] SKIP_CHECKPOINT_CHECK=1 to try anyway"
        exit 2
    fi
fi

OUT_DIR=${OUT_DIR:-/workspace/minimax_h3_logs}
mkdir -p "$OUT_DIR" 2>/dev/null || OUT_DIR=$HERE/logs
mkdir -p "$OUT_DIR"
# Every knob that changes the arithmetic goes in the tag, so a quantized run
# cannot overwrite the bf16 one it is being compared against -- the outputs are
# MP4s in a shared directory and the whole point of these runs is the A/B.
TAG=${TAG:-${TASK}_${CARDS}c_${STEPS}step$( [ "$CACHE" = 1 ] && echo _cd1 )$( [ -n "$QUANT" ] && echo "_q$QUANT" )$( [ -n "$KV_DTYPE" ] && echo "_kv$KV_DTYPE" )$( [ -n "$ATTN" ] && echo "_$(echo "$ATTN" | tr 'A-Z_' 'a-z-')" )}
LOG=$OUT_DIR/$TAG.log; OUT=$OUT_DIR/$TAG.mp4

# ---- environment ------------------------------------------------------------
# Auto large-GRF mode: the H3 DiT kernels are register-hungry, and letting IGC
# pick the large-GRF variant per kernel is what the validated CRI configuration
# ran with. It is an IGC (compiler) switch, so it has to be in the environment
# before the first kernel is JIT'd -- setting it after the server is up does
# nothing.
export IGC_ExtraOCLOptions=${IGC_ExtraOCLOptions:--cl-intel-enable-auto-large-GRF-mode}
# PCI-ID device order makes the mask stable across reboots and driver reloads, so
# MASK=0,1 means the same two cards it meant last run. Without it level-zero's
# enumeration order is not pinned and a mask silently selects different cards.
export ZE_AFFINITY_MASK=$MASK ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
# oneCCL's SYCL kernels for the small collectives. On this class of node the
# host-staged fallback path is the slow one for the text-encoder-TP and
# VAE-parallel traffic.
export CCL_ENABLE_SYCL_KERNELS=1
# CCL_TUNE=1 (opt-in, off by default so the run stays faithful to the reference
# configuration): raise the two SYCL-path size thresholds. Their defaults are
# 4 MiB for allreduce and 1 MiB for allgatherv, so out of the box every
# collective above that drops off the SYCL path -- and on this host class that
# slow path measured 1.7x (16 MB) to 2.4x (256 MB) worse busBW on a 2-rank
# same-switch allreduce. Only the text-encoder TP and VAE-patch-parallel traffic
# goes through CCL here (the DiT ring uses its own IPC pull schedule), and the
# 1.7-2.4x is a microbenchmark number, not an end-to-end one, so treat any
# resulting delta as something to measure rather than assume.
if [ "${CCL_TUNE:-0}" = 1 ]; then
    export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD:-1073741824}
    export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD:-1073741824}
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# The sync endpoint's own deadline (api_server.py: VLLM_OMNI_VIDEO_SYNC_TIMEOUT,
# default 600 s). A 50-step 768p generation on one card is far past 600 s, so
# without this the server closes the request while the DiT is still denoising.
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}
# Opt-in (default 0): repack transformer Linear weights transpose-contiguous at
# load. It declines weights DLO owns and weights owned by a quantized
# quant_method, so on CARDS=1 (DLO) expect the "skipped" accounting rather than a
# large relayout count -- that is the switch working as designed, not a failure.
export H3_GEMM_RELAYOUT=${H3_GEMM_RELAYOUT:-1}
# The two pass-throughs from the reference configuration. Neither is read by
# anything in this tree (grepped) nor by the installed packages, so they are
# inert here and only matter if the deployed stack carries the out-of-tree code
# that consumes them; they are kept so this script reproduces that environment
# exactly. H3_NUM_LAYERS=50 is the DiT's full block count, matching
# --dlo-resident-layers 50 below.
export H3_NUM_LAYERS=${H3_NUM_LAYERS:-50}
export SPT_NOENV=${SPT_NOENV:-1}
# Recompiles only -- the noisy TORCH_LOGS settings would bury the self-evidence
# greps at the bottom. H3_VAE_COMPILE wraps 36 VAE decoder blocks in
# torch.compile, so an unexpected recompile is a real per-request cost and this
# is how it becomes visible.
export TORCH_LOGS=${TORCH_LOGS:-recompiles}
export PYTHONPATH=${PYTHONPATH:-$REPO}
export NO_PROXY=${NO_PROXY:-0.0.0.0,127.0.0.1,localhost}

SERVE_ARGS=(
    --omni --host 0.0.0.0 --port "$PORT" --trust-remote-code
    --task-type "$TASK_TYPE" --dtype bfloat16
    --num-gpus "$CARDS" --tensor-parallel-size 1 --usp 1
    --vae-parallel-mode tile --vae-use-tiling
    --init-timeout 1800 --stage-init-timeout 1500
)
if [ "$CACHE" = 1 ]; then
    SERVE_ARGS+=(--cache-backend cache_dit --cache-config "$CACHE_CFG")
else
    SERVE_ARGS+=(--cache-backend none)
fi
[ -n "$QUANT_CFG" ] && SERVE_ARGS+=(--diffusion-quantization-config "$QUANT_CFG")
[ -n "$KV_DTYPE" ] && SERVE_ARGS+=(--diffusion-kv-cache-dtype "$KV_DTYPE")
if [ -n "$ATTN" ]; then
    SERVE_ARGS+=(--diffusion-attention-backend "$ATTN")
    # The sage3 backend counts, per kind, which calls the V3 kernel served and
    # which its flash/SDPA fallback did -- and only when this is on. Without it
    # the run cannot tell "sage ran everywhere" from "sage declined every call
    # and flash quietly produced the number", which is the one thing a
    # sage-vs-bf16 comparison has to establish. The counters are a dict increment
    # per call, and the report is printed at server exit, so it lands at the tail
    # of $LOG after the teardown below -- not in the evidence section.
    case "$ATTN" in SAGE_ATTN_3|SAGE_ATTN) export SAGE_ATTN_REPORT_FALLBACKS=${SAGE_ATTN_REPORT_FALLBACKS:-1} ;; esac
fi
# Per-stage timings. The curl wall time conflates denoise with VAE decode, and for
# H3 those two scale differently across cards (the ring speeds up the DiT; the
# decode is tiled patch-parallel), so a 1-card vs N-card comparison built on wall
# time alone attributes the difference to the wrong stage. PROFILER=0 to drop it.
[ "${PROFILER:-1}" = 1 ] && SERVE_ARGS+=(--enable-diffusion-pipeline-profiler)
if [ "$CARDS" = 1 ]; then
    SERVE_ARGS+=(--text-encoder-tp-size 1 --vae-patch-parallel-size 1)
    # DLO is the bf16 1-card shape and is OFF whenever the DiT is quantized.
    #
    # With --dlo-resident-layers 50 == num_layers, DLO streams nothing per step;
    # its job on 1 card is purely to get 50 bf16 blocks onto the card at all.
    # Quantized weights are about half the bytes, so that job is gone -- and DLO
    # owning the weights actively gets in the way: it keeps its own host-side
    # copies and moves the module between host and device around load, which is
    # what produces the "Stream-offloaded 0 online-quantized layers to CPU" and
    # "Quantization complete, offloaded model back to CPU" pair, and it is also
    # why H3_GEMM_RELAYOUT declines DLO-owned weights. Leaving it on would mean
    # the quantized row differs from the bf16 one in two ways at once.
    #
    # Consequence to state whenever the 1-card rows are compared: a QUANT row and
    # the 889.96 s bf16 cell do NOT differ only in arithmetic -- one has DLO and
    # the other does not. DLO=1 forces it back on, DLO=0 off.
    DLO=${DLO:-auto}
    case "$DLO" in
        auto) _dlo=$( [ -n "$QUANT" ] && echo 0 || echo 1 ) ;;
        0|1) _dlo=$DLO ;;
        *) echo "[h3] DLO must be auto, 0 or 1, got '$DLO'"; exit 2 ;;
    esac
    if [ "$_dlo" = 1 ]; then
        # Fully resident: DLO keeps the rank-local shard in pinned host memory and
        # with resident=50 every block is copied to the card once per denoise stage
        # instead of streamed through the two-buffer window. --dlo-no-use-allgather
        # keeps each worker on its own standard-loader shard (no DP AllGather to
        # reconstruct full blocks), which is what makes a 1-card run meaningful.
        SERVE_ARGS+=(--enable-distributed-layerwise-offload --dlo-no-use-allgather
                     --dlo-resident-layers 50)
    else
        echo "[h3] DLO off on 1 card ($( [ -n "$QUANT" ] && echo "QUANT=$QUANT" || echo "DLO=$DLO" )): weights stay where the loader puts them, no layerwise offload"
    fi
else
    # Ring degree N == world size N. sycl_tla is also what the in-tree autoselect
    # picks on XPU when the extension is built, but pinning it means a build
    # without the extension fails loudly here instead of quietly running the
    # aten-SDPA ring and producing a number that is not the one being reported.
    #
    # VLLM_OMNI_RING_ATTN_PRECISION is inert on this path: the XPU resolver takes
    # the SYCL/TLA precision from kv_cache_dtype, not from this variable (bf16 is
    # the default when unset). fp8 K/V transport would be
    # --diffusion-kv-cache-dtype fp8_sycl_tla. Set here only to match the
    # reference environment, and because the [RING_RESOLVED] line below names the
    # precision that actually resolved -- read that, not this.
    export VLLM_OMNI_RING_ATTN_TYPE=${VLLM_OMNI_RING_ATTN_TYPE:-sycl_tla}
    export VLLM_OMNI_RING_ATTN_PRECISION=${VLLM_OMNI_RING_ATTN_PRECISION:-bf16}
    # Do not pin pad_seq_len far above the packed valid length: the SYCL ring
    # trims each circulated K/V block to the valid prefix, and a padding that
    # leaves any rank a zero-length valid shard raises at request time
    # ("SYCL/TLA ring attention does not support an empty local K/V shard").
    SERVE_ARGS+=(--ring "$CARDS" --text-encoder-tp-size "$CARDS"
                 --vae-patch-parallel-size "$CARDS")
fi

echo "[h3] task=$TASK partition=$PART model=$MODEL"
echo "[h3] cards=$CARDS mask=$MASK steps=$STEPS seed=$SEED reps=$REPS warm_steps=$WARM_STEPS cache=$( [ "$CACHE" = 1 ] && echo "cache_dit" || echo none )"
[ "$CACHE" = 1 ] && echo "[h3] cache_config=$CACHE_CFG"
echo "[h3] precision: linear=${QUANT_CFG:-bf16} kv=${KV_DTYPE:-bf16} attn_backend=${ATTN:-platform-default}$( [ "$CARDS" != 1 ] && echo " ring=${VLLM_OMNI_RING_ATTN_TYPE:-sycl_tla}" )"
# --diffusion-attention-backend does not reach the ring block kernel, so on N
# cards it would look set and change nothing about the DiT self-attention. Say so
# instead of letting the tag claim a kernel that never ran.
[ -n "$ATTN" ] && [ "$CARDS" != 1 ] && \
    echo "[h3] WARNING: CARDS=$CARDS -- DiT self-attention goes through the ring, whose kernel is VLLM_OMNI_RING_ATTN_TYPE, not ATTN=$ATTN"
echo "[h3] shape: ${WIDTH:-default}x${HEIGHT:-default} $( [ -n "$FRAMES" ] && echo "frames=$FRAMES" || echo "duration=${DURATION}s" ) fps=$FPS prompt=${PROMPT_FILE:-inline}"
echo "[h3] log=$LOG out=$OUT"

# Preflight on the four container-level settings this script cannot set itself
# (see the header). These are warnings, not errors: the run will usually still
# complete, but its number is not comparable to the recorded cells, and that is
# far easier to see here than to reconstruct from a time that came out wrong.
# /dev/shm is the one that actually bites -- the workers pass tensors through it,
# and the 64 MB Docker default turns into a mid-run failure rather than a slowdown.
_shm_mb=$(df -m --output=size /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
[ -n "$_shm_mb" ] && [ "$_shm_mb" -lt 30000 ] && \
    echo "[h3] WARNING: /dev/shm is ${_shm_mb} MB, the reference runs had 32 GB (docker --shm-size 32g)"
_stk=$(ulimit -s); [ "$_stk" != unlimited ] && [ "${_stk:-0}" -lt 65536 ] && \
    echo "[h3] WARNING: stack ulimit is ${_stk} KB, the reference runs had 65536 (docker --ulimit stack=67108864)"
_lck=$(ulimit -l); [ "$_lck" != unlimited ] && \
    echo "[h3] WARNING: memlock ulimit is ${_lck}, the reference runs had unlimited (docker --ulimit memlock=-1)"
[ -n "$PROMPT_FILE" ] && echo "[h3] prompt: $PROMPT_FILE md5=$(md5sum < "$PROMPT_FILE" | cut -c1-32) bytes=$(wc -c < "$PROMPT_FILE")"

# setsid: own process group, so the teardown at the bottom kills this run's
# workers and nothing else. pkill -f on the worker process title would match
# every vLLM-Omni worker in the PID namespace, including other people's runs in
# this shared container.
setsid vllm serve "$MODEL" "${SERVE_ARGS[@]}" > "$LOG" 2>&1 &
SRV=$!
PGID=$(ps -o pgid= -p $SRV 2>/dev/null | tr -d ' ')

cleanup() {
    [ "${KEEP_SERVER:-0}" = 1 ] && { echo "[h3] KEEP_SERVER=1, server left on port $PORT (pgid $PGID)"; return; }
    if [ -n "$PGID" ]; then kill -TERM -"$PGID" 2>/dev/null; else kill -TERM $SRV 2>/dev/null; fi
    sleep 8
    if [ -n "$PGID" ]; then kill -KILL -"$PGID" 2>/dev/null; else kill -KILL $SRV 2>/dev/null; fi
}
trap cleanup EXIT

# Two partitions of ~134 GiB each plus the Qwen3-VL encoder and two VAEs: the
# load is minutes, not seconds, hence the 1800/1500 s init deadlines and this
# poll rather than a fixed sleep.
echo -n "[h3] waiting for /health"
until curl -s -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/health"; do
    kill -0 $SRV 2>/dev/null || { echo; echo "[h3] server exited during init -- tail of $LOG:"; tail -40 "$LOG"; exit 1; }
    echo -n .; sleep 10
done
echo " up"

REQ_FIELDS=("${SHAPE_FIELDS[@]}" -F "fps=$FPS" -F "flow_shift=$FLOW_SHIFT" -F "seed=$SEED")
# FRAMES pins num_frames and then duration must NOT also be sent: the two are
# redundant ways to say the same thing and sending both invites the server to
# resolve the conflict silently, which is exactly the ambiguity a shape-sensitive
# measurement cannot carry.
if [ -n "$FRAMES" ]; then
    REQ_FIELDS+=(-F "num_frames=$FRAMES"
                 -F "extra_params={\"task\":\"$TASK\",\"audio_flow_shift\":$AUDIO_FLOW_SHIFT}")
else
    REQ_FIELDS+=(-F "extra_params={\"task\":\"$TASK\",\"duration\":$DURATION,\"audio_flow_shift\":$AUDIO_FLOW_SHIFT}")
fi
if [ -n "$PROMPT_FILE" ]; then
    [ -s "$PROMPT_FILE" ] || { echo "[h3] PROMPT_FILE=$PROMPT_FILE is missing or empty"; exit 2; }
    REQ_FIELDS+=(-F "prompt=<$PROMPT_FILE")
else
    REQ_FIELDS+=(-F "prompt=$PROMPT")
fi
[ -n "$REF_IMAGE" ] && REQ_FIELDS+=(-F "input_reference=@${REF_IMAGE};type=${REF_MIME}")

# send <kind> <steps>: one /v1/videos/sync request, validated. The sync endpoint
# returns the MP4 bytes directly (the async POST /v1/videos returns a job to poll
# instead). --max-time tracks the server's own sync deadline so curl is not the
# one that gives up first.
#
# Report x-inference-time-s, not the curl wall time: the header is the server's
# own measure of the generation and excludes request upload, MP4 handoff and
# client-side buffering. They agree closely here, but only one of them is the
# number a server-side profile can be reconciled against.
send() {
    local kind=$1 steps=$2 out=$OUT_DIR/$TAG.$1.mp4 hdr=$OUT_DIR/$TAG.$1.headers rc code
    echo "[h3] REQ $kind steps=$steps $(date -u +%T)"
    local t0=$(date +%s)
    code=$(curl --max-time "${VLLM_OMNI_VIDEO_SYNC_TIMEOUT}" -sS --noproxy '*' -X POST \
        "http://127.0.0.1:$PORT/v1/videos/sync" "${REQ_FIELDS[@]}" \
        -F "num_inference_steps=$steps" -D "$hdr" -o "$out" -w '%{http_code}')
    rc=$?
    local xinf=$(grep -ioE 'x-inference-time-s: [0-9.]+' "$hdr" 2>/dev/null | tr -d '\r' | awk '{print $2}')
    local peak=$(grep -ioE 'x-peak-memory-mb: [0-9.]+' "$hdr" 2>/dev/null | tr -d '\r' | awk '{print $2}')
    echo "[h3]   $kind rc=$rc http=${code:-?} x-inference-time-s=${xinf:-NA} x-peak-memory-mb=${peak:-NA} elapsed=$(( $(date +%s) - t0 ))s"
    # On an error the body is JSON and -o leaves it sitting there named .mp4,
    # which reads like a successful generation. Check the ISO-BMFF box type, then
    # insist on BOTH streams: H3's whole point is joint video+audio, and a silent
    # MP4 is a failure this script must not pass.
    if [ $rc -ne 0 ] || [ "${code:-}" != 200 ] || [ "$(dd if="$out" bs=1 skip=4 count=4 2>/dev/null)" != "ftyp" ]; then
        echo "[h3] NOT AN MP4 (curl rc=$rc http=${code:-?}) -- server said:"; head -c 500 "$out"; echo
        echo "[h3] tail of $LOG:"; tail -30 "$LOG"
        return 1
    fi
    echo "[h3]   MP4 OK: $out ($(stat -c%s "$out") bytes)"
    if command -v ffprobe > /dev/null; then
        local streams=$(ffprobe -v error -show_entries stream=codec_type,duration -of csv=p=0 "$out" | tr '\n' ' ')
        echo "[h3]   streams: $streams"
        case "$streams" in *video*) ;; *) echo "[h3] FAILED: no video stream"; return 1 ;; esac
        case "$streams" in *audio*) ;; *) echo "[h3] FAILED: no audio stream -- joint generation did not produce audio"; return 1 ;; esac
    fi
    XINF_TABLE+=("$kind steps=$steps x-inference-time-s=${xinf:-NA} peak_mb=${peak:-NA}")
    return 0
}

XINF_TABLE=()
# The warm request is discarded by construction: it exists to move one-time cost
# out of the timed ones, so its own number is not a result.
[ "$WARM_STEPS" -gt 0 ] && { send warm "$WARM_STEPS" || exit 1; }
for i in $(seq 1 "$REPS"); do send "timed$i" "$STEPS" || exit 1; done
OUT=$OUT_DIR/$TAG.timed1.mp4

# Self-evidence, per docs/user_guide/diffusion/models/minimax_h3_xpu_perf.md.
# Read these as two different kinds of claim: an "installed" line proves only
# that a wrap happened at load, while calls=/sliced=/[RING_RESOLVED] are the
# runtime counters that prove the fast path actually ran. And a MISSING line does
# not distinguish "switch off" from "code not in this build" -- for that, pin the
# build out of band, which is what the rev-parse below records.
echo "[h3] --- request times (server-reported) ---"
printf '[h3] %s\n' "${XINF_TABLE[@]}"
# Name the cell this run is comparable to, but only when the shape really is the
# recorded one -- a reference line printed next to a number measured at some other
# shape is worse than no reference line at all.
if [ "$TASK" = t2va ] && [ "$STEPS" = 50 ] && [ "$FRAMES" = 141 ] && [ "$CACHE" = 0 ] && \
   [ -z "$QUANT" ] && [ -z "$KV_DTYPE" ] && [ -z "$ATTN" ] && \
   [ "${WIDTH:-1248}" = 1248 ] && [ "${HEIGHT:-768}" = 768 ]; then
    case "$CARDS" in
        1) echo "[h3] reference: 889.96 s / peak 76846 MB (hq-cri-d93-a, h3port/cri:b98ebcc2-mkstack)" ;;
        4) echo "[h3] reference: 212.07-212.12 s (h3port/cri:b98ebcc2-mkstack) and 222.49-222.61 s (vllm-cri-omni @ 6ed0e9d3), both peak 92934 MB, hq-cri-d93-a cards 0-3" ;;
    esac
fi
echo "[h3] deployed tree: $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[h3] --- switch evidence (absent line != off; see the doc) ---"
# tqdm writes its progress bar with \r, so the whole denoise loop is ONE line as
# far as grep is concerned and any marker logged during it comes back wrapped in
# a screenful of progress bars. Normalize \r to \n first and the markers stand
# alone -- purely cosmetic, but the counters are the point of this section.
FLAT=$OUT_DIR/$TAG.flat.log
tr '\r' '\n' < "$LOG" > "$FLAT"
grep -aE '\[GEMM_RELAYOUT\]|\[RING_RESOLVED\]|\[XPU_PREFIX_KV\]|\[H3_VAE_[A-Z_0-9]+\]' "$FLAT" | tail -20
grep -aE 'ring attention is enabled|falling back to the original attention' "$FLAT" | tail -5
# One line per precision knob, each of which can silently not take effect:
# quantization can be auto-detected off or overridden, the attention backend can
# resolve to the platform default, and the sage3 kernel can decline every call.
grep -aE 'Building quantization config|Auto-detected quantization|Resolved diffusion attention backend|SageAttention3 XPU|Ring attention block kernel' "$FLAT" | sort -u | tail -10
if [ -n "$QUANT" ] && ! grep -aq 'Building quantization config' "$FLAT"; then
    echo "[h3] WARNING: no 'Building quantization config' line -- QUANT=$QUANT may not have been applied"
fi
if [ -n "$ATTN" ] && ! grep -aq "Resolved diffusion attention backend '$ATTN'" "$FLAT"; then
    echo "[h3] WARNING: backend '$ATTN' never resolved -- check for the deepklox import warning above"
fi
if [ "$CARDS" != 1 ] && ! grep -aq '\[RING_RESOLVED\]' "$FLAT"; then
    echo "[h3] WARNING: no [RING_RESOLVED] line -- the SYCL/TLA ring never resolved, so this is not a ring$CARDS result"
fi
echo "[h3] --- stage timings ---"
grep -aE '\[DiffusionPipelineProfiler\]' "$FLAT" | tail -8
# Steady-state per-step cost, which is what scales with cards -- the first steps
# carry compile and allocator warmup and are not representative.
grep -aoE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]' "$FLAT" | tail -1
echo "[h3] --- recompiles (each one is per-request cost) ---"
grep -aic 'recompiling' "$FLAT"
