#!/usr/bin/env python3
"""Turn a unitrace result directory into one file you can drag into ui.perfetto.dev.

hy_image3_ar_dit_graph.sh runs this for you at the end of a PROFILE=1 run; this is
the standalone entry point for a result directory you already have.

    python3 make_perfetto.py /workspace/hy_image3_logs/ar_dit_unitrace.graph1.unitrace
    python3 make_perfetto.py <dir> --only dit       # just one stage
    python3 make_perfetto.py <dir> --keep-flows     # keep the dependency arrows

Three things make a raw result directory look like "unitrace produced nothing
viewable", and this handles all three:

1. **Nearly every chrome_trace.json is 0 bytes.** unitrace writes one directory per
   traced process, `<comm>.<pid>/chrome_trace.json`, and --follow-child-process gives
   one to every short-lived child: ldconfig, the launcher, and -- because the DiT JIT
   compiles its kernels -- a g++/as/cc1plus/ld/collect2 set per compilation. Measured
   on one profiled run: 95 directories, 2 with content. Those two are the model
   processes, the AR worker (MoE grouped GEMMs) and the DiT worker (convolutions).

2. **A non-empty trace may be truncated JSON.** A process only closes its `]}` if it
   exits cleanly; one killed with the process group at teardown is cut mid-line, and
   Perfetto rejects the whole file with a parse error rather than showing what is
   there. This repairs the cut by dropping the partial last record.

3. **Most of the records are flow arrows.** `ph: s`/`t`/`f` dependency links are 50-70%
   of them (28k of 87k in the smallest trace measured, 1.6M of 2.17M in the largest)
   and Perfetto draws every one, which is what makes the timeline slow and unreadable.
   Dropped unless --keep-flows.

Both stages are converted by default -- one request spans the AR (stage 0) and the
DiT (stage 1), which are separate processes with separate timelines, and the stage
goes in the output file name.

Writes `perfetto_<stage>_<run>_<comm>.<pid>.json.gz` into the run directory (or the
working directory, if the log tree is root-owned) and prints the path. ui.perfetto.dev
reads .gz directly, and dropping the flows plus gzip took a measured 11.9 MB DiT trace
to 0.24 MB and a 506 MB AR trace to 17.7 MB, which matters because the file has to come
off this host first. The result contains only M and X records, every X with a dur, on
two tracks (HOST and DEVICE) -- which is exactly what Perfetto's JSON importer wants.

This is for eyeballing a timeline. Measuring one is a different job: it has to count
the HOST and DEVICE tracks separately (see the script's header), and it should read the
raw files rather than this reduced copy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

# Which stage a trace belongs to, from its kernels: the DiT has convolutions (patch embed,
# VAE), the AR is the 80B MoE, so it has grouped GEMMs and the fused RMS norm.
DIT_HINTS = ("conv", "xetla_conv", "nchw", "resample")
AR_HINTS = ("grouped_gemm", "fused_add_rms_norm", "per_token_group_quant")
FLOW_PHASES = {"s", "t", "f"}


def load_events(path: Path) -> tuple[list[dict], str]:
    text = path.read_text(errors="replace")
    try:
        blob = json.loads(text)
        return blob.get("traceEvents", []), "valid"
    except json.JSONDecodeError:
        cut = text.rfind("},")
        if cut < 0:
            return [], "unrepairable"
        try:
            blob = json.loads(text[: cut + 1] + "]}")
        except json.JSONDecodeError as exc:
            return [], f"unrepairable ({exc})"
        return blob.get("traceEvents", []), "truncated, repaired"


def classify(events: list[dict]) -> str:
    """Which stage's worker wrote this trace.

    Only convolutions separate them. Both stages run the same 80B MoE backbone, so
    grouped GEMMs / fused RMS norm / per-token quant appear in both and a "most MoE
    kernels wins" rule labels the DiT as the AR. What the AR never has under t2i is a
    convolution: the DiT's patch embed and VAE are the only conv users (165-176
    instances measured), and a text-only AR context has none.
    """
    names = [e.get("name", "").lower() for e in events]
    if any(any(h in n for h in DIT_HINTS) for n in names):
        return "dit"
    if any(any(h in n for h in AR_HINTS) for n in names):
        return "ar"
    return "other"


def convert(path: Path, events: list[dict], state: str, stage: str,
            out_dir: Path, keep_flows: bool) -> Path | None:
    if not events:
        print(f"  {path.parent.name}: {state}, nothing to convert")
        return None
    kept = events if keep_flows else [e for e in events if e.get("ph") not in FLOW_PHASES]
    # The run label is the trace directory's own name, except under the older
    # <run>/result/<comm>.<pid>/ layout, where "result" says nothing and the run is one
    # level further up. hy_image3_ar_dit_graph.sh writes <tag>.unitrace/<comm>.<pid>/
    # instead, and naming that one after its parent produced "hy_image3_logs".
    run_dir = path.parent.parent
    run = run_dir.parent.name if run_dir.name == "result" else run_dir.name
    run = run.removesuffix(".unitrace")
    out = out_dir / f"perfetto_{stage}_{run}_{path.parent.name}.json.gz"
    with gzip.open(out, "wt") as handle:
        json.dump({"traceEvents": kept}, handle)
    print(
        f"  {path.parent.name} [{stage}]: {state}, {len(events)} events -> {len(kept)} kept"
        f" -> {out.name} ({out.stat().st_size / 1e6:.2f} MB)"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="the run directory or its result/ subdirectory")
    parser.add_argument("--only", choices=("ar", "dit", "other"), default=None,
                        help="convert just this stage (default: every non-empty trace)")
    parser.add_argument("--keep-flows", action="store_true", help="keep ph s/t/f arrows")
    parser.add_argument("-o", "--out-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}")
        return 1
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif os.access(run_dir, os.W_OK):
        out_dir = run_dir
    else:
        # The log tree is written by the container as root, so a host-side run cannot
        # put anything next to the trace. Fall back rather than failing at the last step.
        out_dir = Path.cwd()
        print(f"{run_dir} is not writable; writing to {out_dir} instead")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Anything under 1 KB is a process that never got to write kernels.
    traces = [p for p in sorted(run_dir.rglob("chrome_trace.json")) if p.stat().st_size > 1024]
    empty = sum(1 for p in run_dir.rglob("chrome_trace.json") if p.stat().st_size <= 1024)
    print(f"{len(traces)} non-empty trace(s) under {run_dir} ({empty} empty, see the docstring)")
    if not traces:
        return 1

    # Parse each trace exactly once and carry the events forward: --only has to classify
    # before it can select, and the AR's trace reaches 500+ MB, so parsing it a second
    # time to convert it costs more than everything else here put together.
    loaded = []
    for path in traces:
        events, state = load_events(path)
        loaded.append((path, events, state, classify(events)))

    # Both stages by default: one request spans the AR and the DiT, they are separate
    # processes with separate timelines, and the interesting comparisons need both.
    targets = loaded
    if args.only:
        targets = [t for t in loaded if t[3] == args.only]
        if not targets:
            print(f"no trace classified as '{args.only}'; converting all instead")
            targets = loaded

    made = [convert(p, ev, st, sg, out_dir, args.keep_flows) for p, ev, st, sg in targets]
    made = [m for m in made if m]
    if not made:
        return 1
    print("\nOpen https://ui.perfetto.dev and drag in (copy off this host first):")
    for m in made:
        print(f"  {m}")
    print(
        "\nIn Perfetto: the DEVICE<...> track is kernel execution, HOST<...> is the Level Zero\n"
        "calls that submitted them. A graph replay is the stretch where the DEVICE track is\n"
        "continuous and HOST is nearly empty; a launched step has a matching HOST call per\n"
        "kernel and visible gaps between them."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
