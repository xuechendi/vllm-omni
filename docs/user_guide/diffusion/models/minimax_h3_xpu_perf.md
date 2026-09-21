# MiniMax-H3 on Intel XPU (CRI): performance switches and reference launch configuration

This page documents the environment switches added for MiniMax-H3 video
generation on Intel XPU and the launch configuration validated on 8x CRI D93
(1/2/4/8-card, 50-step x-inference, September 2026). The 1/2/4/8-card
validation and the cross-vendor comparison below were measured on the
September 2026 integration/delivery builds of this series; they have not been
re-measured on this branch tip.

## Performance switches

All switches are read from the server process environment. What the serve log
can prove differs per switch, so the observable states are listed per switch
in the second table below: **installed** (a load-time line printed), **hit**
(runtime counters prove the fast path actually ran), and **off / not
applicable** (what, if anything, is printed then). An install line alone
never proves execution, and for the VAE switches an absent line does not
distinguish "switched off" from "code not in the build".

| Env | Default | What it does |
| --- | --- | --- |
| `H3_VAE_COMPILE` | `1` | Wraps the 36 video-VAE decoder transformer blocks in `torch.compile` on XPU. Disabled (`=0`): eager decoder blocks (VAE decode ~19% slower on CRI) |
| `H3_VAE_FP16_WEIGHTS` | `1` | Persists the 144 VAE decoder-block linear weights as fp16. Disabled: per-forward dtype conversion returns |
| `H3_VAE_XEFMHA` | `1` | Routes VAE decoder attention to the XeFMHA varlen kernel (guarded: xpu / fp16 / batch=1 / non-spatial-parallel / heads x head_dim = 32 x 64 / 64-byte-aligned; otherwise falls back). Disabled: micro-batched SDPA path |
| `H3_GEMM_RELAYOUT` | `0` (opt-in) | Repacks transformer Linear weights transpose-contiguous ("ab" layout) at load time; skips when DLO owns the weights and skips weights owned by a quantized `quant_method`. Disabled: weights keep the checkpoint layout |
| `H3_XPU_PREFIX_KV` | `1` | flash_attn XPU backend slices K/V to the packed-sequence valid prefix instead of the masked unpad/pad path; read once at import. Disabled: mask path (steady-state per-step ~4% slower on CRI 1-card) |
| (no switch) | — | Ring attention prints one per-rank resolve line to stderr |

Per-switch observable states in the serve log:

| Env | Installed (load-time line) | Hit (runtime evidence) | Off / not applicable |
| --- | --- | --- | --- |
| `H3_VAE_COMPILE` | `[H3_VAE_COMPILE] torch.compile applied to 36 decoder transformer blocks device=xpu:0` | No dedicated runtime counter: the wrap is applied at load and compilation happens inside the first decode. The log cannot separate "wrapped and compiled" from "wrapped but never decoded"; use decode activity itself | No line. Off and absent-from-build are indistinguishable in the log (see below) |
| `H3_VAE_FP16_WEIGHTS` | `[H3_VAE_FP16_WEIGHTS] 144 decoder-block linears persisted as fp16 device=xpu:0` -- for this switch the action *is* the load-time conversion, so this line is also the execution evidence | Same line (the work happens at load) | No line. Off and absent-from-build are indistinguishable in the log |
| `H3_VAE_XEFMHA` | `[H3_VAE_XEFMHA] installed on 36 blocks` -- **installed only**, proves the wrap, not that the kernel ever ran | `[H3_VAE_XEFMHA] first fast-path call: shape=... dtype=... strides=...` and periodic `[H3_VAE_XEFMHA] calls=<n> fallbacks=<m>` with `calls > 0`. `calls=0` with growing `fallbacks` plus `falling back to the original attention: <reason>` lines = installed but the contract was never met | No line. Off and absent-from-build are indistinguishable in the log |
| `H3_GEMM_RELAYOUT` | `[GEMM_RELAYOUT][rankN] 266 linear weights -> transpose-contiguous` -- the repack itself runs at load, so this line is also the execution evidence; `skipped: ...` / `skipped_quant=<n>` account for weights it declined | Same line (the work happens at load) | No line (default is `0`). Off and absent-from-build are indistinguishable in the log |
| `H3_XPU_PREFIX_KV` | `[XPU_PREFIX_KV] enabled=1 sliced kv <kv>-><kv'> q=<q> heads=<h> dtype=<dt>` on first slice | Periodic `[XPU_PREFIX_KV] calls=<n> sliced=<m>`: `sliced > 0` = the slicing ran; `sliced=0` = enabled but no call qualified | Prints `[XPU_PREFIX_KV] enabled=0 (mask path)` once -- the only switch here whose off state is positively logged |
| ring resolve line | — | `[RING_RESOLVED][rank<N> pid=<P>] attn_type=sycl_tla precision=bf16 world=<N>` per rank at resolve time | — |

Two things the log alone cannot tell you:

- **A missing line does not identify the build.** For the VAE switches and
  `H3_GEMM_RELAYOUT`, "switched off" and "code not in the build" both print
  nothing. Establish build identity out of band: record the deployed commit
  (`git -C <tree> rev-parse HEAD`) or fingerprint the deployed files
  (`sha256sum` against the branch), then read the log against that identity.
- **An install line is not execution.** `installed on 36 blocks` proves the
  wrap happened at load; whether the fast path actually ran is answered only
  by the runtime counters (`calls=<n> fallbacks=<m>`, `sliced=<m>`).

## Reference launch configuration (CRI D93, validated 2026-09)

- 1 card: DLO resident 50 (fully resident), attention = vllm-xpu-kernels bf16
  FA (XeFMHA), ring degree 1.
- 2/4/8 cards: TP1 / Ulysses 1 / **ring = N**, with
  `VLLM_OMNI_RING_ATTN_TYPE=sycl_tla` (bf16 precision, the default), text
  encoder tp = N, and VAE parallel = N:
  `--vae-patch-parallel-size N --vae-parallel-mode tile --vae-use-tiling`.
  Do not pin the sequence padding (`pad_seq_len`) far above the packed valid
  length: the SYCL ring path trims each circulated K/V block to the valid
  prefix, and a padded length that leaves any ring rank with a zero-length
  valid shard raises at request time ("SYCL/TLA ring attention does not
  support an empty local K/V shard").
- Do **not** pass `--enforce-eager`; use `--cache-backend none` for the
  uncached baseline.

With this configuration and the switches above (plus `H3_GEMM_RELAYOUT=1` as
delivered), CRI beat RTX PRO 5000 (all-strategy best per card count) on all
twelve 50-step cells; R = PRO/CRI for t2va was 1.18 (1c), 1.37 (2c), 1.46
(4c) and 1.35 (8c). These twelve cells are September 2026 measurements of the
integration/delivery stack (not of this branch tip) and are quoted here as the
reference result for that configuration.
