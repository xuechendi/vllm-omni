# S2V Cache-DiT Audio Injection Compatibility Issue

## Problem

Cache-dit degrades lip sync quality in S2V (Speech-to-Video) generation.

**Observed Quality:**
- No cache-dit: Good lip sync
- Cache-dit warmup-only (no caching): Mouth PSNR 20.75 dB (Poor)  
- Cache-dit with caching: Mouth PSNR 21.03 dB (Poor)

## Root Cause

**Architecture:** Audio injection happens AFTER transformer blocks
```python
for block in blocks:
    hidden_states = block(hidden_states, ...)  # Cache-dit can cache THIS
    hidden_states = after_transformer_block(block_idx, hidden_states, audio_emb)  # Audio injected AFTER
```

**The Problem:**
1. Step 1 (warmup):
   - Block 3 processes `hidden_states_1` → `block_output_1`
   - Audio injection: `block_output_1 + audio_features_1` → `final_hidden_1`
   - Cache stores: `block_output_1`

2. Step 5 (cached):
   - Block 3 returns CACHED `block_output_1` (computed from old states)
   - Audio injection: `block_output_1 + audio_features_5` → `final_hidden_5`

**Issue:** `block_output_1` was computed when the model saw different audio context in previous blocks. The block's self-attention and cross-attention are based on OLD state distribution.

Even though we re-inject current audio AFTER the block, the block's computation is stale.

## Why This Matters for S2V

- **T2V/I2V:** Conditioning is global (text/image), doesn't change per-frame
- **S2V:** Audio features are FRAME-SPECIFIC and change rapidly during speech
- Lip sync requires precise frame-level audio-visual alignment
- Cached blocks miss the current audio context → degraded lip sync

## Solutions

### Option 1: Disable caching for audio injection blocks
Mark blocks [0, 4, 8, 12, 16, 20, 24, 27] as non-cacheable.

**Pros:** Simple, guarantees correctness
**Cons:** Reduces cache hit rate from ~70% to ~45% (8/32 blocks never cache)

### Option 2: Lower cache threshold for audio blocks
Use stricter residual threshold for audio injection blocks.

**Pros:** Adaptive, some caching still possible during low-audio-variance segments
**Cons:** More complex, requires tuning

### Option 3: Audio-aware adaptive caching
Compute audio variance per frame, disable caching when audio changes rapidly.

**Pros:** Optimal cache hit rate
**Cons:** Most complex, requires audio feature analysis

## Recommendation

Start with **Option 1** to validate the hypothesis, then move to **Option 2** or **3** for optimization.

## Implementation

Modify `enable_cache_for_wan_s2v()` in `cache_dit_backend.py` to pass block exclusion list:
```python
cache_dit.enable_cache(
    BlockAdapter(
        transformer=pipeline.transformer,
        blocks=[pipeline.transformer.blocks],
        forward_pattern=[ForwardPattern.Pattern_2],
        params_modifiers=[
            ParamsModifier(
                cache_config=db_cache_config,
                excluded_block_indices=pipeline.transformer.audio_injector.injected_block_id.keys(),
            ),
        ],
        has_separate_cfg=True,
    ),
    cache_config=db_cache_config,
)
```
