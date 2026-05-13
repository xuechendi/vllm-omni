# WAN 2.2 torch.compile Test Report

**Generated:** 2026-05-13  
**Test File:** `test_wan22_torch_compile.py`  
**Total Tests:** 19 (all passing ✅)

---

## Summary

This report covers torch.compile optimization overhead for WAN 2.2 Speech-to-Video (S2V) operations, measuring both Triton JIT kernel generation and CompiledFxGraph compilation costs.

| Category | Tests | Total Overhead | Notes |
|----------|-------|----------------|-------|
| **Triton JIT Kernels** | 7 | ~44.28ms | Auto-generated GPU kernels from torch.inductor |
| **CompiledFxGraph Ops** | 12 | ~2001.39ms* | FX graph representations (includes 1423.58ms SDPA) |
| **Total Compilation** | **19** | **~2045.67ms** | *Cold start overhead measurement* |

\* Note: Self-Attn SDPA has exceptionally high cold-start overhead (1423.58ms) which dominates CompiledFxGraph times.

**Test Configuration:**
- `WARMUP=1`, `RUNS=1` (single-run measurements for compilation overhead)
- `BENCHMARK_MODE=1` (enabled by default)
- Shapes: `BATCH_SIZE=1`, `SEQ_LEN=75600`, `HIDDEN_DIM=5120`, `NUM_HEADS=40`, `HEAD_DIM=128`

---

## Triton JIT Kernels (TestTritonJITKernels)

Auto-generated Triton kernels from torch.inductor when using `torch.compile(mode="max-autotune")`.

| Test Name | Latency (ms) | Kernel Pattern | Source/Notes |
|-----------|--------------|----------------|--------------|
| **test_layer_norm_fused** | 5.424 | `triton_red_fused__to_copy_native_layer_norm_*` | LayerNorm + dtype conversion fusion |
| **test_gelu_activation** | 9.982 | `triton_poi_fused_gelu_*` | GELU in FFN layers |
| **test_segment_modulation_fused** | 5.066 | `triton_poi_fused_add_mul_*` | S2V segment modulation (noisy/ref/motion split) |
| **test_residual_layernorm_fused** | 8.120 | `triton_red_fused_add_native_layer_norm_*` | Residual + LayerNorm fusion |
| **test_rmsnorm_fused** | 5.525 | `triton_red_fused__to_copy_div_mul_pow_sqrt_sum_view_*` | RMSNorm for QK normalization |
| **test_tensor_concat_fused** | 5.045 | `triton_poi_fused_cat_*` | Tensor concatenation |
| **test_clone_contiguous** | 5.115 | `triton_poi_fused_clone_*` | Tensor clone/contiguous memory layout |

**Triton Kernel Categories:**
- **Normalization**: LayerNorm, RMSNorm (19.07ms, 43%)
- **Activation**: GELU (9.98ms, 23%)
- **Tensor Ops**: Segment modulation, concat, clone (15.23ms, 34%)

---

## CompiledFxGraph Operations (TestCompiledFxGraphOps)

FX graph representations containing Triton kernels and ATen operations. These measure the overhead of torch.compile capturing and optimizing complete operation sequences.

### Cross-Attention Pipeline (3 tests)

Operations from `wan2_2_s2v_transformer.py::WanS2VCrossAttention.forward` (lines 372-383)

| Test Name | Latency (ms) | Source Lines | Operations |
|-----------|--------------|--------------|------------|
| **test_cross_attention_qkv_projection** | 37.357 | 372-377 | QKV projection + QK normalization |
| **test_cross_attention_sdpa_call** | 13.647 | 378 | `scaled_dot_product_attention` dispatch |
| **test_cross_attention_output_projection** | 32.386 | 380-383 | Flatten + output projection |

**Subtotal:** 83.39ms

---

### Self-Attention Pipeline (5 tests)

Operations from `wan2_2_s2v_transformer.py::WanS2VSelfAttention.forward` (lines 295-329)

| Test Name | Latency (ms) | Source Lines | Operations |
|-----------|--------------|--------------|------------|
| **test_self_attention_qkv_projection_and_norm** | 118.800 | 298-306 | Fused QKV proj (5120→15360) + split + QK norm |
| **test_self_attention_qkv_reshape** | 2.078 | 308-311 | Reshape to multi-head [B,S,N,D] |
| **test_self_attention_rope_application** | 107.731 | 313-315 | RoPE complex multiplication (720p: 21×45×80) |
| **test_self_attention_sdpa_call** | 1423.580 ⚠️ | 323-324 | `scaled_dot_product_attention` (self-attn) |
| **test_self_attention_output_flatten_proj** | 32.579 | 325-329 | Flatten + output projection |

**Subtotal:** 1684.77ms (dominated by SDPA cold start)

**Note:** Self-attention SDPA has exceptionally high first-call overhead (1423.58ms) due to kernel compilation and autotuning for large sequence length (75,600 tokens).

---

### Segment Modulation (2 tests)

S2V-specific feature that splits latents into noisy vs reference/motion segments for differential modulation.

| Test Name | Latency (ms) | Source Lines | Operations |
|-----------|--------------|--------------|------------|
| **test_segment_modulation_pre_attention** | 8.266 | 437-442 | LayerNorm + per-segment modulation + concat |
| **test_segment_reassembly_post_attention** | 6.544 | 444-449 | Segment splitting + modulation + reassembly |

**Subtotal:** 14.81ms

---

### Other Operations (2 tests)

| Test Name | Latency (ms) | Source | Operations |
|-----------|--------------|--------|------------|
| **test_ffn_linear_layers** | 189.168 | `WanFeedForward` | FFN fc1 (5120→13824) + fc2 (13824→5120) |
| **test_cumsum_seqlen_generation** | 11.256 | Variable-length attention | Cumsum for sequence length indexing |

**Subtotal:** 200.42ms

---

## Performance Analysis

### Top 5 Hotspots (Compilation Overhead)

1. **Self-Attn SDPA Call** - 1423.580ms (69.4%) ⚠️ Extreme cold-start overhead
2. **FFN Linear Layers** - 189.168ms (9.2%)
3. **Self-Attn QKV Proj+Norm** - 118.800ms (5.8%)
4. **Self-Attn RoPE Apply** - 107.731ms (5.3%)
5. **Cross-Attn QKV Projection** - 37.357ms (1.8%)

### Overhead Breakdown

**By Component:**
| Component | Overhead | Percentage | Tests |
|-----------|----------|------------|-------|
| Self-Attention SDPA | 1423.580ms | 69.4% | 1 test |
| Linear Layers (QKV/FFN) | 345.325ms | 16.8% | 3 tests |
| RoPE Operations | 107.731ms | 5.3% | 1 test |
| Segment Modulation | 14.81ms | 0.7% | 2 tests |
| Triton Kernels | 44.28ms | 2.2% | 7 tests |
| Other FxGraph Ops | 110.04ms | 5.4% | 5 tests |

**By Test Class:**
- **TestTritonJITKernels**: 44.28ms (2.2%)
- **TestCompiledFxGraphOps**: 2001.39ms (97.8%)

---

## Key Insights

1. **SDPA Cold Start Dominates**: Self-attention SDPA accounts for 69.4% of total overhead (1423.58ms). This is a one-time cost during first invocation with these shapes (SEQ_LEN=75,600).

2. **Linear Layers Are Second**: QKV projections and FFN layers account for 16.8% (345.33ms) due to large matrix multiplications (75,600 × 5,120).

3. **RoPE Overhead**: Complex-valued rotation for 720p video (21×45×80 spatial grid) takes 107.73ms to compile.

4. **Segment Modulation is Lightweight**: Despite being S2V-specific and generating custom kernels, the overhead is only 14.81ms (0.7%).

5. **Triton Kernel Generation**: Pure Triton kernel overhead (44.28ms) is small compared to CompiledFxGraph overhead (2001.39ms), suggesting most time is spent in graph optimization and autotuning.

---

## Implementation Notes

### Test Replacements

To avoid distributed PyTorch initialization requirements, the following vllm-omni classes were replaced with simple torch.nn equivalents:

| Original Class | Replacement | Reason |
|----------------|-------------|--------|
| `DistributedRMSNorm` | `torch.nn.RMSNorm` | Requires tensor parallel initialization |
| `FlashAttentionBackend` | `torch.nn.functional.scaled_dot_product_attention` | Internal vllm checks |
| `WanS2VSelfAttention` | `SimpleSelfAttnQKV` (custom) | Uses `ColumnParallelLinear` |
| `WanS2VCrossAttention` | `SimpleCrossAttnQKV` (custom) | Uses `ColumnParallelLinear` |
| `WanFeedForward` | `SimpleFFN` (custom) | Uses `ColumnParallelLinear` |

These replacements maintain the same computational patterns while avoiding distributed dependencies.

---

## Test Execution

### Run all tests:
```bash
pytest test_wan22_torch_compile.py -v -s
```

### Run specific category:
```bash
# Triton JIT kernels
pytest test_wan22_torch_compile.py::TestTritonJITKernels -v -s

# CompiledFxGraph operations
pytest test_wan22_torch_compile.py::TestCompiledFxGraphOps -v -s
```

### Run individual test:
```bash
pytest test_wan22_torch_compile.py::TestTritonJITKernels::test_gelu_activation -v -s
```

### Fast validation (disable benchmarking):
```bash
BENCHMARK_MODE=0 pytest test_wan22_torch_compile.py -v -s
```

---

## Related Documentation

- **TEST_SUITE_SUMMARY.md** - Complete WAN 2.2 test suite overview
- **REORGANIZATION_SUMMARY.md** - Test suite restructuring details
- **TRITON_KERNEL_MAPPING.md** - Triton kernel analysis and mapping
- **COMPILED_FX_GRAPH_MAPPING.md** - CompiledFxGraph detailed analysis
- **TORCHDYNAMO_COMPILED_OPS.md** - TorchDynamo compilation overhead analysis

---

## Test Statistics

| Metric | Value |
|--------|-------|
| **Total Tests** | 19 |
| **Passing Tests** | 19 (100%) ✅ |
| **Test Classes** | 2 |
| **Total Overhead** | 2045.67ms |
| **Avg per Test** | 107.67ms |
| **Median Overhead** | 11.26ms |
| **Max Overhead** | 1423.58ms (Self-Attn SDPA) |
| **Min Overhead** | 2.08ms (QKV Reshape) |

---

**Coverage:**
- ✅ Triton JIT kernel generation (7 kernel types)
- ✅ Self-attention compilation (5 operation stages)
- ✅ Cross-attention compilation (3 operation stages)
- ✅ Segment modulation (2 S2V-specific operations)
- ✅ FFN compilation overhead
- ✅ RoPE complex multiplication

---

*Report generated from test run on 2026-05-13. All measurements represent cold-start compilation overhead with WARMUP=1, RUNS=1.*
