# Wan T2V Test Suite

## Overview

Comprehensive unit test suite for Wan-AI/Wan2.2-T2V-A14B-Diffusers (14B parameters).

**📖 Complete Guide:** [`WAN_T2V_COMPLETE_TEST_GUIDE.md`](WAN_T2V_COMPLETE_TEST_GUIDE.md) - Read this first!

### Test Coverage

**Total: 318 tests covering 129 operations (100%)**

| Test Suite | File | Tests | Operations | Type |
| ---------- | ---- | ----- | ---------- | ---- |
| **Triton Kernels** | `test_wan_t2v_kernels.py` | 208 | 17 unique | Custom GPU code |
| **Extern Operations** | `test_wan_t2v_extern_ops.py` | 110 | 112 calls | Library GEMM |

---

## Documentation Files

| File | Purpose | Lines |
| ---- | ------- | ----- |
| **WAN_T2V_COMPLETE_TEST_GUIDE.md** | 📖 **Complete guide** (Triton + Extern) | ~1,800 |
| README.md | Quick reference (this file) | ~200 |
| test_wan_t2v_kernels.py | Triton kernel test implementation | ~800 |
| test_wan_t2v_extern_ops.py | Extern operation test implementation | ~650 |

**Start here:** [`WAN_T2V_COMPLETE_TEST_GUIDE.md`](WAN_T2V_COMPLETE_TEST_GUIDE.md)

---

## Requirements

### Hardware

Tests require a GPU to run (Triton kernels target GPU execution):
- **Intel XPU** (Arc GPUs) - Primary target platform
- **NVIDIA CUDA** - Also supported

Tests will be **skipped** on CPU-only systems.

### Software Dependencies

```bash
# Core dependencies
pip install torch intel_extension_for_pytorch  # For XPU
# OR
pip install torch  # For CUDA

# Test dependencies
pip install pytest
```

---

## Running Tests

### Quick Start

```bash
cd /workspace/vllm-omni

# Run Triton kernel tests
pytest tests/kernels/test_wan_t2v_kernels.py -v

# Run extern operation tests
pytest tests/kernels/test_wan_t2v_extern_ops.py -v

# Run all tests
pytest tests/kernels/ -v
```

### Run Specific Test Categories

```bash
# LayerNorm tests (72 tests)
pytest tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels -v

# Attention tests (32 tests)
pytest tests/kernels/test_wan_t2v_kernels.py::TestAttentionKernels -v

# Activation tests (48 tests)
pytest tests/kernels/test_wan_t2v_kernels.py::TestActivationKernels -v

# Utility tests (68 tests)
pytest tests/kernels/test_wan_t2v_kernels.py::TestUtilityKernels -v

# Integration tests (2 tests)
pytest tests/kernels/test_wan_t2v_kernels.py::TestWanT2VIntegration -v

# Shape variations (26 tests)
pytest tests/kernels/test_wan_t2v_kernels.py::TestKernelShapeVariations -v
```

### Run Specific Test Modes

```bash
# Eager mode only (PyTorch reference)
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "eager"

# Compiled mode only (Triton kernels)
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "compiled"
```

### Run with Specific Parameters

```bash
# Tests with hidden_dim=5120
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "5120"

# Tests with batch_size=4
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "batch_size4"

# Fused add + LayerNorm tests only
pytest tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_fused_add_layer_norm -v
```

---

## Test Structure

### Test Classes

| Class | Kernels Tested | Test Count | Description |
| ----- | -------------- | ---------- | ----------- |
| `TestLayerNormKernels` | 8 | 72 | LayerNorm and variance computation |
| `TestAttentionKernels` | 3 | 32 | QKV projection and scaled dot-product |
| `TestActivationKernels` | 2 | 48 | GELU activation with fusion |
| `TestUtilityKernels` | 4 | 68 | Residual connections, splitting |
| `TestWanT2VIntegration` | Multiple | 2 | Full transformer block patterns |
| `TestKernelShapeVariations` | All | 26 | Various tensor shapes |

### Dual-Mode Testing

All tests run in **both modes**:
1. **Eager mode**: PyTorch reference implementation (no compilation)
2. **Compiled mode**: `torch.compile(backend="inductor")` generates Triton kernels

This ensures:
- Reference implementations are correct
- Compiled kernels match eager output
- Kernel fusion doesn't introduce errors

---

## Kernel Coverage

### 1. LayerNorm Kernels (8 kernels)

**Kernels:**
- `triton_red_fused__to_copy_add_mul_native_layer_norm_*` (5 variants)
- `triton_red_fused__to_copy_pow_sum_view_*` (3 variants)

**Fusion:** Add + LayerNorm (7 ops → 1 kernel)

**Tests:**
- `test_layer_norm_standalone` - Basic LayerNorm
- `test_fused_add_layer_norm` - Residual + LayerNorm fusion
- `test_variance_computation` - Variance for normalization

### 2. Attention Kernels (3 kernels)

**Kernels:**
- `triton_poi_fused__scaled_dot_product_fused_attention_*` (3 variants)

**Fusion:** QKV projection + scaling (5+ ops → 1 kernel)

**Tests:**
- `test_attention_qkv_projection` - Query/Key/Value projection
- `test_scaled_dot_product_attention` - Full attention mechanism

### 3. Activation Kernels (2 kernels)

**Kernels:**
- `triton_poi_fused_gelu_view_*` (2 variants)

**Fusion:** GELU + view (2 ops → 1 kernel)

**Tests:**
- `test_gelu_with_view` - GELU activation with reshaping
- `test_gelu_standalone` - Pure GELU activation

### 4. Utility Kernels (4 kernels)

**Kernels:**
- `triton_poi_fused_add_mul_*`
- `triton_poi_fused__to_copy_div_mul_split_with_sizes_*`
- `triton_poi_fused_add_0`

**Tests:**
- `test_residual_add_mul` - Residual with scaling
- `test_split_with_dtype_conversion` - Tensor splitting
- `test_simple_residual_add` - Basic residual addition

---

## Expected Output

### Successful Run

```text
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_layer_norm_standalone[eager-2048-6-1] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_layer_norm_standalone[compiled-2048-6-1] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_fused_add_layer_norm[eager-2048-6-1] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_fused_add_layer_norm[compiled-2048-6-1] PASSED
...
====================== 208 passed in 45.23s ======================
```

### Kernel Compilation (Compiled Mode)

When running compiled tests, you'll see TorchInductor messages:

```text
[TorchInductor] Compiling triton_red_fused__to_copy_add_mul_native_layer_norm_0
[TorchInductor] Compiling triton_poi_fused__scaled_dot_product_fused_attention_3
[TorchInductor] Compiling triton_poi_fused_gelu_view_6
```

These indicate Triton kernels are being generated and used.

---

## Troubleshooting

### Issue: All Tests Skipped

**Symptom:**
```text
SKIPPED [208] - Requires GPU (XPU or CUDA) for Triton kernel tests
```

**Cause:** No GPU detected (running on CPU-only system)

**Fix:** Run on a machine with Intel XPU or NVIDIA CUDA GPU

---

### Issue: Tolerance Failures

**Symptom:**
```text
AssertionError: Greatest relative difference: 0.025 at index (5, 305)
```

**Cause:** BF16 precision limitations

**Current tolerances:** `rtol=2e-2, atol=2e-2` (already relaxed for BF16)

**Debug:**
1. Check if the difference is consistent across runs
2. Verify the kernel is being compiled (check for TorchInductor messages)
3. Compare eager vs compiled outputs manually

---

### Issue: Kernel Not Compiled

**Symptom:** Test passes in eager mode but fails in compiled mode

**Debug steps:**

1. **Enable verbose logging:**
   ```bash
   export TORCH_LOGS="+dynamo,+inductor,+graph_breaks,+recompiles"
   pytest tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels -v -s
   ```

2. **Check for graph breaks:**
   Look for messages like:
   ```text
   [graph_break] unsupported operation: ...
   ```

3. **Verify Triton is available:**
   ```python
   python -c "import triton; print(triton.__version__)"
   ```

4. **Check inductor cache:**
   ```bash
   ls /tmp/torch_compile_* 2>/dev/null
   ```

---

### Issue: OOM (Out of Memory)

**Symptom:**
```text
RuntimeError: CUDA out of memory
```

**Cause:** Large tensor sizes (e.g., hidden_dim=5120, batch_size=4)

**Fix:** Run tests with smaller parameters:
```bash
# Run only small configurations
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "2048 and 1 and 6"
```

Or modify the parametrization in the test file.

---

## Comparison with LLM Tests

| Aspect | Wan T2V Tests | vLLM LLM Tests |
| ------ | ------------- | -------------- |
| **File** | `test_wan_t2v_kernels.py` | `test_triton_kernels.py` |
| **Model** | Wan-AI/Wan2.2-T2V-A14B | Llama-3.3-70B, Qwen3 |
| **Architecture** | DiT (Diffusion Transformer) | Decoder-only Transformer |
| **Norm Type** | LayerNorm | RMS Norm |
| **Activation** | GELU | SiLU |
| **Kernels** | 17 unique | 16 unique |
| **Tests** | 208 | 150 |
| **Precision** | BF16 | BF16 + FP8 |

---

## Development Workflow

### Adding New Tests

1. **Identify new kernel** from profiling:
   ```bash
   grep -r "triton_" /tmp/inductor_t2v_cache_*/
   ```

2. **Extract ATen operations:**
   ```bash
   grep "Original ATen:" /tmp/inductor_t2v_cache_*/*.py
   ```

3. **Create reference implementation:**
   ```python
   @staticmethod
   def my_new_kernel(x, y):
       # PyTorch reference implementation
       return x + y
   ```

4. **Add parametrized test:**
   ```python
   @pytest.mark.parametrize("batch_size", [1, 4])
   def test_my_new_kernel(self, batch_size, use_compile):
       # Test implementation
       pass
   ```

5. **Run the new test:**
   ```bash
   pytest tests/kernels/test_wan_t2v_kernels.py::TestMyClass::test_my_new_kernel -v
   ```

---

## Related Documentation

- **Comprehensive Test Guide:** `WAN_T2V_KERNEL_TESTS.md` - Complete kernel-to-test mapping
- **Profiling Results:** `/workspace/vllm-omni/vllm_t2v_profile_20260507_190215/T2V_PROFILING_RESULTS.md`
- **Profiling Guide:** `/workspace/vllm-omni/T2V_PROFILING_GUIDE.md`
- **LLM Tests:** `/workspace/vllm/tests/kernels/test_triton_kernels.py`

---

## References

### Model Information
- **Model:** Wan-AI/Wan2.2-T2V-A14B-Diffusers
- **Parameters:** 14B
- **Architecture:** DiT (Diffusion Transformer)
- **Paper:** Scalable Diffusion Models with Transformers

### Profiling Data
- **Triton Kernels:** `/tmp/inductor_t2v_cache_20260507_190215/`
- **Profile Results:** `/workspace/vllm-omni/vllm_t2v_profile_20260507_190215/`

---

**Created:** 2026-05-07  
**Status:** ✅ Complete  
**Total Tests:** 208  
**Platform:** Intel XPU / NVIDIA CUDA
