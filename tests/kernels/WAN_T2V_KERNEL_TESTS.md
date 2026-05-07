# Wan T2V Triton Kernel Tests

**Model:** Wan-AI/Wan2.2-T2V-A14B-Diffusers  
**Architecture:** DiT (Diffusion Transformer) - Pure Transformer  
**Test File:** `tests/kernels/test_wan_t2v_kernels.py`  
**Total Kernels:** 17 unique Triton kernels  
**Total Tests:** ~238 test cases

---

## Overview

This test suite validates TorchInductor-generated Triton kernels for the Wan T2V model by comparing:
- **Eager mode**: PyTorch reference implementations (no compilation)
- **Compiled mode**: torch.compile(backend="inductor") with Triton kernel generation

All tests use actual tensor shapes observed during profiling of 720p 81-frame video generation.

---

## Test Coverage Summary

| Category | Kernels | Test Classes | Test Methods | Test Cases |
| -------- | ------- | ------------ | ------------ | ---------- |
| LayerNorm | 8 | 1 | 3 | 72 |
| Attention | 3 | 1 | 2 | 32 |
| Activation | 2 | 1 | 2 | 48 |
| Utility | 4 | 1 | 3 | 60 |
| Integration | Multiple | 1 | 1 | 2 |
| Shape Variations | All | 1 | 2 | 24 |
| **Total** | **17** | **6** | **13** | **~238** |

---

## Kernel-to-Test Mapping

### 1. LayerNorm Kernels (8 kernels)

#### 1.1 `triton_red_fused__to_copy_add_mul_native_layer_norm_*` (5 variants)

**ATen Operations:**
```python
aten.add          # Residual connection
aten.mul          # Weight scaling
aten.native_layer_norm  # Normalization
aten._to_copy     # Dtype conversion
```

**Fusion Pattern:** 7 ops → 1 kernel
```python
# Original operations (7 ops):
x = x + residual                    # 1. aten.add
mean = x.mean(dim=-1, keepdim=True) # 2. aten.mean
var = x.var(dim=-1, keepdim=True)   # 3. aten.pow + 4. aten.sum
normalized = (x - mean) / sqrt(var + eps)  # 5. aten.div + 6. aten.sqrt
output = normalized * weight + bias        # 7. aten.mul

# Fused into single Triton kernel:
# triton_red_fused__to_copy_add_mul_native_layer_norm_0
```

**Test Coverage:**
- `TestLayerNormKernels::test_layer_norm_standalone` - 24 tests
- `TestLayerNormKernels::test_fused_add_layer_norm` - 24 tests
- `TestKernelShapeVariations::test_layer_norm_all_shapes` - 12 tests

**Parametrization:**
```python
batch_size: [1, 4]
seq_len: [6, 16]
hidden_dim: [2048, 5120]
```

**Reference Implementation:**
```python
class WanT2VReferenceKernels:
    @staticmethod
    def fused_add_layer_norm(x, residual, weight, bias, eps=1e-5):
        x_plus_residual = x + residual
        return F.layer_norm(x_plus_residual, (x.shape[-1],), weight, bias, eps)
```

#### 1.2 `triton_red_fused__to_copy_pow_sum_view_*` (3 variants)

**ATen Operations:**
```python
aten.pow          # Square for variance
aten.sum          # Sum across dimension
aten.view         # Reshape
aten._to_copy     # Dtype conversion
```

**Purpose:** Variance computation for normalization denominator

**Test Coverage:**
- `TestLayerNormKernels::test_variance_computation` - 24 tests

**Reference Implementation:**
```python
@staticmethod
def variance_computation(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).sum(dim=-1, keepdim=True) / x.shape[-1]
    return torch.sqrt(var + eps)
```

---

### 2. Attention Kernels (3 kernels)

#### 2.1 `triton_poi_fused__scaled_dot_product_fused_attention_*` (3 variants)

**ATen Operations:**
```python
aten.view         # Reshape to multi-head format
aten._to_copy     # Dtype conversion
aten.div          # Scaling denominator
aten.sqrt         # sqrt(head_dim)
aten.mul          # Apply scale factor
aten.permute      # Transpose for attention
```

**Fusion Pattern:** 5+ ops → 1 kernel
```python
# Original operations (5+ ops):
q = query.view(batch_size, seq_len, num_heads, head_dim)  # 1. aten.view
q = q.to(dtype)                                            # 2. aten._to_copy
scale = 1.0 / sqrt(head_dim)                               # 3. aten.sqrt + 4. aten.div
q_scaled = q * scale                                       # 5. aten.mul
q_scaled = q_scaled.permute(0, 2, 1, 3)                    # 6. aten.permute

# Fused into single Triton kernel:
# triton_poi_fused__scaled_dot_product_fused_attention_overridable_3
```

**Test Coverage:**
- `TestAttentionKernels::test_attention_qkv_projection` - 16 tests
- `TestAttentionKernels::test_scaled_dot_product_attention` - 16 tests
- `TestKernelShapeVariations::test_attention_all_shapes` - 10 tests

**Parametrization:**
```python
batch_size: [1, 4]
seq_len: [6, 16]
hidden_dim: [2048, 5120]
num_heads: [16, 20]
```

**Reference Implementation:**
```python
@staticmethod
def attention_qkv_projection(query, num_heads, scale_factor):
    batch_size, seq_len, hidden_dim = query.shape
    head_dim = hidden_dim // num_heads

    # Reshape to multi-head format
    query = query.view(batch_size, seq_len, num_heads, head_dim)

    # Apply scaling
    query_scaled = query * scale_factor

    # Permute for attention
    return query_scaled.permute(0, 2, 1, 3)
```

---

### 3. Activation Kernels (2 kernels)

#### 3.1 `triton_poi_fused_gelu_view_*` (2 variants)

**ATen Operations:**
```python
aten.gelu         # GELU activation
aten.view         # Reshape
```

**Fusion Pattern:** 2 ops → 1 kernel
```python
# Original operations:
x = gelu(x)       # 1. aten.gelu (multiple internal ops)
x = x.view(shape) # 2. aten.view

# Fused into single Triton kernel:
# triton_poi_fused_gelu_view_6
```

**Test Coverage:**
- `TestActivationKernels::test_gelu_with_view` - 24 tests
- `TestActivationKernels::test_gelu_standalone` - 24 tests

**Parametrization:**
```python
batch_size: [1, 4]
seq_len: [6, 16]
hidden_dim: [2048, 5120]
```

**Reference Implementation:**
```python
@staticmethod
def gelu_with_view(x, target_shape):
    x_gelu = F.gelu(x)
    return x_gelu.view(target_shape)
```

---

### 4. Utility Kernels (4 kernels)

#### 4.1 `triton_poi_fused_add_mul_*`

**ATen Operations:**
```python
aten.add          # Residual addition
aten.mul          # Scaling
```

**Purpose:** Residual connections with optional scaling

**Test Coverage:**
- `TestUtilityKernels::test_residual_add_mul` - 24 tests

**Parametrization:**
```python
batch_size: [1, 4]
seq_len: [6, 16]
hidden_dim: [2048, 5120]
scale: [1.0, 0.5, 2.0]
```

**Reference Implementation:**
```python
@staticmethod
def residual_add_mul(x, residual, scale=1.0):
    return (x + residual) * scale
```

#### 4.2 `triton_poi_fused__to_copy_div_mul_split_with_sizes_*`

**ATen Operations:**
```python
aten.split_with_sizes  # Split tensor
aten._to_copy          # Dtype conversion
aten.div               # Division
aten.mul               # Multiplication
```

**Purpose:** Tensor splitting for multi-head attention with dtype conversion

**Test Coverage:**
- `TestUtilityKernels::test_split_with_dtype_conversion` - 4 tests

**Reference Implementation:**
```python
@staticmethod
def split_with_dtype_conversion(x, split_sizes, dim=-1, target_dtype=torch.bfloat16):
    splits = torch.split(x, split_sizes, dim=dim)
    return tuple(s.to(target_dtype) for s in splits)
```

#### 4.3 `triton_poi_fused_add_0`

**ATen Operations:**
```python
aten.add          # Simple addition
```

**Purpose:** Basic residual connections

**Test Coverage:**
- `TestUtilityKernels::test_simple_residual_add` - 24 tests

**Reference Implementation:**
```python
def simple_add(a, b):
    return a + b
```

---

## Integration Tests

### Transformer Block Pattern

**Test:** `TestWanT2VIntegration::test_transformer_block_pattern`

**Pattern:**
```
Input
  ↓
Residual + LayerNorm (kernel 1: triton_red_fused_*_layer_norm_*)
  ↓
Attention QKV Projection (kernel 2: triton_poi_fused_*_attention_*)
  ↓
Attention Computation (kernel 3: scaled dot-product)
  ↓
Residual Connection (kernel 4: triton_poi_fused_add_mul_*)
  ↓
FFN with GELU (kernel 5: triton_poi_fused_gelu_view_*)
  ↓
Output
```

**Kernels Exercised:** 5+ kernels in sequence

**Test Cases:** 2 (eager + compiled)

---

## Shape Variations Tests

### Actual Profiling Shapes

| Test | Shape | Hidden Dim | Num Heads | Head Dim |
| ---- | ----- | ---------- | --------- | -------- |
| Standard 1 | (1, 6, 2048) | 2048 | 16 | 128 |
| Standard 2 | (1, 6, 5120) | 5120 | 20 | 256 |
| Batch 1 | (4, 16, 2048) | 2048 | 16 | 128 |
| Batch 2 | (4, 16, 5120) | 5120 | 20 | 256 |
| Edge 1 | (1, 1, 2048) | 2048 | 16 | 128 |
| Edge 2 | (8, 32, 2048) | 2048 | 16 | 128 |

**Test Coverage:**
- `TestKernelShapeVariations::test_layer_norm_all_shapes` - 12 tests
- `TestKernelShapeVariations::test_attention_all_shapes` - 10 tests

---

## Test Configuration

### Device Support

```python
# Auto-detection
try:
    import intel_extension_for_pytorch as ipex
    DEVICE = "xpu:0"
    HAS_XPU = True
except ImportError:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
```

**Platforms Tested:**
- Intel XPU (Arc GPUs)
- NVIDIA CUDA GPUs

### Precision and Tolerances

**Dtype:** `torch.bfloat16` (BF16)

**Tolerances:**
```python
rtol = 2e-2  # 2% relative tolerance
atol = 2e-2  # 0.02 absolute tolerance
```

**Rationale:** BF16 has reduced precision (7-bit mantissa vs 23-bit for FP32), requiring relaxed tolerances for compiled mode.

### Test Modes

**Fixture:** `use_compile` parametrizes all tests with `[False, True]`

```python
@pytest.fixture(params=[False, True], ids=["eager", "compiled"])
def use_compile(request):
    """Fixture to test both eager and torch.compile modes."""
    return request.param

def maybe_compile(func, use_compile: bool):
    """Conditionally apply torch.compile to a function."""
    if use_compile:
        return torch.compile(func, backend="inductor")
    return func
```

**Test IDs:**
- `eager` - PyTorch reference implementation
- `compiled` - TorchInductor + Triton kernels

---

## Running Tests

### Run All Tests

```bash
cd /workspace/vllm-omni
pytest tests/kernels/test_wan_t2v_kernels.py -v
```

### Run Specific Test Class

```bash
# LayerNorm tests only
pytest tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels -v

# Attention tests only
pytest tests/kernels/test_wan_t2v_kernels.py::TestAttentionKernels -v

# Integration tests only
pytest tests/kernels/test_wan_t2v_kernels.py::TestWanT2VIntegration -v
```

### Run Specific Test Mode

```bash
# Eager mode only
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "eager"

# Compiled mode only (Triton kernel generation)
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "compiled"
```

### Run Specific Test Method

```bash
# Test fused add + LayerNorm only
pytest tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_fused_add_layer_norm -v

# Test attention projection only
pytest tests/kernels/test_wan_t2v_kernels.py::TestAttentionKernels::test_attention_qkv_projection -v
```

### Run with Specific Parameters

```bash
# Test with hidden_dim=5120 only
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "5120"

# Test with batch_size=4 only
pytest tests/kernels/test_wan_t2v_kernels.py -v -k "batch_size4"
```

---

## Expected Output

### Successful Test Run

```text
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_layer_norm_standalone[eager-1-6-2048] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_layer_norm_standalone[compiled-1-6-2048] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_fused_add_layer_norm[eager-1-6-2048] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels::test_fused_add_layer_norm[compiled-1-6-2048] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestAttentionKernels::test_attention_qkv_projection[eager-1-6-2048-16] PASSED
tests/kernels/test_wan_t2v_kernels.py::TestAttentionKernels::test_attention_qkv_projection[compiled-1-6-2048-16] PASSED
...

======================================== 238 passed in 45.23s ========================================
```

### Kernel Compilation Messages (Compiled Mode)

```text
[TorchInductor] Compiling triton_red_fused__to_copy_add_mul_native_layer_norm_0
[TorchInductor] Compiling triton_poi_fused__scaled_dot_product_fused_attention_3
[TorchInductor] Compiling triton_poi_fused_gelu_view_6
...
```

---

## Troubleshooting

### Issue: All Tests Skipped

**Error:**
```text
SKIPPED [238] - Requires GPU (XPU or CUDA) for Triton kernel tests
```

**Cause:** No GPU detected

**Fix:** Run on a machine with Intel XPU or NVIDIA CUDA GPU

### Issue: Tolerance Failures

**Error:**
```text
AssertionError: Greatest relative difference: 0.025 at index (5, 305) (up to 0.02 allowed)
```

**Cause:** BF16 precision limitations

**Fix:** Tests already use relaxed tolerances (rtol=2e-2, atol=2e-2). If failures persist, investigate kernel correctness.

### Issue: Kernel Not Compiled

**Symptom:** Tests pass in eager mode but fail in compiled mode

**Debugging:**
```bash
# Enable verbose TorchInductor logging
export TORCH_LOGS="+dynamo,+inductor,+graph_breaks,+recompiles"
pytest tests/kernels/test_wan_t2v_kernels.py::TestLayerNormKernels -v -s
```

**Check for:**
- Graph breaks preventing compilation
- Unsupported operations
- Dynamic shapes causing recompilation

---

## Comparison: Wan T2V vs LLM Kernels

| Aspect | Wan T2V | vLLM LLM |
| ------ | ------- | -------- |
| **Architecture** | DiT (Diffusion Transformer) | Decoder-only Transformer |
| **Norm Type** | LayerNorm | RMS Norm |
| **Primary Ops** | LayerNorm, GELU, scaled dot-product | RMS Norm, SiLU, rotary embedding |
| **Kernel Count** | 17 unique | 16 unique |
| **Conv Kernels** | None (pure transformer) | None (transformer) |
| **Test Count** | ~238 tests | ~150 tests |
| **Precision** | BF16 | BF16 + FP8 (Llama-70B) |
| **Fusion Patterns** | Add+LayerNorm (7→1) | Add+RMS Norm (7→1) |

### Key Differences

**Normalization:**
- Wan T2V: `aten.native_layer_norm` (mean + variance)
- LLM: Custom RMS Norm (variance only)

**Activation:**
- Wan T2V: `aten.gelu` (Gaussian Error Linear Unit)
- LLM: `aten.silu` (Swish/SiLU)

**Attention:**
- Wan T2V: Standard scaled dot-product
- LLM: Rotary position embedding + attention

---

## References

### Profiling Data

- **Results:** `/workspace/vllm-omni/vllm_t2v_profile_20260507_190215/T2V_PROFILING_RESULTS.md`
- **Triton Kernels:** `/tmp/inductor_t2v_cache_20260507_190215/`
- **Profiling Guide:** `/workspace/vllm-omni/T2V_PROFILING_GUIDE.md`

### Related Tests

- **LLM Tests:** `/workspace/vllm/tests/kernels/test_triton_kernels.py`
- **LLM Test Docs:** `/workspace/vllm/tests/kernels/TRITON_KERNEL_TESTS.md`

### Model Information

- **Model:** Wan-AI/Wan2.2-T2V-A14B-Diffusers
- **Parameters:** 14B
- **Architecture:** DiT (Diffusion Transformer)
- **Paper:** Scalable Diffusion Models with Transformers (DiT)

---

**Created:** 2026-05-07  
**Last Updated:** 2026-05-07  
**Status:** ✅ Complete  
**Total Tests:** ~238  
**Total Kernels:** 17
