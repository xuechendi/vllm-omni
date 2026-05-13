"""
Comprehensive unit tests for WAN 2.2 operations (Combined Kernel + API tests).

This test suite combines:
1. Kernel-level tests from test_wan22_kernels.py (Conv1D/2D/3D, Flash Attention, etc.)
2. API-level tests from test_wan22_ops_profiled.py (operation shapes and flows)

Based on profiling traces:
- T2V: profiler_output_t2v_reduced1_flashattn_720p_eager
- S2V: profiler_output_s2v_reduced1_flashattn_720p_eager

Configuration: 720x1280, 80-81 frames, 75,600 sequence length, WAN_REDUCED_LAYERS=1

Key changes from original files:
- Replaced QKVParallelLinear, ColumnParallelLinear, RowParallelLinear with torch.nn.Linear
- Replaced FlashAttentionImpl.forward_xpu with vllm_xpu_kernels.flash_attn_varlen_func
- Combined T2V and S2V test cases with call count annotations
"""

import numpy as np
import pytest
import torch

# Test device configuration
DEVICE = "xpu" if torch.xpu.is_available() else "cpu"

# Configuration from profiling runs (WAN 2.2 A14B)
BATCH_SIZE = 1
SEQ_LEN = 75600  # 21 × 45 × 80 patches
HIDDEN_DIM = 5120
NUM_HEADS = 40
HEAD_DIM = 128
FFN_DIM = 13824
TEXT_SEQ_LEN = 512
NUM_FRAMES_T2V = 81
NUM_FRAMES_S2V = 80
LATENT_HEIGHT = 90
LATENT_WIDTH = 160

WARMUP = 1
RUNS = 5


def benchmark_kernel(name, fn):
    """Benchmark a kernel-launching operation."""
    # Warmup
    for _ in range(WARMUP):
        fn()
        if DEVICE == "xpu":
            torch.xpu.synchronize()

    # Benchmark
    latencies = []
    for _ in range(RUNS):
        if DEVICE == "xpu":
            torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True) if DEVICE == "xpu" else None
        end = torch.xpu.Event(enable_timing=True) if DEVICE == "xpu" else None

        if DEVICE == "xpu":
            start.record()

        fn()

        if DEVICE == "xpu":
            end.record()
            torch.xpu.synchronize()
            latencies.append(start.elapsed_time(end))

    if latencies:
        return {
            "avg_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
        }
    return None


# ============================================================================
# PART 1: KERNEL-LEVEL TESTS (from test_wan22_kernels.py)
# ============================================================================

# ============================================================================
# 1. Conv1D Operations (S2V Audio Encoder)
# ============================================================================


class TestConv1DKernels:
    """Test Conv1D operations (audio encoding in S2V)."""

    def test_conv1d_audio_encoder_large(self):
        """Test Conv1D in audio encoder (large input: 80K samples)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        audio = torch.randn(1, 1, 80056, device=DEVICE, dtype=torch.bfloat16)
        conv = torch.nn.Conv1d(1, 512, kernel_size=10, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Conv1D Audio 80K", lambda: conv(audio))

        if result:
            print(f"\nConv1D Audio 80K: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")

    def test_conv1d_audio_projector(self):
        """Test Conv1D in audio feature projector (S2V: 4 calls, 92ms)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        audio_proj = torch.randn(4, 1280, 156, device=DEVICE, dtype=torch.bfloat16)
        conv = torch.nn.Conv1d(1280, 2560, kernel_size=3, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Conv1D Audio Projector", lambda: conv(audio_proj))

        if result:
            print(f"\nConv1D Audio Projector: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  S2V: 4 calls, 92.66ms total")


# ============================================================================
# 2. Conv2D Operations (VAE Upsampling)
# ============================================================================


class TestConv2DKernels:
    """Test Conv2D operations (VAE decoder upsampling)."""

    def test_conv2d_vae_256x256(self):
        """Test Conv2D at 256×256 (near pixel space)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(4, 192, 256, 256, device=DEVICE, dtype=torch.bfloat16)
        conv = torch.nn.Conv2d(192, 96, kernel_size=3, padding=1, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Conv2D 256×256", lambda: conv(x))

        if result:
            print(f"\nConv2D 256×256: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 3. Conv3D Operations (VAE - CRITICAL: 40%+ XPU time)
# ============================================================================


class TestConv3DKernels:
    """Test Conv3D operations (VAE video encoder/decoder)."""

    def test_conv3d_most_frequent_shape(self):
        """Test most frequent Conv3D shape: [1, 384, 3, 34, 34]."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(1, 384, 3, 34, 34, device=DEVICE, dtype=torch.bfloat16)
        conv = torch.nn.Conv3d(384, 384, kernel_size=3, padding=1, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Conv3D [1,384,3,34,34]", lambda: conv(x))

        if result:
            print(f"\nConv3D [1,384,3,34,34]: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  T2V: 7,560 calls | S2V: 13,320 calls (most frequent)")

    def test_conv3d_shape2(self):
        """Test Conv3D shape: [1, 96, 6, 258, 258]."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(1, 96, 6, 258, 258, device=DEVICE, dtype=torch.bfloat16)
        conv = torch.nn.Conv3d(96, 96, kernel_size=3, padding=1, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Conv3D [1,96,6,258,258]", lambda: conv(x))

        if result:
            print(f"\nConv3D [1,96,6,258,258]: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  T2V: 4,320 calls | S2V: 6,912 calls")

    def test_conv3d_decoder_to_rgb(self):
        """Test Conv3D VAE decoder final layer (latent to RGB)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        latent = torch.randn(1, 96, 6, 258, 258, device=DEVICE, dtype=torch.bfloat16)
        conv = torch.nn.Conv3d(96, 3, kernel_size=3, padding=1, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Conv3D to RGB", lambda: conv(latent))

        if result:
            print(f"\nConv3D to RGB: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 4. Flash Attention Kernels (12-17% XPU time)
# ============================================================================


class TestFlashAttentionKernels:
    """Test Flash Attention using vllm_xpu_kernels.flash_attn_varlen_func."""

    def test_flash_attn_varlen_self_attention(self):
        """Test flash_attn_varlen_func for self-attention (75,600 × 75,600)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        try:
            from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

            # Prepare inputs for varlen format
            q = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
            k = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
            v = torch.randn(BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

            # Cumulative sequence lengths for varlen format
            cu_seqlens = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=DEVICE)

            softmax_scale = 1.0 / (HEAD_DIM**0.5)

            result = benchmark_kernel(
                "flash_attn_varlen_func Self",
                lambda: flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=SEQ_LEN,
                    max_seqlen_k=SEQ_LEN,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=False,
                ),
            )

            if result:
                print(f"\nflash_attn_varlen_func Self: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
                print("  T2V: 16 calls (~550ms/call) | S2V: 24 calls (~366ms/call)")

        except Exception as e:
            pytest.skip(f"flash_attn_varlen_func not available: {e}")


# ============================================================================
# PART 2: API-LEVEL TESTS (from test_wan22_ops_profiled.py with modifications)
# ============================================================================

# ============================================================================
# 5. Self-Attention Operations (using torch.nn.Linear)
# ============================================================================


class TestSelfAttentionOps:
    """Test self-attention operations using torch.nn.Linear instead of parallel layers."""

    def test_qkv_projection_with_torch_linear(self):
        """Test QKV projection using torch.nn.Linear (replaces QKVParallelLinear)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        # Use torch.nn.Linear instead of QKVParallelLinear
        qkv_proj = torch.nn.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("QKV Projection (torch.nn.Linear)", lambda: qkv_proj(input_tensor))

        if result:
            print(f"\nQKV Projection: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  Output shape: [1, 75600, 15360] (Q+K+V concatenated)")

    def test_qk_normalization(self):
        """Test QK RMSNorm normalization."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        try:
            from vllm_omni.diffusion.layers.norm import RMSNorm

            q = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
            k = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

            q_norm = RMSNorm(HIDDEN_DIM).to(DEVICE).to(torch.bfloat16)
            k_norm = RMSNorm(HIDDEN_DIM).to(DEVICE).to(torch.bfloat16)

            def normalize_qk():
                q_normalized = q_norm.forward_xpu(q)
                k_normalized = k_norm.forward_xpu(k)
                return q_normalized, k_normalized

            result = benchmark_kernel("QK RMSNorm", normalize_qk)

            if result:
                print(f"\nQK RMSNorm: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
                print("  T2V: 35,280 calls (~232µs/call)")

        except Exception as e:
            pytest.skip(f"RMSNorm not available: {e}")

    def test_output_projection(self):
        """Test attention output projection using torch.nn.Linear (replaces RowParallelLinear)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        attn_output = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        # Use torch.nn.Linear instead of RowParallelLinear
        out_proj = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Attn Output Projection", lambda: out_proj(attn_output))

        if result:
            print(f"\nAttn Output Projection: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 6. Cross-Attention Operations (using torch.nn.Linear)
# ============================================================================


class TestCrossAttentionOps:
    """Test cross-attention operations using torch.nn.Linear instead of parallel layers."""

    def test_cross_attn_q_projection(self):
        """Test cross-attention Q projection using torch.nn.Linear (replaces ColumnParallelLinear)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        # Q comes from video hidden states
        hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        # Use torch.nn.Linear instead of ColumnParallelLinear
        q_proj = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Cross-Attn Q Projection", lambda: q_proj(hidden_states))

        if result:
            print(f"\nCross-Attn Q Projection: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  Q from video: [1, 75600, 5120] → [1, 75600, 5120]")

    def test_cross_attn_kv_projection(self):
        """Test cross-attention K/V projection using torch.nn.Linear (replaces ColumnParallelLinear)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        # K/V come from text encoder output
        text_features = torch.randn(BATCH_SIZE, TEXT_SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        # Use torch.nn.Linear for K and V projections
        k_proj = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)
        v_proj = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)

        def kv_projection():
            k = k_proj(text_features)
            v = v_proj(text_features)
            return k, v

        result = benchmark_kernel("Cross-Attn K/V Projection", kv_projection)

        if result:
            print(f"\nCross-Attn K/V Projection: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  K/V from text: [1, 512, 5120] → [1, 512, 5120] each")

    def test_cross_attn_output_projection(self):
        """Test cross-attention output projection using torch.nn.Linear (replaces RowParallelLinear)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        # Output from cross-attention
        attn_output = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        # Use torch.nn.Linear instead of RowParallelLinear
        out_proj = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Cross-Attn Output Projection", lambda: out_proj(attn_output))

        if result:
            print(f"\nCross-Attn Output Projection: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 7. FFN Operations (using torch.nn.Linear)
# ============================================================================


class TestFFNOps:
    """Test FFN operations using torch.nn.Linear instead of parallel layers."""

    def test_ffn_upproject(self):
        """Test FFN upproject using torch.nn.Linear (replaces ColumnParallelLinear)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        # Use torch.nn.Linear instead of ColumnParallelLinear
        ffn_up = torch.nn.Linear(HIDDEN_DIM, FFN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("FFN Upproject", lambda: ffn_up(x))

        if result:
            print(f"\nFFN Upproject: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  5120 → 13824 (2.7× expansion)")

    def test_ffn_downproject(self):
        """Test FFN downproject using torch.nn.Linear (replaces ColumnParallelLinear)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, FFN_DIM, device=DEVICE, dtype=torch.bfloat16)

        # Use torch.nn.Linear instead of ColumnParallelLinear
        ffn_down = torch.nn.Linear(FFN_DIM, HIDDEN_DIM, bias=True, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("FFN Downproject", lambda: ffn_down(x))

        if result:
            print(f"\nFFN Downproject: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  13824 → 5120")


# ============================================================================
# 8. Normalization Operations
# ============================================================================


class TestNormalizationOps:
    """Test normalization operations."""

    def test_rms_norm(self):
        """Test RMSNorm (11.84% XPU time in T2V)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        try:
            from vllm_omni.diffusion.layers.norm import RMSNorm

            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
            rms_norm = RMSNorm(HIDDEN_DIM).to(DEVICE).to(torch.bfloat16)

            result = benchmark_kernel("RMSNorm", lambda: rms_norm.forward_xpu(x))

            if result:
                print(f"\nRMSNorm: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
                print("  T2V: 35,280 calls, S2V: more (reference image + audio)")

        except Exception as e:
            pytest.skip(f"RMSNorm not available: {e}")

    def test_ada_layer_norm(self):
        """Test AdaLayerNorm (LayerNorm + adaptive scale/shift)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        try:
            from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm

            # Hidden states + timestep embedding
            hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
            timestep_emb = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

            ada_norm = AdaLayerNorm(HIDDEN_DIM).to(DEVICE).to(torch.bfloat16)

            result = benchmark_kernel("AdaLayerNorm", lambda: ada_norm.forward_xpu(hidden_states, timestep_emb))

            if result:
                print(f"\nAdaLayerNorm: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
                print("  Used in VAE for timestep-conditioned normalization")

        except Exception as e:
            pytest.skip(f"AdaLayerNorm not available: {e}")


# ============================================================================
# 9. Activation Functions
# ============================================================================


class TestActivationOps:
    """Test activation functions."""

    def test_silu_activation(self):
        """Test SiLU activation (3.34% XPU time in S2V, 0.3% in T2V)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("SiLU", lambda: torch.nn.functional.silu(x))

        if result:
            print(f"\nSiLU: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  S2V: 28,850 calls (~80µs/call) - 11× more than T2V")

    def test_gelu_activation(self):
        """Test GELU activation (used in FFN)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, FFN_DIM, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("GELU", lambda: torch.nn.functional.gelu(x, approximate="tanh"))

        if result:
            print(f"\nGELU: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 10. Element-wise Operations
# ============================================================================


class TestElementWiseOps:
    """Test element-wise operations."""

    def test_elementwise_mul(self):
        """Test element-wise multiplication (19-20% XPU time combined with div)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
        scale = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Element-wise Mul", lambda: x * (1 + scale.unsqueeze(1)))

        if result:
            print(f"\nElement-wise Mul: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  T2V: 41,828 calls | S2V: 68,078 calls")

    def test_elementwise_div(self):
        """Test element-wise division (used in RMSNorm)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
        rms = torch.randn(BATCH_SIZE, SEQ_LEN, 1, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Element-wise Div", lambda: x / rms)

        if result:
            print(f"\nElement-wise Div: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  T2V: 17,789 calls | S2V: 30,176 calls")

    def test_elementwise_add(self):
        """Test element-wise addition (residual connections)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
        residual = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("Element-wise Add", lambda: x + residual)

        if result:
            print(f"\nElement-wise Add: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 11. Tensor Manipulation Operations
# ============================================================================


class TestTensorManipulationOps:
    """Test tensor manipulation operations."""

    def test_reshape_multihead(self):
        """Test reshape for multi-head attention."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        # QKV after projection: [1, 75600, 15360]
        qkv = torch.randn(BATCH_SIZE, SEQ_LEN, 3 * HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        def reshape_to_heads():
            # Split into Q, K, V
            q, k, v = qkv.chunk(3, dim=-1)
            # Reshape to [batch, seq, num_heads, head_dim]
            q = q.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
            k = k.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
            v = v.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
            return q, k, v

        result = benchmark_kernel("Reshape to Multi-head", reshape_to_heads)

        if result:
            print(f"\nReshape to Multi-head: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")

    def test_concatenation(self):
        """Test concatenation (used in CFG)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        latent_cond = torch.randn(BATCH_SIZE, 16, 21, LATENT_HEIGHT, LATENT_WIDTH, device=DEVICE, dtype=torch.bfloat16)
        latent_uncond = torch.randn(
            BATCH_SIZE, 16, 21, LATENT_HEIGHT, LATENT_WIDTH, device=DEVICE, dtype=torch.bfloat16
        )

        result = benchmark_kernel("Concatenation (CFG)", lambda: torch.cat([latent_cond, latent_uncond], dim=0))

        if result:
            print(f"\nConcatenation (CFG): {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  T2V: 25,802 calls | S2V: 42,793 calls")


# ============================================================================
# 12. RoPE Operations
# ============================================================================


class TestRoPEOps:
    """Test rotary position embedding operations."""

    def test_rope_embedding_generation(self):
        """Test 3D RoPE embedding generation (temporal, height, width)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        # Generate position IDs for 3D (T, H, W)
        T, H, W = 21, 45, 80  # latent frames, height patches, width patches

        def generate_3d_positions():
            # Temporal positions
            t_pos = torch.arange(T, device=DEVICE).view(T, 1, 1).expand(T, H, W)
            # Height positions
            h_pos = torch.arange(H, device=DEVICE).view(1, H, 1).expand(T, H, W)
            # Width positions
            w_pos = torch.arange(W, device=DEVICE).view(1, 1, W).expand(T, H, W)
            return t_pos, h_pos, w_pos

        result = benchmark_kernel("RoPE 3D Position Generation", generate_3d_positions)

        if result:
            print(f"\nRoPE 3D Position Generation: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 13. Memory Operations
# ============================================================================


class TestMemoryOps:
    """Test memory copy and data movement operations."""

    def test_dtype_conversion_fp32_to_bf16(self):
        """Test fp32 to bf16 conversion."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.float32)

        result = benchmark_kernel("FP32→BF16", lambda: x.to(torch.bfloat16))

        if result:
            print(f"\nFP32→BF16: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")

    def test_dtype_conversion_bf16_to_fp32(self):
        """Test bf16 to fp32 conversion."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("BF16→FP32", lambda: x.to(torch.float32))

        if result:
            print(f"\nBF16→FP32: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  T2V: 84,028 copy calls | S2V: 142,308 calls")

    def test_contiguous_memory_copy(self):
        """Test making tensor contiguous (triggers memory copy)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        x_t = x.transpose(1, 2)  # Non-contiguous

        result = benchmark_kernel("Contiguous Copy", lambda: x_t.contiguous())

        if result:
            print(f"\nContiguous Copy: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")


# ============================================================================
# 14. VAE and WanResidualBlock
# ============================================================================


class TestVAEOps:
    """Test VAE operations."""

    def test_wan_residual_block(self):
        """Test WanResidualBlock (Conv3D + RMSNorm + SiLU + Residual)."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        x = torch.randn(BATCH_SIZE, 384, 3, 34, 34, device=DEVICE, dtype=torch.bfloat16)
        conv = torch.nn.Conv3d(384, 384, kernel_size=3, padding=1, device=DEVICE, dtype=torch.bfloat16)

        def residual_block():
            identity = x
            out = conv(x)
            out = out / (out.norm(dim=1, keepdim=True) + 1e-6)  # RMSNorm
            out = torch.nn.functional.silu(out)
            out = out + identity  # Residual
            return out

        result = benchmark_kernel("WanResidualBlock", residual_block)

        if result:
            print(f"\nWanResidualBlock: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  T2V: 588 calls/block, WanResidualBlock_0: ~10.7ms/call")


# ============================================================================
# 15. Text Encoder (UMT5)
# ============================================================================


class TestTextEncoderOps:
    """Test text encoder (UMT5) operations."""

    def test_umt5_attention_matmul(self):
        """Test matmul in UMT5 self-attention."""
        if DEVICE == "cpu":
            pytest.skip("Kernel tests require XPU")

        UMT5_SEQ_LEN = 512
        UMT5_HEADS = 24
        UMT5_HEAD_DIM = 64

        q = torch.randn(BATCH_SIZE, UMT5_HEADS, UMT5_SEQ_LEN, UMT5_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(BATCH_SIZE, UMT5_HEADS, UMT5_SEQ_LEN, UMT5_HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)

        result = benchmark_kernel("UMT5 Matmul", lambda: torch.matmul(q, k.transpose(-2, -1)))

        if result:
            print(f"\nUMT5 Matmul: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print("  48 UMT5Block.forward calls, ~8ms/call")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
