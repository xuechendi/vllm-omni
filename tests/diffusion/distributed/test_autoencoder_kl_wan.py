# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock

from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_wan as wan_vae_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyOmniAutoencoderKLWan(OmniAutoencoderKLWan):
    def __init__(self, *, dtype: torch.dtype):
        torch.nn.Module.__init__(self)
        self.register_parameter("dummy_weight", torch.nn.Parameter(torch.ones(1, dtype=dtype)))


def test_wan_vae_execution_context_handles_fp32():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.float32)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.float32


def test_wan_vae_execution_context_handles_bf16():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.bfloat16


def test_wan_vae_execution_context_uses_platform_autocast(mocker):
    sentinel = object()
    platform = mocker.Mock()
    platform.create_autocast_context.return_value = sentinel
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)

    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)

    assert model._execution_context() is sentinel
    platform.create_autocast_context.assert_called_once_with(
        device_type=model.dummy_weight.device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )


def _tiny_wan_vae(cls=OmniAutoencoderKLWan):
    torch.manual_seed(0)
    model = cls(base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, attn_scales=[], temperal_downsample=[True])
    for module in model.modules():
        if isinstance(module, wan_vae_module.WanRMS_norm):
            module.gamma.data.uniform_(0.5, 1.5)
    return model.eval()


@pytest.mark.parametrize(("env", "expected"), [(None, True), ("1", True), ("0", False)])
def test_wan_vae_compile_is_on_by_default(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv(wan_vae_module.COMPILE_ENV, raising=False)
    else:
        monkeypatch.setenv(wan_vae_module.COMPILE_ENV, env)
    model = _tiny_wan_vae()
    model._install_vae_compile()
    assert any(getattr(m, "_vllm_omni_compiled", False) for m in model.modules()) is expected


def test_wan_vae_compiled_norms_match_upstream(monkeypatch):
    """The fused norm+SiLU path reproduces upstream encode and decode.

    The compile itself is replaced by the eager function it wraps, so this checks the install
    logic and the math (which norms absorb the SiLU, which nonlinearities become identities),
    not inductor.
    """
    monkeypatch.setenv(wan_vae_module.COMPILE_ENV, "1")
    monkeypatch.setattr(wan_vae_module, "_get_compiled_rms_norm_act", lambda: wan_vae_module._rms_norm_act)
    reference = _tiny_wan_vae()
    model = _tiny_wan_vae()
    model.load_state_dict(reference.state_dict())

    x = torch.randn(1, 3, 5, 16, 16)
    with torch.no_grad():
        ref_latent = reference.encode(x).latent_dist.mode()
        ref_video = reference.decode(ref_latent).sample
        model._install_vae_compile()
        latent = model.encode(x).latent_dist.mode()
        video = model.decode(ref_latent).sample

    norms = [m for m in model.modules() if isinstance(m, wan_vae_module.WanRMS_norm)]
    assert all(getattr(m, "_vllm_omni_compiled", False) for m in norms)
    # Only the attention block's norm has no SiLU after it.
    attn_norms = {id(m.norm) for m in model.modules() if isinstance(m, WanAttentionBlock)}
    assert attn_norms and all(m._vllm_omni_norm_silu == (id(m) not in attn_norms) for m in norms)
    blocks = [m for m in model.modules() if isinstance(m, wan_vae_module.WanResidualBlock)]
    assert blocks and all(isinstance(b.nonlinearity, torch.nn.Identity) for b in blocks)
    torch.testing.assert_close(latent, ref_latent, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(video, ref_video, rtol=1e-4, atol=1e-5)


def test_wan_vae_compile_leaves_fast_path_decoder_alone(monkeypatch):
    """The decoder fast path calls the norms through ``_norm_act``, not ``norm.forward``.

    Compiling its norms would move the SiLU into a forward that is never called while the
    block's ``nonlinearity`` became an identity, dropping the activation. The compile must
    skip that component and still cover the encoder, which the fast path does not own here.
    """
    from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import install_wan_vae_fastpath

    monkeypatch.setenv(wan_vae_module.COMPILE_ENV, "1")
    monkeypatch.setattr(wan_vae_module, "_get_compiled_rms_norm_act", lambda: wan_vae_module._rms_norm_act)
    reference = _tiny_wan_vae()
    model = _tiny_wan_vae()
    model.load_state_dict(reference.state_dict())
    assert install_wan_vae_fastpath(model, level="lossless").installed

    x = torch.randn(1, 3, 5, 16, 16)
    with torch.no_grad():
        ref_latent = reference.encode(x).latent_dist.mode()
        ref_video = reference.decode(ref_latent).sample
        latent = model.encode(x).latent_dist.mode()
        video = model.decode(ref_latent).sample

    assert not any(getattr(m, "_vllm_omni_compiled", False) for m in model.decoder.modules())
    assert any(getattr(m, "_vllm_omni_compiled", False) for m in model.encoder.modules())
    decoder_blocks = [m for m in model.decoder.modules() if isinstance(m, wan_vae_module.WanResidualBlock)]
    assert decoder_blocks and all(isinstance(b.nonlinearity, torch.nn.SiLU) for b in decoder_blocks)
    torch.testing.assert_close(latent, ref_latent, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(video, ref_video, rtol=1e-4, atol=1e-5)
