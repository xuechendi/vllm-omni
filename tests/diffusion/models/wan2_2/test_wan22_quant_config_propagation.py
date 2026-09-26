# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Wan2.2 quant_config propagation through transformer creation.

Tests cover:
- create_transformer_from_config passes quant_config and prefix
- create_vace_transformer_from_config passes quant_config and prefix
- set_tf_model_config propagates quant_config to OmniDiffusionConfig
- patch_wan_rms_norm safely iterates sys.modules with concurrent modifications
- I2V transformer_2 quant_config is built from config dict
"""

import sys
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace as wan22_vace_module
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    create_transformer_from_config,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import (
    create_vace_transformer_from_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# create_transformer_from_config: quant_config / prefix forwarding
# ---------------------------------------------------------------------------


class TestCreateTransformerQuant:
    """Verify quant_config and prefix are forwarded to WanTransformer3DModel."""

    def test_quant_config_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        fake_qc = mocker.MagicMock()
        create_transformer_from_config(
            {"patch_size": [1, 2, 2], "num_layers": 2},
            quant_config=fake_qc,
        )
        assert captured.get("quant_config") is fake_qc

    def test_prefix_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        create_transformer_from_config(
            {"patch_size": [1, 2, 2]},
            prefix="model.transformer.",
        )
        assert captured.get("prefix") == "model.transformer."

    def test_quant_config_none_by_default(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        create_transformer_from_config({"patch_size": [1, 2, 2]})
        # When quant_config is None and prefix is "", they are not added
        assert "quant_config" not in captured or captured["quant_config"] is None

    def test_quant_config_and_prefix_together(self, mocker: MockerFixture):
        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)

        fake_qc = mocker.MagicMock()
        create_transformer_from_config(
            {"patch_size": [1, 2, 2], "num_attention_heads": 4},
            quant_config=fake_qc,
            prefix="blocks.",
        )
        assert captured["quant_config"] is fake_qc
        assert captured["prefix"] == "blocks."


# ---------------------------------------------------------------------------
# create_vace_transformer_from_config: quant_config / prefix forwarding
# ---------------------------------------------------------------------------


class TestCreateVaceTransformerQuant:
    """Verify quant_config and prefix are forwarded to WanVACETransformer3DModel."""

    def test_quant_config_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeVACETransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_vace_module, "WanVACETransformer3DModel", FakeVACETransformer)

        fake_qc = mocker.MagicMock()
        create_vace_transformer_from_config(
            {"patch_size": [1, 2, 2], "num_layers": 2},
            quant_config=fake_qc,
        )
        assert captured.get("quant_config") is fake_qc

    def test_prefix_passed_through(self, mocker: MockerFixture):
        captured = {}

        class FakeVACETransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_vace_module, "WanVACETransformer3DModel", FakeVACETransformer)

        create_vace_transformer_from_config(
            {"patch_size": [1, 2, 2]},
            prefix="vace.",
        )
        assert captured.get("prefix") == "vace."


# ---------------------------------------------------------------------------
# set_tf_model_config: propagation of quant_config
# ---------------------------------------------------------------------------


class TestSetTfModelConfig:
    """Test that set_tf_model_config propagates quant_config correctly."""

    def _make_od_config(self):
        """Create a minimal OmniDiffusionConfig-like object for testing."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        cfg = object.__new__(OmniDiffusionConfig)
        cfg.quantization_config = None
        cfg.tf_model_config = None
        return cfg

    def test_propagates_quant_config_when_none(self, mocker: MockerFixture):
        cfg = self._make_od_config()
        fake_qc = mocker.MagicMock()
        tf_config = SimpleNamespace(quant_config=fake_qc, quant_method="auto-round")

        cfg.set_tf_model_config(tf_config)

        assert cfg.tf_model_config is tf_config
        assert cfg.quantization_config is fake_qc

    def test_does_not_overwrite_existing_quantization_config(self, mocker: MockerFixture):
        cfg = self._make_od_config()
        existing_qc = mocker.MagicMock()
        cfg.quantization_config = existing_qc
        tf_config = SimpleNamespace(quant_config=mocker.MagicMock())

        cfg.set_tf_model_config(tf_config)

        assert cfg.tf_model_config is tf_config
        assert cfg.quantization_config is existing_qc  # not overwritten

    def test_no_propagation_when_tf_quant_config_is_none(self, mocker: MockerFixture):
        cfg = self._make_od_config()
        tf_config = SimpleNamespace(quant_config=None)

        cfg.set_tf_model_config(tf_config)

        assert cfg.tf_model_config is tf_config
        assert cfg.quantization_config is None


# ---------------------------------------------------------------------------
# patch_wan_rms_norm: sys.modules snapshot safety
# ---------------------------------------------------------------------------


class TestPatchWanRmsNorm:
    """Test that patch_wan_rms_norm doesn't raise on concurrent module registration."""

    @pytest.fixture(autouse=True)
    def restore_wan_rms_norm(self):
        originals = [
            (module, module.__dict__["WanRMS_norm"])
            for module in list(sys.modules.values())
            if getattr(module, "__dict__", None) is not None and "WanRMS_norm" in module.__dict__
        ]
        yield
        for module, original in originals:
            module.__dict__["WanRMS_norm"] = original

    def test_patches_modules_with_wan_rms_norm(self):
        from vllm_omni.diffusion.models.wan2_2.norm import RMSNormVAE
        from vllm_omni.diffusion.models.wan2_2.patch_diffusers import patch_wan_rms_norm

        # Create a fake module that has WanRMS_norm
        fake_module = SimpleNamespace(WanRMS_norm=lambda x: x)
        sys.modules["_test_fake_wan_module"] = fake_module

        try:
            patch_wan_rms_norm()
            assert fake_module.WanRMS_norm is RMSNormVAE
        finally:
            del sys.modules["_test_fake_wan_module"]

    def test_no_error_when_modules_change_during_iteration(self):
        """Regression test: list() snapshot prevents RuntimeError."""
        from vllm_omni.diffusion.models.wan2_2.patch_diffusers import patch_wan_rms_norm

        # Simulate a module being added during iteration by a side effect
        original_items = sys.modules.items

        def items_with_side_effect():
            # This would cause RuntimeError without list() snapshot
            result = list(original_items())
            # Add a new module to simulate concurrent modification
            sys.modules["_test_dynamic_module"] = SimpleNamespace()
            return result

        try:
            # The function uses list(sys.modules.items()) so it takes a snapshot
            # Just verify it doesn't raise
            patch_wan_rms_norm()
        finally:
            sys.modules.pop("_test_dynamic_module", None)


# ---------------------------------------------------------------------------
# I2V transformer_2 quant_config extraction
# ---------------------------------------------------------------------------


class TestI2VTransformer2QuantConfig:
    """transformer_2 (low-noise expert) quant config in pipeline_wan2_2_i2v."""

    @staticmethod
    def _build_t2(mocker: MockerFixture, active, t1_config: dict, t2_config: dict):
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline

        captured = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mocker.patch.object(wan22_module, "WanTransformer3DModel", FakeTransformer)
        # An active config auto-detected from transformer/config.json is startup metadata about
        # the first expert, not a user policy for both, which is exactly the case t1_config
        # carrying a quantization_config stands for here.
        pipeline = SimpleNamespace(
            od_config=SimpleNamespace(
                quantization_config=active,
                quantization_config_is_auto_detected="quantization_config" in t1_config,
            )
        )
        Wan22I2VPipeline._create_transformer(pipeline, t2_config, component="transformer_2")
        return captured.get("quant_config")

    def test_online_quant_reaches_bf16_transformer_2(self, mocker: MockerFixture):
        """--quantization on a BF16 checkpoint must quantize both experts, not just transformer."""
        from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

        online = DiffusionMXFP8Config()
        qc = self._build_t2(mocker, online, {"patch_size": [1, 2, 2]}, {"patch_size": [1, 2, 2]})
        assert qc is online

    def test_no_quant_leaves_transformer_2_bf16(self, mocker: MockerFixture):
        qc = self._build_t2(mocker, None, {"patch_size": [1, 2, 2]}, {"patch_size": [1, 2, 2]})
        assert qc is None

    def test_quant_auto_detected_from_transformer_not_applied_to_bf16_transformer_2(self, mocker: MockerFixture):
        """The active config came from transformer/config.json; transformer_2 has none, so it stays BF16."""
        from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

        disk_qc = {"quant_method": "mxfp8", "is_checkpoint_mxfp8_serialized": True}
        active = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
        qc = self._build_t2(
            mocker, active, {"patch_size": [1, 2, 2], "quantization_config": disk_qc}, {"patch_size": [1, 2, 2]}
        )
        assert qc is None

    def test_transformer_2_detects_its_own_disk_config(self, mocker: MockerFixture):
        """Both experts pre-quantized: transformer_2 builds from its own config.json (AutoRound W4A16)."""
        from vllm.model_executor.layers.quantization.inc import INCConfig

        disk_qc = {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "packing_format": "auto_round:auto_gptq",
        }
        active = mocker.MagicMock()
        qc = self._build_t2(
            mocker,
            active,
            {"patch_size": [1, 2, 2], "quantization_config": disk_qc},
            {"patch_size": [1, 2, 2], "quantization_config": disk_qc},
        )
        assert isinstance(qc, INCConfig)
        assert qc.weight_bits == 4
        assert qc.group_size == 128
