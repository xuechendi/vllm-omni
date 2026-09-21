# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanModel,
)


def test_split_qkv_weight_supports_block_scales():
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=128,
        )
    )
    rows = (32 + 2 * 8) * 128

    weight = torch.empty(rows, 4096)
    scale = torch.empty(rows, 128, dtype=torch.uint8)

    assert HunyuanModel._split_qkv_weight(model, weight).shape == weight.shape
    assert HunyuanModel._split_qkv_weight(model, scale).shape == scale.shape
