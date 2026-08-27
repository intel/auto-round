# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from auto_round.export.svdquant_adapters import (
    SDXL_SVDQUANT_TARGET_MODULES,
    detect_svdquant_model_adapter,
)


class ConfiguredModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def test_detects_sdxl_unet_from_runtime_relevant_config():
    model = ConfiguredModel(
        {
            "_class_name": "UNet2DConditionModel",
            "addition_embed_type": "text_time",
            "cross_attention_dim": 2048,
            "projection_class_embeddings_input_dim": 2816,
        }
    )

    assert detect_svdquant_model_adapter(model) == "sdxl"


def test_does_not_treat_stable_diffusion_v1_unet_as_sdxl():
    model = ConfiguredModel(
        {
            "_class_name": "UNet2DConditionModel",
            "cross_attention_dim": 768,
        }
    )

    assert detect_svdquant_model_adapter(model) == "identity"


def test_sdxl_target_allowlist_matches_nunchaku_patched_linears():
    assert set(SDXL_SVDQUANT_TARGET_MODULES) == {
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }

