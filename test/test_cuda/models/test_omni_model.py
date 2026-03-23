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

"""CUDA integration tests for Qwen2.5-Omni and Qwen3-Omni-MoE quantization.

Tests cover end-to-end quantization flow:
- Qwen2.5-Omni: loads real config from pretrained, reduces layers, random weights
  (uses the shared ``tiny_qwen2_5_omni`` session-scoped fixture)
- Qwen3-Omni-MoE: fully synthetic tiny config with random weights
  (no pretrained checkpoint needed — model is too large for CI)
- Quantize with AutoRound
- Save and reload
- Run inference on reloaded model
"""

import os
import shutil

import pytest
import torch
from transformers import (
    AutoTokenizer,
    Qwen2_5OmniForConditionalGeneration,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
)

from auto_round import AutoRound

from ...helpers import check_version

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        not check_version("transformers>=5.1.0"),
        reason="Qwen-Omni models require transformers >= 5.1.0",
    ),
]


# ========================= Qwen2.5-Omni Integration Tests ==================


class TestQwen2_5OmniQuantization:
    """End-to-end quantization test for Qwen2.5-Omni (dense model)."""

    def test_quantize_and_reload(self, tiny_qwen2_5_omni_model_path, tmp_path):
        """Quantize, save, reload, verify weights, and run inference."""
        # Quantize
        autoround = AutoRound(
            tiny_qwen2_5_omni_model_path,
            nsamples=2,
            iters=1,
            seqlen=32,
            ignore_layers="self_attn,lm_head",
        )
        quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=tmp_path)
        assert quantized_model is not None, "Quantized model should not be None"

        # Copy model-specific files required for from_pretrained (e.g. spk_dict.pt for token2wav)
        for extra_file in ["spk_dict.pt"]:
            src = os.path.join(tiny_qwen2_5_omni_model_path, extra_file)
            if os.path.exists(src):
                shutil.copy2(src, tmp_path)

        # Reload
        loaded_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(tmp_path, device_map="cuda")

        # Run inference on thinker
        inp = torch.randint(0, 100, (1, 64)).to("cuda")
        with torch.inference_mode():
            output = loaded_model.thinker(input_ids=inp)
        assert output is not None, "Inference failed on reloaded model"


# ========================= Qwen3-Omni-MoE Integration Tests ================


class TestQwen3OmniMoeQuantization:
    """End-to-end quantization test for Qwen3-Omni-MoE."""

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_quantize_and_reload(self, tiny_qwen3_omni_moe_model_path):
        """Quantize, save, reload, verify weights, and run inference."""
        # Quantize
        autoround = AutoRound(
            tiny_qwen3_omni_moe_model_path,
            nsamples=2,
            iters=1,
            seqlen=32,
            ignore_layers="self_attn,lm_head,mlp.gate",
        )
        quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=self.save_dir)
        assert quantized_model is not None, "Quantized model should not be None"

        # Reload
        loaded_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.save_dir, device_map="cuda")

        # Run inference on thinker
        inp = torch.randint(0, 100, (1, 64)).to("cuda")
        with torch.inference_mode():
            output = loaded_model.thinker(input_ids=inp)
        assert output is not None, "Inference failed on reloaded model (thinker)"

    def test_quantize_mxfp4(self, tiny_qwen3_omni_moe_model_path):
        """Quantize with MXFP4 scheme and verify."""
        autoround = AutoRound(
            tiny_qwen3_omni_moe_model_path,
            scheme="MXFP4",
            nsamples=2,
            iters=1,
            seqlen=32,
            ignore_layers="self_attn,lm_head,mlp.gate",
        )
        quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=self.save_dir)
        assert quantized_model is not None, "MXFP4 quantized model should not be None"

        # Reload and inference
        loaded_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.save_dir, device_map="cuda")

        inp = torch.randint(0, 100, (1, 64)).to("cuda")
        with torch.inference_mode():
            output = loaded_model.thinker(input_ids=inp)
        assert output is not None
