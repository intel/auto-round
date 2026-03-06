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

from ...helpers import check_version, qwen2_5_omni_name_or_path

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        not check_version("transformers>=5.1.0"),
        reason="Qwen-Omni models require transformers >= 5.1.0",
    ),
]

# ---------------------------------------------------------------------------
# Fixture: tiny Qwen3-Omni-MoE
# Priority: use real config from qwen3_omni_name_or_path (skipped if absent);
# fall back to fully synthetic config using qwen_name_or_path tokenizer.
# ---------------------------------------------------------------------------
@pytest.fixture
def setup_qwen3_omni_moe(tiny_qwen3_omni_moe):
    """Create a tiny Qwen3-Omni-MoE model.

    Uses the session-scoped ``tiny_qwen3_omni_moe`` fixture which loads the
    real tokenizer/processor from ``qwen3_omni_name_or_path`` and builds a
    model with reduced layers and random weights.
    """
    model, tokenizer, processor = tiny_qwen3_omni_moe
    output_dir = "./tmp/test_quantized_qwen3_omni_moe"
    return model, tokenizer, processor, output_dir, model.config


# ========================= Qwen2.5-Omni Integration Tests ==================


class TestQwen2_5OmniQuantization:
    """End-to-end quantization test for Qwen2.5-Omni (dense model)."""

    def test_quantize_and_reload(self, tiny_qwen2_5_omni):
        """Quantize, save, reload, verify weights, and run inference."""
        model, tokenizer, processor = tiny_qwen2_5_omni
        output_dir = "./tmp/test_quantized_qwen2_5_omni"

        # Quantize
        autoround = AutoRound(
            model,
            tokenizer,
            processor=processor,
            nsamples=2,
            iters=1,
            seqlen=32,
            ignore_layers="self_attn,lm_head",
        )
        quantized_model, save_folder = autoround.quantize_and_save(
            format="auto_round", output_dir=output_dir
        )
        assert quantized_model is not None, "Quantized model should not be None"

        # Copy model-specific files required for from_pretrained (e.g. spk_dict.pt for token2wav)
        for extra_file in ["spk_dict.pt"]:
            src = os.path.join(qwen2_5_omni_name_or_path, extra_file)
            if os.path.exists(src):
                shutil.copy2(src, output_dir)

        # Reload
        loaded_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(output_dir)
        loaded_model.to("cuda")

        # Run inference on thinker
        inp = torch.randint(0, 100, (1, 64)).to("cuda")
        with torch.inference_mode():
            output = loaded_model.thinker(input_ids=inp)
        assert output is not None, "Inference failed on reloaded model"

        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)


# ========================= Qwen3-Omni-MoE Integration Tests ================


class TestQwen3OmniMoeQuantization:
    """End-to-end quantization test for Qwen3-Omni-MoE."""

    def test_quantize_and_reload(self, setup_qwen3_omni_moe):
        """Quantize, save, reload, verify weights, and run inference."""
        model, tokenizer, processor, output_dir, config = setup_qwen3_omni_moe

        # Quantize
        autoround = AutoRound(
            model,
            tokenizer,
            processor=processor,
            nsamples=2,
            iters=1,
            seqlen=32,
            ignore_layers="self_attn,lm_head,mlp.gate",
        )
        quantized_model, save_folder = autoround.quantize_and_save(
            format="auto_round", output_dir=output_dir
        )
        assert quantized_model is not None, "Quantized model should not be None"

        # Reload
        loaded_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(output_dir)
        loaded_model.to("cuda")

        # Run inference on thinker
        inp = torch.randint(0, 100, (1, 64)).to("cuda")
        with torch.inference_mode():
            output = loaded_model.thinker(input_ids=inp)
        assert output is not None, "Inference failed on reloaded model (thinker)"

        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_quantize_mxfp4(self, setup_qwen3_omni_moe):
        """Quantize with MXFP4 scheme and verify."""
        model, tokenizer, processor, output_dir, config = setup_qwen3_omni_moe

        autoround = AutoRound(
            model,
            tokenizer,
            processor=processor,
            scheme="MXFP4",
            nsamples=2,
            iters=1,
            seqlen=32,
            ignore_layers="self_attn,lm_head,mlp.gate",
        )
        quantized_model, save_folder = autoround.quantize_and_save(
            format="auto_round", output_dir=output_dir
        )
        assert quantized_model is not None, "MXFP4 quantized model should not be None"

        # Reload and inference
        loaded_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(output_dir)
        loaded_model.to("cuda")

        inp = torch.randint(0, 100, (1, 64)).to("cuda")
        with torch.inference_mode():
            output = loaded_model.thinker(input_ids=inp)
        assert output is not None

        shutil.rmtree(output_dir, ignore_errors=True)
