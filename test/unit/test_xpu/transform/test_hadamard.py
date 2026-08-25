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

"""XPU CI tests for the Hadamard rotation transform.

XPU counterpart of test/test_cuda/transform/test_mxfp4_transform.py. Covers:
- Direct ``apply_rotation`` (hadamard backend) on an XPU-resident model
- AutoRound pipeline with ``rotation_config="default"`` / ``"random_hadamard"``
  on XPU: quantize → save → load → generate
"""

import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms import apply_rotation, normalize_rotation_config

from test.helpers import generate_prompt

DEVICE = "xpu"

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU is not available on this host",
)


class TestHadamardApplyXPU:
    """Direct apply_rotation on an XPU-resident model."""

    def test_apply_rotation_forward(self, tiny_opt_model_path):
        """Hadamard rotation applied on XPU should produce valid logits."""
        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, dtype=torch.float32).to(DEVICE).eval()
        cfg = normalize_rotation_config("hadamard")
        model = apply_rotation(model, cfg)

        tokenizer = AutoTokenizer.from_pretrained(tiny_opt_model_path)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits

        assert logits.device.type == DEVICE
        assert not torch.isnan(logits).any(), "NaN logits after hadamard rotation on XPU"
        assert not torch.isinf(logits).any(), "Inf logits after hadamard rotation on XPU"
        assert logits.abs().sum() > 0, "All-zero logits after hadamard rotation on XPU"
        del model
        torch.xpu.empty_cache()


class TestHadamardPipelineXPU:
    """AutoRound pipeline with hadamard rotation on XPU (MXFP4 scheme)."""

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.parametrize("rotation", ["default", "random_hadamard"])
    def test_transform_mxfp4_quant_infer(self, tiny_opt_model_path, rotation):
        """MXFP4 + hadamard rotation: quantize → save → load on XPU → generate."""
        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            seqlen=8,
            nsamples=2,
            scheme="MXFP4",
            rotation_config=rotation,
            device_map=DEVICE,
        )
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, dtype="auto", device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        output = generate_prompt(model, tokenizer, device=DEVICE)
        assert len(output) > 0, "Quantized model should produce non-empty output"
        del model
        torch.xpu.empty_cache()
