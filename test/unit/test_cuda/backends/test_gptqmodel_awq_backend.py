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

"""Tests for the GPTQModel ``awq_torch`` backend (low-priority, GPU-only kernel path)."""

import shutil
from test.helpers import evaluate_accuracy, generate_prompt, get_model_path, model_infer

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_gptqmodel


class TestGptqmodelAwqTorchBackend:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_gptqmodel
    @pytest.mark.skip_ci(reason="Not necessary to test low priority backend in CI")
    def test_gptqmodel_awq_torch_4bits_group_size_16(self, dataloader):
        """AWQ-quantized (group_size=16) model loads via the gptqmodel:awq_torch backend and meets accuracy."""
        model_path = get_model_path("facebook/opt-125m")
        autoround = AutoRound(model_path, bits=4, group_size=16, sym=True, iters=0, disable_opt_rtn=True)
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round:auto_awq")

        quantization_config = AutoRoundConfig(backend="gptqmodel:awq_torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)

        output = model_infer(model, tokenizer)
        assert isinstance(output, str) and len(output.strip()) > 0, "Model failed to generate non-empty output"
        generated = generate_prompt(model, tokenizer, "There is a girl who likes adventure,")
        assert len(generated) > len("There is a girl who likes adventure,"), "Generation did not produce new tokens"
        evaluate_accuracy(model, tokenizer, threshold=0.2, batch_size=16)
        torch.cuda.empty_cache()
