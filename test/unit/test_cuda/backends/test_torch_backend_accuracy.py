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

"""Accuracy regression checks for the ``torch`` inference backend.

These use lm_eval on a real (non-tiny) model, so they're slow and only run on
one representative device (cuda) rather than every accelerator -- functional
backend-loading coverage across all devices lives in
``common/backends/test_torch_backend.py``.
"""

import shutil
from test.helpers import evaluate_accuracy, get_model_path, model_infer

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound


class TestTorchBackendAccuracy:
    @classmethod
    def setup_class(cls):
        cls.model_name = get_model_path("facebook/opt-125m")

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.mark.timeout(90)
    def test_torch_backend_accuracy_asym_rtn(self, dataloader):
        """RTN-quantized 4-bit asym model meets the accuracy threshold in both fp16 and bf16."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        autoround = AutoRound(
            model, tokenizer, bits=4, group_size=128, sym=False, iters=0, seqlen=2, dataset=dataloader
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round:gptqmodel")

        quantization_config = AutoRoundConfig(backend="torch")
        for dtype in (torch.float16, torch.bfloat16):
            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, dtype=dtype, device_map="cuda", quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            model_infer(model, tokenizer)
            evaluate_accuracy(model, tokenizer, threshold=0.35, batch_size=16, limit=10)
            torch.cuda.empty_cache()

    @pytest.mark.timeout(90)
    def test_torch_backend_accuracy_sym_rtn(self, dataloader):
        """RTN-quantized 4-bit sym model meets the accuracy threshold."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=32, sym=True, iters=0, seqlen=2, dataset=dataloader)
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=torch.float16, device_map="cuda", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.28, batch_size=32, limit=1000)
        torch.cuda.empty_cache()

    @pytest.mark.timeout(90)
    def test_torch_backend_accuracy_asym_tuning(self, dataloader):
        """Tuned (iters=1) 4-bit asym model meets the accuracy threshold."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        autoround = AutoRound(
            model, tokenizer, bits=4, group_size=128, sym=False, iters=1, seqlen=2, dataset=dataloader
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round:gptqmodel")

        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=torch.float16, device_map="cuda", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.35, batch_size=16)
        torch.cuda.empty_cache()

    @pytest.mark.skip_ci(reason="Accuracy: Time-consuming; covered by the RTN accuracy check above")
    def test_torch_backend_accuracy_sym_tuning(self, dataloader):
        """Tuned (iters=1) 4-bit sym model meets the accuracy threshold."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        autoround = AutoRound(model, tokenizer, bits=4, group_size=128, sym=True, iters=1, seqlen=2, dataset=dataloader)
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=torch.float16, device_map="cuda", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.28, batch_size=16)
        torch.cuda.empty_cache()
