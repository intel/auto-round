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

import shutil

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from auto_round import AutoRound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestFp8SchemesGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def _assert_fp8_weight(self, compressed_model):
        found_fp8 = False
        for module in compressed_model.modules():
            w = getattr(module, "weight", None)
            if isinstance(w, torch.nn.Parameter) and w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                found_fp8 = True
                break
        assert found_fp8, "No fp8 weight found in the compressed model"

    @pytest.mark.timeout(180)
    def test_fp8_block_export(self, tiny_qwen_model_path):
        autoround = AutoRound(tiny_qwen_model_path, scheme="FP8_BLOCK", iters=0, disable_opt_rtn=True, seqlen=2)
        compressed_model, _ = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._assert_fp8_weight(compressed_model)

    @pytest.mark.timeout(180)
    def test_fp8_static_export(self, tiny_qwen_model_path):
        autoround = AutoRound(tiny_qwen_model_path, scheme="FP8_STATIC", iters=0, disable_opt_rtn=True, seqlen=2)
        compressed_model, quantized_model_path = autoround.quantize_and_save(
            output_dir=self.save_dir, format="auto_round"
        )
        self._assert_fp8_weight(compressed_model)
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.timeout(180)
    def test_fp8_block_forward(self, tiny_qwen_model_path):
        """Reloaded FP8_BLOCK model must run a forward pass on GPU."""
        autoround = AutoRound(tiny_qwen_model_path, scheme="FP8_BLOCK", iters=0, disable_opt_rtn=True, seqlen=2)
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        config = AutoConfig.from_pretrained(quantized_model_path, trust_remote_code=True)
        config.quantization_config = None
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            config=config,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        input_ids = torch.randint(0, 1000, (1, 8), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    def test_fp8_plain_export(self, tiny_qwen_model_path):
        """Weight-only FP8 (FPW8A16, per-channel scale) export."""
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme="FPW8A16",
            iters=0,
            disable_opt_rtn=True,
            seqlen=2,
        )
        compressed_model, _ = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._assert_fp8_weight(compressed_model)
