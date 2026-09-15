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
from transformers import AutoModelForCausalLM

from auto_round import AutoRound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestMxfpNvfpExportGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4", "MXFP8", "MXFP4_RCEIL"])
    def test_export_reload_forward(self, tiny_qwen_model_path, scheme):
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme=scheme,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4"])
    def test_export_qlinear_modules(self, tiny_qwen_model_path, scheme):
        """The reloaded model must contain the expected MXFP/NVFP quant-linear module."""
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme=scheme,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        from auto_round.experimental import qmodules as ar_qmodules

        mapping = {
            "MXFP4": ar_qmodules.MXFP4QuantLinear,
            "NVFP4": ar_qmodules.NVFP4QuantLinear,
        }
        found = False
        for module in model.modules():
            if isinstance(module, mapping[scheme]):
                found = True
                break
        assert found, f"No {mapping[scheme].__name__} found in reloaded model"
