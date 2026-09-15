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


class TestNvfpAutoroundExport:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(180)
    def test_nvfp4_autoround_export(self, tiny_opt_model_path):
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme="NVFP4",
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        # Reload the saved model and confirm the NVFP4 quantized weight
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        qcfg = model.config.quantization_config
        assert getattr(qcfg, "quant_method", None) == "auto-round"
