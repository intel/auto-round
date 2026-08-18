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


class TestSchemePresetsGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("scheme", ["W4A16", "W2A16", "W8A16", "W3A16", "INT8"])
    def test_scheme_preset_quantize_forward(self, tiny_opt_model_path, scheme):
        autoround = AutoRound(tiny_opt_model_path, scheme=scheme, iters=0, disable_opt_rtn=True, nsamples=1, seqlen=16)
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize(
        ("bits", "expected"),
        [(4, 4), (8, 8)],
    )
    def test_scheme_bits_exported(self, tiny_opt_model_path, bits, expected):
        """The resolved scheme bits are persisted in the exported quantization config."""
        autoround = AutoRound(tiny_opt_model_path, bits=bits, group_size=128, sym=True, iters=0, disable_opt_rtn=True)
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        qcfg = model.config.quantization_config
        assert getattr(qcfg, "quant_method", None) == "auto-round"
        assert getattr(qcfg, "bits", None) == expected
