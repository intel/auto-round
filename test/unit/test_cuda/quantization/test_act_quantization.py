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


class TestActQuantizationGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def _reload_and_forward(self, quantized_model_path):
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    def test_w4a8_act_dynamic(self, tiny_opt_model_path):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            act_bits=8,
            group_size=128,
            act_group_size=32,
            act_dynamic=True,
            sym=True,
            iters=1,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w8a8_act_dynamic(self, tiny_opt_model_path):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=8,
            act_bits=8,
            group_size=128,
            act_group_size=32,
            act_dynamic=True,
            sym=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w4a8_act_static(self, tiny_opt_model_path):
        """Static (non-dynamic) activation quantization path."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            act_bits=8,
            group_size=128,
            act_group_size=128,
            act_dynamic=False,
            sym=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w4a8_act_group128(self, tiny_opt_model_path):
        """Wider act_group_size exercises a different act-hook grouping path."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            act_bits=8,
            group_size=128,
            act_group_size=128,
            act_dynamic=True,
            sym=True,
            iters=1,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w8a8_asym(self, tiny_opt_model_path):
        """Asymmetric W8A8 cannot be exported to the auto_round format.

        ``auto_round`` supports W8A8 export only for the symmetric path; an
        asymmetric W8A8 config is rejected during format resolution.
        """
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=8,
            act_bits=8,
            group_size=128,
            act_group_size=32,
            act_dynamic=True,
            sym=False,
            iters=0,
            disable_opt_rtn=True,
            nsamples=2,
            seqlen=16,
        )
        with pytest.raises(ValueError, match="does not support exporting"):
            autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
