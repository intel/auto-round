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
from auto_round.special_model_handler import get_predefined_ignore_layers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestSpecialModelHandlerGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_moe_gate_ignore_layers(self, tiny_qwen_moe_model_path):
        # Ignore-layer discovery is architecture-only; avoid a needless CUDA
        # transfer in this smoke test.
        model = AutoModelForCausalLM.from_pretrained(tiny_qwen_moe_model_path, trust_remote_code=True)
        layers = get_predefined_ignore_layers(model)
        assert any(".gate" in name for name in layers), f"No MoE gate ignore layers found: {layers}"

    @pytest.mark.timeout(240)
    def test_moe_quantize_reload(self, tiny_qwen_moe_model_path):
        autoround = AutoRound(
            tiny_qwen_moe_model_path,
            bits=4,
            group_size=128,
            sym=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=8,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        # Gate modules must remain unquantized (registered as ignore layers).
        for name, module in model.named_modules():
            if name.endswith(".gate"):
                assert not hasattr(module, "qweight"), f"Gate layer {name} was unexpectedly quantized"

    def test_non_special_model_ignore_layers_empty(self, tiny_opt_model_path):
        """A plain OPT model is not special: no predefined ignore layers are returned."""
        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, trust_remote_code=True)
        layers = get_predefined_ignore_layers(model)
        assert layers == []

    @pytest.mark.timeout(240)
    def test_moe_lm_head_not_quantized(self, tiny_qwen_moe_model_path):
        """lm_head is not in the quant block list and must stay unquantized after export."""
        autoround = AutoRound(
            tiny_qwen_moe_model_path,
            bits=4,
            group_size=128,
            sym=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=8,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            assert not hasattr(lm_head, "qweight"), "lm_head was unexpectedly quantized"

    @pytest.mark.timeout(240)
    def test_moe_quantize_iters_reload(self, tiny_qwen_moe_model_path):
        """MoE quantization with sign-SGD (iters>0) + reload."""
        autoround = AutoRound(
            tiny_qwen_moe_model_path,
            bits=4,
            group_size=128,
            sym=True,
            iters=1,
            nsamples=1,
            seqlen=8,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
