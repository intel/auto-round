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

"""Tests for quantizing models built from ``Conv1D`` layers (e.g. GPT-2 style architectures)."""

import shutil
from test.helpers import model_infer

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils.device_manager import get_available_device_types

from ...envs import is_gptqmodel_available

# "cpu" is always available; extend with whatever accelerators AutoRound detects (cuda/xpu/hpu/...).
_AVAILABLE_DEVICES = ["cpu"] + [d for d in get_available_device_types() if d != "cpu"]


class TestQuantizationConv1d:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.mark.parametrize("device", _AVAILABLE_DEVICES)
    @pytest.mark.timeout(300)
    def test_quant(self, dataloader, device, tiny_lamini_model_path):
        """Quantize a Conv1D-based model, save it, reload on `device`, and run inference."""
        if device != "cpu" and not is_gptqmodel_available():
            pytest.skip("test requires gptqmodel>=2.0")

        model = AutoModelForCausalLM.from_pretrained(tiny_lamini_model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(tiny_lamini_model_path, trust_remote_code=True)

        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )

        autoround.quantize()
        _, quantized_model_path = autoround.save_quantized(self.save_dir, return_folders=True)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=device, trust_remote_code=True
        )
        quantized_tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, trust_remote_code=True)
        model_infer(quantized_model, quantized_tokenizer)


def test_find_layers_from_config_lamini():
    """Conv1D layers should be detectable directly from the model config (no weights loaded)."""
    from test.helpers import lamini_name_or_path

    from auto_round.utils.model import find_layers_from_config

    res = find_layers_from_config(lamini_name_or_path, class_names="Conv1d")
    assert "Conv1D" in res, "Conv1D should be detected in the model config"
    assert "h.0.attn.c_attn" in res["Conv1D"], "Conv1D should be detected in the model config with correct prefix"
    assert len(res["Conv1D"]) == 48, "At least one Conv1D layer should be detected in the model config"
