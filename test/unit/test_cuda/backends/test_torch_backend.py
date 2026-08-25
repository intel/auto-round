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

"""Functional (non-accuracy) checks for the ``torch`` inference backend.

Verifies that models quantized via RTN or tuning, at various bit-widths/group
sizes/formats, can be reloaded with ``AutoRoundConfig(backend="torch")`` and
produce output -- no accuracy thresholds here (see
``test_cuda/backends/test_torch_backend_accuracy.py`` for that, which needs a
real, non-tiny model to produce meaningful numbers).
"""

import shutil
from test.helpers import model_infer

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils.device_manager import get_major_device

_AVAILABLE_DEVICES = [get_major_device()]

# (sym, group_size, format) combinations for 4-bit quantization.
_4BIT_CASES = [
    (False, 128, "auto_round:gptqmodel"),
    (True, 32, "auto_round"),
]


class TestTorchBackendFunctional:
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
    @pytest.mark.parametrize("iters", [0, 1], ids=["rtn", "tuning"])
    @pytest.mark.parametrize("sym,group_size,format", _4BIT_CASES)
    def test_torch_backend_4bit(self, tiny_opt_model_path, sym, group_size, format, iters, device):
        """4-bit (a)symmetric model, RTN or tuned, loads via the torch backend and generates text."""
        ar = AutoRound(tiny_opt_model_path, bits=4, group_size=group_size, sym=sym, iters=iters, seqlen=2, nsamples=1)
        _, quantized_model_path = ar.quantize_and_save(output_dir=self.save_dir, format=format)
        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)

    @pytest.mark.parametrize("device", _AVAILABLE_DEVICES)
    def test_torch_backend_3bit_asym(self, tiny_opt_model_path, dataloader, device):
        """3-bit asymmetric model loads via the torch backend and generates text."""
        ar = AutoRound(tiny_opt_model_path, bits=3, group_size=128, sym=False, iters=2, seqlen=2, dataset=dataloader)
        ar.quantize()
        ar.save_quantized(output_dir=self.save_dir, inplace=False, format="auto_round:gptqmodel")

        model = AutoModelForCausalLM.from_pretrained(self.save_dir, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)

    @pytest.mark.skip_ci(reason="Not necessary to test both symmetric and asymmetric for 3-bit quantization in CI")
    @pytest.mark.parametrize("device", _AVAILABLE_DEVICES)
    def test_torch_backend_3bit_sym(self, tiny_opt_model_path, dataloader, device):
        """3-bit symmetric model loads via the torch backend and generates text."""
        ar = AutoRound(tiny_opt_model_path, bits=3, group_size=128, sym=True, iters=2, seqlen=2, dataset=dataloader)
        ar.quantize()
        _, quantized_model_path = ar.save_quantized(
            output_dir=self.save_dir, inplace=False, format="auto_round", return_folders=True
        )

        quantization_config = AutoRoundConfig(backend="auto")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=device, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
