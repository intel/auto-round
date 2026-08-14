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

import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM

from auto_round import AutoRound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestSignSgdPipeline:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(180)
    def test_sign_sgd_quantize_reload_forward(self, tiny_opt_model_path):
        """Run a real sign-SGD quantization on GPU, reload and forward-pass it."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=128,
            sym=True,
            iters=2,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        qcfg = model.config.quantization_config
        assert getattr(qcfg, "quant_method", None) == "auto-round"

        # A real forward pass on CUDA proves the quantized graph is runnable.
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    def test_w4a8_quantize_reload(self, tiny_opt_model_path):
        """W4A8 with iters>0 exercises the activation-quantization hooks on GPU."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            act_bits=8,
            group_size=128,
            act_group_size=32,
            sym=True,
            iters=1,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)

        # Run the act-quantized model on GPU to prove the W4A8 graph is runnable.
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    def test_sign_sgd_asym(self, tiny_opt_model_path):
        """Asymmetric sign-SGD quantization + export on GPU.

        Reload of the AWQ-packed asymmetric export requires the optional
        ``auto_awq`` package, so we verify the export itself completes.
        """
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=128,
            sym=False,
            iters=2,
            nsamples=2,
            seqlen=16,
        )
        compressed_model, quantized_model_path = autoround.quantize_and_save(
            output_dir=self.save_dir, format="auto_round"
        )
        assert isinstance(compressed_model, torch.nn.Module)
        assert os.path.isdir(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_sign_sgd_bits2_group32(self, tiny_opt_model_path):
        """2-bit, small-group sign-SGD exercises low-bit block optimization."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=2,
            group_size=32,
            sym=True,
            iters=2,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1
