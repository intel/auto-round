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

from auto_round import AutoRound, AutoScheme

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestAutoSchemeGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(240)
    def test_auto_scheme_quantize_reload(self, tiny_qwen_model_path):
        scheme = AutoScheme(avg_bits=5.0, options=("W4A16", "W8A16"))
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

    @pytest.mark.timeout(240)
    def test_auto_scheme_avg_bits_in_range(self, tiny_qwen_model_path):
        """The auto-generated mixed scheme must land near the requested avg_bits."""
        from auto_round.auto_scheme.utils import compute_avg_bits_for_model

        target = 5.0
        scheme = AutoScheme(avg_bits=target, options=("W4A16", "W8A16"))
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme=scheme,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        model, _ = autoround.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model, ignore_scale_zp_bits=True)
        assert target - 0.3 < avg_bits <= target + 0.3

    @pytest.mark.timeout(240)
    def test_auto_scheme_iters1_reload(self, tiny_qwen_model_path):
        """Auto scheme with sign-SGD (iters>0) + reload."""
        scheme = AutoScheme(avg_bits=5.0, options=("W4A16", "W8A16"))
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme=scheme,
            iters=1,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
