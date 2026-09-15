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

import json
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig

from auto_round import AutoRound, AutoScheme

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


def _local_calibration_dataset(tmp_path):
    dataset_path = tmp_path / "autoscheme_calibration.json"
    dataset_path.write_text(
        json.dumps(
            [
                "The quick brown fox jumps over the lazy dog. " * 32,
                "Quantization calibration needs distinct activation patterns. " * 32,
                "AutoRound assigns mixed precision to transformer layers. " * 32,
                "A compact local dataset keeps CUDA unit tests deterministic. " * 32,
            ]
        ),
        encoding="utf-8",
    )
    return str(dataset_path)


class TestAutoSchemeGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(240)
    def test_auto_scheme_quantize_reload(self, tiny_qwen_model_path, tmp_path, monkeypatch):
        monkeypatch.setenv("AR_AUTO_SCHEME_NSAMPLES", "1")
        monkeypatch.setenv("AR_AUTO_SCHEME_BATCH_SIZE", "1")
        calibration_dataset = _local_calibration_dataset(tmp_path)
        target = 5.0
        scheme = AutoScheme(avg_bits=target, options=("W4A16", "W8A16"))
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme=scheme,
            iters=0,
            disable_opt_rtn=True,
            disable_model_free=True,
            nsamples=1,
            seqlen=16,
            dataset=calibration_dataset,
        )
        quantized_model, quantized_model_path = autoround.quantize_and_save(
            output_dir=self.save_dir, format="auto_round"
        )

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            device_map="cuda:0",
            trust_remote_code=True,
            quantization_config=AutoRoundConfig(backend="torch"),
        )
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.timeout(120)
    def test_auto_scheme_avg_bits_in_range(self, tiny_qwen_model_path, tmp_path, monkeypatch):
        """The generated mixed scheme lands near the requested average bit-width."""
        from auto_round.auto_scheme.utils import compute_avg_bits_for_model

        monkeypatch.setenv("AR_AUTO_SCHEME_NSAMPLES", "4")
        monkeypatch.setenv("AR_AUTO_SCHEME_BATCH_SIZE", "1")
        calibration_dataset = _local_calibration_dataset(tmp_path)
        target = 5.0
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme=AutoScheme(avg_bits=target, options=("W4A16", "W8A16")),
            iters=0,
            disable_opt_rtn=True,
            disable_model_free=True,
            nsamples=1,
            seqlen=16,
            dataset=calibration_dataset,
        )
        model, _ = autoround.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model, ignore_scale_zp_bits=True)
        assert target - 0.3 < avg_bits <= target + 0.3

    @pytest.mark.timeout(240)
    def test_auto_scheme_iters1_reload(self, tiny_qwen_model_path, tmp_path, monkeypatch):
        """Auto scheme with sign-SGD (iters>0) + reload."""
        monkeypatch.setenv("AR_AUTO_SCHEME_NSAMPLES", "1")
        monkeypatch.setenv("AR_AUTO_SCHEME_BATCH_SIZE", "1")
        calibration_dataset = _local_calibration_dataset(tmp_path)
        scheme = AutoScheme(avg_bits=5.0, options=("W4A16", "W8A16"))
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme=scheme,
            iters=1,
            nsamples=1,
            seqlen=16,
            dataset=calibration_dataset,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            device_map="cuda:0",
            trust_remote_code=True,
            quantization_config=AutoRoundConfig(backend="torch"),
        )
        assert isinstance(model, torch.nn.Module)
