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
from transformers import AutoModelForCausalLM, AutoRoundConfig

from auto_round import AutoRound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


def _has_packed_weight(model):
    """Return True if the reloaded model contains at least one packed quant layer."""
    for module in model.modules():
        if hasattr(module, "qweight") or hasattr(module, "packed_weight"):
            return True
    return False


def _load_with_torch_backend(quantized_model_path):
    """Reload generic export coverage with the reference backend.

    Optimized backend selection and kernel compilation are covered in their
    dedicated CUDA backend tests.
    """
    return AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        device_map="cuda:0",
        trust_remote_code=True,
        quantization_config=AutoRoundConfig(backend="torch"),
    )


class TestBlockRunnerGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(180)
    def test_3bit_asym_sign_sgd_reload(self, tiny_opt_model_path, dataloader):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=3,
            group_size=32,
            sym=False,
            iters=2,
            nsamples=2,
            seqlen=8,
            dataset=dataloader,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = _load_with_torch_backend(quantized_model_path)
        assert isinstance(model, torch.nn.Module)
        assert _has_packed_weight(model)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.skip_ci(
        reason="Matrix: group_size=64 packing is covered by test_autoround_int_export; retain full Sign-SGD case weekly"
    )
    @pytest.mark.timeout(180)
    def test_group64_sign_sgd_reload(self, tiny_opt_model_path):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=64,
            sym=True,
            iters=3,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = _load_with_torch_backend(quantized_model_path)
        assert isinstance(model, torch.nn.Module)
        assert _has_packed_weight(model)

    @pytest.mark.skip_ci(
        reason="Coverage: symmetric INT export/reload is covered by test_autoround_int_export; retain tuning case weekly"
    )
    @pytest.mark.timeout(180)
    def test_4bit_sym_group128_reload(self, tiny_opt_model_path):
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

        model = _load_with_torch_backend(quantized_model_path)
        assert isinstance(model, torch.nn.Module)
        assert _has_packed_weight(model)

    @pytest.mark.skip_ci(
        reason="Coverage: asymmetric INT2 runtime is covered by the Triton backend smoke; retain full tuning case weekly"
    )
    @pytest.mark.timeout(180)
    def test_2bit_asym_group32_reload(self, tiny_opt_model_path):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=2,
            group_size=32,
            sym=False,
            iters=2,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = _load_with_torch_backend(quantized_model_path)
        assert isinstance(model, torch.nn.Module)
        assert _has_packed_weight(model)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.skip_ci(
        reason="Coverage: test_3bit_asym_sign_sgd_reload already verifies tuned reload and forward; retain longer convergence case weekly"
    )
    @pytest.mark.timeout(180)
    def test_sign_sgd_forward_after_reload(self, tiny_opt_model_path):
        """Sign-SGD quantized model must forward-pass on GPU after reload."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=32,
            sym=True,
            iters=5,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = _load_with_torch_backend(quantized_model_path)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1
