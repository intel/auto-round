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

"""Tests for asymmetric (sym=False) quantization: RTN and tuning paths, across formats and quant params."""

import shutil
from test.helpers import model_infer

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils.device_manager import get_available_device_types

# "cpu" is always available; extend with whatever accelerators AutoRound detects (cuda/xpu/hpu/...).
_AVAILABLE_DEVICES = ["cpu"] + [d for d in get_available_device_types() if d != "cpu"]
_CUDA_AVAILABLE = "cuda" in _AVAILABLE_DEVICES
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires a CUDA device")

# (bits, group_size) combinations exercised by the quant-param sweep below.
_QUANT_PARAMS = [(4, 32), (4, 64), (4, 128), (2, 128), (8, 128)]

_FORMATS = [
    "auto_round",
    "auto_round:auto_gptq",
    # GPTQModel's ExLlama post-init can segfault on some CUDA/Python
    # combinations.  Its CUDA backend has dedicated coverage; keep this
    # common test active on CPU/XPU and skip only the unstable CUDA variant.
    pytest.param(
        "auto_round:gptqmodel",
        marks=pytest.mark.skipif(torch.cuda.is_available(), reason="GPTQModel ExLlama CUDA path is covered separately"),
    ),
]


def _device_params(devices):
    """Skip only XPU parameterizations until the asymmetric XPU backend is available."""
    return [
        (
            pytest.param(
                device,
                marks=pytest.mark.skip(reason="asymmetric quantization backend is not available on XPU"),
            )
            if device == "xpu"
            else device
        )
        for device in devices
    ]


class TestAutoRoundAsym:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    # ------------------------------------------------------------------
    # RTN path (iters=0): cheap, so runs unconditionally in CI on every device.
    # ------------------------------------------------------------------
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("device", _device_params(_AVAILABLE_DEVICES))
    @pytest.mark.parametrize("format", _FORMATS)
    def test_asym_format_rtn(self, tiny_opt_model_path, format, device):
        """RTN-quantized asym model can be saved in each export format and reloaded for inference."""
        if str(device).startswith("cuda") and format in {
            "auto_round",
            "auto_round:auto_gptq",
            "auto_round:gptqmodel",
        }:
            pytest.skip("GPTQModel ExLlamaV2 CUDA path is covered by dedicated backend tests")
        bits, group_size, sym = 4, 128, False
        ar = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            seqlen=2,
            nsamples=1,
            disable_opt_rtn=True,
        )
        _, quantized_model_path = ar.quantize_and_save(format=format, output_dir=self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)

    @pytest.mark.parametrize("device", _device_params(_AVAILABLE_DEVICES))
    @pytest.mark.parametrize("bits,group_size", _QUANT_PARAMS)
    def test_asym_quant_params_rtn(self, tiny_opt_model_path, bits, group_size, device):
        """RTN asym quantization works across bit-widths and group sizes, and the result is loadable."""
        ar = AutoRound(tiny_opt_model_path, bits=bits, group_size=group_size, sym=False, iters=0, seqlen=2, nsamples=1)
        _, quantized_model_path = ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)

    # ------------------------------------------------------------------
    # Tuning path (iters=1): exercises the real sign-gradient tuning loop.
    # This is genuinely slow (real gradient tuning), so it's cheaper in wall-clock
    # time to run once on cuda than on every device (bits/format sweeps below are
    # heavier still and stay skip_ci -- manual/nightly coverage only).
    # ------------------------------------------------------------------
    @requires_cuda
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("group_size", [32])
    def test_asym_group_size_tuning(self, tiny_opt_model_path, group_size, dataloader):
        """Tuned (iters=1) asym quantization works across group sizes."""
        device = "cuda"
        ar = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=group_size,
            sym=False,
            iters=1,
            seqlen=2,
            nsamples=1,
            dataset=dataloader,
        )
        _, quantized_model_path = ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)

    @pytest.mark.skip_ci(reason="Coverage: Not necessary since it's covered by backend tests")
    @pytest.mark.parametrize("device", _device_params(_AVAILABLE_DEVICES))
    @pytest.mark.parametrize("bits", [2, 3, 8])
    def test_asym_bits_tuning(self, tiny_opt_model_path, bits, device):
        """Tuned (iters=1) asym quantization works across bit-widths."""
        ar = AutoRound(tiny_opt_model_path, bits=bits, group_size=128, sym=False, iters=1, seqlen=2, nsamples=1)
        _, quantized_model_path = ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)

    @pytest.mark.skip_ci(reason="Coverage: Not necessary since it's covered by backend tests")
    @pytest.mark.parametrize("device", _device_params(_AVAILABLE_DEVICES))
    @pytest.mark.parametrize("format", _FORMATS)
    def test_asym_format_tuning(self, tiny_opt_model_path, format, device):
        """Tuned (iters=1) asym model can be saved in each export format and reloaded for inference."""
        bits, group_size, sym = 4, 128, False
        ar = AutoRound(tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, nsamples=1)
        _, quantized_model_path = ar.quantize_and_save(format=format, output_dir=self.save_dir)

        if format == "auto_round:auto_gptq":
            # Cannot load correctly, skip auto_gptq since it's deprecated.
            return

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
