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

"""CPU-only quantization coverage for audio model architectures."""

import os

import pytest
import torch

from auto_round import AutoRound
from auto_round.special_model_handler import resolve_model_type


class _LocalCalibrationDataLoader:
    """Provide deterministic token batches without downloading a calibration dataset."""

    def __iter__(self):
        yield torch.ones((1, 32), dtype=torch.long)


class TestMiMoAudioQuantization:
    def test_quantize_rtn(self, tiny_mimo_audio_model_path, tmp_path):
        """RTN-quantizing a tiny MiMo-Audio model produces QuantLinear layers.

        Algorithm correctness -- runs once on cpu.
        """
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(
            tiny_mimo_audio_model_path, torch_dtype="auto", trust_remote_code=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_mimo_audio_model_path)
        assert resolve_model_type(model) == "mimo_audio"

        autoround = AutoRound(
            model,
            tokenizer=tokenizer,
            scheme="W4A16",
            nsamples=1,
            iters=0,
            seqlen=32,
            device_map="cpu",
            enable_torch_compile=True,
            disable_opt_rtn=True,
        )
        quantized_model, save_folder = autoround.quantize_and_save(output_dir=str(tmp_path / "saved"))
        assert quantized_model is not None
        assert save_folder is not None
        has_quantlinear = any(m.__class__.__name__ == "QuantLinear" for m in quantized_model.modules())
        assert has_quantlinear, "Quantized model should contain QuantLinear layers"

    @pytest.mark.timeout(90)
    def test_quantize_with_tuning(self, tiny_mimo_audio_model_path, tmp_path):
        """Tuned MiMo-Audio quantization uses local calibration data and produces QuantLinear layers."""
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(
            tiny_mimo_audio_model_path, torch_dtype="auto", trust_remote_code=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_mimo_audio_model_path)

        autoround = AutoRound(
            model,
            tokenizer=tokenizer,
            scheme="W4A16",
            nsamples=1,
            iters=1,
            seqlen=32,
            dataset=_LocalCalibrationDataLoader(),
            device_map="cpu",
            enable_torch_compile=True,
        )
        quantized_model, _ = autoround.quantize_and_save(output_dir=str(tmp_path / "saved"))
        assert quantized_model is not None
        has_quantlinear = any(m.__class__.__name__ == "QuantLinear" for m in quantized_model.modules())
        assert has_quantlinear, "Quantized model should contain QuantLinear layers after tuning"


class TestStableAudioQuantization:
    def test_quantize_rtn(self, tiny_stable_audio_pipe, tmp_path):
        """RTN-quantizing a tiny StableAudio pipeline saves both quantized and non-quantized components.

        Algorithm/export correctness -- runs once on cpu.
        """
        from diffusers import StableAudioPipeline

        pipe = StableAudioPipeline.from_pretrained(tiny_stable_audio_pipe)
        output_dir = str(tmp_path / "stable_audio_rtn")

        autoround = AutoRound(
            pipe,
            tokenizer=None,
            scheme="W4A16",
            nsamples=1,
            iters=0,
            disable_opt_rtn=True,
            calib_num_inference_steps=2,
            device_map="cpu",
            enable_torch_compile=True,
        )
        autoround.quantize_and_save(output_dir)

        has_quantlinear = any(m.__class__.__name__ == "QuantLinear" for m in pipe.transformer.modules())
        assert has_quantlinear, "Quantized transformer should contain QuantLinear layers"
        assert os.path.exists(os.path.join(output_dir, "model_index.json")), "model_index.json missing"
        assert os.path.exists(
            os.path.join(output_dir, "transformer", "quantization_config.json")
        ), "quantization_config.json missing in transformer directory"
        # Non-quantized pipeline components must still be saved alongside the quantized one.
        assert os.path.exists(os.path.join(output_dir, "text_encoder")), "text_encoder directory missing"
        assert os.path.exists(os.path.join(output_dir, "vae")), "vae directory missing"
        assert os.path.exists(os.path.join(output_dir, "projection_model")), "projection_model directory missing"
