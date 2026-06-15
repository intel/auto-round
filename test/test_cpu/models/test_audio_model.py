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

"""CPU unit tests for Audio model support.

Tests cover:
- MiMo-Audio: architecture-based model_type resolution, block detection,
  forward patching, processor/template registration, quantization with
  a tiny dummy Qwen model patched to MiMo-Audio config.
- StableAudio: diffusion pipeline function attachment, output config
  registration, shared cache keys, and quantization (RTN) with a tiny
  dummy StableAudioPipeline built from random weights.
"""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from auto_round import AutoRound
from auto_round.compressors.mllm.processor import PROCESSORS
from auto_round.special_model_handler import (
    ARCHITECTURE_MODEL_TYPE_MAP,
    SPECIAL_MULTIMODAL_BLOCK,
    SPECIAL_SHARED_CACHE_KEYS,
    SUPPORT_ONLY_TEXT_MODELS,
    _get_mimo_audio_multimodal_block,
    _handle_special_model,
    check_mllm_model_batch,
    mllms_with_limited_bs,
    resolve_model_type,
)

# ====================== Mock Helpers =========================================


def _make_mock_config(model_type, architectures=None):
    cfg = SimpleNamespace(model_type=model_type)
    if architectures is not None:
        cfg.architectures = architectures
    return cfg


def _make_mimo_audio_mock(n_main_layers=2, n_input_local_layers=1, n_local_layers=2):
    """Build a mock MiMo-Audio model with correct module hierarchy."""
    model = MagicMock()
    model.config = _make_mock_config("qwen2", architectures=["MiMoAudioModel"])

    main_layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(n_main_layers)])
    inner_model = MagicMock()
    inner_model.layers = main_layers
    model.model = inner_model

    input_local = MagicMock()
    input_local.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(n_input_local_layers)])
    model.input_local_transformer = input_local

    local_xfm = MagicMock()
    local_xfm.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(n_local_layers)])
    model.local_transformer = local_xfm

    return model


# ====================== MiMo-Audio Unit Tests ================================


class TestResolveModelType:
    """Test architecture-based model_type resolution."""

    def test_mimo_audio_architecture_override(self):
        model = MagicMock()
        model.config = _make_mock_config("qwen2", architectures=["MiMoAudioModel"])
        assert resolve_model_type(model) == "mimo_audio"

    def test_mimo_audio_causal_lm_variant(self):
        model = MagicMock()
        model.config = _make_mock_config("qwen2", architectures=["MiMoAudioForCausalLM"])
        assert resolve_model_type(model) == "mimo_audio"

    def test_standard_qwen2_not_overridden(self):
        model = MagicMock()
        model.config = _make_mock_config("qwen2", architectures=["Qwen2ForCausalLM"])
        assert resolve_model_type(model) == "qwen2"


class TestMiMoAudioBlockDetection:
    """Test _get_mimo_audio_multimodal_block with mock models."""

    def test_main_decoder_only(self):
        model = _make_mimo_audio_mock(n_main_layers=4)
        blocks = _get_mimo_audio_multimodal_block(model)
        assert len(blocks) == 1
        assert blocks[0] == [f"model.layers.{i}" for i in range(4)]


class TestMiMoAudioRegistration:
    """Verify MiMo-Audio is properly registered in all required registries."""

    def test_special_map(self):
        assert "MiMoAudioModel" in ARCHITECTURE_MODEL_TYPE_MAP
        assert "MiMoAudioForCausalLM" in ARCHITECTURE_MODEL_TYPE_MAP
        assert "mimo_audio" in SPECIAL_MULTIMODAL_BLOCK
        assert "mimo_audio" in SUPPORT_ONLY_TEXT_MODELS

        assert "mimo_audio" in PROCESSORS
        from auto_round.compressors.mllm.template import TEMPLATES

        assert "mimo_audio" in TEMPLATES


class TestMiMoAudioForwardPatching:
    """Test that _handle_special_model patches the forward function."""

    def test_forward_is_patched(self, tiny_mimo_audio_model_path):
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(tiny_mimo_audio_model_path)
        original_forward = model.forward
        model = _handle_special_model(model)
        assert model.forward != original_forward


class TestMiMoAudioQuantization:
    """End-to-end quantization with a tiny Qwen model patched to MiMo-Audio config."""

    def test_quantize_rtn(self, tiny_mimo_audio_model_path, tmp_path):
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
            enable_torch_compile=True,
            disable_opt_rtn=True,
        )
        quantized_model, save_folder = autoround.quantize_and_save(output_dir=str(tmp_path / "saved"))
        assert quantized_model is not None
        assert save_folder is not None
        # Verify quantized model has QuantLinear layers
        has_quantlinear = any(m.__class__.__name__ == "QuantLinear" for m in quantized_model.modules())
        assert has_quantlinear, "Quantized model should contain QuantLinear layers"

    def test_quantize_with_tuning(self, tiny_mimo_audio_model_path, tmp_path):
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
            enable_torch_compile=True,
            disable_opt_rtn=True,
        )
        quantized_model, _ = autoround.quantize_and_save(format="auto_round", output_dir=str(tmp_path / "saved"))
        assert quantized_model is not None
        # Verify quantized model has QuantLinear layers
        has_quantlinear = any(m.__class__.__name__ == "QuantLinear" for m in quantized_model.modules())
        assert has_quantlinear, "Quantized model should contain QuantLinear layers after tuning"


# ====================== StableAudio Unit Tests ===============================


class TestStableAudioRegistration:
    """Verify StableAudio-specific registrations."""

    def test_config_and_special_registered(self):
        from auto_round.algorithms.quantization.base import DiffusionMixin

        assert "StableAudioDiTBlock" in DiffusionMixin.DIFFUSION_OUTPUT_CONFIGS
        assert DiffusionMixin.DIFFUSION_OUTPUT_CONFIGS["StableAudioDiTBlock"] == ["hidden_states"]

        assert "StableAudioDiTModel" in SPECIAL_SHARED_CACHE_KEYS
        assert "encoder_hidden_states" in SPECIAL_SHARED_CACHE_KEYS["StableAudioDiTModel"]


class TestStableAudioPipelineFunction:
    """Test custom pipeline function attachment for StableAudio."""

    def test_attach_pipeline_fn(self):
        from auto_round.utils.model import _attach_diffusion_pipeline_fn

        pipe = MagicMock()
        type(pipe).__name__ = "StableAudioPipeline"
        _attach_diffusion_pipeline_fn(pipe)
        assert hasattr(pipe, "_autoround_pipeline_fn")
        pipe._autoround_pipeline_fn(pipe, ["test prompt"], guidance_scale=3.5, num_inference_steps=10)
        pipe.assert_called_once()
        _, kwargs = pipe.call_args
        assert "audio_end_in_s" in kwargs

    def test_noop_for_non_stable_audio(self):
        from auto_round.utils.model import _attach_diffusion_pipeline_fn

        pipe = MagicMock(spec=["config", "__call__"])
        type(pipe).__name__ = "FluxPipeline"
        _attach_diffusion_pipeline_fn(pipe)
        assert not hasattr(pipe, "_autoround_pipeline_fn")


class TestStableAudioQuantization:
    """End-to-end quantization with a tiny StableAudioPipeline (random weights)."""

    def test_quantize_rtn(self, tiny_stable_audio_pipe, tmp_path):
        """RTN quantization on tiny StableAudio pipeline."""
        from diffusers import StableAudioPipeline

        pipe = StableAudioPipeline.from_pretrained(tiny_stable_audio_pipe)
        print(pipe)
        output_dir = str(tmp_path / "stable_audio_rtn")

        autoround = AutoRound(
            pipe,
            tokenizer=None,
            scheme="W4A16",
            nsamples=1,
            iters=0,
            disable_opt_rtn=True,
            num_inference_steps=2,
        )
        autoround.quantize_and_save(output_dir)
        # Verify quantized transformer has QuantLinear layers
        has_quantlinear = any(m.__class__.__name__ == "QuantLinear" for m in pipe.transformer.modules())
        assert has_quantlinear, "Quantized transformer should contain QuantLinear layers"
        # Verify StableAudio-specific output: pipeline config and quantized transformer
        import os

        assert os.path.exists(os.path.join(output_dir, "model_index.json")), "model_index.json missing"
        assert os.path.exists(
            os.path.join(output_dir, "transformer", "quantization_config.json")
        ), "quantization_config.json missing in transformer directory"
        # Verify non-quantized pipeline components are also saved
        assert os.path.exists(os.path.join(output_dir, "text_encoder")), "text_encoder directory missing"
        assert os.path.exists(os.path.join(output_dir, "vae")), "vae directory missing"
        assert os.path.exists(os.path.join(output_dir, "projection_model")), "projection_model directory missing"
