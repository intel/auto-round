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

"""Tests for audio model support: MiMo-Audio, Qwen3-TTS, and StableAudio."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from auto_round import AutoRound
from auto_round.compressors.mllm.processor import PROCESSORS
from auto_round.compressors.mllm.template import TEMPLATES
from auto_round.special_model_handler import (
    ARCHITECTURE_MODEL_TYPE_MAP,
    SPECIAL_MULTIMODAL_BLOCK,
    SPECIAL_SHARED_CACHE_KEYS,
    SUPPORT_ONLY_TEXT_MODELS,
    _get_mimo_audio_multimodal_block,
    _get_qwen3_tts_multimodal_block,
    _handle_special_model,
    _qwen3_tts_forward,
    mllms_with_limited_bs,
    resolve_model_type,
)


def _make_mock_config(model_type, architectures=None):
    cfg = SimpleNamespace(model_type=model_type)
    if architectures is not None:
        cfg.architectures = architectures
    return cfg


def _make_mimo_audio_mock(n_main_layers=2, n_input_local_layers=1, n_local_layers=2):
    """Build a mock MiMo-Audio model with the expected module hierarchy."""
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


def _make_qwen3_tts_mock(n_layers=2, use_tts_model=True):
    """Build a mock Qwen3-TTS model, routed through either ``tts_model`` or ``talker``."""
    model = MagicMock()
    model.config = _make_mock_config("qwen3_tts", architectures=["Qwen3TTSForConditionalGeneration"])

    layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(n_layers)])
    inner = MagicMock()
    inner.layers = layers

    if use_tts_model:
        tts = MagicMock()
        tts.model = inner
        model.tts_model = tts
        del model.talker
    else:
        talker = MagicMock()
        talker.model = inner
        model.talker = talker
        del model.tts_model

    return model


# ====================== MiMo-Audio ======================


class TestResolveModelType:
    def test_mimo_audio_architecture_override(self):
        """architectures=[MiMoAudioModel] resolves to the mimo_audio model type."""
        model = MagicMock()
        model.config = _make_mock_config("qwen2", architectures=["MiMoAudioModel"])
        assert resolve_model_type(model) == "mimo_audio"

    def test_mimo_audio_causal_lm_variant(self):
        """architectures=[MiMoAudioForCausalLM] also resolves to mimo_audio."""
        model = MagicMock()
        model.config = _make_mock_config("qwen2", architectures=["MiMoAudioForCausalLM"])
        assert resolve_model_type(model) == "mimo_audio"

    def test_qwen3_tts_uses_config_model_type(self):
        """qwen3_tts architecture resolves via its config model_type."""
        model = MagicMock()
        model.config = _make_mock_config("qwen3_tts", architectures=["Qwen3TTSForConditionalGeneration"])
        assert resolve_model_type(model) == "qwen3_tts"

    def test_standard_qwen2_not_overridden(self):
        """A plain Qwen2ForCausalLM is not remapped to a special model type."""
        model = MagicMock()
        model.config = _make_mock_config("qwen2", architectures=["Qwen2ForCausalLM"])
        assert resolve_model_type(model) == "qwen2"

    def test_no_architectures_falls_back(self):
        """Missing `architectures` falls back to the config's model_type."""
        model = MagicMock()
        model.config = _make_mock_config("qwen2")
        assert resolve_model_type(model) == "qwen2"

    def test_no_config_returns_none(self):
        """A model without a config resolves to None instead of raising."""
        model = MagicMock(spec=[])
        assert resolve_model_type(model) is None


class TestMiMoAudioBlockDetection:
    def test_main_decoder_only(self):
        """Main decoder layers are detected as a single multimodal block."""
        model = _make_mimo_audio_mock(n_main_layers=4)
        blocks = _get_mimo_audio_multimodal_block(model)
        assert len(blocks) == 1
        assert blocks[0] == [f"model.layers.{i}" for i in range(4)]

    def test_no_audio_modules(self):
        """Detection degrades gracefully when the audio-specific submodules are absent."""
        model = MagicMock()
        model.config = _make_mock_config("qwen2", architectures=["MiMoAudioModel"])
        inner = MagicMock()
        inner.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])
        model.model = inner
        del model.input_local_transformer
        del model.local_transformer
        blocks = _get_mimo_audio_multimodal_block(model)
        assert len(blocks) == 1


class TestMiMoAudioRegistration:
    def test_special_map(self):
        """MiMo-Audio is registered in every registry it needs to be (block/text/processor/template)."""
        assert "MiMoAudioModel" in ARCHITECTURE_MODEL_TYPE_MAP
        assert "MiMoAudioForCausalLM" in ARCHITECTURE_MODEL_TYPE_MAP
        assert ARCHITECTURE_MODEL_TYPE_MAP["MiMoAudioModel"] == "mimo_audio"
        assert "mimo_audio" in SPECIAL_MULTIMODAL_BLOCK
        assert "mimo_audio" in SUPPORT_ONLY_TEXT_MODELS
        assert "mimo_audio" in PROCESSORS
        assert "mimo_audio" in TEMPLATES


class TestMiMoAudioForwardPatching:
    def test_forward_is_patched(self, tiny_mimo_audio_model_path):
        """_handle_special_model replaces forward on a real (tiny) MiMo-Audio model."""
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(tiny_mimo_audio_model_path)
        original_forward = model.forward
        model = _handle_special_model(model)
        assert model.forward != original_forward


class TestMiMoAudioQuantization:
    def test_quantize_rtn(self, tiny_mimo_audio_model_path, tmp_path):
        """RTN-quantizing a tiny MiMo-Audio model produces QuantLinear layers.

        Algorithm correctness -- runs once on cpu.
        """
        device = "cpu"
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
            device_map=device,
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
        """Tuned (iters=1) quantization of a tiny MiMo-Audio model also produces QuantLinear layers.

        Algorithm correctness -- runs once on cpu (measured cpu ~21s vs cuda ~18s here,
        not different enough to justify pinning to cuda).
        """
        device = "cpu"
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
            device_map=device,
            enable_torch_compile=True,
        )
        quantized_model, _ = autoround.quantize_and_save(output_dir=str(tmp_path / "saved"))
        assert quantized_model is not None
        has_quantlinear = any(m.__class__.__name__ == "QuantLinear" for m in quantized_model.modules())
        assert has_quantlinear, "Quantized model should contain QuantLinear layers after tuning"


# ====================== Qwen3-TTS ======================


class TestQwen3TTSDetection:
    def test_special_map(self):
        """Qwen3-TTS is registered in every registry it needs to be."""
        assert "qwen3_tts" in SPECIAL_MULTIMODAL_BLOCK
        assert "qwen3_tts" in SUPPORT_ONLY_TEXT_MODELS
        assert "qwen3_tts" in mllms_with_limited_bs
        assert "qwen3_tts" in PROCESSORS
        assert "qwen3_tts" in TEMPLATES

    def test_tts_model_path(self):
        """Block detection routes through `tts_model` when present."""
        model = _make_qwen3_tts_mock(n_layers=3, use_tts_model=True)
        blocks = _get_qwen3_tts_multimodal_block(model)
        assert len(blocks) == 1
        assert blocks[0] == [f"tts_model.model.layers.{i}" for i in range(3)]

    def test_talker_path(self):
        """Block detection routes through `talker` when `tts_model` is absent."""
        model = _make_qwen3_tts_mock(n_layers=4, use_tts_model=False)
        blocks = _get_qwen3_tts_multimodal_block(model)
        assert len(blocks) == 1
        assert blocks[0] == [f"talker.model.layers.{i}" for i in range(4)]

    def test_fallback_to_model_layers(self):
        """Block detection falls back to `model.layers` when neither submodule is present."""
        model = MagicMock()
        model.config = _make_mock_config("qwen3_tts")
        inner = MagicMock()
        inner.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])
        model.model = inner
        del model.tts_model
        del model.talker
        blocks = _get_qwen3_tts_multimodal_block(model)
        assert len(blocks) == 1
        assert blocks[0] == [f"model.layers.{i}" for i in range(2)]


class TestQwen3TTSForward:
    def test_routes_through_tts_model(self):
        """Forward is routed through `tts_model` when present."""
        model = MagicMock()
        tts = MagicMock()
        model.tts_model = tts
        del model.talker
        dummy_ids = torch.randint(0, 100, (1, 4))
        _qwen3_tts_forward(model, input_ids=dummy_ids)
        tts.assert_called_once()

    def test_routes_through_talker(self):
        """Forward is routed through `talker` when `tts_model` is None."""
        model = MagicMock()
        talker = MagicMock()
        model.talker = talker
        model.tts_model = None
        dummy_ids = torch.randint(0, 100, (1, 4))
        _qwen3_tts_forward(model, input_ids=dummy_ids)
        talker.assert_called_once()


# ====================== StableAudio ======================


class TestStableAudioRegistration:
    def test_config_and_special_registered(self):
        """StableAudio is registered for diffusion output handling and shared cache keys."""
        from auto_round.algorithms.block_runner import _DIFFUSION_OUTPUT_REGISTRY

        assert "StableAudioDiTBlock" in _DIFFUSION_OUTPUT_REGISTRY
        assert _DIFFUSION_OUTPUT_REGISTRY["StableAudioDiTBlock"] == ["hidden_states"]

        assert "StableAudioDiTModel" in SPECIAL_SHARED_CACHE_KEYS
        assert "encoder_hidden_states" in SPECIAL_SHARED_CACHE_KEYS["StableAudioDiTModel"]


class TestStableAudioPipelineFunction:
    def test_attach_pipeline_fn(self):
        """StableAudio pipelines get a custom call wrapper injecting audio_end_in_s."""
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
        """Non-StableAudio pipelines are left untouched."""
        from auto_round.utils.model import _attach_diffusion_pipeline_fn

        pipe = MagicMock(spec=["config", "__call__"])
        type(pipe).__name__ = "FluxPipeline"
        _attach_diffusion_pipeline_fn(pipe)
        assert not hasattr(pipe, "_autoround_pipeline_fn")


class TestStableAudioQuantization:
    def test_quantize_rtn(self, tiny_stable_audio_pipe, tmp_path):
        """RTN-quantizing a tiny StableAudio pipeline saves both quantized and non-quantized components.

        Algorithm/export correctness -- runs once on cpu.
        """
        device = "cpu"
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
            num_inference_steps=2,
            device_map=device,
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
