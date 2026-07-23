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

"""Tests for auto_round/special_model_handler.py"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestMllmsWithLimitedBs:
    """Test mllms_with_limited_bs tuple."""

    def test_is_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert isinstance(mllms_with_limited_bs, tuple)

    def test_llava_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "llava" in mllms_with_limited_bs

    def test_qwen2_vl_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "qwen2_vl" in mllms_with_limited_bs

    def test_phi3_v_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "phi3_v" in mllms_with_limited_bs

    def test_mllama_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "mllama" in mllms_with_limited_bs

    def test_qwen2_5_omni_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "qwen2_5_omni" in mllms_with_limited_bs

    def test_qwen3_omni_moe_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "qwen3_omni_moe" in mllms_with_limited_bs

    def test_glm_image_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "glm_image" in mllms_with_limited_bs

    def test_mimo_audio_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "mimo_audio" in mllms_with_limited_bs

    def test_qwen3_tts_in_tuple(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert "qwen3_tts" in mllms_with_limited_bs

    def test_tuple_length(self):
        from auto_round.special_model_handler import mllms_with_limited_bs

        assert len(mllms_with_limited_bs) == 9


class TestSupportOnlyTextModels:
    """Test SUPPORT_ONLY_TEXT_MODELS list."""

    def test_is_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert isinstance(SUPPORT_ONLY_TEXT_MODELS, list)

    def test_phi3_v_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "phi3_v" in SUPPORT_ONLY_TEXT_MODELS

    def test_cogvlm2_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "cogvlm2" in SUPPORT_ONLY_TEXT_MODELS

    def test_llava_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "llava" in SUPPORT_ONLY_TEXT_MODELS

    def test_qwen2_vl_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen2_vl" in SUPPORT_ONLY_TEXT_MODELS

    def test_qwen2_5_vl_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen2_5_vl" in SUPPORT_ONLY_TEXT_MODELS

    def test_deepseek_vl_v2_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "deepseek_vl_v2" in SUPPORT_ONLY_TEXT_MODELS

    def test_chatglm_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "chatglm" in SUPPORT_ONLY_TEXT_MODELS

    def test_idefics3_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "idefics3" in SUPPORT_ONLY_TEXT_MODELS

    def test_llama4_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "llama4" in SUPPORT_ONLY_TEXT_MODELS

    def test_internvl_chat_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "internvl_chat" in SUPPORT_ONLY_TEXT_MODELS

    def test_glm4v_moe_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "glm4v_moe" in SUPPORT_ONLY_TEXT_MODELS

    def test_glm_image_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "glm_image" in SUPPORT_ONLY_TEXT_MODELS

    def test_qwen3_vl_moe_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen3_vl_moe" in SUPPORT_ONLY_TEXT_MODELS

    def test_qwen2_5_omni_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen2_5_omni" in SUPPORT_ONLY_TEXT_MODELS

    def test_qwen3_omni_moe_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen3_omni_moe" in SUPPORT_ONLY_TEXT_MODELS

    def test_gemma3_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "gemma3" in SUPPORT_ONLY_TEXT_MODELS

    def test_bagel_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "bagel" in SUPPORT_ONLY_TEXT_MODELS

    def test_mimo_audio_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "mimo_audio" in SUPPORT_ONLY_TEXT_MODELS

    def test_qwen3_tts_in_list(self):
        from auto_round.special_model_handler import SUPPORT_ONLY_TEXT_MODELS

        assert "qwen3_tts" in SUPPORT_ONLY_TEXT_MODELS


class TestNotSupportOnlyTextModels:
    """Test NOT_SUPPORT_ONLY_TEXT_MODELS list."""

    def test_is_list(self):
        from auto_round.special_model_handler import NOT_SUPPORT_ONLY_TEXT_MODELS

        assert isinstance(NOT_SUPPORT_ONLY_TEXT_MODELS, list)

    def test_mllama_in_list(self):
        from auto_round.special_model_handler import NOT_SUPPORT_ONLY_TEXT_MODELS

        assert "mllama" in NOT_SUPPORT_ONLY_TEXT_MODELS

    def test_mistral3_2_in_list(self):
        from auto_round.special_model_handler import NOT_SUPPORT_ONLY_TEXT_MODELS

        assert "mistral3_2" in NOT_SUPPORT_ONLY_TEXT_MODELS


class TestSpecialSharedCacheKeys:
    """Test SPECIAL_SHARED_CACHE_KEYS dict."""

    def test_is_dict(self):
        from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS

        assert isinstance(SPECIAL_SHARED_CACHE_KEYS, dict)

    def test_gemma3_for_conditional_generation_key(self):
        from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS

        assert "Gemma3ForConditionalGeneration" in SPECIAL_SHARED_CACHE_KEYS
        keys = SPECIAL_SHARED_CACHE_KEYS["Gemma3ForConditionalGeneration"]
        assert "position_embeddings_global" in keys
        assert "position_embeddings_local" in keys

    def test_minimax_key(self):
        from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS

        assert "MiniMaxText01ForCausalLM" in SPECIAL_SHARED_CACHE_KEYS
        keys = SPECIAL_SHARED_CACHE_KEYS["MiniMaxText01ForCausalLM"]
        assert "slope_rate" in keys

    def test_stable_audio_dit_model_key(self):
        from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS

        assert "StableAudioDiTModel" in SPECIAL_SHARED_CACHE_KEYS
        keys = SPECIAL_SHARED_CACHE_KEYS["StableAudioDiTModel"]
        assert "encoder_hidden_states" in keys

    def test_gemma4_for_conditional_generation_key(self):
        from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS

        assert "Gemma4ForConditionalGeneration" in SPECIAL_SHARED_CACHE_KEYS
        keys = SPECIAL_SHARED_CACHE_KEYS["Gemma4ForConditionalGeneration"]
        assert "position_ids" in keys

    def test_wan_transformer_3d_model_key(self):
        from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS

        assert "WanTransformer3DModel" in SPECIAL_SHARED_CACHE_KEYS
        keys = SPECIAL_SHARED_CACHE_KEYS["WanTransformer3DModel"]
        assert "rotary_emb" in keys


class TestMistral32Models:
    """Test MISTRAL_3_2_MODELS list."""

    def test_is_list(self):
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS

        assert isinstance(MISTRAL_3_2_MODELS, list)

    def test_mistral_small_3_2_in_list(self):
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS

        assert "Mistral-Small-3.2" in MISTRAL_3_2_MODELS

    def test_magistral_small_in_list(self):
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS

        assert "Magistral-Small" in MISTRAL_3_2_MODELS

    def test_devstral_small_in_list(self):
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS

        assert "Devstral-Small" in MISTRAL_3_2_MODELS


class TestModelNameMatcher:
    """Test ModelNameMatcher class."""

    def test_match_qwen_in_mode(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("Qwen3-0.6B", mode="in")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "Qwen/Qwen3-0.6B"
        assert matcher(mock_model) is True

    def test_match_qwen_case_insensitive(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("Qwen", mode="in")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "Qwen/Qwen2.5-3B"
        assert matcher(mock_model) is True

    def test_match_deepseek_in_mode(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("deepseek-ai", mode="in")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "deepseek-ai/DeepSeek-V2-Lite"
        assert matcher(mock_model) is True

    def test_match_gemma_in_mode(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("gemma", mode="in")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "google/gemma-2b-it"
        assert matcher(mock_model) is True

    def test_no_match(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("Qwen", mode="in")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "facebook/opt-125m"
        assert matcher(mock_model) is False

    def test_full_mode_match(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("Qwen/Qwen3-0.6B", mode="full")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "Qwen/Qwen3-0.6B"
        assert matcher(mock_model) is True

    def test_full_mode_no_match(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("Qwen/Qwen3-0.6B", mode="full")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "Qwen/Qwen2.5-3B"
        assert matcher(mock_model) is False

    def test_regex_mode_match(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher(r"Qwen\d*-", mode="regex")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "Qwen/Qwen3-0.6B"
        assert matcher(mock_model) is True

    def test_regex_mode_no_match(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher(r"Mistral-", mode="regex")
        mock_model = MagicMock()
        mock_model.config.name_or_path = "Qwen/Qwen3-0.6B"
        assert matcher(mock_model) is False

    def test_unsupported_mode_raises(self):
        from auto_round.special_model_handler import ModelNameMatcher

        matcher = ModelNameMatcher("test", mode="unsupported")
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="unsupported mode"):
            matcher(mock_model)


class TestArchitectureMatcher:
    """Test ArchitectureMatcher class."""

    def test_match_qwen3_5_moe_in_mode(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("Qwen3_5Moe", mode="in")
        mock_model = MagicMock()
        mock_model.config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        assert matcher(mock_model) is True

    def test_match_qwen3_omni_moe_in_mode(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("Qwen3OmniMoe", mode="in")
        mock_model = MagicMock()
        mock_model.config.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
        assert matcher(mock_model) is True

    def test_match_deepseek_v2_in_mode(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("DeepSeekV2", mode="in")
        mock_model = MagicMock()
        mock_model.config.architectures = ["DeepSeekV2ForCausalLM"]
        assert matcher(mock_model) is True

    def test_match_gemma3_in_mode(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("Gemma3", mode="in")
        mock_model = MagicMock()
        mock_model.config.architectures = ["Gemma3ForConditionalGeneration"]
        assert matcher(mock_model) is True

    def test_no_match(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("Qwen", mode="in")
        mock_model = MagicMock()
        mock_model.config.architectures = ["OPTForCausalLM"]
        assert matcher(mock_model) is False

    def test_full_mode_match(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("Qwen3_5MoeForConditionalGeneration", mode="full")
        mock_model = MagicMock()
        mock_model.config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        assert matcher(mock_model) is True

    def test_full_mode_no_match(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("Qwen3_5MoeForConditionalGeneration", mode="full")
        mock_model = MagicMock()
        mock_model.config.architectures = ["Qwen3_5MoEForConditionalGeneration"]
        assert matcher(mock_model) is False

    def test_regex_mode_match(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher(r"Qwen\d*_?\d*Moe", mode="regex")
        mock_model = MagicMock()
        mock_model.config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        assert matcher(mock_model) is True

    def test_regex_mode_no_match(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher(r"Mistral-", mode="regex")
        mock_model = MagicMock()
        mock_model.config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        assert matcher(mock_model) is False

    def test_unsupported_mode_raises(self):
        from auto_round.special_model_handler import ArchitectureMatcher

        matcher = ArchitectureMatcher("test", mode="unsupported")
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="unsupported mode"):
            matcher(mock_model)


class TestPreDefinedIgnoreLayers:
    """Test PreDefinedIgnoreLayers dataclass."""

    def test_dataclass_fields(self):
        from auto_round.special_model_handler import PreDefinedIgnoreLayers

        ignore = PreDefinedIgnoreLayers(matchers=[], ignore_layers=[])
        assert hasattr(ignore, "matchers")
        assert hasattr(ignore, "ignore_layers")

    def test_dataclass_assignment(self):
        from auto_round.special_model_handler import PreDefinedIgnoreLayers

        matcher = MagicMock()
        ignore = PreDefinedIgnoreLayers(matchers=[matcher], ignore_layers=["layer.0", "layer.1"])
        assert ignore.matchers == [matcher]
        assert ignore.ignore_layers == ["layer.0", "layer.1"]

    def test_default_ignore_layers_empty_list(self):
        from auto_round.special_model_handler import PreDefinedIgnoreLayers

        ignore = PreDefinedIgnoreLayers(matchers=[])
        assert ignore.ignore_layers == []


class TestCheckMllmModelBatch:
    """Test check_mllm_model_batch function."""

    def _create_mock_model(self, model_type):
        mock_model = MagicMock()
        mock_model.config.model_type = model_type
        return mock_model

    def test_llava_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("llava")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_qwen2_vl_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("qwen2_vl")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_qwen2_5_omni_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("qwen2_5_omni")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_qwen3_omni_moe_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("qwen3_omni_moe")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_glm_image_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("glm_image")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_mimo_audio_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("mimo_audio")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_qwen3_tts_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("qwen3_tts")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_phi3_v_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("phi3_v")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_mllama_rejects_batch_greater_than_1(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("mllama")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 2, gradient_accumulate_steps=1)
        assert result_bs == 1
        assert result_gas == 2

    def test_single_batch_allowed(self):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("llava")
        result_bs, result_gas = check_mllm_model_batch(mock_model, 1)
        assert result_bs == 1
        assert result_gas == 1

    def test_non_mllm_batch_allowed(self):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("gpt2")
        result_bs, result_gas = check_mllm_model_batch(mock_model, 4)
        assert result_bs == 4
        assert result_gas == 1

    def test_gradient_accumulate_steps(self, caplog):
        from auto_round.special_model_handler import check_mllm_model_batch

        mock_model = self._create_mock_model("llava")
        with caplog.at_level("WARNING"):
            result_bs, result_gas = check_mllm_model_batch(mock_model, 4, gradient_accumulate_steps=2)
        assert result_bs == 1
        assert result_gas == 8


class TestNormalizeGemma4PerLayerInput:
    """Test _normalize_gemma4_per_layer_input function."""

    def test_none_input(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        result = _normalize_gemma4_per_layer_input(None, torch.randn(1, 10, 128))
        assert result is None

    def test_empty_list(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        result = _normalize_gemma4_per_layer_input([], torch.randn(1, 10, 128))
        assert result == []

    def test_same_shape(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        positional_inputs = (torch.randn(1, 10, 128),)
        hidden_states = torch.randn(1, 10, 128)
        result = _normalize_gemma4_per_layer_input(positional_inputs, hidden_states)
        assert torch.equal(result[0], positional_inputs[0])

    def test_truncate_longer_input(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        positional_inputs = (torch.randn(1, 20, 128),)
        hidden_states = torch.randn(1, 10, 128)
        result = _normalize_gemma4_per_layer_input(positional_inputs, hidden_states)
        assert result[0].shape[1] == 10

    def test_pad_shorter_input(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        positional_inputs = (torch.randn(1, 5, 128),)
        hidden_states = torch.randn(1, 10, 128)
        result = _normalize_gemma4_per_layer_input(positional_inputs, hidden_states)
        assert result[0].shape[1] == 10

    def test_returns_tuple_when_input_is_tuple(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        positional_inputs = (torch.randn(1, 10, 128), torch.randn(1, 10, 128))
        hidden_states = torch.randn(1, 10, 128)
        result = _normalize_gemma4_per_layer_input(positional_inputs, hidden_states)
        assert isinstance(result, tuple)

    def test_returns_list_when_input_is_list(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        positional_inputs = [torch.randn(1, 10, 128), torch.randn(1, 10, 128)]
        hidden_states = torch.randn(1, 5, 128)
        result = _normalize_gemma4_per_layer_input(positional_inputs, hidden_states)
        assert isinstance(result, list)

    def test_non_tensor_input_not_modified(self):
        from auto_round.special_model_handler import _normalize_gemma4_per_layer_input

        positional_inputs = ("not_a_tensor",)
        hidden_states = torch.randn(1, 10, 128)
        result = _normalize_gemma4_per_layer_input(positional_inputs, hidden_states)
        assert result == positional_inputs


class TestPrepareSpecialModelBlockInputs:
    """Test prepare_special_model_block_inputs function."""

    def test_position_ids_none_creates_tensor(self):
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = MagicMock()
        block._autoround_special_replay = None
        rotary_input = torch.randn(1, 10, 128)
        input_others = {"position_ids": None}
        positional_inputs = None
        result_others, result_pos = prepare_special_model_block_inputs(
            block, rotary_input, input_others, positional_inputs
        )
        assert result_others["position_ids"] is not None
        assert result_others["position_ids"].shape == (1, 10)

    def test_position_ids_list_single_element(self):
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = MagicMock()
        block._autoround_special_replay = None
        rotary_input = torch.randn(1, 10, 128)
        input_others = {"position_ids": [torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]}
        positional_inputs = None
        result_others, result_pos = prepare_special_model_block_inputs(
            block, rotary_input, input_others, positional_inputs
        )
        assert isinstance(result_others["position_ids"], torch.Tensor)

    def test_position_ids_list_empty_creates_tensor(self):
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = MagicMock()
        block._autoround_special_replay = None
        rotary_input = torch.randn(1, 10, 128)
        input_others = {"position_ids": []}
        positional_inputs = None
        result_others, result_pos = prepare_special_model_block_inputs(
            block, rotary_input, input_others, positional_inputs
        )
        assert result_others["position_ids"].shape == (1, 10)

    def test_position_ids_already_tensor_unchanged(self):
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = MagicMock()
        block._autoround_special_replay = None
        rotary_input = torch.randn(1, 10, 128)
        input_others = {"position_ids": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        positional_inputs = None
        result_others, result_pos = prepare_special_model_block_inputs(
            block, rotary_input, input_others, positional_inputs
        )
        assert torch.equal(result_others["position_ids"], input_others["position_ids"])

    def test_position_ids_not_in_input_others(self):
        from auto_round.special_model_handler import prepare_special_model_block_inputs

        block = MagicMock()
        block._autoround_special_replay = None
        rotary_input = torch.randn(1, 10, 128)
        input_others = {}
        positional_inputs = None
        result_others, result_pos = prepare_special_model_block_inputs(
            block, rotary_input, input_others, positional_inputs
        )
        assert "position_ids" not in result_others


class TestGetDeepseekVl2MultimodalBlock:
    """Test _get_deepseek_vl2_multimodal_block function."""

    def test_returns_language_layers(self):
        from auto_round.special_model_handler import _get_deepseek_vl2_multimodal_block

        mock_model = MagicMock()
        mock_model.language.model.layers = [MagicMock() for _ in range(10)]
        block_names = _get_deepseek_vl2_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 10

    def test_with_quant_vision_true(self):
        from auto_round.special_model_handler import _get_deepseek_vl2_multimodal_block

        mock_model = MagicMock()
        mock_model.vision.blocks = [MagicMock() for _ in range(5)]
        mock_model.projector.layers = [MagicMock() for _ in range(3)]
        mock_model.language.model.layers = [MagicMock() for _ in range(10)]
        block_names = _get_deepseek_vl2_multimodal_block(mock_model, quant_vision=True)
        assert len(block_names) == 3
        assert len(block_names[0]) == 5  # vision blocks
        assert len(block_names[1]) == 3  # projector layers
        assert len(block_names[2]) == 10  # language layers

    def test_model_forward_replaced(self):
        from auto_round.special_model_handler import _get_deepseek_vl2_multimodal_block

        mock_model = MagicMock()
        mock_model.language.forward = MagicMock()
        mock_model.language.model.layers = [MagicMock() for _ in range(10)]
        original_forward = mock_model.forward
        _get_deepseek_vl2_multimodal_block(mock_model)
        assert mock_model.forward == mock_model.language.forward


class TestGetQwen25OmniMultimodalBlock:
    """Test _get_qwen2_5_omni_multimodal_block function."""

    def test_returns_thinker_layers(self):
        from auto_round.special_model_handler import _get_qwen2_5_omni_multimodal_block

        mock_model = MagicMock()
        mock_model.thinker.model.layers = [MagicMock() for _ in range(8)]
        block_names = _get_qwen2_5_omni_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 8

    def test_with_quant_vision_true(self):
        from auto_round.special_model_handler import _get_qwen2_5_omni_multimodal_block

        mock_model = MagicMock()
        mock_model.thinker.visual.blocks = [MagicMock() for _ in range(5)]
        mock_model.thinker.audio_tower.layers = [MagicMock() for _ in range(3)]
        mock_model.thinker.model.layers = [MagicMock() for _ in range(8)]
        block_names = _get_qwen2_5_omni_multimodal_block(mock_model, quant_vision=True)
        assert len(block_names) == 3
        assert len(block_names[0]) == 5  # visual blocks
        assert len(block_names[1]) == 3  # audio tower layers
        assert len(block_names[2]) == 8  # thinker model layers

    def test_no_thinker_model(self):
        from auto_round.special_model_handler import _get_qwen2_5_omni_multimodal_block

        mock_model = MagicMock(spec=[])
        block_names = _get_qwen2_5_omni_multimodal_block(mock_model)
        assert block_names == []


class TestGetQwen3OmniMoeMultimodalBlock:
    """Test _get_qwen3_omni_moe_multimodal_block function."""

    def test_returns_thinker_layers(self):
        from auto_round.special_model_handler import _get_qwen3_omni_moe_multimodal_block

        mock_model = MagicMock()
        mock_model.thinker.model.layers = [MagicMock() for _ in range(8)]
        block_names = _get_qwen3_omni_moe_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 8

    def test_with_quant_vision_true(self):
        from auto_round.special_model_handler import _get_qwen3_omni_moe_multimodal_block

        mock_model = MagicMock()
        mock_model.thinker.visual.blocks = [MagicMock() for _ in range(5)]
        mock_model.thinker.audio_tower.layers = [MagicMock() for _ in range(3)]
        mock_model.thinker.model.layers = [MagicMock() for _ in range(8)]
        block_names = _get_qwen3_omni_moe_multimodal_block(mock_model, quant_vision=True)
        assert len(block_names) == 3


class TestGetGlmImageMultimodalBlock:
    """Test _get_glm_image_multimodal_block function."""

    def test_returns_language_model_layers(self):
        from auto_round.special_model_handler import _get_glm_image_multimodal_block

        mock_model = MagicMock()
        mock_model.model.language_model.layers = [MagicMock() for _ in range(12)]
        block_names = _get_glm_image_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 12

    def test_with_quant_vision_true(self):
        from auto_round.special_model_handler import _get_glm_image_multimodal_block

        mock_model = MagicMock()
        mock_model.model.visual.blocks = [MagicMock() for _ in range(8)]
        mock_model.model.language_model.layers = [MagicMock() for _ in range(12)]
        block_names = _get_glm_image_multimodal_block(mock_model, quant_vision=True)
        assert len(block_names) == 2
        assert len(block_names[0]) == 8  # visual blocks
        assert len(block_names[1]) == 12  # language model layers

    def test_no_language_model(self):
        from auto_round.special_model_handler import _get_glm_image_multimodal_block

        mock_model = MagicMock(spec=[])
        block_names = _get_glm_image_multimodal_block(mock_model)
        assert block_names == []


class TestGetMimoAudioMultimodalBlock:
    """Test _get_mimo_audio_multimodal_block function."""

    def test_returns_model_layers(self):
        from auto_round.special_model_handler import _get_mimo_audio_multimodal_block

        mock_model = MagicMock()
        mock_model.model.layers = [MagicMock() for _ in range(28)]
        block_names = _get_mimo_audio_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 28

    def test_with_base_model(self):
        from auto_round.special_model_handler import _get_mimo_audio_multimodal_block

        mock_model = MagicMock(spec=[])
        mock_model.layers = [MagicMock() for _ in range(20)]
        block_names = _get_mimo_audio_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 20

    def test_no_layers(self):
        from auto_round.special_model_handler import _get_mimo_audio_multimodal_block

        mock_model = MagicMock(spec=[])
        block_names = _get_mimo_audio_multimodal_block(mock_model)
        assert block_names == []


class TestGetQwen3TtsMultimodalBlock:
    """Test _get_qwen3_tts_multimodal_block function."""

    def test_tts_model_model_layers(self):
        from auto_round.special_model_handler import _get_qwen3_tts_multimodal_block

        mock_model = MagicMock()
        mock_model.tts_model.model.layers = [MagicMock() for _ in range(6)]
        block_names = _get_qwen3_tts_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 6

    def test_talker_model_layers(self):
        from auto_round.special_model_handler import _get_qwen3_tts_multimodal_block

        mock_model = MagicMock()
        mock_model.tts_model = MagicMock(spec=[])
        mock_model.talker.model.layers = [MagicMock() for _ in range(6)]
        block_names = _get_qwen3_tts_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 6

    def test_model_model_layers_fallback(self):
        from auto_round.special_model_handler import _get_qwen3_tts_multimodal_block

        mock_model = MagicMock()
        mock_model.tts_model = MagicMock(spec=[])
        mock_model.talker = MagicMock(spec=[])
        mock_model.model.layers = [MagicMock() for _ in range(6)]
        block_names = _get_qwen3_tts_multimodal_block(mock_model)
        assert len(block_names) == 1
        assert len(block_names[0]) == 6


class TestSpecialMultimodalBlockRegistry:
    """Test SPECIAL_MULTIMODAL_BLOCK registry."""

    def test_registry_contains_deepseek_vl_v2(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "deepseek_vl_v2" in SPECIAL_MULTIMODAL_BLOCK

    def test_registry_contains_qwen2_5_omni(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "qwen2_5_omni" in SPECIAL_MULTIMODAL_BLOCK

    def test_registry_contains_qwen3_omni_moe(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "qwen3_omni_moe" in SPECIAL_MULTIMODAL_BLOCK

    def test_registry_contains_glm_image(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "glm_image" in SPECIAL_MULTIMODAL_BLOCK

    def test_registry_contains_mimo_audio(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "mimo_audio" in SPECIAL_MULTIMODAL_BLOCK

    def test_registry_contains_qwen3_tts(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "qwen3_tts" in SPECIAL_MULTIMODAL_BLOCK

    def test_registry_contains_bagel(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        assert "bagel" in SPECIAL_MULTIMODAL_BLOCK


class TestRegisterIgnoreLayers:
    """Test register_ignore_layers function."""

    def test_register_ignore_layers(self):
        from auto_round.special_model_handler import _PRE_DEFINED_IGNORE_LAYERS, register_ignore_layers

        initial_count = len(_PRE_DEFINED_IGNORE_LAYERS)
        matcher = MagicMock(return_value=True)
        register_ignore_layers(matchers=[matcher], ignore_layers=["layer.0"])
        assert len(_PRE_DEFINED_IGNORE_LAYERS) == initial_count + 1


class TestGetPredefinedIgnoreLayers:
    """Test get_predefined_ignore_layers function."""

    def test_longcat_matcher(self):
        from auto_round.special_model_handler import get_predefined_ignore_layers

        mock_model = MagicMock()
        mock_model.config.architectures = ["LongcatConfig"]
        layers = get_predefined_ignore_layers(mock_model)
        assert "classifier" in layers

    def test_glm_flash_matcher(self):
        from auto_round.special_model_handler import get_predefined_ignore_layers

        mock_model = MagicMock()
        mock_model.config.model_type = "glm_moe_dsa"
        mock_model.config.first_k_dense_replace = 2
        layers = get_predefined_ignore_layers(mock_model)
        assert "layers.0.mlp" in layers
        assert "layers.1.mlp" in layers

    def test_step3p5_matcher(self):
        from auto_round.special_model_handler import get_predefined_ignore_layers

        mock_model = MagicMock()
        mock_model.config.model_type = "step3p5"
        layers = get_predefined_ignore_layers(mock_model)
        assert "g_proj" in layers
        assert "moe.gate" in layers
        assert "eh_proj" in layers
        assert "shared_head" in layers
        assert "layers.45" in layers

    def test_kimi_k25_matcher(self):
        from auto_round.special_model_handler import get_predefined_ignore_layers

        mock_model = MagicMock()
        mock_model.config.model_type = "kimi_k25"
        layers = get_predefined_ignore_layers(mock_model)
        assert "vision_tower" in layers
        assert "mm_projector" in layers

    def test_bagel_matcher(self):
        from auto_round.special_model_handler import get_predefined_ignore_layers

        mock_model = MagicMock()
        mock_model.config.model_type = "bagel"
        mock_model.language_model.model.layers = [MagicMock() for _ in range(32)]
        layers = get_predefined_ignore_layers(mock_model)
        assert "moe_gen" in layers
        assert "self_attn.q_proj" in layers
        assert "self_attn.k_proj" in layers
        assert "self_attn.v_proj" in layers
        assert "self_attn.o_proj" in layers

    def test_moe_model_via_config(self):
        from auto_round.special_model_handler import get_predefined_ignore_layers

        mock_model = MagicMock()
        mock_model.config.model_type = "test_moe"
        mock_model.config.architectures = ["TestMoE"]
        mock_model.named_modules.return_value = iter([])
        layers = get_predefined_ignore_layers(mock_model)
        # Should not add any layers without matching rules


class TestGetBagelIgnoreLayers:
    """Test get_bagel_ignore_layers function."""

    def test_returns_expected_layers(self):
        from auto_round.special_model_handler import get_bagel_ignore_layers

        mock_model = MagicMock()
        mock_model.language_model.model.layers = [MagicMock() for _ in range(32)]
        layers = get_bagel_ignore_layers(mock_model)
        assert "moe_gen" in layers
        assert "self_attn.q_proj" in layers
        assert "self_attn.k_proj" in layers
        assert "self_attn.v_proj" in layers
        assert "self_attn.o_proj" in layers

    def test_no_language_model(self):
        from auto_round.special_model_handler import get_bagel_ignore_layers

        mock_model = MagicMock(spec=[])
        layers = get_bagel_ignore_layers(mock_model)
        assert "moe_gen" in layers


class TestGetGlmFlashIgnoreLayers:
    """Test get_glm_flash_ignore_layers function."""

    def test_default_num_dense_layer(self):
        from auto_round.special_model_handler import get_glm_flash_ignore_layers

        mock_model = MagicMock(spec=[])
        layers = get_glm_flash_ignore_layers(mock_model)
        assert "layers.0.mlp" in layers

    def test_custom_num_dense_layer(self):
        from auto_round.special_model_handler import get_glm_flash_ignore_layers

        mock_model = MagicMock()
        mock_model.config.first_k_dense_replace = 3
        layers = get_glm_flash_ignore_layers(mock_model)
        assert "layers.0.mlp" in layers
        assert "layers.1.mlp" in layers
        assert "layers.2.mlp" in layers


class TestGetPredefinedFixedAttr:
    """Test get_predefined_fixed_attr function."""

    def test_gemma4_unified_returns_attrs(self):
        from auto_round.special_model_handler import get_predefined_fixed_attr

        mock_model = MagicMock()
        mock_model.config.model_type = "gemma4_unified"
        attrs = get_predefined_fixed_attr(mock_model)
        assert attrs is not None
        assert "has_variable_block_shape" in attrs

    def test_unknown_model_type_returns_none(self):
        from auto_round.special_model_handler import get_predefined_fixed_attr

        mock_model = MagicMock()
        mock_model.config.model_type = "unknown_model"
        attrs = get_predefined_fixed_attr(mock_model)
        assert attrs is None

    def test_no_config_returns_none(self):
        from auto_round.special_model_handler import get_predefined_fixed_attr

        mock_model = MagicMock(spec=[])
        del mock_model.config
        attrs = get_predefined_fixed_attr(mock_model)
        assert attrs is None

    def test_config_without_model_type_returns_none(self):
        from auto_round.special_model_handler import get_predefined_fixed_attr

        mock_model = MagicMock()
        mock_model.config.model_type = None
        attrs = get_predefined_fixed_attr(mock_model)
        assert attrs is None


class TestUpdateModule:
    """Test update_module function."""

    def test_gguf_format_returns_unchanged(self):
        from auto_round.formats import OutputFormat
        from auto_round.special_model_handler import update_module

        mock_model = MagicMock()
        gguf_format = MagicMock(spec=OutputFormat)
        gguf_format.is_gguf.return_value = True
        result = update_module(mock_model, formats=[gguf_format])
        assert result is mock_model

    def test_non_gguf_format_applies_replacements(self):
        from auto_round.formats import OutputFormat
        from auto_round.special_model_handler import update_module

        mock_model = MagicMock()
        non_gguf_format = MagicMock(spec=OutputFormat)
        non_gguf_format.is_gguf.return_value = False
        result = update_module(mock_model, formats=[non_gguf_format])
        # The function should call apply_replacements
        assert result is not None

    def test_no_formats_applies_replacements(self):
        from auto_round.special_model_handler import update_module

        mock_model = MagicMock()
        result = update_module(mock_model, formats=None)
        # The function should call apply_replacements
        assert result is not None


class TestDeepseekVl2Forward:
    """Test _deepseek_vl2_forward function."""

    def test_calls_prepare_inputs_embeds(self):
        from auto_round.special_model_handler import _deepseek_vl2_forward

        mock_model = MagicMock()
        mock_model.prepare_inputs_embeds.return_value = torch.randn(1, 10, 128)
        mock_model.language.return_value = MagicMock()

        input_ids = torch.tensor([1, 2, 3])
        _deepseek_vl2_forward(mock_model, input_ids=input_ids, images=None)

        mock_model.prepare_inputs_embeds.assert_called_once()
        mock_model.language.assert_called_once()


class TestQwen25OmniForward:
    """Test _qwen2_5_omni_forward function."""

    def test_calls_thinker_forward(self):
        from auto_round.special_model_handler import _qwen2_5_omni_forward

        mock_model = MagicMock()
        mock_model.thinker.return_value = MagicMock(hidden_states=[torch.randn(1, 10, 128)])
        mock_model.has_talker = False
        mock_model.thinker.get_input_embeddings.return_value = MagicMock(return_value=torch.randn(1, 10, 128))

        input_ids = torch.tensor([1, 2, 3])
        _qwen2_5_omni_forward(mock_model, input_ids=input_ids)

        mock_model.thinker.assert_called_once()


class TestMimoAudioForward:
    """Test _mimo_audio_forward function."""

    def test_converts_input_ids_to_embeds(self):
        from auto_round.special_model_handler import _mimo_audio_forward

        mock_model = MagicMock()
        mock_model.model.embed_tokens.return_value = torch.randn(1, 10, 128)
        mock_model.model.return_value = MagicMock()

        input_ids = torch.tensor([[1, 2, 3]])
        _mimo_audio_forward(mock_model, input_ids=input_ids)

        mock_model.model.embed_tokens.assert_called_once_with(input_ids)
        mock_model.model.assert_called_once()


class TestQwen3TtsForward:
    """Test _qwen3_tts_forward function."""

    def test_uses_tts_model_backbone(self):
        from auto_round.special_model_handler import _qwen3_tts_forward

        mock_model = MagicMock(spec=[])
        mock_tts_backbone = MagicMock()
        mock_model.tts_model = mock_tts_backbone
        mock_tts_backbone.model.text_embedding = MagicMock(return_value=torch.randn(1, 10, 128))
        mock_tts_backbone.text_projection = MagicMock(return_value=torch.randn(1, 10, 128))
        mock_tts_backbone.return_value = MagicMock()

        input_ids = torch.tensor([[1, 2, 3]])
        _qwen3_tts_forward(mock_model, input_ids=input_ids)

        mock_tts_backbone.assert_called_once()

    def test_uses_talker_backbone(self):
        from auto_round.special_model_handler import _qwen3_tts_forward

        mock_model = MagicMock(spec=[])
        mock_talker = MagicMock()
        mock_model.talker = mock_talker
        mock_model.tts_model = None
        mock_talker.model.text_embedding = MagicMock(return_value=torch.randn(1, 10, 128))
        mock_talker.text_projection = MagicMock(return_value=torch.randn(1, 10, 128))
        mock_talker.return_value = MagicMock()

        input_ids = torch.tensor([[1, 2, 3]])
        _qwen3_tts_forward(mock_model, input_ids=input_ids)

        mock_talker.assert_called_once()

    def test_raises_if_missing_text_embedding(self):
        from auto_round.special_model_handler import _qwen3_tts_forward

        mock_model = MagicMock(spec=[])
        mock_tts_backbone = MagicMock()
        mock_model.tts_model = mock_tts_backbone
        mock_tts_backbone.model.text_embedding = None
        mock_tts_backbone.text_projection = MagicMock()
        mock_model.talker = None

        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(RuntimeError, match="missing text_embedding"):
            _qwen3_tts_forward(mock_model, input_ids=input_ids)


class TestPredefinedIgnoreLayersRegistry:
    """Test _PRE_DEFINED_IGNORE_LAYERS global list has expected entries."""

    def test_registry_has_longcat_rule(self):
        from auto_round.special_model_handler import _PRE_DEFINED_IGNORE_LAYERS

        assert len(_PRE_DEFINED_IGNORE_LAYERS) > 0

    def test_registry_contains_multiple_rules(self):
        from auto_round.special_model_handler import _PRE_DEFINED_IGNORE_LAYERS

        assert len(_PRE_DEFINED_IGNORE_LAYERS) >= 5


class TestPredefinedFixedAttr:
    """Test _PRE_DEFINED_FIXED_ATTR global dict."""

    def test_gemma4_unified_in_dict(self):
        from auto_round.special_model_handler import _PRE_DEFINED_FIXED_ATTR

        assert "gemma4_unified" in _PRE_DEFINED_FIXED_ATTR


class TestGemma4HelperFunctions:
    """Test Gemma4 helper functions."""

    def test_get_gemma4_shared_kv_states_global_with_ref(self):
        from auto_round.special_model_handler import _get_gemma4_shared_kv_states_global

        mock_block = MagicMock()
        mock_block._shared_kv_states_global_ref = {"test": "value"}
        result = _get_gemma4_shared_kv_states_global(mock_block)
        assert result == {"test": "value"}

    def test_get_gemma4_shared_kv_states_global_without_ref(self):
        from auto_round.special_model_handler import _get_gemma4_shared_kv_states_global

        mock_block = MagicMock()
        mock_block._shared_kv_states_global_ref = None
        result = _get_gemma4_shared_kv_states_global(mock_block)
        assert result == {}

    def test_get_gemma4_rotary_emb_with_ref(self):
        from auto_round.special_model_handler import _get_gemma4_rotary_emb

        mock_block = MagicMock()
        mock_rotary_emb = MagicMock()
        mock_block._rotary_emb_ref = [mock_rotary_emb]
        result = _get_gemma4_rotary_emb(mock_block)
        assert result is mock_rotary_emb

    def test_get_gemma4_rotary_emb_without_ref(self):
        from auto_round.special_model_handler import _get_gemma4_rotary_emb

        mock_block = MagicMock()
        mock_block._rotary_emb_ref = None
        mock_block._rotary_emb = "default_rotary"
        result = _get_gemma4_rotary_emb(mock_block, default_rotary_emb="default_rotary")
        assert result == "default_rotary"
