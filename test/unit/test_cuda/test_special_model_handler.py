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
"""GPU-side fast unit tests for ``auto_round.special_model_handler``.

These tests focus on the pure-Python helpers that don't require any actual
heavy models to load, so they run in milliseconds and bring meaningful
coverage of the 451-statement module to the GPU CI pipeline.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from auto_round import special_model_handler
from auto_round.special_model_handler import (
    ArchitectureMatcher,
    ModelNameMatcher,
    ModelTypeMatcher,
    get_glm_flash_ignore_layers,
    get_predefined_ignore_layers,
    prepare_special_model_block_inputs,
    register_ignore_layers,
    update_module,
)


def _make_model(name_or_path="some/model", architectures=None, model_type=None):
    cfg = SimpleNamespace(
        name_or_path=name_or_path,
        architectures=architectures or [],
        model_type=model_type,
    )
    model = nn.Module()
    model.config = cfg
    return model


# ==============================================================================
# ModelNameMatcher
# ==============================================================================


class TestModelNameMatcher:
    def test_in_mode_matches_substring(self):
        # Case-sensitive substring match against name_or_path
        m = ModelNameMatcher("Qwen", mode="in")
        assert m(_make_model(name_or_path="Qwen/Qwen3-0.6B")) is True
        assert m(_make_model(name_or_path="facebook/opt-125m")) is False

    def test_in_mode_no_match_returns_false(self):
        m = ModelNameMatcher("Qwen", mode="in")
        assert m(_make_model(name_or_path="facebook/opt-125m")) is False

    def test_in_mode_empty_pattern_matches_everything(self):
        # Empty pattern is a substring of any string
        m = ModelNameMatcher("", mode="in")
        assert m(_make_model(name_or_path="anything")) is True

    def test_full_mode_requires_exact_match(self):
        m = ModelNameMatcher("facebook/opt-125m", mode="full")
        assert m(_make_model(name_or_path="facebook/opt-125m")) is True
        assert m(_make_model(name_or_path="facebook/opt-125m-fp16")) is False

    def test_regex_mode_uses_re_search(self):
        m = ModelNameMatcher(r"Qwen[23]", mode="regex")
        assert m(_make_model(name_or_path="Qwen/Qwen2-7B")) is True
        assert m(_make_model(name_or_path="Qwen/Qwen3-0.6B")) is True
        assert m(_make_model(name_or_path="facebook/opt-125m")) is False

    def test_unsupported_mode_raises(self):
        m = ModelNameMatcher("foo", mode="invalid")
        with pytest.raises(ValueError):
            m(_make_model(name_or_path="foo"))


# ==============================================================================
# ArchitectureMatcher
# ==============================================================================


class TestArchitectureMatcher:
    def test_in_mode_matches_substring(self):
        m = ArchitectureMatcher("LlamaForCausalLM", mode="in")
        assert m(_make_model(architectures=["LlamaForCausalLM"])) is True
        assert m(_make_model(architectures=["Qwen2ForCausalLM"])) is False

    def test_full_mode_requires_exact_match(self):
        m = ArchitectureMatcher("LlamaForCausalLM", mode="full")
        assert m(_make_model(architectures=["LlamaForCausalLM"])) is True
        assert m(_make_model(architectures=["LlamaForCausalLMWithX"])) is False

    def test_regex_mode_uses_re_search(self):
        m = ArchitectureMatcher(r"ForCausal", mode="regex")
        assert m(_make_model(architectures=["LlamaForCausalLM"])) is True
        assert m(_make_model(architectures=["BertLMHeadModel"])) is False

    def test_handles_missing_architectures(self):
        # When architectures is missing entirely, archs_str becomes "" so no match
        m = ArchitectureMatcher("Llama", mode="in")
        cfg = SimpleNamespace()  # no architectures attr
        model = nn.Module()
        model.config = cfg
        assert m(model) is False

    def test_unsupported_mode_raises(self):
        m = ArchitectureMatcher("foo", mode="bad")
        with pytest.raises(ValueError):
            m(_make_model(architectures=["foo"]))


# ==============================================================================
# ModelTypeMatcher
# ==============================================================================


class TestModelTypeMatcher:
    def test_in_mode_matches_substring(self):
        m = ModelTypeMatcher("qwen2", mode="in")
        assert m(_make_model(model_type="qwen2")) is True
        assert m(_make_model(model_type="qwen3")) is False

    def test_full_mode_requires_exact_match(self):
        m = ModelTypeMatcher("qwen2", mode="full")
        assert m(_make_model(model_type="qwen2")) is True
        assert m(_make_model(model_type="qwen2_moe")) is False

    def test_regex_mode(self):
        m = ModelTypeMatcher(r"qwen[23]", mode="regex")
        assert m(_make_model(model_type="qwen2")) is True
        assert m(_make_model(model_type="qwen3")) is True
        assert m(_make_model(model_type="llama")) is False

    def test_missing_model_type_returns_false(self):
        m = ModelTypeMatcher("anything", mode="in")
        cfg = SimpleNamespace()
        model = nn.Module()
        model.config = cfg
        assert m(model) is False

    def test_unsupported_mode_raises(self):
        m = ModelTypeMatcher("foo", mode="bad")
        with pytest.raises(ValueError):
            m(_make_model(model_type="foo"))


# ==============================================================================
# get_glm_flash_ignore_layers
# ==============================================================================


class TestGetGlmFlashIgnoreLayers:
    def test_default_one_dense_layer(self):
        model = _make_model()
        layers = get_glm_flash_ignore_layers(model)
        assert "layers.0.mlp" in layers

    def test_uses_first_k_dense_replace_when_present(self):
        model = _make_model()
        model.config.first_k_dense_replace = 3
        layers = get_glm_flash_ignore_layers(model)
        for i in range(3):
            assert f"layers.{i}.mlp" in layers


# ==============================================================================
# register_ignore_layers / get_predefined_ignore_layers
# ==============================================================================


class TestRegisterAndPredefinedIgnoreLayers:
    def test_register_then_match(self):
        # Register a temporary rule
        register_ignore_layers(
            matchers=[ModelTypeMatcher("test_type_xyz", mode="full")],
            ignore_layers=["custom_ignore_me"],
        )
        try:
            model = _make_model(model_type="test_type_xyz")
            layers = get_predefined_ignore_layers(model)
            assert "custom_ignore_me" in layers
        finally:
            # Pop the just-registered rule so we don't pollute later tests
            from auto_round.special_model_handler import _PRE_DEFINED_IGNORE_LAYERS

            _PRE_DEFINED_IGNORE_LAYERS.pop()

    def test_no_match_returns_default(self):
        model = _make_model(model_type="unmatched_random_type")
        layers = get_predefined_ignore_layers(model)
        # Default behavior: may be empty or have defaults — must at least be a list
        assert isinstance(layers, list)


# ==============================================================================
# update_module
# ==============================================================================


class TestUpdateModule:
    def test_gguf_format_calls_apply_replacements_with_gguf_flag(self, monkeypatch):
        original_model = object()
        replaced_model = object()
        calls = []

        def apply_replacements(model, **kwargs):
            calls.append((model, kwargs))
            return replaced_model

        class GGUFFormat:
            @staticmethod
            def is_gguf():
                return True

        monkeypatch.setattr(special_model_handler, "apply_replacements", apply_replacements)

        result = update_module(original_model, formats=[GGUFFormat()], cleanup_original=False)

        assert result is replaced_model
        assert calls == [(original_model, {"gguf_export": True})]

    def test_non_gguf_format_passes_false_flag(self, monkeypatch):
        original_model = object()
        replaced_model = object()
        calls = []

        def apply_replacements(model, **kwargs):
            calls.append((model, kwargs))
            return replaced_model

        class NonGgufFormat:
            @staticmethod
            def is_gguf():
                return False

        monkeypatch.setattr(special_model_handler, "apply_replacements", apply_replacements)

        result = update_module(original_model, formats=[NonGgufFormat()], cleanup_original=False)

        assert result is replaced_model
        assert calls == [(original_model, {"gguf_export": False})]

    def test_none_formats_passes_false_flag(self, monkeypatch):
        original_model = object()
        replaced_model = object()
        calls = []

        def apply_replacements(model, **kwargs):
            calls.append((model, kwargs))
            return replaced_model

        monkeypatch.setattr(special_model_handler, "apply_replacements", apply_replacements)
        result = update_module(original_model, formats=None, cleanup_original=False)
        assert result is replaced_model
        assert calls == [(original_model, {"gguf_export": False})]

    def test_cleanup_original_calls_release(self, monkeypatch):
        seen_release = []

        def apply_replacements(model, **kwargs):
            return model

        def release(model):
            seen_release.append(model)

        monkeypatch.setattr(special_model_handler, "apply_replacements", apply_replacements)
        monkeypatch.setattr(special_model_handler, "release_original_module_", release)

        model = object()
        update_module(model, formats=None, cleanup_original=True)
        assert seen_release == [model]


# ==============================================================================
# prepare_special_model_block_inputs
# ==============================================================================


class TestPrepareSpecialModelBlockInputs:
    def test_normalizes_position_ids_when_list(self):
        """When input_others['position_ids'] is a list, it should be unpacked/synthesized."""

        class _Block:
            pass

        block = _Block()
        rotary_input = torch.zeros(1, 4, dtype=torch.long)
        # Length-1 list -> unwrapped
        input_others = {"position_ids": [torch.arange(4).unsqueeze(0)]}
        new_input_others, _ = prepare_special_model_block_inputs(block, rotary_input, input_others)
        assert not isinstance(new_input_others["position_ids"], list)
        assert new_input_others["position_ids"].shape == (1, 4)

    def test_synthesizes_position_ids_when_none(self):
        class _Block:
            pass

        block = _Block()
        rotary_input = torch.zeros(2, 3, dtype=torch.long)
        input_others = {"position_ids": None}
        new_input_others, _ = prepare_special_model_block_inputs(block, rotary_input, input_others)
        assert new_input_others["position_ids"].shape == (2, 3)
        assert new_input_others["position_ids"].dtype == torch.long

    def test_synthesizes_position_ids_when_empty_list(self):
        class _Block:
            pass

        block = _Block()
        rotary_input = torch.zeros(1, 2, dtype=torch.long)
        input_others = {"position_ids": []}
        new_input_others, _ = prepare_special_model_block_inputs(block, rotary_input, input_others)
        assert new_input_others["position_ids"].shape == (1, 2)

    def test_no_position_ids_key_no_op(self):
        class _Block:
            pass

        block = _Block()
        rotary_input = torch.zeros(1, 2, dtype=torch.long)
        input_others = {"other": "value"}
        new_input_others, _ = prepare_special_model_block_inputs(block, rotary_input, input_others)
        # The key wasn't there, so it should not be created
        assert "position_ids" not in new_input_others
        assert new_input_others.get("other") == "value"
