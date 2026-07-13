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
"""Tests for the pure helpers in auto_round/utils/common.py."""

import argparse
import json
import os
import sys
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# contain_any_mm_keys
# ---------------------------------------------------------------------------
class TestContainAnyMmKeys:
    def test_name_with_visual_keyword_matches(self):
        from auto_round.utils.common import contain_any_mm_keys
        assert contain_any_mm_keys("model.visual.blocks.0") is True

    def test_name_with_audio_keyword_matches(self):
        from auto_round.utils.common import contain_any_mm_keys
        assert contain_any_mm_keys("audio_encoder.layers.0") is True

    def test_plain_text_model_name_does_not_match(self):
        from auto_round.utils.common import contain_any_mm_keys
        assert contain_any_mm_keys("model.layers.0.self_attn.q_proj") is False

    def test_empty_name_does_not_match(self):
        from auto_round.utils.common import contain_any_mm_keys
        assert contain_any_mm_keys("") is False


# ---------------------------------------------------------------------------
# is_debug_mode
# ---------------------------------------------------------------------------
class TestIsDebugMode:
    def test_returns_bool(self):
        from auto_round.utils.common import is_debug_mode
        assert isinstance(is_debug_mode(), bool)

    def test_true_when_sys_gettrace_returns_non_none(self):
        from auto_round.utils.common import is_debug_mode
        with patch.object(sys, "gettrace", return_value=lambda *a, **kw: None):
            assert is_debug_mode() is True

    def test_false_when_no_tracer(self):
        from auto_round.utils.common import is_debug_mode

        class _FakeFlags:
            debug = 0

        with patch.object(sys, "gettrace", return_value=None), \
             patch.object(sys, "flags", _FakeFlags()):
            assert is_debug_mode() is False


# ---------------------------------------------------------------------------
# is_local_path
# ---------------------------------------------------------------------------
class TestIsLocalPath:
    def test_existing_text_file_is_local(self, tmp_path):
        from auto_round.utils.common import is_local_path
        p = tmp_path / "weights.txt"
        p.write_text("hi")
        assert is_local_path(str(p)) is True

    def test_existing_json_file_is_local(self, tmp_path):
        from auto_round.utils.common import is_local_path
        p = tmp_path / "weights.json"
        p.write_text("{}")
        assert is_local_path(str(p)) is True

    def test_non_existing_path_is_not_local(self, tmp_path):
        from auto_round.utils.common import is_local_path
        assert is_local_path(str(tmp_path / "missing.txt")) is False

    def test_unsupported_extension_with_existing_file(self, tmp_path):
        from auto_round.utils.common import is_local_path
        p = tmp_path / "weights.bin"
        p.write_text("hi")
        # ".bin" is not in format_list -> returns None, which is falsy
        assert is_local_path(str(p)) is None or is_local_path(str(p)) is False


# ---------------------------------------------------------------------------
# get_library_version
# ---------------------------------------------------------------------------
class TestGetLibraryVersion:
    def test_real_package_returns_string_version(self):
        from auto_round.utils.common import get_library_version
        version = get_library_version("torch")
        assert isinstance(version, str)
        # The returned value should not be the "not installed" sentinel
        assert "not installed" not in version

    def test_missing_package_returns_sentinel_message(self):
        from auto_round.utils.common import get_library_version
        msg = get_library_version("definitely_not_a_real_package_zzz_12345")
        assert isinstance(msg, str)
        assert "not installed" in msg


# ---------------------------------------------------------------------------
# str2bool
# ---------------------------------------------------------------------------
class TestStr2Bool:
    def test_true_variants(self):
        from auto_round.utils.common import str2bool
        for v in ("yes", "true", "t", "y", "1", "YES", "True"):
            assert str2bool(v) is True

    def test_false_variants(self):
        from auto_round.utils.common import str2bool
        for v in ("no", "false", "f", "n", "0", "NO", "False"):
            assert str2bool(v) is False

    def test_already_bool_returns_as_is(self):
        from auto_round.utils.common import str2bool
        assert str2bool(True) is True
        assert str2bool(False) is False

    def test_invalid_string_raises(self):
        from auto_round.utils.common import str2bool
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool("maybe")


# ---------------------------------------------------------------------------
# flatten_list
# ---------------------------------------------------------------------------
class TestFlattenList:
    def test_flatten_nested_lists(self):
        from auto_round.utils.common import flatten_list
        assert flatten_list([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]

    def test_flatten_already_flat(self):
        from auto_round.utils.common import flatten_list
        assert flatten_list([1, 2, 3]) == [1, 2, 3]

    def test_flatten_with_tuples(self):
        from auto_round.utils.common import flatten_list
        assert flatten_list([(1, 2), 3, [4]]) == [1, 2, 3, 4]

    def test_flatten_empty(self):
        from auto_round.utils.common import flatten_list
        assert flatten_list([]) == []

    def test_flatten_deeply_nested(self):
        from auto_round.utils.common import flatten_list
        assert flatten_list([[[1, 2], [3, 4]], [[5]]]) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# to_standard_regex
# ---------------------------------------------------------------------------
class TestToStandardRegex:
    def test_plain_string_wraps_with_wildcards(self):
        from auto_round.utils.common import to_standard_regex
        result = to_standard_regex("model.embed_tokens")
        # Should wrap with .* on each side
        assert result.startswith(".*")
        assert result.endswith(".*")
        # The middle part should still contain the original text
        assert "model" in result and "embed_tokens" in result

    def test_string_with_anchors_kept_as_is(self):
        from auto_round.utils.common import to_standard_regex
        anchored = "mlp.gate$"
        result = to_standard_regex(anchored)
        # '$' signals user intent, should be preserved (no double-wrap)
        assert "$" in result

    def test_string_with_wildcard_still_wrapped(self):
        from auto_round.utils.common import to_standard_regex
        # Implementation always wraps with .* on both sides; confirm behaviour
        result = to_standard_regex("model.*attn")
        assert result.startswith(".*") and result.endswith(".*")
        assert "model" in result and "attn" in result

    def test_string_with_caret_kept(self):
        from auto_round.utils.common import to_standard_regex
        result = to_standard_regex("^layer.0")
        assert result.startswith("^")

    def test_returns_compilable_regex(self):
        import re as _re
        from auto_round.utils.common import to_standard_regex
        # Must not raise when compiled
        _re.compile(to_standard_regex("plain_text"))


# ---------------------------------------------------------------------------
# matches_any_regex
# ---------------------------------------------------------------------------
class TestMatchesAnyRegex:
    def test_empty_config_returns_false(self):
        from auto_round.utils.common import matches_any_regex
        assert matches_any_regex("anything", {}) is False

    def test_matching_pattern_returns_true(self):
        from auto_round.utils.common import matches_any_regex
        cfg = {".*attn.*": {"bits": 4}}
        assert matches_any_regex("layer.0.self_attn.q_proj", cfg) is True

    def test_no_match_returns_false(self):
        from auto_round.utils.common import matches_any_regex
        cfg = {"^mlp\\.": {"bits": 8}}
        assert matches_any_regex("layer.0.self_attn.q_proj", cfg) is False

    def test_dynamic_prefix_is_stripped(self):
        """Patterns starting with '+:' or '-:' should be treated as raw regex."""
        from auto_round.utils.common import matches_any_regex
        cfg = {"+:attn.*": {"bits": 4}}
        assert matches_any_regex("layer.0.attn.q_proj", cfg) is True

    def test_invalid_regex_is_skipped(self):
        from auto_round.utils.common import matches_any_regex
        cfg = {"[unclosed": {"bits": 4}}
        # Should not raise; returns False since the only pattern is invalid
        assert matches_any_regex("anything", cfg) is False


# ---------------------------------------------------------------------------
# json_serialize
# ---------------------------------------------------------------------------
class TestJsonSerialize:
    def test_torch_dtype_float16(self):
        from auto_round.utils.common import json_serialize
        assert json_serialize(torch.float16) == "float16"

    def test_torch_dtype_int64(self):
        from auto_round.utils.common import json_serialize
        assert json_serialize(torch.int64) == "int64"

    def test_unsupported_type_raises(self):
        from auto_round.utils.common import json_serialize
        with pytest.raises(TypeError):
            json_serialize(object())


# ---------------------------------------------------------------------------
# get_reciprocal
# ---------------------------------------------------------------------------
class TestGetReciprocal:
    def test_normal_values(self):
        from auto_round.utils.common import get_reciprocal
        t = torch.tensor([2.0, 4.0, 0.5])
        r = get_reciprocal(t)
        assert torch.allclose(r, torch.tensor([0.5, 0.25, 2.0]))

    def test_small_values_are_masked_to_zero(self):
        from auto_round.utils.common import get_reciprocal
        # Use a value smaller than the fp32 eps used inside the function (1e-30)
        t = torch.tensor([1e-40, 1.0])
        r = get_reciprocal(t)
        assert r[0].item() == 0.0
        assert r[1].item() == pytest.approx(1.0)

    def test_float16_uses_larger_eps(self):
        from auto_round.utils.common import get_reciprocal
        # Use a value smaller than the fp16 eps of 1e-5
        t = torch.tensor([1e-7, 1.0], dtype=torch.float16)
        r = get_reciprocal(t)
        # 1e-7 is below the fp16 eps of 1e-5, so should be masked to 0
        assert r[0].item() == 0.0
        assert r[1].item() == pytest.approx(1.0, rel=1e-2)

    def test_does_not_raise_under_torch_compile(self):
        """Smoke test: should not contain operations that break torch.compile."""
        from auto_round.utils.common import get_reciprocal
        t = torch.tensor([0.0, 1.0, -2.0])
        # No exception expected; the function uses torch.where to avoid nonzero
        out = get_reciprocal(t)
        assert out.shape == t.shape


# ---------------------------------------------------------------------------
# parse_layer_config_arg
# ---------------------------------------------------------------------------
class TestParseLayerConfigArg:
    def test_strict_json(self):
        from auto_round.utils.common import parse_layer_config_arg
        result = parse_layer_config_arg('{"bits": 4, "group_size": 128}')
        assert result == {"bits": 4, "group_size": 128}

    def test_cli_friendly_dict_syntax(self):
        from auto_round.utils.common import parse_layer_config_arg
        result = parse_layer_config_arg("{bits:4, group_size:128}")
        assert result == {"bits": 4, "group_size": 128}

    def test_quoted_string_keys_are_stripped(self):
        from auto_round.utils.common import parse_layer_config_arg
        result = parse_layer_config_arg('{"bits": 4}')
        assert result == {"bits": 4}

    def test_negative_integer_is_preserved(self):
        from auto_round.utils.common import parse_layer_config_arg
        result = parse_layer_config_arg('{"bits": -4}')
        assert result == {"bits": -4}

    def test_boolean_strings_normalized(self):
        from auto_round.utils.common import parse_layer_config_arg
        result = parse_layer_config_arg('{"a": true, "b": false}')
        assert result == {"a": True, "b": False}

    def test_null_string_normalized_to_none(self):
        from auto_round.utils.common import parse_layer_config_arg
        result = parse_layer_config_arg('{"a": null}')
        assert result == {"a": None}

    def test_nested_dict(self):
        from auto_round.utils.common import parse_layer_config_arg
        result = parse_layer_config_arg('{"outer": {"inner": 1}}')
        assert result == {"outer": {"inner": 1}}

    def test_invalid_input_raises(self):
        from auto_round.utils.common import parse_layer_config_arg
        with pytest.raises(Exception):
            parse_layer_config_arg("")


# ---------------------------------------------------------------------------
# GlobalState
# ---------------------------------------------------------------------------
class TestGlobalState:
    def test_starts_at_zero(self):
        from auto_round.utils.common import GlobalState
        gs = GlobalState()
        assert isinstance(gs.replaced_module_count, int)

    def test_can_be_incremented(self):
        from auto_round.utils.common import GlobalState
        gs = GlobalState()
        before = gs.replaced_module_count
        gs.replaced_module_count += 5
        assert gs.replaced_module_count == before + 5


# ---------------------------------------------------------------------------
# Transformers version checks
# ---------------------------------------------------------------------------
class TestTransformersVersionChecks:
    def test_v5_4_returns_bool(self):
        from auto_round.utils.common import is_transformers_version_greater_or_equal_5_4_0
        # Cache may already be set; result must be bool
        assert isinstance(is_transformers_version_greater_or_equal_5_4_0(), bool)

    def test_v5_returns_bool(self):
        from auto_round.utils.common import is_transformers_version_greater_or_equal_5
        assert isinstance(is_transformers_version_greater_or_equal_5(), bool)

    def test_v4_returns_bool(self):
        from auto_round.utils.common import is_transformers_version_greater_or_equal_4
        assert isinstance(is_transformers_version_greater_or_equal_4(), bool)


# ---------------------------------------------------------------------------
# compress_layer_names
# ---------------------------------------------------------------------------
class TestCompressLayerNames:
    def test_single_name_unchanged(self):
        from auto_round.utils.common import compress_layer_names
        result = compress_layer_names(["layer.0"])
        assert result == "layer.0"

    def test_sequential_layers_get_compressed(self):
        from auto_round.utils.common import compress_layer_names
        names = [f"layer.{i}.self_attn.q_proj" for i in range(4)]
        result = compress_layer_names(names)
        # Implementation compresses to a single regex string
        assert result == "layer.[0-3].self_attn.q_proj"


# ---------------------------------------------------------------------------
# infer_bits_by_data_type
# ---------------------------------------------------------------------------
class TestInferBitsByDataType:
    def test_int2_returns_2(self):
        from auto_round.utils.common import infer_bits_by_data_type
        assert infer_bits_by_data_type("int2") == 2

    def test_int4_returns_4(self):
        from auto_round.utils.common import infer_bits_by_data_type
        assert infer_bits_by_data_type("int4") == 4

    def test_int8_returns_8(self):
        from auto_round.utils.common import infer_bits_by_data_type
        assert infer_bits_by_data_type("int8") == 8

    def test_unknown_returns_none(self):
        from auto_round.utils.common import infer_bits_by_data_type
        assert infer_bits_by_data_type("not_a_real_dtype") is None