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
"""GPU-side fast unit tests for ``auto_round.export.export_to_llmcompressor``.

Covers both ``utils.generate_ignore_regex_list`` and the
``config.initialize_quantization`` helper. These tests do not require
``compressed_tensors`` to be installed for the regex-list branch, and
they gracefully skip when the package is missing for the quantization
config branch.
"""

import pytest

from auto_round.export.export_to_llmcompressor.utils import generate_ignore_regex_list

from ...envs import is_compressed_tensors_available

pytestmark_init = pytest.mark.skipif(
    not is_compressed_tensors_available(), reason="test requires compressed-tensors"
)


# ==============================================================================
# generate_ignore_regex_list
# ==============================================================================


class TestGenerateIgnoreRegexList:
    """Test generate_ignore_regex_list function."""

    def test_empty_configs_returns_empty_list(self):
        assert generate_ignore_regex_list({}, {}) == []

    def test_regex_config_bits_above_8_added_as_regex(self):
        regex_config = {"^model\\.decoder\\.layers\\.0.*$": {"bits": 16}}
        result = generate_ignore_regex_list(regex_config, {})
        assert len(result) == 1
        assert result[0].startswith("re:")
        assert "decoder" in result[0]

    def test_regex_config_bits_at_or_below_8_ignored(self):
        regex_config = {
            "low_bits": {"bits": 8},
            "high_bits": {"bits": 16},
        }
        result = generate_ignore_regex_list(regex_config, {})
        # Only high_bits should be added; low_bits is at the boundary (8) and
        # is therefore excluded by the strict-greater-than 8 check.
        assert len(result) == 1
        assert "high_bits" in result[0]

    def test_layer_config_above_8_added_as_full_name(self):
        layer_config = {"lm_head": {"bits": 16}}
        result = generate_ignore_regex_list({}, layer_config)
        assert "lm_head" in result

    def test_combined_regex_and_layer_config(self):
        regex_config = {"r_pattern": {"bits": 16}}
        layer_config = {"full_name_layer": {"bits": 16}}
        result = generate_ignore_regex_list(regex_config, layer_config)
        assert len(result) == 2
        # one item should be the regex pattern with re: prefix, one should be the full name
        assert any(item.startswith("re:") for item in result)
        assert any(item == "full_name_layer" for item in result)

    def test_layer_config_below_8_not_added(self):
        layer_config = {"keep_layer": {"bits": 4}, "drop_layer": {"bits": 16}}
        result = generate_ignore_regex_list({}, layer_config)
        assert "keep_layer" not in result
        assert "drop_layer" in result


# ==============================================================================
# initialize_quantization
# ==============================================================================


@pytestmark_init
class TestInitializeQuantization:
    """Test initialize_quantization function in config.py."""

    def test_preset_scheme_mxfp4(self):
        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        config = initialize_quantization(scheme="MXFP4")
        assert config is not None
        # Round-trip via dict to confirm structure
        d = config.to_dict()
        assert "config_groups" in d
        assert "group_0" in d["config_groups"]

    def test_preset_scheme_mxfp8_uses_mxfp4_fallback(self):
        """mxfp8 is not natively supported; the helper rewrites it to mxfp4 + num_bits=8."""
        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        config = initialize_quantization(scheme="MXFP8")
        d = config.to_dict()
        assert d["config_groups"]["group_0"]["weights"]["num_bits"] == 8
        assert d["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8

    def test_preset_scheme_nvfp4(self):
        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        config = initialize_quantization(scheme="NVFP4")
        d = config.to_dict()
        assert "config_groups" in d
        assert "group_0" in d["config_groups"]

    def test_explicit_config_groups(self):
        from compressed_tensors.quantization import QuantizationScheme  # noqa: E0401

        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        scheme = QuantizationScheme(targets=["Linear"], weights={"num_bits": 4})
        config = initialize_quantization(scheme=None, config_groups={"group_0": scheme})
        d = config.to_dict()
        assert "group_0" in d["config_groups"]

    def test_both_scheme_and_config_groups_raises(self):
        from compressed_tensors.quantization import QuantizationScheme  # noqa: E0401

        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        scheme = QuantizationScheme(targets=["Linear"])
        with pytest.raises(ValueError, match="either `scheme` or `config_groups`"):
            initialize_quantization(scheme="MXFP4", config_groups={"group_0": scheme})

    def test_neither_scheme_nor_config_groups_uses_default(self):
        from auto_round.export.export_to_llmcompressor.config import initialize_quantization

        # scheme=None + config_groups=None -> default QuantizationScheme is used
        config = initialize_quantization(scheme=None, config_groups=None)
        d = config.to_dict()
        assert "group_0" in d["config_groups"]
