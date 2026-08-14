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
"""GPU-side fast unit tests for ``auto_round.compressors.config_resolution``.

Covers ``freeze_mapping`` / ``thaw_mapping`` (incl. MappingProxyType deepcopy
dispatch), ``ResolvedScheme`` value isolation (mutating returned value must not
affect the DTO), ``FormatResolution`` post-init freezing,
``resolve_scheme_value`` (string / dict / QuantizationScheme input + error
wrapping via ``SchemeResolutionError``), and ``resolve_quantization_config``
combination logic.

All tests run in milliseconds; no model loading or GPU kernel launches.
"""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.compressors.config_resolution.contracts import (
    FormatResolution,
    LayerConfig,
    ResolvedQuantizationConfig,
    ResolvedScheme,
    freeze_block_groups,
    freeze_mapping,
    thaw_mapping,
)
from auto_round.compressors.config_resolution.errors import SchemeResolutionError
from auto_round.compressors.config_resolution.resolve import (
    resolve_quantization_config,
    resolve_scheme_value,
)
from auto_round.schemes import QuantizationScheme


def _scheme(**overrides):
    base = dict(bits=4, group_size=128, sym=True, data_type="int", act_bits=16)
    base.update(overrides)
    return QuantizationScheme.from_dict(base)


# ==============================================================================
# freeze_mapping / thaw_mapping
# ==============================================================================


class TestFreezeThawMapping:
    def test_freeze_then_thaw_roundtrip(self):
        original = {"layer1": {"bits": 4}, "layer2": {"bits": 8}}
        frozen = freeze_mapping(original)
        # Should be a read-only MappingProxyType
        with pytest.raises(TypeError):
            frozen["layer1"] = {"bits": 8}
        # Thawed back to plain dict
        thawed = thaw_mapping(frozen)
        assert thawed == original
        # Now mutable
        thawed["layer3"] = {"bits": 2}
        assert thawed == {"layer1": {"bits": 4}, "layer2": {"bits": 8}, "layer3": {"bits": 2}}

    def test_freeze_none(self):
        # None passes through freeze/thaw
        assert thaw_mapping(None) == {}

    def test_freeze_non_mapping_values_preserved(self):
        # Non-mapping values are stored as-is (e.g. GGUF preset name strings)
        original = {"layer1": "gguf:q4_k_m", "layer2": {"bits": 4}}
        frozen = freeze_mapping(original)
        # Thawed values are intact
        thawed = thaw_mapping(frozen)
        assert thawed["layer1"] == "gguf:q4_k_m"
        assert thawed["layer2"] == {"bits": 4}

    def test_freeze_deepcopy(self):
        # Frozen mapping is independent of the original
        original = {"layer": {"bits": 4}}
        frozen = freeze_mapping(original)
        # Mutating original doesn't affect frozen
        original["layer"]["bits"] = 8
        assert thaw_mapping(frozen)["layer"]["bits"] == 4

    def test_freeze_mapping_proxy_supports_deepcopy(self):
        # MappingProxyType isn't picklable by default; verify our dispatch handles it
        frozen = freeze_mapping({"a": {"bits": 4}})
        # copy.deepcopy should not raise
        copy.deepcopy(frozen)


# ==============================================================================
# freeze_block_groups
# ==============================================================================


class TestFreezeBlockGroups:
    def test_none_passthrough(self):
        assert freeze_block_groups(None) is None

    def test_tuple_of_tules(self):
        groups = (("a", "b"), ("c", "d"))
        result = freeze_block_groups(groups)
        assert result == (("a", "b"), ("c", "d"))
        assert isinstance(result, tuple)
        assert isinstance(result[0], tuple)

    def test_immutable(self):
        groups = [["a", "b"], ["c", "d"]]
        result = freeze_block_groups(groups)
        # Resulting outer tuple can't be reassigned
        with pytest.raises(TypeError):
            result[0] = ("e", "f")


# ==============================================================================
# ResolvedScheme
# ==============================================================================


class TestResolvedScheme:
    def test_value_is_deepcopied(self):
        s = _scheme()
        resolved = ResolvedScheme.from_scheme(s)
        # Mutating the returned value must not affect the DTO
        value = resolved.value
        value.bits = 8
        assert resolved.value.bits == 4

    def test_preset_name_preserved(self):
        resolved = ResolvedScheme.from_scheme(_scheme(), preset_name="W4A16")
        assert resolved.preset_name == "W4A16"

    def test_preset_name_default_none(self):
        resolved = ResolvedScheme.from_scheme(_scheme())
        assert resolved.preset_name is None

    def test_deepcopy(self):
        # ResolvedScheme is frozen; deepcopy must work via our dispatch
        resolved = ResolvedScheme.from_scheme(_scheme(), preset_name="W4A16")
        copy.deepcopy(resolved)  # must not raise


# ==============================================================================
# FormatResolution
# ==============================================================================


class TestFormatResolution:
    def test_post_init_freezes_layer_config(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        # Pass a regular dict; should be frozen by post_init
        resolution = FormatResolution(formats=(scheme,), scheme=scheme, layer_config_patch={"a": {"bits": 4}})
        # The stored layer_config_patch is a MappingProxyType
        with pytest.raises(TypeError):
            resolution.layer_config_patch["b"] = {"bits": 8}

    def test_post_init_freezes_quant_block_list(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        resolution = FormatResolution(
            formats=(scheme,),
            scheme=scheme,
            quant_block_list=(("a", "b"),),
        )
        # Tuple of tuples -> immutable
        assert isinstance(resolution.quant_block_list, tuple)
        assert isinstance(resolution.quant_block_list[0], tuple)

    def test_formats_converted_to_tuple(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        # Pass formats as a list -> post_init freezes to tuple
        resolution = FormatResolution(formats=[scheme, scheme], scheme=scheme)
        assert isinstance(resolution.formats, tuple)
        assert len(resolution.formats) == 2


# ==============================================================================
# ResolvedQuantizationConfig
# ==============================================================================


class TestResolvedQuantizationConfig:
    def test_basic_construction(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        resolution = FormatResolution(formats=(scheme,), scheme=scheme)
        cfg = ResolvedQuantizationConfig(scheme=scheme, formats=resolution.formats, layer_config={"a": {"bits": 4}})
        assert cfg.scheme is scheme
        assert cfg.formats == resolution.formats
        assert "a" in cfg.layer_config

    def test_layer_config_is_mapping_proxy(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        resolution = FormatResolution(formats=(scheme,), scheme=scheme)
        cfg = ResolvedQuantizationConfig(scheme=scheme, formats=resolution.formats, layer_config={"a": {"bits": 4}})
        with pytest.raises(TypeError):
            cfg.layer_config["b"] = {"bits": 8}

    def test_regex_config_default_empty(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        cfg = ResolvedQuantizationConfig(scheme=scheme, formats=(scheme,), layer_config={})
        assert cfg.regex_config is not None
        assert dict(cfg.regex_config) == {}

    def test_has_qlayer_outside_block_default_false(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        cfg = ResolvedQuantizationConfig(scheme=scheme, formats=(scheme,), layer_config={})
        assert cfg.has_qlayer_outside_block is False


# ==============================================================================
# resolve_scheme_value
# ==============================================================================


class TestResolveSchemeValue:
    def test_string_input(self):
        resolved = resolve_scheme_value("W4A16", {})
        assert resolved.preset_name == "W4A16"

    def test_string_input_gguf_lowercase(self):
        resolved = resolve_scheme_value("gguf:q4_k_m", {})
        assert resolved.preset_name == "gguf:q4_k_m"

    def test_dict_input(self):
        resolved = resolve_scheme_value({"bits": 4, "data_type": "int", "group_size": 128, "sym": True, "act_bits": 16}, {})
        # No preset_name since input is a dict
        assert resolved.preset_name is None

    def test_scheme_input(self):
        s = _scheme()
        resolved = resolve_scheme_value(s, {})
        assert resolved.value == s

    def test_unknown_string_raises_resolution_error(self):
        with pytest.raises(SchemeResolutionError):
            resolve_scheme_value("TOTALLY_UNKNOWN_SCHEME", {})

    def test_autoscheme_raises_resolution_error(self):
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        a = AutoScheme(avg_bits=4.0, options=["W4A16"])
        with pytest.raises(SchemeResolutionError, match="AutoScheme"):
            resolve_scheme_value(a, {})

    def test_invalid_dict_raises_resolution_error(self):
        # Empty dict has no bits/data_type -> validation may fail
        # Actually _override_scheme_with_user_specify requires bits
        with pytest.raises(SchemeResolutionError):
            resolve_scheme_value({}, {})


# ==============================================================================
# resolve_quantization_config
# ==============================================================================


class TestResolveQuantizationConfig:
    def test_layer_config_merged_with_patch(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        # Patch has layer1; caller also has layer1 with conflicting bits
        # setdefault creates the dict but update overwrites values,
        # so the caller-supplied layer_config wins on conflict.
        patch = {"layer1": {"bits": 8}}
        resolution = FormatResolution(formats=(scheme,), scheme=scheme, layer_config_patch=patch)
        cfg = resolve_quantization_config(
            resolution, {"layer2": {"bits": 4}, "layer1": {"bits": 16}}
        )
        # layer2: only caller has it
        assert cfg.layer_config["layer2"]["bits"] == 4
        # layer1: caller wins on conflict (16 overwrites patch's 8)
        assert cfg.layer_config["layer1"]["bits"] == 16

    def test_layer_config_patch_only_layer(self):
        # Layer only in patch, not in caller -> patch wins
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        patch = {"patch_only": {"bits": 8}}
        resolution = FormatResolution(formats=(scheme,), scheme=scheme, layer_config_patch=patch)
        cfg = resolve_quantization_config(resolution, {"caller_only": {"bits": 4}})
        assert cfg.layer_config["patch_only"]["bits"] == 8
        assert cfg.layer_config["caller_only"]["bits"] == 4

    def test_regex_config_default(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        resolution = FormatResolution(formats=(scheme,), scheme=scheme)
        cfg = resolve_quantization_config(resolution, {})
        assert cfg.regex_config is not None

    def test_regex_config_passed_through(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        resolution = FormatResolution(formats=(scheme,), scheme=scheme)
        regex = {"pattern1": {"bits": 4}}
        cfg = resolve_quantization_config(resolution, {}, regex_config=regex)
        assert cfg.regex_config == regex

    def test_has_qlayer_outside_block_true(self):
        s = _scheme()
        scheme = ResolvedScheme.from_scheme(s)
        resolution = FormatResolution(formats=(scheme,), scheme=scheme)
        cfg = resolve_quantization_config(resolution, {}, has_qlayer_outside_block=True)
        assert cfg.has_qlayer_outside_block is True