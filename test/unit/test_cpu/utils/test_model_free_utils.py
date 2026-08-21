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

"""Unit tests for model_free_utils helpers."""

from unittest.mock import Mock

import pytest
import torch

from auto_round.compressors.model_free import _ModelFreeCompressorCore
from auto_round.schemes import QuantizationScheme
from auto_round.utils.model_free_utils import (
    _add_routed_experts_if_moe,
    _build_mxfp_autoround_quantization_config,
    _build_mxfp_quantization_config,
    _build_quantization_config,
    _convert_auto_scheme_layer_config,
    _dequant_mxfp_tensors,
    _expand_e8m0_block_scale,
    _handle_mxfp_source_tensors,
    _looks_like_auto_scheme,
    _PatternMatcher,
    _quantize_weight_mxfp,
    _validate_auto_scheme_options,
)
from auto_round.utils.model_free_utils import (
    preprocess_model_type_source_tensors as _preprocess_model_type_source_tensors,
)

from ...envs import require_compressed_tensors

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _matcher(ignore=None, layer_config=None, default=None):
    return _PatternMatcher(ignore or [], layer_config or {}, default or {})


# ---------------------------------------------------------------------------
# _add_routed_experts_if_moe
# TODO: Remove when the issue is fixed
#   https://github.com/vllm-project/llm-compressor/issues/3069
# ---------------------------------------------------------------------------


class TestAddRoutedExpertsIfMoe:
    """Unit tests for the _add_routed_experts_if_moe helper.

    TODO: Remove when the issue is fixed
    https://github.com/vllm-project/llm-compressor/issues/3069
    """

    def test_w1_pattern_triggers(self):
        targets = ["Linear"]
        layers = ["model.layers.0.mlp.experts.0.w1.weight"]
        result = _add_routed_experts_if_moe(targets, layers)
        assert "RoutedExperts" in result

    def test_w3_pattern_triggers(self):
        targets = ["Linear"]
        layers = ["model.layers.0.mlp.experts.0.w3.weight"]
        result = _add_routed_experts_if_moe(targets, layers)
        assert "RoutedExperts" in result

    def test_block_sparse_moe_pattern_triggers(self):
        targets = ["Linear"]
        layers = ["model.layers.0.block_sparse_moe.experts.0.w1.weight"]
        result = _add_routed_experts_if_moe(targets, layers)
        assert "RoutedExperts" in result

    def test_non_moe_no_change(self):
        targets = ["Linear"]
        layers = ["model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.up_proj.weight"]
        result = _add_routed_experts_if_moe(targets, layers)
        assert result == ["Linear"]
        assert "RoutedExperts" not in result

    def test_w2_alone_does_not_trigger(self):
        """w2 (down-projection) alone should not trigger RoutedExperts."""
        targets = ["Linear"]
        layers = ["model.layers.0.mlp.experts.0.w2.weight"]
        result = _add_routed_experts_if_moe(targets, layers)
        assert "RoutedExperts" not in result

    def test_already_present_not_duplicated(self):
        targets = ["Linear", "RoutedExperts"]
        layers = ["model.layers.0.mlp.experts.0.w1.weight"]
        result = _add_routed_experts_if_moe(targets, layers)
        assert result.count("RoutedExperts") == 1

    def test_original_list_not_mutated(self):
        targets = ["Linear"]
        layers = ["model.layers.0.mlp.experts.0.w1.weight"]
        _add_routed_experts_if_moe(targets, layers)
        assert targets == ["Linear"]

    def test_mixed_moe_and_non_moe_layers_triggers(self):
        """Even if only some layers are MoE, RoutedExperts should be injected."""
        targets = ["Linear"]
        layers = [
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.mlp.experts.0.w1.weight",
        ]
        result = _add_routed_experts_if_moe(targets, layers)
        assert "RoutedExperts" in result

    def test_empty_layers_no_change(self):
        targets = ["Linear"]
        result = _add_routed_experts_if_moe(targets, [])
        assert result == ["Linear"]


# ---------------------------------------------------------------------------
# _build_mxfp_quantization_config: RoutedExperts in single-scheme groups
# TODO: Remove when the issue is fixed
#   https://github.com/vllm-project/llm-compressor/issues/3069
# ---------------------------------------------------------------------------

# MoE-like layer names (w1/w3)
_MOE_W1_W3_LAYERS = [
    "model.layers.0.mlp.experts.0.w1",
    "model.layers.0.mlp.experts.0.w2",
    "model.layers.0.mlp.experts.0.w3",
    "model.layers.1.mlp.experts.0.w1",
    "model.layers.1.mlp.experts.0.w2",
    "model.layers.1.mlp.experts.0.w3",
]

_DENSE_LAYERS = [
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.gate_proj",
]


def _get_targets_from_qconfig(qconfig: dict, group: str = "group_0") -> list:
    return qconfig.get("config_groups", {}).get(group, {}).get("targets", [])


@pytest.mark.parametrize(
    "scheme_name,bits",
    [
        ("MXFP4", 4),
        ("MXFP8", 8),
    ],
)
class TestBuildMxfpQuantizationConfigSingleSchemeMoe:
    """Test RoutedExperts injection for single-group MXFP configs.

    TODO: Remove when the issue is fixed
    https://github.com/vllm-project/llm-compressor/issues/3069
    """

    def _default_scheme(self, bits):
        return {"bits": bits, "data_type": "mx_fp", "group_size": 32}

    def test_moe_layers_add_routed_experts(self, scheme_name, bits):
        """BF16-default with only MoE layers quantized via layer_config should
        include RoutedExperts in the single config group's targets."""
        # Use block_sparse_moe-style names so the layer_config pattern can
        # actually be matched by _PatternMatcher (layer ends with .w1 so
        # .w1. does NOT match as substring; block_sparse_moe does).
        moe_quantized = [
            "model.layers.0.block_sparse_moe.experts.0.w1",
            "model.layers.0.block_sparse_moe.experts.0.w3",
        ]
        default_scheme = {"bits": 16, "data_type": "float", "group_size": -1}
        layer_cfg = {".block_sparse_moe.": {"bits": bits, "data_type": "mx_fp", "group_size": 32}}
        qconfig = _build_mxfp_quantization_config(
            default_scheme=default_scheme,
            quantized_layers=moe_quantized,
            ignored_layers=[],
            layer_config=layer_cfg,
        )
        targets = _get_targets_from_qconfig(qconfig, "group_0")
        assert "RoutedExperts" in targets, f"Expected RoutedExperts in targets, got: {targets}"

    def test_dense_layers_no_routed_experts(self, scheme_name, bits):
        """Dense-only quantized layers should NOT get RoutedExperts."""
        # Use the exact layer name as key so PatternMatcher can match it
        # (suffix patterns like .down_proj. don't match layer names that end
        # with down_proj since the compiled regex requires a trailing dot).
        dense_layer = "model.layers.0.mlp.down_proj"
        default_scheme = {"bits": 16, "data_type": "float", "group_size": -1}
        layer_cfg = {dense_layer: {"bits": bits, "data_type": "mx_fp", "group_size": 32}}
        qconfig = _build_mxfp_quantization_config(
            default_scheme=default_scheme,
            quantized_layers=[dense_layer],
            ignored_layers=[],
            layer_config=layer_cfg,
        )
        targets = _get_targets_from_qconfig(qconfig, "group_0")
        assert "RoutedExperts" not in targets, f"Unexpected RoutedExperts in targets: {targets}"

    def test_default_mxfp_global_targets_not_affected(self, scheme_name, bits):
        """When the default scheme is already MXFP (not BF16), targets stay as
        ``['Linear']`` for a uniform-scheme model without layer_config."""
        default_scheme = {"bits": bits, "data_type": "mx_fp", "group_size": 32}
        qconfig = _build_mxfp_quantization_config(
            default_scheme=default_scheme,
            quantized_layers=_MOE_W1_W3_LAYERS,
            ignored_layers=[],
            layer_config=None,
        )
        targets = _get_targets_from_qconfig(qconfig, "group_0")
        # Uniform-scheme → targets should be the catch-all ["Linear"], no explicit layer list
        assert targets == ["Linear"]


# ---------------------------------------------------------------------------
# _build_mxfp_quantization_config: RoutedExperts in mixed-precision groups
# TODO: Remove when the issue is fixed
#   https://github.com/vllm-project/llm-compressor/issues/3069
# ---------------------------------------------------------------------------


class TestBuildMxfpQuantizationConfigMixedPrecisionMoe:
    """RoutedExperts must be injected per config_group in mixed-precision configs.

    TODO: Remove when the issue is fixed
    https://github.com/vllm-project/llm-compressor/issues/3069
    """

    def test_mixed_moe_groups_both_get_routed_experts(self):
        """Mixed precision: MXFP4 default + MXFP8 dense override.
        The MXFP8 override group has an explicit dense-layer list;
        it must NOT contain RoutedExperts.
        The MXFP4 default group uses ["Linear"] — RoutedExperts not needed."""
        moe_layers = [
            "model.layers.0.block_sparse_moe.experts.0.w1",
            "model.layers.0.block_sparse_moe.experts.0.w3",
        ]
        dense_layer = "model.layers.0.mlp.down_proj"
        all_layers = moe_layers + [dense_layer]

        default_scheme = {"bits": 4, "data_type": "mx_fp", "group_size": 32}
        # Override down_proj to MXFP8 so it lands in an override group.
        layer_cfg = {dense_layer: {"bits": 8, "data_type": "mx_fp", "group_size": 32}}
        qconfig = _build_mxfp_quantization_config(
            default_scheme=default_scheme,
            quantized_layers=all_layers,
            ignored_layers=[],
            layer_config=layer_cfg,
        )
        assert qconfig.get("format") == "mixed-precision"
        config_groups = qconfig.get("config_groups", {})
        # The override group (MXFP8) has only dense_layer — no RoutedExperts.
        for grp in config_groups.values():
            t = grp.get("targets", [])
            if t != ["Linear"] and dense_layer in str(t):
                assert "RoutedExperts" not in t, f"Dense override group should not have RoutedExperts: {t}"

    def test_mixed_moe_override_group_gets_routed_experts(self):
        """When MoE layers are in an *override* group (not the default ["Linear"]
        catch-all), the explicit targets list must include RoutedExperts."""
        # default=MXFP8, block_sparse_moe layers override to MXFP4
        # → they land in the override group with an explicit targets list.
        moe_layers = [
            "model.layers.0.block_sparse_moe.experts.0.w1",
            "model.layers.0.block_sparse_moe.experts.0.w3",
        ]
        dense_layers = [
            "model.layers.0.mlp.down_proj",
            "model.layers.0.mlp.up_proj",
        ]
        all_layers = moe_layers + dense_layers

        default_scheme = {"bits": 8, "data_type": "mx_fp", "group_size": 32}
        layer_cfg = {
            ".block_sparse_moe.": {"bits": 4, "data_type": "mx_fp", "group_size": 32},
        }
        qconfig = _build_mxfp_quantization_config(
            default_scheme=default_scheme,
            quantized_layers=all_layers,
            ignored_layers=[],
            layer_config=layer_cfg,
        )
        assert qconfig.get("format") == "mixed-precision"
        config_groups = qconfig.get("config_groups", {})
        # Find the override group (MXFP4, explicit layer list) containing MoE layers.
        moe_group_targets = None
        for grp in config_groups.values():
            t = grp.get("targets", [])
            if t != ["Linear"] and any("block_sparse_moe" in str(x) for x in t):
                moe_group_targets = t
                break
        assert moe_group_targets is not None, "Expected a non-default group for MoE layers"
        assert "RoutedExperts" in moe_group_targets, f"Missing RoutedExperts in: {moe_group_targets}"


# ---------------------------------------------------------------------------
# _build_quantization_config: BF16-default + NVFP4 llm_compressor format
# TODO: Remove when the issue is fixed
#   https://github.com/vllm-project/llm-compressor/issues/3069
# ---------------------------------------------------------------------------

# Only the w1/w3 layers that actually match the nv_fp layer_config are passed
# as quantized_layers (w2 stays at BF16 in this scenario).
_MOE_NV_QUANTIZED = [
    "model.layers.0.mlp.experts.0.w1",
    "model.layers.0.mlp.experts.0.w3",
    "model.layers.1.mlp.experts.0.w1",
    "model.layers.1.mlp.experts.0.w3",
]


class TestBuildQuantizationConfigNvfp4MoeLlmCompressor:
    """RoutedExperts must be present when BF16-default with NVFP4 layer_config
    is exported in llm_compressor format and quantized layers are MoE.

    TODO: Remove when the issue is fixed
    https://github.com/vllm-project/llm-compressor/issues/3069
    """

    def test_nvfp4_moe_adds_routed_experts(self):
        default_scheme = {"bits": 16, "data_type": "float", "group_size": -1}
        layer_cfg = {
            ".w1.": {"bits": 4, "data_type": "nv_fp", "group_size": 16},
            ".w3.": {"bits": 4, "data_type": "nv_fp", "group_size": 16},
        }
        qconfig = _build_quantization_config(
            default_scheme=default_scheme,
            layer_config=layer_cfg,
            ignore_patterns=[],
            quantized_layers=_MOE_NV_QUANTIZED,
            ignored_layers=[],
            format="llm_compressor",
        )
        targets = _get_targets_from_qconfig(qconfig, "group_0")
        assert "RoutedExperts" in targets, f"Expected RoutedExperts in targets, got: {targets}"

    def test_nvfp4_dense_no_routed_experts(self):
        default_scheme = {"bits": 16, "data_type": "float", "group_size": -1}
        layer_cfg = {".down_proj.": {"bits": 4, "data_type": "nv_fp", "group_size": 16}}
        qconfig = _build_quantization_config(
            default_scheme=default_scheme,
            layer_config=layer_cfg,
            ignore_patterns=[],
            quantized_layers=["model.layers.0.mlp.down_proj"],
            ignored_layers=[],
            format="llm_compressor",
        )
        targets = _get_targets_from_qconfig(qconfig, "group_0")
        assert "RoutedExperts" not in targets, f"Unexpected RoutedExperts in targets: {targets}"


# ===========================================================================
#  _PatternMatcher
# ===========================================================================


class TestPatternMatcher:
    def test_ignore_substring(self):
        m = _matcher(ignore=["mlp"])
        assert m.should_ignore("model.layers.0.mlp.fc1.weight") is True
        assert m.should_ignore("model.layers.0.self_attn.q_proj.weight") is False

    def test_ignore_trailing_dot(self):
        m = _matcher(ignore=["layers.4."])
        assert m.should_ignore("model.layers.4.mlp.fc1.weight") is True
        assert m.should_ignore("model.layers.45.mlp.fc1.weight") is False

    def test_skip_predefined(self):
        m = _matcher()
        assert m.should_skip("model.layers.0.shared_expert_gate.weight") is True
        assert m.should_skip("model.layers.0.mlp.gate.weight") is True
        assert m.should_skip("model.layers.0.mlp.gate_proj.weight") is False
        assert m.should_skip("model.embed_tokens.weight") is True
        assert m.should_skip("model.layers.0.mlp.fc1.weight") is False

    def test_resolve_scheme_exact_regex_and_default(self):
        default = {"bits": 4, "group_size": 128, "sym": True}
        lc = {
            "model.layers.0.mlp.fc1": {"bits": 8, "group_size": 32},
            r".*k_proj": {"bits": 8},
        }
        m = _matcher(layer_config=lc, default=default)
        assert m.resolve_scheme("model.layers.0.mlp.fc1.weight")["bits"] == 8
        assert m.resolve_scheme("model.layers.0.self_attn.k_proj.weight")["bits"] == 8
        assert m.resolve_scheme("model.layers.0.mlp.fc2.weight") == default

    def test_resolve_bits16_returns_none(self):
        m = _matcher(layer_config={"model.layers.0.fc1": {"bits": 16}}, default={"bits": 4, "group_size": 128})
        assert m.resolve_scheme("model.layers.0.fc1.weight") is None

    def test_resolve_substring_pattern(self):
        default = {"bits": 4, "group_size": 128, "sym": True}
        m = _matcher(layer_config={".ffn.experts.": {"bits": 2, "group_size": 64}}, default=default)
        r = m.resolve_scheme("model.layers.0.ffn.experts.3.gate_proj.weight")
        assert r["bits"] == 2 and r["group_size"] == 64
        assert m.resolve_scheme("model.layers.0.self_attn.q_proj.weight") == default


# ===========================================================================
#  _parse_layer_config — scheme key resolution
# ===========================================================================


class TestParseLayerConfig:
    @staticmethod
    def _make_core(layer_config_input, scheme="W4A16"):
        core = _ModelFreeCompressorCore(
            model_name_or_path="dummy",
            output_dir="dummy_out",
            scheme=scheme,
        )
        core.layer_config_input = layer_config_input
        core._parse_scheme()
        core._parse_layer_config()
        return core

    def test_scheme_key_resolves(self):
        core = self._make_core({".ffn.experts.": {"scheme": "W2A16"}})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2 and "scheme" not in cfg

        m = _matcher(layer_config=core.layer_config, default=core.default_scheme)
        assert m.resolve_scheme("model.layers.0.ffn.experts.3.gate_proj.weight")["bits"] == 2

    def test_scheme_key_with_overrides(self):
        core = self._make_core({".ffn.experts.": {"scheme": "W2A16", "group_size": 32}})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2 and cfg["group_size"] == 32

    def test_string_value(self):
        core = self._make_core({".ffn.experts.": "W2A16"})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2

    def test_quantization_scheme_value(self):
        core = self._make_core({".ffn.experts.": QuantizationScheme(bits=2, group_size=64)})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2 and cfg["group_size"] == 64

    def test_w4a16_mixed_recipe_in_model_free(self):
        core = self._make_core({}, scheme="W4A16_MIXED")
        assert core.default_scheme["bits"] == 8
        assert core.layer_config[".experts."]["bits"] == 4
        assert core.layer_config[".moe."]["bits"] == 4
        assert core.layer_config[".shared_expert."]["bits"] == 8


# ===========================================================================
#  _build_quantization_config (generic INT path)
# ===========================================================================


class TestBuildQuantizationConfig:
    def test_extra_config_filters_embed_conv_only(self):
        default = {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"}
        ignored = [
            "model.embed_tokens",
            "model.conv1",
            "model.layers.0.shared_expert_gate",
            "model.layers.0.mlp.gate",
        ]

        cfg = _build_quantization_config(
            default_scheme=default,
            layer_config={},
            ignore_patterns=[],
            quantized_layers=[],
            ignored_layers=ignored,
        )

        extra = cfg.get("extra_config", {})
        assert "model.embed_tokens" not in extra
        assert "model.conv1" not in extra
        assert extra["model.layers.0.shared_expert_gate"] == {"bits": 16, "data_type": "float"}
        assert extra["model.layers.0.mlp.gate"] == {"bits": 16, "data_type": "float"}


# ===========================================================================
#  _quantize_weight_mxfp
# ===========================================================================


class TestQuantizeWeightMXFP:
    def test_disable_opt_rtn_skips_mxfp_scale_search(self, monkeypatch):
        from auto_round.data_type import mxfp

        search_mock = Mock(side_effect=AssertionError("optimized RTN must be disabled"))
        monkeypatch.setattr(mxfp, "quant_mx_opt_rtn", search_mock)

        weight = torch.randn(8, 32, dtype=torch.bfloat16)
        out = _quantize_weight_mxfp(
            weight,
            "layer",
            bits=4,
            group_size=32,
            data_type="mx_fp",
            disable_opt_rtn=True,
        )

        assert "layer.weight_packed" in out
        search_mock.assert_not_called()

    def test_quantize_weight_mxfp4_shapes(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        out = _quantize_weight_mxfp(w, "layer", bits=4, group_size=32, data_type="mx_fp")
        assert out["layer.weight_packed"].shape == (64, 64)
        assert out["layer.weight_packed"].dtype == torch.uint8
        assert out["layer.weight_scale"].shape == (64, 4)
        assert out["layer.weight_scale"].dtype == torch.uint8

    def test_quantize_weight_mxfp8_shapes(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        out = _quantize_weight_mxfp(w, "layer", bits=8, group_size=32, data_type="mx_fp")
        assert out["layer.weight"].shape == (64, 128)
        assert out["layer.weight"].dtype == torch.float8_e4m3fn
        assert out["layer.weight_scale"].shape == (64, 4)
        assert out["layer.weight_scale"].dtype == torch.uint8


# ===========================================================================
#  _build_mxfp_quantization_config (non-MoE, structural tests)
# ===========================================================================


class TestBuildMxfpConfig:
    @require_compressed_tensors
    def test_build_mxfp_mixed_config_uniform(self):
        default = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        quantized = ["model.layers.0.fc1", "model.layers.0.fc2"]
        ignored = ["lm_head"]
        cfg = _build_mxfp_quantization_config(default, quantized, ignored, layer_config={})
        assert cfg["format"] == "mxfp4-pack-quantized"
        assert "lm_head" in cfg["ignore"]
        assert len(cfg["config_groups"]) == 1

    @require_compressed_tensors
    def test_build_mxfp_config_bf16_default_targets_only_overrides(self):
        default = {"bits": 16, "group_size": -1, "sym": True, "data_type": "bf16"}
        layer_config = {
            "model.layers.0.self_attn.q_proj": {"bits": 4, "group_size": 32, "data_type": "mx_fp"},
        }
        quantized = ["model.layers.0.self_attn.q_proj"]
        cfg = _build_mxfp_quantization_config(default, quantized, [], layer_config=layer_config)
        assert cfg["format"] == "mxfp4-pack-quantized"
        assert cfg["config_groups"]["group_0"]["targets"] == quantized

    @require_compressed_tensors
    def test_build_mxfp_mixed_config_two_groups(self):
        default = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        layer_config = {
            "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
            "model.layers.0.self_attn.k_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
        }
        quantized = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.mlp.fc1",
            "model.layers.0.mlp.fc2",
        ]
        ignored = ["lm_head"]
        cfg = _build_mxfp_quantization_config(default, quantized, ignored, layer_config=layer_config)
        assert cfg["format"] == "mixed-precision"
        assert len(cfg["config_groups"]) == 2
        mxfp8_group = next(g for g in cfg["config_groups"].values() if g["format"] == "mxfp8-quantized")
        assert set(mxfp8_group["targets"]) == {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        }
        mxfp4_group = next(g for g in cfg["config_groups"].values() if g["format"] == "mxfp4-pack-quantized")
        assert mxfp4_group["targets"] == ["Linear"]


# ===========================================================================
#  _expand_e8m0_block_scale
# ===========================================================================


class TestExpandE8M0BlockScale:
    def test_expand_repeat_interleave(self):
        scale = torch.tensor([[100, 101], [102, 103]], dtype=torch.uint8)
        out = _expand_e8m0_block_scale(scale, out_features=64, in_features=128, group_size=32)
        assert out.shape == (64, 4)
        assert out.dtype == torch.uint8
        assert out[0].tolist() == [100, 100, 101, 101]
        assert out[63].tolist() == [102, 102, 103, 103]

    def test_expand_noop_when_already_fine(self):
        scale = torch.full((64, 4), 127, dtype=torch.uint8)
        out = _expand_e8m0_block_scale(scale, out_features=64, in_features=128, group_size=32)
        assert out.shape == (64, 4) and torch.equal(out, scale)

    def test_expand_invalid_shape_raises(self):
        scale = torch.full((3, 4), 127, dtype=torch.uint8)
        with pytest.raises(ValueError):
            _expand_e8m0_block_scale(scale, out_features=64, in_features=128, group_size=32)


# ===========================================================================
#  _handle_mxfp_source_tensors / _dequant_mxfp_tensors
# ===========================================================================

_DEEPSEEK_V4_CFG = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}


def _make_deepseek_v4_mxfp8(out_f, in_f, block_h, block_w):
    weight_fp8 = torch.randn(out_f, in_f, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale = torch.full((out_f // block_h, in_f // block_w), 127, dtype=torch.uint8)
    return weight_fp8, scale


class TestHandleMXFPSourceTensors:
    def test_passthrough_mxfp8_same_target(self):
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {"layer.weight": weight_fp8.clone(), "layer.weight_scale": weight_scale.clone()}
        matcher = _matcher(default={"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)
        assert layers == ["layer"]
        assert passthrough["layer.weight"].dtype == torch.float8_e4m3fn
        assert torch.equal(passthrough["layer.weight"].view(torch.uint8), weight_fp8.view(torch.uint8))
        assert passthrough["layer.weight_scale"].dtype == torch.uint8
        assert "layer.weight" not in raw_out
        assert "layer.weight_scale" not in raw_out

    def test_passthrough_mxfp4_same_target(self):
        weight_packed = torch.randint(0, 255, (64, 64), dtype=torch.uint8)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {"layer.weight_packed": weight_packed.clone(), "layer.weight_scale": weight_scale.clone()}
        matcher = _matcher(default={"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)
        assert layers == ["layer"]
        assert torch.equal(passthrough["layer.weight_packed"], weight_packed)
        assert passthrough["layer.weight_scale"].dtype == torch.uint8
        assert "layer.weight_packed" not in raw_out

    def test_mxfp8_dequant_when_int_target(self):
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {"layer.weight": weight_fp8.clone(), "layer.weight_scale": weight_scale.clone()}
        matcher = _matcher(default={"bits": 4, "group_size": 128, "sym": True, "data_type": "int"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)
        assert layers == [] and passthrough == {}
        assert raw_out["layer.weight"].dtype == torch.bfloat16
        assert torch.allclose(raw_out["layer.weight"], weight_fp8.to(torch.bfloat16))
        assert "layer.weight_scale" not in raw_out

    def test_mixed_passthrough_and_dequant(self):
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_fp8 = torch.full((64, 4), 127, dtype=torch.uint8)
        weight_packed = torch.randint(0, 255, (64, 64), dtype=torch.uint8)
        scale_packed = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {
            "attn.weight": weight_fp8.clone(),
            "attn.weight_scale": scale_fp8.clone(),
            "mlp.weight_packed": weight_packed.clone(),
            "mlp.weight_scale": scale_packed.clone(),
        }
        matcher = _matcher(default={"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)
        assert "attn" in layers
        assert passthrough["attn.weight"].dtype == torch.float8_e4m3fn
        assert "mlp" not in layers
        assert raw_out["mlp.weight"].dtype == torch.bfloat16
        assert raw_out["mlp.weight"].shape == (64, 128)

    def test_noop_without_mxfp_tensors(self):
        raw = {"layer.weight": torch.randn(64, 128, dtype=torch.bfloat16)}
        matcher = _matcher(default={"bits": 8, "group_size": 32, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)
        assert raw_out is raw and passthrough == {} and layers == []

    def test_reject_non_mxfp_packed_float_scale(self):
        raw = {
            "layer.weight_packed": torch.randint(0, 255, (64, 64), dtype=torch.uint8),
            "layer.weight_scale": torch.ones(64, 8, dtype=torch.float16),
        }
        matcher = _matcher(default={"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)
        assert raw_out is raw and passthrough == {} and layers == []

    def test_convert_passthrough_when_target_mxfp8(self):
        weight_fp8, scale = _make_deepseek_v4_mxfp8(64, 128, block_h=32, block_w=64)
        raw = {"layer.weight": weight_fp8.clone(), "layer.scale": scale.clone()}
        matcher = _matcher(default={"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, state = _preprocess_model_type_source_tensors(raw, model_type="deepseek_v4")
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw_out, matcher, source_state=state)
        assert layers == ["layer"]
        assert passthrough["layer.weight"].dtype == torch.float8_e4m3fn
        assert torch.equal(passthrough["layer.weight"].view(torch.uint8), weight_fp8.view(torch.uint8))
        assert passthrough["layer.weight_scale"].dtype == torch.uint8
        assert passthrough["layer.weight_scale"].shape == (64, 4)
        assert "layer.weight" not in raw_out and "layer.scale" not in raw_out

    def test_convert_dequant_when_target_int(self):
        weight_fp8, scale = _make_deepseek_v4_mxfp8(64, 128, block_h=32, block_w=64)
        raw = {"layer.weight": weight_fp8.clone(), "layer.scale": scale.clone()}
        matcher = _matcher(default={"bits": 4, "group_size": 128, "sym": True, "data_type": "int"})
        raw_out, state = _preprocess_model_type_source_tensors(raw, model_type="deepseek_v4")
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw_out, matcher, source_state=state)
        assert layers == [] and passthrough == {}
        assert raw_out["layer.weight"].dtype == torch.bfloat16
        assert raw_out["layer.weight"].shape == (64, 128)
        assert torch.allclose(raw_out["layer.weight"], weight_fp8.to(torch.bfloat16))
        assert "layer.scale" not in raw_out and "layer.weight_scale" not in raw_out

    def test_convert_noop_without_quantized(self):
        raw = {"layer.weight": torch.randn(64, 128, dtype=torch.bfloat16)}
        matcher = _matcher(default={"bits": 8, "group_size": 32, "data_type": "mx_fp"})
        raw_out, state = _preprocess_model_type_source_tensors(raw, model_type="deepseek_v4")
        assert state == {}
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw_out, matcher, source_state=state)
        assert raw_out is raw and passthrough == {} and layers == []

    def test_dequant_mxfp_tensors_mxfp8(self):
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {"layer.weight": weight_fp8.clone(), "layer.weight_scale": weight_scale.clone()}
        out = _dequant_mxfp_tensors(raw)
        assert out["layer.weight"].dtype == torch.bfloat16
        assert torch.allclose(out["layer.weight"], weight_fp8.to(torch.bfloat16))
        assert "layer.weight_scale" not in out


# ===========================================================================
#  AutoScheme helpers
# ===========================================================================


class TestAutoSchemeHelpers:
    def test_looks_like_auto_scheme(self):
        from auto_round import AutoScheme

        assert _looks_like_auto_scheme(AutoScheme(avg_bits=3, options=("W2A16", "W4A16")))
        assert not _looks_like_auto_scheme("W4A16")
        assert not _looks_like_auto_scheme(QuantizationScheme(bits=4))

    def test_validate_options_int_family(self):
        from auto_round import AutoScheme

        assert _validate_auto_scheme_options(AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "W8A16"))) == "int"

    def test_validate_options_int_family_with_bf16(self):
        from auto_round import AutoScheme

        assert _validate_auto_scheme_options(AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))) == "int"

    def test_validate_options_mxfp_family(self):
        from auto_round import AutoScheme

        assert _validate_auto_scheme_options(AutoScheme(avg_bits=6, options=("MXFP4", "MXFP8"))) == "mx_fp"

    def test_validate_options_mixed_family_raises(self):
        from auto_round import AutoScheme

        with pytest.raises(ValueError, match="mix INT and MXFP"):
            _validate_auto_scheme_options(AutoScheme(avg_bits=4, options=("W4A16", "MXFP4")))

    @pytest.mark.parametrize(
        "options",
        [
            ("W3A16", "W4A16"),
            ("GGUF:Q4_K_M", "W8A16"),
            ("NVFP4", "W4A16"),
            ("MXFP4_RCEIL", "MXFP4"),
        ],
    )
    def test_validate_options_unsupported_raises(self, options):
        from auto_round import AutoScheme

        with pytest.raises(ValueError, match="unsupported option"):
            _validate_auto_scheme_options(AutoScheme(avg_bits=4, options=options))

    def test_convert_layer_config(self):
        generated = {
            "model.layers.0.q_proj": {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
            "model.layers.0.k_proj": {"bits": 2, "group_size": 128, "sym": True, "data_type": "int"},
            "model.layers.0.v_proj": {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
            "model.embed_tokens": {"bits": 16, "group_size": 128, "sym": True, "data_type": "int"},
        }
        base_scheme, per_layer, fp16_layers = _convert_auto_scheme_layer_config(generated)
        assert base_scheme.bits == 4 and base_scheme.group_size == 128
        assert per_layer["model.layers.0.k_proj"]["bits"] == 2
        assert "model.embed_tokens" not in per_layer
        assert fp16_layers == ["model.embed_tokens"]

    def test_convert_layer_config_infers_mxfp_bits_from_dtype_alias(self):
        generated = {
            "model.layers.0.q_proj": {"group_size": 32, "sym": True, "data_type": "mxfp8"},
            "model.layers.0.k_proj": {"group_size": 32, "sym": True, "data_type": "MXFP4"},
        }
        base_scheme, per_layer, fp16_layers = _convert_auto_scheme_layer_config(generated)
        assert fp16_layers == []
        assert per_layer["model.layers.0.q_proj"]["bits"] == 8
        assert per_layer["model.layers.0.q_proj"]["data_type"] == "mx_fp"
        assert per_layer["model.layers.0.k_proj"]["bits"] == 4
        assert per_layer["model.layers.0.k_proj"]["data_type"] == "mx_fp"
        assert base_scheme.data_type == "mx_fp"
        assert base_scheme.bits in (4, 8)


# ===========================================================================
#  _build_mxfp_autoround_quantization_config
# ===========================================================================


class TestBuildMxfpAutoRoundConfig:
    @require_compressed_tensors
    def test_mxfp4_top_level_fields(self):
        from dataclasses import asdict

        from auto_round.schemes import PRESET_SCHEMES

        default = {k: v for k, v in asdict(PRESET_SCHEMES["MXFP4"]).items() if v is not None}
        cfg = _build_mxfp_autoround_quantization_config(default, quantized_layers=["layer.fc1"], ignored_layers=[])
        assert cfg["quant_method"] == "auto-round"
        assert cfg["packing_format"] == "auto_round:llm_compressor"
        assert cfg["bits"] == 4
        assert cfg["group_size"] == 32
        assert cfg["data_type"] == "mx_fp"
        assert cfg["sym"] is True
        assert cfg["enable_quanted_input"] is False
        assert cfg["model_free"] is True
        assert "autoround_version" in cfg

    @require_compressed_tensors
    def test_mxfp4_activation_fields(self):
        from dataclasses import asdict

        from auto_round.schemes import PRESET_SCHEMES

        default = {k: v for k, v in asdict(PRESET_SCHEMES["MXFP4"]).items() if v is not None}
        cfg = _build_mxfp_autoround_quantization_config(default, quantized_layers=["layer.fc1"], ignored_layers=[])
        assert cfg["act_bits"] == 4
        assert cfg["act_data_type"] == "mx_fp"
        assert cfg["act_dynamic"] is True
        assert cfg["act_group_size"] == 32
        assert cfg["act_sym"] is True

    @require_compressed_tensors
    def test_mxfp8_bits(self):
        from dataclasses import asdict

        from auto_round.schemes import PRESET_SCHEMES

        default = {k: v for k, v in asdict(PRESET_SCHEMES["MXFP8"]).items() if v is not None}
        cfg = _build_mxfp_autoround_quantization_config(default, quantized_layers=["layer.fc1"], ignored_layers=[])
        assert cfg["bits"] == 8
        assert cfg["act_bits"] == 8
        assert cfg["packing_format"] == "auto_round:llm_compressor"

    @require_compressed_tensors
    def test_ignored_layers_in_extra_config(self):
        default = {
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "data_type": "mx_fp",
            "act_bits": 4,
            "act_data_type": "mx_fp",
            "act_dynamic": True,
            "act_group_size": 32,
            "act_sym": True,
        }
        ignored = ["lm_head", "model.embed_tokens", "model.conv1", "model.layers.0.mlp.gate"]
        cfg = _build_mxfp_autoround_quantization_config(
            default,
            quantized_layers=["model.layers.0.fc1"],
            ignored_layers=ignored,
        )
        extra = cfg.get("extra_config", {})
        full_precision = {"bits": 16, "data_type": "float", "act_bits": 16, "act_data_type": "float"}
        assert extra.get("lm_head") == full_precision
        assert extra.get("model.layers.0.mlp.gate") == full_precision
        assert "model.embed_tokens" not in extra
        assert "model.conv1" not in extra

    @require_compressed_tensors
    def test_quantized_lm_head_in_extra_config(self):
        default = {
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "data_type": "mx_fp",
            "act_bits": 4,
            "act_data_type": "mx_fp",
            "act_dynamic": True,
            "act_group_size": 32,
            "act_sym": True,
        }
        cfg = _build_mxfp_autoround_quantization_config(
            default,
            quantized_layers=["model.layers.0.fc1", "lm_head"],
            ignored_layers=[],
        )
        extra = cfg.get("extra_config", {})
        assert "lm_head" in extra
        assert extra["lm_head"]["bits"] == 4
        assert extra["lm_head"]["data_type"] == "mx_fp"

    @require_compressed_tensors
    def test_build_quantization_config_routes_mxfp_to_autoround(self):
        default = {
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "data_type": "mx_fp",
            "act_bits": 4,
            "act_data_type": "mx_fp",
            "act_dynamic": True,
            "act_group_size": 32,
            "act_sym": True,
        }
        cfg = _build_quantization_config(
            default_scheme=default,
            layer_config={},
            ignore_patterns=[],
            quantized_layers=["layer.fc1"],
            ignored_layers=[],
            format="auto_round",
        )
        assert cfg["quant_method"] == "auto-round"
        assert cfg["packing_format"] == "auto_round:llm_compressor"

    @require_compressed_tensors
    def test_build_quantization_config_mxfp_llmcompressor_path_unchanged(self):
        default = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        cfg = _build_quantization_config(
            default_scheme=default,
            layer_config={},
            ignore_patterns=[],
            quantized_layers=["layer.fc1"],
            ignored_layers=["lm_head"],
            format="llm_compressor",
        )
        assert cfg["quant_method"] == "compressed-tensors"
        assert cfg["format"] == "mxfp4-pack-quantized"
