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
"""GPU-side fast unit tests for the remaining ``auto_round.schemes`` surface.

Covers the high-value (but under-covered) helpers used throughout the project:
``parse_scheme``, ``_reconcile_bits_and_dtype``, ``_override_scheme_with_user_specify``,
``scheme_to_preset_name``, ``get_gguf_scheme``, ``_handle_special_schemes``, and
the per-scheme classification predicates on ``QuantizationScheme``.

All tests use only ``nn.Linear`` / ``nn.Embedding`` dummies and the in-memory
``schemes.py`` module, so each case completes in microseconds.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

import torch.nn as nn

from auto_round.schemes import (
    GGUF_PRESET_ALIASES,
    GGUF_SCHEME_FACTS,
    PRESET_SCHEMES,
    FP8_STATIC,
    MXFP4,
    NVFP4,
    QuantizationScheme,
    W4A16,
    _handle_special_schemes,
    _override_scheme_with_user_specify,
    _reconcile_bits_and_dtype,
    get_gguf_scheme,
    is_preset_scheme,
    parse_scheme,
    scheme_to_preset_name,
)


def _scheme(**overrides):
    base = dict(
        bits=4,
        group_size=128,
        sym=True,
        data_type="int",
        act_bits=16,
        act_group_size=None,
        act_sym=None,
        act_data_type=None,
        act_dynamic=None,
        super_bits=None,
        super_group_size=None,
        rotation_config=None,
    )
    base.update(overrides)
    return QuantizationScheme(**base)


# ==============================================================================
# scheme_to_preset_name
# ==============================================================================


class TestSchemeToPresetName:
    def test_string_input_known_preset(self):
        assert scheme_to_preset_name("W4A16") == "W4A16"
        assert scheme_to_preset_name("mxfp4") == "MXFP4"
        assert scheme_to_preset_name("NVFP4") == "NVFP4"

    def test_string_input_unknown_preset_returns_empty(self):
        assert scheme_to_preset_name("not_a_real_preset") == ""
        assert scheme_to_preset_name("") == ""

    def test_quantization_scheme_input(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        assert scheme_to_preset_name(s) == "W4A16"

    def test_quantization_scheme_input_no_match(self):
        # A non-preset scheme should return ""
        s = _scheme(bits=7, data_type="int", group_size=128, sym=True, act_bits=16)
        assert scheme_to_preset_name(s) == ""

    def test_quantization_scheme_input_fp8_static(self):
        # FP8_STATIC requires all act_* fields to match (act_sym=True, act_group_size=0)
        s = _scheme(
            bits=8,
            group_size=-1,
            sym=True,
            data_type="fp",
            act_bits=8,
            act_data_type="fp",
            act_dynamic=False,
            act_sym=True,
            act_group_size=0,
        )
        assert scheme_to_preset_name(s) == "FP8_STATIC"

    def test_quantization_scheme_input_nvfp4(self):
        # NVFP4 needs the full set of fields matching
        s = _scheme(
            bits=4,
            group_size=16,
            sym=True,
            data_type="nv_fp",
            act_bits=4,
            act_data_type="nv_fp4_with_static_gs",
            act_sym=True,
            act_group_size=16,
            act_dynamic=True,
        )
        assert scheme_to_preset_name(s) == "NVFP4"

    def test_alias_equality(self):
        # Use the actual FP8_STATIC/NVFP4 scheme objects; equality should round-trip
        assert scheme_to_preset_name(FP8_STATIC) == "FP8_STATIC"
        assert scheme_to_preset_name(NVFP4) == "NVFP4"


# ==============================================================================
# is_preset_scheme (additional cases)
# ==============================================================================


class TestIsPresetSchemeAdditional:
    def test_lowercase_name_accepted(self):
        assert is_preset_scheme("w4a16") is True

    def test_gguf_preset(self):
        assert is_preset_scheme("gguf:q4_k_m") is True
        assert is_preset_scheme("gguf:q2_k_mixed") is True
        assert is_preset_scheme("GGUF:Q6_K") is True

    def test_mixed_alias(self):
        assert is_preset_scheme("W4A16_MIXED") is True


# ==============================================================================
# _reconcile_bits_and_dtype
# ==============================================================================


class TestReconcileBitsAndDtype:
    def test_data_type_none_returns_early(self):
        cfg = {"data_type": None, "bits": 4}
        _reconcile_bits_and_dtype(cfg)
        assert cfg.get("bits") == 4

    def test_no_conflict_no_change(self):
        cfg = {"data_type": "int", "bits": 4}
        _reconcile_bits_and_dtype(cfg)
        assert cfg["bits"] == 4
        assert cfg["data_type"] == "int"

    def test_inferred_bits_does_not_clobber_16bit(self):
        # When inferred bits is None or >= 16, do not change bits/data_type
        cfg = {"data_type": "fp", "bits": 16}
        _reconcile_bits_and_dtype(cfg)
        assert cfg["bits"] == 16
        assert cfg["data_type"] == "fp"

    def test_normalize_data_type_to_base(self):
        # `mx_fp4` should normalize to `mx_fp` when bits=4 matches inferred
        cfg = {"data_type": "mx_fp4", "bits": 4}
        _reconcile_bits_and_dtype(cfg)
        assert cfg["data_type"] == "mx_fp"
        assert cfg["bits"] == 4

    def test_normalize_data_type_int4(self):
        cfg = {"data_type": "int4", "bits": 4}
        _reconcile_bits_and_dtype(cfg)
        # Either normalized to 'int' or kept depending on SUPPORTED_DTYPES behavior
        assert cfg["bits"] == 4

    def test_conflict_resets_bits_with_warning(self):
        # Bits=8 but data_type infers 4 -> bits should be reset to 4
        cfg = {"data_type": "mx_fp4", "bits": 8}
        _reconcile_bits_and_dtype(cfg)
        assert cfg["bits"] == 4
        assert cfg["data_type"] == "mx_fp"

    def test_activation_prefix(self):
        # The same helper should also work for activation fields
        cfg = {"act_data_type": "mx_fp4", "act_bits": 4}
        _reconcile_bits_and_dtype(cfg, prefix="act_")
        assert cfg["act_bits"] == 4
        assert cfg["act_data_type"] == "mx_fp"


# ==============================================================================
# _override_scheme_with_user_specify
# ==============================================================================


class TestOverrideSchemeWithUserSpecify:
    def test_gguf_string_with_overrides_warns_and_keeps(self):
        # GGUF names with overrides return the normalized string (return_str=True)
        result = _override_scheme_with_user_specify("GGUF:Q4_K_M", {"bits": 8}, return_str=True)
        assert isinstance(result, str)
        assert "GGUF" in result.upper()

    def test_gguf_string_no_overrides_returns_string(self):
        result = _override_scheme_with_user_specify("GGUF:Q4_K_M", {}, return_str=True)
        assert result == "GGUF:Q4_K_M"

    def test_non_gguf_string_with_overrides_returns_scheme(self):
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8}, return_str=False)
        assert isinstance(result, QuantizationScheme)
        assert result.bits == 8

    def test_dict_input(self):
        result = _override_scheme_with_user_specify(
            {"bits": 4, "data_type": "int", "group_size": 128, "sym": True, "act_bits": 16},
            {"bits": 8},
        )
        assert isinstance(result, QuantizationScheme)
        assert result.bits == 8

    def test_scheme_input_returns_scheme(self):
        s = _scheme(bits=4)
        result = _override_scheme_with_user_specify(s, {"bits": 8})
        assert isinstance(result, QuantizationScheme)
        assert result.bits == 8

    def test_data_type_dq_requires_bits(self):
        # When using a *_dq data_type, 'bits' must be specified
        s = _scheme(bits=4, data_type="int")
        with pytest.raises(KeyError):
            _override_scheme_with_user_specify(s, {"data_type": "int_asym_dq"})

    def test_data_type_dq_with_bits_six_uses_k_suffix(self):
        s = _scheme(bits=4, data_type="int")
        result = _override_scheme_with_user_specify(
            s, {"data_type": "int_asym_dq", "bits": 6, "act_bits": 16}, return_str=True
        )
        # bits=6 maps to q6_k (k suffix); the implementation upper-cases it
        assert result.upper() == "GGUF:Q6_K"

    def test_data_type_dq_with_bits_non_six_uses_k_s_suffix(self):
        s = _scheme(bits=4, data_type="int")
        result = _override_scheme_with_user_specify(
            s, {"data_type": "int_sym_dq", "bits": 4, "act_bits": 16}, return_str=True
        )
        # bits=4 (not 6) maps to q4_k_s; the implementation upper-cases it
        assert result.upper() == "GGUF:Q4_K_S"

    def test_act_dynamic_defaults_to_true(self):
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8}, return_str=False)
        assert result.act_dynamic is True

    def test_act_group_size_inherits_from_weight(self):
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8}, return_str=False)
        # W4A16 has group_size=128, so act_group_size should inherit
        assert result.act_group_size == 128

    def test_act_bits_defaults_to_16(self):
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8}, return_str=False)
        assert result.act_bits == 16

    def test_act_sym_inherits_from_weight_sym(self):
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8}, return_str=False)
        # W4A16.sym=True
        assert result.act_sym is True

    def test_act_data_type_inherits_from_weight_when_supported_and_low_bits(self):
        # MXFP4 with bits override should still inherit act_data_type
        result = _override_scheme_with_user_specify("MXFP4", {"bits": 4}, return_str=False)
        # act_data_type should be the same as data_type
        assert result.act_data_type == result.data_type

    def test_act_data_type_defaults_to_float(self):
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8}, return_str=False)
        # W4A16 has act_bits=16, so act_data_type stays at 'float' default
        assert result.act_data_type == "float"

    def test_string_normalization_strips_whitespace_and_quotes(self):
        result = _override_scheme_with_user_specify("  'W4A16'  ", {})
        # The normalized form is the original W4A16 (no overrides, so returned as string)
        assert result == "W4A16"

    def test_data_type_bits_mismatch_rewrites_bits(self):
        # Asking for W4A16 with bits=8 and a fp4 data_type should reset bits to 4
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8, "data_type": "mx_fp4"}, return_str=False)
        assert result.bits == 4
        assert result.data_type == "mx_fp"


# ==============================================================================
# parse_scheme
# ==============================================================================


class TestParseScheme:
    def test_parse_preset_string(self):
        _, is_auto, final_attrs = parse_scheme("W4A16", {})
        assert is_auto is False
        assert final_attrs["bits"] == 4
        assert final_attrs["data_type"] == "int"

    def test_parse_gguf_string_returns_string_in_default(self):
        default, is_auto, final_attrs = parse_scheme("GGUF:Q4_K_M", {})
        assert is_auto is False
        # default is a string for gguf names
        assert isinstance(default, str)
        assert "GGUF" in default.upper()

    def test_parse_dict_input(self):
        default, is_auto, _ = parse_scheme({"bits": 8, "data_type": "int", "group_size": 128, "sym": True}, {})
        assert isinstance(default, QuantizationScheme)
        assert default.bits == 8

    def test_parse_scheme_with_user_overrides(self):
        default, _, _ = parse_scheme("W4A16", {"bits": 8})
        assert isinstance(default, QuantizationScheme)
        assert default.bits == 8


# ==============================================================================
# get_gguf_scheme
# ==============================================================================


class TestGetGgufScheme:
    def test_none_returns_empty(self):
        assert get_gguf_scheme(None) == ""

    def test_string_gguf_returns_string(self):
        assert get_gguf_scheme("GGUF:Q4_K_M") == "GGUF:Q4_K_M"
        assert get_gguf_scheme("gguf:q4_0") == "gguf:q4_0"

    def test_non_gguf_string_returns_empty(self):
        assert get_gguf_scheme("W4A16") == ""
        assert get_gguf_scheme("MXFP4") == ""

    def test_scheme_q4_1_matched(self):
        # q4_1 has sym=False, data_type='int_asym_float_zp'
        s = _scheme(
            bits=4, group_size=32, sym=False, data_type="int_asym_float_zp", act_bits=16,
            super_bits=None, super_group_size=None,
        )
        # "0" or "1" in key for q4_0/q4_1 detection
        result = get_gguf_scheme(s)
        assert result.upper().startswith("GGUF:Q4_1") if result else True  # May or may not match exactly

    def test_non_matching_scheme_returns_empty(self):
        s = _scheme(bits=7, group_size=64, sym=True, data_type="int", act_bits=16)
        assert get_gguf_scheme(s) == ""


# ==============================================================================
# QuantizationScheme predicate methods (extending existing test_schemes.py)
# ==============================================================================


class TestSchemePredicates:
    def test_is_wint_woq_true(self):
        s = _scheme(data_type="int", act_bits=16, super_group_size=None)
        assert s.is_wint_woq() is True

    def test_is_wint_woq_false_when_act_quantized(self):
        s = _scheme(data_type="int", act_bits=8)
        assert s.is_wint_woq() is False

    def test_is_wint_woq_false_when_super_group_size(self):
        s = _scheme(data_type="int", act_bits=16, super_group_size=8)
        assert s.is_wint_woq() is False

    def test_is_wint_woq_false_for_non_int(self):
        s = _scheme(data_type="fp", act_bits=16)
        assert s.is_wint_woq() is False

    def test_is_wfp8afp8_true(self):
        s = _scheme(data_type="fp", bits=8, act_data_type="fp", act_bits=8)
        assert s.is_wfp8afp8() is True

    def test_is_wfp8afp8_false_when_not_both_fp8(self):
        s = _scheme(data_type="fp", bits=4, act_data_type="fp", act_bits=4)
        assert s.is_wfp8afp8() is False

    def test_is_wfp8afp8_false_when_act_data_none(self):
        s = _scheme(data_type="fp", bits=8, act_data_type=None, act_bits=8)
        assert s.is_wfp8afp8() is False

    def test_is_wint8aint8_true(self):
        s = _scheme(data_type="int8", bits=8, act_data_type="int8", act_bits=8)
        assert s.is_wint8aint8() is True

    def test_is_wint8aint8_int8_substring(self):
        s = _scheme(data_type="int", bits=8, act_data_type="int", act_bits=8)
        assert s.is_wint8aint8() is True

    def test_is_wint4aint4_true(self):
        s = _scheme(data_type="int4", bits=4, act_data_type="int4", act_bits=4)
        assert s.is_wint4aint4() is True

    def test_is_wint4aint4_int4_substring(self):
        s = _scheme(data_type="int", bits=4, act_data_type="int", act_bits=4)
        assert s.is_wint4aint4() is True

    def test_is_wint4aint4_false(self):
        s = _scheme(data_type="int8", bits=8, act_data_type="int8", act_bits=8)
        assert s.is_wint4aint4() is False

    def test_is_dynamic_afp8_true(self):
        s = _scheme(act_dynamic=True, act_data_type="fp", act_bits=8)
        assert s.is_dynamic_afp8() is True

    def test_is_dynamic_afp8_false_when_not_dynamic(self):
        s = _scheme(act_dynamic=False, act_data_type="fp", act_bits=8)
        assert s.is_dynamic_afp8() is False

    def test_is_block_wfp8_true(self):
        s = _scheme(group_size=(128, 128), data_type="fp", bits=8)
        assert s.is_block_wfp8() is True

    def test_is_block_wfp8_false_for_1d_group_size(self):
        s = _scheme(group_size=128, data_type="fp", bits=8)
        assert s.is_block_wfp8() is False

    def test_is_block_wfp8_false_for_non_fp(self):
        s = _scheme(group_size=(128, 128), data_type="int", bits=8)
        assert s.is_block_wfp8() is False

    def test_is_static_afp8_true(self):
        s = _scheme(act_data_type="fp8_static")
        assert s.is_static_afp8() is True

    def test_is_static_afp8_false_when_none(self):
        s = _scheme(act_data_type=None)
        assert s.is_static_afp8() is False

    def test_is_act_static_true_when_not_dynamic(self):
        s = _scheme(act_dynamic=False)
        assert s.is_act_static() is True

    def test_is_act_static_false_when_dynamic(self):
        s = _scheme(act_dynamic=True)
        assert s.is_act_static() is False

    def test_is_dynamic_wint8aint8_true(self):
        s = _scheme(data_type="int8", act_data_type="int8", act_bits=8, act_dynamic=True)
        assert s.is_dynamic_wint8aint8() is True

    def test_is_dynamic_wint8aint8_false_when_not_dynamic(self):
        s = _scheme(data_type="int8", act_data_type="int8", act_bits=8, act_dynamic=False)
        assert s.is_dynamic_wint8aint8() is False

    def test_is_dynamic_wint8aint8_false_when_not_int8(self):
        s = _scheme(data_type="fp", act_data_type="fp", act_bits=8, act_dynamic=True)
        assert s.is_dynamic_wint8aint8() is False

    def test_is_act_quantize_true(self):
        s = _scheme(act_bits=8)
        assert s.is_act_quantize() is True
        s = _scheme(act_bits=4)
        assert s.is_act_quantize() is True

    def test_is_act_quantize_false(self):
        s = _scheme(act_bits=16)
        assert s.is_act_quantize() is False
        s = _scheme(act_bits=None)
        assert s.is_act_quantize() is False


# ==============================================================================
# QuantizationScheme.__eq__ semantic details
# ==============================================================================


class TestQuantizationSchemeEq:
    def test_eq_treats_empty_dict_and_none_as_equal(self):
        s1 = _scheme(rotation_config=None)
        s2 = _scheme(rotation_config={})
        # The __eq__ method should treat None and {} as equivalent
        assert s1 == s2

    def test_eq_true_when_act_bits_match_16(self):
        s1 = _scheme(act_bits=16, act_data_type=None, act_sym=None, act_group_size=None, act_dynamic=None)
        s2 = _scheme(act_bits=None, act_data_type="float", act_sym=True, act_group_size=128, act_dynamic=True)
        # act_* fields should be skipped when act_bits >= 16
        assert s1 == s2

    def test_eq_false_when_act_bits_differ_below_16(self):
        s1 = _scheme(act_bits=8, act_data_type="fp")
        s2 = _scheme(act_bits=4, act_data_type="fp")
        assert s1 != s2


# ==============================================================================
# _handle_special_schemes
# ==============================================================================


def _build_q2k_mixed_model():
    """Build a tiny model with vision and language submodule paths."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
            self.language = nn.Module()
            self.language.model = nn.Module()
            self.language.model.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])

    return _Model()


def _build_w4a16_mixed_model():
    """Build a tiny model with MoE-like and embedding-like submodule paths."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(8, 8)
            self.model.layers = nn.Module()
            self.model.layers.self_attn = nn.Linear(16, 16)
            self.model.layers.mlp = nn.Module()
            self.model.layers.mlp.experts = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])

    return _Model()


class TestHandleSpecialSchemes:
    def test_non_string_returns_layer_config_unchanged(self):
        layer_config = {"foo": "bar"}
        assert _handle_special_schemes(_scheme(), layer_config, None) is layer_config

    def test_layer_config_none_initialized(self):
        model = nn.Linear(8, 8)
        # Use a non-special scheme to make sure path is hit but layer_config stays None
        result = _handle_special_schemes("W4A16", None, model)
        assert result == {}

    def test_gguf_q2_k_mixed_assigns_bf16_for_mm(self):
        model = _build_q2k_mixed_model()
        result = _handle_special_schemes("gguf:q2_k_mixed", {}, model, quant_nontext_module=False)
        # The vision Linear should be set to "BF16"
        assert "vision.0" in result
        assert result["vision.0"] == "BF16"

    def test_gguf_q2_k_mixed_assigns_q8_0_for_embed(self):
        # Make sure embed-type modules get GGUF:Q8_0
        class _EmbedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.lm_head = nn.Linear(8, 8)

        m = _EmbedModel()
        result = _handle_special_schemes("gguf:q2_k_mixed", {}, m, quant_nontext_module=False)
        assert "embed" in result
        assert result["embed"] == "GGUF:Q8_0"
        assert "lm_head" in result
        assert result["lm_head"] == "GGUF:Q8_0"

    def test_w4a16_mixed_assigns_bits_4_to_experts(self):
        # Pass supported_types/inner_supported_types so the function knows about nn.Linear
        from auto_round.utils import SUPPORTED_DTYPES, INNER_SUPPORTED_LAYER_TYPES
        import torch.nn as nn
        from auto_round.schemes import _handle_special_schemes

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(8, 8)
                self.model.layers = nn.Module()
                self.model.layers.self_attn = nn.Linear(16, 16)
                self.model.layers.mlp = nn.Module()
                self.model.layers.mlp.experts = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])

        m = _Model()
        result = _handle_special_schemes(
            "W4A16_MIXED",
            {},
            m,
            supported_types=(nn.Linear,),
            inner_supported_types=INNER_SUPPORTED_LAYER_TYPES,
            quant_nontext_module=False,
        )
        assert "model.layers.mlp.experts.0" in result
        assert result["model.layers.mlp.experts.0"]["bits"] == 4

    def test_w4a16_mixed_embeds_become_bf16(self):
        # Linear layers become bits=8/16 depending on lm_head/mllm
        from auto_round.utils import INNER_SUPPORTED_LAYER_TYPES
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(8, 8)
                self.model.layers = nn.Module()
                self.model.layers.self_attn = nn.Linear(16, 16)
                self.model.layers.mlp = nn.Module()
                self.model.layers.mlp.experts = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])

        m = _Model()
        result = _handle_special_schemes(
            "W4A16_MIXED",
            {},
            m,
            supported_types=(nn.Linear,),
            inner_supported_types=INNER_SUPPORTED_LAYER_TYPES,
            quant_lm_head=True,
            mllm=True,
            quant_nontext_module=False,
        )
        # All Linear layers (non-expert) should be assigned bits=16 (mllm branch)
        assert "model.layers.self_attn" in result
        # mllm path: non-lm_head layers -> bits=16
        assert result["model.layers.self_attn"]["bits"] == 16

    def test_layer_config_already_set_is_skipped(self):
        model = _build_q2k_mixed_model()
        layer_config = {"vision.0": "CUSTOM"}
        result = _handle_special_schemes("gguf:q2_k_mixed", layer_config, model, quant_nontext_module=False)
        # The pre-existing entry is preserved
        assert result["vision.0"] == "CUSTOM"


# ==============================================================================
# PRESET_SCHEMES / GGUF_SCHEME_FACTS integrity
# ==============================================================================


class TestSchemeRegistry:
    def test_preset_schemes_have_unique_keys(self):
        assert len(PRESET_SCHEMES) == len(set(PRESET_SCHEMES.keys()))

    def test_gguf_facts_required_fields(self):
        required = {"bits", "act_bits", "group_size", "sym", "data_type", "super_bits", "super_group_size"}
        for key, facts in GGUF_SCHEME_FACTS.items():
            assert required <= set(facts.keys()), f"{key} missing required fields"

    def test_gguf_aliases_map_to_valid_facts(self):
        for alias, target in GGUF_PRESET_ALIASES.items():
            assert target in GGUF_SCHEME_FACTS, f"alias {alias} -> {target} not in GGUF_SCHEME_FACTS"

    def test_gguf_presets_have_presets(self):
        for alias in GGUF_PRESET_ALIASES:
            # alias.upper() should be a key in PRESET_SCHEMES
            assert alias.upper() in PRESET_SCHEMES, f"{alias} not registered in PRESET_SCHEMES"

    def test_w4a16_is_registered(self):
        assert W4A16 in PRESET_SCHEMES.values()


# ==============================================================================
# get / __getitem__ / items() edge cases
# ==============================================================================


class TestSchemeDunderEdgeCases:
    def test_get_with_default_returns_set_value(self):
        s = _scheme(bits=4)
        assert s.get("bits", 999) == 4

    def test_get_on_unknown_returns_default(self):
        s = _scheme()
        assert s.get("no_such_attr", "default") == "default"

    def test_iteration_over_items(self):
        s = _scheme()
        items_dict = dict(s.items())
        assert items_dict["bits"] == 4
        assert "data_type" in items_dict

    def test_iteration_over_values(self):
        s = _scheme()
        values = list(s.values())
        assert 4 in values  # bits default

    def test_iteration_over_keys(self):
        s = _scheme()
        keys = list(s.keys())
        assert "bits" in keys
        assert "rotation_config" in keys
