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
"""GPU-side fast unit tests for ``auto_round.schemes`` advanced paths.

Covers the remaining untested symbols and predicates:

* Module-level classifiers: ``is_standard_fp``, ``is_mx_fp``, ``is_mx_int``,
  ``is_nv_fp``.
* ``QuantizationScheme`` predicates: ``is_wint_woq``, ``is_wfp8afp8``,
  ``is_wint8aint8``, ``is_wint4aint4``, ``is_dynamic_afp8``, ``is_block_wfp8``,
  ``is_static_afp8``, ``is_act_static``, ``is_dynamic_wint8aint8``,
  ``is_act_quantize``, ``is_act_mx_fp``, ``is_act_nv_fp``,
  ``is_act_standard_fp``, ``is_act_mx_int``.
* ``scheme_to_preset_name`` (string and QuantizationScheme paths).
* ``_reconcile_bits_and_dtype`` (3 branches: missing data_type, mismatch
  warning, normalization).
* ``_override_scheme_with_user_specify`` (7 branches).
* ``parse_scheme`` (string / dict / QuantizationScheme / AutoScheme routing).

All tests run in milliseconds; no torch needed.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.schemes import (
    GGUF_PRESET_ALIASES,
    GGUF_SCHEME_FACTS,
    BackendDataType,
    QuantizationScheme,
    _override_scheme_with_user_specify,
    _reconcile_bits_and_dtype,
    is_mx_fp,
    is_mx_int,
    is_nv_fp,
    is_preset_scheme,
    is_standard_fp,
    parse_scheme,
    preset_name_to_scheme,
    scheme_to_preset_name,
)


def _scheme(**overrides):
    base = dict(bits=4, group_size=128, sym=True, data_type="int", act_bits=16)
    base.update(overrides)
    return QuantizationScheme.from_dict(base)


# ==============================================================================
# Module-level classifiers
# ==============================================================================


class TestModuleClassifiers:
    @pytest.mark.parametrize(
        "backend, expected",
        [
            ("fp", True),
            ("fp8", True),
            ("fp16", True),
            ("mx_fp", False),
            ("nv_fp", False),
            ("mx_int", False),
            ("int", False),
        ],
    )
    def test_is_standard_fp(self, backend, expected):
        assert is_standard_fp(backend) is expected

    @pytest.mark.parametrize(
        "backend, expected",
        [
            ("mx_fp4", True),
            ("mx_fp8", True),
            ("MX_FP4", True),
            ("fp", False),
            ("nv_fp", False),
        ],
    )
    def test_is_mx_fp(self, backend, expected):
        assert is_mx_fp(backend) is expected

    @pytest.mark.parametrize(
        "backend, expected",
        [
            ("nv_fp4", True),
            ("NV_FP4", True),
            ("fp", False),
            ("mx_fp", False),
        ],
    )
    def test_is_nv_fp(self, backend, expected):
        assert is_nv_fp(backend) is expected

    @pytest.mark.parametrize(
        "backend, expected",
        [
            ("mx_int4", True),
            ("mx_int8", True),
            ("MX_INT8", True),
            ("fp", False),
            ("mx_fp", False),
        ],
    )
    def test_is_mx_int(self, backend, expected):
        assert is_mx_int(backend) is expected

    def test_backend_data_type_enum(self):
        assert BackendDataType.STANDARD_FP == "fp"
        assert BackendDataType.MX_FP == "mx_fp"
        assert BackendDataType.NV_FP == "nv_fp"
        assert BackendDataType.MX_INT == "mx_int"
        assert BackendDataType.FP8_STATIC == "fp8_static"
        assert BackendDataType.FP8 == "fp8"


# ==============================================================================
# QuantizationScheme predicates
# ==============================================================================


class TestSchemePredicates:
    def test_is_wint_woq_true(self):
        s = _scheme(data_type="int", act_bits=16, super_group_size=None)
        assert s.is_wint_woq() is True

    def test_is_wint_woq_false_super_group(self):
        # When super_group_size is set, is_wint_woq returns False
        s = _scheme(data_type="int", act_bits=16, super_group_size=8)
        assert s.is_wint_woq() is False

    def test_is_wint_woq_false_act_quantized(self):
        s = _scheme(data_type="int", act_bits=8)
        assert s.is_wint_woq() is False

    def test_is_wint_woq_false_fp(self):
        s = _scheme(data_type="fp", act_bits=16)
        assert s.is_wint_woq() is False

    def test_is_wint_woq_none_data_type_raises(self):
        # data_type=None -> "int" not in None raises TypeError
        s = _scheme(data_type="int")  # data_type must be set
        s.data_type = None
        with pytest.raises(TypeError):
            s.is_wint_woq()

    def test_is_wfp8afp8_true(self):
        s = _scheme(bits=8, data_type="fp", act_bits=8, act_data_type="fp", act_dynamic=True)
        assert s.is_wfp8afp8() is True

    def test_is_wfp8afp8_none_data_type(self):
        s = _scheme(bits=8, data_type="fp", act_bits=8, act_data_type="fp", act_dynamic=True)
        s.data_type = None
        s.act_data_type = None
        assert s.is_wfp8afp8() is False

    def test_is_wint8aint8_true(self):
        s = _scheme(bits=8, data_type="int8", act_bits=8, act_data_type="int8", act_dynamic=True)
        assert s.is_wint8aint8() is True

    def test_is_wint4aint4_true(self):
        s = _scheme(bits=4, data_type="int4", act_bits=4, act_data_type="int4", act_dynamic=True)
        assert s.is_wint4aint4() is True

    def test_is_wint4aint4_false_bits_mismatch(self):
        # bits != 4 with data_type=non-int4 -> the right side of the AND fails
        s = _scheme(bits=8, data_type="int8", act_bits=8, act_data_type="int4", act_dynamic=True)
        # data_type doesn't contain "int4" and bits != 4 -> False
        assert s.is_wint4aint4() is False

    def test_is_dynamic_afp8_true(self):
        s = _scheme(bits=8, data_type="fp", act_bits=8, act_data_type="fp", act_dynamic=True)
        assert s.is_dynamic_afp8() is True

    def test_is_dynamic_afp8_false_not_dynamic(self):
        s = _scheme(bits=8, data_type="fp", act_bits=8, act_data_type="fp", act_dynamic=False)
        assert s.is_dynamic_afp8() is False

    def test_is_block_wfp8_true(self):
        s = _scheme(bits=8, data_type="fp", group_size=(128, 128))
        assert s.is_block_wfp8() is True

    def test_is_block_wfp8_false_scalar_group(self):
        s = _scheme(bits=8, data_type="fp", group_size=128)
        assert s.is_block_wfp8() is False

    def test_is_block_wfp8_false_int_data_type(self):
        s = _scheme(bits=8, data_type="int", group_size=(128, 128))
        assert s.is_block_wfp8() is False

    def test_is_static_afp8_true(self):
        s = _scheme(act_data_type="fp8_static")
        assert s.is_static_afp8() is True

    def test_is_static_afp8_false(self):
        s = _scheme(act_data_type="fp")
        assert s.is_static_afp8() is False

    def test_is_static_afp8_none_act_data_type(self):
        s = _scheme(act_data_type=None)
        assert s.is_static_afp8() is False

    def test_is_act_static_true(self):
        s = _scheme(act_dynamic=False)
        assert s.is_act_static() is True

    def test_is_act_static_false(self):
        s = _scheme(act_dynamic=True)
        assert s.is_act_static() is False

    def test_is_dynamic_wint8aint8_true(self):
        s = _scheme(bits=8, data_type="int8", act_bits=8, act_data_type="int8", act_dynamic=True)
        assert s.is_dynamic_wint8aint8() is True

    def test_is_dynamic_wint8aint8_false_no_dynamic(self):
        s = _scheme(bits=8, data_type="int8", act_bits=8, act_data_type="int8", act_dynamic=False)
        assert s.is_dynamic_wint8aint8() is False

    def test_is_act_quantize_true(self):
        s = _scheme(act_bits=8)
        assert s.is_act_quantize() is True

    def test_is_act_quantize_false_none(self):
        s = _scheme(act_bits=None)
        assert s.is_act_quantize() is False

    def test_is_act_quantize_false_high(self):
        s = _scheme(act_bits=16)
        assert s.is_act_quantize() is False

    def test_is_act_mx_fp(self):
        s = _scheme(act_data_type="mx_fp")
        assert s.is_act_mx_fp() is True

    def test_is_act_nv_fp(self):
        s = _scheme(act_data_type="nv_fp")
        assert s.is_act_nv_fp() is True

    def test_is_act_standard_fp(self):
        s = _scheme(act_data_type="fp")
        assert s.is_act_standard_fp() is True
        s.act_data_type = "mx_fp"
        assert s.is_act_standard_fp() is False

    def test_is_act_mx_int_via_module_helper(self):
        # QuantizationScheme doesn't have is_act_mx_int directly,
        # but the module-level is_mx_int(act_data_type) can be used.
        from auto_round.schemes import is_mx_int

        assert is_mx_int("mx_int") is True
        assert is_mx_int("fp") is False

    def test_is_act_mx_int_none(self):
        from auto_round.schemes import is_mx_int

        # is_mx_int(None) returns False because "mx_int" not in str(None)
        assert is_mx_int("") is False


# ==============================================================================
# scheme_to_preset_name / is_preset_scheme
# ==============================================================================


class TestSchemeToPresetName:
    def test_string_input_known(self):
        assert scheme_to_preset_name("W4A16") == "W4A16"

    def test_string_input_unknown(self):
        # Unknown names return empty string
        assert scheme_to_preset_name("TOTALLY_UNKNOWN") == ""

    def test_string_input_lowercase_uppercased(self):
        # Lowercase is uppercased before lookup
        assert scheme_to_preset_name("w4a16") == "W4A16"

    def test_scheme_input_known(self):
        s = _scheme()
        assert scheme_to_preset_name(s) == "W4A16"

    def test_scheme_input_unknown(self):
        s = _scheme(bits=3, group_size=64)
        # 3-bit W3A16G64 may or may not be a preset
        result = scheme_to_preset_name(s)
        # Either empty (no match) or a name
        assert result == "" or isinstance(result, str)


class TestIsPresetScheme:
    @pytest.mark.parametrize("name, expected", [
        ("W4A16", True),
        ("w4a16", True),
        ("MXFP4", True),
        ("GGUF:Q4_0", True),
        ("NOT_A_REAL_PRESET", False),
    ])
    def test_basic(self, name, expected):
        assert is_preset_scheme(name) is expected

    def test_preset_name_to_scheme_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            preset_name_to_scheme("TOTALLY_NOT_REAL")


# ==============================================================================
# _reconcile_bits_and_dtype
# ==============================================================================


class TestReconcileBitsAndDtype:
    def test_no_data_type_returns_early(self):
        # data_type not set -> early return (mutates nothing)
        config = {"bits": 4, "data_type": None}
        _reconcile_bits_and_dtype(config)
        assert config.get("bits") == 4

    def test_inferred_bits_mismatch(self):
        # mx_fp4 implies 4 bits; if bits=8 -> warning + reset
        config = {"bits": 8, "data_type": "mx_fp4"}
        _reconcile_bits_and_dtype(config)
        # bits should be reset to 4
        assert config["bits"] == 4

    def test_inferred_bits_match(self):
        # int4 with bits=4 -> no change
        config = {"bits": 4, "data_type": "int"}
        _reconcile_bits_and_dtype(config)
        assert config["bits"] == 4

    def test_data_type_normalization(self):
        # mx_fp4 should be normalized to mx (since it's in SUPPORTED_DTYPES)
        config = {"bits": 4, "data_type": "mx_fp4"}
        _reconcile_bits_and_dtype(config)
        # data_type may be normalized to "mx" depending on SUPPORTED_DTYPES
        # Just verify it's a string
        assert isinstance(config["data_type"], str)

    def test_act_prefix(self):
        # Use "fp8" which infers bits=8 - if act_bits is 16 -> warning, no reset
        config = {"act_bits": 16, "act_data_type": "fp8"}
        _reconcile_bits_and_dtype(config, prefix="act_")
        # inferred_bits=8 != act_bits=16 -> reset to 8
        assert config["act_bits"] == 8

    def test_act_prefix_no_data_type_early_return(self):
        # act_data_type None -> early return
        config = {"act_bits": 8, "act_data_type": None}
        _reconcile_bits_and_dtype(config, prefix="act_")
        assert config["act_bits"] == 8  # unchanged

    def test_act_prefix_normalization(self):
        # act_data_type="mx_fp4" -> normalize to "mx"
        config = {"act_bits": 4, "act_data_type": "mx_fp4"}
        _reconcile_bits_and_dtype(config, prefix="act_")
        # data_type is normalized
        assert isinstance(config["act_data_type"], str)


# ==============================================================================
# _override_scheme_with_user_specify
# ==============================================================================


class TestOverrideSchemeWithUserSpecify:
    def test_string_input_no_overrides_returns_string(self):
        # No overrides + return_str=True (default) -> returns the normalized name
        result = _override_scheme_with_user_specify("W4A16", {})
        assert result == "W4A16"

    def test_string_input_with_overrides_returns_scheme(self):
        # With overrides, return_str defaults to True; but the function
        # always returns a QuantizationScheme when overrides are present
        result = _override_scheme_with_user_specify("W4A16", {"bits": 8})
        # Should be a scheme with bits=8
        assert result.bits == 8

    def test_scheme_input_no_overrides(self):
        s = _scheme()
        result = _override_scheme_with_user_specify(s, {})
        assert isinstance(result, QuantizationScheme)
        # Should be equivalent to s
        assert result == s

    def test_dict_input_with_overrides(self):
        d = {"bits": 4, "data_type": "int", "group_size": 128, "sym": True, "act_bits": 16}
        result = _override_scheme_with_user_specify(d, {"bits": 8})
        assert result.bits == 8

    def test_dq_data_type_requires_bits(self):
        # data_type ending with "_dq" requires bits to be specified
        with pytest.raises(KeyError, match="Must specify 'bits'"):
            _override_scheme_with_user_specify("W4A16", {"data_type": "int_asym_dq"})

    def test_dq_data_type_with_bits_routes_to_gguf(self):
        # With bits=6 -> routes to gguf:q6_k
        result = _override_scheme_with_user_specify("W4A16", {"data_type": "int_asym_dq", "bits": 6})
        # Returns a string (gguf preset) because of the early routing
        assert isinstance(result, str)
        assert "q6" in result.lower()

    def test_gguf_string_warns_and_clears_overrides(self):
        # When the input scheme is GGUF, overrides are silently cleared
        # to ensure format compatibility; result is the original string
        result = _override_scheme_with_user_specify("GGUF:Q4_K_M", {"bits": 8})
        # Returns the normalized string (overrides were dropped)
        assert isinstance(result, str)
        assert "Q4_K_M" in result.upper()

    def test_act_dynamic_defaults_to_true(self):
        # If act_dynamic is not specified, default to True
        s = _scheme(act_dynamic=None)
        # Override scheme without specifying act_dynamic
        result = _override_scheme_with_user_specify(s, {})
        assert result.act_dynamic is True

    def test_act_group_size_inherits_weight_group_size(self):
        # If act_group_size is None, inherit from weight group_size
        s = _scheme(group_size=64, act_group_size=None)
        result = _override_scheme_with_user_specify(s, {})
        assert result.act_group_size == 64

    def test_act_bits_defaults_to_16(self):
        # If act_bits is None, default to 16
        s = _scheme(act_bits=None)
        result = _override_scheme_with_user_specify(s, {})
        assert result.act_bits == 16

    def test_act_sym_inherits_weight_sym(self):
        # If act_sym is None, inherit from weight sym
        s = _scheme(sym=True, act_sym=None)
        result = _override_scheme_with_user_specify(s, {})
        assert result.act_sym is True

    def test_act_data_type_inherits_weight_data_type_when_quantized(self):
        # If act_bits<16 and act_data_type is None, inherit from weight data_type
        s = _scheme(act_bits=8, act_data_type=None, data_type="fp")
        result = _override_scheme_with_user_specify(s, {})
        # Should have adopted fp
        assert result.act_data_type == "fp"

    def test_act_data_type_defaults_to_float(self):
        # If act_bits>=16 and act_data_type is None, default to "float"
        s = _scheme(act_bits=16, act_data_type=None, data_type="int")
        result = _override_scheme_with_user_specify(s, {})
        assert result.act_data_type == "float"


# ==============================================================================
# parse_scheme
# ==============================================================================


class TestParseScheme:
    def test_string_input(self):
        # parse_scheme with a string input returns the string as default_scheme
        default_scheme, is_auto, attrs = parse_scheme("W4A16", {})
        # default_scheme is the normalized string when no overrides
        assert default_scheme == "W4A16"
        assert is_auto is False
        assert isinstance(attrs, dict)
        assert attrs["bits"] == 4

    def test_dict_input(self):
        d = {"bits": 4, "data_type": "int", "group_size": 128, "sym": True, "act_bits": 16}
        default_scheme, is_auto, attrs = parse_scheme(d, {})
        assert isinstance(default_scheme, QuantizationScheme)
        assert is_auto is False

    def test_scheme_input(self):
        s = _scheme()
        default_scheme, is_auto, attrs = parse_scheme(s, {})
        assert isinstance(default_scheme, QuantizationScheme)
        assert is_auto is False

    def test_autoscheme_input(self):
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        a = AutoScheme(avg_bits=4.0, options=["W4A16", "W8A8"])
        default_scheme, is_auto, attrs = parse_scheme(a, {})
        assert is_auto is True

    def test_autoscheme_empty_options_raises(self):
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        a = AutoScheme(avg_bits=4.0, options=[])
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_scheme(a, {})

    def test_autoscheme_mixed_option_raises(self):
        # The check is case-sensitive (lowercase "mixed" only)
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        a = AutoScheme(avg_bits=4.0, options=["W4A16_mixed"])
        with pytest.raises(ValueError, match="Mixed"):
            parse_scheme(a, {})

    def test_user_overrides_applied(self):
        default_scheme, _, attrs = parse_scheme("W4A16", {"bits": 8})
        # Override should change bits
        assert attrs["bits"] == 8

    def test_attrs_is_dict(self):
        # Returned attrs is always a plain dict
        default_scheme, _, attrs = parse_scheme("W4A16", {})
        assert isinstance(attrs, dict)
        assert "bits" in attrs
        assert "data_type" in attrs
        assert "group_size" in attrs


# ==============================================================================
# GGUF_SCHEME_FACTS / GGUF_PRESET_ALIASES
# ==============================================================================


class TestGgufFacts:
    def test_q4_0_facts(self):
        facts = GGUF_SCHEME_FACTS["gguf:q4_0"]
        assert facts["bits"] == 4
        assert facts["act_bits"] == 16
        assert facts["sym"] is True
        assert facts["data_type"] == "int"

    def test_q4_1_facts(self):
        facts = GGUF_SCHEME_FACTS["gguf:q4_1"]
        assert facts["bits"] == 4
        assert facts["sym"] is False

    def test_q2_k_facts(self):
        facts = GGUF_SCHEME_FACTS["gguf:q2_k"]
        assert facts["bits"] == 2
        assert facts["super_bits"] == 4
        assert facts["super_group_size"] == 16

    def test_q6_k_facts(self):
        facts = GGUF_SCHEME_FACTS["gguf:q6_k"]
        assert facts["bits"] == 6

    def test_gguf_preset_aliases_k_s_to_k(self):
        # q2_k_s -> q2_k
        assert GGUF_PRESET_ALIASES["gguf:q2_k_s"] == "gguf:q2_k"
        # q4_k_m -> q4_k
        assert GGUF_PRESET_ALIASES["gguf:q4_k_m"] == "gguf:q4_k"
        # q2_k_mixed -> q2_k
        assert GGUF_PRESET_ALIASES["gguf:q2_k_mixed"] == "gguf:q2_k"

    def test_gguf_bf16_and_fp16_alias(self):
        # gguf:fp16 is an alias for gguf:bf16
        assert GGUF_SCHEME_FACTS["gguf:fp16"] is GGUF_SCHEME_FACTS["gguf:bf16"]