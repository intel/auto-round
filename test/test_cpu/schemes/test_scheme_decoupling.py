# Copyright (c) 2025 Intel Corporation
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
"""Decoupling regression for the `quantization-scheme` capability.

Covers the single-authority classification predicates on ``QuantizationScheme``
(D1), the two user-confirmed drift-point semantics (D2, Option A), and the GGUF
scheme-facts source that was pushed down into ``schemes.py`` (D4).
"""

import copy

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.schemes import GGUF_PRESET_ALIASES, GGUF_SCHEME_FACTS, BackendDataType, QuantizationScheme


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
    )
    base.update(overrides)
    return QuantizationScheme(**base)


# --- D1: classification predicates are single-authority on QuantizationScheme ---


def test_backend_data_type_members():
    assert BackendDataType.STANDARD_FP == "fp"
    assert BackendDataType.MX_FP == "mx_fp"
    assert BackendDataType.NV_FP == "nv_fp"
    assert BackendDataType.MX_INT == "mx_int"
    assert BackendDataType.FP8_STATIC == "fp8_static"
    assert BackendDataType.FP8 == "fp8"


def test_standard_and_mx_and_nv_predicates():
    assert _scheme(data_type="fp").is_standard_fp() is True
    assert _scheme(data_type="mx_fp").is_standard_fp() is False
    assert _scheme(data_type="nv_fp").is_standard_fp() is False
    assert _scheme(data_type="mx_fp").is_mx_fp() is True
    assert _scheme(data_type="nv_fp").is_nv_fp() is True
    assert _scheme(data_type="mx_int").is_mx_int() is True
    assert _scheme(act_data_type="fp").is_act_standard_fp() is True
    assert _scheme(act_data_type="mx_fp").is_act_mx_fp() is True
    assert _scheme(act_data_type="nv_fp").is_act_nv_fp() is True


def test_int_and_fp8_shape_predicates():
    assert _scheme(data_type="int", act_bits=16, super_group_size=None).is_wint_woq() is True
    assert _scheme(data_type="int", act_bits=8).is_wint_woq() is False
    assert _scheme(data_type="int", act_bits=16, super_group_size=8).is_wint_woq() is False
    assert _scheme(data_type="fp", bits=8, act_data_type="fp", act_bits=8).is_wfp8afp8() is True
    assert _scheme(data_type="int8", act_data_type="int8").is_wint8aint8() is True
    assert _scheme(data_type="int4", bits=4, act_data_type="int4", act_bits=4).is_wint4aint4() is True
    assert _scheme(act_dynamic=True, act_data_type="fp", act_bits=8).is_dynamic_afp8() is True
    assert _scheme(group_size=(128, 128), data_type="fp", bits=8).is_block_wfp8() is True
    assert _scheme(act_data_type="fp8_static").is_static_afp8() is True
    assert _scheme(act_bits=8).is_act_quantize() is True
    assert _scheme(act_bits=16).is_act_quantize() is False


# --- D2: the two confirmed drift points use Option A (utils.py) semantics ---


def test_is_static_wfp8afp8_matches_fp8_static_preset_option_a():
    # FP8_STATIC preset itself has data_type="fp"/act_data_type="fp" (no literal
    # "fp8_static"); Option A must still classify it as static wfp8afp8.
    assert _scheme(data_type="fp", bits=8, act_data_type="fp", act_bits=8, act_dynamic=False).is_static_wfp8afp8()
    assert not _scheme(data_type="fp", bits=8, act_data_type="fp", act_bits=8, act_dynamic=True).is_static_wfp8afp8()


def test_is_dynamic_wint8aint8_requires_int8_shape_option_a():
    # Option A: act_dynamic alone is not sufficient (AND, not OR).
    assert _scheme(data_type="int8", act_data_type="int8", act_dynamic=True).is_dynamic_wint8aint8()
    assert not _scheme(data_type="fp", act_data_type="fp", act_dynamic=True, bits=8, act_bits=8).is_dynamic_wint8aint8()


def test_quantization_config_predicates_delegate_to_scheme_option_a():
    # config.py properties must delegate to the scheme's single-authority predicates.
    assert (
        QuantizationConfig(
            bits=8, data_type="fp", act_bits=8, act_data_type="fp", act_dynamic=True
        ).is_dynamic_wint8aint8
        is False
    )
    assert (
        QuantizationConfig(bits=8, data_type="fp", act_bits=8, act_data_type="fp", act_dynamic=False).is_static_wfp8afp8
        is True
    )


# --- D4: GGUF scheme facts pushed down into schemes.py ---


def test_gguf_scheme_facts_only_hold_scheme_fields():
    scheme_field_names = {"bits", "act_bits", "group_size", "sym", "data_type", "super_bits", "super_group_size"}
    for key, facts in GGUF_SCHEME_FACTS.items():
        assert set(facts.keys()) <= scheme_field_names, f"{key} leaks format fields: {facts.keys()}"
        assert not ({"embedding", "lm_head", "mostly"} & set(facts.keys()))


def test_gguf_preset_aliases_and_facts_are_independent_objects():
    # Aliases resolve to a base facts key...
    assert GGUF_PRESET_ALIASES["gguf:q4_k_m"] == "gguf:q4_k"
    assert GGUF_PRESET_ALIASES["gguf:q2_k_mixed"] == "gguf:q2_k"
    # ...and mutating one facts entry must not bleed into a sibling (no shared refs).
    facts_copy = copy.deepcopy(GGUF_SCHEME_FACTS)
    GGUF_SCHEME_FACTS["gguf:q4_k"]["bits"] = 999
    try:
        assert GGUF_SCHEME_FACTS["gguf:q3_k"]["bits"] == facts_copy["gguf:q3_k"]["bits"]
    finally:
        GGUF_SCHEME_FACTS["gguf:q4_k"]["bits"] = facts_copy["gguf:q4_k"]["bits"]
