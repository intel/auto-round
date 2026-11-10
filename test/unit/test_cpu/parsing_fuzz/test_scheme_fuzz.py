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
"""Fuzzing of quantization-scheme plumbing: preset lookup, ``QuantizationScheme`` dict round-trips,
its lenient ``__eq__`` and bits/data_type reconciliation."""

import dataclasses

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from auto_round.schemes import (
    PRESET_SCHEMES,
    QuantizationScheme,
    _reconcile_bits_and_dtype,
    is_preset_scheme,
    preset_name_to_scheme,
    scheme_to_preset_name,
)
from auto_round.utils.common import SUPPORTED_DTYPES, infer_bits_by_data_type

_PRESET_NAMES = sorted(PRESET_SCHEMES)


@st.composite
def preset_names_any_case(draw):
    name = draw(st.sampled_from(_PRESET_NAMES))
    flips = draw(st.lists(st.booleans(), min_size=len(name), max_size=len(name)))
    return "".join(c.lower() if flip else c for c, flip in zip(name, flips))


_data_types = st.one_of(
    st.none(),
    st.builds(
        lambda base, bits, suffix: f"{base}{bits}{suffix}",
        st.sampled_from(SUPPORTED_DTYPES),
        st.sampled_from(["", "2", "4", "8", "16", "1x"]),
        st.sampled_from(["", "_e4m3", "_e5m2", "_rceil", "_dq", "e2m1"]),
    ),
)
_bits = st.one_of(st.none(), st.integers(0, 32))
_maybe_bool = st.one_of(st.none(), st.booleans())

_FIELD_STRATEGIES = {
    "bits": _bits,
    "group_size": st.one_of(st.none(), st.sampled_from([-1, 0, 32, 64, 128])),
    "sym": _maybe_bool,
    "data_type": _data_types,
    "act_bits": st.one_of(st.none(), st.sampled_from([4, 8, 16, 32])),
    "act_group_size": st.one_of(st.none(), st.sampled_from([-1, 0, 32])),
    "act_sym": _maybe_bool,
    "act_data_type": _data_types,
    "act_dynamic": _maybe_bool,
    "super_bits": st.one_of(st.none(), st.sampled_from([4, 6, 8])),
    "super_group_size": st.one_of(st.none(), st.sampled_from([8, 16])),
    "rotation_config": st.one_of(st.none(), st.just({}), st.just({"group_size": 32})),
}
schemes = st.builds(QuantizationScheme, **_FIELD_STRATEGIES)


@given(name=preset_names_any_case())
def test_preset_lookup_is_case_insensitive_and_returns_isolated_copies(name):
    assert is_preset_scheme(name)
    assert scheme_to_preset_name(name) == name.upper()
    scheme = preset_name_to_scheme(name)
    assert isinstance(scheme, QuantizationScheme)
    assert scheme.to_dict() == PRESET_SCHEMES[name.upper()].to_dict()
    scheme.bits = -1
    scheme.data_type = "mutated"
    assert PRESET_SCHEMES[name.upper()].bits != -1
    assert PRESET_SCHEMES[name.upper()].data_type != "mutated"


@given(name=st.sampled_from(_PRESET_NAMES))
def test_scheme_to_preset_name_finds_an_equal_preset(name):
    found = scheme_to_preset_name(PRESET_SCHEMES[name])
    assert found and PRESET_SCHEMES[found] == PRESET_SCHEMES[name]


@given(text=st.text(max_size=12))
def test_unknown_names_are_rejected_consistently(text):
    known = text.upper() in PRESET_SCHEMES
    assert is_preset_scheme(text) == known
    assert scheme_to_preset_name(text) == (text.upper() if known else "")
    if not known:
        with pytest.raises(KeyError):
            preset_name_to_scheme(text)


@given(scheme=schemes, junk=st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=3))
def test_dict_round_trip_is_exact_and_ignores_unknown_keys(scheme, junk):
    as_dict = scheme.to_dict()
    assert QuantizationScheme.from_dict(as_dict).to_dict() == as_dict
    field_names = set(QuantizationScheme.get_attributes())
    extra = {k: v for k, v in junk.items() if k not in field_names}
    assert QuantizationScheme.from_dict({**as_dict, **extra}).to_dict() == as_dict
    assert dict(scheme.items()) == as_dict
    assert list(scheme.keys()) == list(as_dict)


@given(a=schemes, b=schemes, c=schemes)
def test_scheme_equality_is_an_equivalence_relation(a, b, c):
    assert a.__eq__(a) and a == a.copy() and a == QuantizationScheme.from_dict(a.to_dict())
    assert (a == b) == (b == a)
    if a == b and b == c:
        assert a == c


@given(scheme=schemes, data=st.data())
def test_scheme_equality_only_ignores_documented_differences(scheme, data):
    field = data.draw(st.sampled_from(QuantizationScheme.get_attributes()))
    if field == "act_bits":
        replacements = [None, 3, 8, 16, 32]
    else:
        replacements = [None, 3, "other", {}, {"k": 1}]
    other = scheme.copy()
    setattr(other, field, data.draw(st.sampled_from(replacements)))
    before, after = getattr(scheme, field), getattr(other, field)
    note(f"field={field} before={before!r} after={after!r}")

    def act_bits(s):
        return 16 if s.act_bits is None else s.act_bits

    act_ignored = field.startswith("act_") and act_bits(scheme) == act_bits(other) >= 16
    empty_vs_none = {type(before), type(after)} <= {dict, type(None)} and not before and not after
    assert (scheme == other) == (before == after or act_ignored or empty_vs_none)


@given(data_type=_data_types.filter(lambda d: d is not None), bits=_bits, prefix=st.sampled_from(["", "act_"]))
def test_reconcile_bits_and_dtype_is_idempotent_and_follows_inference(data_type, bits, prefix):
    config = {f"{prefix}data_type": data_type, f"{prefix}bits": bits}
    _reconcile_bits_and_dtype(config, prefix)
    once = dict(config)
    _reconcile_bits_and_dtype(config, prefix)
    assert config == once

    inferred = infer_bits_by_data_type(data_type)
    if inferred is not None and inferred < 16:
        assert once[f"{prefix}bits"] == inferred
        assert infer_bits_by_data_type(once[f"{prefix}data_type"]) in (None, inferred)
    else:
        assert once == {f"{prefix}data_type": data_type, f"{prefix}bits": bits}


def test_strategy_covers_every_scheme_field():
    assert set(_FIELD_STRATEGIES) == {f.name for f in dataclasses.fields(QuantizationScheme)}
