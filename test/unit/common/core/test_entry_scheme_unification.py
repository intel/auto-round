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
"""Entry-layer scheme unification: routing decisions in ``entry.py`` read scheme
fields from a single resolved-scheme source (``preview_resolved_attrs``) rather
than double-reading algorithm config attrs. These lock that the resolved values — and
the resulting compressor-class routing — are identical whether the user passes
``scheme=`` alone or equivalent entry-level bit/dtype overrides.
"""

import pytest

from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors.builder import _resolve_rtn_config
from auto_round.scheme_entry import (
    preview_resolved_attrs,
    resolve_entry_scheme,
)


@pytest.mark.parametrize("candidates", [("W4A16", "W8A16"), ["W4A16", "W8A16"]])
def test_scheme_candidates_and_bits_build_auto_scheme(candidates):
    """scheme candidates + bits select AutoScheme options and average target bits."""
    scheme, kwargs = resolve_entry_scheme(candidates, {"bits": 4.2})
    assert isinstance(scheme, AutoScheme)
    assert scheme.options == ["W4A16", "W8A16"]
    assert scheme.avg_bits == 4.2
    assert "bits" not in kwargs  # consumed as the AutoScheme target, not a scheme override


def test_auto_scheme_threads_aux_kwargs():
    """shared_layers/ignore_scale_zp_bits are consumed by the AutoScheme."""
    scheme, kwargs = resolve_entry_scheme(
        ("W4A16", "W8A16"), {"bits": 5.0, "shared_layers": [["a", "b"]], "ignore_scale_zp_bits": True}
    )
    assert scheme.shared_layers == [["a", "b"]]
    assert scheme.ignore_scale_zp_bits is True
    assert "shared_layers" not in kwargs and "ignore_scale_zp_bits" not in kwargs


def test_schemes_mixed_string_and_scheme_object():
    """schemes may mix preset names and QuantizationScheme instances."""
    from auto_round.schemes import QuantizationScheme

    obj = QuantizationScheme.from_dict({"bits": 4, "group_size": 64, "sym": True, "data_type": "int"})
    scheme, kwargs = resolve_entry_scheme(("W4A16", obj), {"bits": 4.5})
    assert isinstance(scheme, AutoScheme)
    assert scheme.options == ["W4A16", obj]
    assert scheme.avg_bits == 4.5
    assert "bits" not in kwargs


def test_group_size_override_applies_to_all_schemes():
    """scheme candidates + group_size=32 overrides every candidate group size."""
    from auto_round.schemes import parse_scheme

    scheme, kwargs = resolve_entry_scheme(("W4A16", "W8A16"), {"bits": 4.5, "group_size": 32})
    # group_size stays an entry override (it is not an AutoScheme field); the
    # authoritative parse then applies it across every candidate option.
    assert kwargs == {"group_size": 32}
    overrides = kwargs
    _, is_auto_scheme, _ = parse_scheme(scheme, overrides)
    assert is_auto_scheme
    for option in scheme.options:
        assert option.group_size == 32


def test_legacy_options_avg_bits_still_resolve():
    """options=/avg_bits= remain accepted without warning."""
    scheme, kwargs = resolve_entry_scheme("W4A16", {"options": ("W4A16", "W8A16"), "avg_bits": 4.0})
    assert isinstance(scheme, AutoScheme)
    assert scheme.options == ["W4A16", "W8A16"]
    assert scheme.avg_bits == 4.0
    assert "options" not in kwargs and "avg_bits" not in kwargs


def test_bits_without_schemes_coerces_integral_float():
    """bits=8.0 without schemes stays a plain integer weight bit override."""
    scheme, kwargs = resolve_entry_scheme("W4A16", {"bits": 8.0, "group_size": 128})
    assert scheme == "W4A16"
    assert kwargs == {"bits": 8, "group_size": 128}


def test_old_schemes_parameter_is_rejected():
    with pytest.raises(TypeError, match="pass candidate schemes through `scheme`"):
        resolve_entry_scheme("W4A16", {"schemes": ("W4A16", "W8A16")})


def test_fractional_bits_requires_schemes():
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_entry_scheme("W4A16", {"bits": 4.2})


def test_scheme_candidates_and_options_conflict():
    with pytest.raises(ValueError, match="cannot be used together"):
        resolve_entry_scheme(("W4A16", "W8A16"), {"options": ("W4A16", "W8A16")})


def test_auto_scheme_instance_passthrough_and_overrides():
    original = AutoScheme(options=("W4A16", "W8A16"), avg_bits=4.2, nsamples=16)
    scheme, kwargs = resolve_entry_scheme(original, {})
    assert scheme is original
    assert kwargs == {}
    scheme, kwargs = resolve_entry_scheme(original, {"bits": 5.0, "ignore_scale_zp_bits": True, "group_size": 32})
    assert scheme is not original
    assert scheme.avg_bits == 5.0
    assert scheme.ignore_scale_zp_bits is True
    assert scheme.nsamples == 16
    assert original.avg_bits == 4.2
    assert original.ignore_scale_zp_bits is False
    assert kwargs == {"group_size": 32}


@pytest.mark.parametrize("candidates", [[], ()])
def test_empty_scheme_candidates_rejected(candidates):
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_entry_scheme(candidates, {"bits": 4.2})


def test_scheme_candidates_require_average_bits():
    with pytest.raises(ValueError, match="`bits` is required"):
        resolve_entry_scheme(["W4A16", "W8A16"], {})


def test_preview_resolves_scheme_only_and_override_to_same_attrs():
    # scheme="W8A16" vs bits=8 override on top of a W-generic scheme must resolve
    # to the same bits/data_type for routing.
    from_scheme = preview_resolved_attrs({}, "W8A16")
    from_override = preview_resolved_attrs({"bits": 8}, "W8A16")
    assert from_scheme.get("bits") == from_override.get("bits") == 8


def test_preview_threads_output_format_for_w8_asym(monkeypatch):
    # W8 asym is format-scoped: refused for native formats, allowed for
    # llm_compressor. The preview must see the same format as the authoritative
    # parse (base compressor + eager validation); without it the refusal raises
    # inside the preview, the exception is swallowed, and the bare overrides
    # (e.g. {"sym": False}) are expanded through QuantizationScheme.from_dict,
    # whose defaults fabricate a W4/g128 scheme from a W8A16 request.
    monkeypatch.delenv("AR_ALLOW_W8_ASYM", raising=False)
    overrides = {"sym": False}
    resolved = preview_resolved_attrs(overrides, "W8A16", format="auto_round:llm_compressor")
    assert resolved.get("bits") == 8
    assert resolved.get("sym") is False


def test_route_scheme_not_fabricated_from_defaults_for_w8_asym_llmc(monkeypatch):
    from auto_round.schemes import QuantizationScheme

    monkeypatch.delenv("AR_ALLOW_W8_ASYM", raising=False)
    overrides = {"sym": False}
    resolved = preview_resolved_attrs(overrides, "W8A16", format="auto_round:llm_compressor")
    scheme_obj = QuantizationScheme.from_dict(resolved)
    assert scheme_obj.bits == 8
    assert scheme_obj.sym is False


def test_preview_degrades_to_overrides_for_w8_asym_native_format(monkeypatch):
    # For formats that cannot serve W8 asym the preview still degrades to the
    # entry overrides (the refusal surfaces at eager validation, which does
    # see the format); it must never fabricate bits.
    monkeypatch.delenv("AR_ALLOW_W8_ASYM", raising=False)
    overrides = {"sym": False}
    resolved = preview_resolved_attrs(overrides, "W8A16", format="auto_round")
    assert resolved.get("bits") is None
    assert resolved.get("sym") is False


def test_preview_falls_back_to_entry_overrides_when_preview_skipped(monkeypatch):
    # An unknown scheme string makes parse_scheme raise; the resolver must then
    # surface the entry's explicit overrides (not an empty dict) so routing still
    # sees the user's bits. The degradation must also be logged, never silent.
    import auto_round.scheme_entry as scheme_entry_mod

    calls = []
    monkeypatch.setattr(scheme_entry_mod.logger, "warning_once", lambda *a, **k: calls.append(a))

    overrides = {"bits": 4, "data_type": "int"}
    resolved = preview_resolved_attrs(overrides, "definitely-not-a-real-scheme-xyz")
    assert resolved.get("bits") == 4
    assert resolved.get("data_type") == "int"
    assert len(calls) == 1
    assert "routing falls back to explicit overrides" in calls[0][0]


@pytest.mark.parametrize("sym", [True, False])
def test_routing_matches_between_scheme_only_and_equivalent_override(sym):
    from auto_round.schemes import QuantizationScheme

    scheme_config = RTNConfig()
    override_config = RTNConfig()
    scheme = QuantizationScheme(bits=4, data_type="int", sym=sym)
    via_scheme = _resolve_rtn_config(scheme_config, scheme, None, {})
    via_override = _resolve_rtn_config(
        override_config, "W4A16", None, {}, scheme_overrides={"bits": 4, "data_type": "int", "sym": sym}
    )
    assert via_scheme is scheme_config
    assert via_override is override_config
    assert scheme_config.enable_imatrix is override_config.enable_imatrix is sym
    assert isinstance(scheme_config, OptimizedRTNConfig) is sym
    assert isinstance(override_config, OptimizedRTNConfig) is sym
    assert not hasattr(scheme_config, "bits")
    assert not hasattr(override_config, "sym")


def test_w8a16_symmetric_routes_to_zero_shot():
    # sym W8A16 (bits>=8, sym True) -> no imatrix, no act calib -> zero-shot RTN path.
    cfg = RTNConfig()
    resolved = _resolve_rtn_config(cfg, "W8A16", None, {})
    assert resolved is cfg
    assert cfg.enable_imatrix is False
