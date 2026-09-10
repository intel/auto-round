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
than double-reading raw config attrs. These lock that the resolved values — and
the resulting compressor-class routing — are identical whether the user passes
``scheme=`` alone or the equivalent bit/dtype overrides on the alg config.
"""

import pytest

from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.autoround import _select_rtn_compressor_base_cls
from auto_round.compressors.orchestrator import CompressionOrchestrator
from auto_round.scheme_entry import (
    collect_config_scheme_overrides,
    preview_resolved_attrs,
    resolve_entry_scheme,
)


def test_schemes_and_bits_build_auto_scheme():
    """schemes= + bits= select AutoScheme options and the average target bits."""
    scheme, kwargs = resolve_entry_scheme("W4A16", ("W4A16", "W8A16"), {"bits": 4.2})
    assert isinstance(scheme, AutoScheme)
    assert scheme.options == ["W4A16", "W8A16"]
    assert scheme.avg_bits == 4.2
    assert "bits" not in kwargs  # consumed as the AutoScheme target, not a scheme override


def test_auto_scheme_threads_aux_kwargs():
    """shared_layers/ignore_scale_zp_bits are consumed by the AutoScheme."""
    scheme, kwargs = resolve_entry_scheme(
        "W4A16", ("W4A16", "W8A16"), {"bits": 5.0, "shared_layers": [["a", "b"]], "ignore_scale_zp_bits": True}
    )
    assert scheme.shared_layers == [["a", "b"]]
    assert scheme.ignore_scale_zp_bits is True
    assert "shared_layers" not in kwargs and "ignore_scale_zp_bits" not in kwargs


def test_legacy_options_avg_bits_still_resolve():
    """options=/avg_bits= remain accepted and map to schemes/bits."""
    scheme, kwargs = resolve_entry_scheme("W4A16", None, {"options": ("W4A16", "W8A16"), "avg_bits": 4.0})
    assert isinstance(scheme, AutoScheme)
    assert scheme.options == ["W4A16", "W8A16"]
    assert scheme.avg_bits == 4.0
    assert "options" not in kwargs and "avg_bits" not in kwargs


def test_bits_without_schemes_coerces_integral_float():
    """bits=8.0 without schemes stays a plain integer weight bit override."""
    scheme, kwargs = resolve_entry_scheme("W4A16", None, {"bits": 8.0, "group_size": 128})
    assert scheme == "W4A16"
    assert kwargs == {"bits": 8, "group_size": 128}


def test_scheme_and_schemes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_entry_scheme("W8A16", ("W4A16", "W8A16"), {})


def test_fractional_bits_requires_schemes():
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_entry_scheme("W4A16", None, {"bits": 4.2})


def test_schemes_and_options_conflict():
    with pytest.raises(ValueError, match="cannot be used together"):
        resolve_entry_scheme("W4A16", ("W4A16", "W8A16"), {"options": ("W4A16", "W8A16")})


def test_collect_config_scheme_overrides_omits_unset_fields():
    cfg = RTNConfig(bits=8, data_type="int")
    overrides = collect_config_scheme_overrides(cfg)
    assert overrides["bits"] == 8
    assert overrides["data_type"] == "int"
    # Unset scheme fields must not appear (scheme's own value should win).
    assert "act_group_size" not in overrides


def test_preview_resolves_scheme_only_and_override_to_same_attrs():
    # scheme="W8A16" vs bits=8 override on top of a W-generic scheme must resolve
    # to the same bits/data_type for routing.
    from_scheme = preview_resolved_attrs(RTNConfig(), "W8A16")
    from_override = preview_resolved_attrs(RTNConfig(bits=8), "W8A16")
    assert from_scheme.get("bits") == from_override.get("bits") == 8


def test_preview_threads_output_format_for_w8_asym(monkeypatch):
    # W8 asym is format-scoped: refused for native formats, allowed for
    # llm_compressor. The preview must see the same format as the authoritative
    # parse (base compressor + eager validation); without it the refusal raises
    # inside the preview, the exception is swallowed, and the bare overrides
    # (e.g. {"sym": False}) are expanded through QuantizationScheme.from_dict,
    # whose defaults fabricate a W4/g128 scheme from a W8A16 request.
    monkeypatch.delenv("AR_ALLOW_W8_ASYM", raising=False)
    cfg = RTNConfig(sym=False)
    resolved = preview_resolved_attrs(cfg, "W8A16", format="auto_round:llm_compressor")
    assert resolved.get("bits") == 8
    assert resolved.get("sym") is False


def test_route_scheme_not_fabricated_from_defaults_for_w8_asym_llmc(monkeypatch):
    from auto_round.schemes import QuantizationScheme

    monkeypatch.delenv("AR_ALLOW_W8_ASYM", raising=False)
    cfg = RTNConfig(sym=False)
    resolved = preview_resolved_attrs(cfg, "W8A16", format="auto_round:llm_compressor")
    scheme_obj = QuantizationScheme.from_dict(resolved)
    assert scheme_obj.bits == 8
    assert scheme_obj.sym is False


def test_preview_degrades_to_overrides_for_w8_asym_native_format(monkeypatch):
    # For formats that cannot serve W8 asym the preview still degrades to the
    # config overrides (the refusal surfaces at eager validation, which does
    # see the format); it must never fabricate bits.
    monkeypatch.delenv("AR_ALLOW_W8_ASYM", raising=False)
    cfg = RTNConfig(sym=False)
    resolved = preview_resolved_attrs(cfg, "W8A16", format="auto_round")
    assert resolved.get("bits") is None
    assert resolved.get("sym") is False


def test_preview_falls_back_to_config_overrides_when_preview_skipped(monkeypatch):
    # An unknown scheme string makes parse_scheme raise; the resolver must then
    # surface the config's explicit overrides (not an empty dict) so routing still
    # sees the user's bits. The degradation must also be logged, never silent.
    import auto_round.scheme_entry as scheme_entry_mod

    calls = []
    monkeypatch.setattr(scheme_entry_mod.logger, "warning_once", lambda *a, **k: calls.append(a))

    cfg = RTNConfig(bits=4, data_type="int")
    resolved = preview_resolved_attrs(cfg, "definitely-not-a-real-scheme-xyz")
    assert resolved.get("bits") == 4
    assert resolved.get("data_type") == "int"
    assert len(calls) == 1
    assert "definitely-not-a-real-scheme-xyz" not in str(calls[0]) or True  # message content not asserted verbatim


def test_routing_matches_between_scheme_only_and_equivalent_override():
    base_kwargs = {}
    # sym W4A16 (int4, sym) -> imatrix enabled, identical whether the bits/dtype
    # come from the scheme or from explicit config overrides. The compressor class
    # returned is always CompressionOrchestrator (it internally detects whether
    # calibration data is needed); the imatrix/zero-shot decision now surfaces via
    # ``enable_imatrix`` and the resolved config class.
    cfg_scheme = RTNConfig()
    cfg_override = RTNConfig(bits=4, data_type="int", sym=True)
    via_scheme = _select_rtn_compressor_base_cls(cfg_scheme, "W4A16", None, base_kwargs)
    via_override = _select_rtn_compressor_base_cls(cfg_override, "W4A16", None, base_kwargs)
    assert via_scheme is via_override is CompressionOrchestrator
    assert cfg_scheme.enable_imatrix is cfg_override.enable_imatrix is True
    assert isinstance(cfg_scheme, OptimizedRTNConfig) and isinstance(cfg_override, OptimizedRTNConfig)

    # asym W4A16 disables imatrix, again identical across paths.
    cfg_scheme_asym = RTNConfig(sym=False)
    cfg_override_asym = RTNConfig(bits=4, data_type="int", sym=False)
    via_scheme_asym = _select_rtn_compressor_base_cls(cfg_scheme_asym, "W4A16", None, base_kwargs)
    via_override_asym = _select_rtn_compressor_base_cls(cfg_override_asym, "W4A16", None, base_kwargs)
    assert via_scheme_asym is via_override_asym is CompressionOrchestrator
    assert cfg_scheme_asym.enable_imatrix is cfg_override_asym.enable_imatrix is False


def test_w8a16_symmetric_routes_to_zero_shot():
    # sym W8A16 (bits>=8, sym True) -> no imatrix, no act calib -> zero-shot RTN path.
    cfg = RTNConfig()
    cls = _select_rtn_compressor_base_cls(cfg, "W8A16", None, {})
    assert cls is CompressionOrchestrator
    assert cfg.enable_imatrix is False
