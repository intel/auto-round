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
fields from a single resolved-scheme source (``_preview_resolved_attrs``) rather
than double-reading raw config attrs. These lock that the resolved values — and
the resulting compressor-class routing — are identical whether the user passes
``scheme=`` alone or the equivalent bit/dtype overrides on the alg config.
"""

from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.compressors.data_driven import CalibratedRTNCompressor
from auto_round.compressors.entry import (
    _collect_config_scheme_overrides,
    _preview_resolved_attrs,
    _select_rtn_compressor_base_cls,
)
from auto_round.compressors.zero_shot import ZeroShotCompressor


def test_collect_config_scheme_overrides_omits_unset_fields():
    cfg = RTNConfig(bits=8, data_type="int")
    overrides = _collect_config_scheme_overrides(cfg)
    assert overrides["bits"] == 8
    assert overrides["data_type"] == "int"
    # Unset scheme fields must not appear (scheme's own value should win).
    assert "act_group_size" not in overrides


def test_preview_resolves_scheme_only_and_override_to_same_attrs():
    # scheme="W8A16" vs bits=8 override on top of a W-generic scheme must resolve
    # to the same bits/data_type for routing.
    from_scheme = _preview_resolved_attrs(RTNConfig(), "W8A16")
    from_override = _preview_resolved_attrs(RTNConfig(bits=8), "W8A16")
    assert from_scheme.get("bits") == from_override.get("bits") == 8


def test_preview_falls_back_to_config_overrides_when_preview_skipped():
    # An unknown scheme string makes parse_scheme raise; the resolver must then
    # surface the config's explicit overrides (not an empty dict) so routing still
    # sees the user's bits.
    cfg = RTNConfig(bits=4, data_type="int")
    resolved = _preview_resolved_attrs(cfg, "definitely-not-a-real-scheme-xyz")
    assert resolved.get("bits") == 4
    assert resolved.get("data_type") == "int"


def test_routing_matches_between_scheme_only_and_equivalent_override():
    base_kwargs = {}
    # sym W4A16 (int4, sym) -> imatrix -> CalibratedRTNCompressor, identical whether
    # the bits/dtype come from the scheme or from explicit config overrides.
    via_scheme = _select_rtn_compressor_base_cls(RTNConfig(), "W4A16", None, base_kwargs)
    via_override = _select_rtn_compressor_base_cls(
        RTNConfig(bits=4, data_type="int", sym=True), "W4A16", None, base_kwargs
    )
    assert via_scheme is via_override is CalibratedRTNCompressor

    # asym W4A16 disables imatrix -> ZeroShotCompressor, again identical across paths.
    via_scheme_asym = _select_rtn_compressor_base_cls(RTNConfig(sym=False), "W4A16", None, base_kwargs)
    via_override_asym = _select_rtn_compressor_base_cls(
        RTNConfig(bits=4, data_type="int", sym=False), "W4A16", None, base_kwargs
    )
    assert via_scheme_asym is via_override_asym is ZeroShotCompressor


def test_w8a16_symmetric_routes_to_zero_shot():
    # sym W8A16 (bits>=8, sym True) -> no imatrix, no act calib -> ZeroShotCompressor.
    cls = _select_rtn_compressor_base_cls(RTNConfig(), "W8A16", None, {})
    assert cls is ZeroShotCompressor
