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
"""Tests for the pure helpers in ``auto_round/auto_scheme/utils.py``."""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# merge_lists_unionfind
# ---------------------------------------------------------------------------
class TestMergeListsUnionFind:
    def test_empty_input(self):
        from auto_round.auto_scheme.utils import merge_lists_unionfind
        assert merge_lists_unionfind([]) == []

    def test_single_list(self):
        from auto_round.auto_scheme.utils import merge_lists_unionfind
        result = merge_lists_unionfind([["a", "b", "c"]])
        assert sorted(result) == [["a", "b", "c"]]

    def test_disjoint_lists_remain_separate(self):
        from auto_round.auto_scheme.utils import merge_lists_unionfind
        result = merge_lists_unionfind([["a", "b"], ["c", "d"]])
        groups = sorted([sorted(g) for g in result])
        assert groups == [["a", "b"], ["c", "d"]]

    def test_overlapping_lists_are_merged(self):
        from auto_round.auto_scheme.utils import merge_lists_unionfind
        # "b" and "c" overlap -> all 4 should end up in a single group
        result = merge_lists_unionfind([["a", "b"], ["b", "c"], ["c", "d"]])
        assert len(result) == 1
        assert sorted(result[0]) == ["a", "b", "c", "d"]

    def test_three_way_overlap(self):
        from auto_round.auto_scheme.utils import merge_lists_unionfind
        result = merge_lists_unionfind([["a", "b"], ["c", "d"], ["b", "c"]])
        assert len(result) == 1
        assert sorted(result[0]) == ["a", "b", "c", "d"]

    def test_long_chain(self):
        from auto_round.auto_scheme.utils import merge_lists_unionfind
        result = merge_lists_unionfind(
            [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]]
        )
        assert len(result) == 1
        assert sorted(result[0]) == ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------------------
# compute_layer_bits
# ---------------------------------------------------------------------------
class TestComputeLayerBits:
    """Compute-layer-bits reference values come from the actual code path:

    * ``scale_bits = 8`` for ``mx_fp / nv_fp / fp4`` data types, ``16`` otherwise.
    * ``zp_bits   = bits if (not sym) OR ("int" in data_type) else 0``
    * aux per group = scale_bits + zp_bits
    * n_group: ``out_features * ceil(in_features / group_size)`` for group_size>0;
      1 for 0; out_features for -1.
    """

    def _make_layer(self, **attrs):
        layer = nn.Linear(8, 4)  # 32 params
        for k, v in attrs.items():
            setattr(layer, k, v)
        return layer

    def test_unquantized_layer_default_16_bits(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer()
        total, avg = compute_layer_bits(layer)
        assert total == 16 * 32
        assert avg == 16.0

    def test_unquantized_with_ignore_overhead(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=16)
        total, _ = compute_layer_bits(layer, ignore_scale_zp_bits=True)
        assert total == 16 * 32

    def test_int4_sym_with_group(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=4, group_size=4, sym=True, data_type="int")
        # sym=True but data_type contains "int" -> zp_bits = 4
        # scale_bits = 16 (default)
        # aux per group = 20; n_group = 4 * ceil(8/4) = 8 -> aux_total = 160
        # weight = 128 -> total = 288
        total, avg = compute_layer_bits(layer)
        assert total == 288
        assert avg == pytest.approx(288 / 32)

    def test_int4_asym_with_group(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=4, group_size=4, sym=False, data_type="int")
        # asym -> zp_bits = 4; aux per group = 20; aux_total = 160; total = 288
        total, _ = compute_layer_bits(layer)
        assert total == 288

    def test_mx_fp_uses_8bit_scale_no_zp(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=4, group_size=4, sym=True, data_type="mx_fp4")
        # scale_bits=8, zp_bits=0 (sym AND not "int" in data_type)
        # aux per group = 8; n_group = 8 -> aux_total = 64
        # weight = 128 -> total = 192
        total, _ = compute_layer_bits(layer)
        assert total == 192

    def test_group_size_zero(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=4, group_size=0, sym=True, data_type="int")
        # n_group = 1; aux = 20; weight = 128 -> total = 148
        total, _ = compute_layer_bits(layer)
        assert total == 148

    def test_group_size_neg1(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=4, group_size=-1, sym=True, data_type="int")
        # n_group = out_features = 4; aux = 80; weight = 128 -> total = 208
        total, _ = compute_layer_bits(layer)
        assert total == 208

    def test_invalid_group_size_raises(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=4, group_size=-99, sym=True, data_type="int")
        with pytest.raises(ValueError):
            compute_layer_bits(layer)

    def test_super_group_uses_super_bits(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(
            bits=4, group_size=4, sym=True, data_type="int",
            super_group_size=2, super_bits=6,
        )
        # aux1 = 8 * 6 * 2 = 96; n_super_group = ceil(8/2) = 4; aux2 = 4 * 32 * 2 = 256
        # total aux = 352; weight = 128 -> total = 480
        total, _ = compute_layer_bits(layer)
        assert total == 480

    def test_cached_weight_numel_used_when_weight_empty(self):
        from auto_round.auto_scheme.utils import compute_layer_bits
        layer = self._make_layer(bits=4, group_size=4, sym=True, data_type="int")
        layer._cached_weight_numel = 32
        layer.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        total, _ = compute_layer_bits(layer)
        # Same numbers as test_int4_sym_with_group
        assert total == 288


# ---------------------------------------------------------------------------
# apply_quant_scheme / remove_quant_scheme
# ---------------------------------------------------------------------------
class TestApplyRemoveQuantScheme:
    def test_apply_with_string_scheme(self):
        from auto_round.auto_scheme.utils import apply_quant_scheme
        from dataclasses import fields
        from auto_round.schemes import QuantizationScheme

        model = nn.Sequential(nn.Linear(8, 4))
        apply_quant_scheme(
            model,
            quant_layer_names=["0"],
            fixed_layer_scheme={},
            scheme="W4A16",  # valid preset
        )
        for f in fields(QuantizationScheme):
            assert hasattr(model[0], f.name)

    def test_apply_with_dict_scheme(self):
        from auto_round.auto_scheme.utils import apply_quant_scheme

        model = nn.Sequential(nn.Linear(8, 4))
        apply_quant_scheme(
            model,
            quant_layer_names=["0"],
            fixed_layer_scheme={},
            scheme={"bits": 8, "group_size": 64, "sym": True, "data_type": "int"},
        )
        assert model[0].bits == 8
        assert model[0].group_size == 64
        assert model[0].sym is True

    def test_apply_with_per_layer_override(self):
        from auto_round.auto_scheme.utils import apply_quant_scheme

        model = nn.Sequential(nn.Linear(8, 4))
        fixed = {"0": {"bits": 2, "group_size": 32, "sym": True, "data_type": "int"}}
        apply_quant_scheme(
            model,
            quant_layer_names=["0"],
            fixed_layer_scheme=fixed,
            scheme="W4A16",
        )
        # Per-layer override beats the preset
        assert model[0].bits == 2
        assert model[0].group_size == 32

    def test_remove_clears_scheme_attrs(self):
        from auto_round.auto_scheme.utils import apply_quant_scheme, remove_quant_scheme

        model = nn.Sequential(nn.Linear(8, 4))
        scheme = {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"}
        apply_quant_scheme(
            model,
            quant_layer_names=["0"],
            fixed_layer_scheme={},
            scheme=scheme,
        )
        # Pre-condition: every key in the supplied scheme exists on the layer
        for key in scheme:
            assert hasattr(model[0], key)

        remove_quant_scheme(model)

        # After removal, every key in the scheme dict is gone from the layer
        for key in scheme:
            assert not hasattr(model[0], key)

    def test_remove_preserves_root_rotation_config(self):
        from auto_round.auto_scheme.utils import (
            apply_quant_scheme,
            remove_quant_scheme,
        )

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.rotation_config = "must_survive"
                self.linear = nn.Linear(4, 4)

        m = _Model()
        apply_quant_scheme(
            m,
            quant_layer_names=["linear"],
            fixed_layer_scheme={},
            scheme={"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
        )
        remove_quant_scheme(m)
        # Root.rotation_config must NOT be touched
        assert m.rotation_config == "must_survive"
        # But the inner layer's scheme attributes should be cleared
        assert not hasattr(m.linear, "bits")