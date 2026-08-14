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
"""GPU-side fast unit tests for ``auto_round.auto_scheme.utils``.

Covers the pure-Python control-flow helpers that were not previously exercised:
``merge_lists_unionfind`` (set grouping via union-find),
``_expert_key_from_layer_name`` (MoE expert path collapse),
``_short_summary_name`` (string shortening),
``_scheme_short_name`` (label formatting from QuantizationScheme/str/dict),
``apply_quant_scheme`` / ``remove_quant_scheme`` (set/get attributes on a tiny
``nn.Module``), and ``compute_layer_bits`` branches (weight_bits >= 16,
group_size == 0, group_size == -1, mx_fp/nv_fp data types, super_group_size).

All tests run on CPU in milliseconds; no model loading.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

import torch.nn as nn

from auto_round.auto_scheme.utils import (
    _expert_key_from_layer_name,
    _scheme_short_name,
    _short_summary_name,
    apply_quant_scheme,
    compute_layer_bits,
    merge_lists_unionfind,
    remove_quant_scheme,
)
from auto_round.schemes import QuantizationScheme


# ==============================================================================
# merge_lists_unionfind
# ==============================================================================


class TestMergeListsUnionFind:
    def test_disjoint_lists(self):
        groups = merge_lists_unionfind([["a", "b"], ["c", "d"]])
        # Two disjoint groups, each preserving order
        sorted_groups = [sorted(g) for g in groups]
        assert ["a", "b"] in sorted_groups
        assert ["c", "d"] in sorted_groups

    def test_overlapping_lists_merged(self):
        # Lists 1 and 2 share "b"; list 3 shares "c" with list 2
        groups = merge_lists_unionfind([["a", "b"], ["b", "c"], ["c", "d"]])
        # All four should land in a single group
        sorted_groups = [sorted(g) for g in groups]
        assert ["a", "b", "c", "d"] in sorted_groups

    def test_empty_input(self):
        assert merge_lists_unionfind([]) == []

    def test_single_list(self):
        groups = merge_lists_unionfind([["a", "b", "c"]])
        assert len(groups) == 1
        assert sorted(groups[0]) == ["a", "b", "c"]

    def test_self_referential_chain(self):
        groups = merge_lists_unionfind([["a", "a"]])
        assert sorted(groups[0]) == ["a"]


# ==============================================================================
# _expert_key_from_layer_name
# ==============================================================================


class TestExpertKeyFromLayerName:
    def test_experts_dot_id_pattern(self):
        # "model.layers.3.mlp.experts.7.gate_proj" -> "model.layers.3.mlp.experts.7"
        result = _expert_key_from_layer_name("model.layers.3.mlp.experts.7.gate_proj")
        assert result == "model.layers.3.mlp.experts.7"

    def test_experts_dot_id_with_subpath(self):
        # Even with multiple subpaths after the experts ID, the regex
        # matches the path up to ".experts.<id>" only
        result = _expert_key_from_layer_name("model.layers.3.mlp.experts.7.w.weight")
        assert result == "model.layers.3.mlp.experts.7"

    def test_moe_fallback(self):
        # "model.layers.0.moe.gate" -> strip last segment
        result = _expert_key_from_layer_name("model.layers.0.moe.gate")
        assert result == "model.layers.0.moe"

    def test_no_match_returns_none(self):
        result = _expert_key_from_layer_name("model.layers.3.self_attn.q_proj")
        assert result is None

    def test_single_segment_moe(self):
        # ".moe." in name with single segment before split
        result = _expert_key_from_layer_name("moe_experts")
        # Falls through; no .moe. pattern
        assert result is None


# ==============================================================================
# _short_summary_name
# ==============================================================================


class TestShortSummaryName:
    def test_last_segment_numeric(self):
        # Last 2 segments when final is numeric
        result = _short_summary_name("a.b.c.experts.7.gate_proj")
        # The gate_proj (last) is not numeric, so unchanged
        assert result == "a.b.c.experts.7.gate_proj"

    def test_last_segment_numeric_works(self):
        # Find an example where the LAST segment IS numeric
        result = _short_summary_name("a.b.c.5")
        # 3 parts from rsplit: ['a.b.c', '5'] -- len >= 2 and last is digit
        assert result == "c.5"

    def test_last_segment_non_numeric(self):
        # Return unchanged if final segment is not numeric
        result = _short_summary_name("model.layers.3.gate_proj")
        assert result == "model.layers.3.gate_proj"

    def test_single_segment_numeric(self):
        result = _short_summary_name("layers.0")
        # Both segments kept
        assert result == "layers.0"


# ==============================================================================
# _scheme_short_name
# ==============================================================================


class TestSchemeShortName:
    def test_string_input_returns_unchanged(self):
        assert _scheme_short_name("W4A16") == "W4A16"
        assert _scheme_short_name("MXFP4") == "MXFP4"

    def test_quantization_scheme_preset(self):
        s = QuantizationScheme.from_dict({"bits": 4, "sym": True, "group_size": 128, "data_type": "int", "act_bits": 16})
        result = _scheme_short_name(s)
        # W4A16 is the preset, so result should match
        assert result == "W4A16"

    def test_dict_mx_fp(self):
        d = {"bits": 4, "sym": True, "group_size": 32, "data_type": "mx_fp", "act_bits": 16}
        result = _scheme_short_name(d)
        assert result == "MXFP4"

    def test_dict_nv_fp(self):
        d = {"bits": 4, "sym": True, "group_size": 32, "data_type": "nv_fp", "act_bits": 16}
        result = _scheme_short_name(d)
        assert result == "NVFP4"

    def test_dict_gguf(self):
        d = {"bits": 4, "sym": True, "group_size": 32, "data_type": "int", "act_bits": 16, "super_bits": 6}
        result = _scheme_short_name(d)
        assert result.startswith("W") or result.startswith("GGUF")

    def test_dict_act_quantized(self):
        d = {"bits": 4, "sym": True, "group_size": 128, "data_type": "int", "act_bits": 8}
        result = _scheme_short_name(d)
        assert result == "W4A8"


# ==============================================================================
# apply_quant_scheme / remove_quant_scheme
# ==============================================================================


class TestApplyRemoveQuantScheme:
    def _cuda_linear(self, in_f, out_f):
        return nn.Linear(in_f, out_f).to("cuda")

    def _cuda_seq(self, *linears):
        return nn.Sequential(*[self._cuda_linear(8, 8) for _ in linears])

    def test_apply_dict_scheme_to_module(self):
        # Use a valid module name (empty string targets the root module)
        m = self._cuda_linear(8, 8)
        scheme = {"bits": 4, "sym": True, "group_size": 128, "data_type": "int", "act_bits": 16}
        apply_quant_scheme(m, [""], {}, scheme)
        assert m.bits == 4
        assert m.sym is True
        assert m.group_size == 128

    def test_apply_string_scheme(self):
        # Build a tiny model with two layers on CUDA
        model = self._cuda_seq(0, 1)
        apply_quant_scheme(model, ["0"], {}, "W4A16")
        assert model[0].bits == 4

    def test_apply_uses_fixed_layer_override(self):
        m = self._cuda_seq(0, 1)
        # Override per-layer scheme
        fixed = {"1": {"bits": 8, "sym": True, "data_type": "int"}}
        apply_quant_scheme(m, ["0", "1"], fixed, "W4A16")
        assert m[0].bits == 4  # From default
        assert m[1].bits == 8  # From fixed

    def test_remove_quant_scheme_strips_attrs(self):
        m = self._cuda_linear(8, 8)
        m.bits = 4
        m.sym = True
        m.data_type = "int"
        m.scale_dtype = torch.float16  # not a scheme field
        remove_quant_scheme(m)
        # Scheme attrs are removed
        assert not hasattr(m, "bits")
        assert not hasattr(m, "sym")
        # scale_dtype is also a scheme attr (added in register.py)
        assert not hasattr(m, "scale_dtype")


# ==============================================================================
# compute_layer_bits
# ==============================================================================


class TestComputeLayerBits:
    def _cuda_linear(self, in_f, out_f):
        # Build a real CUDA linear layer to exercise the actual CUDA tensor path
        return nn.Linear(in_f, out_f).to("cuda")

    def test_unquantized_layer_bits_16(self):
        # Layer without `bits` attr defaults to 16 (uses CUDA tensor)
        m = self._cuda_linear(8, 8)
        if hasattr(m, "bits"):
            delattr(m, "bits")
        total_bits, avg_bits = compute_layer_bits(m, ignore_scale_zp_bits=False)
        assert total_bits == 16 * 64  # 8 * 8 weight numel * 16 bits
        assert avg_bits == 16

    def test_quantized_layer_w4(self):
        m = self._cuda_linear(64, 64)
        m.bits = 4
        m.group_size = 128
        m.data_type = "int"
        m.sym = True
        total_bits, avg_bits = compute_layer_bits(m)
        # W4 only (sym int) -> 4 * numel + scales (16 each)
        assert total_bits > 4 * 64 * 64
        assert avg_bits > 4

    def test_quantized_layer_w4_ignore_overhead(self):
        m = self._cuda_linear(64, 64)
        m.bits = 4
        m.group_size = 128
        m.data_type = "int"
        m.sym = True
        total_bits, _ = compute_layer_bits(m, ignore_scale_zp_bits=True)
        # No scale/zp overhead -> exactly 4 * numel
        assert total_bits == 4 * 64 * 64

    def test_mx_fp_dtype(self):
        m = self._cuda_linear(64, 64)
        m.bits = 4
        m.group_size = 32
        m.data_type = "mx_fp"
        m.sym = True
        total_bits, _ = compute_layer_bits(m)
        # MXFP uses 8-bit scales
        assert total_bits > 0

    def test_nv_fp_dtype(self):
        m = self._cuda_linear(64, 64)
        m.bits = 4
        m.group_size = 32
        m.data_type = "nv_fp"
        m.sym = True
        total_bits, _ = compute_layer_bits(m)
        assert total_bits > 0

    def test_group_size_zero(self):
        m = self._cuda_linear(64, 64)
        m.bits = 4
        m.group_size = 0
        m.data_type = "int"
        m.sym = True
        total_bits, _ = compute_layer_bits(m)
        # Group_size == 0 -> n_group = 1
        assert total_bits > 0

    def test_group_size_neg1(self):
        m = self._cuda_linear(64, 64)
        m.bits = 4
        m.group_size = -1
        m.data_type = "int"
        m.sym = True
        total_bits, _ = compute_layer_bits(m)
        assert total_bits > 0

    def test_super_group_size_path(self):
        m = self._cuda_linear(64, 64)
        m.bits = 4
        m.group_size = 32
        m.data_type = "int_asym_dq"
        m.sym = False
        m.super_bits = 6
        m.super_group_size = 8
        total_bits, _ = compute_layer_bits(m)
        # Double-quantization path adds extra aux bits
        assert total_bits > 0

    def test_unquantized_with_super_bits(self):
        m = nn.Linear(64, 64)
        m.bits = 16  # unquantized
        m.group_size = 32
        m.data_type = "int"
        m.sym = True
        m.super_bits = 6
        m.super_group_size = 8
        total_bits, avg_bits = compute_layer_bits(m)
        # GGUF-bf16 path: returns 32 * numel, 32
        assert total_bits == 32 * 64 * 64
        assert avg_bits == 32

    def test_invalid_group_size_raises(self):
        m = nn.Linear(64, 64)
        m.bits = 4
        m.group_size = -5  # not in {0, -1, > 0}
        m.data_type = "int"
        m.sym = True
        with pytest.raises(ValueError, match="Invalid group_size"):
            compute_layer_bits(m)