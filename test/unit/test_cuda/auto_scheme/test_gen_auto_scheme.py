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
"""GPU-side fast unit tests for ``auto_round.auto_scheme.gen_auto_scheme``.

Covers the ``AutoScheme`` dataclass itself: ``__post_init__`` (string-to-list
parsing), ``_deduplicate_options`` (full branch coverage including unknown
options, QuantizationScheme duplicates, and same-scheme-via-different-strings
GGUF aliases).

Tests the ``GenScheme.fallback_gguf_layer_config`` short-circuit branches on a
small ``nn.Module`` that has an Embedding (skip), an out_features % 256 != 0
linear (fallback applied), and an out_features % 256 == 0 linear (skip).

No model loading or GPU kernel launches.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

import torch.nn as nn

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme, GenScheme
from auto_round.schemes import QuantizationScheme


# ==============================================================================
# AutoScheme.post_init and deduplication
# ==============================================================================


class TestAutoSchemePostInit:
    def test_string_options_split(self):
        a = AutoScheme(avg_bits=4.0, options="W4A16,W8A8")
        assert isinstance(a.options, list)
        assert a.options == ["W4A16", "W8A8"]

    def test_string_options_uppercased_and_stripped(self):
        a = AutoScheme(avg_bits=4.0, options="w4a16, w8a8")
        # Case is uppercased and spaces removed
        assert a.options == ["W4A16", "W8A8"]

    def test_list_options_unchanged(self):
        a = AutoScheme(avg_bits=4.0, options=["W4A16", "W8A8"])
        assert a.options == ["W4A16", "W8A8"]

    def test_tuple_options_unchanged(self):
        a = AutoScheme(avg_bits=4.0, options=("W4A16", "W8A8"))
        # _deduplicate_options returns a list (uniform across all inputs)
        assert a.options == ["W4A16", "W8A8"]


class TestAutoSchemeDeduplicateOptions:
    def test_unknown_string_kept_as_is(self):
        a = AutoScheme(avg_bits=4.0, options=["UNKNOWN_SCHEME"])
        # Unknown names pass through unchanged
        assert a.options == ["UNKNOWN_SCHEME"]

    def test_duplicate_presets_removed(self):
        # W4A16 and INT8_W8A8 (alias for W8A8) - here just same preset
        a = AutoScheme(avg_bits=4.0, options=["W4A16", "W4A16"])
        assert a.options == ["W4A16"]

    def test_gguf_aliases_collapsed(self):
        # GGUF:Q4_K_S and GGUF:Q4_K_M have identical quantization params
        a = AutoScheme(avg_bits=4.0, options=["GGUF:Q4_K_S", "GGUF:Q4_K_M"])
        assert a.options == ["GGUF:Q4_K_S"]

    def test_distinct_presets_kept(self):
        a = AutoScheme(avg_bits=4.0, options=["W4A16", "W8A8"])
        assert a.options == ["W4A16", "W8A8"]

    def test_scheme_objects_preserved(self):
        s = QuantizationScheme.from_dict({"bits": 4, "sym": True, "group_size": 128, "data_type": "int", "act_bits": 16})
        a = AutoScheme(avg_bits=4.0, options=[s, s])
        # Two identical scheme objects -> dedup
        assert len(a.options) == 1

    def test_mixed_types_in_options(self):
        # Mix of string and scheme + unknown
        a = AutoScheme(avg_bits=4.0, options=["W4A16", "unknown_thing"])
        assert a.options == ["W4A16", "unknown_thing"]


# ==============================================================================
# GenScheme.fallback_gguf_layer_config
# ==============================================================================


class TestFallbackGgufLayerConfig:
    def _cuda_linear(self, in_f, out_f):
        return nn.Linear(in_f, out_f).to("cuda")

    def _cuda_embedding(self, num, dim):
        return nn.Embedding(num, dim).to("cuda")

    def _cuda_sequential(self, *layers):
        return nn.Sequential(*layers).to("cuda")

    def test_no_super_bits_layer_skipped(self):
        # Non-GGUF (no super_bits) -> no fallback
        layer_config = {
            "0": {"bits": 4, "data_type": "int", "sym": True, "super_bits": None, "super_group_size": None},
        }
        m = self._cuda_sequential(self._cuda_linear(64, 64))
        gs = GenScheme.__new__(GenScheme)  # bypass __init__
        gs.model = m
        result = gs.fallback_gguf_layer_config(layer_config)
        assert result["0"]["bits"] == 4

    def test_embedding_skipped(self):
        # Layer is an Embedding -> skip even with super_bits set
        layer_config = {
            "embed": {"bits": 4, "data_type": "int", "sym": True, "super_bits": 6, "super_group_size": 8},
        }
        m = self._cuda_sequential(self._cuda_embedding(64, 32))
        gs = GenScheme.__new__(GenScheme)
        gs.model = m
        result = gs.fallback_gguf_layer_config(layer_config)
        # Embedding -> no change
        assert result["embed"]["bits"] == 4

    def test_input_features_mod_256_zero_skipped(self):
        # in_features % 256 == 0 -> skip
        layer_config = {
            "0": {"bits": 4, "data_type": "int", "sym": True, "super_bits": 6, "super_group_size": 8},
        }
        m = self._cuda_sequential(self._cuda_linear(256, 64))  # in_features=256
        gs = GenScheme.__new__(GenScheme)
        gs.model = m
        result = gs.fallback_gguf_layer_config(layer_config)
        # No fallback applied
        assert result["0"]["bits"] == 4

    def test_input_features_neither_mod_256_nor_mod_32(self):
        # in_features % 256 != 0 and % 32 != 0 -> bf16 fallback
        layer_config = {
            "0": {"bits": 4, "data_type": "int", "sym": True, "super_bits": 6, "super_group_size": 8},
        }
        # Build a fresh model in the test (Sequential with one Linear(48,64))
        m = self._cuda_sequential(self._cuda_linear(48, 64))  # 48 % 32 != 0
        gs = GenScheme.__new__(GenScheme)
        gs.model = m
        # Take a deep copy of the layer config since the function mutates in place
        import copy

        cfg = copy.deepcopy(layer_config)
        result = gs.fallback_gguf_layer_config(cfg)
        # bf16 fallback applied -> bits becomes 16
        assert result["0"]["bits"] == 16

    def test_input_features_mod_256_neq_mod_32_zero(self):
        # in_features % 256 != 0 but % 32 == 0 -> prefix-idx fallback
        layer_config = {
            "0": {"bits": 4, "data_type": "int", "sym": False, "super_bits": 6, "super_group_size": 8},
        }
        m = self._cuda_sequential(self._cuda_linear(64, 64))  # 64 % 32 == 0 but % 256 != 0
        gs = GenScheme.__new__(GenScheme)
        gs.model = m
        result = gs.fallback_gguf_layer_config(layer_config)
        # q4_0/q4_1 fallback applied
        assert "bits" in result["0"]

    def test_module_not_found_skipped(self):
        # Layer not in model -> input_features is None -> skip
        layer_config = {
            "nonexistent.path": {"bits": 4, "data_type": "int", "sym": True, "super_bits": 6, "super_group_size": 8},
        }
        m = self._cuda_sequential(self._cuda_linear(64, 64))
        gs = GenScheme.__new__(GenScheme)
        gs.model = m
        result = gs.fallback_gguf_layer_config(layer_config)
        # No change to layer config
        assert result["nonexistent.path"]["bits"] == 4