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
"""Tests for the small helpers in ``auto_round/export/export_to_gguf/gguf_dtype.py``."""

import gguf  # provided by the optional dep we just installed
import pytest


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
class TestMappings:
    def test_values_unique(self):
        from auto_round.export.export_to_gguf.gguf_dtype import _GGUF_TYPE_TO_QTYPE_NAME

        qtype_names = list(_GGUF_TYPE_TO_QTYPE_NAME.values())
        # F16 appears twice (under f16 and fp16); rest are unique
        assert qtype_names.count("F16") == 2
        others = [n for n in qtype_names if n != "F16"]
        assert len(set(others)) == len(others)

    def test_qtype_name_to_gguf_type_set(self):
        """For each entry in the forward map, the reverse must agree on F16
        (since F16 has two aliases)."""
        from auto_round.export.export_to_gguf.gguf_dtype import (
            _GGUF_TYPE_TO_QTYPE_NAME,
            _QTYPE_NAME_TO_GGUF_TYPE,
        )

        # F16 has two forward aliases (f16, fp16) but only one reverse entry
        # (fp16 wins because of later assignment). The reverse map only stores
        # the canonical alias -> qtype_name pair.
        forward = _GGUF_TYPE_TO_QTYPE_NAME
        reverse = _QTYPE_NAME_TO_GGUF_TYPE
        # All forward entries map to some qtype_name; reverse is total on qtype_name
        for qtype_name in set(forward.values()):
            assert qtype_name in reverse

    def test_fp16_canonical(self):
        """F16's reverse mapping should be the canonical 'gguf:fp16'."""
        from auto_round.export.export_to_gguf.gguf_dtype import (
            _QTYPE_NAME_TO_GGUF_TYPE,
        )

        assert _QTYPE_NAME_TO_GGUF_TYPE["F16"] == "gguf:fp16"

    def test_fp16_aliasing(self):
        """`gguf:fp16` and `gguf:f16` both map to F16."""
        from auto_round.export.export_to_gguf.gguf_dtype import _GGUF_TYPE_TO_QTYPE_NAME

        assert _GGUF_TYPE_TO_QTYPE_NAME["gguf:fp16"] == "F16"
        assert _GGUF_TYPE_TO_QTYPE_NAME["gguf:f16"] == "F16"


# ---------------------------------------------------------------------------
# TensorCategory
# ---------------------------------------------------------------------------
class TestTensorCategory:
    def test_values_are_strings(self):
        from auto_round.export.export_to_gguf.gguf_dtype import TensorCategory

        for c in TensorCategory:
            assert isinstance(c.value, str)

    def test_token_embd_value(self):
        from auto_round.export.export_to_gguf.gguf_dtype import TensorCategory

        assert TensorCategory.TOKEN_EMBD.value == "token_embd"

    def test_per_layer_token_embd_value(self):
        from auto_round.export.export_to_gguf.gguf_dtype import TensorCategory

        assert TensorCategory.TOKEN_EMBD.value == "token_embd"
        # _tensor_category recognizes the per-layer prefix, so verify behavior below.


# ---------------------------------------------------------------------------
# _tensor_category
# ---------------------------------------------------------------------------
class TestTensorCategoryFunction:
    def test_output_weight(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        assert _tensor_category("output.weight") == TensorCategory.OUTPUT

    def test_token_embd(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        assert _tensor_category("token_embd.weight") == TensorCategory.TOKEN_EMBD
        assert _tensor_category("per_layer_token_embd.weight") == TensorCategory.TOKEN_EMBD

    def test_attn_qkv(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        assert _tensor_category("blk.0.attn_qkv.weight") == TensorCategory.ATTENTION_QKV

    def test_attn_kv_b(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        assert _tensor_category("blk.0.attn_kv_b.weight") == TensorCategory.ATTENTION_KV_B

    def test_attn_q_k_v(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        assert _tensor_category("blk.0.attn_q.weight") == TensorCategory.ATTENTION_Q
        assert _tensor_category("blk.0.attn_k.weight") == TensorCategory.ATTENTION_K
        assert _tensor_category("blk.0.attn_v.weight") == TensorCategory.ATTENTION_V

    def test_attn_output(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        assert _tensor_category("blk.0.attn_output.weight") == TensorCategory.ATTENTION_OUTPUT

    def test_ffn_up_gate_down(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        assert _tensor_category("blk.0.ffn_up.weight") == TensorCategory.FFN_UP
        assert _tensor_category("blk.0.ffn_gate.weight") == TensorCategory.FFN_GATE
        assert _tensor_category("blk.0.ffn_down.weight") == TensorCategory.FFN_DOWN

    def test_other(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _tensor_category,
        )

        # Anything not matching above falls into OTHER
        assert _tensor_category("blk.0.unknown.weight") == TensorCategory.OTHER


# ---------------------------------------------------------------------------
# _is_attn_v_like
# ---------------------------------------------------------------------------
class TestIsAttnVLike:
    def test_true_for_v_qkv_kv_b(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _is_attn_v_like,
        )

        assert _is_attn_v_like(TensorCategory.ATTENTION_V) is True
        assert _is_attn_v_like(TensorCategory.ATTENTION_QKV) is True
        assert _is_attn_v_like(TensorCategory.ATTENTION_KV_B) is True

    def test_false_for_others(self):
        from auto_round.export.export_to_gguf.gguf_dtype import (
            TensorCategory,
            _is_attn_v_like,
        )

        assert _is_attn_v_like(TensorCategory.ATTENTION_Q) is False
        assert _is_attn_v_like(TensorCategory.ATTENTION_K) is False
        assert _is_attn_v_like(TensorCategory.ATTENTION_OUTPUT) is False
        assert _is_attn_v_like(TensorCategory.TOKEN_EMBD) is False


# ---------------------------------------------------------------------------
# _use_more_bits
# ---------------------------------------------------------------------------
class TestUseMoreBits:
    def test_first_eighth(self):
        """First 1/8 of layers should use more bits."""
        from auto_round.export.export_to_gguf.gguf_dtype import _use_more_bits

        # 8 layers: first 8/8=1 layer uses more bits
        for i in range(0, 1):
            assert _use_more_bits(i, 8) is True

    def test_last_eighth(self):
        """Last 1/8 of layers should use more bits."""
        from auto_round.export.export_to_gguf.gguf_dtype import _use_more_bits

        # Last 1 of 8 layers
        assert _use_more_bits(7, 8) is True

    def test_middle_layer_no_extra_bits(self):
        """Layers that don't satisfy any of the three predicates should return False.

        Formula: ``i < n//8 or i >= 7*n//8 or (i - n//8) % 3 == 2``
        For n=16, n//8=2, 7*n//8=14:
          i=0 -> first eighth
          i=8 -> (8-2)%3=0 -> False
        """
        from auto_round.export.export_to_gguf.gguf_dtype import _use_more_bits

        # 16 layers, index 8 -> (8-2)%3 == 0
        assert _use_more_bits(8, 16) is False

    def test_use_more_bits_periodic(self):
        """``(i - n//8) % 3 == 2`` produces a periodic True pattern."""
        from auto_round.export.export_to_gguf.gguf_dtype import _use_more_bits

        # For n=24, n//8=3; check the predicate directly via formula
        # i=5 -> (5-3)%3 = 2 -> True
        assert _use_more_bits(5, 24) is True


# ---------------------------------------------------------------------------
# _get_layer_id
# ---------------------------------------------------------------------------
class TestGetLayerId:
    def test_blk_prefix_digits(self):
        from auto_round.export.export_to_gguf.gguf_dtype import _get_layer_id

        assert _get_layer_id("blk.5.attn_q.weight", fallback=99) == 5

    def test_no_blk_prefix_returns_fallback(self):
        from auto_round.export.export_to_gguf.gguf_dtype import _get_layer_id

        assert _get_layer_id("some.other.weight", fallback=42) == 42

    def test_single_segment_returns_fallback(self):
        from auto_round.export.export_to_gguf.gguf_dtype import _get_layer_id

        assert _get_layer_id("weight", fallback=10) == 10

    def test_blk_zero(self):
        from auto_round.export.export_to_gguf.gguf_dtype import _get_layer_id

        assert _get_layer_id("blk.0.attn_q.weight", fallback=99) == 0

    def test_blk_with_negative(self):
        """blk.-1 is not a digit-only part -> fallback."""
        from auto_round.export.export_to_gguf.gguf_dtype import _get_layer_id

        assert _get_layer_id("blk.-1.attn_q.weight", fallback=99) == 99


# ---------------------------------------------------------------------------
# gguf_format_to_ftype
# ---------------------------------------------------------------------------
class TestGgufFormatToFtype:
    @pytest.mark.parametrize(
        "format_name,expected_name",
        [
            ("gguf:f32", "ALL_F32"),
            ("gguf:fp16", "MOSTLY_F16"),
            ("gguf:f16", "MOSTLY_F16"),
            ("gguf:bf16", "MOSTLY_BF16"),
            ("gguf:q4_0", "MOSTLY_Q4_0"),
            ("gguf:q4_1", "MOSTLY_Q4_1"),
            ("gguf:q5_0", "MOSTLY_Q5_0"),
            ("gguf:q5_1", "MOSTLY_Q5_1"),
            ("gguf:q8_0", "MOSTLY_Q8_0"),
            ("gguf:q4_k_m", "MOSTLY_Q4_K_M"),
            ("gguf:q5_k_m", "MOSTLY_Q5_K_M"),
            ("gguf:q6_k", "MOSTLY_Q6_K"),
        ],
    )
    def test_known_formats(self, format_name, expected_name):
        from auto_round.export.export_to_gguf.gguf_dtype import gguf_format_to_ftype

        ftype = gguf_format_to_ftype(format_name)
        assert ftype.name == expected_name

    def test_q2_k_mixed_renames(self):
        from auto_round.export.export_to_gguf.gguf_dtype import gguf_format_to_ftype

        ftype = gguf_format_to_ftype("gguf:q2_k_mixed")
        assert ftype.name == "MOSTLY_Q2_K_S"

    def test_unknown_raises(self):
        from auto_round.export.export_to_gguf.gguf_dtype import gguf_format_to_ftype

        with pytest.raises(ValueError):
            gguf_format_to_ftype("gguf:not_a_real_format")


# ---------------------------------------------------------------------------
# GGUFDTypeSelector
# ---------------------------------------------------------------------------
class TestGGUFDTypeSelector:
    def test_construction_stores_args(self):
        from auto_round.export.export_to_gguf.gguf_dtype import GGUFDTypeSelector

        hparams = {"num_hidden_layers": 24, "num_attention_heads": 8, "num_key_value_heads": 4}
        ftype = gguf.LlamaFileType.MOSTLY_Q4_K_M
        sel = GGUFDTypeSelector(hparams, ftype)
        assert sel.hparams is hparams
        assert sel.ftype is ftype
        assert sel.i_attention_wv == 0
        assert sel.i_ffn_down == 0
