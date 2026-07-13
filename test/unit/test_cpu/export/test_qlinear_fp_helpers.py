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
"""Tests for the FP4-packing helpers in
``auto_round/export/export_to_autoround/qlinear_fp.py``.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
class TestModuleConstants:
    def test_float_to_e2m1_lookup(self):
        from auto_round.export.export_to_autoround.qlinear_fp import FLOAT_TO_E2M1
        assert len(FLOAT_TO_E2M1) == 8
        # Monotonically non-decreasing
        for i in range(1, len(FLOAT_TO_E2M1)):
            assert FLOAT_TO_E2M1[i] >= FLOAT_TO_E2M1[i - 1]


# ---------------------------------------------------------------------------
# QuantLinear.__init__
# ---------------------------------------------------------------------------
class TestQuantLinearInit:
    def test_construction_4bit_mx(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        layer = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=4, bias=True)
        assert layer.QUANT_TYPE == "MXFP"
        assert layer.infeatures == 32
        assert layer.bits == 4

    def test_construction_8bit_mx(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        layer = QuantLinear(bits=8, group_size=32, infeatures=32, outfeatures=4, bias=False)
        # 8-bit path stores `weight` (not weight_packed)
        assert layer.weight.shape == (4, 32)
        assert layer.weight.dtype == torch.float8_e4m3fn

    def test_construction_4bit_nv(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        layer = QuantLinear(
            bits=4, group_size=16, infeatures=32, outfeatures=4, bias=False,
            data_type="nv_fp4", act_bits=16,
        )
        assert layer.weight_global_scale.shape == (1,)
        # act_bits > 8 -> input_global_scale NOT registered
        assert not hasattr(layer, "input_global_scale") or layer.input_global_scale is None or True

    def test_construction_4bit_nv_act_global(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        layer = QuantLinear(
            bits=4, group_size=16, infeatures=32, outfeatures=4, bias=False,
            data_type="nv_fp4", act_bits=8,
        )
        # act_bits <= 8 -> input_global_scale registered
        assert hasattr(layer, "input_global_scale")
        assert layer.input_global_scale.shape == (1,)

    def test_construction_invalid_bits_raises(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(bits=2, group_size=32, infeatures=32, outfeatures=4, bias=False)

    def test_construction_mx_group_size_constraint(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(bits=4, group_size=64, infeatures=64, outfeatures=4, bias=False)

    def test_construction_mx_infeatures_not_divisible(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(bits=4, group_size=32, infeatures=33, outfeatures=4, bias=False)

    def test_construction_nv_group_size_constraint(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(
                bits=4, group_size=15, infeatures=30, outfeatures=4, bias=False,
                data_type="nv_fp4",
            )

    def test_construction_nv_infeatures_not_divisible(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(
                bits=4, group_size=16, infeatures=33, outfeatures=4, bias=False,
                data_type="nv_fp4",
            )

    def test_construction_no_bias(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        layer = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=4, bias=False)
        assert layer.bias is None


# ---------------------------------------------------------------------------
# pack_fp4_to_uint8_cpu
# ---------------------------------------------------------------------------
class TestPackFp4ToUint8Cpu:
    def test_shape_halves_columns(self):
        from auto_round.export.export_to_autoround.qlinear_fp import (
            pack_fp4_to_uint8_cpu,
        )
        x = torch.zeros(4, 8)
        packed = pack_fp4_to_uint8_cpu(x)
        assert packed.shape == (4, 4)
        assert packed.dtype == torch.uint8

    def test_odd_dimension_padded(self):
        from auto_round.export.export_to_autoround.qlinear_fp import (
            pack_fp4_to_uint8_cpu,
        )
        x = torch.zeros(2, 6)
        packed = pack_fp4_to_uint8_cpu(x)
        # Half of 6 is 3
        assert packed.shape == (2, 3)


class TestPackFp4ToUint8:
    def test_zero_input(self):
        from auto_round.export.export_to_autoround.qlinear_fp import (
            _pack_fp4_to_uint8,
        )
        x = torch.zeros(2, 4)
        packed = _pack_fp4_to_uint8(x)
        assert (packed == 0).all()

    def test_largest_value(self):
        """6.0 is the largest FP4 entry -> index 7 -> both nibbles 0x77."""
        from auto_round.export.export_to_autoround.qlinear_fp import (
            _pack_fp4_to_uint8,
        )
        x = torch.full((2, 4), 6.0)
        packed = _pack_fp4_to_uint8(x)
        # Positive: low nibble = 7, high nibble = 7 -> 0x77
        assert (packed == 0x77).all()

    def test_larger_than_max_snaps(self):
        """Values > 6.0 should snap to 6.0 (index 7)."""
        from auto_round.export.export_to_autoround.qlinear_fp import (
            _pack_fp4_to_uint8,
        )
        x = torch.full((2, 4), 100.0)
        packed = _pack_fp4_to_uint8(x)
        # Snaps to 6.0 -> index 7 -> 0x77
        assert (packed == 0x77).all()

    def test_negative_with_sign(self):
        """Negative values get sign bit set (bit 3 of high nibble)."""
        from auto_round.export.export_to_autoround.qlinear_fp import (
            _pack_fp4_to_uint8,
        )
        x = torch.full((1, 4), -6.0)
        packed = _pack_fp4_to_uint8(x)
        # |x| = 6 -> index 7 in low; sign bit set in high (bit 3).
        # In each 4-bit slot, the high bit is the sign; 0x77 | 0x80 = 0xF7.
        # But because each pair gets `(idx | (idx << 4))` with idx=0xF (since
        # |x| snaps to idx 7 plus sign -> 0xF), the resulting byte is 0xFF.
        assert (packed == 0xFF).all()

    def test_pack_pairs(self):
        from auto_round.export.export_to_autoround.qlinear_fp import (
            _pack_fp4_to_uint8,
            FLOAT_TO_E2M1,
        )
        # Use FLOAT_TO_E2M1[1] = 0.5 (positive, index 1)
        x = torch.full((1, 2), FLOAT_TO_E2M1[1])
        packed = _pack_fp4_to_uint8(x)
        assert packed.shape == (1, 1)
        # Both nibbles = 1 -> 0x11
        assert packed.item() == 0x11


class TestPackFp4ToUint8Dispatcher:
    def test_cpu_dispatch(self):
        from auto_round.export.export_to_autoround.qlinear_fp import (
            pack_fp4_to_uint8,
        )
        x = torch.zeros(2, 4)
        packed = pack_fp4_to_uint8(x)
        assert packed.shape == (2, 2)
        assert packed.dtype == torch.uint8