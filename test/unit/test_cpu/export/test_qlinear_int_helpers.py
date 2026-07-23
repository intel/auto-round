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
"""Tests for the int4-packing helpers in
``auto_round/export/export_to_autoround/qlinear_int.py``.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
class TestModuleConstants:
    def test_float_to_e0m4_lookup(self):
        from auto_round.export.export_to_autoround.qlinear_int import FLOAT_TO_E0M4

        assert len(FLOAT_TO_E0M4) == 8
        # monotonically non-decreasing
        for i in range(1, len(FLOAT_TO_E0M4)):
            assert FLOAT_TO_E0M4[i] >= FLOAT_TO_E0M4[i - 1]

    def test_e8m0_constants(self):
        from auto_round.export.export_to_autoround.qlinear_int import (
            E8M0_EXPONENT_BIAS,
            E8M0_EXPONENT_NAN_VAL,
        )

        assert E8M0_EXPONENT_BIAS == 127
        assert E8M0_EXPONENT_NAN_VAL == 255
        assert E8M0_EXPONENT_NAN_VAL > E8M0_EXPONENT_BIAS


# ---------------------------------------------------------------------------
# QuantLinear.__init__
# ---------------------------------------------------------------------------
class TestQuantLinearInit:
    def test_construction_basic(self):
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        layer = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=4, bias=True)
        assert layer.infeatures == 32
        assert layer.outfeatures == 4
        assert layer.bits == 4
        assert layer.group_size == 32
        assert layer.QUANT_TYPE == "MXINT"
        assert layer.bias is not None

    def test_construction_no_bias(self):
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        layer = QuantLinear(bits=4, group_size=32, infeatures=64, outfeatures=8, bias=False)
        assert layer.bias is None

    def test_construction_group_size_neg1_rejected(self):
        """MXINT path requires group_size == 32 explicitly; -1 is not supported."""
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(bits=4, group_size=-1, infeatures=64, outfeatures=4, bias=False)

    def test_invalid_bits_raises(self):
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(bits=8, group_size=32, infeatures=32, outfeatures=4, bias=False)

    def test_invalid_group_size_raises(self):
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(bits=4, group_size=64, infeatures=32, outfeatures=4, bias=False)

    def test_infeatures_not_divisible_raises(self):
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        with pytest.raises(NotImplementedError):
            QuantLinear(bits=4, group_size=32, infeatures=33, outfeatures=4, bias=False)

    def test_weight_buffer_shape_bits_4(self):
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        layer = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=4, bias=False)
        # 4-bit packing means infeatures/2 cols
        assert layer.weight_packed.shape == (4, 16)
        assert layer.weight_packed.dtype == torch.uint8

    def test_weight_scale_buffer_shape(self):
        from auto_round.export.export_to_autoround.qlinear_int import QuantLinear

        layer = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=4, bias=False)
        # scale has (outfeatures, ceil(infeatures/group_size)) entries
        assert layer.weight_scale.shape == (4, 1)


# ---------------------------------------------------------------------------
# pack_int4_to_uint8_cpu
# ---------------------------------------------------------------------------
class TestPackInt4ToUint8Cpu:
    def test_shape_halves_columns(self):
        from auto_round.export.export_to_autoround.qlinear_int import (
            pack_int4_to_uint8_cpu,
        )

        x = torch.zeros(4, 8, dtype=torch.float32)  # 4 rows, 8 cols
        packed = pack_int4_to_uint8_cpu(x)
        assert packed.shape == (4, 4)
        assert packed.dtype == torch.uint8

    def test_output_dtype(self):
        from auto_round.export.export_to_autoround.qlinear_int import (
            pack_int4_to_uint8_cpu,
        )

        x = torch.zeros(2, 4)
        packed = pack_int4_to_uint8_cpu(x)
        assert packed.dtype == torch.uint8

    def test_odd_dimension_padded(self):
        from auto_round.export.export_to_autoround.qlinear_int import (
            pack_int4_to_uint8_cpu,
        )

        # Odd number of columns should be padded, then reshaped to (rows, ceil(cols/2))
        x = torch.zeros(2, 6)
        packed = pack_int4_to_uint8_cpu(x)
        # Half of 6 is 3
        assert packed.shape == (2, 3)


# ---------------------------------------------------------------------------
# _pack_int4_to_uint8 (the heavy lifter)
# ---------------------------------------------------------------------------
class TestPackInt4ToUint8:
    def test_zero_input(self):
        from auto_round.export.export_to_autoround.qlinear_int import (
            _pack_int4_to_uint8,
        )

        # All-zero values map to index 0 (the 0.0 entry in FLOAT_TO_E0M4)
        x = torch.zeros(2, 4)
        packed = _pack_int4_to_uint8(x)
        assert packed.shape == (2, 2)
        assert (packed == 0).all()

    def test_positive_values_highest_index(self):
        """Very large values should map to the largest entry (1.75, index 7)
        and yield packed bytes where the low nibble is 0b0111."""
        from auto_round.export.export_to_autoround.qlinear_int import (
            _pack_int4_to_uint8,
        )

        # Use a value > 1.75 -> should snap to 1.75 (index 7)
        x = torch.full((2, 4), 100.0)
        packed = _pack_int4_to_uint8(x)
        # Both nibbles should encode (idx=7, sign=0) => 0x77
        assert (packed == 0x77).all()

    def test_negative_values_with_sign_bit(self):
        """Negative values get the sign bit set (high nibble bit 3)."""
        from auto_round.export.export_to_autoround.qlinear_int import (
            _pack_int4_to_uint8,
        )

        x = torch.full((1, 4), -100.0)  # large negative -> absolute 100 snaps to 1.75 (idx 7)
        packed = _pack_int4_to_uint8(x)
        # negative -> sign bit in high nibble of each slot
        # Pair (idx, idx). low nibble = 7, high nibble = 7
        # sign bit is bit 3 of the high nibble -> 0x8 added
        # => 0x77 | 0x88 = 0xFF (because abs 100 > 1.75, both nibbles want sign+abs_max)
        assert packed.shape == (1, 2)
        # Each byte should be 0xFF (low=7, high=7+0x8 sign bit)
        assert (packed == 0xFF).all()

    def test_positive_max_value(self):
        """Positive large values: both nibbles should be 0x07 (idx 7 + sign 0)."""
        from auto_round.export.export_to_autoround.qlinear_int import (
            _pack_int4_to_uint8,
        )

        x = torch.full((1, 4), 100.0)
        packed = _pack_int4_to_uint8(x)
        assert (packed == 0x77).all()

    def test_4bit_values_packed_per_byte(self):
        """The packer packs two int4 values per uint8 (low + high nibble)."""
        from auto_round.export.export_to_autoround.qlinear_int import (
            FLOAT_TO_E0M4,
            _pack_int4_to_uint8,
        )

        # Use only the smallest positive value (0.25 -> index 1)
        x = torch.full((1, 2), FLOAT_TO_E0M4[1])
        packed = _pack_int4_to_uint8(x)
        # packed: (1, 1) containing one byte with both nibbles = 1
        assert packed.shape == (1, 1)
        assert packed.item() == 0x11  # low=1, high=1


# ---------------------------------------------------------------------------
# pack_int4_to_uint8 (dispatcher)
# ---------------------------------------------------------------------------
class TestPackInt4ToUint8Dispatcher:
    def test_cpu_dispatch(self):
        """On CPU, dispatcher should call the CPU path."""
        from auto_round.export.export_to_autoround.qlinear_int import (
            pack_int4_to_uint8,
        )

        x = torch.zeros(2, 4)
        packed = pack_int4_to_uint8(x)
        assert packed.shape == (2, 2)
        assert packed.dtype == torch.uint8
