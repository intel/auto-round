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
"""Tests for ``auto_round/export/export_to_awq/utils.py``."""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# unpack_awq
# ---------------------------------------------------------------------------
class TestUnpackAwq:
    def test_unpack_4bit_basic(self):
        from auto_round.export.export_to_awq.utils import unpack_awq

        # 0x5B = 91 = 0b01011011.
        # Right-shifts by [0, 4, 8, 12, 16, 20, 24, 28] produce 8 successive
        # 4-bit slices, cast to int8.
        packed = torch.tensor([0x5B], dtype=torch.int32).view(1, 1, 1)
        zeros = torch.zeros((1, 1, 1), dtype=torch.int32)
        iw, iz = unpack_awq(packed, zeros, bits=4)
        assert iw.shape == (1, 8)
        # iw[0,0] = 0x5B (no shift, the whole int32, fits in int8: 91)
        assert iw[0, 0].item() == 0x5B
        # iw[0,1] = 0x5B >> 4 = 5
        assert iw[0, 1].item() == 0x5
        # iw[0,2] = 0x5B >> 8 = 0 (0x5B is only 8 bits)
        assert iw[0, 2].item() == 0

    def test_unpack_qzeros_none_raises(self):
        """If qzeros is None, the implementation requires a device for shifts."""
        from auto_round.export.export_to_awq.utils import unpack_awq

        packed = torch.zeros((2, 1, 1), dtype=torch.int32)
        with pytest.raises(AttributeError):
            unpack_awq(packed, None, bits=4)


# ---------------------------------------------------------------------------
# reverse_awq_order
# ---------------------------------------------------------------------------
class TestReverseAwqOrder:
    def test_reverse_identity(self):
        from auto_round.export.export_to_awq.utils import reverse_awq_order

        iw = torch.arange(8, dtype=torch.int32).view(1, 8)
        iz = torch.zeros(1, 8, dtype=torch.int32)
        out_iw, out_iz = reverse_awq_order(iw, iz, bits=4)
        # AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
        expected = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32).view(1, 8)
        assert torch.equal(out_iw, expected)


# ---------------------------------------------------------------------------
# dequantize_gemm
# ---------------------------------------------------------------------------
class TestDequantizeGemm:
    def test_zero_pack_zero_scale_returns_zeros(self):
        from auto_round.export.export_to_awq.utils import dequantize_gemm

        # Pack all zeros, no quantization -> output should be 0 (zero - 0) * 1 = 0
        in_f, out_f, group_size = 8, 8, 8
        qweight = torch.zeros((in_f, out_f // 8), dtype=torch.int32)
        qzeros = torch.zeros((in_f // group_size, out_f // 8), dtype=torch.int32)
        scales = torch.ones((in_f // group_size, out_f), dtype=torch.float16)
        out = dequantize_gemm(qweight, qzeros, scales, bits=4, group_size=group_size)
        assert out.shape == (in_f, out_f)
        assert torch.equal(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# WQLinear_GEMM
# ---------------------------------------------------------------------------
class TestWQLinearGEMM:
    def test_construction(self):
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        layer = WQLinear_GEMM(
            w_bit=4,
            group_size=4,
            in_features=8,
            out_features=8,
            bias=True,
            dev="cpu",
        )
        assert layer.w_bit == 4
        assert layer.in_features == 8
        assert layer.out_features == 8
        assert layer.qweight.shape == (8, 1)  # 8 // (32/4) = 8 / 8 = 1
        assert layer.bias is not None

    def test_construction_no_bias(self):
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        layer = WQLinear_GEMM(
            w_bit=4,
            group_size=4,
            in_features=8,
            out_features=8,
            bias=False,
            dev="cpu",
        )
        assert layer.bias is None

    def test_construction_neg1_group(self):
        """group_size=-1 should be replaced with in_features."""
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        layer = WQLinear_GEMM(
            w_bit=4,
            group_size=-1,
            in_features=8,
            out_features=8,
            bias=False,
            dev="cpu",
        )
        assert layer.group_size == 8

    def test_invalid_w_bit_raises(self):
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        with pytest.raises(NotImplementedError):
            WQLinear_GEMM(w_bit=8, group_size=4, in_features=8, out_features=8, bias=False, dev="cpu")

    def test_infeatures_not_divisible_raises(self):
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        with pytest.raises(ValueError):
            WQLinear_GEMM(w_bit=4, group_size=4, in_features=9, out_features=8, bias=False, dev="cpu")

    def test_outfeatures_not_aligned_raises(self):
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        # out_features must be divisible by (32 // w_bit) = 8
        with pytest.raises(ValueError):
            WQLinear_GEMM(w_bit=4, group_size=4, in_features=8, out_features=7, bias=False, dev="cpu")

    def test_from_linear_init_only(self):
        from torch.nn import Linear

        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        linear = Linear(8, 8, bias=True)
        layer = WQLinear_GEMM.from_linear(linear, w_bit=4, group_size=4, init_only=True)
        # In init_only, just creates the buffer shell
        assert isinstance(layer, WQLinear_GEMM)
        # The buffers remain zeros
        assert torch.equal(layer.qweight, torch.zeros_like(layer.qweight))

    def test_from_linear_requires_scales_and_zeros(self):
        from torch.nn import Linear

        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        linear = Linear(8, 8, bias=True)
        with pytest.raises(ValueError, match="scales"):
            WQLinear_GEMM.from_linear(linear, w_bit=4, group_size=4)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
class TestModuleConstants:
    def test_reverse_order_table(self):
        from auto_round.export.export_to_awq.utils import AWQ_REVERSE_ORDER

        assert len(AWQ_REVERSE_ORDER) == 8
        # Permutation of 0..7
        assert sorted(AWQ_REVERSE_ORDER) == list(range(8))


# ---------------------------------------------------------------------------
# WQLinearMMFunction forward
# ---------------------------------------------------------------------------
class TestWQLinearMMFunction:
    def test_forward_shape(self):
        from auto_round.export.export_to_awq.utils import WQLinearMMFunction

        in_f, out_f, group_size = 8, 8, 8
        qweight = torch.zeros((in_f, out_f // 8), dtype=torch.int32)
        qzeros = torch.zeros((in_f // group_size, out_f // 8), dtype=torch.int32)
        scales = torch.ones((in_f // group_size, out_f), dtype=torch.float16)
        bias = torch.zeros(out_f, dtype=torch.float16)
        x = torch.randn(2, in_f, dtype=torch.float16)
        out = WQLinearMMFunction.apply(
            x,
            qweight,
            qzeros,
            scales,
            4,
            group_size,
            bias,
            out_f,
        )
        # Output should be (1, 2, 8) because the function unsqueezes 2D tensors
        assert out.shape == (1, 2, 8)
