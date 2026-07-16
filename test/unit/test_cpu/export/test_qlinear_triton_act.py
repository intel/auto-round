# Copyright (c) 2024 Intel Corporation
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

"""Tests for export_to_autoround/qlinear_triton_act.py."""

import pytest
import torch
import torch.nn as nn
import transformers

from auto_round.export.export_to_autoround.qlinear_triton_act import QuantLinear


class TestQuantLinearInit:
    """Tests for QuantLinear.__init__."""

    def test_init_4bit_standard(self):
        """Test init with 4-bit, group_size=128."""
        ql = QuantLinear(bits=4, group_size=128, infeatures=1024, outfeatures=256, bias=False)
        assert ql.bits == 4
        assert ql.group_size == 128
        assert ql.maxq == 15

    def test_init_2bit(self):
        """Test init with 2-bit."""
        ql = QuantLinear(bits=2, group_size=64, infeatures=512, outfeatures=128, bias=True)
        assert ql.bits == 2
        assert ql.maxq == 3
        assert ql.bias is not None

    def test_init_8bit(self):
        """Test init with 8-bit, group_size=-1 (whole matrix)."""
        ql = QuantLinear(bits=8, group_size=-1, infeatures=512, outfeatures=256, bias=False)
        assert ql.bits == 8
        assert ql.group_size == 512

    def test_init_not_implemented_bits(self):
        """Test that unsupported bits raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Only 2,4,8 bits"):
            QuantLinear(bits=3, group_size=64, infeatures=512, outfeatures=128, bias=False)

    def test_init_infeatures_not_divisible(self):
        """Test that infeatures not divisible by 32 raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="must be divisible by 32"):
            QuantLinear(bits=4, group_size=64, infeatures=511, outfeatures=256, bias=False)

    def test_init_outfeatures_not_divisible(self):
        """Test that outfeatures not divisible by 32 raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="must be divisible by 32"):
            QuantLinear(bits=4, group_size=64, infeatures=512, outfeatures=255, bias=False)

    def test_init_buffers_shapes(self):
        """Test that buffers have correct shapes."""
        ql = QuantLinear(bits=4, group_size=64, infeatures=512, outfeatures=256, bias=False)
        assert ql.qweight.shape == (64, 256)
        assert ql.scales.shape == (8, 256)
        assert ql.qzeros.shape == (8, 32)
        assert ql.act_scales.shape == (1,)
        assert ql.w_bf16_to_fp8_scale.shape == (1,)

    def test_init_use_pc_true(self):
        """Test init with use_pc=True sets w_bf16_to_fp8_scale shape to (1, outfeatures)."""
        ql = QuantLinear(
            bits=4, group_size=64, infeatures=512, outfeatures=256,
            bias=False, use_pc=True
        )
        assert ql.w_bf16_to_fp8_scale.shape == (1, 256)

    def test_repr(self):
        """Test __repr__ produces a String."""
        ql = QuantLinear(bits=4, group_size=64, infeatures=512, outfeatures=256, bias=False)
        r = repr(ql)
        assert "QuantLinear" in r
        assert "bits=4" in r


class TestQuantLinearPack:
    """Tests for QuantLinear.pack.

    Note: pack requires specific shapes for the repeat_interleave broadcasting.
    The safe configuration is group_size=infeatures (num_groups=1),
    with outfeatures=infeatures=256 so all shapes align.
    """

    def test_pack_with_bias(self):
        """Test pack copies bias from linear."""
        linear = nn.Linear(256, 256, bias=True)
        linear.weight.data = torch.randn(256, 256)
        linear.bias.data = torch.randn(256)

        scales = torch.ones(1, 256)
        zeros = torch.zeros(1, 256)
        act_scales = torch.ones(1)
        w_bf16 = torch.ones(1)

        ql = QuantLinear(bits=4, group_size=256, infeatures=256, outfeatures=256, bias=True)
        ql.pack(linear, scales, zeros, act_scales, w_bf16)

        assert ql.bias is not None
        assert ql.bias.shape == (256,)

    def test_pack_without_bias(self):
        """Test pack handles linear without bias."""
        linear = nn.Linear(256, 256, bias=False)
        linear.weight.data = torch.randn(256, 256)

        scales = torch.ones(1, 256)
        zeros = torch.zeros(1, 256)
        act_scales = torch.ones(1)
        w_bf16 = torch.ones(1)

        ql = QuantLinear(bits=4, group_size=256, infeatures=256, outfeatures=256, bias=False)
        ql.pack(linear, scales, zeros, act_scales, w_bf16)
        assert ql.bias is None

    def test_pack_conv1d(self):
        """Test pack flattens Conv1D weights before quantization."""
        linear = transformers.pytorch_utils.Conv1D(256, 256)
        linear.weight.data = torch.randn(256, 256)
        linear.bias = None

        scales = torch.ones(1, 256)
        zeros = torch.zeros(1, 256)
        act_scales = torch.ones(1)
        w_bf16 = torch.ones(1)

        ql = QuantLinear(bits=4, group_size=256, infeatures=256, outfeatures=256, bias=False)
        ql.pack(linear, scales, zeros, act_scales, w_bf16)

        assert ql.qweight.shape[1] == 256

    def test_pack_sets_act_scales(self):
        """Test pack copies act_scales and w_bf16_to_fp8_scale buffers."""
        linear = nn.Linear(256, 256, bias=False)
        linear.weight.data = torch.randn(256, 256)

        scales = torch.ones(1, 256)
        zeros = torch.zeros(1, 256)
        act_scales = torch.tensor([2.5])
        w_bf16 = torch.tensor([1.5])

        ql = QuantLinear(bits=4, group_size=256, infeatures=256, outfeatures=256, bias=False)
        ql.pack(linear, scales, zeros, act_scales, w_bf16)

        assert ql.act_scales.item() == 2.5
        assert ql.w_bf16_to_fp8_scale.item() == 1.5


class TestQuantLinearWarmup:
    """Tests for QuantLinear.warmup."""

    def test_warmup_does_nothing(self):
        """Test warmup is a no-op (returns None)."""
        result = QuantLinear.warmup(None)
        assert result is None
