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
"""Numerical tests for int_sym / int_sym_gptq and mx_uint / mx_fp4_rceil_v2 quantizers."""

import pytest
import torch

from auto_round.data_type.int import quant_tensor_sym, quant_tensor_sym_gptq
from auto_round.data_type.mxfp import quant_mx_rceil, quant_mx_rceil_v2, quant_mx_uint
from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(4, 64)


def _assert_on_grid(qdq, scale, lo, hi):
    q = qdq.float() / scale.float()
    assert torch.allclose(q, q.round(), atol=1e-3)
    assert q.min() >= lo - 1e-3 and q.max() <= hi + 1e-3


class TestIntSym:
    def test_registered(self):
        assert QUANT_FUNC_WITH_DTYPE["int_sym"] is quant_tensor_sym
        assert QUANT_FUNC_WITH_DTYPE["int_sym_gptq"] is quant_tensor_sym_gptq

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_extreme_value_is_exact_and_grid(self, weight, bits):
        qdq, scale, maxq = quant_tensor_sym(weight, bits=bits, group_size=32, scale_dtype=torch.float32)
        assert maxq == 2 ** (bits - 1) and qdq.shape == weight.shape
        _assert_on_grid(qdq.reshape(-1, 32), scale, -maxq, maxq)
        groups, qgroups = weight.reshape(-1, 32), qdq.reshape(-1, 32)
        idx = groups.abs().argmax(-1, keepdim=True)
        # The sign of the scale follows the larger-magnitude side so that extreme maps to -maxq exactly.
        assert torch.allclose(groups.gather(-1, idx), qgroups.gather(-1, idx), atol=1e-6)
        assert torch.all((scale.squeeze(-1) > 0) == (groups.gather(-1, idx).squeeze(-1) < 0))

    def test_padding_and_explicit_min_max(self):
        x = torch.linspace(-1, 2, 40).reshape(1, 40)
        qdq, scale, _ = quant_tensor_sym(x, bits=4, group_size=32)
        assert qdq.shape == x.shape and scale.shape == (2, 1)
        tmin, tmax = torch.full((2,), -4.0), torch.full((2,), 4.0)
        _, wide_scale, _ = quant_tensor_sym(x, bits=4, group_size=32, tensor_min=tmin, tensor_max=tmax)
        assert torch.allclose(wide_scale.abs().float(), torch.full((2, 1), 0.5))

    def test_init_scale_with_scalar_and_tensor_max_scale(self):
        x = torch.tensor([[0.3, -0.7, 1.2, 0.0]])
        init = torch.tensor([[0.25]])
        qdq, scale, _ = quant_tensor_sym(x, bits=4, group_size=4, init_scale=init, scale_dtype=torch.float32)
        assert torch.allclose(qdq, torch.tensor([[0.25, -0.75, 1.25, 0.0]]))
        qdq2, scale2, _ = quant_tensor_sym(
            x, bits=4, group_size=4, init_scale=init, max_scale=torch.tensor([2.0]), scale_dtype=torch.float32
        )
        assert torch.allclose(scale2, torch.tensor([[0.5]]))
        assert torch.allclose(qdq2, torch.tensor([[0.5, -0.5, 1.0, 0.0]]))

    def test_init_scale_threshold_keeps_sign(self):
        x = torch.tensor([[1.0, -1.0]])
        _, scale, _ = quant_tensor_sym(x, group_size=2, init_scale=torch.tensor([[-1e-9]]), scale_dtype=torch.float32)
        assert torch.allclose(scale, torch.tensor([[-1e-5]]))


class TestIntSymGptq:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_grid_and_zero_point(self, weight, bits):
        qdq, scale, zp = quant_tensor_sym_gptq(weight, bits=bits, group_size=32, scale_dtype=torch.float32)
        maxq = 2**bits - 1
        assert torch.all(zp == (maxq + 1) / 2) and scale.shape == (8, 1)
        expected_scale = 2 * weight.reshape(-1, 32).abs().amax(-1, keepdim=True) / maxq
        assert torch.allclose(scale, expected_scale)
        _assert_on_grid(qdq.reshape(-1, 32), scale, -(maxq + 1) / 2, (maxq - 1) / 2)
        assert (qdq - weight).abs().reshape(-1, 32).max(-1).values.le(scale.squeeze(-1) * 1.5 + 1e-6).all()

    def test_tensor_min_max_scales_and_bounds(self):
        x = torch.tensor([[0.5, -1.0, 0.25, 0.75]])
        min_scale, max_scale = torch.tensor([0.5]), torch.tensor([0.5])
        _, scale, _ = quant_tensor_sym_gptq(
            x, bits=4, group_size=4, min_scale=min_scale, max_scale=max_scale, scale_dtype=torch.float32
        )
        assert torch.allclose(scale, torch.tensor([[1.0 / 15]]))
        _, scale, _ = quant_tensor_sym_gptq(
            x,
            bits=4,
            group_size=4,
            tensor_min=torch.tensor([-3.0]),
            tensor_max=torch.tensor([1.5]),
            scale_dtype=torch.float32,
        )
        assert torch.allclose(scale, torch.tensor([[6.0 / 15]]))

    def test_all_zero_group_uses_threshold(self):
        _, scale, _ = quant_tensor_sym_gptq(torch.zeros(1, 8), group_size=8, q_scale_thresh=1e-4)
        assert torch.all(scale.float() >= 1e-4 * 0.99)


class TestMxUint:
    def test_exact_codes(self):
        x = torch.tensor([[1.0, -0.5, 0.25, 0.0, 1.9, -2.0, 0.125, 1.75]])
        qdq, shared_exp, zp = quant_mx_uint(x, bits=4, group_size=4)
        assert zp == 8
        assert torch.equal(shared_exp, torch.tensor([[0.0], [1.0]]))
        # group 2: scale 2, step 0.5 -> 0.125 rounds to 0 and 1.75 (half) rounds to even 2.0
        assert torch.equal(qdq, torch.tensor([[1.0, -0.5, 0.25, 0.0, 2.0, -2.0, 0.0, 2.0]]))

    def test_positive_saturation_and_zero_group(self):
        x = torch.tensor([[1.99, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        qdq, shared_exp, _ = quant_mx_uint(x, bits=4, group_size=2)
        assert torch.equal(shared_exp.flatten(), torch.tensor([3.0, 0.0, 0.0, 0.0]))
        assert torch.equal(qdq, torch.tensor([[2.0, 8.0] + [0.0] * 6]))
        qdq, shared_exp, _ = quant_mx_uint(torch.tensor([[1.99, -1.0]]), bits=4, group_size=2)
        # 1.99 * 4 rounds to code 16, which saturates at 15 -> 7 / 4
        assert shared_exp.item() == 0.0
        assert torch.equal(qdq, torch.tensor([[1.75, -1.0]]))

    def test_max_scale_tensor_and_dtype(self):
        x = torch.tensor([[1.0, 0.5, -0.5, 0.25]], dtype=torch.bfloat16)
        qdq, shared_exp, _ = quant_mx_uint(x, bits=8, group_size=4, max_scale=torch.tensor([4.0]), init_scale=None)
        assert qdq.dtype == torch.bfloat16 and shared_exp.item() == 2.0
        assert torch.equal(qdq, x)

    @pytest.mark.parametrize("rounding", ["nearest", "floor", "stochastic"])
    def test_rounding_modes(self, weight, rounding):
        qdq, shared_exp, zp = quant_mx_uint(weight, bits=4, group_size=32, mantissa_rounding=rounding)
        scale = 2.0 ** shared_exp.float() / 4
        _assert_on_grid(qdq.reshape(-1, 32), scale, -zp, zp - 1)
        if rounding == "floor":
            assert torch.all(qdq.abs() <= weight.abs() + 1e-6)

    def test_invalid_args(self, weight):
        with pytest.raises(ValueError, match="bits must be positive"):
            quant_mx_uint(weight, bits=0)
        with pytest.raises(ValueError, match="mantissa_rounding"):
            quant_mx_uint(weight, group_size=32, mantissa_rounding="up")


class TestMxRceilV2:
    def test_registered(self):
        assert QUANT_FUNC_WITH_DTYPE["mx_fp4_rceil_v2"] is quant_mx_rceil_v2

    def test_uses_wider_threshold_than_rceil(self):
        x = torch.tensor([[7.0, 1.0, -2.0, 0.5]])
        qdq_v1, exp_v1, _ = quant_mx_rceil(x, bits=4, group_size=4, data_type="mx_fp4")
        qdq_v2, exp_v2, none = quant_mx_rceil_v2(x, bits=4, group_size=4, data_type="mx_fp4")
        assert none is None
        assert exp_v1.item() == 1.0 and exp_v2.item() == 0.0
        assert torch.equal(qdq_v2, torch.tensor([[6.0, 1.0, -2.0, 0.5]]))
        assert (qdq_v2 - x).abs().sum() < (qdq_v1 - x).abs().sum()

    def test_grid_padding_and_dtype(self, weight):
        x = weight[:, :40].to(torch.bfloat16)
        qdq, shared_exp, _ = quant_mx_rceil_v2(x, bits=4, group_size=32, data_type="unknown")
        assert qdq.shape == x.shape and qdq.dtype == torch.bfloat16
        padded = torch.nn.functional.pad(qdq.float(), (0, 24)).reshape(-1, 32)
        normalized = (padded / 2.0 ** shared_exp.float()).abs()
        assert torch.isin(normalized, FP4_GRID).all()

    def test_zero_group_and_tensor_max_scale(self):
        x = torch.zeros(1, 32)
        qdq, shared_exp, _ = quant_mx_rceil_v2(x, bits=4, group_size=32)
        assert torch.equal(qdq, x) and shared_exp.item() == 1.0
        y = torch.full((1, 32), 3.0)
        _, exp_scaled, _ = quant_mx_rceil_v2(y, bits=4, group_size=32, max_scale=torch.tensor([4.0]))
        assert exp_scaled.item() == 1.0
