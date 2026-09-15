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

import pytest
import torch

from auto_round.data_type.utils import (
    ceil_ste,
    float8_e4m3fn_ste,
    float8_e5m2_ste,
    floor_ste,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    round_ste,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestSteGpu:
    def test_round_floor_ceil_ste_grad_flow(self):
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        for fn in (round_ste, floor_ste, ceil_ste):
            y = fn(x).sum()
            y.backward()
            assert x.grad is not None
            assert torch.all(x.grad == 1.0)
            x.grad = None

    def test_float8_ste_on_cuda(self):
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        y = float8_e4m3fn_ste(x)
        assert y.device.type == "cuda"
        y.sum().backward()
        assert x.grad is not None

    def test_reshape_pad_revert_on_cuda(self):
        t = torch.arange(0, 30, dtype=torch.float32, device="cuda").reshape(3, 10)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=4)
        assert pad == 2
        assert out.device.type == "cuda"
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert reverted.shape == t.shape
        assert torch.equal(reverted, t)

    def test_float8_e5m2_ste_on_cuda(self):
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        y = float8_e5m2_ste(x)
        assert y.device.type == "cuda"
        y.sum().backward()
        assert x.grad is not None

    def test_reshape_pad_group_size_zero(self):
        # group_size == 0 -> flat reshape to a single row, no padding
        t = torch.randn(3, 4, 8, device="cuda")
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=0)
        assert pad == 0
        assert out.shape == (1, t.numel())
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert torch.equal(reverted, t)

    def test_reshape_pad_group_size_minus_one(self):
        # group_size == -1 -> returned unchanged (single row)
        t = torch.randn(3, 8, device="cuda")
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=-1)
        assert pad == 0
        assert out is t

    def test_reshape_pad_2d_group_size(self):
        # 2D group_size tuple pads both M and N dimensions
        t = torch.randn(5, 9, device="cuda")
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=(2, 4))
        assert pad == (1, 3)  # ceil(5/2)*2-5=1, ceil(9/4)*4-9=3
        assert out.device.type == "cuda"
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert reverted.shape == t.shape
        assert torch.equal(reverted, t)

    def test_reshape_pad_3d_input_revert(self):
        # 3D input is flattened to 2D before grouping, then restored
        t = torch.randn(2, 3, 10, device="cuda")
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=4)
        assert pad == 2
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert reverted.shape == t.shape
        assert torch.equal(reverted, t)

    def test_ste_through_linear_backward(self):
        # STE is differentiable: gradient flows through round_ste inside a Linear
        torch.manual_seed(0)
        linear = torch.nn.Linear(8, 4, device="cuda")
        x = torch.randn(2, 8, device="cuda", requires_grad=True)
        out = linear(round_ste(x))
        loss = (out.square() + out).sum()
        loss.backward()
        assert x.grad is not None
        assert linear.weight.grad is not None

    def test_round_ste_value_on_cuda(self):
        x = torch.tensor([1.2, 2.7, -0.5], device="cuda", requires_grad=True)
        y = round_ste(x)
        # STE output equals the rounded value but keeps gradient flow
        assert torch.equal(y, x.round())
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))
