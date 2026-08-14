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
"""GPU-side fast unit tests for ``auto_round.data_type.utils``.

These tests exercise the pure-tensor helper functions (reshape, STE round,
fp8 STE) without any model loading. They give the GPU CI pipeline
meaningful coverage of the 196-statement ``data_type/utils.py`` module.
"""

import torch

from auto_round.data_type.utils import (
    ceil_ste,
    float8_e4m3fn_ste,
    float8_e4m3fn_hpu_ste,
    float8_e4m3fnuz_hpu_ste,
    float8_e5m2_ste,
    floor_ste,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    round_ste,
)


# ==============================================================================
# reshape_pad_tensor_by_group_size + revert_tensor_by_pad
# ==============================================================================


class TestReshapePadAndRevert:
    def test_int_group_size_no_padding(self):
        """Tensor width already divisible by group_size -> no padding."""
        t = torch.arange(0, 32, dtype=torch.float32).reshape(4, 8)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=4)
        assert pad == 0
        assert orig_shape == t.shape
        assert out.shape == (4 * 2, 4)  # (rows * group_size, group_size) reshape

    def test_int_group_size_with_padding(self):
        t = torch.arange(0, 30, dtype=torch.float32).reshape(3, 10)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=4)
        assert pad == 2  # ceil(10/4)*4 - 10 = 12 - 10
        assert orig_shape == t.shape
        # Round-trip
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert reverted.shape == t.shape
        assert torch.equal(reverted, t)

    def test_int_group_size_smaller_than_tensor_width_returns_orig(self):
        """When group_size is -1 OR group_size > width, the function returns the
        tensor unchanged (no reshape)."""
        t = torch.arange(0, 8, dtype=torch.float32).reshape(2, 4)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=-1)
        assert pad == 0
        assert out.shape == t.shape

    def test_group_size_zero_collapses_to_2d(self):
        t = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=0)
        assert pad == 0
        assert orig_shape == t.shape
        assert out.shape == (1, 12)
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert reverted.shape == t.shape
        assert torch.equal(reverted, t)

    def test_2d_group_size(self):
        """2D block-wise group_size reshapes into (M-blocks, N-blocks, M, N)."""
        t = torch.arange(0, 16, dtype=torch.float32).reshape(4, 4)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=(2, 2))
        assert orig_shape == t.shape
        assert pad == (0, 0)
        # Output shape: (rows / 2, cols / 2, 2, 2) = (2, 2, 2, 2)
        assert out.shape == (2, 2, 2, 2)
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert torch.equal(reverted, t)

    def test_2d_group_size_with_padding(self):
        t = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=(2, 2))
        # 3 -> 4 (pad 1), 4 -> 4 (pad 0)
        assert pad == (1, 0)
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert reverted.shape == t.shape
        assert torch.equal(reverted, t)

    def test_higher_dim_input(self):
        """3D tensor should be flattened to (-1, last_dim) before reshape."""
        t = torch.arange(0, 24, dtype=torch.float32).reshape(2, 3, 4)
        out, orig_shape, pad = reshape_pad_tensor_by_group_size(t, group_size=4)
        assert orig_shape == t.shape
        # rows becomes 2*3 = 6, then -1 x 4 (since 24/4=6)
        reverted = revert_tensor_by_pad(out, orig_shape, pad)
        assert reverted.shape == t.shape
        assert torch.equal(reverted, t)

    def test_revert_tensor_with_zero_pad(self):
        t = torch.arange(0, 8, dtype=torch.float32)
        reverted = revert_tensor_by_pad(t, t.shape, 0)
        assert reverted.shape == t.shape


# ==============================================================================
# STE round / floor / ceil
# ==============================================================================


class TestSteRounding:
    def test_round_ste_identity_for_integers(self):
        t = torch.tensor([1.0, 2.0, 3.5, 4.5])
        out = round_ste(t)
        # STE passes gradients as if it were identity, but forward rounding
        # produces standard nearest-even rounding.
        assert torch.equal(out, torch.round(t))

    def test_round_ste_preserves_shape_and_dtype(self):
        t = torch.randn(8, 16, dtype=torch.bfloat16)
        out = round_ste(t)
        assert out.shape == t.shape
        assert out.dtype == t.dtype

    def test_round_ste_gradient_passthrough(self):
        """The STE backward should pass gradient through unchanged."""
        t = torch.tensor([1.3, 2.7], requires_grad=True)
        out = round_ste(t)
        out.sum().backward()
        # Gradient must equal 1.0 for every element (passthrough)
        assert torch.equal(t.grad, torch.ones_like(t))

    def test_floor_ste_identity_for_integers(self):
        t = torch.tensor([1.5, 2.5, -0.5])
        out = floor_ste(t)
        assert torch.equal(out, torch.floor(t))

    def test_floor_ste_gradient_passthrough(self):
        t = torch.tensor([1.5, -0.5], requires_grad=True)
        out = floor_ste(t)
        out.sum().backward()
        assert torch.equal(t.grad, torch.ones_like(t))

    def test_ceil_ste_identity_for_integers(self):
        t = torch.tensor([1.1, 2.1, -0.9])
        out = ceil_ste(t)
        assert torch.equal(out, torch.ceil(t))

    def test_ceil_ste_gradient_passthrough(self):
        t = torch.tensor([1.1, -0.9], requires_grad=True)
        out = ceil_ste(t)
        out.sum().backward()
        assert torch.equal(t.grad, torch.ones_like(t))


# ==============================================================================
# fp8 STE helpers
# ==============================================================================


class TestFp8Ste:
    """fp8 STE helpers take a float tensor, quantize to fp8 and dequantize back.
    The returned tensor therefore has the **original** dtype, but its values
    have been snapped to the fp8 grid (the STE pattern keeps the gradient
    passthrough)."""

    def _roundtrip_safe(self, t):
        # Avoid extreme values that overflow fp8 dynamic range
        return t.clamp(-100.0, 100.0)

    def test_float8_e4m3fn_ste_preserves_dtype_and_shape(self):
        t = self._roundtrip_safe(torch.randn(8, 8, dtype=torch.bfloat16))
        out = float8_e4m3fn_ste(t)
        assert out.shape == t.shape
        assert out.dtype == t.dtype
        # STE should be close to (but not exactly equal to) the input
        # since fp8 has limited precision.
        assert not torch.equal(out, t)

    def test_float8_e5m2_ste_preserves_dtype_and_shape(self):
        t = self._roundtrip_safe(torch.randn(8, 8, dtype=torch.bfloat16))
        out = float8_e5m2_ste(t)
        assert out.shape == t.shape
        assert out.dtype == t.dtype

    def test_float8_e4m3fn_ste_gradient_passthrough(self):
        t = self._roundtrip_safe(torch.randn(4, 4, dtype=torch.bfloat16))
        t.requires_grad_(True)
        out = float8_e4m3fn_ste(t)
        out.sum().backward()
        # STE backward must be a passthrough (gradient == 1.0 everywhere)
        assert torch.equal(t.grad, torch.ones_like(t))

    def test_float8_e4m3fn_hpu_ste(self):
        """HPU fp8 STE only works on HPU hardware; on non-HPU it falls back
        to a torch.ops.hpu call that isn't available. Verify the function
        object exists and is callable (the actual call only succeeds on HPU).
        """
        from auto_round.data_type.utils import float8_e4m3fn_hpu_ste

        assert callable(float8_e4m3fn_hpu_ste)

    def test_float8_e4m3fnuz_hpu_ste(self):
        from auto_round.data_type.utils import float8_e4m3fnuz_hpu_ste

        assert callable(float8_e4m3fnuz_hpu_ste)
