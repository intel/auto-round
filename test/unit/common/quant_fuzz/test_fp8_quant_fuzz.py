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
"""Property-based fuzzing of the scaled FP8 QDQ kernels in ``auto_round.data_type.fp8``."""

import math
from test.fuzz_helpers import (
    FP8_E4M3,
    FP8_E5M2,
    assert_monotone_within_groups,
    assert_on_scaled_grid,
    assert_shape_dtype_finite,
    assert_where,
    assert_zero_and_sign_preserved,
    dtype_slack,
    group_view,
    tensors,
    to_f64,
    weight_shapes,
)

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from auto_round.data_type.fp8 import (
    quant_block_fp_sym,
    quant_fp8_e5m2,
    quant_fp8_e5m2_unit_scale,
    quant_fp8_sym,
    quant_fp8_unit_scale,
)
from auto_round.utils import is_gaudi2

pytestmark = pytest.mark.skipif(is_gaudi2(), reason="Gaudi2 re-registers fp8_sym with float8_e4m3fnuz")

GROUP_SIZES = st.one_of(st.sampled_from([-1, 0, 16, 32, 128]), st.integers(1, 64))
BLOCKS = st.one_of(
    st.sampled_from([(128, 128), (1, 128), (16, 16), (8, 1)]), st.tuples(st.integers(1, 64), st.integers(1, 64))
)
PER_GROUP = {
    "e4m3": (quant_fp8_sym, FP8_E4M3, torch.float8_e4m3fn),
    "e5m2": (quant_fp8_e5m2, FP8_E5M2, torch.float8_e5m2),
}
UNIT_SCALE = {"e4m3": (quant_fp8_unit_scale, FP8_E4M3), "e5m2": (quant_fp8_e5m2_unit_scale, FP8_E5M2)}


def _effective_input(x):
    """The kernels route fp16 through bf16 before dividing by the scale (to avoid NaN gradients)."""
    return x.to(torch.bfloat16) if x.dtype == torch.float16 else x


def _coarser(*dtypes):
    return max(dtypes, key=lambda d: torch.finfo(d).eps)


def _check_fp8_values(xg, xe, qg, s, grid, compute, out_dtype):
    """Elements on ``scale * grid`` and within the grid's rounding/saturation error of the effective input."""
    tol_dtype = _coarser(compute, out_dtype)
    assert_on_scaled_grid(qg, s, grid, tol_dtype)
    eps_c = torch.finfo(compute).eps
    bound = s * grid.max_error(xe.abs() / s) + eps_c * xe.abs() + (xg - xe).abs() + dtype_slack(tol_dtype, qg)
    assert_where((xg - qg).abs() <= bound, "FP8 error exceeds the grid bound", x=xg, qdq=qg, scale=s, bound=bound)
    assert_zero_and_sign_preserved(xg, qg)
    assert_monotone_within_groups(xg, qg, out_dtype)


@given(x=tensors(), kind=st.sampled_from(sorted(PER_GROUP)), group_size=GROUP_SIZES)
def test_fp8_per_group_invariants(x, kind, group_size):
    func, grid, fp8_dtype = PER_GROUP[kind]
    qdq, scale, extra = func(x, group_size=group_size)
    assert extra is None
    assert_shape_dtype_finite(qdq, x)

    xg, qg = to_f64(group_view(x, group_size)), to_f64(group_view(qdq, group_size))
    xe = to_f64(group_view(_effective_input(x), group_size))
    assert scale.dtype == torch.float32 and scale.shape == (xg.shape[0], 1)
    s = to_f64(scale)
    fmax = torch.finfo(fp8_dtype).max
    expected = (xg.abs().amax(-1, keepdim=True) / fmax).clamp(min=1.0 / (fmax * 512.0))
    assert_where(
        (s - expected).abs() <= 2**-22 * expected, "scale != max(amax / fp8_max, floor)", scale=s, want=expected
    )
    _check_fp8_values(xg, xe, qg, s, grid, torch.float32, x.dtype)


@given(x=tensors(), kind=st.sampled_from(sorted(PER_GROUP)), group_size=GROUP_SIZES, signed_range=st.booleans())
def test_fp8_static_range_matches_dynamic(x, kind, group_size, signed_range):
    """Passing the calibrated per-group range (static path) must reproduce the dynamic per-token path."""
    func = PER_GROUP[kind][0]
    xg = group_view(x, group_size)
    if signed_range:
        kwargs = {"tensor_max": xg.max(-1)[0], "tensor_min": xg.min(-1)[0]}
    else:
        kwargs = {"tensor_max": xg.abs().amax(-1)}
    dynamic, dyn_scale, _ = func(x, group_size=group_size)
    static, static_scale, _ = func(x, group_size=group_size, **kwargs)
    assert torch.equal(static_scale, dyn_scale)
    assert torch.equal(static, dynamic)


def _block_view(t, block):
    m, n = block
    rows, cols = t.shape
    mb, nb = math.ceil(rows / m), math.ceil(cols / n)
    t = torch.nn.functional.pad(t, (0, nb * n - cols, 0, mb * m - rows))
    return t.reshape(mb, m, nb, n).permute(0, 2, 1, 3).reshape(mb, nb, m * n)


@given(x=tensors(shapes=weight_shapes(max_rows=96, max_cols=160)), block=BLOCKS)
def test_block_fp8_invariants(x, block):
    qdq, scale, extra = quant_block_fp_sym(x, group_size=block)
    assert extra is None
    assert_shape_dtype_finite(qdq, x)

    xg, qg = to_f64(_block_view(x, block)), to_f64(_block_view(qdq, block))
    xe = to_f64(_block_view(_effective_input(x), block))
    # the block scale stays in the input dtype
    assert scale.dtype == x.dtype and scale.shape == xg.shape[:2]
    s = to_f64(scale).unsqueeze(-1)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    expected = (xg.abs().amax(-1, keepdim=True) / fmax).clamp(min=1.0 / (fmax * 512.0))
    assert_where(
        (s - expected).abs() <= dtype_slack(x.dtype, expected), "block scale != max(amax / 448, floor)", scale=s
    )
    compute = torch.promote_types(_effective_input(x).dtype, x.dtype)
    _check_fp8_values(xg.flatten(0, 1), xe.flatten(0, 1), qg.flatten(0, 1), s.flatten(0, 1), FP8_E4M3, compute, x.dtype)


@given(x=tensors(), kind=st.sampled_from(sorted(UNIT_SCALE)))
def test_fp8_unit_scale_is_saturating_cast(x, kind):
    func, grid = UNIT_SCALE[kind]
    qdq, scale, _ = func(x)
    assert_shape_dtype_finite(qdq, x)
    assert torch.equal(scale.cpu(), torch.ones(1))
    xg, qg, xe = to_f64(x).reshape(1, -1), to_f64(qdq).reshape(1, -1), to_f64(_effective_input(x)).reshape(1, -1)
    _check_fp8_values(xg, xe, qg, torch.ones(1, 1, dtype=torch.float64), grid, _effective_input(x).dtype, x.dtype)
