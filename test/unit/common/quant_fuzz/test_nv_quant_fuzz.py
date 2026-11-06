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
"""Property-based fuzzing of the two-level (global + per-group) NVFP4 kernels in ``auto_round.data_type.nvfp``."""

from test.fuzz_helpers import (
    FP4_E2M1,
    FP8_E4M3,
    FUZZ_DEVICE,
    UE5M3,
    assert_monotone_within_groups,
    assert_on_scaled_grid,
    assert_shape_dtype_finite,
    assert_where,
    assert_zero_and_sign_preserved,
    dtype_slack,
    group_view,
    scaled_settings,
    tensors,
    to_f64,
)

import torch
from hypothesis import given
from hypothesis import strategies as st

from auto_round.data_type.nvfp import (
    cast_to_fp4,
    cast_to_ue5m3,
    e5m3_to_float_tensor,
    float_to_e5m3_frexp,
    nv_fp4,
    nv_fp4_rtn,
    nv_fp4_with_static_gs,
    nvfp4_v2,
    nvfp4_v2_gs,
    nvfp4_v2_with_global_scale,
    nvfp4_v2_with_static_gs,
    opt_rtn_fast_nvfp4,
    opt_rtn_nvfp4_v2_gs,
    rtn_nv_fp4_with_static_gs,
)

E4M3_GLOBAL = 448.0 * 6.0
UE5M3_GLOBAL = 114688.0 * 6.0
# reciprocal-of-reciprocal dequant scales and fp32 products drift by a few fp32 ulps
FP32_DRIFT = 2.0**-20
# the opt-RTN sweeps multiply the group max by 0.50 .. 1.51
SEARCH_COEFF = (0.5, 1.51)


def _amax(x):
    return x.float().abs().max()


def _global_scale(x, numerator):
    amax = _amax(x)
    return 0.0 if amax == 0 else to_f64(numerator / amax).item()


def _check_two_level(x, qdq, scale, global_scale, group_size, scale_grid, coeff=(1.0, 1.0)):
    """Per-group scale = ``scale_grid``-rounded ``c * gs * vmax / 6`` with ``c`` in ``coeff``;
    elements on the FP4 grid of ``scale / gs``."""
    assert_shape_dtype_finite(qdq, x)
    xg, qg = to_f64(group_view(x, group_size)), to_f64(group_view(qdq, group_size))
    s = to_f64(scale).reshape(-1, 1)
    assert s.shape[0] == xg.shape[0]
    assert_where(scale_grid.on_grid(s, rtol=0), "group scale is not representable in its scale format", scale=s)
    if global_scale == 0:
        assert_where(qg == 0, "an all-zero tensor must stay zero", qdq=qg)
        return

    ideal = global_scale * xg.abs().amax(-1, keepdim=True) / 6
    lo, hi = ((ideal * c).clamp(max=scale_grid.max_norm) for c in coeff)
    assert_where(
        (s >= lo - scale_grid.max_error(lo) - FP32_DRIFT * lo) & (s <= hi + scale_grid.max_error(hi) + FP32_DRIFT * hi),
        "group scale is not the rounded ideal scale",
        scale=s,
        lo=lo,
        hi=hi,
    )

    dq = s / global_scale
    dead = dq == 0
    assert_where(~dead | (qg == 0), "a group whose scale underflowed must dequantize to zero", qdq=qg, scale=s)
    safe = torch.where(dead, torch.ones_like(dq), dq)
    assert_on_scaled_grid(qg, safe, FP4_E2M1, x.dtype, rtol=FP32_DRIFT)
    y = xg.abs() / safe
    bound = dq * FP4_E2M1.max_error(y) + FP32_DRIFT * (xg.abs() + qg.abs()) + dtype_slack(x.dtype, qg)
    assert_where(dead | ((xg - qg).abs() <= bound), "NVFP4 error exceeds the FP4 grid bound", x=xg, qdq=qg, dq=dq)
    assert_zero_and_sign_preserved(xg, qg)
    assert_monotone_within_groups(xg, qg, x.dtype)


@given(values=st.lists(st.floats(-8, 8, width=32), min_size=1, max_size=256))
def test_cast_to_fp4_is_round_nearest_even_on_e2m1(values):
    x = torch.tensor(values, dtype=torch.float32, device=FUZZ_DEVICE)
    assert torch.equal(to_f64(cast_to_fp4(x)), FP4_E2M1.snap(to_f64(x)))


@given(values=st.lists(st.floats(-1e3, 2.5e5, width=32), min_size=1, max_size=256))
def test_ue5m3_cast_is_round_nearest_even_and_unsigned(values):
    x = torch.tensor(values, dtype=torch.float32, device=FUZZ_DEVICE)
    assert torch.equal(to_f64(cast_to_ue5m3(x)), UE5M3.snap(to_f64(x).clamp(min=0)))


def test_ue5m3_codebook_roundtrip():
    codes = torch.arange(255, dtype=torch.uint8, device=FUZZ_DEVICE)
    values = e5m3_to_float_tensor(codes)
    assert torch.equal(float_to_e5m3_frexp(values), codes)
    assert bool((values[1:] > values[:-1]).all()), "UE5M3 decode must be strictly increasing"
    assert values[-1].item() == UE5M3.max_norm
    nan_code = torch.tensor([0xFF], dtype=torch.uint8, device=FUZZ_DEVICE)
    assert torch.isnan(e5m3_to_float_tensor(nan_code)).all()


@given(x=tensors(), group_size=st.sampled_from([16, 32]))
def test_nv_fp4_two_level_invariants(x, group_size):
    qdq, scale, extra = nv_fp4(x, group_size=group_size)
    assert extra is None and scale.dtype == torch.float32
    _check_two_level(x, qdq, scale, _global_scale(x, E4M3_GLOBAL), group_size, FP8_E4M3)


@given(x=tensors(), group_size=st.sampled_from([16, 32]))
def test_nv_fp4_entry_points_agree(x, group_size):
    """In-place, static-global-scale and RTN variants are refactors of ``nv_fp4`` and must match it bit-for-bit."""
    ref, ref_scale, _ = nv_fp4(x, group_size=group_size)
    for name, (qdq, scale, _) in {
        "nv_fp4_rtn": nv_fp4_rtn(x, group_size=group_size),
        "nv_fp4_with_static_gs": nv_fp4_with_static_gs(x, group_size=group_size),
        "nv_fp4_with_static_gs(tensor_max)": nv_fp4_with_static_gs(x, group_size=group_size, tensor_max=_amax(x)),
        "rtn_nv_fp4_with_static_gs": rtn_nv_fp4_with_static_gs(x, group_size=group_size),
    }.items():
        assert torch.equal(qdq, ref), f"{name} qdq differs from nv_fp4"
        assert torch.equal(scale, ref_scale), f"{name} scale differs from nv_fp4"


@given(x=tensors(), group_size=st.sampled_from([16, 32]))
def test_nvfp4_v2_ue5m3_invariants(x, group_size):
    qdq, scale, _ = nvfp4_v2(x, group_size=group_size)
    _check_two_level(x, qdq, scale, 1.0, group_size, UE5M3)
    qdq, scale, _ = nvfp4_v2_with_global_scale(x, group_size=group_size)
    _check_two_level(x, qdq, scale, _global_scale(x, UE5M3_GLOBAL), group_size, UE5M3)


@given(x=tensors())
def test_nvfp4_v2_global_scale_entry_points_agree(x):
    ref, ref_scale, _ = nvfp4_v2_with_global_scale(x, group_size=16)
    for name, (qdq, scale, _) in {
        "nvfp4_v2_gs": nvfp4_v2_gs(x, group_size=16),
        "nvfp4_v2_with_static_gs": nvfp4_v2_with_static_gs(x, group_size=16),
        "nvfp4_v2_with_static_gs(tensor_max)": nvfp4_v2_with_static_gs(x, group_size=16, tensor_max=_amax(x)),
    }.items():
        assert torch.equal(qdq, ref), f"{name} qdq differs from nvfp4_v2_with_global_scale"
        assert torch.equal(scale, ref_scale), f"{name} scale differs from nvfp4_v2_with_global_scale"


def _assert_not_worse(x, opt, ref):
    xg = to_f64(group_view(x, 16))
    opt_sse = ((xg - to_f64(group_view(opt, 16))) ** 2).sum(-1)
    ref_sse = ((xg - to_f64(group_view(ref, 16))) ** 2).sum(-1)
    slack = ((FP32_DRIFT * xg) ** 2).sum(-1)
    assert_where(
        opt_sse <= ref_sse * (1 + 1e-6) + slack, "opt-RTN ended worse than plain RTN", opt=opt_sse, ref=ref_sse
    )


@scaled_settings(0.3)
@given(x=tensors(dtypes=(torch.float32,)))
def test_opt_rtn_nv_fp4_never_worse_than_rtn(x):
    """The 0.50..1.51 scale sweep includes the RTN scale, so per group it can only lower the SSE."""
    qdq, scale, _ = opt_rtn_fast_nvfp4(x.clone(), group_size=16)
    _check_two_level(x, qdq, scale, _global_scale(x, E4M3_GLOBAL), 16, FP8_E4M3, coeff=SEARCH_COEFF)
    _assert_not_worse(x, qdq, nv_fp4(x, group_size=16)[0])


@scaled_settings(0.3)
@given(x=tensors(dtypes=(torch.float32,)))
def test_opt_rtn_nvfp4_v2_gs_never_worse_than_rtn(x):
    qdq, scale, _ = opt_rtn_nvfp4_v2_gs(x.clone(), group_size=16)
    _check_two_level(x, qdq, scale, _global_scale(x, UE5M3_GLOBAL), 16, UE5M3, coeff=SEARCH_COEFF)
    _assert_not_worse(x, qdq, nvfp4_v2_gs(x, group_size=16)[0])
