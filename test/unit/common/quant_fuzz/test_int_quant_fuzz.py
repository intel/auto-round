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
"""Property-based fuzzing of the integer QDQ kernels in ``auto_round.data_type.int``."""

from test.fuzz_helpers import (
    FUZZ_DEVICE,
    assert_monotone_within_groups,
    assert_shape_dtype_finite,
    assert_where,
    assert_zero_and_sign_preserved,
    dtype_slack,
    group_view,
    scaled_settings,
    tensors,
    to_f64,
)

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from auto_round.data_type.int import (
    quant_tensor_asym,
    quant_tensor_opt_rtn_sym,
    quant_tensor_rtn_sym,
    quant_tensor_sym,
)

GROUP_SIZES = st.one_of(st.sampled_from([-1, 0, 16, 32, 64, 128]), st.integers(1, 64))
SCALE_DTYPES = st.sampled_from([torch.float16, torch.float32])
Q_SCALE_THRESH = 1e-5


def _grouped(t, group_size):
    return to_f64(group_view(t, group_size))


def _error_bound(x, qg, scale, maxq, scale_dtype, base):
    """``base`` code steps of rounding/clipping error plus fp error of the stored scale and of x / scale.

    The range -> scale division runs in ``x.dtype`` before the cast to ``scale_dtype``, so a scale that lands in the
    subnormal range of *either* dtype loses relative precision, which shrinks the covered range by up to
    ``maxq * subnormal_step`` and lets the group extremes clip.
    """
    compute = torch.promote_types(x.dtype, scale_dtype)
    rel = torch.finfo(scale_dtype).eps + torch.finfo(x.dtype).eps + torch.finfo(compute).eps
    subnormal_step = max(torch.finfo(d).smallest_normal * torch.finfo(d).eps for d in (x.dtype, scale_dtype))
    return scale.abs() * (base + maxq * rel) + maxq * subnormal_step + dtype_slack(x.dtype, qg)


@given(x=tensors(), bits=st.sampled_from([2, 3, 4, 5, 6, 8]), group_size=GROUP_SIZES, scale_dtype=SCALE_DTYPES)
def test_int_sym_invariants(x, bits, group_size, scale_dtype):
    qdq, scale, maxq = quant_tensor_sym(x, bits=bits, group_size=group_size, scale_dtype=scale_dtype)
    assert_shape_dtype_finite(qdq, x)
    assert maxq == 2 ** (bits - 1)

    xg, qg = _grouped(x, group_size), _grouped(qdq, group_size)
    assert scale.shape == (xg.shape[0], 1) and scale.dtype == scale_dtype
    s = to_f64(scale)
    assert_where(s.abs() >= Q_SCALE_THRESH * (1 - 2**-10), "|scale| fell below q_scale_thresh", scale=s)

    codes = (qg / s).round().clamp(-maxq, maxq - 1)
    assert_where(
        (qg - s * codes).abs() <= dtype_slack(x.dtype, s * codes),
        "qdq is not scale * code with code in [-maxq, maxq - 1]",
        qdq=qg,
        scale=s,
    )
    err = (xg - qg).abs()
    bound = _error_bound(x, qg, s, maxq, scale_dtype, base=1.0)
    assert_where(err <= bound, "int_sym error exceeds one quantization step", x=xg, qdq=qg, scale=s, bound=bound)
    assert_zero_and_sign_preserved(xg, qg)
    assert_monotone_within_groups(xg, qg, x.dtype)

    if x.dtype != torch.bfloat16 or bits <= 4:
        again, scale2, _ = quant_tensor_sym(
            qdq, bits=bits, group_size=group_size, init_scale=scale, scale_dtype=scale_dtype
        )
        assert torch.equal(scale2, scale)
        assert torch.equal(again, qdq), "re-quantizing a QDQ output with its own scale must be a fixed point"


@given(x=tensors(), bits=st.sampled_from([2, 3, 4, 8]), group_size=GROUP_SIZES, scale_dtype=SCALE_DTYPES)
def test_int_asym_invariants(x, bits, group_size, scale_dtype):
    qdq, scale, zp = quant_tensor_asym(x, bits=bits, group_size=group_size, scale_dtype=scale_dtype)
    assert_shape_dtype_finite(qdq, x)
    maxq = 2**bits - 1

    xg, qg = _grouped(x, group_size), _grouped(qdq, group_size)
    assert scale.shape == (xg.shape[0], 1) and zp.shape == scale.shape
    s, z = to_f64(scale), to_f64(zp)
    assert_where(s >= Q_SCALE_THRESH * (1 - 2**-10), "scale fell below q_scale_thresh", scale=s)
    # Groups hit by the known zp == maxq + 1 overflow (pinned by test_int_asym_zero_point_fits_bits_with_subnormal_scale)
    # are excluded from the per-group checks below; every other group is still checked in full.
    known = ((s < torch.finfo(scale_dtype).smallest_normal) & (z == maxq + 1)).reshape(-1)
    xg, qg, s, z = xg[~known], qg[~known], s[~known], z[~known]
    assert_where((z == z.round()) & (z >= 0) & (z <= maxq), "zero point must be an integer in [0, maxq]", zp=z)

    codes = (qg / s + z).round().clamp(0, maxq)
    assert_where(
        (qg - s * (codes - z)).abs() <= dtype_slack(x.dtype, s * (codes - z)),
        "qdq is not scale * (code - zp) with code in [0, maxq]",
        qdq=qg,
        scale=s,
        zp=z,
    )
    err = (xg - qg).abs()
    bound = _error_bound(x, qg, s, maxq, scale_dtype, base=0.5)
    assert_where(err <= bound, "int_asym error exceeds half a step", x=xg, qdq=qg, scale=s, zp=z, bound=bound)
    assert_zero_and_sign_preserved(xg, qg)
    assert_monotone_within_groups(xg, qg, x.dtype)

    if x.dtype != torch.bfloat16 or bits <= 4:
        again, _, _ = quant_tensor_asym(
            qdq, bits=bits, group_size=group_size, init_scale=scale, init_zp=zp, scale_dtype=scale_dtype
        )
        assert torch.equal(again, qdq), "re-quantizing a QDQ output with its own scale/zp must be a fixed point"


@pytest.mark.xfail(
    strict=True,
    reason="fuzz-found: quant_tensor_asym casts the scale to scale_dtype before zp = round(-wmin / scale) and never "
    "clamps zp; an fp16 scale near q_scale_thresh is subnormal and can round down by ~0.3 %, so a non-positive 8-bit "
    "group with range < ~0.0156 gets zp = 256: zero then dequantizes to -scale, and the plain auto_round zero-point "
    "packing cannot hold the value",
)
def test_int_asym_zero_point_fits_bits_with_subnormal_scale():
    neg = -torch.linspace(0.00256, 0.004, 64, device=FUZZ_DEVICE)
    x = torch.stack([neg, torch.zeros_like(neg)], dim=-1)  # one [-a, 0] group per row
    qdq, scale, zp = quant_tensor_asym(x, bits=8, group_size=2, scale_dtype=torch.float16)
    assert (scale < torch.finfo(torch.float16).smallest_normal).all()
    assert int(zp.max()) <= 255
    assert (qdq[:, 1] == 0).all()


@given(x=tensors(), bits=st.sampled_from([2, 4, 8]), group_size=GROUP_SIZES, sym=st.booleans())
def test_int_static_range_matches_dynamic(x, bits, group_size, sym):
    """Passing the per-group range explicitly (static/calibrated path) must equal the dynamic path."""
    xg = group_view(x, group_size)
    tensor_min = xg.min(-1)[0].clamp(max=0)
    tensor_max = xg.max(-1)[0].clamp(min=0)
    func = quant_tensor_sym if sym else quant_tensor_asym
    dynamic = func(x, bits=bits, group_size=group_size)[0]
    static = func(x, bits=bits, group_size=group_size, tensor_min=tensor_min, tensor_max=tensor_max)[0]
    assert torch.equal(static, dynamic)


@given(x=tensors(), bits=st.sampled_from([2, 3, 4, 8]), group_size=GROUP_SIZES)
def test_rtn_int_sym_equals_int_sym(x, bits, group_size):
    rtn, rtn_scale, rtn_maxq = quant_tensor_rtn_sym(x.clone(), bits=bits, group_size=group_size)
    ref, ref_scale, ref_maxq = quant_tensor_sym(x, bits=bits, group_size=group_size)
    assert rtn_maxq == ref_maxq
    assert torch.equal(rtn_scale, ref_scale)
    assert torch.equal(rtn, ref)


@scaled_settings(0.4)
@given(
    x=tensors(dtypes=(torch.float32,)),
    bits=st.sampled_from([2, 3, 4, 8]),
    group_size=st.sampled_from([-1, 16, 32, 64]),
)
def test_opt_rtn_int_sym_codes_and_never_worse_than_rtn(x, bits, group_size):
    qdq, scale, maxq = quant_tensor_opt_rtn_sym(x.clone(), bits=bits, group_size=group_size)
    assert_shape_dtype_finite(qdq, x)
    xg, qg = _grouped(x, group_size), _grouped(qdq, group_size)
    s = to_f64(scale)
    codes = (qg / s).round().clamp(-maxq, maxq - 1)
    assert_where((qg - s * codes).abs() <= dtype_slack(x.dtype, s * codes), "opt-RTN output off its grid", qdq=qg)

    rtn = _grouped(quant_tensor_sym(x, bits=bits, group_size=group_size, scale_dtype=torch.float32)[0], group_size)
    opt_sse = ((xg - qg) ** 2).sum(-1)
    rtn_sse = ((xg - rtn) ** 2).sum(-1)
    # a few fp32 ulps per element of absolute slack: RTN can be exact (sse == 0) where the searched scale is 1 ulp off
    ulp_slack = (xg.abs().amax(-1) * 2**-21) ** 2 * xg.shape[-1]
    # the search is unaware of q_scale_thresh: a searched |scale| below it is clipped afterwards, by design
    clipped = s.reshape(-1).abs() <= Q_SCALE_THRESH * (1 + 1e-6)
    assert_where(
        (opt_sse <= rtn_sse * (1 + 1e-3) + ulp_slack) | clipped,
        "scale search ended worse than its own RTN starting point",
        opt_sse=opt_sse,
        rtn_sse=rtn_sse,
    )
