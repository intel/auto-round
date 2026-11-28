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
"""Property-based fuzzing of the MX (shared power-of-two exponent) QDQ kernels in ``auto_round.data_type.mxfp``."""

import re
from test.fuzz_helpers import (
    FUZZ_DEVICE,
    assert_monotone_within_groups,
    assert_on_scaled_grid,
    assert_shape_dtype_finite,
    assert_where,
    assert_zero_and_sign_preserved,
    dtype_slack,
    group_view,
    make_tensor,
    mx_grid,
    scaled_settings,
    tensors,
    to_f64,
)

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from auto_round.data_type import QUANT_FUNC_WITH_DTYPE, get_quant_func
from auto_round.data_type.mxfp import (
    MXFP_FORMAT_CACHE,
    quant_mx,
    quant_mx_opt_rtn,
    quant_mx_rceil,
    quant_mx_rceil_v2,
    quant_mx_uint,
)

MX_KEYS = sorted(MXFP_FORMAT_CACHE)
GROUP_SIZES = st.one_of(st.sampled_from([-1, 16, 32, 64]), st.integers(1, 64))
ROUNDINGS = st.sampled_from(["even", "nearest", "floor"])
SCALE_EMAX = 127
# fp32 log2 of a value within this relative distance of a power of two may land on either side of it
LOG2_REL_TOL = 2.0**-20
_EXPLICIT_MANTISSA_BITS = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}


def _bits_of(key):
    return int(re.search(r"\d+", key).group())


def _floor_log2(v):
    _, e = torch.frexp(v)
    return (e - 1).to(torch.float64)


def _ceil_log2(v):
    m, e = torch.frexp(v)
    return torch.where(m == 0.5, e - 1, e).to(torch.float64)


def _rceil_ratio_underflows(x, group_size, max_norm):
    amax = group_view(x, group_size).float().abs().amax(-1)
    return bool(((amax > 0) & (amax / max_norm == 0)).any())


def _assert_exponent_bracketed(se, amax, log2_fn, offset, zero_value, what):
    """``se`` must equal ``clamp(log2_fn(amax) - offset)``, allowing fp32 log2 slop next to powers of two."""
    lo = (log2_fn(amax * (1 - LOG2_REL_TOL)) - offset).clamp(-SCALE_EMAX, SCALE_EMAX)
    hi = (log2_fn(amax * (1 + LOG2_REL_TOL)) - offset).clamp(-SCALE_EMAX, SCALE_EMAX)
    zero = torch.full_like(se, float(zero_value))
    ok = torch.where(amax == 0, se == zero, (se >= lo) & (se <= hi))
    assert_where(ok, f"{what} shared exponent differs from the oracle", amax=amax, shared_exp=se, lo=lo, hi=hi)


def _check_elements(x, qdq, shared_exp, key, group_size, mantissa_rounding="even"):
    """Shared checks for every MX kernel whose elements live on ``key``'s element grid."""
    ebits, mbits, _, max_norm, _ = MXFP_FORMAT_CACHE[key]
    grid = mx_grid(ebits, mbits, max_norm)
    xg, qg = to_f64(group_view(x, group_size)), to_f64(group_view(qdq, group_size))
    se = to_f64(shared_exp)
    assert se.shape == (xg.shape[0], 1)
    assert_where(se == se.round(), "shared exponent is not an integer", shared_exp=se)
    scale = torch.exp2(se)

    assert_on_scaled_grid(qg, scale, grid, x.dtype)
    y = xg.abs() / scale
    step_err = grid.max_error(y) * (2 if mantissa_rounding == "floor" else 1)
    bound = scale * step_err + dtype_slack(x.dtype, qg)
    assert_where((xg - qg).abs() <= bound, f"{key} error exceeds the element grid bound", x=xg, qdq=qg, scale=scale)
    if mantissa_rounding == "floor":
        assert_where(qg.abs() <= xg.abs() + dtype_slack(x.dtype, qg), "floor rounding grew a magnitude", x=xg, qdq=qg)
    assert_zero_and_sign_preserved(xg, qg)
    assert_monotone_within_groups(xg, qg, x.dtype)
    return xg, qg, se, grid


@given(x=tensors(), key=st.sampled_from(MX_KEYS), group_size=GROUP_SIZES, mantissa_rounding=ROUNDINGS)
def test_mx_floor_scale_invariants(x, key, group_size, mantissa_rounding):
    qdq, shared_exp, extra = quant_mx(
        x, bits=_bits_of(key), group_size=group_size, data_type=key, mantissa_rounding=mantissa_rounding
    )
    assert extra is None
    assert_shape_dtype_finite(qdq, x)
    assert shared_exp.dtype == x.dtype
    xg, _, se, grid = _check_elements(x, qdq, shared_exp, key, group_size, mantissa_rounding)

    emax = MXFP_FORMAT_CACHE[key][2]
    amax = xg.abs().amax(-1, keepdim=True)
    _assert_exponent_bracketed(se, amax, _floor_log2, emax, max(1 - emax, -SCALE_EMAX), "floor")

    fits = x.dtype != torch.float16 and grid.mbits <= _EXPLICIT_MANTISSA_BITS[x.dtype]
    if mantissa_rounding == "even" and fits:
        again = quant_mx(qdq, bits=_bits_of(key), group_size=group_size, data_type=key)[0]
        assert torch.equal(again, qdq), "re-quantizing an MX QDQ output must be a fixed point"


@given(x=tensors(), key=st.sampled_from(MX_KEYS), group_size=GROUP_SIZES)
def test_mx_rceil_scale_never_saturates(x, key, group_size):
    """RCEIL picks the smallest power-of-two scale with ``amax / scale <= max_norm``."""
    max_norm = MXFP_FORMAT_CACHE[key][3]
    assume(not _rceil_ratio_underflows(x, group_size, max_norm))  # covered by test_mx_rceil_tiny_group_stays_finite
    qdq, shared_exp, extra = quant_mx_rceil(x, bits=_bits_of(key), group_size=group_size, data_type=key)
    assert extra is None
    assert_shape_dtype_finite(qdq, x)
    xg, _, se, _ = _check_elements(x, qdq, shared_exp, key, group_size)

    amax = xg.abs().amax(-1, keepdim=True)
    _assert_exponent_bracketed(se, amax / max_norm, _ceil_log2, 0, 1, "rceil")
    ratio = amax / torch.exp2(se)
    unclamped = (se > -SCALE_EMAX) & (se < SCALE_EMAX) & (amax > 0)
    assert_where(
        ~unclamped | (ratio <= max_norm * (1 + LOG2_REL_TOL)), "rceil scale saturates the group max", ratio=ratio
    )
    assert_where(
        ~unclamped | (ratio > max_norm / 2 * (1 - LOG2_REL_TOL)), "rceil scale is not the smallest", ratio=ratio
    )


@given(x=tensors(), group_size=GROUP_SIZES)
def test_mx_fp4_rceil_v2_scale_window(x, group_size):
    """The v2 variant targets ``amax / scale`` in (3.625, 7.25] and never uses a larger scale than RCEIL."""
    qdq, shared_exp, _ = quant_mx_rceil_v2(x, bits=4, group_size=group_size, data_type="mx_fp4")
    assert_shape_dtype_finite(qdq, x)
    xg, _, se, _ = _check_elements(x, qdq, shared_exp, "mx_fp4", group_size)

    amax = xg.abs().amax(-1, keepdim=True)
    _assert_exponent_bracketed(se, amax / 7.25, _ceil_log2, 0, 1, "rceil_v2")
    rceil_se = to_f64(quant_mx_rceil(x, bits=4, group_size=group_size, data_type="mx_fp4")[1])
    assert_where(se <= rceil_se, "rceil_v2 chose a coarser scale than rceil", v2=se, rceil=rceil_se)


@given(x=tensors(), bits=st.sampled_from([2, 3, 4, 8]), group_size=GROUP_SIZES)
def test_mx_uint_codes_and_error(x, bits, group_size):
    qdq, shared_exp, zero_point = quant_mx_uint(x, bits=bits, group_size=group_size)
    assert_shape_dtype_finite(qdq, x)
    assert zero_point == 2 ** (bits - 1)
    maxq = 2**bits - 1

    xg, qg = to_f64(group_view(x, group_size)), to_f64(group_view(qdq, group_size))
    se = to_f64(shared_exp)
    amax = xg.abs().amax(-1, keepdim=True)
    _assert_exponent_bracketed(se, amax, _floor_log2, 0, 0, "mx_uint")

    step = torch.exp2(se) / 2.0 ** (bits - 2)
    codes = (qg / step).round()
    assert_where(
        (qg - codes * step).abs() <= dtype_slack(x.dtype, qg), "mx_uint output is not code * step", qdq=qg, step=step
    )
    assert_where((codes >= -zero_point) & (codes <= maxq - zero_point), "mx_uint code out of range", code=codes)
    t = xg / step
    overflow = (t - (maxq - zero_point)).clamp(min=0) + (-zero_point - t).clamp(min=0)
    bound = step * (0.5 + overflow) + dtype_slack(x.dtype, qg)
    assert_where((xg - qg).abs() <= bound, "mx_uint error exceeds half a step", x=xg, qdq=qg, step=step)
    assert_zero_and_sign_preserved(xg, qg)


@scaled_settings(0.5)
@given(x=tensors(), key=st.sampled_from(MX_KEYS), group_size=st.sampled_from([-1, 16, 32, 64]))
def test_mx_opt_rtn_never_worse_than_floor_rtn(x, key, group_size):
    """The exponent search tries 0.5x/1x/2x of the group max, so per group it can only lower the SSE."""
    qdq, shared_exp, _ = quant_mx_opt_rtn(x.clone(), bits=_bits_of(key), group_size=group_size, data_type=key)
    assert_shape_dtype_finite(qdq, x)
    xg, qg, se, _ = _check_elements(x, qdq, shared_exp, key, group_size)

    ref, ref_se, _ = quant_mx(x, bits=_bits_of(key), group_size=group_size, data_type=key)
    assert_where((se - to_f64(ref_se)).abs() <= 1, "opt-RTN moved the exponent by more than one binade", se=se)
    rg = to_f64(group_view(ref, group_size))
    opt_sse = ((xg - qg) ** 2).sum(-1)
    ref_sse = ((xg - rg) ** 2).sum(-1)
    slack = (dtype_slack(x.dtype, xg) ** 2).sum(-1) * 4
    assert_where(
        opt_sse <= ref_sse * (1 + 1e-6) + slack, "opt-RTN ended worse than floor RTN", opt=opt_sse, ref=ref_sse
    )


@pytest.mark.xfail(
    strict=True,
    reason="fuzz-found: quant_mx_rceil takes ceil_ste(log2(amax / max_norm)); for mx_bf16 the ratio underflows fp32 "
    "to 0 when 0 < amax < ~4.7e-7, log2 yields -inf and ceil_ste(-inf) is NaN, so the whole group becomes NaN",
)
@pytest.mark.parametrize("key", ["mx_bf16", "mx_bfloat16"])
def test_mx_rceil_tiny_group_stays_finite(key):
    x = torch.tensor([[1.5e-7, -1e-8, 0.0, 3e-9]], device=FUZZ_DEVICE)
    qdq = quant_mx_rceil(x, bits=16, group_size=-1, data_type=key)[0]
    assert torch.isfinite(qdq).all()


# ``quant_mx_opt_rtn`` / ``quant_mx_rceil`` resolve the element format with
# ``data_type if data_type in MXFP_FORMAT_CACHE else "mx_fp" + str(bits)``, but the
# wrapper passes the *registry key* (e.g. "opt_rtn_mx_int4", "mx_int8_rceil"), so every
# format whose name is not ``mx_fp<bits>``-equivalent silently falls back to another one.
_MISROUTED = {"mx_int8", "mx_int4", "mx_int2", "mx_fp8e5m2", "mx_fp6e3m2", "mx_bfloat16", "mx_bf16"}
_MISROUTE_REASON = (
    "fuzz-found: registry key {key!r} is not in MXFP_FORMAT_CACHE, so the kernel falls back to "
    "'mx_fp{bits}' and quantizes onto the wrong element grid"
)


def _registry_routing_cases():
    for base in MX_KEYS:
        for key in (f"opt_rtn_{base}", f"{base}_rceil"):
            marks = ()
            if base in _MISROUTED:
                marks = pytest.mark.xfail(strict=True, reason=_MISROUTE_REASON.format(key=key, bits=_bits_of(base)))
            yield pytest.param(key, base, id=key, marks=marks)


@pytest.mark.parametrize("key, base", list(_registry_routing_cases()))
def test_mx_registry_key_uses_its_own_element_format(key, base):
    """Calling a kernel the way ``WrapperLinear`` does (``data_type=<registry key>``) must match the base format."""
    func = QUANT_FUNC_WITH_DTYPE[key]
    x = make_tensor((4, 64), torch.float32, "normal", seed=0, magnitude=1.0, device=FUZZ_DEVICE)
    got = func(x.clone(), bits=_bits_of(base), group_size=32, data_type=key)[0]
    want = func(x.clone(), bits=_bits_of(base), group_size=32, data_type=base)[0]
    assert torch.equal(got, want)


@pytest.mark.xfail(
    strict=True,
    reason="fuzz-found: the MXINT4 scheme with iters=0 resolves to 'opt_rtn_mx_int4', which falls back to "
    "the mx_fp4 element grid {0, .5, 1, 1.5, 2, 3, 4, 6} instead of the int4 grid",
)
def test_mxint4_scheme_opt_rtn_stays_on_int4_grid():
    func, resolved = get_quant_func("mx_int", 4, True, disable_opt_rtn=False, group_size=32, iters=0)
    x = make_tensor((4, 64), torch.float32, "normal", seed=0, magnitude=1.0, device=FUZZ_DEVICE)
    qdq, shared_exp, _ = func(x.clone(), bits=4, group_size=32, data_type=resolved)
    ebits, mbits, _, max_norm, _ = MXFP_FORMAT_CACHE["mx_int4"]
    scale = torch.exp2(to_f64(shared_exp))
    assert_on_scaled_grid(to_f64(group_view(qdq, 32)), scale, mx_grid(ebits, mbits, max_norm), x.dtype)
