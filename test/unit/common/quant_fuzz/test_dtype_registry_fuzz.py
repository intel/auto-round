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
"""Registry-wide fuzzing of every ``QUANT_FUNC_WITH_DTYPE`` entry.

The per-family files check each format's numerics against an oracle. This file checks the
contract every registered QDQ function shares, calling it exactly the way ``WrapperLinear``
does, in both the RTN layout (``iters == 0``: scalar ``v``/margins, no ``tensor_min/max``) and
the tuning layout (per-element ``v``, per-group margins and min/max). ``CASES`` must cover the
whole registry, so a newly registered data type is fuzzed as soon as it gets a case.
"""

import math
import re
from dataclasses import dataclass
from test.fuzz_helpers import (
    FLOAT_DTYPES,
    FUZZ_DEVICE,
    Q_SCALE_THRESH,
    TENSOR_PATTERNS,
    assert_shape_dtype_finite,
    assert_where,
    dtype_slack,
    group_view,
    make_tensor,
    scaled_settings,
    to_f64,
    weight_shapes,
    wrapper_quant_kwargs,
)

import pytest
import torch
from hypothesis import assume, given, note
from hypothesis import strategies as st

from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

SCALE_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@dataclass(frozen=True)
class Case:
    configs: tuple  # kwargs dicts holding ``bits``/``group_size`` plus key-specific extras
    col_multiple: int = 1  # GGUF k-quants work on 256-wide super-blocks
    budget: float = 0.4  # fraction of the active profile's ``max_examples``
    max_cols: int = 300


def _grid(bits, group_sizes, **extra):
    return tuple(dict(bits=b, group_size=g, **extra) for b in bits for g in group_sizes)


QK_K = 256
INT = Case(_grid((2, 3, 4, 8), (-1, 16, 32, 128)))
GGUF_SYM_DQ = Case(
    (
        dict(bits=2, group_size=16, super_group_size=16, super_bits=4),
        dict(bits=3, group_size=16, super_group_size=16, super_bits=6),
        dict(bits=6, group_size=16, super_group_size=16, super_bits=8),
    ),
    col_multiple=QK_K,
)
GGUF_ASYM_DQ = Case(
    (
        dict(bits=2, group_size=16, super_group_size=16, super_bits=4),
        dict(bits=4, group_size=32, super_group_size=8, super_bits=6),
        dict(bits=5, group_size=32, super_group_size=8, super_bits=6),
    ),
    col_multiple=QK_K,
)
FLOAT_ZP = Case(_grid((2, 3, 4, 5, 8), (-1, 32, 128)))
NV = Case(_grid((4,), (16,)))
NV_V2 = Case(_grid((4,), (16, 32)))
FP8 = Case(_grid((8,), (-1, 32, 128)))
FP8_PER_TENSOR = Case(_grid((8,), (-1,)))
BLOCK_FP8 = Case(_grid((8,), ((16, 16), (8, 32), (128, 128))))


def _mx_case(key):
    fmt = key.removeprefix("opt_rtn_").removesuffix("_v2").removesuffix("_rceil")
    if fmt in ("mx_uint", "mx_uint4", "mxuint4"):
        bits = (4,)
    elif fmt == "mx_fp":
        bits = (4, 8)
    else:
        bits = (int(re.search(r"\d+", fmt).group()),)
    return Case(_grid(bits, (32,)))


CASES = {
    "int_sym": INT,
    "int_asym": INT,
    "rtn_int_sym": INT,
    "int_sym_gptq": INT,
    "opt_rtn_int_sym": Case(INT.configs, budget=0.2),
    "opt_rtn_int_asym": Case(INT.configs, budget=0.2),
    "opt_rtn_int_sym_neuqi": Case(INT.configs, budget=0.1, max_cols=128),
    "int_sym_dq": GGUF_SYM_DQ,
    "int_asym_dq": GGUF_ASYM_DQ,
    "rtn_int_sym_dq": Case(GGUF_SYM_DQ.configs[1:], col_multiple=QK_K),  # Q3_K / Q6_K only
    "rtn_int_asym_dq": GGUF_ASYM_DQ,
    "int_asym_float_zp": FLOAT_ZP,
    "opt_rtn_int_asym_float_zp": FLOAT_ZP,
    "nv_fp4": NV,
    "nv_fp4_rtn": NV,
    "nv_fp4_with_static_gs": NV,
    "rtn_nv_fp4_with_static_gs": NV,
    "opt_rtn_nv_fp4": NV,
    "nvfp4_v2": NV_V2,
    "nvfp4_v2_with_global_scale": NV_V2,
    "nvfp4_v2_gs": NV,
    "nvfp4_v2_with_static_gs": NV,
    "opt_rtn_nvfp4_v2_gs": NV,
    "fp8_sym": FP8,
    "fp8": FP8,
    "fp8_e4m3": FP8,
    "fp8_e5m2": FP8,
    "fp8_unit_scale": FP8_PER_TENSOR,
    "fp8_e5m2_unit_scale": FP8_PER_TENSOR,
    "block_fp8_sym": BLOCK_FP8,
    "block_fp8": BLOCK_FP8,
    "block_fp8_e4m3": BLOCK_FP8,
    "fp8_to_int_sym": Case(_grid((4,), (-1, 32, 128))),
    **{key: _mx_case(key) for key in QUANT_FUNC_WITH_DTYPE if "mx" in key},
}

EXCLUDED = {"fp8_gaudi3_sym": "HPU-only kernel, exercised by the HPU test suite"}

# ``opt_rtn_*`` / ``rtn_*`` are only resolved for ``iters == 0`` (see ``get_quant_func``);
# ``nv_fp4_rtn`` is the RTN candidate/fallback of the opt-RTN NVFP4 search.
TUNABLE = [key for key in CASES if not key.startswith(("opt_rtn_", "rtn_")) and key != "nv_fp4_rtn"]

_MX_INT2_REASON = (
    "fuzz-found: {key!r} is not in MXFP_FORMAT_CACHE and the bits fallback looks up 'mx_fp2', "
    "raising KeyError; see test_mx_quant_fuzz.py::test_mx_registry_key_uses_its_own_element_format"
)
CALL_FAILS = {key: _MX_INT2_REASON.format(key=key) for key in ("opt_rtn_mx_int2", "mx_int2_rceil")}

_MUTATES_REASON = (
    "fuzz-found: the kernel runs in-place ops on reshape_pad_tensor_by_group_size(...) / .to(float32), "
    "which alias the caller's tensor for fp32 input that needs no padding; WrapperLinear passes "
    "orig_layer.weight itself, so the layer weight is overwritten"
)
MUTATES_INPUT = {
    **{key: _MUTATES_REASON for key in QUANT_FUNC_WITH_DTYPE if key.startswith("opt_rtn_mx_")},
    "rtn_int_sym_dq": _MUTATES_REASON,
    "opt_rtn_int_asym_float_zp": _MUTATES_REASON,
}

_FLOAT_ZP_RECIPROCAL_REASON = (
    "fuzz-found: opt_rtn_int_asym_float_zp clamps a degenerate fp16 scale to q_scale_thresh=1e-5, and "
    "get_reciprocal(fp16 1e-5) ~ 99864 overflows fp16 to inf: zero weights become 0 * inf = NaN and tiny "
    "weights saturate to 0 or maxq * scale, where int_asym_float_zp rounds them to ~0"
)
_NVFP4_GS_DTYPE_REASON = (
    "fuzz-found: nvfp4_v2_with_global_scale only recasts tensor_max to fp32 when numel() != 1; a "
    "one-element bf16/fp16 tensor_max (WrapperLinear's weight_max for a single-group weight) yields a "
    "low-precision global_scale and trips ref_fp4_quant's float32 assertion"
)

# Float-zp kernels round ``x / s`` *before* adding the fractional zero point, so interior values sit
# on ``s * k`` while the clamped ends sit on ``m + s * q``: up to ``2**bits + 1`` levels per group.
# The resulting q4_1/q5_1 export mismatch is pinned in gguf_fuzz/test_gguf_roundtrip_fuzz.py.
LEVELS_UNCHECKED = {"int_asym_float_zp", "opt_rtn_int_asym_float_zp"}


def _float_zp_reciprocal_overflows(x, config, scale_dtype, mode):
    """Some group's fp16 scale lies in the overflow window (see ``_FLOAT_ZP_RECIPROCAL_REASON``)."""
    if scale_dtype != torch.float16:
        return False
    filled = torch.nan_to_num(_groups(to_f64(x), config["group_size"]), nan=0.0)
    scale = (filled.amax(-1).clamp(min=0) - filled.amin(-1).clamp(max=0)) / (2 ** config["bits"] - 1)
    return bool((scale < 2 * Q_SCALE_THRESH).any())


def _single_low_precision_group(x, config, scale_dtype, mode):
    """See ``_NVFP4_GS_DTYPE_REASON``."""
    return mode == "tune" and x.dtype != torch.float32 and _groups(x, config["group_size"]).shape[0] == 1


# Inputs that trigger a known bug, excluded from the sweeps and pinned by a strict xfail below.
KNOWN_PRECONDITIONS = {
    "opt_rtn_int_asym_float_zp": _float_zp_reciprocal_overflows,
    "nvfp4_v2_with_global_scale": _single_low_precision_group,
}


def _params(keys, failing):
    params = []
    for key in keys:
        reason = CALL_FAILS.get(key) or failing.get(key)
        raises = KeyError if key in CALL_FAILS else AssertionError
        marks = pytest.mark.xfail(strict=True, raises=raises, reason=reason) if reason else ()
        params.append(pytest.param(key, marks=marks))
    return params


@st.composite
def _shapes(draw, case, group_size, mode):
    if isinstance(group_size, tuple):
        if mode == "tune":  # get_scale_shape floors, so the tuning layout needs whole blocks
            return (group_size[0] * draw(st.integers(1, 2)), group_size[1] * draw(st.integers(1, 2)))
        return (draw(st.integers(1, 2 * group_size[0])), draw(st.integers(1, 2 * group_size[1])))
    if case.col_multiple > 1:
        return (draw(st.integers(1, 4)), case.col_multiple * draw(st.integers(1, 2)))
    return draw(weight_shapes(max_cols=case.max_cols))


@st.composite
def _calls(draw, key, mode):
    case = CASES[key]
    config = draw(st.sampled_from(case.configs))
    shape = draw(_shapes(case, config["group_size"], mode))
    dtype = draw(st.sampled_from(FLOAT_DTYPES))
    pattern = draw(st.sampled_from(TENSOR_PATTERNS))
    seed = draw(st.integers(0, 2**16))
    magnitude = 10.0 ** draw(st.integers(-3, 2))
    scale_dtype = draw(st.sampled_from(SCALE_DTYPES))
    note(
        f"{key} {mode}: config={config} shape={shape} dtype={dtype} pattern={pattern} seed={seed} "
        f"magnitude={magnitude:g} scale_dtype={scale_dtype}"
    )
    x = make_tensor(shape, dtype, pattern, seed, magnitude, device=FUZZ_DEVICE)
    return x, config, scale_dtype, (lambda: wrapper_quant_kwargs(key, x, config, scale_dtype, mode, seed))


def _groups(t, group_size):
    """Quantization groups of a float64 2-D tensor as rows, padded with NaN."""
    if not isinstance(group_size, tuple):
        return group_view(t, group_size, pad_value=math.nan)
    m, n = group_size
    t = torch.nn.functional.pad(t, (0, (-t.shape[1]) % n, 0, (-t.shape[0]) % m), value=math.nan)
    return t.view(t.shape[0] // m, m, t.shape[1] // n, n).permute(0, 2, 1, 3).reshape(-1, m * n)


def _distinct_per_group(t, group_size):
    s = _groups(t, group_size).sort(dim=-1).values
    valid = ~torch.isnan(s)
    changes = (s[:, 1:] != s[:, :-1]) & valid[:, 1:]
    return valid[:, 0].long() + changes.sum(dim=-1)


def _qdq(key, x, kwargs):
    out = QUANT_FUNC_WITH_DTYPE[key](x.clone(), **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _check_contract(key, x, config, scale_dtype, make_kwargs, mode):
    precondition = KNOWN_PRECONDITIONS.get(key)
    assume(precondition is None or not precondition(x, config, scale_dtype, mode))
    qdq = _qdq(key, x, make_kwargs())
    assert_shape_dtype_finite(qdq, x)
    xd, qd = to_f64(x), to_f64(qdq)

    bits = config["bits"]
    if key not in LEVELS_UNCHECKED and 2**bits < _groups(qd, config["group_size"]).shape[-1]:
        levels = _distinct_per_group(qd, config["group_size"])
        assert_where(levels <= 2**bits, f"more than 2**{bits} distinct values in a group", levels=levels)

    if mode == "rtn":
        # Tuning offsets may legitimately round an all-zero group up by one step, so the range
        # bound only holds without them. Scales clamped to q_scale_thresh add up to one such step.
        bound = 2 * xd.abs().max() + 2 * Q_SCALE_THRESH
        assert_where(
            qd.abs() <= bound + dtype_slack(x.dtype, bound), "output exceeds twice the input range", x=xd, qdq=qd
        )

    if qdq.device.type == "cpu":
        assert torch.equal(qdq, _qdq(key, x, make_kwargs())), "same input, different output"


def test_every_registry_key_has_a_case():
    assert not set(CASES) & set(EXCLUDED)
    assert set(CASES) | set(EXCLUDED) == set(QUANT_FUNC_WITH_DTYPE)


@pytest.mark.parametrize("key", _params(CASES, {}))
def test_registry_contract_rtn_layout(key):
    @scaled_settings(CASES[key].budget)
    @given(call=_calls(key, "rtn"))
    def check(call):
        x, config, scale_dtype, make_kwargs = call
        _check_contract(key, x, config, scale_dtype, make_kwargs, "rtn")

    check()


@pytest.mark.parametrize("key", _params(TUNABLE, {}))
def test_registry_contract_tuning_layout(key):
    @scaled_settings(CASES[key].budget * 0.75)
    @given(call=_calls(key, "tune"))
    def check(call):
        x, config, scale_dtype, make_kwargs = call
        _check_contract(key, x, config, scale_dtype, make_kwargs, "tune")

    check()


def _fixed_shape(case, config):
    group_size = config["group_size"]
    if isinstance(group_size, tuple):
        return (group_size[0], 2 * group_size[1])
    return (3, max(case.col_multiple, 256))


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("key", _params(CASES, {"opt_rtn_int_asym_float_zp": _FLOAT_ZP_RECIPROCAL_REASON}))
def test_registry_zero_input_maps_to_zero(key, dtype):
    case = CASES[key]
    for config in case.configs:
        x = torch.zeros(_fixed_shape(case, config), dtype=dtype, device=FUZZ_DEVICE)
        qdq = _qdq(key, x, wrapper_quant_kwargs(key, x, config, torch.float16, "rtn", 0))
        assert_shape_dtype_finite(qdq, x)
        assert_where(qdq == 0, f"zero input does not map to zero for {config}", qdq=qdq)


@pytest.mark.parametrize("key", _params(CASES, MUTATES_INPUT))
def test_registry_does_not_modify_its_input(key):
    case = CASES[key]
    for mode in ("rtn", "tune") if key in TUNABLE else ("rtn",):
        for config in case.configs:
            x = make_tensor(_fixed_shape(case, config), torch.float32, "normal", 0, 1.0, device=FUZZ_DEVICE)
            before = x.clone()
            QUANT_FUNC_WITH_DTYPE[key](x, **wrapper_quant_kwargs(key, x, config, torch.float16, mode, 0))
            assert torch.equal(x, before), f"{mode} call with {config} modified the input tensor"


def _assert_float_zp_kernels_agree(x, config, kwargs):
    fast, scale, _ = QUANT_FUNC_WITH_DTYPE["opt_rtn_int_asym_float_zp"](x.clone(), **kwargs)
    assert_shape_dtype_finite(fast, x)
    ref = _qdq("int_asym_float_zp", x, dict(kwargs, data_type="int_asym_float_zp"))
    group_size = config["group_size"]
    diff = _groups(to_f64(fast) - to_f64(ref), group_size).nan_to_num(0.0)
    ref_groups = _groups(to_f64(ref), group_size).nan_to_num(0.0)
    step = to_f64(scale).reshape(-1, 1)
    # One rounding step, plus both outputs being rounded to x.dtype. The fast kernel works in fp32,
    # while the reference keeps x/s + zp in x.dtype, whose resolution near maxq is maxq*eps steps
    # (about 2 steps for 8-bit bf16).
    maxq = 2 ** config["bits"] - 1
    rel = 1 + 2 * torch.finfo(scale.dtype).eps + maxq * torch.finfo(x.dtype).eps
    tol = step * rel + 2 * dtype_slack(x.dtype, ref_groups)
    assert_where(diff.abs() <= tol, "fast float-zp RTN deviates by more than one step", diff=diff, step=step)


@scaled_settings(0.6)
@given(call=_calls("opt_rtn_int_asym_float_zp", "rtn"))
def test_opt_rtn_float_zp_matches_int_asym_float_zp(call):
    """The fast in-place RTN kernel must agree with the reference float-zp QDQ to within one step."""
    x, config, scale_dtype, make_kwargs = call
    assume(not _float_zp_reciprocal_overflows(x, config, scale_dtype, "rtn"))
    _assert_float_zp_kernels_agree(x, config, make_kwargs())


@pytest.mark.xfail(strict=True, reason=_FLOAT_ZP_RECIPROCAL_REASON)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("degenerate", ["pruned", "tiny"])
def test_opt_rtn_float_zp_degenerate_group_matches_reference(degenerate, dtype):
    x = make_tensor((2, 64), dtype, "normal", 0, 1.0, device=FUZZ_DEVICE)
    x[0, :32] = 0 if degenerate == "pruned" else x[0, :32] * 1e-6  # one degenerate group among regular ones
    config = dict(bits=4, group_size=32)
    _assert_float_zp_kernels_agree(
        x, config, wrapper_quant_kwargs("opt_rtn_int_asym_float_zp", x, config, torch.float16, "rtn", 0)
    )


@pytest.mark.xfail(strict=True, reason=_NVFP4_GS_DTYPE_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_nvfp4_v2_with_global_scale_single_group_tuning_layout(dtype):
    x = make_tensor((1, 16), dtype, "normal", 0, 1.0, device=FUZZ_DEVICE)
    config = dict(bits=4, group_size=16)
    kwargs = wrapper_quant_kwargs("nvfp4_v2_with_global_scale", x, config, torch.float16, "tune", 0)
    assert_shape_dtype_finite(_qdq("nvfp4_v2_with_global_scale", x, kwargs), x)
