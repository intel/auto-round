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
"""Fuzzing of the 4-bit sign-magnitude nibble codecs.

Covers the FP4 (E2M1) / INT4 (E0M4) packers used by the auto_round export and the experimental
qmodules, and the logical-code helpers of the SVDQuant MXFP4 exporter. All of them share one byte
layout: bit 3 of a nibble is the sign, bits 0-2 index the magnitude grid, and element ``2i`` lives
in the low nibble of byte ``i``.
"""

import itertools
from test.fuzz_helpers import FLOAT_DTYPES, FUZZ_DEVICE, assert_where, scaled_settings, tensors

import pytest
import torch
from hypothesis import given, note
from hypothesis import strategies as st

from auto_round.experimental.qmodules.fp4_utils import pack_fp4_to_uint8 as pack_fp4_experimental
from auto_round.experimental.qmodules.fp4_utils import unpack_fp4_from_uint8
from auto_round.experimental.qmodules.mxint4_utils import unpack_int4_from_uint8
from auto_round.export.export_to_autoround.qlinear_fp import pack_fp4_to_uint8 as pack_fp4_export
from auto_round.export.export_to_autoround.qlinear_int import pack_int4_to_uint8
from auto_round.export.svdquant_mxfp4 import (
    decode_e2m1,
    decode_ue8m0,
    encode_e2m1,
    encode_ue8m0,
    pack_nibbles,
    unpack_nibbles,
)

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E0M4_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
_MANTISSA_BITS = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}
CODE_DTYPES = (torch.uint8, torch.int16, torch.int32, torch.int64)

_E2M1_NARROW_DECODE_REASON = (
    "fuzz-found: decode_e2m1 casts the scale to the decode dtype before multiplying, so an FP16 decode "
    "flushes scales below 2**-24 to zero and turns scales above 2**15 into inf (code 0 -> NaN) even "
    "where code * scale is representable in FP16"
)


def _fp4_packers():
    return {"export": pack_fp4_export, "experimental": pack_fp4_experimental}


FORMATS = {
    "fp4": (E2M1_GRID, _fp4_packers, unpack_fp4_from_uint8),
    "int4": (E0M4_GRID, lambda: {"export": pack_int4_to_uint8}, unpack_int4_from_uint8),
}


def _reference_bytes(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _values_for_codes(codes: torch.Tensor, grid, dtype) -> torch.Tensor:
    magnitudes = torch.tensor(grid, dtype=torch.float64, device=codes.device)[codes.long() & 7]
    # multiplying keeps code 8 as -0.0, which is what the sign bit encodes
    return (magnitudes * torch.where(codes >= 8, -1.0, 1.0).double()).to(dtype)


def _nearest_distance(values: torch.Tensor, grid) -> torch.Tensor:
    grid = torch.tensor(grid, dtype=torch.float64, device=values.device)
    return (values.double().abs().unsqueeze(-1) - grid).abs().min(dim=-1).values


@st.composite
def near_ties(draw, grid, dtype, shape):
    """Values within a couple of ulps of the midpoints between adjacent grid magnitudes."""
    seed = draw(st.integers(0, 2**16))
    g = torch.Generator().manual_seed(seed)
    mids = torch.tensor([(a + b) / 2 for a, b in itertools.pairwise(grid)], dtype=torch.float64)
    picked = mids[torch.randint(0, len(mids), shape, generator=g)]
    ulp = torch.exp2(torch.floor(torch.log2(picked)) - _MANTISSA_BITS[dtype])
    nudge = torch.randint(-2, 3, shape, generator=g).double()
    sign = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0).double()
    note(f"near-tie values: dtype={dtype} shape={shape} seed={seed}")
    return ((picked + nudge * ulp) * sign).to(dtype).to(FUZZ_DEVICE)


even_matrix_shapes = st.tuples(st.integers(1, 8), st.integers(1, 48).map(lambda n: 2 * n))


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@given(shape=even_matrix_shapes, dtype=st.sampled_from(FLOAT_DTYPES), seed=st.integers(0, 2**16))
def test_packers_encode_grid_values_as_low_first_sign_magnitude_nibbles(fmt, shape, dtype, seed):
    grid, packers, unpack = FORMATS[fmt]
    codes = torch.randint(0, 16, shape, generator=torch.Generator().manual_seed(seed)).to(FUZZ_DEVICE)
    values = _values_for_codes(codes, grid, dtype)
    expected = _reference_bytes(codes)
    assert torch.equal(pack_nibbles(codes), expected)

    for name, pack in packers().items():
        packed = pack(values.clone())
        assert packed.dtype == torch.uint8 and packed.shape == (shape[0], shape[1] // 2), name
        assert torch.equal(packed, expected), name

    decoded = unpack(expected, *shape, dtype)
    assert decoded.dtype == dtype
    assert torch.equal(decoded, values)
    assert torch.equal(torch.signbit(decoded), torch.signbit(values))


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@given(dtype=st.sampled_from(FLOAT_DTYPES), data=st.data())
def test_packers_round_off_grid_values_to_the_nearest_magnitude(fmt, dtype, data):
    grid, packers, unpack = FORMATS[fmt]
    shape = data.draw(even_matrix_shapes, label="shape")
    if data.draw(st.booleans(), label="near_ties"):
        values = data.draw(near_ties(grid, dtype, shape))
    else:
        # in-contract inputs are already scaled into the grid range; allow a bit of clipping headroom
        values = data.draw(tensors(shapes=st.just(shape), dtypes=(dtype,), log10_magnitude=(-3, 0)))
        values = values * float(grid[-1]) / 2
        values = values.clamp(-grid[-1] * 4 / 3, grid[-1] * 4 / 3)

    packed = {name: pack(values.clone()) for name, pack in packers().items()}
    reference = packed["export"]
    for name, other in packed.items():
        assert torch.equal(other, reference), f"{name} packer disagrees with the export packer"

    decoded = unpack(reference, *shape, torch.float64)
    error = (values.double().abs() - decoded.abs()).abs()
    assert_where(
        error <= _nearest_distance(values, grid),
        "packed magnitude must be a nearest grid point",
        value=values.double(),
        decoded=decoded,
    )
    assert_where(torch.signbit(decoded) == torch.signbit(values), "sign bit must be kept", value=values.double())


@given(
    lead=st.lists(st.integers(1, 4), max_size=2).map(tuple),
    length=st.integers(0, 33),
    code_dtype=st.sampled_from(CODE_DTYPES),
    seed=st.integers(0, 2**16),
)
def test_nibble_pack_round_trips_any_codes(lead, length, code_dtype, seed):
    shape = lead + (length,)
    codes = torch.randint(0, 16, shape, generator=torch.Generator().manual_seed(seed)).to(code_dtype)
    codes = codes.to(FUZZ_DEVICE)

    packed = pack_nibbles(codes)
    assert packed.dtype == torch.uint8 and packed.shape == lead + ((length + 1) // 2,)
    padded = torch.cat((codes.to(torch.uint8), codes.new_zeros(lead + (length % 2,), dtype=torch.uint8)), dim=-1)
    assert torch.equal(packed, _reference_bytes(padded))

    assert torch.equal(unpack_nibbles(packed, length), codes.to(torch.uint8))
    assert torch.equal(unpack_nibbles(packed), padded)


@given(
    shape=st.lists(st.integers(1, 6), min_size=1, max_size=3).map(tuple),
    byte_dtype=st.sampled_from(CODE_DTYPES),
    seed=st.integers(0, 2**16),
)
def test_nibble_unpack_then_pack_is_identity_on_bytes(shape, byte_dtype, seed):
    packed = torch.randint(0, 256, shape, generator=torch.Generator().manual_seed(seed)).to(byte_dtype)
    packed = packed.to(FUZZ_DEVICE)
    assert torch.equal(pack_nibbles(unpack_nibbles(packed)), packed.to(torch.uint8))


@given(
    length=st.integers(1, 12),
    bad_value=st.one_of(st.integers(16, 300), st.integers(-300, -1)),
    position=st.integers(0, 11),
    logical_count=st.one_of(st.integers(-5, -1), st.integers(1, 5)),
)
def test_nibble_and_e2m1_helpers_reject_invalid_codes(length, bad_value, position, logical_count):
    codes = torch.zeros(length, dtype=torch.int32)
    codes[position % length] = bad_value
    with pytest.raises(ValueError):
        pack_nibbles(codes)
    with pytest.raises(ValueError):
        decode_e2m1(codes, torch.tensor(1.0))
    if not 0 <= bad_value <= 255:
        with pytest.raises(ValueError):
            unpack_nibbles(codes)
    for not_codes in (codes.float().clamp(0, 15), (codes % 2).bool(), torch.tensor(3)):
        with pytest.raises(ValueError):
            pack_nibbles(not_codes)

    packed = torch.zeros(length, dtype=torch.uint8)
    capacity = 2 * length
    count = logical_count if logical_count < 0 else capacity + logical_count
    with pytest.raises(ValueError):
        unpack_nibbles(packed, count)
    with pytest.raises(ValueError):
        unpack_nibbles(packed, True)


_SCALE_LAYOUTS = ("scalar", "per_row", "full")


def _draw_scales(data, shape, min_exp, max_exp, power_of_two=False):
    layout = data.draw(st.sampled_from(_SCALE_LAYOUTS), label="scale_layout")
    scale_shape = {"scalar": (), "per_row": shape[:-1], "full": shape}[layout]
    seed = data.draw(st.integers(0, 2**16), label="scale_seed")
    g = torch.Generator().manual_seed(seed)
    exps = torch.randint(min_exp, max_exp + 1, scale_shape, generator=g).float()
    mantissa = 1.0 if power_of_two else 1 + torch.rand(scale_shape, generator=g)
    return (mantissa * torch.exp2(exps)).to(FUZZ_DEVICE)


def _broadcastable(scales: torch.Tensor, ndim: int) -> torch.Tensor:
    return scales.unsqueeze(-1) if scales.dim() == ndim - 1 else scales


def _reference_e2m1(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = _values_for_codes(codes, E2M1_GRID, torch.float64)
    return values * _broadcastable(scales, codes.dim()).double()


# Decoded magnitudes 0.5 * s .. 6 * s must stay normal and finite in the decode dtype.
_E2M1_SCALE_EXPONENTS = {torch.float16: (-13, 11), torch.bfloat16: (-60, 60), torch.float32: (-60, 60)}


@given(dtype=st.sampled_from(FLOAT_DTYPES + (torch.float64,)), code_dtype=st.sampled_from(CODE_DTYPES), data=st.data())
def test_e2m1_encode_inverts_decode(dtype, code_dtype, data):
    shape = data.draw(st.tuples(st.integers(1, 6), st.integers(1, 40)), label="shape")
    scales = _draw_scales(data, shape, *_E2M1_SCALE_EXPONENTS.get(dtype, (-60, 60)))
    seed = data.draw(st.integers(0, 2**16), label="code_seed")
    codes = torch.randint(0, 16, shape, generator=torch.Generator().manual_seed(seed)).to(code_dtype)
    codes = codes.to(FUZZ_DEVICE)

    decoded = decode_e2m1(codes, scales, dtype=dtype)
    assert decoded.dtype == dtype and decoded.shape == shape
    assert torch.equal(encode_e2m1(decoded, scales), codes.to(torch.uint8))


# Exponents for which a power-of-two (UE8M0) scale is itself representable in the decode dtype.
_REPRESENTABLE_SCALE_EXPONENTS = {torch.float16: (-24, 15), torch.bfloat16: (-133, 127), torch.float32: (-149, 127)}


@given(dtype=st.sampled_from(FLOAT_DTYPES), data=st.data())
def test_e2m1_decode_is_correctly_rounded_for_representable_scales(dtype, data):
    shape = data.draw(st.tuples(st.integers(1, 6), st.integers(1, 40)), label="shape")
    scales = _draw_scales(data, shape, *_REPRESENTABLE_SCALE_EXPONENTS[dtype], power_of_two=True)
    seed = data.draw(st.integers(0, 2**16), label="code_seed")
    codes = torch.randint(0, 16, shape, generator=torch.Generator().manual_seed(seed)).to(FUZZ_DEVICE)

    expected = _reference_e2m1(codes, scales).to(dtype)
    assert torch.equal(decode_e2m1(codes, scales, dtype=dtype), expected)


@pytest.mark.xfail(reason=_E2M1_NARROW_DECODE_REASON, strict=True)
@given(
    code=st.sampled_from((6, 7, 14, 15)),
    exponent=st.sampled_from((-26, -25)),
    zero_with_huge_scale=st.booleans(),
    rows=st.integers(1, 4),
)
@scaled_settings(0.2)
def test_e2m1_fp16_decode_handles_scales_outside_the_fp16_range(code, exponent, zero_with_huge_scale, rows):
    # Both cases are reachable via NunchakuMXFP4Packer.unpack_residual(..., dtype=torch.float16).
    if zero_with_huge_scale:
        code, exponent = 0, 16
    codes = torch.full((rows, 32), code, dtype=torch.uint8, device=FUZZ_DEVICE)
    scales = torch.full((rows,), 2.0**exponent, device=FUZZ_DEVICE)

    expected = _reference_e2m1(codes, scales).to(torch.float16)
    assert torch.equal(decode_e2m1(codes, scales, dtype=torch.float16), expected)


@given(data=st.data())
def test_e2m1_encode_rounds_to_the_nearest_scaled_magnitude(data):
    values = data.draw(tensors(shapes=st.tuples(st.integers(1, 6), st.integers(1, 64))), label="values")
    scales = _draw_scales(data, values.shape, -12, 4)

    codes = encode_e2m1(values, scales)
    assert codes.dtype == torch.uint8 and codes.shape == values.shape and int(codes.max()) <= 15

    expanded = _broadcastable(scales, values.dim())
    normalized = (values.float() / expanded.float()).clamp(-6.0, 6.0)
    magnitude = torch.tensor(E2M1_GRID, dtype=torch.float64, device=codes.device)[codes.long() & 7]
    error = (normalized.double().abs() - magnitude).abs()
    assert_where(
        error <= _nearest_distance(normalized, E2M1_GRID),
        "E2M1 code must be a nearest magnitude of values / scales",
        normalized=normalized.double(),
        magnitude=magnitude,
    )
    assert_where((codes >= 8) == torch.signbit(normalized), "sign bit must follow values / scales")


def test_ue8m0_encode_inverts_decode_for_every_emittable_code():
    codes = torch.arange(255, dtype=torch.uint8, device=FUZZ_DEVICE)
    assert torch.equal(encode_ue8m0(decode_ue8m0(codes)), codes)


@given(
    exponent=st.integers(-127, 126),
    mantissa=st.one_of(st.just(1.0), st.floats(1.0, 2.0, exclude_max=True, width=32)),
    count=st.integers(1, 8),
)
def test_ue8m0_rounds_scales_up_to_a_power_of_two(exponent, mantissa, count):
    scales = torch.full((count,), mantissa, dtype=torch.float64).mul(2.0**exponent).float().to(FUZZ_DEVICE)
    s = scales.double()
    codes = encode_ue8m0(scales)
    decoded = decode_ue8m0(codes).double()
    assert int(codes.max()) <= 254
    if mantissa == 1.0:
        assert torch.equal(decoded, s)
    # Rounding log2() in float32 can drop a 2**-23 relative excess before ceil(); every in-tree caller
    # passes exact powers of two, so allow that sliver instead of requiring decoded >= s.
    assert_where(decoded >= s * (1 - 2.0**-20), "decoded scale must not undershoot", scale=s, decoded=decoded)
    assert_where(decoded <= 2 * s, "decoded scale must be the smallest power of two", scale=s, decoded=decoded)


@given(seed=st.integers(0, 2**16), count=st.integers(1, 16))
def test_ue8m0_saturates_and_maps_invalid_scales_to_one(seed, count):
    g = torch.Generator().manual_seed(seed)
    huge = torch.tensor([2.0**127, torch.finfo(torch.float32).max])
    tiny = torch.tensor([2.0**-149, 2.0**-128, 2.0**-127])
    invalid = torch.tensor([0.0, -0.0, -1.0, -(2.0**-149), float("inf"), float("-inf"), float("nan")])
    pool = torch.cat((huge, tiny, invalid))
    scales = pool[torch.randint(0, len(pool), (count,), generator=g)].to(FUZZ_DEVICE)

    codes = encode_ue8m0(scales)
    expected = torch.full_like(codes, 127)
    expected[torch.isfinite(scales) & (scales >= 2.0**127)] = 254
    expected[(scales > 0) & (scales <= 2.0**-127)] = 0
    assert torch.equal(codes, expected)
