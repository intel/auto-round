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
"""Fuzzing of the Nunchaku/SVDQuant physical tensor layouts.

Every packer here is a padding step followed by a fixed permutation (plus nibble packing), so the
oracles are: unpack(pack(x)) restores the padded input, the packed tensor is a permutation of the
padded input, and the dequantized payload equals the reference quantizer in runtime channel order.
"""

from test.fuzz_helpers import FLOAT_DTYPES, FUZZ_DEVICE, TENSOR_PATTERNS, assert_where, scaled_settings, tensors

import pytest
import torch
from hypothesis import given, note
from hypothesis import strategies as st

from auto_round.data_type.mxfp import quant_mx_rceil
from auto_round.export.formats.svdquant.mxfp4 import NunchakuMXFP4Packer, pack_lowrank_weight, unpack_lowrank_weight
from auto_round.export.formats.svdquant.nunchaku import pack_nunchaku_16bit_vector, unpack_nunchaku_16bit_vector
from auto_round.export.formats.svdquant.w4a16 import (
    dequantize_adanorm_w4a16,
    pack_adanorm_w4a16,
    quantize_adanorm_w4a16_rtn,
    unpack_adanorm_w4a16,
)

HALF_DTYPES = (torch.float16, torch.bfloat16)

_W4A16_SUBNORMAL_SCALE_REASON = (
    "fuzz-found: quantize_adanorm_w4a16_rtn rounds absmax / 7 to an FP16 subnormal scale, so for a group "
    "whose absmax is a few multiples of 2**-24 the stored scale is too small and |w / scale| rounds past 7"
)


def _ceil_to(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _assert_is_permutation(packed: torch.Tensor, padded: torch.Tensor) -> None:
    assert packed.numel() == padded.numel()
    assert torch.equal(packed.flatten().sort().values, padded.flatten().sort().values)


def _pad(t: torch.Tensor, shape, value=0.0) -> torch.Tensor:
    out = torch.full(shape, value, dtype=t.dtype, device=t.device)
    out[tuple(slice(0, s) for s in t.shape)] = t
    return out


@given(
    down=st.booleans(),
    weight=tensors(shapes=st.tuples(st.integers(1, 40), st.integers(1, 300)), dtypes=HALF_DTYPES),
)
def test_lowrank_pack_is_a_padded_permutation_and_unpack_inverts_it(down, weight):
    rows, cols = weight.shape
    padded = _pad(weight, (_ceil_to(rows, 16 if down else 128), _ceil_to(cols, 128 if down else 16)))

    packed = pack_lowrank_weight(weight.clone(), down)
    # The physical layout is always [channels, rank]; for ``down`` the logical matrix is [rank, channels].
    assert packed.dtype == weight.dtype and packed.shape == (padded.T.shape if down else padded.shape)
    _assert_is_permutation(packed, padded)
    assert torch.equal(unpack_lowrank_weight(packed, down), padded)


@given(vector=tensors(shapes=st.integers(1, 400).map(lambda n: (n,)), dtypes=HALF_DTYPES))
def test_16bit_vector_pack_is_a_one_padded_permutation_and_unpack_inverts_it(vector):
    padded = _pad(vector, (_ceil_to(vector.numel(), 128),), value=1.0)

    packed = pack_nunchaku_16bit_vector(vector.clone())
    assert packed.dtype == vector.dtype
    _assert_is_permutation(packed, padded)
    assert torch.equal(unpack_nunchaku_16bit_vector(packed), padded)


@given(
    dtype=st.sampled_from(FLOAT_DTYPES),
    shape=st.tuples(st.integers(1, 3), st.integers(1, 200)),
    down=st.booleans(),
    defect=st.sampled_from(("inf", "nan", "fp32", "1d", "empty", "int_down")),
)
def test_svdquant_layout_packers_reject_malformed_inputs(dtype, shape, down, defect):
    weight = torch.ones(shape, dtype=dtype if dtype in HALF_DTYPES else torch.float16)
    vector = weight[0].clone()
    if defect in ("inf", "nan"):
        weight[-1, -1] = vector[-1] = float(defect)
    elif defect == "fp32":
        weight, vector = weight.float(), vector.float()
    elif defect == "1d":
        weight, vector = vector, weight
    elif defect == "empty":
        weight, vector = weight[:, :0], vector[:0]
    with pytest.raises(ValueError):
        pack_lowrank_weight(weight, 1 if defect == "int_down" else down)
    if defect != "int_down":
        with pytest.raises(ValueError):
            pack_nunchaku_16bit_vector(vector)

    residual = torch.ones(shape, dtype=dtype)
    if defect in ("inf", "nan"):
        residual[0, 0] = float(defect)
    elif defect == "1d":
        residual = residual[0]
    else:
        with pytest.raises(ValueError):
            NunchakuMXFP4Packer().pack_residual(residual, group_size=64)
        return
    with pytest.raises(ValueError):
        NunchakuMXFP4Packer().pack_residual(residual)


@given(weight=tensors(shapes=st.tuples(st.integers(1, 200), st.integers(1, 300))))
@scaled_settings(0.5)
def test_mxfp4_residual_unpacks_to_quant_mx_rceil(weight):
    packer = NunchakuMXFP4Packer()
    n, k = weight.shape
    packed = packer.pack_residual(weight.clone())
    n_pad, k_pad = _ceil_to(n, 128), _ceil_to(k, 128)
    assert packed.logical_shape == (n, k) and packed.padded_shape == (n_pad, k_pad)
    assert packed.qweight.dtype == torch.int8 and packed.qweight.shape == (n_pad, k_pad // 2)
    assert packed.wscales.dtype == torch.uint8 and packed.wscales.shape == (k_pad // 32, n_pad)

    expected = quant_mx_rceil(weight.clone(), bits=4, group_size=32, data_type="mx_fp4e2m1")[0]
    # Narrower decode dtypes are covered by the decode_e2m1 properties in test_nibble_codec_fuzz.py.
    for dtype in (torch.float32, torch.float64):
        decoded = packer.unpack_residual(packed.qweight, packed.wscales, (n, k), dtype=dtype)
        assert decoded.dtype == dtype
        assert_where(decoded == expected.to(dtype), "residual must decode to the MXFP4 QDQ", qdq=expected)

    full = packer.unpack_residual(packed.qweight, packed.wscales, (n_pad, k_pad))
    assert torch.equal(full[:n, :k], packer.unpack_residual(packed.qweight, packed.wscales, (n, k)))
    padding = torch.ones_like(full, dtype=torch.bool)
    padding[:n, :k] = False
    assert not bool(full[padding].any()), "padding must decode to zero"


@given(n_tiles=st.integers(1, 3), k_tiles=st.integers(1, 3), seed=st.integers(0, 2**16))
def test_mxfp4_physical_layout_is_a_bijection_on_codes(n_tiles, k_tiles, seed):
    packer = NunchakuMXFP4Packer()
    n, k = 128 * n_tiles, 128 * k_tiles
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8).to(FUZZ_DEVICE)
    scale_codes = torch.randint(0, 256, (n, k // 32), generator=g, dtype=torch.uint8).to(FUZZ_DEVICE)

    qweight = packer._pack_weight_codes(codes)
    assert qweight.dtype == torch.int8 and qweight.shape == (n, k // 2)
    nibbles = torch.stack((qweight.view(torch.uint8) & 0xF, qweight.view(torch.uint8) >> 4))
    histogram = [torch.bincount(t.flatten().long(), minlength=16) for t in (nibbles, codes)]
    assert torch.equal(*histogram)
    assert torch.equal(packer._unpack_weight_codes(qweight), codes)

    wscales = packer._pack_scale_codes(scale_codes)
    assert wscales.shape == (k // 32, n)
    _assert_is_permutation(wscales, scale_codes)
    assert torch.equal(packer._unpack_scale_codes(wscales), scale_codes)


def _runtime_order(t: torch.Tensor, splits: int) -> torch.Tensor:
    """Interleave the ``splits`` logical output fields the way the Nunchaku kernel reads them."""
    out = t.shape[0]
    return t.reshape(splits, out // splits, *t.shape[1:]).transpose(0, 1).reshape(t.shape)


w4a16_layouts = st.sampled_from((3, 6)).flatmap(
    lambda splits: st.tuples(st.just(splits), st.integers(1, 4).map(lambda m: 12 * m), st.sampled_from((1024, 2048)))
)


@given(
    layout=w4a16_layouts,
    dtype=st.sampled_from(HALF_DTYPES),
    with_bias=st.booleans(),
    chunk_rows=st.sampled_from((4, 8, 12, 256)),
    seed=st.integers(0, 2**16),
)
@scaled_settings(0.5)
def test_w4a16_pack_round_trips_in_runtime_channel_order(layout, dtype, with_bias, chunk_rows, seed):
    splits, out_features, in_features = layout
    groups = in_features // 64
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-7, 8, (out_features, in_features), generator=g)
    exps = torch.randint(-10, 4, (out_features, groups), generator=g).float()
    scale = ((1 + torch.rand(out_features, groups, generator=g)) * torch.exp2(exps)).to(dtype)
    # q * scale may round in ``dtype``, but by far less than half a step for |q| <= 7.
    weight = (q.float() * scale.float().repeat_interleave(64, dim=1)).to(dtype)
    bias = ((torch.rand(out_features, generator=g) * 8 - 4).to(dtype)) if with_bias else None
    weight, scale, q = weight.to(FUZZ_DEVICE), scale.to(FUZZ_DEVICE), q.to(FUZZ_DEVICE)
    bias = bias.to(FUZZ_DEVICE) if bias is not None else None

    packed = pack_adanorm_w4a16(weight, scale, bias=bias, splits=splits, chunk_rows=chunk_rows)
    unchunked = pack_adanorm_w4a16(weight, scale, bias=bias, splits=splits, chunk_rows=None)
    for name in ("qweight", "wscales", "wzeros", "bias"):
        assert torch.equal(getattr(packed, name), getattr(unchunked, name)), f"{name} depends on chunk_rows"

    assert packed.qweight.dtype == torch.int32 and packed.qweight.shape == (out_features // 4, in_features // 2)
    assert torch.equal(unpack_adanorm_w4a16(packed), _runtime_order(q, splits).to(torch.int8))
    runtime_scale = _runtime_order(scale, splits)
    assert torch.equal(packed.wscales, runtime_scale.t())
    expected_qdq = (_runtime_order(q, splits).float() * runtime_scale.float().repeat_interleave(64, dim=1)).to(dtype)
    assert torch.equal(dequantize_adanorm_w4a16(packed), expected_qdq)

    expected_bias = _runtime_order(bias if bias is not None else torch.zeros_like(scale[:, 0]), splits)
    expected_bias = expected_bias.reshape(-1, splits).clone()
    expected_bias[:, sorted({1, splits - 2})] += 1
    assert torch.equal(packed.bias, expected_bias.reshape(-1))


_RTN_PATTERNS = tuple(p for p in TENSOR_PATTERNS if p != "tiny")


@given(
    layout=w4a16_layouts,
    weight_recipe=st.data(),
)
@scaled_settings(0.5)
def test_w4a16_rtn_matches_symmetric_absmax_reference(layout, weight_recipe):
    splits, out_features, in_features = layout
    weight = weight_recipe.draw(
        tensors(shapes=st.just((out_features, in_features)), dtypes=HALF_DTYPES, patterns=_RTN_PATTERNS),
        label="weight",
    )
    dtype = weight.dtype
    grouped = weight.float().reshape(out_features, -1, 64)
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    # Groups whose absmax / 7 is subnormal in ``dtype`` are covered by the xfail test below.
    subnormal = (absmax > 0) & (absmax / 7 < torch.finfo(dtype).smallest_normal)
    if bool(subnormal.any()):
        note(f"zeroing {int(subnormal.sum())} group(s) with subnormal scales")
        grouped = torch.where(subnormal, torch.zeros_like(grouped), grouped)
        weight = grouped.reshape(out_features, in_features).to(dtype)
        absmax = torch.where(subnormal, torch.zeros_like(absmax), absmax)

    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 7).to(dtype)
    q = torch.round(grouped / scale.float()).clamp(-7, 7)
    expected_qdq = (q * scale.float()).reshape(out_features, in_features).to(dtype)

    packed = quantize_adanorm_w4a16_rtn(weight, splits=splits)
    assert torch.equal(packed.wscales, _runtime_order(scale.squeeze(-1), splits).t())
    assert torch.equal(dequantize_adanorm_w4a16(packed), _runtime_order(expected_qdq, splits))


# Multiples m of 2**-24 (the FP16 subnormal step) whose absmax / 7 rounds to q steps with m / q > 7.5.
_FP16_SUBNORMAL_FAILING_STEPS = [m for m in range(4, 128) if (q := round(m / 7)) >= 1 and m / q > 7.5]


@pytest.mark.xfail(reason=_W4A16_SUBNORMAL_SCALE_REASON, strict=True)
@given(
    steps=st.sampled_from(_FP16_SUBNORMAL_FAILING_STEPS),
    negative=st.booleans(),
    row=st.integers(0, 11),
    col=st.integers(0, 1023),
)
@scaled_settings(0.2)
def test_w4a16_rtn_handles_fp16_groups_with_subnormal_scales(steps, negative, row, col):
    weight = torch.zeros(12, 1024, dtype=torch.float16, device=FUZZ_DEVICE)
    weight[row, col] = (-1 if negative else 1) * steps * 2.0**-24
    packed = quantize_adanorm_w4a16_rtn(weight)
    error = (dequantize_adanorm_w4a16(packed).float() - _runtime_order(weight, 3).float()).abs()
    step = packed.wscales.t().float().repeat_interleave(64, dim=1)
    assert_where(error <= step / 2, "RTN error must stay within half a step", step=step)
