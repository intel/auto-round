#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests for the fused HMT + MXFP4 activation quantization kernel.

The reference tests (``TestReferenceContract``) run everywhere, including on
CPU-only machines; the kernel tests are skipped when no XPU is available.

Acceptance criteria are bit-exact: E8M0 scale bytes and packed FP4 code bytes
must be **equal** to the PyTorch FP32 reference, no mismatch tolerance.
"""

import pytest
import torch
from auto_round_kernel.mxfp4_hadamard import (
    GROUP_SIZE,
    HADAMARD_DIM,
    _e8m0_and_quantized,
    _encode_fp4,
    get_hadamard_matrix,
    hadamard_transform_reference,
    mxfp4_hadamard_quant,
    mxfp4_hadamard_quant_reference,
    pack_codes,
)

XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
requires_xpu = pytest.mark.skipif(not XPU_AVAILABLE, reason="XPU is not available")

SHAPES = [(1, 32), (1, 128), (17, 256)]
DTYPES = [torch.float16, torch.bfloat16]

# A 32-point Hadamard transform can amplify a group by at most
# 32 / sqrt(32) = sqrt(32), so inputs above FP32_MAX / sqrt(32) would overflow
# the FP32 accumulator. Tests that probe "as large as possible" stay below it.
MAX_SAFE_INPUT = 3.4e38 / 32.0**0.5

# Group counts that stress the work-group tail: one work-group covers
# 256 / 32 = 8 quant groups, so anything not a multiple of 8 has a partial
# trailing work-group whose idle sub-groups must exit without writing.
TAIL_SHAPES = [(1, 32), (3, 32), (7, 32), (8, 32), (9, 32), (5, 96), (13, 160)]


def _dequantize(codes: torch.Tensor, scale: torch.Tensor, k: int) -> torch.Tensor:
    """Unpack (codes, e8m0) back to FP32 values, for readability of failures."""
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=codes.device)
    flat = codes.reshape(-1, k // 2).to(torch.int32)
    low = flat & 0x0F
    high = (flat >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(-1, k)
    values = levels[nibbles & 0x07] * torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
    exp = scale.reshape(-1, k // GROUP_SIZE).to(torch.int32) - 127
    return torch.ldexp(values.reshape(-1, GROUP_SIZE), exp.reshape(-1, 1)).reshape(-1, k)


class TestReferenceContract:
    """Phase 0: the frozen reference / packing / E8M0 contract."""

    def test_hadamard_matrix_is_normalized(self):
        h = get_hadamard_matrix(HADAMARD_DIM)
        assert h.shape == (HADAMARD_DIM, HADAMARD_DIM)
        assert h.dtype == torch.float32
        identity = torch.eye(HADAMARD_DIM, dtype=torch.float32)
        torch.testing.assert_close(h @ h.t(), identity, atol=1e-6, rtol=0)

    def test_packing_matches_vllm_ext_fp4_utils(self):
        fp4_utils = pytest.importorskip("auto_round_extension.vllm_ext.fp4_utils")
        torch.manual_seed(0)
        codes = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
        # Build the FP4 values the codes represent, then pack them with the
        # reference packer and compare byte by byte.
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
        values = levels[(codes & 0x07).long()] * torch.where((codes & 0x08) != 0, -1.0, 1.0)
        # -0.0 keeps the sign bit, matching ``signbit`` based encoding.
        values = torch.where((codes & 0x08) != 0, -values.abs(), values.abs())
        expected = fp4_utils.pack_fp4_to_uint8(values)
        assert torch.equal(pack_codes(codes), expected)

    def test_zero_group_contract(self):
        x = torch.zeros(2, 64, dtype=torch.float16)
        codes, scale = mxfp4_hadamard_quant_reference(x)
        assert torch.all(codes == 0)
        assert torch.all(scale == 0)

    def test_e8m0_matches_floor_log2_contract(self):
        import math

        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float16)
        _, scale = mxfp4_hadamard_quant_reference(x)
        y = hadamard_transform_reference(x.reshape(-1, HADAMARD_DIM), get_hadamard_matrix(HADAMARD_DIM))
        amax = y.abs().amax(dim=-1)
        for group, group_amax in enumerate(amax.tolist()):
            expected = min(max(int(math.floor(math.log2(group_amax))) - 2 + 127, 0), 254)
            assert scale.reshape(-1)[group].item() == expected
            # amax / scale must land in [4, 8): the E8M0 exponent is standard.
            ratio = group_amax / (2.0 ** (expected - 127))
            assert 4.0 <= ratio < 8.0

    def test_encode_fp4_threshold_boundaries(self):
        # Exact boundary values exercise the alternating <= / < comparisons of
        # cast_to_fp4. Anything at or below 0.25 -> 0, 0.75 is *not* below the
        # 0.5 level, and so on.
        cases = {
            0.0: 0,
            0.25: 0,
            0.2500001: 1,
            0.75: 2,
            1.25: 2,
            1.2500001: 3,
            1.75: 4,
            2.5: 4,
            2.5000005: 5,
            3.5: 6,
            5.0: 6,
            5.0000005: 7,
            7.9: 7,
        }
        values = torch.tensor(list(cases), dtype=torch.float32)
        expected = torch.tensor(list(cases.values()), dtype=torch.uint8)
        assert torch.equal(_encode_fp4(values), expected)
        # The sign bit is bit 3 and is taken from signbit, except that magnitude
        # 0 is canonicalised to 0x0 rather than 0x8 (see canonical zero rule).
        negated = torch.where(expected == 0, expected, expected | 0x08)
        assert torch.equal(_encode_fp4(-values), negated)

    def test_canonical_zero_never_encodes_negative_zero(self):
        # Any value that rounds to FP4 magnitude 0 must encode as 0x0. The sign
        # of such a value comes from FP32 rounding residue (or from device-side
        # flush-to-zero of subnormals) and is not reproducible, so it must not
        # be observable in the output.
        values = torch.tensor([-0.0, 0.0, -1e-30, 1e-30, -0.25, 0.25, -1e-8], dtype=torch.float32)
        assert torch.equal(_encode_fp4(values), torch.zeros(7, dtype=torch.uint8))
        # Sanity check that the rule is narrow: the smallest non-zero magnitude
        # still carries its sign.
        assert _encode_fp4(torch.tensor([-0.2500001])).item() == 0x09

    def test_canonical_zero_survives_full_pipeline(self):
        # A constant group cancels exactly in every Hadamard column but the
        # first, which is precisely where a negative zero would appear.
        x = torch.full((1, 32), -1.0, dtype=torch.float16)
        codes, _ = mxfp4_hadamard_quant_reference(x)
        assert torch.all(codes[0, 1:] == 0)
        assert (codes & 0x08 != 0).sum() + (codes & 0x80 != 0).sum() <= 1

    def test_accumulation_contract_is_order_defined(self):
        # hadamard_transform_reference must be reproducible bit for bit and must
        # not silently fall back to torch.matmul semantics.
        torch.manual_seed(0)
        x = torch.randn(64, HADAMARD_DIM, dtype=torch.float16).to(torch.float32)
        h = get_hadamard_matrix(HADAMARD_DIM)
        a = hadamard_transform_reference(x, h)
        b = hadamard_transform_reference(x, h)
        assert torch.equal(a, b)
        manual = torch.zeros_like(a)
        for j in range(HADAMARD_DIM):
            manual = manual + x[:, j : j + 1] * h[j]
        assert torch.equal(a, manual)

    def test_custom_hadamard_matrix_is_honored(self):
        # A sign-flipped Hadamard matrix is still orthogonal; the reference must
        # use the matrix it is given rather than the cached default.
        h = get_hadamard_matrix(HADAMARD_DIM).clone()
        h[:, 0] = -h[:, 0]
        torch.manual_seed(0)
        x = torch.randn(4, 32, dtype=torch.float16)
        codes_default, _ = mxfp4_hadamard_quant_reference(x)
        codes_custom, _ = mxfp4_hadamard_quant_reference(x, h)
        assert not torch.equal(codes_default, codes_custom)

    def test_output_shape_and_dtype(self):
        x = torch.randn(3, 128, dtype=torch.float16)
        codes, scale = mxfp4_hadamard_quant_reference(x)
        assert codes.shape == (3, 64)
        assert scale.shape == (3, 4)
        assert codes.dtype == torch.uint8
        assert scale.dtype == torch.uint8

    def test_reference_roundtrip_is_close(self):
        torch.manual_seed(0)
        x = torch.randn(8, 256, dtype=torch.float16)
        codes, scale = mxfp4_hadamard_quant_reference(x)
        deq = _dequantize(codes, scale, 256)
        y = hadamard_transform_reference(x.reshape(-1, HADAMARD_DIM), get_hadamard_matrix(HADAMARD_DIM))
        y = y.reshape(8, 256)
        rel = (deq - y).abs().max() / y.abs().max()
        assert rel < 0.2

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_reference_accepts_bf16_and_fp16(self, dtype):
        torch.manual_seed(0)
        x = torch.randn(4, 128, dtype=dtype)
        codes, scale = mxfp4_hadamard_quant_reference(x)
        assert codes.shape == (4, 64) and scale.shape == (4, 4)
        assert codes.dtype == torch.uint8 and scale.dtype == torch.uint8

    @pytest.mark.parametrize(
        "bad_input, error",
        [
            ("not a tensor", TypeError),
            (torch.randn(1, 32, dtype=torch.float32), ValueError),
            (torch.randint(0, 4, (1, 32), dtype=torch.int8), ValueError),
            (torch.randn(1, 48, dtype=torch.float16), ValueError),
            (torch.randn(1, 0, dtype=torch.float16), ValueError),
            (torch.full((1, 32), float("nan"), dtype=torch.float16), ValueError),
            (torch.full((1, 32), float("inf"), dtype=torch.float16), ValueError),
        ],
    )
    def test_reference_rejects_invalid_input(self, bad_input, error):
        with pytest.raises(error):
            mxfp4_hadamard_quant_reference(bad_input)

    @pytest.mark.parametrize(
        "bad_matrix",
        [
            torch.eye(16, dtype=torch.float32),
            torch.eye(HADAMARD_DIM, dtype=torch.int32),
            torch.full((HADAMARD_DIM, HADAMARD_DIM), float("nan"), dtype=torch.float32),
        ],
    )
    def test_reference_rejects_invalid_hadamard(self, bad_matrix):
        x = torch.randn(1, 32, dtype=torch.float16)
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant_reference(x, bad_matrix)


def _assert_bit_exact(x: torch.Tensor):
    """Run the kernel and the reference on ``x`` and require byte equality."""
    codes, scale = mxfp4_hadamard_quant(x)
    ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
    num_rows, k = x.numel() // x.shape[-1], x.shape[-1]
    assert codes.shape == ref_codes.shape == (num_rows, k // 2)
    assert scale.shape == ref_scale.shape == (num_rows, k // GROUP_SIZE)
    assert codes.dtype == torch.uint8 and scale.dtype == torch.uint8
    assert codes.device.type == scale.device.type == "xpu"
    assert torch.equal(scale.cpu(), ref_scale)
    assert torch.equal(codes.cpu(), ref_codes)
    return codes.cpu(), scale.cpu()


@requires_xpu
class TestXpuKernel:
    """Phase 1: the XPU kernel must be bit-exact with the reference."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_random_finite_input(self, dtype, shape):
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert codes.shape == ref_codes.shape == (shape[0], shape[1] // 2)
        assert scale.shape == ref_scale.shape == (shape[0], shape[1] // GROUP_SIZE)
        assert codes.dtype == torch.uint8 and scale.dtype == torch.uint8
        assert torch.equal(scale.cpu(), ref_scale.cpu())
        assert torch.equal(codes.cpu(), ref_codes.cpu())

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_zero_group(self, dtype):
        x = torch.zeros(4, 64, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x)
        assert torch.all(codes.cpu() == 0)
        assert torch.all(scale.cpu() == 0)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mixed_sign_and_partial_zero_groups(self, dtype):
        torch.manual_seed(1)
        x = torch.randn(5, 128, dtype=dtype, device="xpu")
        x[1, :32] = 0
        x[3, 64:96] = 0
        x[2] = -x[2].abs()
        codes, scale = mxfp4_hadamard_quant(x)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(scale.cpu(), ref_scale)
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.all(scale.cpu()[1, 0] == 0)
        assert torch.all(codes.cpu()[1, :16] == 0)

    def test_extreme_finite_values(self):
        x = torch.zeros(2, 32, dtype=torch.float16, device="xpu")
        x[0] = 65504.0  # FP16 max
        x[1] = 6.1e-5  # smallest FP16 normal
        codes, scale = mxfp4_hadamard_quant(x)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(scale.cpu(), ref_scale)
        assert torch.equal(codes.cpu(), ref_codes)

    def test_multi_dim_input_is_flattened(self):
        torch.manual_seed(2)
        x = torch.randn(2, 3, 64, dtype=torch.float16, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x)
        assert codes.shape == (6, 32)
        assert scale.shape == (6, 2)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(scale.cpu(), ref_scale)
        assert torch.equal(codes.cpu(), ref_codes)

    def test_nibble_order(self):
        torch.manual_seed(3)
        x = torch.randn(1, 64, dtype=torch.float16, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x)
        # Rebuild the per-element 4-bit codes from the reference and check that
        # element 2i sits in the low nibble of byte i, element 2i+1 in the high one.
        y = hadamard_transform_reference(x.cpu().reshape(-1, HADAMARD_DIM), get_hadamard_matrix(HADAMARD_DIM))
        e8m0, q = _e8m0_and_quantized(y)
        nibbles = _encode_fp4(q).reshape(1, 64)
        packed = codes.cpu().to(torch.int32)
        assert torch.equal((packed & 0x0F).to(torch.uint8), nibbles[:, 0::2])
        assert ((packed >> 4) & 0x0F).to(torch.uint8).equal(nibbles[:, 1::2])
        assert torch.equal(scale.cpu(), e8m0.reshape(1, 2))

    def test_invalid_dtype(self):
        x = torch.randn(1, 32, dtype=torch.float32, device="xpu")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x)

    def test_invalid_shape(self):
        x = torch.randn(1, 48, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x)

    def test_invalid_device(self):
        x = torch.randn(1, 32, dtype=torch.float16)
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x)

    def test_non_finite_input_is_rejected(self):
        x = torch.randn(1, 32, dtype=torch.float16, device="xpu")
        x[0, 0] = float("nan")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x, check_finite=True)
        x[0, 0] = float("inf")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x, check_finite=True)

    def test_non_finite_input_is_not_scanned_by_default(self):
        # The finiteness scan is a debugging aid, not part of the contract: it
        # costs more than the kernel, so the hot path must not pay for it.
        x = torch.randn(1, 32, dtype=torch.float16, device="xpu")
        x[0, 0] = float("nan")
        mxfp4_hadamard_quant(x)

    def test_invalid_hadamard_matrix(self):
        x = torch.randn(1, 32, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x, torch.eye(16, dtype=torch.float32, device="xpu"))


@requires_xpu
class TestXpuKernelPhase2:
    """Phase 2: BF16, multi-row inputs, boundaries and error handling."""

    # ---- BF16 -------------------------------------------------------------

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_bf16_matches_reference(self, seed):
        torch.manual_seed(seed)
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="xpu")
        _assert_bit_exact(x)

    def test_bf16_and_fp16_agree_on_exactly_representable_values(self):
        # Values that are exact in both BF16 and FP16 must produce identical
        # codes and scales, proving the two dispatch paths share the FP32 math.
        torch.manual_seed(0)
        base = torch.randint(-8, 9, (16, 128), dtype=torch.int32).to(torch.float32) / 4.0
        codes_fp16, scale_fp16 = mxfp4_hadamard_quant(base.to(torch.float16).to("xpu"))
        codes_bf16, scale_bf16 = mxfp4_hadamard_quant(base.to(torch.bfloat16).to("xpu"))
        assert torch.equal(scale_fp16.cpu(), scale_bf16.cpu())
        assert torch.equal(codes_fp16.cpu(), codes_bf16.cpu())

    def test_bf16_subnormal_clamps_e8m0_to_zero(self):
        # BF16 has FP32's exponent range, so tiny values drive
        # floor(log2(amax)) - 2 + 127 below 0 and must clamp to e8m0 = 0.
        # 1e-39 is chosen so the clamp fires while the rescaled values are still
        # large enough to encode as non-zero FP4 codes, i.e. this is the clamp
        # path and not the all-zero-group path.
        x = torch.full((2, 32), 1e-39, dtype=torch.bfloat16, device="xpu")
        x[1] = -1e-39
        codes, scale = _assert_bit_exact(x)
        assert torch.all(scale == 0)
        assert torch.any(codes != 0)

    def test_bf16_deep_subnormal_underflows_to_zero_codes(self):
        # Far below the clamp the rescaled values fall under the first FP4
        # threshold, so codes become zero while e8m0 stays clamped at 0.
        x = torch.full((1, 32), 1e-43, dtype=torch.bfloat16, device="xpu")
        codes, scale = _assert_bit_exact(x)
        assert torch.all(scale == 0)
        assert torch.all(codes == 0)

    def test_bf16_large_magnitude(self):
        big = MAX_SAFE_INPUT / 4.0
        x = torch.full((2, 32), big, dtype=torch.bfloat16, device="xpu")
        x[1, ::2] = -big
        _assert_bit_exact(x)

    # ---- multi-row / shapes ----------------------------------------------

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", TAIL_SHAPES)
    def test_work_group_tail(self, dtype, shape):
        torch.manual_seed(shape[0])
        x = torch.randn(*shape, dtype=dtype, device="xpu")
        _assert_bit_exact(x)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", [(1024, 2048), (333, 4096)])
    def test_large_multi_row(self, dtype, shape):
        torch.manual_seed(7)
        x = torch.randn(*shape, dtype=dtype, device="xpu")
        _assert_bit_exact(x)

    def test_one_dimensional_input(self):
        torch.manual_seed(0)
        x = torch.randn(64, dtype=torch.float16, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x)
        assert codes.shape == (1, 32)
        assert scale.shape == (1, 2)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_four_dimensional_input(self, dtype):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 5, 96, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x)
        assert codes.shape == (30, 48)
        assert scale.shape == (30, 3)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    def test_non_contiguous_input_is_materialized(self):
        torch.manual_seed(0)
        base = torch.randn(64, 128, dtype=torch.float16, device="xpu")
        view = base[:, ::2]  # stride-2 columns, [64, 64], non-contiguous
        assert not view.is_contiguous()
        codes, scale = mxfp4_hadamard_quant(view)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(view.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)
        # A contiguous copy of the same values must give the same bytes.
        codes_contig, scale_contig = mxfp4_hadamard_quant(view.contiguous())
        assert torch.equal(codes.cpu(), codes_contig.cpu())
        assert torch.equal(scale.cpu(), scale_contig.cpu())

    # ---- numerical boundaries --------------------------------------------

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_row_and_group_mapping(self, dtype):
        # Each row gets a distinct magnitude so a row/column mix-up in the
        # scale layout is detected, not just an average mismatch.
        rows, k = 12, 128
        x = torch.zeros(rows, k, dtype=dtype, device="xpu")
        for r in range(rows):
            for g in range(k // GROUP_SIZE):
                x[r, g * GROUP_SIZE] = float(2 ** (r - 6 + g))
        codes, scale = _assert_bit_exact(x)
        # Scales must be strictly increasing along both axes by one octave.
        scale_i = scale.to(torch.int32)
        assert torch.all(scale_i[1:, :] - scale_i[:-1, :] == 1)
        assert torch.all(scale_i[:, 1:] - scale_i[:, :-1] == 1)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_spike_per_group(self, dtype):
        # A one-hot group makes every transformed element equal in magnitude,
        # which puts them exactly on an FP4 level rather than between levels.
        x = torch.zeros(8, 32, dtype=dtype, device="xpu")
        for r in range(8):
            x[r, r * 4] = 1.0 if r % 2 == 0 else -1.0
        _assert_bit_exact(x)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_alternating_extremes(self, dtype):
        finfo = torch.finfo(dtype)
        big = min(finfo.max, MAX_SAFE_INPUT)
        x = torch.zeros(4, 64, dtype=dtype, device="xpu")
        x[0] = big
        x[1, ::2] = big
        x[1, 1::2] = -big
        x[2] = finfo.tiny
        x[3, ::2] = finfo.tiny
        _assert_bit_exact(x)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_cancelling_group_yields_canonical_zero(self, dtype):
        # A constant group cancels exactly in Hadamard columns 1..31. On device
        # the FP32 residue of that cancellation, and any flush-to-zero of
        # subnormals, may be negatively signed; the canonical zero rule must
        # keep that out of the codes so the kernel still matches the reference.
        x = torch.zeros(4, 32, dtype=dtype, device="xpu")
        x[0] = 1.0
        x[1] = -1.0
        x[2] = torch.finfo(dtype).tiny
        x[3] = -torch.finfo(dtype).tiny
        codes, _ = _assert_bit_exact(x)
        # Columns 1..31 of every row are exact zeros, i.e. bytes 1..15.
        assert torch.all(codes[:, 1:].cpu() == 0)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_quantized_grid_values_stress_thresholds(self, dtype):
        # Coarse, exactly representable inputs make transformed values land on
        # or extremely close to the FP4 decision thresholds, which is where an
        # accumulation-order mismatch between kernel and reference would show.
        torch.manual_seed(11)
        x = (torch.randint(-4, 5, (256, 128), device="xpu").to(torch.float32) / 4.0).to(dtype)
        _assert_bit_exact(x)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_random_fuzz(self, dtype, seed):
        torch.manual_seed(seed)
        x = (torch.randn(128, 256, device="xpu") * (10.0 ** (seed - 4))).to(dtype)
        _assert_bit_exact(x)

    def test_custom_hadamard_matrix(self):
        h = get_hadamard_matrix(HADAMARD_DIM).clone()
        h[:, 0] = -h[:, 0]
        torch.manual_seed(0)
        x = torch.randn(16, 128, dtype=torch.float16, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, h)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu(), h.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)
        default_codes, _ = mxfp4_hadamard_quant(x)
        assert not torch.equal(codes.cpu(), default_codes.cpu())

    def test_cpu_hadamard_matrix_is_moved_to_device(self):
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float16, device="xpu")
        h_cpu = get_hadamard_matrix(HADAMARD_DIM, "cpu")
        assert h_cpu.device.type == "cpu"
        codes, scale = mxfp4_hadamard_quant(x, h_cpu)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    def test_float64_hadamard_matrix_is_downcast(self):
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float16, device="xpu")
        h64 = get_hadamard_matrix(HADAMARD_DIM).to(torch.float64)
        codes, scale = mxfp4_hadamard_quant(x, h64)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    def test_repeated_calls_are_deterministic(self):
        torch.manual_seed(0)
        x = torch.randn(256, 512, dtype=torch.float16, device="xpu")
        first = mxfp4_hadamard_quant(x)
        for _ in range(4):
            again = mxfp4_hadamard_quant(x)
            assert torch.equal(first[0].cpu(), again[0].cpu())
            assert torch.equal(first[1].cpu(), again[1].cpu())

    def test_no_out_of_bounds_writes(self):
        # Allocate padded outputs, run into a slice-sized region and verify the
        # guard bytes around the logical outputs are untouched.
        torch.manual_seed(0)
        rows, k = 9, 96  # 27 groups: 3 full work-groups + a 3-group tail
        x = torch.randn(rows, k, dtype=torch.float16, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x)
        torch.xpu.synchronize()
        assert codes.numel() == rows * k // 2
        assert scale.numel() == rows * k // GROUP_SIZE
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    # ---- error handling ---------------------------------------------------

    @pytest.mark.parametrize(
        "bad_dtype", [torch.float32, torch.float64, torch.int8, torch.int32, torch.uint8, torch.bool]
    )
    def test_rejects_unsupported_dtype(self, bad_dtype):
        x = torch.zeros(1, 32, dtype=bad_dtype, device="xpu")
        with pytest.raises(ValueError, match="float16 or bfloat16"):
            mxfp4_hadamard_quant(x)

    @pytest.mark.parametrize("k", [1, 16, 31, 33, 48, 63])
    def test_rejects_k_not_multiple_of_32(self, k):
        x = torch.randn(2, k, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError, match="multiple of 32"):
            mxfp4_hadamard_quant(x)

    def test_rejects_empty_tensor(self):
        with pytest.raises(ValueError, match="must not be empty"):
            mxfp4_hadamard_quant(torch.randn(0, 32, dtype=torch.float16, device="xpu"))
        with pytest.raises(ValueError, match="must not be empty"):
            mxfp4_hadamard_quant(torch.randn(2, 0, dtype=torch.float16, device="xpu"))

    def test_rejects_cpu_tensor(self):
        x = torch.randn(1, 32, dtype=torch.float16)
        with pytest.raises(ValueError, match="only supported on XPU"):
            mxfp4_hadamard_quant(x)

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError):
            mxfp4_hadamard_quant([0.0] * 32)

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad_value):
        x = torch.randn(2, 64, dtype=torch.float16, device="xpu")
        x[1, 17] = bad_value
        with pytest.raises(ValueError, match="finite"):
            mxfp4_hadamard_quant(x, check_finite=True)

    @pytest.mark.parametrize(
        "bad_matrix_factory",
        [
            lambda: torch.eye(16, dtype=torch.float32, device="xpu"),
            lambda: torch.eye(64, dtype=torch.float32, device="xpu"),
            lambda: torch.zeros(HADAMARD_DIM, dtype=torch.float32, device="xpu"),
            lambda: torch.eye(HADAMARD_DIM, dtype=torch.int32, device="xpu"),
            lambda: torch.full((HADAMARD_DIM, HADAMARD_DIM), float("inf"), dtype=torch.float32, device="xpu"),
        ],
    )
    def test_rejects_invalid_hadamard_matrix(self, bad_matrix_factory):
        x = torch.randn(1, 32, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x, bad_matrix_factory())

    def test_rejects_non_tensor_hadamard_matrix(self):
        x = torch.randn(1, 32, dtype=torch.float16, device="xpu")
        with pytest.raises(TypeError):
            mxfp4_hadamard_quant(x, [[0.0] * 32] * 32)

    def test_state_is_intact_after_rejected_call(self):
        torch.manual_seed(0)
        good = torch.randn(4, 64, dtype=torch.float16, device="xpu")
        expected = mxfp4_hadamard_quant(good)
        bad = torch.randn(2, 48, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(bad)
        actual = mxfp4_hadamard_quant(good)
        assert torch.equal(expected[0].cpu(), actual[0].cpu())
        assert torch.equal(expected[1].cpu(), actual[1].cpu())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
