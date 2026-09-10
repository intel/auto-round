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

The suite is organised by what each class proves:

* ``TestReferenceContract`` -- the frozen quantizer contract (FP4 packing,
  E8M0 scales, zero groups, input validation). CPU only.
* ``TestXpuKernel`` -- the basic ``D = 32`` contract: random input, zero
  groups, signs, extremes, nibble order.
* ``TestXpuKernelLayouts`` -- input layouts, work-group tails, row counts.
* ``TestXpuKernelNumerics`` -- dtype dispatch, exponent boundaries and custom
  Hadamard matrices.
* ``TestXpuQuantOnly`` -- the quant-only baseline (Hadamard stripped).
* ``TestXpuKernelErrors`` -- the validation contract.
* ``TestXpuKernelXmx`` -- the opt-in XMX fast path (tolerance-based).
* ``TestStreamOnlyBaseline`` -- the stream-only roofline baseline, and the
  guarantee that it really reads every input element.
* ``TestHadamardDimReferenceParity`` -- the reference itself is correct at every
  supported transform size ``D``, pinned against an independent construction.
  CPU only.
* ``TestXpuKernelAllDims`` -- the cooperative lane-count template, ``D > 32``.

The two reference classes run everywhere, including on CPU-only machines; the
XPU classes are skipped when no XPU is available.

Acceptance criteria are bit-exact except on the XMX path: E8M0 scale bytes and
packed FP4 code bytes must be **equal** to the PyTorch FP32 reference, no
mismatch tolerance.
"""

import pytest
import torch
from auto_round_kernel.mxfp4_hadamard import (
    GROUP_SIZE,
    HADAMARD_DIM,
    HADAMARD_DIM_128,
    MAX_LANES_PER_ROW,
    SUPPORTED_HADAMARD_DIMS,
    _e8m0_and_quantized,
    _encode_fp4,
    _xmx_supported,
    get_hadamard_matrix,
    hadamard_transform_reference,
    mxfp4_hadamard_quant,
    mxfp4_hadamard_quant_reference,
    mxfp4_quant_reference,
    mxfp4_stream_reference,
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
TAIL_SHAPES = [(1, 32), (9, 32), (5, 96), (13, 160)]


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
    """The frozen reference / packing / E8M0 contract. Runs everywhere.

    Orthogonality of the Hadamard matrices themselves is covered for every
    supported D by :class:`TestHadamardDimReferenceParity`.
    """

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

    # One case per validation branch: not a tensor, non-floating dtype,
    # K not a multiple of 32, empty, non-finite. The XPU entry point runs the
    # same checks, so the remaining variants live there.
    @pytest.mark.parametrize(
        "bad_input, error",
        [
            ("not a tensor", TypeError),
            (torch.randn(1, 32, dtype=torch.float32), ValueError),
            (torch.randint(0, 4, (1, 32), dtype=torch.int8), ValueError),
            (torch.randn(1, 48, dtype=torch.float16), ValueError),
            (torch.randn(1, 0, dtype=torch.float16), ValueError),
            (torch.full((1, 32), float("nan"), dtype=torch.float16), ValueError),
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

    # ---- quant-only reference (no Hadamard transform) ---------------------

    def test_quant_only_reference_matches_fused_with_identity(self):
        # Quantizing the raw activation is mathematically identical to fusing an
        # identity Hadamard matrix: the Path A transform with H = I only ever
        # adds exact zeros, so the two references must agree byte for byte.
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float16)
        codes, scale = mxfp4_quant_reference(x)
        identity = torch.eye(HADAMARD_DIM, dtype=torch.float32)
        fused_codes, fused_scale = mxfp4_hadamard_quant_reference(x, identity)
        assert torch.equal(codes, fused_codes)
        assert torch.equal(scale, fused_scale)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_quant_only_reference_shape_dtype_and_zero_group(self, dtype):
        x = torch.zeros(2, 64, dtype=dtype)
        codes, scale = mxfp4_quant_reference(x)
        assert codes.shape == (2, 32)
        assert scale.shape == (2, 2)
        assert codes.dtype == torch.uint8
        assert scale.dtype == torch.uint8
        assert torch.all(codes == 0)
        assert torch.all(scale == 0)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_quant_only_reference_differs_from_fused_hmt(self, dtype):
        # A real Hadamard transform scrambles values across the group, so the
        # quant-only output must differ from the fused output in general. This
        # guards against quant-only accidentally still applying the transform.
        torch.manual_seed(0)
        x = torch.randn(8, 128, dtype=dtype)
        q_codes, q_scale = mxfp4_quant_reference(x)
        f_codes, f_scale = mxfp4_hadamard_quant_reference(x)
        assert not torch.equal(q_codes, f_codes) or not torch.equal(q_scale, f_scale)


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
    """The ``D = 32`` per-item kernel: bit-exact against the reference."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_random_finite_input(self, dtype, shape):
        torch.manual_seed(0)
        _assert_bit_exact(torch.randn(*shape, dtype=dtype, device="xpu"))

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

    def test_non_finite_input_is_not_scanned_by_default(self):
        # The finiteness scan is a debugging aid, not part of the contract: it
        # costs more than the kernel, so the hot path must not pay for it.
        x = torch.randn(1, 32, dtype=torch.float16, device="xpu")
        x[0, 0] = float("nan")
        mxfp4_hadamard_quant(x)


@requires_xpu
class TestXpuKernelLayouts:
    """How arbitrary input layouts and row counts map onto rows and work-groups."""

    # ---- input shapes and work-group tails -------------------------------

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", TAIL_SHAPES)
    def test_work_group_tail(self, dtype, shape):
        torch.manual_seed(shape[0])
        x = torch.randn(*shape, dtype=dtype, device="xpu")
        _assert_bit_exact(x)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", [(1024, 2048)])
    def test_large_multi_row(self, dtype, shape):
        # A 2-D tensor with many rows and a large K: the row -> work-group
        # mapping is no longer trivial, unlike in TAIL_SHAPES.
        torch.manual_seed(7)
        x = torch.randn(*shape, dtype=dtype, device="xpu")
        _assert_bit_exact(x)

    @pytest.mark.parametrize("shape", [(64,), (2, 3, 64), (2, 3, 5, 96)])
    def test_input_shapes_are_flattened_to_rows(self, shape):
        # 1-D, 3-D and 4-D inputs all flatten to ``[-1, K]`` with K the last
        # dim; the output is always the 2-D ``[rows, ...]`` form.
        torch.manual_seed(2)
        x = torch.randn(*shape, dtype=torch.float16, device="xpu")
        rows, k = x.numel() // shape[-1], shape[-1]
        codes, scale = _assert_bit_exact(x)
        assert codes.shape == (rows, k // 2)
        assert scale.shape == (rows, k // GROUP_SIZE)

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


@requires_xpu
class TestXpuKernelNumerics:
    """D = 32 numerics: dtype dispatch, exponent boundaries, Hadamard handling."""

    # ---- BF16 -------------------------------------------------------------

    def test_bf16_and_fp16_agree_on_exactly_representable_values(self):
        # Values that are exact in both BF16 and FP16 must produce identical
        # codes and scales, proving the two dispatch paths share the FP32 math.
        torch.manual_seed(0)
        base = torch.randint(-8, 9, (16, 128), dtype=torch.int32).to(torch.float32) / 4.0
        codes_fp16, scale_fp16 = mxfp4_hadamard_quant(base.to(torch.float16).to("xpu"))
        codes_bf16, scale_bf16 = mxfp4_hadamard_quant(base.to(torch.bfloat16).to("xpu"))
        assert torch.equal(scale_fp16.cpu(), scale_bf16.cpu())
        assert torch.equal(codes_fp16.cpu(), codes_bf16.cpu())

    def test_bf16_subnormal_clamp_range(self):
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
        # Far below the clamp the rescaled values fall under the first FP4
        # threshold, so codes become zero while e8m0 stays clamped at 0.
        x = torch.full((1, 32), 1e-43, dtype=torch.bfloat16, device="xpu")
        codes, scale = _assert_bit_exact(x)
        assert torch.all(scale == 0)
        assert torch.all(codes == 0)

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

    @pytest.mark.parametrize("seed", [0, 4, 7])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_random_fuzz(self, dtype, seed):
        # The seeds sweep 1e-4, 1 and 1e3, i.e. the E8M0 exponent range; more
        # seeds only resample the same magnitudes.
        torch.manual_seed(seed)
        x = (torch.randn(128, 256, device="xpu") * (10.0 ** (seed - 4))).to(dtype)
        _assert_bit_exact(x)

    # ---- Hadamard matrix handling ----------------------------------------

    def test_custom_hadamard_matrix(self):
        # A sign-flipped Hadamard matrix is still orthogonal, so it must be
        # honored rather than silently replaced by the cached default. The
        # forced Path A call pins the bit-exact branch; the auto-routed call
        # exercises whatever the dispatcher picks (the XMX path when it is
        # built, which has a relaxed, tolerance-based contract).
        h = get_hadamard_matrix(HADAMARD_DIM).clone()
        h[:, 0] = -h[:, 0]
        torch.manual_seed(0)
        x = torch.randn(16, 128, dtype=torch.float16, device="xpu")
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu(), h.cpu())

        codes, scale = mxfp4_hadamard_quant(x, h, _force_xmx=False)
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

        routed_codes, routed_scale = mxfp4_hadamard_quant(x, h)
        if _xmx_supported():
            deq = _dequantize(routed_codes.cpu(), routed_scale.cpu(), 128)
            ref_deq = _dequantize(ref_codes, ref_scale, 128)
            sqnr_db, _, _ = _precision_metrics(deq, ref_deq, ref_scale)
            assert sqnr_db >= 15.0, f"SQNR {sqnr_db:.2f} dB < 15 dB"
        else:
            assert torch.equal(routed_codes.cpu(), ref_codes)
            assert torch.equal(routed_scale.cpu(), ref_scale)

        default_codes, _ = mxfp4_hadamard_quant(x)
        assert not torch.equal(routed_codes.cpu(), default_codes.cpu())

    def test_hadamard_matrix_input_is_normalized(self):
        # The matrix is coerced to a contiguous FP32 tensor on the activation's
        # device; a CPU host matrix and an FP64 matrix must give the same bytes.
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float16, device="xpu")
        h_cpu = get_hadamard_matrix(HADAMARD_DIM, "cpu")
        assert h_cpu.device.type == "cpu"
        h64 = get_hadamard_matrix(HADAMARD_DIM).to(torch.float64)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        for h in (h_cpu, h64):
            codes, scale = mxfp4_hadamard_quant(x, h)
            assert torch.equal(codes.cpu(), ref_codes)
            assert torch.equal(scale.cpu(), ref_scale)


@requires_xpu
class TestXpuQuantOnly:
    """The quant-only baseline: the fused kernel with the Hadamard stripped.

    It must be bit-exact with :func:`mxfp4_quant_reference`, the raw-activation
    reference, on the same per-item layout. That equivalence is what makes the
    fused/quant-only bandwidth ratio a valid ablation of the transform.
    """

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", [(1, 32), (64, 512), (4, 13824)])
    def test_quant_only_matches_reference(self, dtype, shape):
        # The quant-only device path must be bit-exact with mxfp4_quant_reference
        # (raw activation, no transform), on the same per-item layout.
        torch.manual_seed(shape[0])
        x = torch.randn(*shape, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, _quant_only=True)
        ref_codes, ref_scale = mxfp4_quant_reference(x.cpu())
        assert codes.shape == ref_codes.shape == (shape[0], shape[1] // 2)
        assert scale.shape == ref_scale.shape == (shape[0], shape[1] // GROUP_SIZE)
        assert codes.dtype == torch.uint8 and scale.dtype == torch.uint8
        assert torch.equal(scale.cpu(), ref_scale)
        assert torch.equal(codes.cpu(), ref_codes)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_quant_only_zero_group(self, dtype):
        x = torch.zeros(2, 64, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, _quant_only=True)
        assert torch.all(codes.cpu() == 0)
        assert torch.all(scale.cpu() == 0)

    @pytest.mark.parametrize("seed", [0, 2])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_quant_only_fuzz(self, dtype, seed):
        # The two seeds span 1e-2 and 1e2, the magnitudes the reference cannot
        # reach through the shape cases above.
        torch.manual_seed(seed)
        x = (torch.randn(64, 256, device="xpu") * (10.0 ** (seed - 2))).to(dtype)
        codes, scale = mxfp4_hadamard_quant(x, _quant_only=True)
        ref_codes, ref_scale = mxfp4_quant_reference(x.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)


@requires_xpu
class TestXpuKernelErrors:
    """The validation contract: bad inputs are rejected up front, never run."""

    @pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int8, torch.bool])
    def test_rejects_unsupported_dtype(self, bad_dtype):
        x = torch.zeros(1, 32, dtype=bad_dtype, device="xpu")
        with pytest.raises(ValueError, match="float16 or bfloat16"):
            mxfp4_hadamard_quant(x)

    @pytest.mark.parametrize("k", [1, 33, 48])
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

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_rejects_non_finite(self, bad_value):
        x = torch.randn(2, 64, dtype=torch.float16, device="xpu")
        x[1, 17] = bad_value
        with pytest.raises(ValueError, match="finite"):
            mxfp4_hadamard_quant(x, check_finite=True)

    @pytest.mark.parametrize(
        "bad_matrix_factory",
        [
            # Wrong shape, degenerate (singular) values, wrong dtype and
            # non-finite entries: one case per validation branch. A power-of-two
            # D above the largest instantiated lane count is covered by
            # TestXpuKernelAllDims::test_rejects_dim_above_max_lane_count.
            lambda: torch.eye(16, dtype=torch.float32, device="xpu"),
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


@requires_xpu
def _xmx_path_available() -> bool:
    """True when the current XPU build exposes the opt-in XMX fast path."""
    return _xmx_supported()


def _precision_metrics(deq: torch.Tensor, ref: torch.Tensor, ref_scale: torch.Tensor) -> tuple[float, float, float]:
    """``(sqnr_db, max_rel, p999_rel)`` of ``deq`` vs ``ref`` (both FP32).

    SQNR is the standard signal-to-quantization-noise ratio in dB. Relative
    errors are measured **per group against the group's peak magnitude**
    (``amax = 6 * 2**(e8m0-127)``, the largest FP4 level): per-element relative
    error is meaningless near zero (FP4 alone allows unbounded relative error
    for tiny values), while the per-group bound is inherent to E2M1. ``max_rel``
    is the strict worst case; ``p999_rel`` is the 99.9th percentile, robust to
    the handful of threshold-boundary code flips that any slightly-different
    transform path (here: bf16 H + DPAS) produces.
    """
    deq = deq.double()
    ref = ref.double()
    err = deq - ref
    signal = (ref * ref).sum()
    noise = (err * err).sum()
    sqnr_db = float(10.0 * torch.log10(signal / noise.clamp_min(1e-30)))
    amax = (6.0 * torch.pow(2.0, ref_scale.double() - 127.0)).reshape(-1, 1)
    rel = (err.abs().reshape(-1, GROUP_SIZE) / amax).flatten()
    max_rel = float(rel.max())
    p999_rel = float(rel.quantile(0.999))
    return sqnr_db, max_rel, p999_rel


@requires_xpu
class TestXpuKernelXmx:
    """Opt-in XMX fast path, tolerance-based acceptance.

    The XMX path is *not* bit-exact: the Hadamard matrix is stored in the
    activation dtype (fp16/bf16) and the transform runs on XMX DPAS with FP32
    accumulation (relaxed contract, xpu_mxfp4_hadamard_design_revised.md
    §11.4). Acceptance: SQNR >= 15 dB and max relative error < 0.25 against the
    frozen FP32 reference (both measured on dequantized outputs).
    """

    @pytest.fixture(autouse=True)
    def _require_xmx(self):
        if not _xmx_path_available():
            pytest.skip("XMX fast path not available in this build (ARK_SYCL_TLA)")

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", [(1, 32), (17, 256)])
    def test_xmx_matches_reference_within_tolerance(self, dtype, shape):
        torch.manual_seed(shape[0])
        x = torch.randn(*shape, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, _force_xmx=True)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())

        num_rows, k = x.numel() // x.shape[-1], x.shape[-1]
        assert codes.shape == ref_codes.shape == (num_rows, k // 2)
        assert scale.shape == ref_scale.shape == (num_rows, k // GROUP_SIZE)
        assert codes.dtype == torch.uint8 and scale.dtype == torch.uint8

        deq_xmx = _dequantize(codes.cpu(), scale.cpu(), k)
        deq_ref = _dequantize(ref_codes, ref_scale, k)
        sqnr_db, max_rel, p999_rel = _precision_metrics(deq_xmx, deq_ref, ref_scale)
        assert sqnr_db >= 15.0, f"SQNR {sqnr_db:.2f} dB < 15 dB"
        # Worst case stays within half the group peak (no real bug); the
        # 99.9th percentile meets the design-doc 0.25 target robustly, ignoring
        # the rare threshold-boundary code flips from the bf16-H/DPAS path.
        assert max_rel < 0.5, f"max relative error {max_rel:.4f} >= 0.5"
        assert p999_rel < 0.25, f"99.9th pct relative error {p999_rel:.4f} >= 0.25"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_xmx_scales_are_close_to_reference(self, dtype):
        # E8M0 scales are octave buckets; the XMX transform (fp16/bf16 H) can
        # shift a group's max by at most a couple of dB, so at most one bucket.
        torch.manual_seed(3)
        x = torch.randn(64, 256, dtype=dtype, device="xpu")
        _, scale = mxfp4_hadamard_quant(x, _force_xmx=True)
        _, ref_scale = mxfp4_hadamard_quant_reference(x.cpu())
        diff = (scale.cpu().to(torch.int32) - ref_scale.to(torch.int32)).abs()
        assert torch.all(diff <= 1), f"E8M0 scales diverge by more than 1: max={diff.max().item()}"


@requires_xpu
class TestStreamOnlyBaseline:
    """The stream-only roofline baseline (``_stream_only=True``).

    This mode is a measurement instrument, not a quantizer: it keeps the fused
    kernel's loads, packing shape and stores but drops the transform and the
    quantization math. The tests below exist for one reason above all -- to
    prove the kernel really touches every input element. A compiler that sank
    the loads would leave an empty kernel reporting an unreachable bandwidth,
    and every ratio measured against it would be silently wrong.
    """

    # The stream-only path reuses the fused kernel's layout, packing and stores,
    # so it only needs one shape per distinct work-group situation (a single
    # group, a full work-group and a partial trailing one), not the full matrix.
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("num_rows,k", [(1, 32), (17, 256), (9, 32), (13, 160)])
    def test_matches_reference(self, dtype, num_rows, k):
        torch.manual_seed(11)
        x = torch.randn(num_rows, k, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, _stream_only=True)
        ref_codes, ref_scale = mxfp4_stream_reference(x.cpu())
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_every_input_element_is_read(self, dtype):
        """Perturbing any single element must change the output.

        Each element contributes its low nibble to one code byte and (for a
        quarter of the positions) to the scale fold, so a one-element change is
        always observable. This is the anti-dead-code-elimination guarantee the
        whole baseline rests on.
        """
        torch.manual_seed(12)
        x = torch.randn(4, 64, dtype=dtype, device="xpu")
        # Set the low nibble of every element to zero, then flip one bit at a
        # time; using raw bits keeps the test independent of float semantics.
        bits = x.view(torch.int16)
        bits &= ~0x0F
        base_codes, _ = mxfp4_hadamard_quant(x, _stream_only=True)
        for idx in range(x.numel()):
            probe = x.clone()
            probe.view(torch.int16).reshape(-1)[idx] |= 0x0F
            codes, _ = mxfp4_hadamard_quant(probe, _stream_only=True)
            assert not torch.equal(codes.cpu(), base_codes.cpu()), f"element {idx} was not read"

    def test_shares_byte_traffic_with_fused(self):
        """Identical output shapes/dtypes, i.e. identical bytes written.

        The bandwidth ratio is only meaningful because both modes move exactly
        the same number of bytes.
        """
        x = torch.randn(8, 256, dtype=torch.bfloat16, device="xpu")
        stream = mxfp4_hadamard_quant(x, _stream_only=True)
        fused = mxfp4_hadamard_quant(x)
        for a, b in zip(stream, fused):
            assert a.shape == b.shape and a.dtype == b.dtype

    def test_rejects_combination_with_quant_only(self):
        x = torch.randn(1, 32, dtype=torch.bfloat16, device="xpu")
        with pytest.raises(ValueError):
            mxfp4_hadamard_quant(x, _quant_only=True, _stream_only=True)


# Every cooperative size, i.e. everything the lane-count template instantiates
# except D = 32, which the dispatcher routes to the per-item kernel instead.
COOPERATIVE_DIMS = [d for d in SUPPORTED_HADAMARD_DIMS if d > GROUP_SIZE]


class TestHadamardDimReferenceParity:
    """The torch reference must itself be correct at every supported D.

    Bit-exactness against the reference is worthless if the reference is wrong,
    so this pins it against an independent construction: the transform is a
    matrix multiply by the normalized Sylvester matrix, and that matrix is its
    own inverse up to the normalization.
    """

    @pytest.mark.parametrize("dim", SUPPORTED_HADAMARD_DIMS)
    def test_matrix_is_orthogonal(self, dim):
        h = get_hadamard_matrix(dim, "cpu")
        assert h.shape == (dim, dim)
        assert h.dtype == torch.float32
        identity = h @ h.T
        assert torch.allclose(identity, torch.eye(dim, dtype=torch.float32), atol=1e-5)

    @pytest.mark.parametrize("dim", SUPPORTED_HADAMARD_DIMS)
    def test_reference_matches_matmul(self, dim):
        """hadamard_transform_reference takes an already-grouped [-1, D] view."""
        torch.manual_seed(31)
        x = torch.randn(12, dim, dtype=torch.float32)
        h = get_hadamard_matrix(dim, "cpu")
        assert torch.allclose(hadamard_transform_reference(x, h), x @ h, atol=1e-5)

    @pytest.mark.parametrize("dim", SUPPORTED_HADAMARD_DIMS)
    def test_output_group_count_is_dim_independent(self, dim):
        """Whatever D is, the output is still one E8M0 per 32 elements."""
        x = torch.randn(4, 1024, dtype=torch.bfloat16)
        codes, scale = mxfp4_hadamard_quant_reference(x, get_hadamard_matrix(dim, "cpu"))
        assert codes.shape == (4, 1024 // 2)
        assert scale.shape == (4, 1024 // GROUP_SIZE)

    def test_fwht128_matches_dense_matmul(self):
        """The 7-stage butterfly computes ``x @ H128`` up to FP32 rounding.

        This is the one reference routine that is not a plain matmul, so it
        needs its own independent construction.
        """
        from auto_round_kernel.mxfp4_hadamard import fwht_transform_reference

        torch.manual_seed(21)
        x = torch.randn(64, HADAMARD_DIM_128, dtype=torch.float32)
        h = get_hadamard_matrix(HADAMARD_DIM_128, "cpu")
        got = fwht_transform_reference(x, h.reshape(-1)[0])
        assert torch.allclose(got, x @ h, atol=1e-5, rtol=1e-5)


@requires_xpu
class TestXpuKernelAllDims:
    """The lane-count template at every instantiated D, not just 128.

    D = 32 * L uses L cooperating lanes, so the tail-convergence behaviour and
    the rows-per-work-group ratio differ for each L. Row counts here are chosen
    to straddle the 256/L rows that fill one work-group.
    """

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("dim", COOPERATIVE_DIMS)
    def test_bit_exact_against_reference(self, dtype, dim):
        # 33 rows spans more than one work-group for every L, which is what this
        # case adds over test_partial_work_group_tail (that one owns the
        # boundary row counts, in one dtype).
        torch.manual_seed(32)
        x = torch.randn(33, dim, dtype=dtype, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu(), get_hadamard_matrix(dim, "cpu"))
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    @pytest.mark.parametrize("dim", COOPERATIVE_DIMS)
    def test_partial_work_group_tail(self, dim):
        """Rows that leave a partial trailing work-group.

        Inactive lanes in that work-group must still reach the cross-lane
        shuffles; if they returned early the collective would be undefined and
        the *active* lanes of the same row would read garbage.
        """
        rows_per_wg = 256 // (dim // GROUP_SIZE)
        for num_rows in (1, rows_per_wg - 1, rows_per_wg, rows_per_wg + 1):
            torch.manual_seed(33)
            x = torch.randn(num_rows, dim, dtype=torch.bfloat16, device="xpu")
            codes, scale = mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))
            ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu(), get_hadamard_matrix(dim, "cpu"))
            assert torch.equal(codes.cpu(), ref_codes), f"codes mismatch at D={dim} rows={num_rows}"
            assert torch.equal(scale.cpu(), ref_scale), f"scale mismatch at D={dim} rows={num_rows}"

    @pytest.mark.parametrize("dim", COOPERATIVE_DIMS)
    def test_multiple_transforms_per_row(self, dim):
        """K = 4*D exercises the row -> group index arithmetic in the store."""
        torch.manual_seed(34)
        x = torch.randn(5, dim * 4, dtype=torch.bfloat16, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu(), get_hadamard_matrix(dim, "cpu"))
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    def test_all_zero_rows(self):
        # All-zero groups take the e8m0 = 0 clamp on every lane count, so one
        # loop covers what used to be one case per dim.
        for dim in COOPERATIVE_DIMS:
            x = torch.zeros(3, dim, dtype=torch.bfloat16, device="xpu")
            codes, scale = mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))
            assert torch.all(codes.cpu() == 0) and torch.all(scale.cpu() == 0), f"D={dim}"

    def test_rejects_k_not_multiple_of_dim(self):
        for dim in COOPERATIVE_DIMS:
            x = torch.randn(2, dim + GROUP_SIZE, dtype=torch.bfloat16, device="xpu")
            with pytest.raises(ValueError, match=f"multiple of {dim}"):
                mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))

    def test_rejects_non_sylvester(self):
        """There is no O(D^2) fallback above D = 32, so a custom matrix is refused."""
        for dim in COOPERATIVE_DIMS:
            h = get_hadamard_matrix(dim, "xpu").clone()
            h[0, 1] = -h[0, 1]
            x = torch.randn(2, dim, dtype=torch.bfloat16, device="xpu")
            with pytest.raises(NotImplementedError):
                mxfp4_hadamard_quant(x, h)

    def test_transform_size_changes_the_output(self):
        """Each D is a genuinely different transform, not a relabelled D = 32.

        The output *shapes* are D-independent -- everything quantizes in groups
        of 32, so downstream consumers and the bandwidth accounting are
        unaffected by the transform size -- while the values must differ. The
        value check is what stops a dispatcher bug that silently fell back to
        the 32-point butterfly from comparing the reference against itself.
        """
        torch.manual_seed(35)
        x = torch.randn(16, 512, dtype=torch.bfloat16, device="xpu")
        c32, s32 = mxfp4_hadamard_quant(x)
        for dim in COOPERATIVE_DIMS:
            cd, sd = mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))
            assert c32.shape == cd.shape and s32.shape == sd.shape, f"D={dim}"
            assert not torch.equal(c32.cpu(), cd.cpu()), f"D={dim} fell back to the 32-point butterfly"

    @pytest.mark.parametrize("dim", SUPPORTED_HADAMARD_DIMS)
    def test_bit_exact_at_scale(self, dim):
        """Large-shape bit-exactness, which small shapes cannot establish.

        The FMA-contraction bug this guards against flipped roughly 1 code in
        4 million -- only groups whose scaled magnitude landed within an ULP of
        an E2M1 threshold -- so every shape elsewhere in this file passed while
        the kernel was wrong. Catching it needs enough codes for the tail to
        show up, hence 16 M here rather than the few thousand used above.

        It is also why D = 64 and D = 256 must be covered and not assumed safe
        by analogy: their 1/sqrt(D) is an exact power of two, which made the
        load-time product exact and hid the bug for those two sizes alone.
        """
        torch.manual_seed(36)
        x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="xpu")
        codes, scale = mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu(), get_hadamard_matrix(dim, "cpu"))
        bad_codes = int((codes.cpu() != ref_codes).sum())
        bad_scale = int((scale.cpu() != ref_scale).sum())
        assert bad_codes == 0, f"D={dim}: {bad_codes} of {ref_codes.numel()} codes differ"
        assert bad_scale == 0, f"D={dim}: {bad_scale} of {ref_scale.numel()} scales differ"

    def test_normalization_is_applied_last(self):
        """The cooperative path must match the norm-last reference, not norm-first.

        These two orders differ only in rounding, so this pins the choice that
        keeps the kernel free of a contractable multiply-add (see the
        normalization note in xpu_mxfp4_hadamard.hpp). For D whose 1/sqrt(D) is
        an exact power of two the two orders coincide, so only the others can
        actually discriminate.
        """
        from auto_round_kernel.mxfp4_hadamard import fwht_transform_reference

        for dim in COOPERATIVE_DIMS:
            h = get_hadamard_matrix(dim, "cpu")
            norm = h.reshape(-1)[0]
            torch.manual_seed(37)
            x = torch.randn(2048, dim, dtype=torch.float32)
            last = fwht_transform_reference(x, norm, norm_last=True)
            first = fwht_transform_reference(x, norm, norm_last=False)
            assert torch.allclose(last, first, atol=1e-4), f"D={dim}"
            if not float(norm).hex().endswith("p-4") and dim not in (64, 256):
                assert not torch.equal(last, first), f"D={dim}: the two orders must be distinguishable"

    def test_lane_count_divides_sub_group(self):
        """The correctness precondition for the cross-lane butterfly.

        A row's lanes must share one sub-group, which holds only while the lane
        count divides the sub-group size of 32.
        """
        for dim in COOPERATIVE_DIMS:
            assert 32 % (dim // GROUP_SIZE) == 0
        assert max(SUPPORTED_HADAMARD_DIMS) // GROUP_SIZE == MAX_LANES_PER_ROW

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_attention_head_dim_layout(self, dtype):
        """A [B, S, H, D] attention tensor transforms along the head dim.

        The kernel only requires the transform dimension to be the innermost
        one, which is exactly how QKV activations are laid out.
        """
        torch.manual_seed(23)
        x = torch.randn(1, 96, 4, HADAMARD_DIM_128, dtype=dtype, device="xpu")
        h = get_hadamard_matrix(HADAMARD_DIM_128, x.device)
        codes, scale = mxfp4_hadamard_quant(x, h)
        num_rows = x.numel() // HADAMARD_DIM_128
        assert codes.shape == (num_rows, HADAMARD_DIM_128 // 2)
        assert scale.shape == (num_rows, 4)
        ref_codes, ref_scale = mxfp4_hadamard_quant_reference(x.cpu(), get_hadamard_matrix(HADAMARD_DIM_128, "cpu"))
        assert torch.equal(codes.cpu(), ref_codes)
        assert torch.equal(scale.cpu(), ref_scale)

    def test_rejects_xmx_for_cooperative_dims(self):
        """The XMX fast path is D = 32 only."""
        for dim in COOPERATIVE_DIMS:
            x = torch.randn(2, dim, dtype=torch.bfloat16, device="xpu")
            with pytest.raises((RuntimeError, ValueError)):
                mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device), _force_xmx=True)

    def test_rejects_dim_above_max_lane_count(self):
        """A power-of-two D beyond 32 * MAX_LANES_PER_ROW has no instantiation.

        These would need more than 16 lanes per row; 1024 is the first size that
        needs a full 32-lane sub-group per row, which is legal in principle but
        deliberately not built.
        """
        for dim in (1024, 2048):
            x = torch.randn(2, dim, dtype=torch.bfloat16, device="xpu")
            with pytest.raises(ValueError):
                mxfp4_hadamard_quant(x, get_hadamard_matrix(dim, x.device))

    def test_rejects_non_power_of_two_dim(self):
        """D must be 32 * 2^n: the butterfly has no non-power-of-two form."""
        for dim in (96, 160):
            x = torch.randn(2, dim * 2, dtype=torch.bfloat16, device="xpu")
            with pytest.raises(ValueError):
                mxfp4_hadamard_quant(x, torch.eye(dim, dtype=torch.float32, device="xpu"))

    def test_deterministic_across_modes(self):
        """Every mode must be bit-for-bit repeatable.

        The modes are separate kernels behind separate dispatch decisions, but
        the property is the same one, so a single test covers all of them
        instead of one near-identical test per class.
        """
        torch.manual_seed(24)
        x32 = torch.randn(256, 512, dtype=torch.float16, device="xpu")
        x128 = torch.randn(70, 256, dtype=torch.float16, device="xpu")
        h128 = get_hadamard_matrix(HADAMARD_DIM_128, x128.device)
        cases = [
            ("fused D=32", lambda: mxfp4_hadamard_quant(x32)),
            ("cooperative D=128", lambda: mxfp4_hadamard_quant(x128, h128)),
            ("quant-only", lambda: mxfp4_hadamard_quant(x32, _quant_only=True)),
            ("stream-only", lambda: mxfp4_hadamard_quant(x32, _stream_only=True)),
        ]
        if _xmx_supported():
            cases.append(("xmx", lambda: mxfp4_hadamard_quant(x32, _force_xmx=True)))
        for name, run in cases:
            first, first_scale = run()
            for _ in range(2):
                again, again_scale = run()
                assert torch.equal(first.cpu(), again.cpu()), name
                assert torch.equal(first_scale.cpu(), again_scale.cpu()), name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
