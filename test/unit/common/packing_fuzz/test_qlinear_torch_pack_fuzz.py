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
"""Fuzzing of the GPTQ-style torch ``QuantLinear`` pack/forward contract used by ``auto_gptq`` export.

``pack`` turns (weight, scale, zero point) into ``qweight``/``qzeros``; ``forward`` must decode
exactly ``fp16(scale) * (clamp(round(W / scale + zp), 0, maxq) - zp)``. The 2/4/8-bit and the
legacy 3-bit routines must also stay bit-identical to the generic bit-stream layout, which is what
lets every width share one kernel path. ``qlinear_torch_zp`` stores GPTQ's ``zp - 1`` and, by
design, degrades an unrepresentable ``zp == 0`` to a decoded zero point of 1.
"""

from dataclasses import dataclass
from test.fuzz_helpers import FUZZ_DEVICE, assert_where, scaled_settings

import pytest
import torch
import torch.nn as nn
from hypothesis import assume, given, note
from hypothesis import strategies as st

from auto_round.utils.bit_packing import pack_bitstream, pack_scalar_zero, requires_generic_bit_packing
from auto_round_extension.torch.qlinear_torch import QuantLinear as TorchQuantLinear
from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as TorchZpQuantLinear

VARIANTS = {"torch": TorchQuantLinear, "torch_zp": TorchZpQuantLinear}
ALL_BITS = (2, 3, 4, 5, 6, 7, 8)
GENERIC_BITS = tuple(b for b in ALL_BITS if requires_generic_bit_packing(b))

_GENERIC_NO_CLAMP_REASON = (
    "fuzz-found: GenericBitPackingMixin.pack_generic_bits (5/6/7-bit) does not clamp round(W/s + zp) "
    "into [0, maxq] like pack_248_bits/pack_3bits do; pack_bitstream then masks out-of-range levels, "
    "so maxq + 1 decodes as 0 and -1 as maxq"
)
_GENERIC_ZP_ZERO_REASON = (
    "fuzz-found: qlinear_torch_zp packs 5/6/7-bit zero points as zp - 1 without the clamp used by the "
    "2/3/4/8-bit paths, so zp == 0 wraps to maxq and decodes as maxq + 1 instead of 1"
)
_G_IDX_FORWARD_REASON = (
    "fuzz-found: QuantLinear.forward with g_idx (as built by auto_gptq pack_layer) slices scales/weight "
    "columns, i.e. the output-feature axis, by the input size, so any layer with out_features > "
    "in_features fails to reshape its output"
)


@dataclass
class PackCase:
    bits: int
    group_size: int
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: object
    g_idx: bool

    @property
    def maxq(self) -> int:
        return (1 << self.bits) - 1


DEFAULT_LAYOUTS = st.tuples(st.sampled_from((32, 64, 128)), st.integers(1, 3), st.sampled_from((32, 64, 96)))
# (group_size, n_groups, out_features) with out_features > in_features.
WIDE_OUTPUT_LAYOUTS = st.integers(1, 2).flatmap(lambda n: st.tuples(st.just(32), st.just(n), st.just(32 * n + 32)))


@st.composite
def pack_cases(draw, bits, noise, min_zp=0, layouts=DEFAULT_LAYOUTS, g_idx=None):
    """Weights built around random levels; ``noise`` is the per-element offset in quantization steps."""
    group_size, n_groups, out_features = draw(layouts)
    # auto_gptq pack_layer builds the module with g_idx; inference (convert_model) builds it without.
    g_idx = draw(st.booleans()) if g_idx is None else g_idx
    sym = draw(st.booleans())
    seed = draw(st.integers(0, 2**16))
    log10_scale = draw(st.integers(-4, 1))
    note(
        f"bits={bits} group_size={group_size} n_groups={n_groups} out={out_features} g_idx={g_idx} sym={sym} "
        f"seed={seed} log10_scale={log10_scale}"
    )
    g = torch.Generator().manual_seed(seed)
    maxq = (1 << bits) - 1
    scales = (torch.rand(out_features, n_groups, generator=g) + 0.5) * 10.0**log10_scale
    # fp16-representable so the stored fp16 scale equals the scale used for packing
    scales = scales.half().float()
    if sym:
        zeros = 1 << (bits - 1)
        zero_cols = torch.full((out_features, n_groups * group_size), float(zeros))
    else:
        zeros = torch.randint(min_zp, maxq + 1, (out_features, n_groups), generator=g).float()
        zero_cols = zeros.repeat_interleave(group_size, 1)
    levels = torch.randint(0, maxq + 1, (out_features, n_groups * group_size), generator=g).float()
    offsets = noise(g, levels.shape)
    weight = (levels - zero_cols + offsets) * scales.repeat_interleave(group_size, 1)
    return PackCase(bits, group_size, weight, scales, zeros, g_idx)


def _within_half_step(g, shape):
    return torch.rand(shape, generator=g) * 0.9 - 0.45


def _up_to_two_steps(g, shape):
    return torch.randn(shape, generator=g) * 2


def _pack(variant: str, case: PackCase):
    out_features, in_features = case.weight.shape
    qlinear = VARIANTS[variant](case.bits, case.group_size, in_features, out_features, False, g_idx=case.g_idx)
    qlinear.device = torch.device("cpu")  # set by auto_gptq pack_layer before packing
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = case.weight.clone()
    zeros = case.zeros.clone() if isinstance(case.zeros, torch.Tensor) else case.zeros
    qlinear.pack(linear, case.scales.clone(), zeros, None, "cpu")
    return qlinear


def _expected_levels(case: PackCase) -> torch.Tensor:
    scale_cols = case.scales.repeat_interleave(case.group_size, 1)
    zero_cols = case.zeros.repeat_interleave(case.group_size, 1) if isinstance(case.zeros, torch.Tensor) else case.zeros
    return torch.round(case.weight / scale_cols + zero_cols).clamp(0, case.maxq)


def _decoded_zeros(variant: str, case: PackCase) -> torch.Tensor:
    zeros = case.zeros if isinstance(case.zeros, torch.Tensor) else torch.full_like(case.scales, case.zeros)
    if variant == "torch_zp":
        zeros = (zeros - 1).clamp(0, case.maxq) + 1
    return zeros


def _assert_forward_dequantizes_exactly(variant: str, case: PackCase, qlinear) -> None:
    levels = _expected_levels(case)
    zero_cols = _decoded_zeros(variant, case).repeat_interleave(case.group_size, 1)
    scale_cols = case.scales.repeat_interleave(case.group_size, 1).double()
    expected = (scale_cols * (levels - zero_cols).double()).half().double()

    qlinear = qlinear.to(FUZZ_DEVICE)
    eye = torch.eye(case.weight.shape[1], dtype=torch.float64, device=FUZZ_DEVICE)
    with torch.no_grad():
        decoded = qlinear(eye).t().cpu()

    assert_where(
        decoded == expected,
        "forward must decode fp16(scale) * (clamp(round(W/s + zp), 0, maxq) - zp)",
        decoded=decoded,
        expected=expected,
        weight=case.weight,
        level=levels,
    )


@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@given(data=st.data())
@scaled_settings(0.5)
def test_quantlinear_pack_matches_bitstream_and_forward_dequantizes(variant, bits, data):
    generic = requires_generic_bit_packing(bits)
    # The generic path is exact only for in-range levels and zp >= 1 (see the xfail tests below).
    noise = _within_half_step if generic else data.draw(st.sampled_from((_within_half_step, _up_to_two_steps)))
    case = data.draw(pack_cases(bits, noise, min_zp=1 if generic and variant == "torch_zp" else 0), label="case")
    out_features, in_features = case.weight.shape
    assume(not (case.g_idx and out_features > in_features))  # see test_g_idx_forward_handles_wide_outputs

    qlinear = _pack(variant, case)

    levels = _expected_levels(case).to(torch.int32)
    assert torch.equal(qlinear.qweight, pack_bitstream(levels, bits, dim=1).t())
    stored_zeros = _decoded_zeros(variant, case) - (1 if variant == "torch_zp" else 0)
    if isinstance(case.zeros, torch.Tensor):
        expected_qzeros = pack_bitstream(stored_zeros.t().to(torch.int32), bits, dim=1)
    else:
        num_words = case.scales.shape[0] * bits // 32
        expected_qzeros = pack_scalar_zero(int(stored_zeros[0, 0]), bits, num_words, (case.scales.shape[1], num_words))
    assert torch.equal(qlinear.qzeros, expected_qzeros)
    _assert_forward_dequantizes_exactly(variant, case, qlinear)


@pytest.mark.xfail(reason=_GENERIC_NO_CLAMP_REASON, strict=True)
@pytest.mark.parametrize("bits", GENERIC_BITS)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@given(data=st.data())
@scaled_settings(0.5)
def test_generic_bits_pack_clamps_out_of_range_levels(variant, bits, data):
    case = data.draw(pack_cases(bits, _within_half_step, min_zp=1, g_idx=False), label="case")
    # A tuned (scale, zp) can leave the original weight just past either end of the level grid.
    zero = case.zeros[0, 0] if isinstance(case.zeros, torch.Tensor) else case.zeros
    case.weight[0, 0] = (case.maxq + 1 - zero) * case.scales[0, 0]
    case.weight[0, 1] = (-1 - zero) * case.scales[0, 0]

    _assert_forward_dequantizes_exactly(variant, case, _pack(variant, case))


@pytest.mark.xfail(reason=_GENERIC_ZP_ZERO_REASON, strict=True)
@pytest.mark.parametrize("bits", GENERIC_BITS)
@given(data=st.data())
@scaled_settings(0.5)
def test_zp_generic_bits_zero_point_zero_degrades_like_legacy_widths(bits, data):
    case = data.draw(pack_cases(bits, _within_half_step, g_idx=False), label="case")
    if not isinstance(case.zeros, torch.Tensor):
        case.zeros = torch.full_like(case.scales, float(case.zeros))
    # An asymmetric group whose minimum is >= 0 gets zp == 0; keep its levels in range so that only
    # the zero point is at stake.
    case.zeros[0, 0] = 0.0
    in_range_levels = torch.arange(case.group_size, dtype=torch.float32) % (case.maxq + 1)
    case.weight[0, : case.group_size] = in_range_levels * case.scales[0, 0]

    _assert_forward_dequantizes_exactly("torch_zp", case, _pack("torch_zp", case))


@pytest.mark.xfail(reason=_G_IDX_FORWARD_REASON, strict=True)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@given(bits=st.sampled_from((2, 3, 4, 8)), data=st.data())
@scaled_settings(0.5)
def test_g_idx_forward_handles_wide_outputs(variant, bits, data):
    case = data.draw(pack_cases(bits, _within_half_step, min_zp=1, layouts=WIDE_OUTPUT_LAYOUTS, g_idx=True))

    _assert_forward_dequantizes_exactly(variant, case, _pack(variant, case))
