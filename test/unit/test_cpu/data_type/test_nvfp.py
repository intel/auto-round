# Copyright (c) 2024 Intel Corporation
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
"""Unit tests for ``auto_round.data_type.nvfp``."""

import math

import pytest
import torch

from auto_round.data_type.nvfp import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    FLOAT8_E4M3_MIN,
    FLOAT8_UE5M3_MAX,
    calculate_gparam,
    cast_to_fp4,
    cast_to_ue5m3,
    cast_to_ue5m3_ste,
    e5m3_to_float_tensor,
    float_to_e5m3_frexp,
    fp4_v2,
    fp4_v2_with_global_scale,
    get_reciprocal,
    nv_fp4,
    nv_fp4_with_static_gs,
    ref_fp4_quant,
    ref_nvfp4_quant,
    search_nvfp4_scale,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Validate the module-level constants exposed by ``nvfp``."""

    def test_float4_e2m1_max(self):
        assert FLOAT4_E2M1_MAX == 6.0

    def test_float8_e4m3_max(self):
        # Must match torch.finfo for float8_e4m3fn
        assert FLOAT8_E4M3_MAX == pytest.approx(448.0)

    def test_float8_e4m3_min(self):
        assert FLOAT8_E4M3_MIN == pytest.approx(-448.0)

    def test_float8_ue5m3_max(self):
        # E5M3 max value with no sign bit is 114688
        assert FLOAT8_UE5M3_MAX == 114688


# ---------------------------------------------------------------------------
# cast_to_fp4
# ---------------------------------------------------------------------------


class TestCastToFp4:
    """Test the cast_to_fp4 function (taken from vllm test_nvfp4_quant)."""

    def test_basic_quantization(self):
        """Validate the ground-truth mapping from the vLLM reference test."""
        data = torch.tensor([0.0, 0.25, 0.4, 0.75, 1.25, 1.4, 1.75, 2.5, 2.9, 3.5, 5.0, 5.1, 6.0, 6.2, 8.9])
        gt = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0])
        out = cast_to_fp4(data)
        assert torch.sum(torch.abs(out - gt)) < 1e-6

    def test_negative_values(self):
        """The cast must be sign-symmetric (negate input, negate output)."""
        data = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0])
        neg = -data
        out_neg = cast_to_fp4(neg)
        out_pos = cast_to_fp4(data)
        assert torch.allclose(out_neg, -out_pos, atol=1e-6)

    def test_clamp_to_six(self):
        """Values outside [-6, 6] must be clamped to ±6."""
        data = torch.tensor([10.0, 100.0, -50.0, 6.5])
        out = cast_to_fp4(data)
        # All values should be at most ±6 (or zero)
        assert torch.max(torch.abs(out)).item() <= 6.0 + 1e-6

    def test_zero(self):
        out = cast_to_fp4(torch.tensor([0.0]))
        assert out.item() == 0.0

    def test_2d_tensor(self):
        data = torch.tensor([[0.0, 0.5, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0]])
        out = cast_to_fp4(data)
        assert out.shape == data.shape


# ---------------------------------------------------------------------------
# get_reciprocal
# ---------------------------------------------------------------------------


class TestGetReciprocal:
    """Test get_reciprocal (tensor / float / int / invalid)."""

    def test_tensor_nonzero(self):
        x = torch.tensor([2.0, 4.0, 8.0])
        r = get_reciprocal(x)
        assert torch.allclose(r, torch.tensor([0.5, 0.25, 0.125]))

    def test_tensor_with_zero(self):
        x = torch.tensor([0.0, 2.0, 0.0, 4.0])
        r = get_reciprocal(x)
        # zeros must produce zeros (no NaN / Inf)
        assert torch.isfinite(r).all()
        assert r[0].item() == 0.0
        assert r[2].item() == 0.0
        assert r[1].item() == 0.5
        assert r[3].item() == 0.25

    def test_float(self):
        assert get_reciprocal(2.0) == 0.5
        assert get_reciprocal(4.0) == 0.25

    def test_int(self):
        assert get_reciprocal(2) == 0.5
        assert get_reciprocal(8) == 0.125

    def test_zero_float_returns_zero(self):
        assert get_reciprocal(0.0) == 0.0

    def test_zero_int_returns_zero(self):
        assert get_reciprocal(0) == 0.0

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            get_reciprocal("not_supported")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# calculate_gparam
# ---------------------------------------------------------------------------


class TestCalculateGparam:
    """Test calculate_gparam (global scaling factor)."""

    def test_tensor_input(self):
        tensor = torch.randn(32, 32, dtype=torch.float32)
        g = calculate_gparam(tensor)
        assert isinstance(g, torch.Tensor)
        assert g.dtype == torch.float32
        assert g.item() > 0

    def test_python_float_input(self):
        g = calculate_gparam(2.0)
        assert isinstance(g, torch.Tensor)
        assert g.item() > 0

    def test_group_size_assertion(self):
        tensor = torch.randn(16, 16)
        with pytest.raises(AssertionError):
            calculate_gparam(tensor, group_size=32)

    def test_value_formula(self):
        # global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor.abs().max()
        t = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        g = calculate_gparam(t)
        expected = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / 4.0
        assert g.item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# ref_nvfp4_quant
# ---------------------------------------------------------------------------


class TestRefNvfp4Quant:
    """Test ref_nvfp4_quant."""

    def _global_scale(self):
        return torch.tensor(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / 1.0, dtype=torch.float32)

    def test_basic(self):
        x = torch.randn(8, 16, dtype=torch.float32)
        gs = self._global_scale()
        q, scale = ref_nvfp4_quant(x, gs)
        assert q.shape == x.shape
        assert scale.shape == (x.shape[0], 1)

    def test_global_scale_dtype_assertion(self):
        # global_scale must be float32
        x = torch.randn(4, 16)
        gs = torch.tensor(1.0, dtype=torch.float64)
        with pytest.raises(AssertionError):
            ref_nvfp4_quant(x, gs)

    def test_ndim_assertion(self):
        x = torch.randn(16)  # 1D, must be 2D
        gs = self._global_scale()
        with pytest.raises(AssertionError):
            ref_nvfp4_quant(x, gs)

    def test_with_v(self):
        x = torch.randn(4, 16, dtype=torch.float32)
        gs = self._global_scale()
        q, scale = ref_nvfp4_quant(x, gs, v=0.5)
        assert q.shape == x.shape

    def test_with_tensor_scale_coeff(self):
        x = torch.randn(4, 16, dtype=torch.float32)
        gs = self._global_scale()
        sc = torch.ones(4, 1)
        q, scale = ref_nvfp4_quant(x, gs, scale_coeff=sc)
        assert q.shape == x.shape


# ---------------------------------------------------------------------------
# search_nvfp4_scale
# ---------------------------------------------------------------------------


class TestSearchNvfp4Scale:
    """Test search_nvfp4_scale.

    The function expects a tensor already reshaped/padded to ``(rows, 16)``.
    Internally it calls ``nv_fp4`` which reshapes any input whose last dim is
    a multiple of 16, so we pass an (8, 16) tensor.
    """

    def test_shape_and_range(self):
        tensor = torch.randn(8, 16, dtype=torch.float32)
        qw = torch.ones_like(tensor)
        scales = search_nvfp4_scale(tensor, qw=qw)
        # The function returns per-row scales (one per row in the 2-D input)
        assert scales.shape == (8, 1)
        # All scales should be in the searched range [0.5, 1.51]
        assert torch.all(scales >= 0.5 - 1e-6)
        assert torch.all(scales <= 1.52 + 1e-6)

    def test_qw_required(self):
        """``qw`` is not optional in practice — without it the function raises.

        This documents the current behaviour: the signature defaults to ``None``
        but the function unconditionally uses ``qw`` to compute the loss.
        """
        tensor = torch.randn(8, 16, dtype=torch.float32)
        with pytest.raises(TypeError):
            search_nvfp4_scale(tensor, qw=None)


# ---------------------------------------------------------------------------
# nv_fp4
# ---------------------------------------------------------------------------


class TestNvFp4:
    """Test the registered nv_fp4 quantization function."""

    def test_basic(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = nv_fp4(t)
        assert q.shape == t.shape
        assert q.dtype == t.dtype
        assert z is None

    def test_explicit_global_scale(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        gs = torch.tensor(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, dtype=torch.float32)
        q, s, z = nv_fp4(t, global_scale=gs)
        assert q.shape == t.shape

    def test_with_init_scale_tensor(self):
        # 4x32 with group_size=16 -> reshapes to 8x16, so init_scale needs 8 rows
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        is_ = torch.ones(8)
        q, s, z = nv_fp4(t, init_scale=is_)
        assert q.shape == t.shape

    def test_with_init_scale_none(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        # init_scale=None must be normalised to 1.0
        q, s, z = nv_fp4(t, init_scale=None)
        assert q.shape == t.shape

    def test_with_max_scale_tensor(self):
        # 4x32 with group_size=16 -> reshapes to 8x16, so max_scale needs 8 rows
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        ms = torch.ones(8)
        q, s, z = nv_fp4(t, max_scale=ms)
        assert q.shape == t.shape

    def test_with_max_scale_and_init_scale(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        ms = torch.ones(8) * 0.9
        is_ = torch.ones(8) * 1.1
        q, s, z = nv_fp4(t, max_scale=ms, init_scale=is_)
        assert q.shape == t.shape

    def test_float16_input(self):
        t = torch.randn(4, 32, dtype=torch.float16)
        q, s, z = nv_fp4(t)
        assert q.dtype == torch.float16

    def test_non_divisible_dim(self):
        # Column dim is not divisible by 16, must trigger padding
        t = torch.randn(4, 20, dtype=torch.bfloat16)
        q, s, z = nv_fp4(t)
        assert q.shape == t.shape


# ---------------------------------------------------------------------------
# nv_fp4_with_static_gs
# ---------------------------------------------------------------------------


class TestNvFp4WithStaticGs:
    """Test nv_fp4_with_static_gs."""

    def test_basic(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = nv_fp4_with_static_gs(t)
        assert q.shape == t.shape
        assert z is None

    def test_tensor_max_as_float(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = nv_fp4_with_static_gs(t, tensor_max=2.0)
        assert q.shape == t.shape

    def test_tensor_max_as_tensor(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        tm = torch.tensor(1.5, dtype=torch.float32)
        q, s, z = nv_fp4_with_static_gs(t, tensor_max=tm)
        assert q.shape == t.shape

    def test_tensor_max_multi_element(self):
        """If tensor_max has more than one element, only max(|.|) is used."""
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        tm = torch.tensor([1.0, 5.0, 0.5])
        q, s, z = nv_fp4_with_static_gs(t, tensor_max=tm)
        assert q.shape == t.shape

    def test_empty_tensor(self):
        t = torch.empty(0, 16, dtype=torch.bfloat16)
        q, s, z = nv_fp4_with_static_gs(t)
        assert q.shape == t.shape
        assert s is None
        assert z is None

    def test_none_tensor(self):
        q, s, z = nv_fp4_with_static_gs(None)
        assert q is None
        assert s is None
        assert z is None

    def test_float16_input(self):
        t = torch.randn(4, 32, dtype=torch.float16)
        q, s, z = nv_fp4_with_static_gs(t)
        assert q.dtype == torch.float16

    def test_with_v(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = nv_fp4_with_static_gs(t, v=0.5)
        assert q.shape == t.shape


# ---------------------------------------------------------------------------
# float_to_e5m3_frexp / e5m3_to_float_tensor (round-trip)
# ---------------------------------------------------------------------------


class TestFloatToE5m3Frexp:
    """Test float_to_e5m3_frexp."""

    def test_zero_returns_zero(self):
        x = torch.tensor([0.0])
        out = float_to_e5m3_frexp(x)
        assert out.dtype == torch.uint8
        assert out.item() == 0

    def test_normal_numbers(self):
        # 1.0 should be representable in normal form (mantissa=0.5, exp=1 -> ...)
        x = torch.tensor([1.0], dtype=torch.float32)
        out = float_to_e5m3_frexp(x)
        # Round-trip to verify
        decoded = e5m3_to_float_tensor(out)
        assert torch.allclose(decoded, x, atol=1e-3)

    def test_subnormal(self):
        # A small subnormal value: 2**-15 (below 2**-14)
        x = torch.tensor([2**-15], dtype=torch.float32)
        out = float_to_e5m3_frexp(x)
        assert out.dtype == torch.uint8

    def test_clamp_negative_to_zero(self):
        # Negative values must clamp to 0
        x = torch.tensor([-1.0, -100.0], dtype=torch.float32)
        out = float_to_e5m3_frexp(x)
        assert (out == 0).all()


class TestE5m3ToFloatTensor:
    """Test e5m3_to_float_tensor."""

    def test_zero(self):
        e = torch.tensor([0], dtype=torch.uint8)
        x = e5m3_to_float_tensor(e)
        assert x.item() == 0.0

    def test_assert_dtype(self):
        # Must assert that the input dtype is uint8
        with pytest.raises(AssertionError):
            e5m3_to_float_tensor(torch.tensor([0], dtype=torch.int32))

    def test_subnormal_decode(self):
        # Exponent 0 -> subnormal value: m/8 * 2^-14
        # m=4 -> 4/8 * 2^-14 = 2^-15
        e = torch.tensor([0x04], dtype=torch.uint8)
        x = e5m3_to_float_tensor(e)
        assert x.item() == pytest.approx(2**-15, rel=1e-5)

    def test_normal_decode(self):
        # Exponent 15 (=0x0F), mantissa 0 -> 1.0 * 2^(15-15) = 1.0
        # e5m3 byte: (e << 3) | m = (15 << 3) | 0 = 0x78
        e = torch.tensor([0x78], dtype=torch.uint8)
        x = e5m3_to_float_tensor(e)
        assert x.item() == pytest.approx(1.0, rel=1e-5)


class TestCastToUe5m3:
    """Test cast_to_ue5m3 and cast_to_ue5m3_ste (round-trip properties)."""

    def test_basic_round_trip(self):
        # Values should map to representable ue5m3 grid
        x = torch.tensor([0.0, 1.0, 100.0, 1000.0], dtype=torch.float32)
        out = cast_to_ue5m3(x)
        assert out.shape == x.shape
        # All must be finite (no NaN for normal inputs)
        assert torch.isfinite(out).all()

    def test_clamp_negative_to_zero(self):
        x = torch.tensor([-1.0, -100.0], dtype=torch.float32)
        out = cast_to_ue5m3(x)
        assert (out == 0).all()

    def test_preserves_dtype(self):
        x = torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16)
        out = cast_to_ue5m3(x)
        assert out.dtype == torch.bfloat16

    def test_ste_returns_same_shape(self):
        x = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
        out = cast_to_ue5m3_ste(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype


class TestUe5m3FrexpReference:
    """Test the e5m3 reference mapping used in __main__."""

    def test_reference_values(self):
        """Spot-check the values reported by the in-file __main__ block."""
        test = torch.tensor(
            [
                0.0,
                1e-38,
                2 ** (-17),
                (2**-14) * 0.875,
                2**-14,
                2**-13,
                2**-6,
                1e-6,
                2.7657e-05,
                0.1,
                1.0,
                3.14,
                1000.0,
                114688,
                1e10,
            ],
            dtype=torch.float32,
        )
        encoded = float_to_e5m3_frexp(test)
        decoded = e5m3_to_float_tensor(encoded)
        # All decoded values must be representable in fp32 (finite)
        assert torch.isfinite(decoded).all()
        # And dtype is uint8
        assert encoded.dtype == torch.uint8
        # 1.0 round-trip
        one_idx = (test == 1.0).nonzero(as_tuple=True)[0][0]
        assert decoded[one_idx].item() == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# ref_fp4_quant
# ---------------------------------------------------------------------------


class TestRefFp4Quant:
    """Test ref_fp4_quant."""

    def test_basic(self):
        x = torch.randn(4, 16, dtype=torch.float32)
        out, scale = ref_fp4_quant(x, global_scale=1.0)
        assert out.shape == x.shape
        assert scale.shape == (x.shape[0], 1)

    def test_ndim_assertion(self):
        x = torch.randn(16)
        with pytest.raises(AssertionError):
            ref_fp4_quant(x, global_scale=1.0)

    def test_with_v(self):
        x = torch.randn(4, 16, dtype=torch.float32)
        out, scale = ref_fp4_quant(x, global_scale=1.0, v=0.5)
        assert out.shape == x.shape

    def test_with_tensor_max_scale(self):
        # ref_fp4_quant unsqueezes max_scale to (m, 1, 1) for broadcasting, so
        # the input must be 1-D of length m.
        x = torch.randn(4, 16, dtype=torch.float32)
        out, scale = ref_fp4_quant(x, global_scale=1.0, max_scale=torch.ones(4))
        assert out.shape == x.shape

    def test_global_scale_tensor_must_be_float32(self):
        x = torch.randn(4, 16, dtype=torch.float32)
        gs = torch.tensor(1.0, dtype=torch.float64)
        with pytest.raises(AssertionError):
            ref_fp4_quant(x, gs)


# ---------------------------------------------------------------------------
# fp4_v2_with_global_scale
# ---------------------------------------------------------------------------


class TestFp4V2WithGlobalScale:
    """Test fp4_v2_with_global_scale."""

    def test_group_size_16(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = fp4_v2_with_global_scale(t, group_size=16)
        assert q.shape == t.shape
        assert z is None

    def test_group_size_32(self):
        t = torch.randn(4, 64, dtype=torch.bfloat16)
        q, s, z = fp4_v2_with_global_scale(t, group_size=32)
        assert q.shape == t.shape

    def test_invalid_group_size(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            fp4_v2_with_global_scale(t, group_size=64)

    def test_tensor_max_as_float(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = fp4_v2_with_global_scale(t, group_size=16, tensor_max=2.0)
        assert q.shape == t.shape

    def test_tensor_max_as_tensor(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        tm = torch.tensor(1.0, dtype=torch.float32)
        q, s, z = fp4_v2_with_global_scale(t, group_size=16, tensor_max=tm)
        assert q.shape == t.shape

    def test_tensor_max_multi_element(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        tm = torch.tensor([1.0, 2.0])
        q, s, z = fp4_v2_with_global_scale(t, group_size=16, tensor_max=tm)
        assert q.shape == t.shape

    def test_with_max_scale(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = fp4_v2_with_global_scale(t, group_size=16, max_scale=1.5)
        assert q.shape == t.shape

    def test_float16_input(self):
        t = torch.randn(4, 32, dtype=torch.float16)
        q, s, z = fp4_v2_with_global_scale(t, group_size=16)
        assert q.dtype == torch.float16

    def test_with_v(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = fp4_v2_with_global_scale(t, group_size=16, v=0.5)
        assert q.shape == t.shape


# ---------------------------------------------------------------------------
# fp4_v2
# ---------------------------------------------------------------------------


class TestFp4V2:
    """Test fp4_v2."""

    def test_group_size_16(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = fp4_v2(t, group_size=16)
        assert q.shape == t.shape
        assert z is None

    def test_group_size_32(self):
        t = torch.randn(4, 64, dtype=torch.bfloat16)
        q, s, z = fp4_v2(t, group_size=32)
        assert q.shape == t.shape

    def test_invalid_group_size(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            fp4_v2(t, group_size=128)

    def test_with_max_scale(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = fp4_v2(t, group_size=16, max_scale=1.5)
        assert q.shape == t.shape

    def test_with_v(self):
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, z = fp4_v2(t, group_size=16, v=0.5)
        assert q.shape == t.shape

    def test_float16_input(self):
        t = torch.randn(4, 32, dtype=torch.float16)
        q, s, z = fp4_v2(t, group_size=16)
        assert q.dtype == torch.float16

    def test_non_divisible_dim(self):
        """Last dim not divisible by group_size -> padding path."""
        t = torch.randn(4, 50, dtype=torch.bfloat16)
        # group_size=32 divides 50 with padding to 64
        q, s, z = fp4_v2(t, group_size=32)
        assert q.shape == t.shape


# ---------------------------------------------------------------------------
# Cross-check: outputs are finite / dtype preserved
# ---------------------------------------------------------------------------


class TestQuantizationProperties:
    """Cross-cutting invariants that must hold for all quant functions."""

    @pytest.mark.parametrize(
        "fn",
        [
            lambda t: nv_fp4(t),
            lambda t: nv_fp4_with_static_gs(t),
            lambda t: fp4_v2(t),
            lambda t: fp4_v2_with_global_scale(t),
        ],
    )
    def test_outputs_are_finite(self, fn):
        torch.manual_seed(0)
        t = torch.randn(4, 32, dtype=torch.bfloat16)
        q, s, _ = fn(t)
        assert torch.isfinite(q).all()
        assert s is not None

    @pytest.mark.parametrize(
        "fn",
        [
            lambda t: nv_fp4(t),
            lambda t: nv_fp4_with_static_gs(t),
            lambda t: fp4_v2(t),
            lambda t: fp4_v2_with_global_scale(t),
        ],
    )
    def test_dtype_preserved(self, fn):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            t = torch.randn(4, 32, dtype=dtype)
            q, _, _ = fn(t)
            assert q.dtype == dtype, f"dtype {dtype} not preserved by {fn}"
