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

"""
Unit tests for MOE GEMM kernel.

This tests the ARK MOE GEMM kernel at the unit test level
"""

import auto_round_kernel
import pytest
import torch

ark = auto_round_kernel


def is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def has_moe_gemm():
    """Check if MOE GEMM kernel is available."""
    if ark.xpu_lib is None:
        return False
    return hasattr(ark.xpu_lib, "moe_gemm")


def has_moe_gemm_decode():
    """Check if MoE decode GEMV kernel is available."""
    if ark.xpu_lib is None:
        return False
    return hasattr(ark.xpu_lib, "moe_gemm_decode")


@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
@pytest.mark.skipif(not has_moe_gemm(), reason="MOE GEMM kernel not built (need ARK_SYCL_TLA=ON)")
class TestMoEGemm:
    """Unit tests for MOE GEMM kernel."""

    def test_moe_gemm_fp16_basic(self):
        """Test basic MOE GEMM with FP16."""
        num_experts = 8
        total_tokens = 64
        N = 256  # output features
        K = 128  # input features

        # Create test tensors
        # Note: weights layout is [num_experts, K, N] (Row major)
        activations = torch.randn(total_tokens, K, dtype=torch.float16, device="xpu")
        weights = torch.randn(num_experts, K, N, dtype=torch.float16, device="xpu")

        # Distribute tokens evenly among experts
        tokens_per_expert = [total_tokens // num_experts] * num_experts
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        # Run MOE GEMM
        output = ark.moe_gemm(activations, weights, num_tokens_per_expert)

        # Verify output shape
        assert output.shape == (total_tokens, N), f"Expected shape {(total_tokens, N)}, got {output.shape}"
        assert output.dtype == torch.float16, f"Expected dtype float16, got {output.dtype}"

        # Compute reference output using PyTorch
        # GEMM: [num_tokens, K] @ [K, N] -> [num_tokens, N]
        ref_output = torch.zeros(total_tokens, N, dtype=torch.float16, device="xpu")
        token_offset = 0
        for expert_idx in range(num_experts):
            num_tokens = tokens_per_expert[expert_idx]
            if num_tokens > 0:
                expert_activations = activations[token_offset : token_offset + num_tokens]
                expert_weight = weights[expert_idx]  # [K, N]
                # GEMM: [num_tokens, K] @ [K, N] -> [num_tokens, N]
                ref_output[token_offset : token_offset + num_tokens] = expert_activations @ expert_weight
                token_offset += num_tokens

        # Compare results (allow some tolerance for FP16)
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
        print("FP16 MOE GEMM test passed!")

    def test_moe_gemm_uneven_distribution(self):
        """Test MOE GEMM with uneven token distribution."""
        num_experts = 4
        total_tokens = 100
        N = 256
        K = 128

        activations = torch.randn(total_tokens, K, dtype=torch.float16, device="xpu")
        weights = torch.randn(num_experts, K, N, dtype=torch.float16, device="xpu")

        # Uneven distribution: 10, 30, 25, 35 tokens
        tokens_per_expert = [10, 30, 25, 35]
        assert sum(tokens_per_expert) == total_tokens
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        output = ark.moe_gemm(activations, weights, num_tokens_per_expert)

        assert output.shape == (total_tokens, N)

        # Reference computation
        ref_output = torch.zeros(total_tokens, N, dtype=torch.float16, device="xpu")
        token_offset = 0
        for expert_idx in range(num_experts):
            num_tokens = tokens_per_expert[expert_idx]
            if num_tokens > 0:
                expert_activations = activations[token_offset : token_offset + num_tokens]
                expert_weight = weights[expert_idx]  # [K, N]
                ref_output[token_offset : token_offset + num_tokens] = expert_activations @ expert_weight
                token_offset += num_tokens

        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
        print("Uneven distribution MOE GEMM test passed!")

    def test_moe_gemm_many_experts(self):
        """Test MOE GEMM with many experts (like DeepSeek models)."""
        num_experts = 64  # DeepSeek uses 64 experts
        total_tokens = 128
        N = 256
        K = 128

        activations = torch.randn(total_tokens, K, dtype=torch.float16, device="xpu")
        weights = torch.randn(num_experts, K, N, dtype=torch.float16, device="xpu")

        # Only activate some experts (sparse activation pattern)
        tokens_per_expert = [0] * num_experts
        # Activate 8 experts with 16 tokens each
        active_experts = [0, 8, 16, 24, 32, 40, 48, 56]
        for i, exp_idx in enumerate(active_experts):
            tokens_per_expert[exp_idx] = 16

        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        output = ark.moe_gemm(activations, weights, num_tokens_per_expert)

        assert output.shape == (total_tokens, N)
        print(f"Many experts ({num_experts}) MOE GEMM test passed!")

    @pytest.mark.parametrize(
        "N,K",
        [
            (256, 128),
            (512, 256),
            (1024, 512),
            (2048, 1024),
        ],
    )
    def test_moe_gemm_various_sizes(self, N, K):
        """Test MOE GEMM with various matrix sizes."""
        num_experts = 8
        total_tokens = 64

        activations = torch.randn(total_tokens, K, dtype=torch.float16, device="xpu")
        weights = torch.randn(num_experts, K, N, dtype=torch.float16, device="xpu")
        tokens_per_expert = [8] * num_experts
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        output = ark.moe_gemm(activations, weights, num_tokens_per_expert)

        assert output.shape == (total_tokens, N)
        print(f"MOE GEMM test passed for N={N}, K={K}")


# ---------------------------------------------------------------------------
# Decode-path tests (M per expert is typically 1-2, mirrors top-k routing
# after the activations have been gathered/sorted by the upper layer).
# ---------------------------------------------------------------------------


def _pack_int4_sym(w_float, scales, group_size):
    """Quantize a [E, N, K] fp tensor to symmetric int4 packed [E, N, K/2].

    scales is filled in-place with [E, N, K/group_size] values.
    """
    E, N, K = w_float.shape
    G = K // group_size
    w = w_float.reshape(E, N, G, group_size)
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = (absmax / 7.0).squeeze(-1).to(scales.dtype)
    scales.copy_(s)
    q = torch.clamp(torch.round(w / (s.to(w.dtype).unsqueeze(-1))), -8, 7).to(torch.int8)
    q = q.reshape(E, N, K)
    # Pack two nibbles per byte: low nibble at lower K, high nibble at higher K.
    q_low = q[..., 0::2] & 0x0F
    q_high = q[..., 1::2] & 0x0F
    packed = (q_low | (q_high << 4)).to(torch.uint8)
    return packed


def _pack_int4_asym(w_float, scales, zeros, group_size):
    """Quantize to asymmetric int4 (range [0, 15]); returns packed weights."""
    E, N, K = w_float.shape
    G = K // group_size
    w = w_float.reshape(E, N, G, group_size)
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    s = ((wmax - wmin) / 15.0).clamp(min=1e-8)
    z = torch.round(-wmin / s).clamp(0, 15)
    scales.copy_(s.squeeze(-1).to(scales.dtype))
    zeros.copy_(z.squeeze(-1).to(zeros.dtype))
    q = torch.clamp(torch.round(w / s + z), 0, 15).to(torch.int32)
    q = q.reshape(E, N, K)
    q_low = q[..., 0::2] & 0x0F
    q_high = q[..., 1::2] & 0x0F
    packed = (q_low | (q_high << 4)).to(torch.uint8)
    return packed


def _dequant_int4_sym(packed, scales, group_size):
    """Inverse of _pack_int4_sym. Returns [E, N, K] in scales.dtype."""
    E, N, K_half = packed.shape
    K = K_half * 2
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # Sign extend 4-bit -> 8-bit
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    q = torch.empty(E, N, K, dtype=torch.int8, device=packed.device)
    q[..., 0::2] = low
    q[..., 1::2] = high
    q = q.reshape(E, N, K // group_size, group_size).to(scales.dtype)
    return (q * scales.unsqueeze(-1)).reshape(E, N, K)


def _dequant_int4_asym(packed, scales, zeros, group_size):
    E, N, K_half = packed.shape
    K = K_half * 2
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    q = torch.empty(E, N, K, dtype=torch.int32, device=packed.device)
    q[..., 0::2] = low
    q[..., 1::2] = high
    q = q.reshape(E, N, K // group_size, group_size).to(scales.dtype)
    deq = (q - zeros.to(scales.dtype).unsqueeze(-1)) * scales.unsqueeze(-1)
    return deq.reshape(E, N, K)


# ---------------------------------------------------------------------------
# Int8 / Int2 / FP8 helpers (decode-path).
# ---------------------------------------------------------------------------


def _pack_int8_sym(w_float, scales, group_size):
    """Quantize [E, N, K] fp -> int8 (signed, [-127, 127]); fills scales in place."""
    E, N, K = w_float.shape
    G = K // group_size
    w = w_float.reshape(E, N, G, group_size)
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = (absmax / 127.0).squeeze(-1).to(scales.dtype)
    scales.copy_(s)
    q = torch.clamp(torch.round(w / s.to(w.dtype).unsqueeze(-1)), -127, 127).to(torch.int8)
    # Reinterpret as uint8 with no value change.
    return q.reshape(E, N, K).view(torch.uint8).contiguous()


def _pack_int8_asym(w_float, scales, zeros, group_size):
    """Quantize [E, N, K] fp -> uint8 ([0, 255]); fills scales/zeros in place."""
    E, N, K = w_float.shape
    G = K // group_size
    w = w_float.reshape(E, N, G, group_size)
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    s = ((wmax - wmin) / 255.0).clamp(min=1e-8)
    z = torch.round(-wmin / s).clamp(0, 255)
    scales.copy_(s.squeeze(-1).to(scales.dtype))
    zeros.copy_(z.squeeze(-1).to(zeros.dtype))
    q = torch.clamp(torch.round(w / s + z), 0, 255).to(torch.int32)
    return q.reshape(E, N, K).to(torch.uint8).contiguous()


def _dequant_int8_sym(packed_u8, scales, group_size):
    # Reinterpret uint8 bytes as signed int8.
    q = packed_u8.view(torch.int8).to(scales.dtype)
    E, N, K = q.shape
    q = q.reshape(E, N, K // group_size, group_size)
    return (q * scales.unsqueeze(-1)).reshape(E, N, K)


def _dequant_int8_asym(packed_u8, scales, zeros, group_size):
    q = packed_u8.to(torch.int32).to(scales.dtype)
    E, N, K = q.shape
    q = q.reshape(E, N, K // group_size, group_size)
    deq = (q - zeros.to(scales.dtype).unsqueeze(-1)) * scales.unsqueeze(-1)
    return deq.reshape(E, N, K)


def _pack_int2_sym(w_float, scales, group_size):
    """Quantize [E, N, K] fp -> packed int2 (signed [-2, 1]); shape [E, N, K/4].

    Packing: byte = q0 | (q1<<2) | (q2<<4) | (q3<<6), where the j-th 2-bit
    field corresponds to K index 4*i + j.
    """
    E, N, K = w_float.shape
    assert K % 4 == 0, "K must be a multiple of 4 for int2 packing"
    G = K // group_size
    w = w_float.reshape(E, N, G, group_size)
    # Symmetric int2 has signed range [-2, 1] (i.e. clip at 2 and -2 but -2 inclusive).
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = (absmax / 2.0).squeeze(-1).to(scales.dtype)
    scales.copy_(s)
    q = torch.clamp(torch.round(w / s.to(w.dtype).unsqueeze(-1)), -2, 1).to(torch.int32)
    q = q.reshape(E, N, K)
    # Pack 4 values per byte.
    q0 = q[..., 0::4] & 0x3
    q1 = q[..., 1::4] & 0x3
    q2 = q[..., 2::4] & 0x3
    q3 = q[..., 3::4] & 0x3
    packed = (q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)).to(torch.uint8)
    return packed


def _pack_int2_asym(w_float, scales, zeros, group_size):
    """Quantize [E, N, K] fp -> packed int2 (unsigned [0, 3]); shape [E, N, K/4]."""
    E, N, K = w_float.shape
    assert K % 4 == 0
    G = K // group_size
    w = w_float.reshape(E, N, G, group_size)
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    s = ((wmax - wmin) / 3.0).clamp(min=1e-8)
    z = torch.round(-wmin / s).clamp(0, 3)
    scales.copy_(s.squeeze(-1).to(scales.dtype))
    zeros.copy_(z.squeeze(-1).to(zeros.dtype))
    q = torch.clamp(torch.round(w / s + z), 0, 3).to(torch.int32)
    q = q.reshape(E, N, K)
    q0 = q[..., 0::4] & 0x3
    q1 = q[..., 1::4] & 0x3
    q2 = q[..., 2::4] & 0x3
    q3 = q[..., 3::4] & 0x3
    packed = (q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)).to(torch.uint8)
    return packed


def _dequant_int2_sym(packed, scales, group_size):
    E, N, K_q = packed.shape
    K = K_q * 4
    p = packed.to(torch.int32)
    fields = torch.empty(E, N, K, dtype=torch.int32, device=packed.device)
    fields[..., 0::4] = p & 0x3
    fields[..., 1::4] = (p >> 2) & 0x3
    fields[..., 2::4] = (p >> 4) & 0x3
    fields[..., 3::4] = (p >> 6) & 0x3
    # Sign-extend 2-bit (>=2 means negative).
    fields = torch.where(fields >= 2, fields - 4, fields).to(scales.dtype)
    fields = fields.reshape(E, N, K // group_size, group_size)
    return (fields * scales.unsqueeze(-1)).reshape(E, N, K)


def _dequant_int2_asym(packed, scales, zeros, group_size):
    E, N, K_q = packed.shape
    K = K_q * 4
    p = packed.to(torch.int32)
    fields = torch.empty(E, N, K, dtype=torch.int32, device=packed.device)
    fields[..., 0::4] = p & 0x3
    fields[..., 1::4] = (p >> 2) & 0x3
    fields[..., 2::4] = (p >> 4) & 0x3
    fields[..., 3::4] = (p >> 6) & 0x3
    fields = fields.to(scales.dtype)
    fields = fields.reshape(E, N, K // group_size, group_size)
    deq = (fields - zeros.to(scales.dtype).unsqueeze(-1)) * scales.unsqueeze(-1)
    return deq.reshape(E, N, K)


def _pack_fp8(w_float, scales, group_size, fp8_dtype):
    """Quantize [E, N, K] fp -> FP8 (e4m3fn/e5m2) with per-group scale.

    Scales are filled in-place; ``w_float`` is divided by the scale, then cast
    to ``fp8_dtype`` (rounding handled by torch). Returns the FP8 tensor.
    """
    assert fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    E, N, K = w_float.shape
    G = K // group_size
    w = w_float.reshape(E, N, G, group_size)
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    # Pick fp8 max representable magnitude for the chosen format.
    if fp8_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:  # e5m2
        fp8_max = 57344.0
    s = (absmax / fp8_max).squeeze(-1).to(scales.dtype)
    scales.copy_(s)
    scaled = (w / s.to(w.dtype).unsqueeze(-1)).reshape(E, N, K)
    # Clamp to fp8 representable range before cast to avoid Inf/NaN.
    scaled = scaled.clamp(-fp8_max, fp8_max)
    return scaled.to(fp8_dtype).contiguous()


def _dequant_fp8(packed_fp8, scales, group_size, out_dtype):
    """Reference dequant: cast fp8 -> out_dtype and multiply per-group scale."""
    E, N, K = packed_fp8.shape
    w = packed_fp8.to(out_dtype).reshape(E, N, K // group_size, group_size)
    return (w * scales.unsqueeze(-1)).reshape(E, N, K)


def _moe_decode_reference(activations, dequant_weights, num_tokens_per_expert):
    """Reference: each token is matmul'd against its routed expert's weights."""
    total_tokens, K = activations.shape
    E, N, _ = dequant_weights.shape
    out = torch.empty(total_tokens, N, dtype=activations.dtype, device=activations.device)
    offset = 0
    for e in range(E):
        n_tokens = int(num_tokens_per_expert[e].item())
        if n_tokens == 0:
            continue
        a = activations[offset : offset + n_tokens]  # [n_tokens, K]
        w = dequant_weights[e]  # [N, K]
        out[offset : offset + n_tokens] = a @ w.T
        offset += n_tokens
    return out


@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
@pytest.mark.skipif(not has_moe_gemm_decode(), reason="MoE decode GEMV kernel not built (need ARK_SYCL_TLA=ON)")
class TestMoEGemmDecode:
    """Unit tests for the MoE decode GEMV kernel.

    The activations layout follows the same convention as ``moe_gemm``: the
    upper layer has already gathered/sorted tokens per expert, so the kernel
    only needs ``num_tokens_per_expert`` (no top-k indices).
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode_fp_basic(self, dtype):
        num_experts = 4
        # One token per expert with one zero-token expert -> typical top-k=3
        # decode pattern after gather.
        tokens_per_expert = [1, 0, 1, 1]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 128

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        weights = torch.randn(num_experts, N, K, dtype=dtype, device="xpu") * 0.1
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(activations, weights, num_tokens_per_expert, weight_bits=16)

        ref = _moe_decode_reference(activations, weights, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        assert out.dtype == dtype
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("group_size", [32, 128])
    def test_decode_int4_sym(self, dtype, group_size):
        num_experts = 4
        tokens_per_expert = [1, 1, 0, 2]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int4_sym(w_float, scales, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
        )

        dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)
        ref = _moe_decode_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode_int4_asym(self, dtype):
        num_experts = 4
        group_size = 128
        tokens_per_expert = [0, 1, 2, 1]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        zeros = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int4_asym(w_float, scales, zeros, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=4,
            group_size=group_size,
            asym=True,
        )

        dequant = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)
        ref = _moe_decode_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    def test_decode_validation_errors(self):
        """Sanity-check that Python-side validation catches misuse."""
        num_experts = 2
        activations = torch.randn(2, 128, dtype=torch.float16, device="xpu")
        num_tokens_per_expert = torch.tensor([1, 1], dtype=torch.int32, device="xpu")

        # N must be a multiple of 16
        bad_weights = torch.randn(num_experts, 17, 128, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError):
            ark.moe_gemm_decode(activations, bad_weights, num_tokens_per_expert, weight_bits=16)

        # weight_bits=4 requires uint8 packed weights
        bad_packed = torch.randn(num_experts, 64, 64, dtype=torch.float16, device="xpu")
        scales = torch.empty(num_experts, 64, 1, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError):
            ark.moe_gemm_decode(
                activations, bad_packed, num_tokens_per_expert, scales=scales, weight_bits=4, group_size=128
            )

        # asym=True without zeros must error
        packed = torch.zeros(num_experts, 64, 64, dtype=torch.uint8, device="xpu")
        with pytest.raises(ValueError):
            ark.moe_gemm_decode(
                activations,
                packed,
                num_tokens_per_expert,
                scales=scales,
                weight_bits=4,
                group_size=128,
                asym=True,
            )

        # FP8 + asym is rejected
        fp8_w = torch.zeros(num_experts, 64, 128, dtype=torch.float8_e4m3fn, device="xpu")
        zeros = torch.empty(num_experts, 64, 1, dtype=torch.float16, device="xpu")
        with pytest.raises(ValueError):
            ark.moe_gemm_decode(
                activations,
                fp8_w,
                num_tokens_per_expert,
                scales=scales,
                zeros=zeros,
                group_size=128,
                asym=True,
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("group_size", [32, 128])
    def test_decode_int8_sym(self, dtype, group_size):
        num_experts = 4
        tokens_per_expert = [1, 1, 0, 2]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int8_sym(w_float, scales, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            weight_bits=8,
            group_size=group_size,
            asym=False,
        )

        dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)
        ref = _moe_decode_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode_int8_asym(self, dtype):
        num_experts = 4
        group_size = 128
        tokens_per_expert = [0, 1, 2, 1]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        zeros = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int8_asym(w_float, scales, zeros, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=8,
            group_size=group_size,
            asym=True,
        )

        dequant = _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype)
        ref = _moe_decode_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("group_size", [32, 128])
    def test_decode_int2_sym(self, dtype, group_size):
        num_experts = 4
        tokens_per_expert = [1, 1, 0, 2]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int2_sym(w_float, scales, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            weight_bits=2,
            group_size=group_size,
            asym=False,
        )

        dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)
        ref = _moe_decode_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        # Int2 has much higher quant error; relax tolerance vs int4.
        torch.testing.assert_close(out, ref, rtol=1e-1, atol=1e-1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode_int2_asym(self, dtype):
        num_experts = 4
        group_size = 128
        tokens_per_expert = [0, 1, 2, 1]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        zeros = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int2_asym(w_float, scales, zeros, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=2,
            group_size=group_size,
            asym=True,
        )

        dequant = _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype)
        ref = _moe_decode_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=1e-1, atol=1e-1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @pytest.mark.parametrize("group_size", [32, 128])
    def test_decode_fp8(self, dtype, fp8_dtype, group_size):
        num_experts = 4
        tokens_per_expert = [1, 0, 2, 1]
        total_tokens = sum(tokens_per_expert)
        N, K = 256, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_fp8(w_float, scales, group_size, fp8_dtype)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_decode(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            group_size=group_size,
            asym=False,
        )

        dequant = _dequant_fp8(packed, scales, group_size, dtype)
        ref = _moe_decode_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        # E5M2 has only 2 mantissa bits -> coarser; relax tolerance for both.
        rtol = 1e-1 if fp8_dtype == torch.float8_e5m2 else 5e-2
        atol = 1e-1 if fp8_dtype == torch.float8_e5m2 else 5e-2
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def has_moe_gemm_prefill():
    """Check if the quantized MoE prefill GEMM kernel is available."""
    if ark.xpu_lib is None:
        return False
    return hasattr(ark.xpu_lib, "moe_gemm_prefill")


@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
@pytest.mark.skipif(not has_moe_gemm_prefill(), reason="MoE prefill GEMM kernel not built (need ARK_SYCL_TLA=ON)")
class TestMoEGemmPrefill:
    """Parity tests for the quantized MoE prefill kernel.

    The prefill kernel accepts the same quantized weight encodings as
    ``moe_gemm_decode`` (decode-style ``[E, N, K_packed]``); these tests check
    that ``moe_gemm_prefill(q_weights)`` matches ``moe_gemm`` evaluated on the
    same dequantized weights, for multi-token-per-expert workloads typical of
    the prefill phase.
    """

    @staticmethod
    def _run_prefill_reference(activations, weights_NK, num_tokens_per_expert):
        """Reference: per-expert ``A @ W.T`` over ``[E, N, K]`` weights."""
        total_tokens, _ = activations.shape
        E, N, _ = weights_NK.shape
        out = torch.empty(total_tokens, N, dtype=activations.dtype, device=activations.device)
        offset = 0
        for e in range(E):
            n_tokens = int(num_tokens_per_expert[e].item())
            if n_tokens == 0:
                continue
            a = activations[offset : offset + n_tokens]  # [n_tokens, K]
            out[offset : offset + n_tokens] = a @ weights_NK[e].T
            offset += n_tokens
        return out

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_prefill_fp_basic(self, dtype):
        # Many tokens per expert (prefill regime).
        num_experts = 4
        tokens_per_expert = [16, 4, 0, 12]
        total_tokens = sum(tokens_per_expert)
        N, K = 128, 128

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        weights_NK = torch.randn(num_experts, N, K, dtype=dtype, device="xpu") * 0.1
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_prefill(activations, weights_NK, num_tokens_per_expert, weight_bits=16)

        ref = self._run_prefill_reference(activations, weights_NK, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        assert out.dtype == dtype
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("group_size", [32, 128])
    def test_prefill_int4_sym(self, dtype, group_size):
        num_experts = 4
        tokens_per_expert = [8, 8, 0, 16]
        total_tokens = sum(tokens_per_expert)
        N, K = 128, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int4_sym(w_float, scales, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_prefill(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
            asym=False,
        )

        dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)
        ref = self._run_prefill_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_prefill_int4_asym(self, dtype):
        num_experts = 4
        group_size = 128
        tokens_per_expert = [4, 8, 12, 8]
        total_tokens = sum(tokens_per_expert)
        N, K = 128, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        zeros = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_int4_asym(w_float, scales, zeros, group_size)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_prefill(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=4,
            group_size=group_size,
            asym=True,
        )

        dequant = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)
        ref = self._run_prefill_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_prefill_int8(self, dtype, asym):
        num_experts = 4
        group_size = 128
        tokens_per_expert = [8, 8, 16, 0]
        total_tokens = sum(tokens_per_expert)
        N, K = 128, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        if asym:
            zeros = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
            packed = _pack_int8_asym(w_float, scales, zeros, group_size)
            dequant = _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype)
        else:
            zeros = None
            packed = _pack_int8_sym(w_float, scales, group_size)
            dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_prefill(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=8,
            group_size=group_size,
            asym=asym,
        )

        ref = self._run_prefill_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_prefill_int2(self, dtype, asym):
        num_experts = 4
        group_size = 128
        tokens_per_expert = [4, 8, 4, 16]
        total_tokens = sum(tokens_per_expert)
        N, K = 128, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        if asym:
            zeros = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
            packed = _pack_int2_asym(w_float, scales, zeros, group_size)
            dequant = _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype)
        else:
            zeros = None
            packed = _pack_int2_sym(w_float, scales, group_size)
            dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_prefill(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=2,
            group_size=group_size,
            asym=asym,
        )

        ref = self._run_prefill_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        # Int2 quantization noise is high; relax tolerances.
        torch.testing.assert_close(out, ref, rtol=1e-1, atol=1e-1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_prefill_fp8(self, dtype, fp8_dtype):
        num_experts = 4
        group_size = 128
        tokens_per_expert = [8, 0, 12, 4]
        total_tokens = sum(tokens_per_expert)
        N, K = 128, 256

        activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
        w_float = (torch.randn(num_experts, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        scales = torch.empty(num_experts, N, K // group_size, dtype=dtype, device="xpu")
        packed = _pack_fp8(w_float, scales, group_size, fp8_dtype)
        num_tokens_per_expert = torch.tensor(tokens_per_expert, dtype=torch.int32, device="xpu")

        out = ark.moe_gemm_prefill(
            activations,
            packed,
            num_tokens_per_expert,
            scales=scales,
            group_size=group_size,
            asym=False,
        )

        dequant = _dequant_fp8(packed, scales, group_size, dtype)
        ref = self._run_prefill_reference(activations, dequant, num_tokens_per_expert)
        assert out.shape == (total_tokens, N)
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
