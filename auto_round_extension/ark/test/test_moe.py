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

ark = auto_round_kernel.ARK()


def is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def has_moe_gemm():
    """Check if MOE GEMM kernel is available."""
    if ark.xpu_lib is None:
        return False
    return hasattr(ark.xpu_lib, "moe_gemm")


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
