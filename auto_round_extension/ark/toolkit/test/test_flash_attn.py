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
Unit tests for Flash Attention kernel.

This tests the ARK Flash Attention kernel at the unit test level
"""

import math

import pytest
import torch
from ut_utils import *

ark = auto_round_kernel.ARK()


def is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def has_flash_attn():
    """Check if Flash Attention kernel is available."""
    if ark.xpu_lib is None:
        return False
    return hasattr(ark.xpu_lib, "sdpa")


def reference_attention(Q, K, V, scale, is_causal=True):
    """Reference scaled dot-product attention implementation."""
    # Q: [batch, num_heads, seq_q, head_dim]
    # K: [batch, num_heads, seq_kv, head_dim]
    # V: [batch, num_heads, seq_kv, head_dim]

    # Compute attention scores: [batch, num_heads, seq_q, seq_kv]
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q,
        K,
        V,
        scale=scale,
        attn_mask=None,
        is_causal=is_causal,
        enable_gqa=True if K.shape[1] != Q.shape[1] else False,
    )
    return ref


@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
@pytest.mark.skipif(not has_flash_attn(), reason="Flash Attention kernel not built (need ARK_SYCL_TLA=ON)")
class TestSDPA:
    """Unit tests for Flash Attention Prefill kernel."""

    def sdpa_basic(self, batch, num_heads, num_headskv, seq_len, seq_lenkv, head_dim, dtype, is_causal):

        Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device="xpu") - 0.5
        K = torch.randn(batch, num_headskv, seq_lenkv, head_dim, dtype=dtype, device="xpu") - 0.5
        V = torch.randn(batch, num_headskv, seq_lenkv, head_dim, dtype=dtype, device="xpu") - 0.5

        scale = 1.0 / math.sqrt(head_dim)

        output = ark.sdpa(Q, K, V, scale=scale, is_causal=is_causal)
        ref_output = reference_attention(Q, K, V, scale, is_causal=is_causal)
        torch.testing.assert_close(output.to(ref_output.dtype), ref_output, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("num_heads_q, num_heads_kv", [(32, 8), (64, 8), (64, 1)])
    def test_prefill_gqa(self, num_heads_q, num_heads_kv):
        """Test prefill with Grouped Query Attention (GQA)."""
        batch = 1
        seq_len = 512
        head_dim = 128
        dtype = torch.float16
        self.sdpa_basic(batch, num_heads_q, num_heads_kv, seq_len, seq_len, head_dim, dtype=dtype, is_causal=True)

    @pytest.mark.parametrize("seq_len", [33, 100, 1000])
    def test_prefill_various_seq_lengths(self, seq_len):
        """Test prefill with various sequence lengths."""
        batch = 1
        num_heads = 8
        head_dim = 64
        dtype = torch.float16
        self.sdpa_basic(batch, num_heads, num_heads, seq_len, seq_len, head_dim, dtype=dtype, is_causal=True)

    @pytest.mark.parametrize("head_dim", [64, 128, 96, 192])
    def test_prefill_various_head_dims(self, head_dim):
        """Test prefill with various head dimensions."""
        batch = 1
        num_heads = 8
        seq_len = 64
        dtype = torch.float16
        self.sdpa_basic(batch, num_heads, num_heads, seq_len, seq_len, head_dim, dtype=dtype, is_causal=True)

    def test_prefill_non_causal(self):
        """Test prefill without causal mask."""
        batch = 1
        num_heads = 8
        seq_len = 64
        head_dim = 64
        dtype = torch.float16
        self.sdpa_basic(batch, num_heads, num_heads, seq_len, seq_len, head_dim, dtype=dtype, is_causal=False)

    @pytest.mark.parametrize("head_dim", [64, 128, 96, 192])
    def test_decode_various_head_dims(self, head_dim):
        """Test decode with various head dimensions."""
        batch = 1
        num_heads = 32
        seq_len = 512
        dtype = torch.float16
        self.sdpa_basic(batch, num_heads, num_heads, 1, seq_len, head_dim, dtype=dtype, is_causal=False)

    @pytest.mark.parametrize("batch", [64, 128, 96, 192])
    def test_decode_various_batches(self, batch):
        """Test decode with various batch sizes."""
        num_heads = 32
        seq_len = 512
        head_dim = 128
        dtype = torch.float16
        self.sdpa_basic(batch, num_heads, num_heads, 1, seq_len, head_dim, dtype=dtype, is_causal=False)


if __name__ == "__main__":
    import pathlib

    test_file = pathlib.Path(__file__).resolve()
    ark_root = test_file.parent.parent
    pytest.main([str(test_file), "-v", "--confcutdir", str(test_file.parent)])
