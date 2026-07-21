# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch


ark = pytest.importorskip("auto_round_kernel", reason="ARK extension is not built")


def has_sage_fp8():
    return hasattr(torch, "xpu") and torch.xpu.is_available() and ark.xpu_lib is not None and hasattr(ark.xpu_lib, "sage_fp8")


pytestmark = [
    pytest.mark.skipif(not has_sage_fp8(), reason="Sage FP8 requires an ARK Xe3P XPU build"),
]


@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_quantize_fp8_is_signed_and_finite(tensor_layout):
    shape = (1, 2, 32, 64) if tensor_layout == "HND" else (1, 32, 2, 64)
    values = torch.linspace(-2.0, 2.0, math.prod(shape), dtype=torch.bfloat16, device="xpu").reshape(shape)

    quantized, scale = ark.quantize_fp8(values, tensor_layout=tensor_layout)

    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.is_contiguous()
    assert math.isfinite(scale.item()) and scale.item() > 0.0
    assert (quantized.float() < 0).any()
    assert (quantized.float() > 0).any()

    zeros, zero_scale = ark.quantize_fp8(torch.zeros_like(values), tensor_layout=tensor_layout)
    assert torch.count_nonzero(zeros.float()) == 0
    assert math.isfinite(zero_scale.item()) and zero_scale.item() > 0.0


@pytest.mark.parametrize(
    "head_dim,num_heads_q,num_heads_kv,is_causal,smooth_v",
    [(64, 4, 4, False, False), (64, 8, 2, True, True), (128, 4, 4, False, True)],
)
def test_sage_fp8_matches_bf16_sdpa(head_dim, num_heads_q, num_heads_kv, is_causal, smooth_v):
    torch.manual_seed(42)
    batch, seq_len = 1, 256
    query = torch.randn(batch, num_heads_q, seq_len, head_dim, dtype=torch.bfloat16, device="xpu") * 0.5
    key = torch.randn(batch, num_heads_kv, seq_len, head_dim, dtype=torch.bfloat16, device="xpu") * 0.5
    value = torch.randn(batch, num_heads_kv, seq_len, head_dim, dtype=torch.bfloat16, device="xpu") * 0.5
    softmax_scale = 1.0 / math.sqrt(head_dim)

    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=softmax_scale,
        is_causal=is_causal,
        enable_gqa=num_heads_q != num_heads_kv,
    )
    actual = ark.sage_fp8(
        query,
        key,
        value,
        scale=softmax_scale,
        is_causal=is_causal,
        enable_gqa=num_heads_q != num_heads_kv,
        smooth_v=smooth_v,
    )

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08)


def test_sage_fp8_fused_vmean_matches_raw_plus_mean():
    torch.manual_seed(7)
    query = torch.randn(1, 4, 1, 64, dtype=torch.bfloat16, device="xpu")
    key = torch.randn(1, 2, 256, 64, dtype=torch.bfloat16, device="xpu")
    value = torch.randn_like(key)
    key_smoothed, _ = ark._smooth_sequence_mean(key)
    value_smoothed, value_mean = ark._smooth_sequence_mean(value)
    query_fp8, qscale = ark.quantize_fp8(query)
    key_fp8, kscale = ark.quantize_fp8(key_smoothed)
    value_fp8, vscale = ark.quantize_fp8(value_smoothed)
    kwargs = dict(qscale=qscale, kscale=kscale, vscale=vscale, scale=1.0 / 8.0)

    raw = ark.fp8_fa2(query_fp8, key_fp8, value_fp8, **kwargs)
    fused = ark.fp8_fa2(query_fp8, key_fp8, value_fp8, vmean=value_mean, **kwargs)

    torch.testing.assert_close(fused.float(), raw.float() + value_mean.unsqueeze(2), rtol=0.02, atol=0.02)


def test_sage_fp8_matches_sagev1_output():
    torch.manual_seed(11)
    query = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="xpu")
    key = torch.randn(1, 2, 128, 64, dtype=torch.bfloat16, device="xpu")
    value = torch.randn_like(key)

    fp8_output = ark.sage_fp8(query, key, value, enable_gqa=True)
    sagev1_output = ark.sagev1(query, key, value, enable_gqa=True, smooth_k=True)

    torch.testing.assert_close(fp8_output, sagev1_output, rtol=0.12, atol=0.12)
