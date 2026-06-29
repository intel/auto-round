# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 Step 3 end-to-end readiness/gating tests for the experimental BestLA
mixed SDPA path.

Two concerns are covered:

1. Gating: by default (no ``ARK_UNSAFE_BESTLA_MIXED_SDPA``) a mixed-dtype call
   (Q=float32, K/V=fp16/bf16) must NOT silently enter the BestLA mixed path; it
   must error clearly. The route is reachable only with the explicit unsafe
   opt-in.
2. Numerical smoke: with ``ARK_UNSAFE_BESTLA_MIXED_SDPA=1`` the mixed path output
   is compared against PyTorch ``scaled_dot_product_attention`` (Q float32,
   K/V fp16/bf16, O float32) for causal on/off across HND and NHD layouts, with
   separate tolerances per KV dtype.

The module is skipped when the compiled ``auto_round_kernel`` extension is not
built. Individual smoke tests skip (with the explicit ISA/runtime reason) when
the wired mixed kernels are unavailable, e.g. fp16->fp32 (NTILE24) needs AVX2 and
bf16->fp32 (NTILE48) needs AVX512F. In those environments the C++ reorder layout
check (wrapper/test/test_reorder_kv.hpp) validates correctness instead.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

auto_round_kernel = pytest.importorskip(
    "auto_round_kernel", reason="compiled ARK extension not built in this environment"
)

# Separate tolerances: bf16 has a much coarser mantissa than fp16.
_TOL = {torch.float16: (3e-2, 3e-2), torch.bfloat16: (8e-2, 8e-2)}


def _to_layout(tensor_hnd, layout):
    if layout == "HND":
        return tensor_hnd.contiguous()
    if layout == "NHD":
        return tensor_hnd.transpose(1, 2).contiguous()
    raise ValueError(layout)


def _to_hnd(tensor, layout):
    return tensor if layout == "HND" else tensor.transpose(1, 2)


def _mixed_sdpa(q, k, v, scale, is_causal, layout):
    prev = os.environ.get("ARK_UNSAFE_BESTLA_MIXED_SDPA")
    os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = "1"
    try:
        return auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=is_causal, tensor_layout=layout)
    finally:
        if prev is None:
            os.environ.pop("ARK_UNSAFE_BESTLA_MIXED_SDPA", None)
        else:
            os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = prev


def test_mixed_dtype_default_is_gated():
    # Default (no unsafe opt-in): mixed Q=fp32 / K-V=fp16 must NOT silently enter
    # the BestLA mixed path. It must raise rather than return a wrong result.
    os.environ.pop("ARK_UNSAFE_BESTLA_MIXED_SDPA", None)
    q = torch.randn(1, 8, 16, 64, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    v = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    with pytest.raises((RuntimeError, ValueError)):
        auto_round_kernel.sdpa(q, k, v, scale=1 / math.sqrt(64))


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_bestla_mixed_sdpa_matches_torch(kv_dtype, is_causal, layout):
    torch.manual_seed(4003)
    batch, heads_q, heads_kv, head_dim, seq = 1, 8, 2, 64, 64
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)

    expected_hnd = torch.nn.functional.scaled_dot_product_attention(
        q, k.float(), v.float(), scale=scale, enable_gqa=True, is_causal=is_causal
    )
    try:
        actual = _mixed_sdpa(
            _to_layout(q, layout), _to_layout(k, layout), _to_layout(v, layout), scale, is_causal, layout
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")

    atol, rtol = _TOL[kv_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(_to_hnd(actual, layout), expected_hnd, atol=atol, rtol=rtol)
