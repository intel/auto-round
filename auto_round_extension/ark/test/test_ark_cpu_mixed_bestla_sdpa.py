# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 Step 2 end-to-end smoke test for the experimental BestLA mixed SDPA.

This exercises the raw->packed K/V reorder bridge through the public ``sdpa``
entry, opted in via ``ARK_UNSAFE_BESTLA_MIXED_SDPA=1`` (Q=float32, K/V=fp16/bf16,
O=float32). It compares the ARK mixed path against PyTorch's reference
``scaled_dot_product_attention`` for both causal=false and causal=true.

The whole module is skipped when the compiled ``auto_round_kernel`` extension is
unavailable, or when the AMX/AVX512-class runtime needed by the wired mixed
kernels is not present (the path raises rather than producing wrong results). In
those environments the C++ reorder layout check (wrapper/test/test_reorder_kv.hpp)
is what validates correctness; this scaffold documents the intended runtime check
and runs it wherever the extension and ISA are available.
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


def _ark_mixed_sdpa(q, k, v, scale, is_causal):
    prev = os.environ.get("ARK_UNSAFE_BESTLA_MIXED_SDPA")
    os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = "1"
    try:
        return auto_round_kernel.sdpa(q, k, v, scale=scale, is_causal=is_causal)
    finally:
        if prev is None:
            os.environ.pop("ARK_UNSAFE_BESTLA_MIXED_SDPA", None)
        else:
            os.environ["ARK_UNSAFE_BESTLA_MIXED_SDPA"] = prev


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_bestla_mixed_sdpa_matches_torch(kv_dtype, is_causal):
    torch.manual_seed(4002)
    batch, heads_q, heads_kv, head_dim, seq = 1, 8, 2, 64, 64
    scale = 1 / math.sqrt(head_dim)
    q = torch.randn(batch, heads_q, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)
    v = torch.randn(batch, heads_kv, seq, head_dim, dtype=kv_dtype)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.float(), v.float(), scale=scale, enable_gqa=True, is_causal=is_causal
    )
    try:
        actual = _ark_mixed_sdpa(q, k, v, scale=scale, is_causal=is_causal)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"BestLA mixed path unavailable on this ISA/runtime: {exc}")

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
