# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UT for return_lse support across SDPA and SAGEV1 kernels.

Verifies:
1. return_lse=False  → only O returned (backward compatible).
2. return_lse=True   → (O, LSE) tuple with correct shapes.
3. LSE correctness   → LSE-merged split-KV matches full attention.
4. Zero regressions  → return_lse=False output identical to True's O.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auto_round_kernel as ark

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)

# ── helpers ──────────────────────────────────────────────────────────────


def _lse_merge(O_chunks, lse_chunks):
    """Merge N split-KV attention outputs via LSE rescaling (log2 domain)."""
    O_all = torch.stack(O_chunks, dim=0)  # [N, B, H, S, D]
    lse_all = torch.stack(lse_chunks, dim=0)  # [N, B, H, S]
    lse_max = torch.max(lse_all, dim=0).values
    weights = torch.exp2(lse_all - lse_max)  # log2 → linear via exp2
    O_weighted = (O_all * weights.unsqueeze(-1)).sum(dim=0)
    weight_sum = weights.sum(dim=0)
    O_merged = O_weighted / weight_sum.unsqueeze(-1)
    lse_merged = lse_max + torch.log2(weight_sum)
    return O_merged.to(O_chunks[0].dtype), lse_merged


def _split_kv(K, V, n_chunks):
    """Split K/V along sequence dim into n_chunks contiguous pieces."""
    B, Hkv, Skv, D = K.shape
    chunk_size = (Skv + n_chunks - 1) // n_chunks
    Kc, Vc = [], []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, Skv)
        Kc.append(K[:, :, start:end, :].contiguous())
        Vc.append(V[:, :, start:end, :].contiguous())
    return Kc, Vc


def _run_full_and_split(cfg, kernel_fn, kernel_kwargs, n_chunks=2):
    """Run full-attention (return_lse=True), split-attention merge, compare."""
    B, Hq, Hkv, Sq, Skv, D, causal = cfg
    torch.manual_seed(3030 + Skv)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, Hq, Sq, D, dtype=torch.float16, device="xpu")
    k = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="xpu")
    v = torch.randn(B, Hkv, Skv, D, dtype=torch.float16, device="xpu")

    # Full attention with LSE
    O_full, lse_full = kernel_fn(q, k, v, scale=scale, is_causal=causal, return_lse=True, **kernel_kwargs)

    # Split KV, compute each chunk, merge via LSE
    K_chunks, V_chunks = _split_kv(k, v, n_chunks)
    O_chunks, lse_chunks = [], []
    for i in range(n_chunks):
        O_c, lse_c = kernel_fn(
            q, K_chunks[i], V_chunks[i], scale=scale, is_causal=causal, return_lse=True, **kernel_kwargs
        )
        O_chunks.append(O_c)
        lse_chunks.append(lse_c)

    O_merged, lse_merged = _lse_merge(O_chunks, lse_chunks)

    return (O_full, lse_full), (O_merged, lse_merged)


# ── test data ────────────────────────────────────────────────────────────

SDPA_ONLY_KWARGS = [{}]
SAGEV1_KWARGS = [{"quant_block_size": bs} for bs in (32, 64, 128)]

NONCAUSAL_CFGS = [
    (1, 96, 8, 4096, 8192, 128, False),
    (1, 96, 8, 4096, 16384, 128, False),
    (1, 128, 8, 4096, 8192, 64, False),
    (1, 64, 8, 4096, 8192, 64, False),
    (1, 32, 8, 2048, 4096, 64, False),
    (1, 16, 4, 2048, 6144, 64, False),  # non-power-of-2 KV
]

CAUSAL_CFGS = [
    (1, 64, 8, 4096, 8192, 64, True),
    (1, 32, 8, 2048, 2048, 64, True),
]

# ── tests ────────────────────────────────────────────────────────────────


class TestReturnLseAPI:
    """Verify API contract: return_lse=False vs True."""

    @pytest.mark.parametrize(
        "kernel_name,kwargs",
        [
            ("sdpa", {}),
            ("sagev1", {"quant_block_size": 64}),
            ("sagev1_pvi8", {"quant_block_size": 64}),
        ],
    )
    def test_return_lse_false_returns_tensor(self, kernel_name, kwargs):
        """return_lse=False (default) returns a single tensor."""
        q = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="xpu")
        k = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="xpu")
        v = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="xpu")
        fn = getattr(ark, kernel_name)
        out = fn(q, k, v, scale=1 / 8.0, **kwargs)
        assert isinstance(out, torch.Tensor), f"{kernel_name} return_lse=False should return Tensor, got {type(out)}"

    @pytest.mark.parametrize(
        "kernel_name,kwargs",
        [
            ("sdpa", {}),
            ("sagev1", {"quant_block_size": 64}),
            ("sagev1_pvi8", {"quant_block_size": 64}),
        ],
    )
    def test_return_lse_true_returns_tuple(self, kernel_name, kwargs):
        """return_lse=True returns (O, LSE) tuple."""
        q = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="xpu")
        k = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="xpu")
        v = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="xpu")
        fn = getattr(ark, kernel_name)
        out = fn(q, k, v, scale=1 / 8.0, return_lse=True, **kwargs)
        assert (
            isinstance(out, tuple) and len(out) == 2
        ), f"{kernel_name} return_lse=True should return (O, LSE), got {type(out)}"
        O, LSE = out
        assert isinstance(O, torch.Tensor) and isinstance(LSE, torch.Tensor)
        assert LSE.shape == (1, 4, 128), f"LSE shape mismatch: {LSE.shape} vs (1,4,128)"

    @pytest.mark.parametrize(
        "kernel_name,kwargs",
        [
            ("sdpa", {}),
            ("sagev1", {"quant_block_size": 64}),
        ],
    )
    def test_return_lse_false_matches_true_O(self, kernel_name, kwargs):
        """O from return_lse=False must match O from return_lse=True."""
        q = torch.randn(1, 8, 256, 64, dtype=torch.float16, device="xpu")
        k = torch.randn(1, 4, 512, 64, dtype=torch.float16, device="xpu")
        v = torch.randn(1, 4, 512, 64, dtype=torch.float16, device="xpu")
        fn = getattr(ark, kernel_name)
        O_false = fn(q, k, v, scale=1 / 8.0, **kwargs)
        O_true, _ = fn(q, k, v, scale=1 / 8.0, return_lse=True, **kwargs)
        torch.testing.assert_close(O_false, O_true, atol=0, rtol=0, msg="return_lse=False O differs from True O")


class TestLseSequenceParallelMerge:
    """LSE-based merging of split-KV attention = full attention."""

    @pytest.mark.parametrize("cfg", NONCAUSAL_CFGS)
    @pytest.mark.parametrize("kwargs", SDPA_ONLY_KWARGS)
    def test_sdpa_2way(self, cfg, kwargs):
        self._check(cfg, ark.sdpa, kwargs, n_chunks=2)

    @pytest.mark.parametrize("cfg", NONCAUSAL_CFGS[:2])  # subset for speed
    @pytest.mark.parametrize("kwargs", SAGEV1_KWARGS)
    def test_sagev1_2way(self, cfg, kwargs):
        self._check(cfg, ark.sagev1, kwargs, n_chunks=2)

    @pytest.mark.parametrize("cfg", [(1, 32, 8, 2048, 4096, 64, False)])
    @pytest.mark.parametrize("kwargs", SDPA_ONLY_KWARGS)
    def test_sdpa_3way(self, cfg, kwargs):
        self._check(cfg, ark.sdpa, kwargs, n_chunks=3)

    def _check(self, cfg, fn, kwargs, n_chunks):
        """Run full + split, then assert both O and LSE match."""
        (O_full, lse_full), (O_merged, lse_merged) = _run_full_and_split(cfg, fn, kwargs, n_chunks)

        bs = kwargs.get("quant_block_size", 0)
        # SAGEV1 with split-then-merge compounds quantization noise across chunks.
        # The larger the error comes from per-chunk INT8 quantization, not from the
        # LSE merge itself. LSE is validated separately with its own tolerance.
        atol_o = 5.0e-2 if not bs else 1.6e-1
        atol_lse = 1e-1

        O_close = torch.allclose(O_merged, O_full, atol=atol_o, rtol=1e-2)
        LSE_close = torch.allclose(lse_merged, lse_full, atol=atol_lse, rtol=1e-1)

        if not O_close:
            err = (O_merged - O_full).abs().max().item()
            pytest.fail(f"O merged vs full max_err={err:.2e} atol={atol_o:.2e}")
        if not LSE_close:
            err = (lse_merged - lse_full).abs().max().item()
            pytest.fail(f"LSE merged vs full max_err={err:.2e} atol={atol_lse:.2e}")
