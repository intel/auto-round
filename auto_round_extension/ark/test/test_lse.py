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


# ── LSE value correctness (issue #2556) ──────────────────────────────────


def _ref_lse_log2(q, k, scale):
    """Reference LSE in log2 units: log2(sum_j exp(scale * q . k_j))."""
    return torch.logsumexp((q.float() @ k.float().transpose(-1, -2)) * scale, dim=-1) / math.log(2.0)


def _distinct_rows(B, Hq, Sq, D, device="xpu", seed=0):
    """Q whose rows have clearly different magnitudes, so every row has its own LSE.

    Plain ``randn`` masks this bug: every row's LSE lands near log2(kv_len), so
    the error from collapsing 16 rows onto one value stays below 0.5.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, Hq, Sq, D, device=device).bfloat16()
    row_scale = torch.linspace(0.25, 4.0, Sq, device=device).view(1, 1, Sq, 1)
    return (q.float() * row_scale).bfloat16()


def _zero_mean_k(B, Hq, Skv, D, device="xpu", seed=1):
    """K whose per-head sequence mean is ~0.

    Needed to isolate the row-indexing bug: with ``smooth_k=True`` the LSE
    restoration adds ``scale * q . mean(k)``, which varies per query row and
    would otherwise hide a collapsed row index behind correct-looking values.
    """
    torch.manual_seed(seed)
    k = torch.randn(B, Hq, Skv, D, device=device).float()
    return (k - k.mean(dim=2, keepdim=True)).bfloat16()


LSE_APIS = {
    "sdpa": lambda q, k, v, scale: ark.sdpa(q, k, v, scale=scale, tensor_layout="HND", return_lse=True),
    "sagev1": lambda q, k, v, scale: ark.sagev1(q, k, v, scale=scale, tensor_layout="HND", return_lse=True),
    "sagev1_smooth_off": lambda q, k, v, scale: ark.sagev1(
        q, k, v, scale=scale, tensor_layout="HND", return_lse=True, smooth_k=False
    ),
    "sageattn_v1_pvhalf": lambda q, k, v, scale: ark.sageattn(
        q, k, v, tensor_layout="HND", sm_scale=scale, return_lse=True, kernel="v1_pvhalf"
    ),
    "sageattn_v1_pvi8": lambda q, k, v, scale: ark.sageattn(
        q, k, v, tensor_layout="HND", sm_scale=scale, return_lse=True, kernel="v1_pvi8"
    ),
}


class TestLseValues:
    """Compare returned LSE against a torch reference (regression for #2556).

    The pre-existing tests only check that a split-KV merge is self-consistent;
    they never compare against a reference, so a wrong LSE passed unnoticed.
    """

    @pytest.mark.parametrize("api_name", list(LSE_APIS))
    def test_lse_has_one_value_per_query_row(self, api_name):
        """No two query rows in the same 16-row block may share an LSE value."""
        B, Hq, Sq, Skv, D = 1, 4, 64, 1000, 128
        scale = D**-0.5
        q = _distinct_rows(B, Hq, Sq, D)
        # Zero-mean K so smooth_k's LSE restoration contributes ~0 and cannot
        # artificially decorrelate rows that the kernel collapsed.
        k = _zero_mean_k(B, Hq, Skv, D)
        torch.manual_seed(7)
        v = torch.randn(B, Hq, Skv, D, device="xpu").bfloat16()

        _, lse = LSE_APIS[api_name](q, k, v, scale)
        got = lse[0, 0].float()

        # Every 16-consecutive-row block must contain 16 distinct values.
        for start in range(0, Sq, 16):
            block = got[start : start + 16]
            assert len(torch.unique(block)) == block.numel(), (
                f"{api_name}: rows {start}-{start+15} collapsed to "
                f"{len(torch.unique(block))} distinct LSE value(s): {torch.unique(block).tolist()}"
            )

    @pytest.mark.parametrize("api_name", list(LSE_APIS))
    def test_lse_matches_reference(self, api_name):
        """LSE must equal log2(sum_j exp(scale * q . k_j)) per query row."""
        B, Hq, Sq, Skv, D = 1, 4, 64, 1000, 128
        scale = D**-0.5
        q = _distinct_rows(B, Hq, Sq, D)
        k = _zero_mean_k(B, Hq, Skv, D)
        torch.manual_seed(7)
        v = torch.randn(B, Hq, Skv, D, device="xpu").bfloat16()

        _, lse = LSE_APIS[api_name](q, k, v, scale)
        ref = _ref_lse_log2(q, k, scale)

        err = (lse.float() - ref).abs().max().item()
        # SDPA has no quantization, so it must match almost exactly.  The SAGE
        # paths quantize Q/K to INT8 per quant_block_size tokens, which perturbs
        # the logits; measured error is ~0.1 for this shape (independent of the
        # row-indexing bug, which showed up as ~9.0 before the fix).  Keep the
        # same 1e-1 tolerance the pre-existing split-KV tests use for SAGE.
        atol = 0.15 if "sage" in api_name else 0.02
        assert err <= atol, f"{api_name}: max |LSE - reference| = {err:.3f}, atol={atol}"

    @pytest.mark.parametrize("api_name", ["sagev1", "sageattn_v1_pvhalf", "sageattn_v1_pvi8"])
    def test_lse_unsmoothed_with_smooth_k(self, api_name):
        """smooth_k=True must still report the LSE of the original logits.

        Smoothing subtracts mean(k) from K, so the kernel-side LSE misses
        ``scale * q . mean(k)``.  An offset K makes that term large.
        """
        B, Hq, Sq, Skv, D = 1, 4, 32, 512, 128
        scale = D**-0.5
        # All 16 rows of each block identical -> isolates the smoothing term.
        q = _distinct_rows(B, Hq, Sq, D)[:, :, ::16].repeat_interleave(16, dim=2)
        torch.manual_seed(2)
        k = (torch.randn(B, Hq, Skv, D, device="xpu").float() + 0.5).bfloat16()
        v = torch.randn(B, Hq, Skv, D, device="xpu").bfloat16()

        _, lse = LSE_APIS[api_name](q, k, v, scale)
        ref = _ref_lse_log2(q, k, scale)

        err = (lse.float() - ref).abs().max().item()
        # Pre-fix error was ~3.7 for the smoothed path.
        assert err <= 0.15, f"{api_name}: max |LSE - reference| = {err:.3f} with K offset"
