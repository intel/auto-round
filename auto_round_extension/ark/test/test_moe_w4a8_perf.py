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

"""Performance + accuracy benchmark for the W4A8 ARK XPU MoE kernel.

W4A8 = int4 weights, **int8 compute**. The int4 weights are converted once
into int8 with ARK's ``AUTO_S8`` re-scale (int4 ``group=32`` -> int8
``group=-1``, i.e. one scale per output channel) and the activations are
dynamically quantized to int8 per token inside the kernel, so the mainloop is
a pure ``s8 x s8 -> s32`` DPAS -- the highest-throughput shape on Xe.

This script covers **both phases**:

* **prefill** -- grouped GEMM over expert-sorted tokens
  (``ark.moe_w4a8(..., phase="prefill")``).
* **decode**  -- GEMV for one/few tokens per expert
  (``ark.moe_w4a8(..., phase="decode")``).

and reports, per shape:

* accuracy of W4A8 against (a) an fp32 reference computed from the
  dequantized int4 weights and (b) the existing W4A16 ARK kernel, as SNR
  (dB), cosine similarity and max relative error;
* latency, TFLOPS and effective weight bandwidth of W4A8 vs the W4A16 ARK
  path and vs a PyTorch matmul baseline on dequantized weights.

How to run (pytest)::

    pytest -v -s auto_round_extension/ark/test/test_moe_w4a8_perf.py

The ``-s`` flag is required to see the printed tables. By default only the
smallest batch group is run; pass ``--all-shapes`` for the full sweep::

    pytest -v -s auto_round_extension/ark/test/test_moe_w4a8_perf.py --all-shapes

It also runs standalone, without pytest::

    python auto_round_extension/ark/test/test_moe_w4a8_perf.py --all-shapes
    python auto_round_extension/ark/test/test_moe_w4a8_perf.py --phase decode
    python auto_round_extension/ark/test/test_moe_w4a8_perf.py --rescale-group-size 256

Useful environment variables (read by the kernel itself):

* ``ARK_MOE_W4A8_AUTO_S8`` -- override the AUTO_S8 re-scale block size.
  Unset / ``-1`` keeps the default (one scale per output channel).
* ``ARK_MOE_W4A8_DECODE_MAX_TOKENS`` -- token count at or below which
  ``phase="auto"`` picks the GEMV (default 128).

.. note::

   The W4A8 kernel is a new SYCL/CuTe port; this script is the intended
   on-hardware validation vehicle for it (the kernel header is marked
   ``STATUS: NEEDS-HARDWARE-VALIDATION``).
"""

import argparse
import math
import os
import sys

import torch

try:  # pytest is optional when the script is run directly
    import pytest
except ImportError:  # pragma: no cover - standalone execution without pytest
    pytest = None

import auto_round_kernel

# Reuse the pack/dequant helpers validated by the correctness tests.
from test_moe import _dequant_int4_sym, _pack_int4_sym  # noqa: E402

ark = auto_round_kernel


# ---------------------------------------------------------------------------
# Skip reasons
# ---------------------------------------------------------------------------


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _xpu_skip_reason() -> str:
    if not hasattr(torch, "xpu"):
        return "torch has no xpu submodule (need an Intel XPU build of torch)"
    if not torch.xpu.is_available():
        return "torch.xpu.is_available() == False (no XPU device or driver visible)"
    return ""


def _w4a8_skip_reason() -> str:
    """Return a non-empty string if the W4A8 MoE kernel can't be exercised."""
    reason = _xpu_skip_reason()
    if reason:
        return reason
    if ark.xpu_lib is None:
        return (
            "ark.xpu_lib is None -- the XPU extension module "
            "(auto_round_kernel_xpu) failed to import; check that auto_round_kernel "
            "was installed for THIS Python env with XPU support enabled"
        )
    for symbol in ("moe_w4a8_prepack", "moe_gemm_w4a8"):
        if not hasattr(ark.xpu_lib, symbol):
            return (
                f"ark.xpu_lib loaded but has no {symbol} symbol -- "
                "rebuild with ARK_SYCL_TLA=ON to compile the W4A8 MoE kernel"
            )
    return ""


_W4A8_SKIP = _w4a8_skip_reason()

print(
    "[moe-w4a8-perf] xpu_available=%s  xpu_lib=%s  has_moe_gemm_w4a8=%s"
    % (
        _xpu_available(),
        "loaded" if ark.xpu_lib is not None else "None",
        hasattr(ark.xpu_lib, "moe_gemm_w4a8") if ark.xpu_lib is not None else False,
    )
)
if _W4A8_SKIP:
    print("[moe-w4a8-perf] suite will SKIP. reason: %s" % _W4A8_SKIP)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS = 30


def _release_xpu_memory() -> None:
    """Free cached XPU memory and synchronize between shapes."""
    if _xpu_available():
        torch.xpu.synchronize()
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()


def _xpu_time_ms(fn, warmup: int = None, iters: int = None) -> float:
    """Time ``fn`` on XPU using device events; returns the median ms per call.

    ``warmup`` / ``iters`` default to the module-level ``WARMUP`` / ``ITERS``
    at *call* time (not at definition time) so the standalone CLI's
    ``--warmup`` / ``--iters`` flags take effect.
    """
    warmup = WARMUP if warmup is None else warmup
    iters = ITERS if iters is None else iters
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    timings.sort()
    return timings[len(timings) // 2]


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------


def _snr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB: ``10 log10(||ref||^2 / ||ref - act||^2)``.

    Higher is better. ``inf`` means bit-exact. As a rule of thumb an int8
    activation path lands around 25-40 dB against an fp32 reference; below
    ~15 dB something is structurally wrong (wrong scale block, wrong layout)
    rather than merely lossy.
    """
    ref = reference.to(torch.float32).flatten()
    act = actual.to(torch.float32).flatten()
    noise = torch.sum((ref - act) ** 2).item()
    signal = torch.sum(ref**2).item()
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal / noise)


def _cosine(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.to(torch.float32).flatten()
    act = actual.to(torch.float32).flatten()
    denom = ref.norm().item() * act.norm().item()
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(ref, act).item() / denom)


def _max_rel_err(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Max relative error, normalized by the reference magnitude scale.

    Elements are normalized by ``max(|ref_elem|, 0.01 * max|ref|)`` so that
    near-zero outputs (where the relative error is meaningless and explodes)
    do not dominate the metric.
    """
    ref = reference.to(torch.float32)
    act = actual.to(torch.float32)
    scale = ref.abs().max().item()
    if scale == 0.0:
        return 0.0
    denom = torch.clamp(ref.abs(), min=0.01 * scale)
    return float(((ref - act).abs() / denom).max().item())


# ---------------------------------------------------------------------------
# Shapes
#
# Qwen3-MoE (the shape group the int4 MoE work targets):
#   hidden_size = 2048, intermediate_size = 768, num_local_experts = 128,
#   num_experts_per_tok = 8, int4-sym weights, group_size = 32
#
#   w13 (gate/up-proj): N = 2 * 768 = 1536, K = 2048
#   w2  (down-proj)   : N = 2048,           K =  768
#
# Routed expert-token rows = ``batch * top_k``, spread round-robin over the
# 128 experts.
#
# NOTE: the W4A8 kernel requires ``K % 64 == 0``; the down-proj K of 768
# satisfies it, and the AUTO_S8 default block (= K) keeps a single full-K
# accumulation for both GEMMs.
# ---------------------------------------------------------------------------

_QWEN3_E = 128
_QWEN3_HIDDEN = 2048
_QWEN3_INTER = 768
_QWEN3_TOPK = 8
_QWEN3_GROUP_SIZE = 32

# (label, N, K) for the two grouped GEMMs of one Qwen3-MoE layer.
_QWEN3_NK = [
    ("qwen3 up  ", 2 * _QWEN3_INTER, _QWEN3_HIDDEN),
    ("qwen3 down", _QWEN3_HIDDEN, _QWEN3_INTER),
]

# Model-token batches (routed rows = batch * top_k).
_DECODE_BATCHES = [1]
_DECODE_BATCHES_EXTENDED = [1, 2, 8, 16]
_PREFILL_BATCHES = [128]
_PREFILL_BATCHES_EXTENDED = [128, 512, 2048, 8192]


def _spread_tokens(total_tokens: int, num_experts: int) -> list:
    """Distribute ``total_tokens`` across ``num_experts`` round-robin."""
    tpe = [0] * num_experts
    for i in range(total_tokens):
        tpe[i % num_experts] += 1
    return tpe


def _decode_batches(all_shapes: bool) -> list:
    return list(_DECODE_BATCHES_EXTENDED if all_shapes else _DECODE_BATCHES)


def _prefill_batches(all_shapes: bool) -> list:
    return list(_PREFILL_BATCHES_EXTENDED if all_shapes else _PREFILL_BATCHES)


# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------


def _build_case(N, K, E, total_tokens, group_size, dtype, device="xpu", seed=0):
    """Build one W4A8 MoE test case.

    Returns a dict with the packed int4 weights + scales, the activations,
    the routing histogram, and the fp32 reference output computed from the
    dequantized weights (so quantization error of the *weights* is excluded
    from the W4A8-vs-reference comparison, isolating the extra error the int8
    activation path introduces).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w_float = torch.randn(E, N, K, generator=generator, dtype=torch.float32) * 0.05
    scales = torch.empty(E, N, K // group_size, dtype=dtype)
    packed = _pack_int4_sym(w_float, scales, group_size)

    activations = (torch.randn(total_tokens, K, generator=generator, dtype=torch.float32) * 0.5).to(dtype)
    tpe = _spread_tokens(total_tokens, E)
    ntpe = torch.tensor(tpe, dtype=torch.int32)

    packed = packed.to(device)
    scales = scales.to(device)
    activations = activations.to(device)
    ntpe = ntpe.to(device)

    dequant = _dequant_int4_sym(packed, scales, group_size)

    # fp32 per-expert reference on the dequantized weights.
    reference = torch.empty(total_tokens, N, device=device, dtype=torch.float32)
    offset = 0
    for e, n_e in enumerate(tpe):
        if n_e == 0:
            continue
        a = activations[offset : offset + n_e].to(torch.float32)
        reference[offset : offset + n_e] = a @ dequant[e].to(torch.float32).t()
        offset += n_e

    return {
        "packed": packed,
        "scales": scales,
        "dequant": dequant,
        "activations": activations,
        "ntpe": ntpe,
        "tpe": tpe,
        "reference": reference,
        "N": N,
        "K": K,
        "E": E,
        "total_tokens": total_tokens,
        "group_size": group_size,
        "dtype": dtype,
    }


def _torch_baseline(case):
    """Per-expert ``A @ W.T`` on pre-dequantized weights (matmul-only cost)."""
    activations = case["activations"]
    dequant = case["dequant"]
    out = torch.empty(case["total_tokens"], case["N"], device=activations.device, dtype=activations.dtype)
    offset = 0
    for e, n_e in enumerate(case["tpe"]):
        if n_e == 0:
            continue
        out[offset : offset + n_e] = activations[offset : offset + n_e] @ dequant[e].t()
        offset += n_e
    return out


def _w4a16(case, phase):
    """The existing W4A16 ARK path (int4 weights, fp16/bf16 compute)."""
    if phase == "decode":
        return ark.moe_gemm_decode(
            case["activations"],
            case["packed"],
            case["ntpe"],
            scales=case["scales"],
            weight_bits=4,
            group_size=case["group_size"],
            asym=False,
        )
    return ark.moe_gemm_prefill(
        case["activations"],
        case["packed"],
        case["ntpe"],
        scales=case["scales"],
        weight_bits=4,
        group_size=case["group_size"],
        asym=False,
    )


def _w4a8(case, weights_s8, wscales, block, phase):
    return ark.moe_gemm_w4a8(
        case["activations"],
        weights_s8,
        wscales,
        case["ntpe"],
        rescale_block_size=block,
        phase=phase,
    )


def _flops(total_tokens, N, K) -> float:
    """MoE grouped GEMM FLOPs: each token does one ``[K] x [K, N]`` product."""
    return float(total_tokens) * N * K * 2.0


def _weight_bytes(E, N, K, bits) -> float:
    return float(E) * N * K * bits / 8.0


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_ACC_WIDTH = 150
_PERF_WIDTH = 168


def _print_acc_header(title: str) -> None:
    """Accuracy table.

    * ``SNR ref`` / ``cos ref`` / ``maxrel ref``: W4A8 against the fp32
      reference computed from the *dequantized* int4 weights. This isolates
      the error added by the int8 activation quantization and the AUTO_S8
      weight re-scale.
    * ``SNR w4a16``: W4A8 against the existing W4A16 ARK kernel -- the
      quality delta a caller sees when switching paths.
    """
    print()
    print("=" * _ACC_WIDTH)
    print(title)
    print(
        f"{'shape':<14}{'E':>5}{'N':>7}{'K':>7}{'tokens':>8}{'block':>8}"
        f"{'SNR ref(dB)':>14}{'cos ref':>12}{'maxrel ref':>13}"
        f"{'SNR w4a16(dB)':>16}{'cos w4a16':>12}{'w4a16 SNR ref':>16}"
    )
    print("-" * _ACC_WIDTH)


def _print_acc_row(label, E, N, K, tokens, block, snr_ref, cos_ref, maxrel_ref, snr_w4a16, cos_w4a16, w4a16_snr_ref):
    def _fmt(v, digits=3):
        if v is None:
            return "--"
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return "inf" if v > 0 else ("nan" if math.isnan(v) else "-inf")
        return f"{v:.{digits}f}"

    print(
        f"{label:<14}{E:>5}{N:>7}{K:>7}{tokens:>8}{block:>8}"
        f"{_fmt(snr_ref, 2):>14}{_fmt(cos_ref, 5):>12}{_fmt(maxrel_ref, 4):>13}"
        f"{_fmt(snr_w4a16, 2):>16}{_fmt(cos_w4a16, 5):>12}{_fmt(w4a16_snr_ref, 2):>16}"
    )


def _print_perf_header(title: str) -> None:
    """Perf table.

    * ``torch(ms)``: per-expert ``A @ W.T`` on pre-dequantized weights
      (matmul-only; the dequant is outside the timed region).
    * ``w4a16(ms)``: the existing ARK int4 kernel for the same phase.
    * ``w4a8(ms)`` / ``TFLOPS`` / ``W GB/s``: the new int8-compute path.
      ``W GB/s`` counts only the expert weight traffic actually touched by
      the routed tokens, which is what a memory-bound decode is limited by.
    * ``vs torch`` / ``vs w4a16``: speedups (``other / w4a8``).
    """
    print()
    print("=" * _PERF_WIDTH)
    print(title)
    print(
        f"{'shape':<14}{'E':>5}{'N':>7}{'K':>7}{'tokens':>8}"
        f"{'torch(ms)':>12}{'w4a16(ms)':>12}{'w4a8(ms)':>12}"
        f"{'TFLOPS':>10}{'W GB/s':>10}{'vs torch':>11}{'vs w4a16':>11}{'prepack(ms)':>13}"
    )
    print("-" * _PERF_WIDTH)


def _print_perf_row(label, E, N, K, tokens, torch_ms, w4a16_ms, w4a8_ms, tflops, gbps, prepack_ms):
    def _fmt(v, digits=3):
        return "--" if v is None else f"{v:.{digits}f}"

    vs_torch = None if (torch_ms is None or not w4a8_ms) else torch_ms / w4a8_ms
    vs_w4a16 = None if (w4a16_ms is None or not w4a8_ms) else w4a16_ms / w4a8_ms
    print(
        f"{label:<14}{E:>5}{N:>7}{K:>7}{tokens:>8}"
        f"{_fmt(torch_ms):>12}{_fmt(w4a16_ms):>12}{_fmt(w4a8_ms):>12}"
        f"{_fmt(tflops, 2):>10}{_fmt(gbps, 1):>10}"
        f"{(_fmt(vs_torch, 2) + 'x') if vs_torch else '--':>11}"
        f"{(_fmt(vs_w4a16, 2) + 'x') if vs_w4a16 else '--':>11}"
        f"{_fmt(prepack_ms, 2):>13}"
    )


# ---------------------------------------------------------------------------
# Core sweeps (shared by pytest and the standalone CLI)
# ---------------------------------------------------------------------------


def run_accuracy(phase, batches, dtype=torch.bfloat16, rescale_group_size=-1, verbose=True):
    """Run the W4A8 accuracy sweep. Returns a list of per-row metric dicts."""
    rows = []
    if verbose:
        _print_acc_header(
            f"W4A8 accuracy [{phase}] (E={_QWEN3_E}, group_size={_QWEN3_GROUP_SIZE}, "
            f"act={str(dtype).split('.')[-1]}, rescale_group_size={rescale_group_size}) "
            f"-- ark.moe_w4a8 vs fp32(dequant int4) and vs W4A16"
        )
    for nk_label, N, K in _QWEN3_NK:
        for batch in batches:
            total_tokens = batch * _QWEN3_TOPK
            case = _build_case(N, K, _QWEN3_E, total_tokens, _QWEN3_GROUP_SIZE, dtype)

            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"],
                case["scales"],
                group_size=_QWEN3_GROUP_SIZE,
                rescale_group_size=rescale_group_size,
            )
            out_w4a8 = _w4a8(case, weights_s8, wscales, block, phase)

            try:
                out_w4a16 = _w4a16(case, phase)
            except Exception as exc:  # pragma: no cover - depends on build
                print(f"[moe-w4a8-perf] W4A16 reference unavailable for {nk_label}: {exc}")
                out_w4a16 = None

            reference = case["reference"]
            row = {
                "label": nk_label,
                "phase": phase,
                "E": _QWEN3_E,
                "N": N,
                "K": K,
                "tokens": total_tokens,
                "block": block,
                "snr_ref": _snr_db(reference, out_w4a8),
                "cos_ref": _cosine(reference, out_w4a8),
                "maxrel_ref": _max_rel_err(reference, out_w4a8),
                "snr_w4a16": None if out_w4a16 is None else _snr_db(out_w4a16, out_w4a8),
                "cos_w4a16": None if out_w4a16 is None else _cosine(out_w4a16, out_w4a8),
                "w4a16_snr_ref": None if out_w4a16 is None else _snr_db(reference, out_w4a16),
            }
            rows.append(row)
            if verbose:
                _print_acc_row(
                    nk_label,
                    _QWEN3_E,
                    N,
                    K,
                    total_tokens,
                    block,
                    row["snr_ref"],
                    row["cos_ref"],
                    row["maxrel_ref"],
                    row["snr_w4a16"],
                    row["cos_w4a16"],
                    row["w4a16_snr_ref"],
                )

            # Drop the (large) int8 weights before the next shape allocates.
            case = weights_s8 = wscales = out_w4a8 = out_w4a16 = None
            ark.clear_moe_w4a8_prepack_cache()
            _release_xpu_memory()
    return rows


def run_perf(phase, batches, dtype=torch.bfloat16, rescale_group_size=-1, verbose=True):
    """Run the W4A8 perf sweep. Returns a list of per-row metric dicts."""
    rows = []
    if verbose:
        _print_perf_header(
            f"W4A8 perf [{phase}] (E={_QWEN3_E}, group_size={_QWEN3_GROUP_SIZE}, "
            f"act={str(dtype).split('.')[-1]}, rescale_group_size={rescale_group_size}) "
            f"-- ark.moe_gemm_w4a8 vs W4A16 vs torch"
        )
    for nk_label, N, K in _QWEN3_NK:
        for batch in batches:
            total_tokens = batch * _QWEN3_TOPK
            case = _build_case(N, K, _QWEN3_E, total_tokens, _QWEN3_GROUP_SIZE, dtype)

            # One-shot int4 -> int8 AUTO_S8 conversion. Timed separately: it
            # happens once at model load, not per forward.
            prepack_ms = _xpu_time_ms(
                lambda: ark.moe_w4a8_prepack(
                    case["packed"],
                    case["scales"],
                    group_size=_QWEN3_GROUP_SIZE,
                    rescale_group_size=rescale_group_size,
                ),
                warmup=1,
                iters=3,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"],
                case["scales"],
                group_size=_QWEN3_GROUP_SIZE,
                rescale_group_size=rescale_group_size,
            )

            w4a8_ms = _xpu_time_ms(lambda: _w4a8(case, weights_s8, wscales, block, phase))
            torch_ms = _xpu_time_ms(lambda: _torch_baseline(case))
            try:
                w4a16_ms = _xpu_time_ms(lambda: _w4a16(case, phase))
            except Exception as exc:  # pragma: no cover - depends on build
                print(f"[moe-w4a8-perf] W4A16 timing unavailable for {nk_label}: {exc}")
                w4a16_ms = None

            active_experts = sum(1 for n_e in case["tpe"] if n_e > 0)
            tflops = _flops(total_tokens, N, K) / (w4a8_ms * 1e-3) / 1e12
            # W4A8 streams int8 weights: 1 byte per element, only for the
            # experts that actually received tokens.
            gbps = _weight_bytes(active_experts, N, K, 8) / (w4a8_ms * 1e-3) / 1e9

            row = {
                "label": nk_label,
                "phase": phase,
                "E": _QWEN3_E,
                "N": N,
                "K": K,
                "tokens": total_tokens,
                "torch_ms": torch_ms,
                "w4a16_ms": w4a16_ms,
                "w4a8_ms": w4a8_ms,
                "tflops": tflops,
                "gbps": gbps,
                "prepack_ms": prepack_ms,
            }
            rows.append(row)
            if verbose:
                _print_perf_row(
                    nk_label, _QWEN3_E, N, K, total_tokens, torch_ms, w4a16_ms, w4a8_ms, tflops, gbps, prepack_ms
                )

            # Drop the (large) int8 weights before the next shape allocates.
            case = weights_s8 = wscales = None
            ark.clear_moe_w4a8_prepack_cache()
            ark.moe_w4a8_release_scratch()
            _release_xpu_memory()
    return rows


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

if pytest is not None:

    @pytest.fixture(autouse=True)
    def _xpu_cleanup_between_tests():
        """Release the XPU allocator cache and W4A8 scratch around every test.

        The prepacked int8 weights are ``E * N * K`` bytes (200+ MB for the
        Qwen3 up-proj shape), so a test that aborts mid-sweep would otherwise
        leave the allocator holding them and starve the next parametrization.
        """
        _release_xpu_memory()
        try:
            yield
        finally:
            ark.clear_moe_w4a8_prepack_cache()
            ark.moe_w4a8_release_scratch()
            _release_xpu_memory()

    @pytest.mark.skipif(bool(_W4A8_SKIP), reason=_W4A8_SKIP or "W4A8 MoE kernel unavailable")
    class TestMoEW4A8:
        """W4A8 MoE: accuracy + performance for prefill and decode."""

        @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
        def test_accuracy_decode(self, request, dtype):
            all_shapes = request.config.getoption("--all-shapes", default=False)
            rows = run_accuracy("decode", _decode_batches(all_shapes), dtype=dtype)
            _assert_accuracy(rows)

        @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
        def test_accuracy_prefill(self, request, dtype):
            all_shapes = request.config.getoption("--all-shapes", default=False)
            rows = run_accuracy("prefill", _prefill_batches(all_shapes), dtype=dtype)
            _assert_accuracy(rows)

        def test_accuracy_rescale_group_size(self, request):
            """AUTO_S8 with an explicit sub-K block must stay at least as accurate.

            A smaller re-scale block tracks the per-group int4 scales more
            closely, so it can only reduce the weight re-quantization error --
            at the cost of an accumulator fold per block in the mainloop.
            """
            all_shapes = request.config.getoption("--all-shapes", default=False)
            batches = _prefill_batches(all_shapes)[:1]
            coarse = run_accuracy("prefill", batches, rescale_group_size=-1)
            fine = run_accuracy("prefill", batches, rescale_group_size=256)
            _assert_accuracy(coarse)
            _assert_accuracy(fine)
            for c, f in zip(coarse, fine):
                assert f["block"] <= c["block"]

        def test_perf_decode(self, request):
            all_shapes = request.config.getoption("--all-shapes", default=False)
            rows = run_perf("decode", _decode_batches(all_shapes))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)

        def test_perf_prefill(self, request):
            all_shapes = request.config.getoption("--all-shapes", default=False)
            rows = run_perf("prefill", _prefill_batches(all_shapes))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)


# Minimum quality gate for the int8 activation path against an fp32 reference
# built from the same dequantized int4 weights. Per-token absmax int8
# activations lose ~7 bits of mantissa, which empirically lands well above
# 20 dB / 0.999 cosine; anything below indicates a structural bug (wrong
# scale block, transposed layout, wrong expert offset) rather than lossiness.
_MIN_SNR_DB = 20.0
_MIN_COSINE = 0.99


def _assert_accuracy(rows) -> None:
    assert rows, "no accuracy rows were produced"
    for row in rows:
        label = f"{row['label']} phase={row['phase']} N={row['N']} K={row['K']} tokens={row['tokens']}"
        assert row["snr_ref"] >= _MIN_SNR_DB, f"{label}: SNR vs fp32 reference {row['snr_ref']:.2f} dB is too low"
        assert row["cos_ref"] >= _MIN_COSINE, f"{label}: cosine vs fp32 reference {row['cos_ref']:.5f} is too low"


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="W4A8 ARK XPU MoE performance / accuracy benchmark")
    parser.add_argument(
        "--phase",
        choices=("decode", "prefill", "both"),
        default="both",
        help="Which phase(s) to benchmark (default: both).",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Sweep the full batch matrix instead of the single smallest batch.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Activation dtype (default: bf16).",
    )
    parser.add_argument(
        "--rescale-group-size",
        type=int,
        default=-1,
        help="AUTO_S8 re-scale block size; -1 (default) = one scale per output channel.",
    )
    parser.add_argument("--skip-accuracy", action="store_true", help="Only run the perf sweep.")
    parser.add_argument("--skip-perf", action="store_true", help="Only run the accuracy sweep.")
    parser.add_argument("--iters", type=int, default=ITERS, help=f"Timed iterations per measurement (default {ITERS}).")
    parser.add_argument("--warmup", type=int, default=WARMUP, help=f"Warmup iterations (default {WARMUP}).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    global WARMUP, ITERS
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if _W4A8_SKIP:
        print(f"[moe-w4a8-perf] cannot run: {_W4A8_SKIP}")
        return 1

    WARMUP = args.warmup
    ITERS = args.iters
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    phases = ("decode", "prefill") if args.phase == "both" else (args.phase,)

    env_block = os.environ.get("ARK_MOE_W4A8_AUTO_S8")
    if env_block is not None:
        print(f"[moe-w4a8-perf] ARK_MOE_W4A8_AUTO_S8={env_block} overrides --rescale-group-size")

    failures = []
    for phase in phases:
        batches = _decode_batches(args.all_shapes) if phase == "decode" else _prefill_batches(args.all_shapes)
        if not args.skip_accuracy:
            rows = run_accuracy(phase, batches, dtype=dtype, rescale_group_size=args.rescale_group_size)
            try:
                _assert_accuracy(rows)
            except AssertionError as exc:
                failures.append(str(exc))
        if not args.skip_perf:
            run_perf(phase, batches, dtype=dtype, rescale_group_size=args.rescale_group_size)

    if failures:
        print()
        print("ACCURACY FAILURES:")
        for message in failures:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
