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
    python auto_round_extension/ark/test/test_moe_w4a8_perf.py --phase prefill --long-seq

The prefill throughput points are two: ``--compute-bound`` derives, per model,
the batch that puts 384 rows on every expert (the smallest routing where the
100 TFLOPS goal is reachable at all), and ``--long-seq`` runs a fixed 8K-token
prompt, whose 65536 routed rows divide into 512 rows per expert for Qwen3-MoE
and 341 for MiniMax -- a higher intensity than the compute-bound batch for one
model and a lower one for the other.

Useful environment variables (read by the kernel itself):

* ``ARK_MOE_W4A8_AUTO_S8`` -- override the AUTO_S8 re-scale block size.
  Unset / ``-1`` keeps the default (one scale per output channel).
* ``ARK_MOE_W4A8_DECODE_MAX_TOKENS`` -- token count at or below which
  ``phase="auto"`` picks the GEMV (default 128).

The dispatch knobs the sweeps here drive -- ``ARK_MOE_W4A8_PREFILL_TILE``,
``ARK_MOE_W4A8_ACT_QUANT_VEC``, ``ARK_MOE_W4A8_ACT_QUANT_UNROLL``,
``ARK_MOE_W4A8_PREFILL_FULL_TILE`` and the ``ARK_MOE_W4A8_DECODE_KSPLIT*``
family -- are documented in ``README_MOE_W4A8.md``; each sweep sets them itself
and restores the environment afterwards.

.. note::

   The W4A8 kernel is a new SYCL/CuTe port; this script is the intended
   on-hardware validation vehicle for it (the kernel header is marked
   ``STATUS: PARTIALLY HARDWARE-VALIDATED`` and names the paths that still
   need a device run).
"""

import argparse
import contextlib
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
# Rounds the config sweeps round-robin over their configurations. Three is
# enough to separate a real ranking from clock drift without making the sweep
# slower: the per-round iteration count is ITERS // SWEEP_ROUNDS, so the total
# number of timed iterations per configuration is unchanged.
SWEEP_ROUNDS = 3


def _release_xpu_memory() -> None:
    """Free cached XPU memory and synchronize between shapes."""
    if _xpu_available():
        torch.xpu.synchronize()
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()


@contextlib.contextmanager
def _env_override(**overrides):
    """Temporarily set environment variables, restoring the previous values.

    The kernel re-reads its dispatch flags on every call (they are never
    cached in a static), so an in-process override is enough to A/B two code
    paths without reloading the extension. A ``None`` value unsets the variable
    for the duration of the block, which is how a sweep expresses "kernel
    default".
    """
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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


_ULP_INT_DTYPE = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
}


def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise distance between two tensors counted in representable steps.

    ``1`` means the two values are neighbours in the format -- the smallest
    disagreement that can be expressed -- and ``0`` means identical. This is
    the right unit for comparing two computations that are algebraically the
    same but not required to round identically: a tolerance in absolute or
    relative terms has to be picked to fit the magnitudes at hand, while "one
    step apart" is a property of the format and holds across the whole range.

    Floats are ordered by their bit pattern within a sign, so the pattern read
    as an integer counts steps directly; the negatives run backwards, so their
    magnitude is negated to get one monotone key over the whole line (which
    also makes ``-0`` and ``+0`` the same key).
    """
    if a.dtype != b.dtype:
        raise TypeError(f"_ulp_diff needs one dtype, got {a.dtype} and {b.dtype}")
    int_dtype = _ULP_INT_DTYPE.get(a.dtype)
    if int_dtype is None:
        raise TypeError(f"_ulp_diff does not know the bit layout of {a.dtype}")
    magnitude = torch.iinfo(int_dtype).max  # 0x7fff / 0x7fffffff: everything but the sign

    def key(t: torch.Tensor) -> torch.Tensor:
        bits = t.contiguous().view(int_dtype).to(torch.int64)
        return torch.where(bits < 0, -(bits & magnitude), bits)

    return (key(a) - key(b)).abs()


def _ulp_report(a: torch.Tensor, b: torch.Tensor) -> str:
    """``max ULP`` plus how much of the tensor moved, for assertion messages.

    The SNR is printed alongside because the two numbers fail differently: a
    contract that reads the wrong scale or the wrong row misses by orders of
    magnitude and takes the SNR down with it, while a difference confined to
    the last bits leaves the SNR high however large the worst ULP distance is
    (an output that cancels to near zero is many steps from its neighbour at
    no cost in energy).
    """
    ulp = _ulp_diff(a, b)
    differing = int((ulp > 0).sum().item())
    total = ulp.numel()
    return (
        f"max {int(ulp.max().item())} ULP, {differing}/{total} elements differ "
        f"({differing / total:.3%}), max |diff| {(a.float() - b.float()).abs().max().item():.6g}, "
        f"SNR {_snr_db(a, b):.2f} dB"
    )


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

# ---------------------------------------------------------------------------
# MiniMax-M2 shapes, matching ``test_moe_prefill_perf.py``:
#   hidden_size = 3072, intermediate_size = 1536, num_local_experts = 192,
#   num_experts_per_tok = 8.
#
# A second shape group matters here because both perf targets are shape
# dependent: with 192 experts a given batch spreads over 1.5x more experts
# (fewer rows per expert, so a *lower* compute ceiling at the same batch), while
# K doubles for the up-proj (3072 vs 2048), which lengthens the decode GEMV's
# sequential stream and gives the prefill tile more K per tile-load. Routing is
# the same round-robin spread used for Qwen3; the heavy-tailed empirical
# distribution lives in ``test_moe_prefill_perf.py``'s ``minimax real`` rows.
# ---------------------------------------------------------------------------
_MINIMAX_E = 192
_MINIMAX_HIDDEN = 3072
_MINIMAX_INTER = 1536
_MINIMAX_TOPK = 8
_MINIMAX_NK = [
    ("minimax up  ", _MINIMAX_INTER, _MINIMAX_HIDDEN),
    ("minimax down", _MINIMAX_HIDDEN, _MINIMAX_INTER),
]

# A "model" is the (experts, top-k, group size, [(label, N, K)]) tuple the
# sweeps iterate over. Everything downstream reads the per-row ``E``/``topk``,
# so adding a group here is enough to get it benchmarked and target-checked.
_MODELS = {
    "qwen3": {"E": _QWEN3_E, "topk": _QWEN3_TOPK, "group_size": _QWEN3_GROUP_SIZE, "nk": _QWEN3_NK},
    "minimax": {"E": _MINIMAX_E, "topk": _MINIMAX_TOPK, "group_size": _QWEN3_GROUP_SIZE, "nk": _MINIMAX_NK},
}
_DEFAULT_MODELS = ["qwen3"]


def _models(names) -> list:
    """Resolve model names (or ``"all"``) to ``(name, spec)`` pairs."""
    if names is None:
        names = _DEFAULT_MODELS
    if isinstance(names, str):
        names = _MODELS.keys() if names == "all" else [names]
    return [(n, _MODELS[n]) for n in names]


def _models_option(request):
    """Read the ``--models`` pytest option (absent under a foreign conftest)."""
    value = request.config.getoption("--models", default=None) if request is not None else None
    if not value:
        return None
    if value == "all":
        return "all"
    return [name.strip() for name in value.split(",") if name.strip()]


# Model-token batches (routed rows = batch * top_k).
_DECODE_BATCHES = [1]
_DECODE_BATCHES_EXTENDED = [1, 2, 8, 16]
_PREFILL_BATCHES = [128]
_PREFILL_BATCHES_EXTENDED = [128, 512, 2048, 8192]
# Rows per expert at which the prefill TFLOPS target is physically reachable.
#
# This was 256, chosen from the weights-only roofline (``2 * rows/E *
# weight_bandwidth``, i.e. "256 rows/expert needs only ~195 GB/s"). Counting
# every stream the call moves (see ``_traffic_bytes``) puts the real
# requirement at 374-423 GB/s, and 423 GB/s -- the qwen3 down-projection, the
# shape with the smallest K and therefore the largest non-weight share -- is
# past what a 456 GB/s part delivers: its ceiling at 256 rows/expert is 94
# TFLOPS, so the target was *unreachable at the batch it was being measured
# at*, whatever the kernel did.
#
# 384 is the smallest round value that clears 100 TFLOPS on every shipped shape
# with margin (ceilings 112 / 130 / 137 / 154 TFLOPS at a 400 GB/s probe;
# qwen3 down alone needs >= 290). The batch is derived per model
# (``rows/E * E / top_k``): 6144 model tokens for Qwen3-MoE (128 experts),
# 9216 for MiniMax (192).
_PREFILL_TARGET_ROWS_PER_EXPERT = 384

# A second prefill point, sized like a real prompt instead of by a target
# rows/expert: one 8K-token sequence, i.e. 65536 routed rows -- the same 8K
# group ``test_moe_prefill_perf.py`` sweeps.
#
# It is not a larger copy of the compute-bound batch. That batch is *derived*
# per model so both land on 384 rows/expert; a fixed token count is divided by
# a different expert count in each model, so the same 8K prompt puts 512 rows
# on each of Qwen3-MoE's 128 experts and 341 on each of MiniMax's 192. The
# arithmetic intensity therefore moves in opposite directions, and the roofline
# with it: at a 400 GB/s probe the ceilings go 129 -> 145 and 112 -> 123 TFLOPS
# for qwen3 up/down, and 137 -> 129 and 154 -> 145 for minimax up/down.
#
# 512 rows/expert also moves the *tile ladder*, which is why this point is
# worth a sweep of its own rather than one more perf row: it is the only
# routing in the suite at which the 256-row tile does not pad
# (``ceil(M/256)*256 == ceil(M/128)*128``, false at 384 and true at 512), i.e.
# the only one that can measure `TileM = 256` on its merits rather than on its
# padding. ``test_perf_prefill_tile_sweep_long_seq`` did, and it came out level
# or behind, which is why the ladder no longer has a 256-row rung.
_PREFILL_LONG_SEQ_LEN = 8192


# Rows per expert for the epilogue equivalence tests: enough to give every
# expert interior *and* ragged tiles against the prefill tile the ladder picks
# for this shape (128 rows: 300 is two full tiles plus a 44-row remainder),
# which is what makes a single launch exercise both epilogue paths. It is
# deliberately not tied to the perf batch above -- these tests need a specific
# tile geometry, not a compute-bound routing, and the smaller batch keeps them
# quick.
_RAGGED_TILE_ROWS_PER_EXPERT = 300


def _compute_bound_batches(model: dict) -> list:
    """Model-token batches that put ``_PREFILL_TARGET_ROWS_PER_EXPERT`` rows on every expert."""
    return [_PREFILL_TARGET_ROWS_PER_EXPERT * model["E"] // model["topk"]]


def _long_seq_batches() -> list:
    """The long-prompt prefill point: one ``_PREFILL_LONG_SEQ_LEN``-token sequence.

    Unlike ``_compute_bound_batches`` this is *not* derived per model -- the
    point of it is that a fixed prompt length routes differently in each model
    (see ``_PREFILL_LONG_SEQ_LEN``).
    """
    return [_PREFILL_LONG_SEQ_LEN]


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


def _int8_grid_activations(total_tokens, K, dtype, generator):
    """Activations every correct absmax quantizer maps to the *same* int8 bytes.

    The pre-quantized contract is only testable if the caller's int8 and the
    kernel's own agree, and on arbitrary data they are not required to. Both
    compute ``inv = 127 / absmax`` and round ``a * inv``, but the harness runs
    that division in exact fp32 while SPIR-V lets the device's carry a few ulp,
    so any product sitting on a ``.5`` tie can land on either side. With 16-bit
    activations that is not a rare accident -- a bf16 row carries 8 mantissa
    bits, so exact ties are common -- and a flipped int8 is *not* a rounding
    difference in the output: it is a different input to the dot product, which
    moves the result by ``|w| * scales`` in absolute terms, i.e. without bound
    in ULP wherever the accumulator happens to cancel to near zero.

    So the rows are put on the quantizer's own grid instead. ``a = q * 2^-e``
    with integer ``|q| <= 127`` is exact in every 16-bit float format, ``127``
    is planted in each row so its absmax is exactly ``127 * 2^-e``, and both
    ``127 / absmax = 2^e`` and ``absmax / 127 = 2^-e`` are powers of two. Every
    product is then the integer ``q`` itself -- half a step from the nearest
    tie, a margin no plausible division error comes close to bridging -- so the
    two quantizers must produce identical bytes and any difference in the
    output belongs to the contract rather than to the rounding.

    ``e`` walks 4..8 down the rows so neighbouring rows have different scales:
    a scale read from the wrong row is then a factor of two, not a coincidence.
    """
    q = torch.randint(-127, 128, (total_tokens, K), generator=generator, dtype=torch.int32)
    q[:, 0] = 127
    exponent = -(4 + torch.arange(total_tokens, dtype=torch.int32) % 5)
    activations = torch.ldexp(q.to(torch.float32), exponent.unsqueeze(1)).to(dtype)

    # The construction is only worth anything if it survives the cast to
    # `dtype` and the reference quantizer's own two divisions, so it is checked
    # here -- cheaply, on the host, before the rows reach the device -- rather
    # than left for a later edit to reintroduce the ties unnoticed.
    q_back, scale_back = _quantize_rows(activations)
    assert torch.equal(q_back.to(torch.int32), q), f"the int8 grid does not survive {dtype}"
    assert torch.equal(
        scale_back, torch.ldexp(torch.ones(total_tokens), exponent)
    ), "the row scales are not the powers of two the grid was built from"
    return activations


def _build_case(
    N,
    K,
    E,
    total_tokens,
    group_size,
    dtype,
    device="xpu",
    seed=0,
    need_reference=True,
    need_dequant=True,
    topk=None,
    act_int8_grid=False,
):
    """Build one W4A8 MoE test case.

    Returns a dict with the packed int4 weights + scales, the activations,
    the routing histogram, and the fp32 reference output computed from the
    dequantized weights (so quantization error of the *weights* is excluded
    from the W4A8-vs-reference comparison, isolating the extra error the int8
    activation path introduces).

    ``need_dequant`` / ``need_reference`` are opt-outs for the perf sweeps:
    the dequantized ``[E, N, K]`` weights (805 MB at the Qwen3 up-proj shape in
    bf16) are only needed by the torch baseline, and the fp32 reference
    (``[total_tokens, N]`` plus a full fp32 grouped matmul) is only needed by
    the accuracy sweep. Skipping them is what keeps the compute-bound batches
    -- the only ones where a prefill TOPS target is physically reachable --
    inside a sane memory and time budget.

    ``topk`` adds the side tables the fused top-k reduction needs: a
    ``row -> model token`` map and a per-row routing weight. Row ``r`` is given
    to token ``r % batch``, which puts each token on exactly ``top_k`` rows and
    -- because an expert's block of rows is shorter than ``batch`` at every
    shipped ``E`` -- never twice on the same expert, i.e. the same structure a
    real router produces after the rows are sorted by expert.

    ``act_int8_grid`` swaps the normally-distributed activations for rows that
    quantize exactly (:func:`_int8_grid_activations`). It is for the tests that
    compare two *quantizers* against each other rather than the kernel against
    a reference, where a tie broken differently on either side would swamp what
    is being measured.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w_float = torch.randn(E, N, K, generator=generator, dtype=torch.float32) * 0.05
    scales = torch.empty(E, N, K // group_size, dtype=dtype)
    packed = _pack_int4_sym(w_float, scales, group_size)

    if act_int8_grid:
        activations = _int8_grid_activations(total_tokens, K, dtype, generator)
    else:
        activations = (torch.randn(total_tokens, K, generator=generator, dtype=torch.float32) * 0.5).to(dtype)
    tpe = _spread_tokens(total_tokens, E)
    ntpe = torch.tensor(tpe, dtype=torch.int32)

    row_to_token = None
    routing_weights = None
    batch = None
    if topk:
        batch = total_tokens // topk
        row_to_token = (torch.arange(total_tokens, dtype=torch.int32) % max(batch, 1)).contiguous()
        routing_weights = (torch.rand(total_tokens, generator=generator, dtype=torch.float32) + 0.5) / topk

    packed = packed.to(device)
    scales = scales.to(device)
    activations = activations.to(device)
    ntpe = ntpe.to(device)
    if topk:
        row_to_token = row_to_token.to(device)
        routing_weights = routing_weights.to(device)

    dequant = _dequant_int4_sym(packed, scales, group_size) if (need_dequant or need_reference) else None

    # fp32 per-expert reference on the dequantized weights.
    reference = None
    if need_reference:
        reference = torch.empty(total_tokens, N, device=device, dtype=torch.float32)
        offset = 0
        for e, n_e in enumerate(tpe):
            if n_e == 0:
                continue
            a = activations[offset : offset + n_e].to(torch.float32)
            reference[offset : offset + n_e] = a @ dequant[e].to(torch.float32).t()
            offset += n_e
        if not need_dequant:
            dequant = None

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
        "topk": topk,
        "batch": batch,
        "row_to_token": row_to_token,
        "routing_weights": routing_weights,
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


def _quantize_rows(activations):
    """Per-row absmax int8 quantization -- the reference for what the kernel does.

    Reproduces ``launch_act_dynamic_quant`` expression for expression: the
    absmax is an exact reduction, ``inv = 127 / absmax`` and
    ``scale = absmax / 127`` are the same two fp32 operations, and ``rint`` is
    round-half-to-even, which is what ``torch.round`` does as well.

    Same expressions is not the same bits, though. Those two divisions are
    exact in IEEE fp32 here and only approximate on the device -- SPIR-V lets
    a division carry a few ulp of error, and the kernel is not built with the
    flags that would forbid that -- so ``inv`` can land one step off, and any
    product that sits on a ``.5`` tie then rounds the other way. The callers
    that need the two to agree byte for byte hand this function rows that
    cannot tie (:func:`_int8_grid_activations`); on arbitrary rows it is a
    faithful model of the kernel, not a bit-exact one.
    """
    a = activations.to(torch.float32)
    absmax = a.abs().amax(dim=1)
    scale = absmax / 127.0
    inv = torch.where(absmax > 0, 127.0 / absmax, torch.zeros_like(absmax))
    q = torch.clamp(torch.round(a * inv.unsqueeze(1)), -127.0, 127.0).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _prequantized(case):
    """Cached ``(int8 activations, fp32 row scales)`` for ``case``."""
    if case.get("qact") is None:
        qact, ascale = _quantize_rows(case["activations"])
        case["qact"], case["ascale"] = qact, ascale
    return case["qact"], case["ascale"]


def _reduces_topk(nk_label: str) -> bool:
    """Whether this projection's output is the one the top-k combine reduces.

    Only the second (down) GEMM of a MoE layer produces rows that are scaled by
    the routing weights and summed back per token. The up/gate output stays
    expanded -- one row per routed token, straight into SiLU -- so it owes no
    caller-side reduction and cannot use the fused epilogue. Charging it one
    would credit the fused contract with removing work that never existed, the
    same overstatement :func:`_traffic_bytes` avoids by leaving ``fused_rows``
    unset.
    """
    return nk_label.strip().endswith("down")


def _reduce_topk(case, out):
    """Reduce an unfused ``[T, N]`` output the way a caller would.

    ``index_add_`` on an fp32 accumulator, i.e. exactly the reduction the fused
    epilogue replaces -- so this is what a fused result is compared against.
    """
    reduced = torch.zeros((case["batch"], case["N"]), device=out.device, dtype=torch.float32)
    reduced.index_add_(0, case["row_to_token"].to(torch.long), out.to(torch.float32) * case["routing_weights"][:, None])
    return reduced


def _w4a8(case, weights_s8, wscales, block, phase, prequant=False, fused=False):
    """One ``moe_gemm_w4a8`` call under the requested call contract.

    ``prequant`` hands the kernel int8 activations and their per-row scales so
    it skips its own quantization pass; ``fused`` asks the epilogue to apply
    the routing weights and scatter-add into a ``[batch, N]`` fp32 accumulator
    instead of writing the unreduced ``[T, N]``.
    """
    kwargs = {}
    activations = case["activations"]
    if prequant:
        activations, ascale = _prequantized(case)
        kwargs["activation_scale"] = ascale
        kwargs["out_dtype"] = case["dtype"]
    if fused:
        kwargs["row_to_token"] = case["row_to_token"]
        kwargs["routing_weights"] = case["routing_weights"]
        kwargs["output_rows"] = case["batch"]
    return ark.moe_gemm_w4a8(
        activations,
        weights_s8,
        wscales,
        case["ntpe"],
        rescale_block_size=block,
        phase=phase,
        **kwargs,
    )


def _flops(total_tokens, N, K) -> float:
    """MoE grouped GEMM FLOPs: each token does one ``[K] x [K, N]`` product."""
    return float(total_tokens) * N * K * 2.0


def _weight_bytes(E, N, K, bits) -> float:
    return float(E) * N * K * bits / 8.0


def _dtype_bytes(dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# Targets and the roofline they have to be read against
#
# The weights are the largest single stream, but they are not the only one. A
# W4A8 MoE call moves, per invocation:
#
#   * ``T * K * sizeof(act)``  -- the routed activations, read by the quantizer
#   * ``T * K``                -- the int8 quantized copy, written
#   * ``T * K``                -- and read back by the GEMM
#   * ``E_active * N * K``     -- every active expert's int8 weights, once
#   * ``T * N * sizeof(out)``  -- the output
#
# Only the fourth line is what ``W GB/s`` reports, and it is a *minority* of
# the total whenever K is small: at 384 rows per expert the qwen3
# down-projection (N = 2048, K = 768) moves 201 MB of weights inside 554 MB of
# traffic, so a roofline built on weights alone understates the bandwidth a
# shape needs by ~2.7x. This matters for the verdict, not just the bookkeeping:
# at the 256 rows per expert this harness used to measure at, the same shape
# needs 423 GB/s to reach 100 TFLOPS -- past what a 456 GB/s part delivers, so
# the target was unreachable there whatever the kernel did. See
# ``_PREFILL_TARGET_ROWS_PER_EXPERT``.
#
# The old model -- ``TFLOPS <= 2 * rows_per_expert * weight_bandwidth``, i.e.
# ``BW@100T = 50 TB/s / rows_per_expert`` -- is the weights-only special case
# and is what the earlier tuning rounds were judged against. It is kept in mind
# here only as the reason those rounds read as "close to target": the target
# was measured against half the traffic.
#
# The intensity still rises with ``rows/E`` (the weight term is the only one
# that does not grow with T), which is why the perf table prints ``rows/E`` and
# why the prefill target sweep uses a compute-bound batch -- but the ceiling
# now saturates instead of growing without bound, because the activation,
# quantized-activation and output streams all scale with T exactly as the
# FLOPs do.
# ---------------------------------------------------------------------------

_TARGET_PREFILL_TFLOPS = 100.0
_TARGET_DECODE_GBPS = 300.0


def _rows_per_expert(total_tokens, active_experts) -> float:
    return float(total_tokens) / float(active_experts) if active_experts else 0.0


def _traffic_bytes(total_tokens, active_experts, N, K, act_bytes=2, out_bytes=2, prequantized=False, fused_rows=None):
    """Compulsory DRAM traffic of one ``moe_gemm_w4a8`` call, in bytes.

    Counts each byte once: re-reads of A across the N tiles are L2 hits at any
    launch this kernel produces (~20 concurrent work-groups against 8 MB of
    L2), so they are not DRAM traffic. This is a lower bound, which keeps the
    derived ceiling optimistic and therefore never excuses a slow kernel.

    The two optional call contracts remove whole streams, so the model has to
    know which one is in force or every derived number (``BW@100T``, the
    ceiling, the PASS/FAIL verdict) is computed against traffic the call no
    longer moves:

    * ``prequantized``: the caller hands over int8 activations, so the 16-bit
      read and the int8 write are gone and only the GEMM's read-back remains.
    * ``fused_rows``: the epilogue reduces into a ``[batch, N]`` fp32
      accumulator, so instead of writing ``T * N`` elements it reads *and*
      writes ``batch * N`` fp32 ones.

    This counts one *call*, so the caller-side reduction an unfused result
    still owes is deliberately not in here: only the second projection's output
    is reduced in a real layer, and the up/gate output feeds SiLU unreduced, so
    charging every unfused shape for a reduction would overstate the traffic on
    half of them. The contract sweep accounts for that reduction in the timing
    instead, where it can be applied per configuration.
    """
    act_read = 0.0 if prequantized else float(total_tokens) * K * act_bytes
    qact_write_read = (1.0 if prequantized else 2.0) * float(total_tokens) * K
    weights = float(active_experts) * N * K
    if fused_rows:
        out_write = 2.0 * float(fused_rows) * N * 4
    else:
        out_write = float(total_tokens) * N * out_bytes
    return act_read + qact_write_read + weights + out_write


def _bw_needed_for_tflops(
    total_tokens,
    active_experts,
    N,
    K,
    act_bytes=2,
    out_bytes=2,
    tflops_target=_TARGET_PREFILL_TFLOPS,
    prequantized=False,
    fused_rows=None,
) -> float:
    """GB/s of DRAM traffic a shape needs to hit ``tflops_target``."""
    flops = _flops(total_tokens, N, K)
    if flops <= 0.0:
        return float("inf")
    seconds_at_target = flops / (tflops_target * 1e12)
    traffic = _traffic_bytes(
        total_tokens, active_experts, N, K, act_bytes, out_bytes, prequantized=prequantized, fused_rows=fused_rows
    )
    return traffic / seconds_at_target / 1e9


def _tflops_ceiling(
    total_tokens, active_experts, N, K, gbps, act_bytes=2, out_bytes=2, prequantized=False, fused_rows=None
) -> float:
    """Best TFLOPS this shape can reach at ``gbps`` of DRAM bandwidth."""
    traffic = _traffic_bytes(
        total_tokens, active_experts, N, K, act_bytes, out_bytes, prequantized=prequantized, fused_rows=fused_rows
    )
    if traffic <= 0.0:
        return float("inf")
    return _flops(total_tokens, N, K) / (traffic / (gbps * 1e9)) / 1e12


_DEVICE_BW_GBPS = None


def _device_bandwidth_gbps():
    """Measure achievable device DRAM bandwidth with a large device-to-device copy.

    Used to turn ``rows/E`` into a *hard* TFLOPS ceiling for the current GPU,
    so a prefill row can be reported as "unreachable at this routing" instead
    of "slow". Deriving the ceiling from the kernel's own measured bandwidth
    would be circular (it reproduces the measured TFLOPS exactly), hence the
    independent probe.

    The copy counts one read plus one write, which slightly *understates*
    read-only weight streaming. That conservatism is not free: a ceiling that
    is too low reads as "bandwidth bound", and :func:`_assert_targets` skips
    enforcement on exactly those rows, so an under-measured probe silently
    excuses a slow kernel. :func:`_apply_bandwidth_evidence` is the guard.

    Taken as the *best* of several rounds rather than one burst's median. A
    single burst samples whatever clock state the device happens to be in, and
    on B70 that spread three consecutive runs across 439 / 373 / 299 GB/s --
    a 47% swing in a number the ceilings treat as a hardware constant. The
    fastest copy is the one least contaminated by throttling, which is the same
    reason every other measurement here min-filters. Cached after the first
    call; returns ``None`` if XPU is unavailable or the probe fails.
    """
    global _DEVICE_BW_GBPS
    if _DEVICE_BW_GBPS is not None:
        return _DEVICE_BW_GBPS
    if not _xpu_available():
        return None
    nbytes = 128 * 1024 * 1024  # far past any device cache
    try:
        src = torch.empty(nbytes, dtype=torch.int8, device="xpu")
        dst = torch.empty_like(src)
        ms = min(_xpu_time_ms(lambda: dst.copy_(src), warmup=2, iters=5) for _ in range(3))
        _DEVICE_BW_GBPS = 2.0 * nbytes / (ms * 1e-3) / 1e9
    except Exception as exc:  # pragma: no cover - depends on device/runtime
        print(f"[moe-w4a8-perf] device bandwidth probe unavailable: {exc}")
        return None
    finally:
        src = dst = None
        _release_xpu_memory()
    return _DEVICE_BW_GBPS


def _apply_bandwidth_evidence(rows) -> None:
    """Raise the ceilings when a row proved the probe under-measured.

    The ceilings are derived from an independent copy probe, but the probe is
    only ever a *lower* bound on what the part can stream: it counts a read
    plus a write, and it samples one moment of one clock state. A kernel row
    that moved its own traffic at a higher rate is direct evidence the device
    sustains at least that much -- so the probe, not the row, is what was
    wrong.

    Without this a run can print "118% of the bandwidth ceiling", which is not
    a thing a roofline can do; worse, :func:`_assert_targets` waives the target
    for any row sitting below its ceiling, so a cold probe hands out the
    "bandwidth bound, unreachable" excuse to rows that are merely slow.

    Ceilings are linear in bandwidth, so each is rescaled by the same ratio.
    This only ever raises them, which only ever makes the verdict stricter.
    """
    if not rows:
        return
    probe = rows[0].get("device_bw_gbps")
    if not probe:
        return
    best = max((r.get("dram_gbps") or 0.0) for r in rows)
    if best <= probe:
        return
    winner = max(rows, key=lambda r: r.get("dram_gbps") or 0.0)
    scale = best / probe
    for row in rows:
        if row.get("tflops_ceiling") is not None:
            row["tflops_ceiling"] *= scale
        row["bw_evidence_gbps"] = best
        row["bw_evidence_label"] = winner["label"]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_ACC_WIDTH = 150
_PERF_WIDTH = 179


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
    * ``DRAM GB/s``: *all* the traffic the call moves -- the activations the
      quantizer reads, the int8 copy it writes, the GEMM's read of that copy,
      the weights and the output. On the small-K shapes the weights are under
      half of this, so it is the number to compare against the device probe.
    * ``rows/E``: routed tokens per active expert. The weight stream is the
      only one that does not grow with the token count, so this is what raises
      the arithmetic intensity -- but the other four streams keep it bounded.
    * ``BW@100T``: DRAM bandwidth the shape would need to reach 100 TFLOPS.
      When it exceeds what the device can stream, ``TFLOPS`` is capped by
      memory and no kernel change can reach the target at that shape.
    * ``vs torch`` / ``vs w4a16``: speedups (``other / w4a8``).
    """
    print()
    print("=" * _PERF_WIDTH)
    print(title)
    print(
        f"{'shape':<14}{'E':>5}{'N':>7}{'K':>7}{'tokens':>8}{'rows/E':>8}"
        f"{'torch(ms)':>12}{'w4a16(ms)':>12}{'w4a8(ms)':>12}"
        f"{'TFLOPS':>10}{'W GB/s':>10}{'DRAM GB/s':>11}{'BW@100T':>10}"
        f"{'vs torch':>11}{'vs w4a16':>11}{'prepack(ms)':>13}"
    )
    print("-" * _PERF_WIDTH)


def _print_perf_row(
    label,
    E,
    N,
    K,
    tokens,
    torch_ms,
    w4a16_ms,
    w4a8_ms,
    tflops,
    gbps,
    prepack_ms,
    rows_per_expert=None,
    bw_at_100t=None,
    dram_gbps=None,
):
    def _fmt(v, digits=3):
        if v is None:
            return "--"
        if isinstance(v, float) and math.isinf(v):
            return "inf"
        return f"{v:.{digits}f}"

    vs_torch = None if (torch_ms is None or not w4a8_ms) else torch_ms / w4a8_ms
    vs_w4a16 = None if (w4a16_ms is None or not w4a8_ms) else w4a16_ms / w4a8_ms
    print(
        f"{label:<14}{E:>5}{N:>7}{K:>7}{tokens:>8}{_fmt(rows_per_expert, 1):>8}"
        f"{_fmt(torch_ms):>12}{_fmt(w4a16_ms):>12}{_fmt(w4a8_ms):>12}"
        f"{_fmt(tflops, 2):>10}{_fmt(gbps, 1):>10}{_fmt(dram_gbps, 1):>11}{_fmt(bw_at_100t, 0):>10}"
        f"{(_fmt(vs_torch, 2) + 'x') if vs_torch else '--':>11}"
        f"{(_fmt(vs_w4a16, 2) + 'x') if vs_w4a16 else '--':>11}"
        f"{_fmt(prepack_ms, 2):>13}"
    )


def _print_targets(phase: str, rows) -> None:
    """Print the goal verdict for a perf sweep.

    Prefill is judged on TFLOPS, decode on weight bandwidth. A prefill row
    whose device-bandwidth ceiling is already below the target is reported as
    ``N/A`` rather than ``FAIL``: at that routing the target is unreachable by
    construction (see the roofline note above ``_TARGET_PREFILL_TFLOPS``), and
    the row's measured bandwidth is what should be judged instead.

    The ceiling counts every stream the call moves, not just the weights, so a
    small-K shape can be ``N/A`` here while the weights-only model of earlier
    rounds called it reachable. A reachable row also prints how much of the
    ceiling it actually reaches, which is the number a kernel change can move.
    """
    if not rows:
        return
    is_prefill = phase == "prefill"
    target = _TARGET_PREFILL_TFLOPS if is_prefill else _TARGET_DECODE_GBPS
    unit = "TFLOPS" if is_prefill else "GB/s"
    device_bw = rows[0].get("device_bw_gbps")
    evidence = rows[0].get("bw_evidence_gbps")
    print()
    print(f"targets [{phase}]: {'prefill compute' if is_prefill else 'decode weight bandwidth'} > {target:g} {unit}")
    if device_bw:
        note = ""
        if evidence:
            note = (
                f" (under-measured: {rows[0]['bw_evidence_label']} streamed {evidence:.0f} GB/s, "
                f"so the ceilings below use that)"
            )
        print(f"  device copy bandwidth probe: {device_bw:.0f} GB/s{note}")
    for row in rows:
        measured = row["tflops"] if is_prefill else row["gbps"]
        ceiling = row.get("tflops_ceiling")
        if is_prefill and ceiling is not None and ceiling < target:
            verdict = (
                f"N/A (bandwidth bound: ceiling {ceiling:.1f} {unit} at {row['rows_per_expert']:.0f} rows/expert; "
                f"reaching {target:g} would need {row['bw_at_100t']:.0f} GB/s of DRAM traffic)"
            )
        else:
            verdict = "PASS" if measured > target else "FAIL"
            if is_prefill and ceiling:
                verdict += f" ({measured / ceiling * 100:.0f}% of the {ceiling:.0f} {unit} bandwidth ceiling)"
        print(
            f"  {row['label']:<12} tokens={row['tokens']:<6} rows/E={row['rows_per_expert']:<6.1f} "
            f"{measured:8.2f} {unit} vs {target:g} -> {verdict}"
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


def run_perf(
    phase,
    batches,
    dtype=torch.bfloat16,
    rescale_group_size=-1,
    verbose=True,
    torch_baseline=True,
    models=None,
    compute_bound=False,
    prequantized=False,
    fused_reduce=False,
):
    """Run the W4A8 perf sweep. Returns a list of per-row metric dicts.

    ``torch_baseline=False`` skips both the dequantized weights and the torch
    matmul timing; the compute-bound batches need it to stay within memory.

    ``models`` selects the shape groups (``"qwen3"``, ``"minimax"``, ``"all"``
    or a list); ``compute_bound=True`` ignores ``batches`` and derives, per
    model, the batch that puts ``_PREFILL_TARGET_ROWS_PER_EXPERT`` rows on every
    expert -- the only regime where the prefill TOPS target is reachable.

    ``prequantized`` / ``fused_reduce`` select the traffic-cutting call
    contracts (see :func:`_w4a8`). They change what the call moves, so they are
    also fed to the traffic model: the printed ``DRAM GB/s``, ``BW@100T`` and
    the ceiling that decides the verdict all follow the contract in force.

    ``fused_reduce`` applies only to the projection whose output is actually
    top-k reduced (:func:`_reduces_topk`). Fusing the up/gate projection is not
    a contract a layer can take -- that output stays expanded into SiLU -- so
    enabling it there measures a configuration nobody can ship and charges the
    contract for it. The header says which shapes it reached.
    """
    rows = []
    # Probed before anything large is allocated (and cached across sweeps).
    device_bw = _device_bandwidth_gbps()
    resolved = _models(models)
    contract = "".join(
        [
            ", A=int8-in" if prequantized else "",
            ", fused top-k reduce (down-proj only)" if fused_reduce else "",
        ]
    )
    if verbose:
        _print_perf_header(
            f"W4A8 perf [{phase}] (models={'+'.join(n for n, _ in resolved)}, "
            f"group_size={_QWEN3_GROUP_SIZE}, "
            f"act={str(dtype).split('.')[-1]}, rescale_group_size={rescale_group_size}{contract}) "
            f"-- ark.moe_gemm_w4a8 vs W4A16 vs torch"
        )
    shapes = [
        (nk_label, N, K, spec, batch)
        for _, spec in resolved
        for nk_label, N, K in spec["nk"]
        for batch in (_compute_bound_batches(spec) if compute_bound else batches)
    ]
    for nk_label, N, K, spec, batch in shapes:
        E, topk, group_size = spec["E"], spec["topk"], spec["group_size"]
        total_tokens = batch * topk
        # Only the reduced projection can take the fused contract.
        fused_here = fused_reduce and _reduces_topk(nk_label)
        case = _build_case(
            N,
            K,
            E,
            total_tokens,
            group_size,
            dtype,
            need_reference=False,
            need_dequant=torch_baseline,
            topk=topk if fused_here else None,
        )

        # One-shot int4 -> int8 AUTO_S8 conversion. Timed separately: it
        # happens once at model load, not per forward.
        prepack_ms = _xpu_time_ms(
            lambda: ark.moe_w4a8_prepack(
                case["packed"],
                case["scales"],
                group_size=group_size,
                rescale_group_size=rescale_group_size,
            ),
            warmup=1,
            iters=3,
        )
        weights_s8, wscales, block = ark.moe_w4a8_prepack(
            case["packed"],
            case["scales"],
            group_size=group_size,
            rescale_group_size=rescale_group_size,
        )

        # Round-robin over the three implementations rather than timing each
        # to completion in turn. Measuring w4a8 first and w4a16 last put the
        # two on opposite ends of the sweep's clock droop, which biases the
        # headline `vs w4a16` ratio *upward* -- both the numerator being
        # measured hot and the denominator cold push it the same way. Taking
        # each one's least-throttled round removes that from the ratio; see
        # `_sweep_timings` for the measurement this is guarding against.
        w4a8_samples, torch_samples, w4a16_samples = [], [], []
        w4a16_failed = None
        iters = max(1, ITERS // SWEEP_ROUNDS)
        for _ in range(SWEEP_ROUNDS):
            w4a8_samples.append(
                _xpu_time_ms(
                    lambda: _w4a8(case, weights_s8, wscales, block, phase, prequant=prequantized, fused=fused_here),
                    iters=iters,
                )
            )
            if torch_baseline:
                torch_samples.append(_xpu_time_ms(lambda: _torch_baseline(case), iters=iters))
            if w4a16_failed is None:
                try:
                    w4a16_samples.append(_xpu_time_ms(lambda: _w4a16(case, phase), iters=iters))
                except Exception as exc:  # pragma: no cover - depends on build
                    w4a16_failed = exc
        if w4a16_failed is not None:
            print(f"[moe-w4a8-perf] W4A16 timing unavailable for {nk_label}: {w4a16_failed}")
        w4a8_ms = min(w4a8_samples)
        torch_ms = min(torch_samples) if torch_samples else None
        w4a16_ms = min(w4a16_samples) if w4a16_samples else None

        active_experts = sum(1 for n_e in case["tpe"] if n_e > 0)
        tflops = _flops(total_tokens, N, K) / (w4a8_ms * 1e-3) / 1e12
        # W4A8 streams int8 weights: 1 byte per element, only for the
        # experts that actually received tokens.
        gbps = _weight_bytes(active_experts, N, K, 8) / (w4a8_ms * 1e-3) / 1e9
        # Everything the call moves, not just the weights: the quantizer's read
        # of A, the int8 copy it writes, the GEMM's read of that copy and the
        # output. On the small-K shapes the weights are under half of it.
        act_bytes = _dtype_bytes(dtype)
        fused_rows = case["batch"] if fused_here else None
        traffic = _traffic_bytes(
            total_tokens,
            active_experts,
            N,
            K,
            act_bytes,
            act_bytes,
            prequantized=prequantized,
            fused_rows=fused_rows,
        )
        dram_gbps = traffic / (w4a8_ms * 1e-3) / 1e9
        rows_per_expert = _rows_per_expert(total_tokens, active_experts)
        bw_at_100t = _bw_needed_for_tflops(
            total_tokens,
            active_experts,
            N,
            K,
            act_bytes,
            act_bytes,
            prequantized=prequantized,
            fused_rows=fused_rows,
        )

        row = {
            "label": nk_label,
            "phase": phase,
            "E": E,
            "N": N,
            "K": K,
            "tokens": total_tokens,
            "torch_ms": torch_ms,
            "w4a16_ms": w4a16_ms,
            "w4a8_ms": w4a8_ms,
            "tflops": tflops,
            "gbps": gbps,
            "dram_gbps": dram_gbps,
            "prepack_ms": prepack_ms,
            "active_experts": active_experts,
            "rows_per_expert": rows_per_expert,
            "bw_at_100t": bw_at_100t,
            # Hard ceiling for this routing on this device (``None`` when
            # the bandwidth probe is unavailable).
            "tflops_ceiling": (
                None
                if device_bw is None
                else _tflops_ceiling(
                    total_tokens,
                    active_experts,
                    N,
                    K,
                    device_bw,
                    act_bytes,
                    act_bytes,
                    prequantized=prequantized,
                    fused_rows=fused_rows,
                )
            ),
            "device_bw_gbps": device_bw,
            "prequantized": prequantized,
            "fused_reduce": fused_here,
        }
        rows.append(row)
        if verbose:
            _print_perf_row(
                nk_label,
                E,
                N,
                K,
                total_tokens,
                torch_ms,
                w4a16_ms,
                w4a8_ms,
                tflops,
                gbps,
                prepack_ms,
                rows_per_expert=rows_per_expert,
                bw_at_100t=bw_at_100t,
                dram_gbps=dram_gbps,
            )

        # Drop the (large) int8 weights before the next shape allocates.
        case = weights_s8 = wscales = None
        ark.clear_moe_w4a8_prepack_cache()
        ark.moe_w4a8_release_scratch()
        _release_xpu_memory()
    # Before any verdict is drawn from the ceilings -- including the one
    # `_assert_targets` draws from the returned rows, which is why this is not
    # gated on `verbose`.
    _apply_bandwidth_evidence(rows)
    if verbose:
        _print_targets(phase, rows)
    return rows


def run_dedup_quant(batches=None, dtype=torch.bfloat16, models=None, verbose=True):
    """End-to-end permute + GEMM for the up/gate projection, two ways.

    The kernel quantizes its activations per row, and the row absmax is a
    property of the *token*: ``a.abs().amax(dim=1)`` does not depend on which
    expert the row was routed to. On the up/gate projection the ``T`` rows the
    kernel is handed are ``top_k`` copies of ``batch`` distinct tokens, so the
    in-call pass computes every scale ``top_k`` times and writes ``top_k``
    copies of every int8 row -- 8x redundant at the shipped ``top_k = 8``.

    Hoisting the same quantization above the permute removes that redundancy
    and lets a caller reach the pre-quantized contract *without* an int8
    producer upstream: the dynamic quantization still happens, just on
    ``batch`` rows instead of ``batch * top_k``, and the permute then moves
    int8 instead of 16-bit.

    Every path is timed **including the caller's permute and its
    quantization**, which is what makes this an honest comparison rather than
    an accounting shift. The pre-quantized contract on its own only moves the
    quantization across the call boundary; it is the deduplication -- and the
    halved permute that comes with moving int8 instead of bf16 -- that makes
    the work actually smaller. A caller who quantized the *permuted* rows
    themselves would see the contract's gain cancel almost exactly.

    **Which quantizer does the deduplicated work matters more than the
    deduplication.** :func:`_quantize_rows` is the eager-torch reference: it
    upcasts to fp32 and walks the tensor about seven times, once per operator,
    materializing a full-size intermediate each time. The in-call path uses the
    fused SYCL quantizer, which reads each row once and keeps the absmax in
    registers. Comparing them directly charges the deduplicated path a constant
    factor roughly fifteen times worse per row, which is more than enough to
    eat an 8x reduction in rows -- so this reports all three points and lets
    the columns say which effect is which:

    * ``in-call quant`` -- what ships today: permute 16-bit, kernel quantizes
      ``T`` rows with the fused quantizer.
    * ``dedup, torch quant`` -- the deduplication done with the reference
      quantizer. Not a shipping configuration; it is here because it is the
      obvious way to try this and it *loses*, and that is worth recording.
    * ``dedup, fused quant`` -- the deduplication with a quantizer of the same
      quality as the one already in the kernel.

    The fused quantizer has no standalone Python entry point, so its cost is
    measured rather than assumed: the same GEMM is timed with 16-bit input and
    with int8 input, on the same shape and the same weights, and the difference
    is the in-kernel quantization of exactly those rows. Doing that at both
    ``T`` and ``batch`` rows also cross-checks that the cost is linear in rows
    (it should divide by ``top_k``), which is reported as ``fused quant
    T/batch`` so a bad measurement cannot pass silently.

    Up/gate only. The down projection's ``T`` rows are the SiLU output, one
    distinct row per routed row, so there is nothing to deduplicate; its route
    to the same contract is folding the quantization into the SiLU epilogue
    that already writes that tensor, which is what
    ``test_perf_prefill_prequant_long_seq`` measures.
    """
    batches = _long_seq_batches() if batches is None else batches
    resolved = _models(models)
    if verbose:
        print()
        print("=" * _PERF_WIDTH)
        print(
            f"W4A8 end-to-end [prefill] (models={'+'.join(n for n, _ in resolved)}, up/gate only) "
            f"-- caller permute + moe_gemm_w4a8, in-call quant vs deduplicated quant"
        )
        print(
            f"{'shape':<14}{'E':>5}{'N':>7}{'K':>7}{'tokens':>8}{'batch':>8}{'topk':>6}"
            f"{'path':>21}{'permute(ms)':>13}{'quant(ms)':>11}{'gemm(ms)':>11}{'total(ms)':>12}"
            f"{'vs in-call':>12}"
        )
        print("-" * _PERF_WIDTH)
    rows = []
    shapes = [
        (nk_label, N, K, spec, batch)
        for _, spec in resolved
        for nk_label, N, K in spec["nk"]
        if not _reduces_topk(nk_label)
        for batch in batches
    ]
    for nk_label, N, K, spec, batch in shapes:
        E, topk, group_size = spec["E"], spec["topk"], spec["group_size"]
        total_tokens = batch * topk
        case = _build_case(
            N, K, E, total_tokens, group_size, dtype, need_reference=False, need_dequant=False, topk=topk
        )
        weights_s8, wscales, block = ark.moe_w4a8_prepack(
            case["packed"], case["scales"], group_size=group_size, rescale_group_size=-1
        )
        # The unpermuted hidden states a real layer holds, and the map the
        # router produces. `index_select` wants a 64-bit index; converting it
        # is one-off setup a caller does once per layer, not per call, so it
        # stays outside both timed paths.
        n_tokens = case["batch"]
        # `clone`, not `contiguous`: a leading slice of a contiguous tensor is
        # already contiguous, so `contiguous()` would hand back a view and keep
        # the whole [T, K] allocation alive through it. This function holds
        # several large tensors live at once -- both permuted forms are
        # materialized so each stage can be timed on its own -- so the 268 MB
        # the full activations occupy at this shape is worth actually
        # releasing.
        hidden = case["activations"][:n_tokens].clone()
        index = case["row_to_token"].long()
        ntpe_full = case["ntpe"]
        case["activations"] = case["packed"] = case["scales"] = None
        _release_xpu_memory()

        # Routing histogram for a `batch`-row call: same weights, same experts,
        # one row per distinct token. Only used to isolate the fused
        # quantizer's cost at that row count.
        ntpe_batch = torch.tensor(_spread_tokens(n_tokens, E), dtype=torch.int32, device=hidden.device)

        def _call(act, ascale=None, ntpe=None):
            kwargs = {"rescale_block_size": block, "phase": "prefill"}
            if ascale is not None:
                kwargs["activation_scale"] = ascale
                kwargs["out_dtype"] = dtype
            return ark.moe_gemm_w4a8(act, weights_s8, wscales, ntpe_full if ntpe is None else ntpe, **kwargs)

        # Materialized once, outside every timed region: each stage is timed on
        # its own so the reported columns are attributable, and the three
        # stages form a true dependency chain (permute -> quantize -> GEMM),
        # which is why summing separately-timed stages is a faithful model of
        # running them back to back.
        qact_batch, ascale_batch = _quantize_rows(hidden)
        permuted_bf16 = hidden.index_select(0, index)
        permuted_int8 = qact_batch.index_select(0, index)
        permuted_scale = ascale_batch.index_select(0, index)

        # Round-robin, min-filtered: the same guard against clock droop the
        # other sweeps use, since these points are compared to each other
        # rather than reported in isolation.
        stages = {
            k: []
            for k in (
                "perm_bf16",
                "perm_int8",
                "quant_torch",
                "gemm_bf16_t",
                "gemm_int8_t",
                "gemm_bf16_b",
                "gemm_int8_b",
            )
        }
        iters = max(1, ITERS // SWEEP_ROUNDS)
        for _ in range(SWEEP_ROUNDS):
            stages["perm_bf16"].append(_xpu_time_ms(lambda: hidden.index_select(0, index), iters=iters))
            stages["perm_int8"].append(
                _xpu_time_ms(
                    lambda: (qact_batch.index_select(0, index), ascale_batch.index_select(0, index)), iters=iters
                )
            )
            stages["quant_torch"].append(_xpu_time_ms(lambda: _quantize_rows(hidden), iters=iters))
            stages["gemm_bf16_t"].append(_xpu_time_ms(lambda: _call(permuted_bf16), iters=iters))
            stages["gemm_int8_t"].append(_xpu_time_ms(lambda: _call(permuted_int8, permuted_scale), iters=iters))
            stages["gemm_bf16_b"].append(_xpu_time_ms(lambda: _call(hidden, ntpe=ntpe_batch), iters=iters))
            stages["gemm_int8_b"].append(
                _xpu_time_ms(lambda: _call(qact_batch, ascale_batch, ntpe=ntpe_batch), iters=iters)
            )
        t = {k: min(v) for k, v in stages.items()}

        # The fused quantizer has no standalone entry point, so difference the
        # same GEMM with 16-bit and int8 input: identical shape, identical
        # weights, identical output -- the only work that differs is the
        # in-kernel quantization of exactly those rows.
        quant_fused_t = max(t["gemm_bf16_t"] - t["gemm_int8_t"], 0.0)
        quant_fused_b = max(t["gemm_bf16_b"] - t["gemm_int8_b"], 0.0)
        # Should be ~top_k: the quantizer is a pure streaming pass, so its cost
        # is linear in rows. A ratio far from top_k means one of the two
        # differences is noise rather than signal.
        fused_ratio = (quant_fused_t / quant_fused_b) if quant_fused_b > 0 else None

        perm_bf16_ms, perm_int8_ms = t["perm_bf16"], t["perm_int8"]
        gemm_ms = t["gemm_int8_t"]
        today_ms = perm_bf16_ms + t["gemm_bf16_t"]
        dedup_torch_ms = perm_int8_ms + t["quant_torch"] + gemm_ms
        dedup_fused_ms = perm_int8_ms + quant_fused_b + gemm_ms

        # The deduplicated path must not change the result: the row absmax is
        # expert-independent, so quantizing before the permute and permuting
        # the int8 is the same arithmetic on the same row values.
        snr_db = _snr_db(_call(permuted_bf16).to(torch.float32), _call(permuted_int8, permuted_scale).to(torch.float32))

        rows.append(
            {
                "label": nk_label,
                "E": E,
                "N": N,
                "K": K,
                "tokens": total_tokens,
                "batch": n_tokens,
                "topk": topk,
                "today_ms": today_ms,
                "dedup_ms": dedup_fused_ms,
                "dedup_torch_ms": dedup_torch_ms,
                "permute_bf16_ms": perm_bf16_ms,
                "permute_int8_ms": perm_int8_ms,
                "gemm_int8_ms": gemm_ms,
                "quant_torch_ms": t["quant_torch"],
                "quant_fused_batch_ms": quant_fused_b,
                "quant_fused_tokens_ms": quant_fused_t,
                "fused_ratio": fused_ratio,
                "speedup": (today_ms / dedup_fused_ms) if dedup_fused_ms else None,
                "speedup_torch": (today_ms / dedup_torch_ms) if dedup_torch_ms else None,
                "snr_db": snr_db,
            }
        )
        if verbose:
            for path, perm_ms, quant_ms, total_ms in (
                ("in-call quant", perm_bf16_ms, quant_fused_t, today_ms),
                ("dedup, torch quant", perm_int8_ms, t["quant_torch"], dedup_torch_ms),
                ("dedup, fused quant", perm_int8_ms, quant_fused_b, dedup_fused_ms),
            ):
                speedup = None if total_ms == today_ms else (today_ms / total_ms if total_ms else None)
                print(
                    f"{nk_label:<14}{E:>5}{N:>7}{K:>7}{total_tokens:>8}{n_tokens:>8}{topk:>6}"
                    f"{path:>21}{perm_ms:>13.3f}{quant_ms:>11.3f}{gemm_ms:>11.3f}{total_ms:>12.3f}"
                    f"{(f'{speedup:.2f}x' if speedup else '--'):>12}"
                )

        case = weights_s8 = wscales = hidden = None
        qact_batch = ascale_batch = permuted_bf16 = permuted_int8 = permuted_scale = None
        ark.clear_moe_w4a8_prepack_cache()
        ark.moe_w4a8_release_scratch()
        _release_xpu_memory()

    if verbose and rows:
        print()
        print("deduplicated dynamic quantization [prefill] (up/gate only; the down projection has no duplicate rows):")
        for row in rows:
            ratio = row["fused_ratio"]
            print(
                f"  {row['label']:<12} {row['today_ms']:.3f} ms -> {row['dedup_ms']:.3f} ms  "
                f"{row['speedup']:.2f}x end-to-end   (quantizes {row['batch']} rows instead of "
                f"{row['tokens']}, permutes int8 instead of {str(dtype).split('.')[-1]})   "
                f"SNR {row['snr_db']:.1f} dB"
            )
            print(
                f"  {'':<12} with the eager-torch quantizer instead: {row['dedup_torch_ms']:.3f} ms "
                f"({row['speedup_torch']:.2f}x) -- {row['quant_torch_ms']:.3f} ms to quantize "
                f"{row['batch']} rows, vs {row['quant_fused_batch_ms']:.3f} ms fused"
            )
            print(
                f"  {'':<12} fused quant measured by differencing: {row['quant_fused_tokens_ms']:.3f} ms at "
                f"{row['tokens']} rows / {row['quant_fused_batch_ms']:.3f} ms at {row['batch']} rows = "
                f"{(f'{ratio:.1f}x' if ratio else 'n/a')} for {row['topk']}x the rows"
                f"{'' if ratio and 0.5 * row['topk'] <= ratio <= 2.0 * row['topk'] else '  <- NOT linear, treat as noise'}"
            )
    return rows


# ---------------------------------------------------------------------------
# Kernel-configuration sweeps
#
# Both perf targets depend on a dispatch choice that can only be settled on
# hardware (how many bytes a decode lane keeps in flight; how wide a prefill
# tile should be). Every choice is reachable through an environment variable
# and re-read per call, so one run can time them all against the same
# workload -- built and prepacked once -- and name the winner.
#
# Each configuration is also checked against the first one for numerical
# equivalence, so a configuration that is fast because it computes the wrong
# thing cannot win.
# ---------------------------------------------------------------------------

# Decode: (label, env overrides). ``None`` unsets a variable.
_DECODE_CONFIGS = [
    ("legacy gemv", {"ARK_MOE_W4A8_DECODE_KSPLIT": "0"}),
] + [
    (
        f"ksplit ch{ch} ncols{ncols}",
        {
            "ARK_MOE_W4A8_DECODE_KSPLIT": "1",
            "ARK_MOE_W4A8_DECODE_KSPLIT_CH": str(ch),
            "ARK_MOE_W4A8_DECODE_KSPLIT_NCOLS": str(ncols),
        },
    )
    for ch in (16, 32)
    for ncols in (1, 2, 4)
]

# Prefill: work-group tile shapes. ``auto`` is the built-in ladder.
_PREFILL_TILES = ["auto", "128x128", "128x256", "256x128", "256x256"]
_PREFILL_TILE_CONFIGS = [
    (f"tile {tile}", {"ARK_MOE_W4A8_PREFILL_TILE": None if tile == "auto" else tile}) for tile in _PREFILL_TILES
]

# Prefill: activation-quantization message width. The vectorized kernel hands
# each lane ``VEC`` *consecutive* elements, turning the scalar mapping's 32-byte
# loads / 16-byte stores into 256-byte / 128-byte ones. At prefill sizes this
# pass streams ~200 MB next to the GEMM's ~400 MB, so it is a real share of the
# call, not a preamble.
_ACT_QUANT_CONFIGS = [
    ("act-quant scalar", {"ARK_MOE_W4A8_ACT_QUANT_VEC": "0"}),
    ("act-quant vec", {"ARK_MOE_W4A8_ACT_QUANT_VEC": "1"}),
]

# Prefill: activation-quantization loads in flight. Widening the messages set
# how many bytes each *request* moves; this sets how many requests a work-item
# has outstanding. ``UNROLL`` vectors are loaded before any is consumed, and
# reduced into ``UNROLL`` partial maxima so they do not serialize behind the
# accumulator either. ``1`` is the kernel as it was before the batching, so the
# first row is an exact A/B baseline.
_ACT_QUANT_UNROLL_CONFIGS = [
    (
        f"act-quant unroll {u}",
        {"ARK_MOE_W4A8_ACT_QUANT_VEC": "1", "ARK_MOE_W4A8_ACT_QUANT_UNROLL": str(u)},
    )
    for u in (1, 2, 4)
]

# Prefill: activation-quantization passes over the row. The two-pass kernel
# reads ``[T, K]`` twice -- once for the absmax, once to quantize -- because the
# scale is not known until the row has been seen. The single-pass kernel keeps
# the row in registers between the two, which removes the second read at the
# cost of ``K / 16`` elements of register pressure per lane. Whether that trade
# pays depends on whether it spills, which only hardware can say; rows longer
# than the register budget take the two-pass kernel either way.
_ACT_QUANT_SINGLE_PASS_CONFIGS = [
    (
        "act-quant two-pass",
        {"ARK_MOE_W4A8_ACT_QUANT_VEC": "1", "ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS": "0"},
    ),
    (
        "act-quant single-pass",
        {"ARK_MOE_W4A8_ACT_QUANT_VEC": "1", "ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS": "1"},
    ),
]

# Prefill: epilogue guard. A tile that touches neither the M nor the N edge
# needs no store predicate and no scale-index clamp, and whether it does is
# uniform across the work-group. The guarded path is what every tile used to
# run; it stays reachable so the saving can be measured (it is largest where
# the mainloop is shortest, i.e. small K).
_EPILOGUE_CONFIGS = [
    ("epilogue guarded", {"ARK_MOE_W4A8_PREFILL_FULL_TILE": "0"}),
    ("epilogue interior", {"ARK_MOE_W4A8_PREFILL_FULL_TILE": "1"}),
]

# Prefill: how D leaves the registers. The DPAS C fragment gives a lane one
# column of each 8x16 atom, so a scalar store is a 32-byte message for 16-bit D
# and a 32x32 sub-group fragment issues 64 of them; the hardware 2D block store
# moves the same bytes in a handful of messages and needs no predicate, because
# it clips to the output surface. D is a third of the tile traffic on the
# down-projection shapes, so this is where the store width should show.
_PREFILL_STORE_CONFIGS = [
    ("store scalar", {"ARK_MOE_W4A8_PREFILL_STORE_2D": "0"}),
    ("store block2d", {"ARK_MOE_W4A8_PREFILL_STORE_2D": "1"}),
]

# Prefill: how many k-tiles the mainloop keeps prefetched ahead of the tile it
# is computing. The prologue issues this many A/B prefetch pairs before the
# first DPAS and the loop then issues one pair per tile, so it is the depth of
# the memory pipeline the DPAS chain runs against. 3 is what the kernel was
# written with; the shipped shapes are short in K (12 k-tiles at K = 768, where
# a prologue of 3 is a quarter of the whole mainloop), which is exactly the
# regime where the depth is worth re-measuring in both directions.
#
# The sweep spans the whole range the kernel accepts (1-8) rather than a window
# around the default. The narrower `2 / 3 / 4 / 6` it used to probe cannot
# distinguish "3 is the optimum" from "3 is the best of the four points that
# were tried": both endpoints are legal values the clamp in
# `moe_w4a8_prefill_prefetch_dist` admits, and on a 12-k-tile mainloop they are
# the two hypotheses that matter -- 1 is the shortest prologue the kernel can
# have, 8 is deep enough to cover two thirds of that mainloop before the first
# DPAS. A K-aware default can only be justified by a sweep that includes them.
_PREFILL_PREFETCH_CONFIGS = [
    (f"prefetch {dist}", {"ARK_MOE_W4A8_PREFILL_PREFETCH": str(dist)}) for dist in (1, 2, 3, 4, 6, 8)
]

# Prefill: when the persistent kernel asks for its next tile.
#
# The grid is sized to the device, not to the problem -- `sm_count` times the
# work-groups that fit on an Xe core -- and the tiles are handed out by a
# device-scope `atomicAdd` on a single dword that every resident work-group
# hits. Claiming *after* the GEMM puts that L2 round trip in the gap between
# two tiles, where the work-group has nothing to overlap it with; claiming
# before puts the message in flight across the DPAS mainloop and only the use
# of its result waits.
#
# The size of that stall is bounded by how much work a tile has to hide it
# behind, so this should show on the down projections and not much on the up
# ones: at K = 768 a tile is 12 k-tiles against 32 at K = 2048, and the down
# projection is where the measured TFLOPS sits furthest below the up
# projection's on the same tile shape and the same kernel. If the two rows tie
# on every shape, the claim was never the stall and this sweep says so.
_PREFILL_CLAIM_CONFIGS = [
    ("claim after gemm", {"ARK_MOE_W4A8_PREFILL_CLAIM_EARLY": "0"}),
    ("claim before gemm", {"ARK_MOE_W4A8_PREFILL_CLAIM_EARLY": "1"}),
]

# Prefill: the register budget, and therefore how many work-groups are resident.
#
# `grf_size` used to be hardwired at 256 in the launcher, which pinned every
# tile in the ladder at 4 hardware threads per vector engine -- 512 work-items
# per Xe-core, which is exactly the `MaxThreadsPerSM` the persistent grid is
# sized from. Asking for 128 registers instead doubles both. The small tiles are
# in the ladder *because* they ask for fewer registers, so every one of them has
# been swept so far paying that cost without ever collecting the occupancy that
# is their reason to exist.
#
# The flag alone would sweep nothing: the ladder sends every shipped qwen3 shape
# to `128x256`, whose 32x64 sub-group accumulator is 128 int32 registers -- the
# entire small-GRF file, nothing left for the staged A/B tiles -- so that policy
# declines the halved budget and both settings launch the identical kernel. The
# tile therefore has to be forced for the knob to be observable at all, and the
# first two rows do exactly that: same 128x128 tile, same everything else, one
# register budget each. That pair is the measurement.
#
# The third row is the shipped default, and it is what the pair is *for*. The
# question is not whether 128 registers beat 256 on the same tile -- it is
# whether the smaller tile at double occupancy finally catches the bigger tile
# that has beaten it in every sweep so far. If row 2 beats row 1 but neither
# reaches row 3, the ladder stays as it is and the answer is that the 256-wide N
# tile wins on A re-reads, not on occupancy.
#
# Expect the effect, if any, on the down projection: occupancy hides latency,
# half of qwen3 down's time is per-tile cost that does not scale with K, and
# ~78% of that is the 64 KB of D each tile writes. Covering a store stream with
# other work-groups is what more residency buys. The up projection is at 86% of
# the DPAS peak with the quantizer subtracted and has nothing to hide.
_PREFILL_GRF_CONFIGS = [
    ("128x128, grf 256", {"ARK_MOE_W4A8_PREFILL_TILE": "128x128", "ARK_MOE_W4A8_PREFILL_SMALL_GRF": "0"}),
    ("128x128, grf 128", {"ARK_MOE_W4A8_PREFILL_TILE": "128x128", "ARK_MOE_W4A8_PREFILL_SMALL_GRF": "1"}),
    ("128x256 (default)", {"ARK_MOE_W4A8_PREFILL_TILE": "128x256"}),
]

# Prefill: the two call contracts that cut traffic instead of cycles.
#
# Neither changes the GEMM. They change what crosses the call boundary, which
# is where the remaining traffic is: on the qwen3 down-projection at 384
# rows/expert the call moves 528 MB, of which only 36% is weights -- 27% is the
# activation quantization round-trip (read fp16, write int8, read the int8
# back) and 36% is a `[T, N]` output that the caller immediately reduces to
# `[batch, N]`. Both are redundancies of the *interface*: the producer of the
# activations already writes `[T, K]` once and could write int8, and the
# reduction the caller performs can be done in the epilogue while the values
# are still in registers.
#
# The fused row is not bit-identical to the others -- fp32 atomics do not
# commit in a fixed order, and the unfused baseline additionally rounds each
# row to the activation dtype before the caller's reduction sees it -- so this
# is the one sweep whose SNR column is a quality gate rather than an identity
# check.
_PREFILL_CONTRACT_CONFIGS = [
    ("A quant in-call", {}, {}),
    ("A int8 in", {}, {"prequant": True}),
    ("fused reduce", {}, {"fused": True}),
    ("A int8 + fused", {}, {"prequant": True, "fused": True}),
]

_SWEEP_MIN_SNR_DB = 40.0


def _sweep_timings(configs, call, extra=None):
    """Time every configuration, round-robin, and return the per-round samples.

    Timing each configuration to completion in turn -- warmup and all its
    iterations, then on to the next -- silently ranks by position as much as
    by configuration. These shapes draw enough power to droop the clock over
    a sweep, so every configuration measured later is handicapped by the
    heat the earlier ones produced. On the qwen3 up-proj shape that artefact
    is worth 6-7% run to run, which is *larger* than the spread the sweep is
    being used to rank: two runs of the prefill prefetch sweep an hour apart
    put the optimum at depth 3-4 and then at depth 1-2, with the ordering
    essentially reversed and the last-measured configuration slowest in both.

    Round-robin instead, so the drift is spread evenly across configurations
    rather than accumulating against the later ones. The caller takes the
    minimum across rounds (each configuration at its least-throttled) and
    keeps the spread as the noise floor a ranking has to clear.

    ``call`` is invoked as ``call(call_kwargs)`` with the configuration's env
    overrides in force. ``extra`` is an optional zero-argument callable timed
    once per round alongside them, for work that is charged to some
    configurations but is not itself one. Returns
    ``(per_config_samples, extra_samples)``, the former positionally aligned
    with ``configs``.
    """
    samples = [[] for _ in configs]
    extra_samples = []
    iters = max(1, ITERS // SWEEP_ROUNDS)
    for _ in range(SWEEP_ROUNDS):
        for idx, (_, overrides, call_kwargs) in enumerate(configs):
            with _env_override(**overrides):
                samples[idx].append(_xpu_time_ms(lambda: call(call_kwargs), iters=iters))
        if extra is not None:
            # Timed inside the round-robin, not after it: a measurement taken
            # once at the end of the sweep lands at the sweep's hottest point
            # and would be biased high against the configurations it is added
            # to, which is the artefact the round-robin exists to remove.
            extra_samples.append(_xpu_time_ms(extra, iters=iters))
    return samples, extra_samples


def _print_sweep_header(title: str, metric: str, with_reduce: bool = False) -> None:
    print()
    print("=" * _PERF_WIDTH)
    print(title)
    extra = f"{'+reduce':>10}" if with_reduce else ""
    print(
        f"{'shape':<14}{'E':>5}{'N':>7}{'K':>7}{'tokens':>8}{'rows/E':>8}  "
        f"{'config':<22}{'ms':>10}{extra}{metric:>10}{'vs default':>12}{'SNR(dB)':>10}{'drift':>9}"
    )
    print("-" * _PERF_WIDTH)


def run_config_sweep(phase, configs, dtype=torch.bfloat16, models=None, verbose=True, compute_bound=None, batches=None):
    """Time every kernel configuration in ``configs`` on the same workload.

    ``phase`` picks the sweep's shapes and its metric: ``decode`` reports
    weight bandwidth (the decode target), ``prefill`` reports TFLOPS (the
    prefill target) at the compute-bound batch. ``batches`` runs the sweep at
    an explicit list of model-token batches instead -- what the long-prompt
    sweeps use, since a fixed prompt length is exactly what the compute-bound
    derivation replaces. Returns one dict per (shape, batch, configuration).

    A configuration is ``(label, env_overrides)`` or, for the sweeps that
    compare *call contracts* rather than dispatch knobs,
    ``(label, env_overrides, call_kwargs)`` -- the extra dict is forwarded to
    :func:`_w4a8`. A contract that reduces inside the kernel returns a
    ``[batch, N]`` tensor where the others return ``[T, N]``, so outputs are
    put in the same frame (:func:`_reduce_topk`) before they are compared.

    When some configuration fuses the top-k reduction, timing the others as a
    bare GEMM would rank them against a baseline that skips work: the fused row
    pays for the reduction inside its epilogue while the unfused rows leave a
    ``[T, N]`` tensor their caller still has to reduce. Every row therefore also
    carries the caller-side reduction it would owe (``+reduce``, zero for the
    fused rows), and ``vs default`` ranks on the sum -- the cost of *producing
    the routed output*, which is the thing the contract actually changes.

    That sum brackets the win rather than pinning it. The reduction is timed as
    :func:`_reduce_topk`, a torch ``index_add_`` that materializes fp32
    temporaries, so it is an upper bound on a hand-written one; the GEMM-only
    ``ms`` column is the lower bound. The truth is between the two columns,
    which is why both are printed.
    """
    is_prefill = phase == "prefill"
    # An explicit batch list opts out of the compute-bound derivation, which
    # ignores ``batches`` and sizes the batch from the target rows/expert.
    if compute_bound is None:
        compute_bound = is_prefill and batches is None
    metric_name = "TFLOPS" if is_prefill else "W GB/s"
    device_bw = _device_bandwidth_gbps()
    resolved = _models(models)
    configs = [(cfg[0], cfg[1], cfg[2] if len(cfg) > 2 else {}) for cfg in configs]
    # Only build the routing side tables when some configuration asks for them.
    need_routing = any(kwargs.get("fused") for _, _, kwargs in configs)
    if verbose:
        _print_sweep_header(
            f"W4A8 config sweep [{phase}] (models={'+'.join(n for n, _ in resolved)}, "
            f"act={str(dtype).split('.')[-1]}) -- same workload, one row per kernel configuration",
            metric_name,
            with_reduce=need_routing,
        )
    if batches is None:
        batches = _DECODE_BATCHES
    rows = []
    for _, spec in resolved:
        E, topk, group_size = spec["E"], spec["topk"], spec["group_size"]
        for nk_label, N, K in spec["nk"]:
            for batch in _compute_bound_batches(spec) if compute_bound else batches:
                total_tokens = batch * topk
                case = _build_case(
                    N,
                    K,
                    E,
                    total_tokens,
                    group_size,
                    dtype,
                    need_reference=False,
                    need_dequant=False,
                    topk=topk if need_routing else None,
                )
                weights_s8, wscales, block = ark.moe_w4a8_prepack(
                    case["packed"], case["scales"], group_size=group_size, rescale_group_size=-1
                )
                active_experts = sum(1 for n_e in case["tpe"] if n_e > 0)
                rows_per_expert = _rows_per_expert(total_tokens, active_experts)

                # Correctness first, one configuration at a time, so that only
                # the baseline output and the one being compared to it are
                # alive: a six-configuration sweep that held every output at
                # once would need six [T, N] tensors (1.2 GB on the qwen3
                # up-proj shape).
                baseline_out = None
                snrs = []
                for label, overrides, call_kwargs in configs:
                    with _env_override(**overrides):
                        out = _w4a8(case, weights_s8, wscales, block, phase, **call_kwargs)
                    if need_routing and not call_kwargs.get("fused"):
                        out = _reduce_topk(case, out)
                    if baseline_out is None:
                        # Cloned: the kernel may hand back a reused scratch
                        # buffer, which would make every later comparison
                        # compare a tensor with itself.
                        baseline_out = out.clone()
                        snrs.append(float("inf"))
                    else:
                        snrs.append(_snr_db(baseline_out.to(torch.float32), out.to(torch.float32)))
                    out = None
                baseline_out = None

                # What an unfused caller still owes after the GEMM returns --
                # but only on the projection whose output is actually reduced.
                # Timed inside the round-robin so it is min-filtered on the
                # same basis as the configurations it is charged to.
                charge_reduce = need_routing and _reduces_topk(nk_label)
                unreduced = _w4a8(case, weights_s8, wscales, block, phase) if charge_reduce else None
                samples, reduce_samples = _sweep_timings(
                    configs,
                    lambda kw: _w4a8(case, weights_s8, wscales, block, phase, **kw),
                    extra=(lambda: _reduce_topk(case, unreduced)) if charge_reduce else None,
                )
                reduce_ms = min(reduce_samples) if reduce_samples else 0.0
                unreduced = None
                _release_xpu_memory()

                baseline_ms = None
                for idx, (label, overrides, call_kwargs) in enumerate(configs):
                    # The least-throttled round, and the spread across rounds
                    # as the noise floor this shape's ranking has to clear.
                    ms = min(samples[idx])
                    noise = (max(samples[idx]) - ms) / ms if ms else 0.0
                    snr = snrs[idx]
                    # The fused epilogue has already done the reduction; every
                    # other configuration leaves it for the caller.
                    row_reduce_ms = 0.0 if call_kwargs.get("fused") else reduce_ms
                    total_ms = ms + row_reduce_ms
                    if baseline_ms is None:
                        baseline_ms = total_ms
                    tflops = _flops(total_tokens, N, K) / (ms * 1e-3) / 1e12
                    gbps = _weight_bytes(active_experts, N, K, 8) / (ms * 1e-3) / 1e9
                    row = {
                        "label": nk_label,
                        "phase": phase,
                        "config": label,
                        "overrides": overrides,
                        "call_kwargs": call_kwargs,
                        "E": E,
                        "N": N,
                        "K": K,
                        "tokens": total_tokens,
                        "rows_per_expert": rows_per_expert,
                        "w4a8_ms": ms,
                        "reduce_ms": row_reduce_ms,
                        "total_ms": total_ms,
                        "noise": noise,
                        "tflops": tflops,
                        "gbps": gbps,
                        "snr_db": snr,
                        "speedup": baseline_ms / total_ms if total_ms else None,
                        "device_bw_gbps": device_bw,
                    }
                    rows.append(row)
                    if verbose:
                        metric = tflops if is_prefill else gbps
                        snr_txt = "--" if math.isinf(snr) else f"{snr:.1f}"
                        speedup_txt = "--" if row["speedup"] is None else f"{row['speedup']:.2f}x"
                        reduce_txt = f"{row_reduce_ms:>10.3f}" if need_routing else ""
                        print(
                            f"{nk_label:<14}{E:>5}{N:>7}{K:>7}{total_tokens:>8}{rows_per_expert:>8.1f}  "
                            f"{label:<22}{ms:>10.3f}{reduce_txt}{metric:>10.2f}{speedup_txt:>12}{snr_txt:>10}"
                            f"{noise * 100:>8.1f}%"
                        )
                    out = None
                case = weights_s8 = wscales = baseline_out = None
                ark.clear_moe_w4a8_prepack_cache()
                ark.moe_w4a8_release_scratch()
                _release_xpu_memory()
    if verbose:
        _print_sweep_best(phase, rows)
    return rows


def _ranked_ms(row):
    """The time a configuration is ranked on: GEMM plus any reduction it left.

    Equal to ``w4a8_ms`` for every sweep that has no fused-reduce
    configuration, so the dispatch-knob sweeps rank exactly as before.
    """
    return row.get("total_ms", row["w4a8_ms"])


def _ranked_noise(row):
    """``row``'s round-to-round drift as a fraction of the time it is ranked on.

    The drift is measured on the GEMM call, so charging a configuration an
    unmeasured constant on top of it dilutes the same absolute jitter.
    """
    ranked = _ranked_ms(row)
    if not ranked:
        return row.get("noise", 0.0)
    return row.get("noise", 0.0) * row["w4a8_ms"] / ranked


def _print_sweep_best(phase, rows) -> None:
    """Report, per shape, the fastest numerically-equivalent configuration.

    A shape swept at more than one batch gets one line per batch: the winner
    is a property of the routing as much as of the shape (the tile ladder
    changes rung with ``rows/expert``), so the two must not be pooled.
    """
    if not rows:
        return
    is_prefill = phase == "prefill"
    print()
    print(f"best configuration [{phase}] (equivalent within {_SWEEP_MIN_SNR_DB:g} dB SNR of the first configuration):")
    shapes = []
    for row in rows:
        if row["label"] not in shapes:
            shapes.append(row["label"])
    for shape in shapes:
        shape_rows = [r for r in rows if r["label"] == shape]
        batches = []
        for row in shape_rows:
            if row["tokens"] not in batches:
                batches.append(row["tokens"])
        for tokens in batches:
            group = [r for r in shape_rows if r["tokens"] == tokens]
            name = shape if len(batches) == 1 else f"{shape.strip()} T={tokens}"
            candidates = [r for r in group if r["snr_db"] >= _SWEEP_MIN_SNR_DB]
            if not candidates:
                print(f"  {name:<14} no numerically-equivalent configuration")
                continue
            best = min(candidates, key=_ranked_ms)
            # A winner is only a winner if its lead over the sweep's own
            # first (= default) configuration is larger than the drift the
            # sweep measured on itself. Below that the ranking is reporting
            # which configuration happened to be timed on the coolest
            # device, and recommending it bakes an artefact into a default.
            # ``noise`` is a spread relative to the GEMM time, but the lead is
            # a ratio of GEMM+reduce times. Adding the same reduction to both
            # sides of that ratio shrinks it, so the gate has to be put on the
            # same basis or a real win between two unfused configurations gets
            # discarded as drift.
            noise = max(_ranked_noise(r) for r in group)
            lead = _ranked_ms(group[0]) / _ranked_ms(best) - 1.0 if _ranked_ms(best) else 0.0
            if lead <= noise:
                print(
                    f"  {name:<14} inconclusive: best is {lead * 100:.1f}% ahead of the default, "
                    f"within the {noise * 100:.1f}% round-to-round drift -- keep the default"
                )
                continue
            metric = f"{best['tflops']:.2f} TFLOPS" if is_prefill else f"{best['gbps']:.1f} GB/s"
            parts = [f"{k}={v}" for k, v in sorted(best["overrides"].items()) if v is not None]
            parts += [f"{k}={v}" for k, v in sorted(best.get("call_kwargs", {}).items()) if v]
            env = " ".join(parts) or "(defaults)"
            # The metric is derived from the GEMM time, so the GEMM time is what
            # is printed beside it; the reduction is shown as a separate term so
            # the ranked total stays recoverable without implying the metric was
            # computed from it.
            ms_txt = f"{best['w4a8_ms']:.3f} ms"
            if best.get("reduce_ms"):
                ms_txt += f" + {best['reduce_ms']:.3f} reduce"
            print(f"  {name:<14} {best['config']:<22} {ms_txt:<26} {metric:<16} {env}")


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
            rows = run_perf("decode", _decode_batches(all_shapes), models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            _assert_targets(request, "decode", rows)

        def test_perf_prefill(self, request):
            all_shapes = request.config.getoption("--all-shapes", default=False)
            rows = run_perf("prefill", _prefill_batches(all_shapes), models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            _assert_targets(request, "prefill", rows)

        def test_perf_prefill_compute_bound(self, request):
            """Prefill throughput at a batch where the TFLOPS target is reachable.

            ``test_perf_prefill`` runs 128 model tokens, i.e. 8 rows per
            expert: at that routing the grouped GEMM only does 16 FLOPs per
            weight byte, so it is pinned to the DRAM roofline and no amount of
            kernel work can push it to 100 TFLOPS. This case routes enough
            tokens to put ``_PREFILL_TARGET_ROWS_PER_EXPERT`` rows on every
            expert, which is the smallest routing where 100 TFLOPS is under the
            device's bandwidth ceiling for *all* the shipped shapes -- counting
            the activation, quantized-activation and output streams, not just
            the weights.
            """
            rows = run_perf("prefill", None, torch_baseline=False, compute_bound=True, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            _assert_targets(request, "prefill", rows)

        def test_perf_prefill_long_seq(self, request):
            """Prefill throughput for one 8K-token prompt.

            ``test_perf_prefill_compute_bound`` derives its batch per model so
            that both land on the same 384 rows per expert. A prompt does not
            work that way: a fixed 8192 model tokens (65536 routed rows) is
            divided by whatever expert count the model has, which puts 512 rows
            on each of Qwen3-MoE's 128 experts and 341 on each of MiniMax's
            192.

            Both directions are worth measuring. More rows per expert raise the
            arithmetic intensity -- the weights are the only stream that does
            not grow with the token count -- so the qwen3 ceilings rise from
            129 / 112 to 145 / 123 TFLOPS at a 400 GB/s probe, which is the
            regime where the 100 TFLOPS target has the most margin. Fewer rows
            lower it, so minimax comes down from 137 / 154 to 129 / 145 and the
            same kernel should read *slower* there: the point of running both
            is that the prompt length alone does not decide the throughput, the
            routing it produces does.
            """
            rows = run_perf(
                "prefill",
                _long_seq_batches(),
                torch_baseline=False,
                models=_models_option(request),
            )
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            _assert_targets(request, "prefill", rows)

        def test_perf_decode_config_sweep(self, request):
            """Time every decode lane mapping on one workload and name the best.

            The decode target is weight bandwidth, and how close the GEMV gets
            to the device's streaming rate is decided by how many bytes a lane
            keeps in flight (``CH``) and how many output columns a sub-group
            blocks over (``NCOLS``). Both are dispatch-time environment knobs,
            so a single run can measure the whole grid -- including the legacy
            mapping -- against the same prepacked weights.
            """
            rows = run_config_sweep("decode", _DECODE_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"decode config {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_tile_sweep(self, request):
            """Time every prefill work-group tile at the compute-bound batch.

            A ``TileM x TileN`` tile re-reads A once per N tile and B once per M
            tile, so the tile shape sets the GEMM's tile-load traffic
            (``~ M*N*K * (1/TileM + 1/TileN)``) and therefore whether a
            compute-bound shape is actually limited by compute. This sweep
            measures the ladder's choice against every explicit tile.
            """
            rows = run_config_sweep("prefill", _PREFILL_TILE_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"prefill tile {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_tile_sweep_long_seq(self, request):
            """Time every prefill work-group tile at the 8K-prompt routing.

            The tile sweep above runs at the derived compute-bound batch, where
            every model sits at 384 rows per expert -- a routing at which the
            256-row tile can only lose, because it would schedule 512 rows for
            384 rows of data. A fixed 8K prompt is the routing that changes
            that: Qwen3-MoE's 128 experts get 512 rows each, an exact multiple
            of 256, so both candidate ``TileM`` values schedule the same rows
            and the sweep measures the tile rather than its padding. It is the
            only case in the suite that can, which is what makes it the
            evidence for (or, as it turned out, against) a 256-row rung.
            """
            rows = run_config_sweep(
                "prefill",
                _PREFILL_TILE_CONFIGS,
                models=_models_option(request),
                batches=_long_seq_batches(),
            )
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"prefill tile {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_act_quant_sweep(self, request):
            """Time both activation-quantization mappings at the compute-bound batch.

            Quantizing the routed activations is a pure streaming pass -- read
            ``[T, K]`` twice, write ``[T, K]`` int8 -- so its cost is set by how
            wide a message the sub-group issues, not by arithmetic. The scalar
            mapping (``k = lane; k += SG_SIZE``) stores 16 bytes per message, a
            quarter of a cache line; the vectorized one gives each lane
            ``VEC`` consecutive elements instead. Both are dispatch-time
            choices, so one run measures the pair against the same workload.
            """
            rows = run_config_sweep("prefill", _ACT_QUANT_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"act-quant config {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_act_quant_unroll_sweep(self, request):
            """Time the activation quantizer's loads-in-flight depth.

            Widening the messages (the sweep above) set how many bytes each
            request moves; it did not change how many requests a work-item has
            outstanding. The pass walks K with a runtime trip count and folds
            every vector into one ``local_max``, so an in-order thread keeps
            about one 256-byte load in flight -- ~320 KB even at a B60's
            1280-thread occupancy ceiling, under what a 456 GB/s part needs to
            stay busy over a memory latency. ``UNROLL`` loads that many
            independent vectors
            before consuming any of them. The first row is the kernel as it was
            before the batching, so this is an exact A/B measurement; the
            arithmetic is unchanged, which
            ``test_act_quant_unroll_matches`` asserts bit-for-bit.
            """
            rows = run_config_sweep("prefill", _ACT_QUANT_UNROLL_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"act-quant unroll {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_act_quant_single_pass_sweep(self, request):
            """Time the quantizer against its own second read of the row.

            The absmax has to see the whole row before the first element can
            be quantized, so the two-pass kernel reads ``[T, K]``, reduces,
            then reads ``[T, K]`` again. The single-pass kernel keeps the row
            in registers across the reduction and deletes the second read --
            ~0.8 MB per expert at 384 rows and K = 2048, against 3.1 MB of
            weights for the whole GEMM. It costs ``K / 16`` elements of
            register pressure per lane (64 dwords of the default 128-dword
            budget at K = 2048), which is exactly the risk this sweep exists to
            settle: if it spills, the single-pass row is *slower*, and the
            default should be flipped. Rows too long for the budget take the
            two-pass kernel in both rows of the sweep, so those shapes should
            read as noise.
            """
            rows = run_config_sweep("prefill", _ACT_QUANT_SINGLE_PASS_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert row["snr_db"] >= _SWEEP_MIN_SNR_DB, (
                    f"act-quant pass count {row['config']} disagrees with {rows[0]['config']}: "
                    f"SNR {row['snr_db']:.2f} dB"
                )

        def test_perf_prefill_store_sweep(self, request):
            """Time the D store width at the compute-bound batch.

            The mainloop is identical in both rows. The DPAS C fragment hands a
            lane one column of each 8x16 atom, so the 16 lanes of a sub-group
            hold 16 consecutive columns of one row: a scalar store is a
            32-byte message for 16-bit D, and a 32x32 sub-group fragment issues
            64 of them. The 2D block store moves the same bytes in a handful of
            messages and needs no predicate, because it clips to the output
            surface in hardware.

            Expect the largest gain where D is the largest share of tile
            traffic and the mainloop the shortest -- the small-K
            down-projections, which are also the shapes furthest from the
            compute target.
            """
            rows = run_config_sweep("prefill", _PREFILL_STORE_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"store config {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_epilogue_sweep(self, request):
            """Time the epilogue guard at the compute-bound batch.

            The mainloop is identical in both rows; only the store differs. A
            tile that lies entirely inside the expert's rows and inside N can
            skip the per-element store predicate and the two scale-index
            clamps, and the answer is uniform across the work-group, so the
            branch costs one comparison per tile rather than per element. The
            saving is a fixed number of instructions per output element, so it
            shows up as a larger fraction where the mainloop is shortest --
            the small-K down-projections -- and should never be negative.
            """
            rows = run_config_sweep("prefill", _EPILOGUE_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"epilogue config {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_prefetch_sweep(self, request):
            """Time the mainloop's prefetch depth at the compute-bound batch.

            The mainloop is otherwise identical in every row: only how far
            ahead of the computing tile the A/B block prefetches run changes.
            The shipped shapes are short in K -- the qwen3 down-projection has
            12 k-tiles at a 64-element k-tile -- so the default depth of 3 is a
            quarter of the whole mainloop, which is the regime where both
            directions are plausible: deeper hides more latency but spends more
            of the tile in a prologue that computes nothing, and the prefetched
            lines have to survive in L2 until the tile that wants them runs.

            Nothing about the arithmetic changes, so every row must be
            bit-identical to the first; only the timing is a measurement.
            """
            rows = run_config_sweep("prefill", _PREFILL_PREFETCH_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"prefetch depth {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_prefetch_sweep_long_seq(self, request):
            """Time the mainloop's prefetch depth at the 8K-prompt routing.

            Same sweep as above at the other prefill point, and the one that
            has to decide whether the depth should depend on K. The compute-
            bound batch derives its token count per model so every shape lands
            on 384 rows per expert; a fixed 8K prompt does not, so the qwen3
            shapes get 512 rows per expert and the minimax ones 341. That is
            the routing the shipped-contract numbers were taken at, and the one
            where the qwen3 down-projection reads 74% of its DRAM ceiling while
            the other three sit at 91-100% -- i.e. the only shape in the suite
            with headroom a scheduling change could still take.

            K is what separates it: at a 64-element k-tile its mainloop is 12
            iterations against 32 for qwen3 up and 48 for minimax up, so a
            prologue of 3 is a quarter of the loop there and a sixteenth here.
            Whether that costs anything is not decidable from the shape alone
            -- a shorter prologue starts computing sooner but runs against a
            shallower memory pipeline -- so `moe_w4a8_prefill_prefetch_dist`
            deliberately stays a single constant until this sweep separates the
            two shapes. Making it K-aware on the strength of the k-tile count
            alone would be picking one of those two effects by assertion.

            Nothing about the arithmetic changes, so every row must be
            bit-identical to the first; only the timing is a measurement.
            """
            rows = run_config_sweep(
                "prefill",
                _PREFILL_PREFETCH_CONFIGS,
                models=_models_option(request),
                batches=_long_seq_batches(),
            )
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"prefetch depth {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_claim_early_sweep(self, request):
            """Time when the persistent kernel claims its next tile.

            The grid is sized to the device, so a work-group does not own one
            tile -- it loops, taking the next index from a device-scope
            ``atomicAdd`` on a single dword that every resident work-group
            hits. Claimed after the GEMM, that L2 round trip lands in the gap
            between two tiles with nothing to overlap it; claimed before, the
            message is in flight across the whole DPAS mainloop and only the
            store of its result waits on it.

            Which tiles are computed does not change -- one claim per tile,
            same indices, same order -- so every row must be bit-identical to
            the first and only the timing is a measurement.
            """
            rows = run_config_sweep("prefill", _PREFILL_CLAIM_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"claim order {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_claim_early_sweep_long_seq(self, request):
            """Time the tile-claim order at the 8K-prompt routing.

            Same sweep at the other prefill point, and the one that should
            decide it. How much a claim stall costs depends on how much work
            the tile it precedes has to hide it behind, so the effect is
            bounded by the mainloop length: 12 k-tiles on the qwen3 down
            projection against 32 on qwen3 up, at the same tile shape and the
            same kernel. That is also the shape the shipped-contract numbers
            leave furthest below its own bandwidth ceiling, so it is where a
            per-tile fixed cost should be visible at all.

            A tie on every shape is a real result: it says the claim was not
            the stall, and the down projection's gap is the prologue, the
            epilogue or the D write instead.
            """
            rows = run_config_sweep(
                "prefill",
                _PREFILL_CLAIM_CONFIGS,
                models=_models_option(request),
                batches=_long_seq_batches(),
            )
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"claim order {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_grf_sweep(self, request):
            """Time the register budget, which is really a test of occupancy.

            The kernel is persistent and its grid is sized to fill the device
            exactly once, so the number of work-groups launched *is* the
            residency. That number comes from a threads-per-Xe-core constant
            that only holds at ``grf_size<256>``; halving the register request
            doubles it, and the two have to move together or nothing changes.

            The first two rows are the measurement -- one tile, two budgets --
            and the third is the question they answer: whether 128x128 at
            double occupancy catches the 128x256 tile that has beaten it in
            every sweep so far. Every row computes the same tiles from the same
            inputs, so all three must be bit-identical and only the timing is a
            measurement.
            """
            rows = run_config_sweep("prefill", _PREFILL_GRF_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"GRF budget {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_grf_sweep_long_seq(self, request):
            """Time the register budget at the 8K-prompt routing.

            The other prefill point, and the one where the occupancy argument
            is weakest for the small tile: 512 rows/expert divides evenly by
            both 128 and 256, so the padding penalty that decides the ladder
            elsewhere is absent here and the tiles compete on their merits.

            A tie across all three rows would say the down projection's gap is
            not a latency that more resident work-groups can cover, which --
            after prefetch depth, tile order and claim order all came back
            flat -- would close the last of the latency knobs and leave the D
            write as the only remaining prefill lever.
            """
            rows = run_config_sweep(
                "prefill",
                _PREFILL_GRF_CONFIGS,
                models=_models_option(request),
                batches=_long_seq_batches(),
            )
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"GRF budget {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_contract_sweep(self, request):
            """Time the two traffic-cutting call contracts at the compute-bound batch.

            This is the sweep that decides whether the qwen3 shapes can reach
            the 100 TFLOPS target at all. Every kernel configuration above
            moves the same bytes and competes for the same ~60-75% of the
            device's bandwidth; the target on the down-projection needs 358
            GB/s of a ~390 GB/s part, which no scheduling change reaches. The
            contracts are the only levers that change the numerator: handing
            the kernel int8 activations removes 27% of the call's traffic and
            reducing in the epilogue removes another 27%.

            Both rows are also a correctness check on the harness's own model:
            the ``TFLOPS`` column here is directly comparable to
            ``test_perf_prefill_compute_bound`` because the workload, the
            weights and the routing are the same objects.
            """
            rows = run_config_sweep("prefill", _PREFILL_CONTRACT_CONFIGS, models=_models_option(request))
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            for row in rows:
                assert (
                    row["snr_db"] >= _SWEEP_MIN_SNR_DB
                ), f"call contract {row['config']} disagrees with {rows[0]['config']}: SNR {row['snr_db']:.2f} dB"

        def test_perf_prefill_dedup_quant_long_seq(self, request):
            """End-to-end win from deduplicating the dynamic quantization.

            The answer for a caller whose upstream operator has no int8 to
            hand over. The quantization still has to happen -- it just does not
            have to happen ``top_k`` times per token. See :func:`run_dedup_quant`
            for why the row absmax makes that safe, and why both paths are
            timed with the caller's permute included.

            Asserts the two paths agree: the deduplicated path is the same
            arithmetic on the same row values, so the only permitted difference
            is the few-ulp fp32 division noise :func:`_quantize_rows` documents.
            """
            rows = run_dedup_quant(models=_models_option(request))
            assert rows and all(r["today_ms"] > 0 and r["dedup_ms"] > 0 for r in rows)
            for row in rows:
                assert row["snr_db"] >= _SWEEP_MIN_SNR_DB, (
                    f"deduplicated quantization changed {row['label']}: "
                    f"SNR {row['snr_db']:.2f} dB below {_SWEEP_MIN_SNR_DB}"
                )
                # The fused quantizer's cost is obtained by differencing two
                # GEMM timings, so it is only meaningful if the difference is
                # signal. A pure streaming pass must scale with rows: if
                # quantizing top_k times as many rows does not cost roughly
                # top_k times as much, the differences are measurement noise
                # and every number derived from them is meaningless.
                ratio, topk = row["fused_ratio"], row["topk"]
                assert ratio is not None and 0.5 * topk <= ratio <= 2.0 * topk, (
                    f"fused activation quantization does not scale with rows on {row['label']}: "
                    f"{row['quant_fused_tokens_ms']:.3f} ms at {row['tokens']} rows vs "
                    f"{row['quant_fused_batch_ms']:.3f} ms at {row['batch']} rows "
                    f"({'n/a' if ratio is None else f'{ratio:.2f}x'} for {topk}x the rows) -- "
                    f"the GEMM difference is noise, not the quantizer"
                )

        def test_perf_prefill_prequant_long_seq(self, request):
            """Prefill throughput for one 8K prompt with the int8-in contract alone.

            The two contracts were only ever measured together, and together
            they regress on B70 (see the README): the fused epilogue's cost
            swamps whatever the activation round-trip saves, so that run says
            nothing about the round-trip itself. This isolates it.

            It is also the contract a serving stack can actually adopt without
            touching its layer code -- the previous op already produced int8 in
            most quantized pipelines -- whereas the fused reduction requires the
            caller to hand over the routing map and give up a bit-identical
            result. Removing the round-trip drops ``3 * T * K`` bytes, a third
            of the up projection's traffic, and the traffic model here is told
            about it so the printed ceiling matches what the call moves.
            """
            rows = run_perf(
                "prefill",
                _long_seq_batches(),
                torch_baseline=False,
                models=_models_option(request),
                prequantized=True,
            )
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            _assert_targets(request, "prefill", rows)

        def test_perf_prefill_contracts_long_seq(self, request):
            """Prefill throughput for one 8K prompt with both contracts enabled.

            ``test_perf_prefill_long_seq`` measures the shipped contract, where
            the qwen3 shapes are bandwidth-bound below the target. This runs
            the same prompt with the activation round-trip removed, and the
            output reduction folded into the epilogue on the one projection
            that is actually reduced; the ceiling printed next to it is
            computed from the same reduced traffic model, so the verdict is
            against the right roof.

            On B70 this is *slower* than the shipped contract even though it
            moves fewer bytes -- the fused epilogue trades a coalesced 2D block
            store for one device-scope atomic per output element. Compare
            against ``test_perf_prefill_prequant_long_seq`` to separate the two
            contracts' contributions.
            """
            rows = run_perf(
                "prefill",
                _long_seq_batches(),
                torch_baseline=False,
                models=_models_option(request),
                prequantized=True,
                fused_reduce=True,
            )
            assert rows and all(r["w4a8_ms"] > 0 for r in rows)
            _assert_targets(request, "prefill", rows)

        def test_prequantized_activations_match_internal(self):
            """Handing the kernel int8 activations must reproduce the in-call quantization.

            The pre-quantized entry point does not change any arithmetic: it
            removes the pass that computes the int8 copy and takes the caller's
            instead. Everything downstream -- the policy, the tiling, the
            accumulation order, the epilogue -- is selected from the same
            arguments on both contracts, so the only thing that can differ is
            the int8 bytes and the row scales.

            :func:`_quantize_rows` is that pass expression for expression, but
            it runs its two divisions in exact fp32 while the device is allowed
            a few ulp on them, so on ordinary activations a product that sits
            on a rounding tie can round the other way. That is not a rounding
            difference in the *output*: a flipped int8 is a different input to
            every dot product it takes part in, and one that lands next to a
            cancelling accumulator moves the result by any number of steps. No
            useful bound survives it, so the case is built on the quantizer's
            own grid instead (:func:`_int8_grid_activations`), where every
            product is an integer half a step from the nearest tie and both
            quantizers must emit the same bytes.

            What is left is the row scale. ``absmax / 127`` is a power of two
            on this grid, so the exact result is representable and the device's
            division can miss it by at most an ulp -- and it enters the
            epilogue as a *factor*, so it perturbs every output by the same
            relative amount rather than by an absolute one, which is at most
            one step of the output format wherever the output lies. Hence the
            bound below, which is not a fitted tolerance but the smallest
            difference the format can express.

            It still fails loudly for the bugs this is here to catch -- a
            transposed scale, an off-by-one row, the scale read as its
            reciprocal -- because none of those are worth one step, and the
            per-row exponent makes a scale taken from the wrong row a factor of
            two rather than a coincidence.

            Both K are covered because the internal quantizer picks its lane
            mapping from K, and only one of the two rungs would be exercised by
            a single shape.
            """
            for nk_label, N, K in _QWEN3_NK:
                case = _build_case(
                    N,
                    K,
                    _QWEN3_E,
                    _PREFILL_BATCHES[0] * _QWEN3_TOPK,
                    _QWEN3_GROUP_SIZE,
                    torch.bfloat16,
                    need_reference=False,
                    need_dequant=False,
                    act_int8_grid=True,
                )
                weights_s8, wscales, block = ark.moe_w4a8_prepack(
                    case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
                )
                internal = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
                external = _w4a8(case, weights_s8, wscales, block, "prefill", prequant=True).clone()
                assert int(_ulp_diff(internal, external).max().item()) <= 1, (
                    f"{nk_label.strip()} (K={K}): pre-quantized activations disagree with the in-call "
                    f"quantizer by more than a rounding step: {_ulp_report(internal, external)}"
                )
                case = weights_s8 = wscales = internal = external = None
                ark.clear_moe_w4a8_prepack_cache()
                ark.moe_w4a8_release_scratch()
                _release_xpu_memory()

        def test_prequantized_activations_match_internal_decode(self):
            """The same, on the decode GEMV, where the expert map comes from elsewhere.

            Decode needs a ``token -> expert`` map, and the shipped path gets
            it for free: the activation-quant kernel already runs one sub-group
            per token, so it fills the map on the way past. Pre-quantized
            activations delete that kernel, so the map has to come from the
            standalone scan instead -- a different code path producing a value
            the GEMV indexes its weights with. If it were wrong every token
            would read another expert's weights, which this test sees as a
            gross mismatch rather than the one-step scale difference the two
            paths are entitled to.
            """
            case = _build_case(
                _QWEN3_NK[0][1],
                _QWEN3_NK[0][2],
                _QWEN3_E,
                _DECODE_BATCHES[0] * _QWEN3_TOPK,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
                act_int8_grid=True,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            internal = _w4a8(case, weights_s8, wscales, block, "decode").clone()
            external = _w4a8(case, weights_s8, wscales, block, "decode", prequant=True).clone()
            assert int(_ulp_diff(internal, external).max().item()) <= 1, (
                "pre-quantized activations disagree with the in-call quantizer on decode by more than a "
                f"rounding step: {_ulp_report(internal, external)}"
            )

        def test_fused_reduce_matches_unfused(self):
            """The fused epilogue must agree with reducing the unfused output.

            Unlike every other prefill A/B in this file, this one is *not* a
            bit-identity check and cannot be. The fused epilogue combines a
            token's ``top_k`` contributions with device-scope fp32 atomics,
            which commit in whatever order the work-groups finish, and fp32
            addition is not associative; the unfused path additionally rounds
            each row to the activation dtype before the caller's reduction ever
            sees it. So the two differ by rounding on both sides, and the gate
            is the accuracy gate the rest of the suite uses against the fp32
            reference (20 dB / 0.99 cosine) -- generous enough to survive
            reassociation, far too tight to survive a wrong token index, a
            missing routing weight or an expert-offset slip, all of which
            misplace whole rows.

            The batch is the ragged one: 300 rows per expert against the
            ladder's 128-row tile gives every expert two interior tiles and one
            partial tile, so both the fast and the guarded scatter run, and the
            partial tile is where a scatter can do damage a predicated store
            cannot -- an out-of-range row would land on a *valid* token.
            """
            rows_per_expert = _RAGGED_TILE_ROWS_PER_EXPERT
            case = _build_case(
                _QWEN3_NK[1][1],
                _QWEN3_NK[1][2],
                _QWEN3_E,
                rows_per_expert * _QWEN3_E,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
                topk=_QWEN3_TOPK,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            unfused = _reduce_topk(case, _w4a8(case, weights_s8, wscales, block, "prefill"))
            fused = _w4a8(case, weights_s8, wscales, block, "prefill", fused=True)
            assert fused.shape == unfused.shape, f"fused output shape {tuple(fused.shape)} != {tuple(unfused.shape)}"
            snr = _snr_db(unfused, fused)
            cos = _cosine(unfused, fused)
            assert snr >= _MIN_SNR_DB, f"the fused top-k reduction disagrees with the unfused one: SNR {snr:.2f} dB"
            assert cos >= _MIN_COSINE, f"the fused top-k reduction disagrees with the unfused one: cosine {cos:.6f}"

        def test_fused_reduce_rejects_decode(self):
            """The fused reduction must refuse the decode phase rather than mis-reduce.

            The scatter lives in the grouped GEMM's epilogue; the decode GEMV
            has no such epilogue, so a decode call with routing tables would
            silently return the unreduced output under a shape that claims to
            be reduced. Both the Python guard and the kernel's own check exist
            for this; the Python one is what a caller hits.
            """
            case = _build_case(
                _QWEN3_NK[1][1],
                _QWEN3_NK[1][2],
                _QWEN3_E,
                _DECODE_BATCHES[0] * _QWEN3_TOPK,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
                topk=_QWEN3_TOPK,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            with pytest.raises(ValueError):
                _w4a8(case, weights_s8, wscales, block, "decode", fused=True)

        def test_act_quant_vec_matches_scalar(self):
            """The vectorized activation quantizer must be bit-identical to the scalar one.

            Unlike the K-split decode rewrite, this one reorders nothing that
            rounds: the per-lane partial reduction is ``fmax``, which is exact
            and order-independent, so both mappings feed the sub-group reduce
            the same absmax and therefore the same reciprocal. Every element is
            then put through the same ``rint``/``clamp`` expression, and the
            GEMM that consumes the int8 is deterministic. Only the lane -> K
            assignment differs, so any difference at all is a bug (a wrong
            vector index, a row-stride slip, or a missed tail).
            """
            case = _build_case(
                _QWEN3_NK[0][1],
                _QWEN3_NK[0][2],
                _QWEN3_E,
                _PREFILL_BATCHES[0] * _QWEN3_TOPK,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            outs = {}
            for flag in ("0", "1"):
                with _env_override(ARK_MOE_W4A8_ACT_QUANT_VEC=flag):
                    outs[flag] = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
            assert torch.equal(outs["0"], outs["1"]), (
                "vectorized activation quantization disagrees with the scalar mapping: "
                f"max |diff| {(outs['0'].float() - outs['1'].float()).abs().max().item():.6g}"
            )

        def test_act_quant_unroll_matches(self):
            """Batching the quantizer's loads must be bit-identical at every depth.

            ``UNROLL`` only changes how many vectors are in flight before any
            is consumed: the per-lane reduction is still ``fmax`` over the same
            values (exact and order-independent, so the partial maxima merge to
            the same bits as one chain), and every element goes through the
            same ``rint``/``clamp``. ``UNROLL = 1`` is the kernel as it was
            before the batching, so it is the reference here.

            Both K are checked because the tail loop is what a wrong bound
            would break: at ``VEC = 8`` a lane walks ``K / 128`` vectors, so
            K = 2048 divides by the default depth of 4 and K = 768 leaves a
            two-vector tail. A missed or double-counted tail changes the
            absmax, and hence every element of the row.
            """
            for nk_label, N, K in _QWEN3_NK:
                case = _build_case(
                    N,
                    K,
                    _QWEN3_E,
                    _PREFILL_BATCHES[0] * _QWEN3_TOPK,
                    _QWEN3_GROUP_SIZE,
                    torch.bfloat16,
                    need_reference=False,
                    need_dequant=False,
                )
                weights_s8, wscales, block = ark.moe_w4a8_prepack(
                    case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
                )
                outs = {}
                for unroll in ("1", "2", "4"):
                    with _env_override(ARK_MOE_W4A8_ACT_QUANT_VEC="1", ARK_MOE_W4A8_ACT_QUANT_UNROLL=unroll):
                        outs[unroll] = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
                for unroll in ("2", "4"):
                    assert torch.equal(outs["1"], outs[unroll]), (
                        f"{nk_label.strip()} (K={K}): activation quantization at unroll {unroll} disagrees with "
                        f"unroll 1: max |diff| "
                        f"{(outs['1'].float() - outs[unroll].float()).abs().max().item():.6g}"
                    )
                case = weights_s8 = wscales = outs = None
                ark.clear_moe_w4a8_prepack_cache()
                ark.moe_w4a8_release_scratch()
                _release_xpu_memory()

        def test_full_tile_epilogue_matches_predicated(self):
            """Skipping the epilogue guard on interior tiles must change nothing.

            The fast path drops only the store predicate and the two
            scale-index clamps, all three of which are no-ops on a tile that
            lies inside the expert's rows and inside N; the convert, the two
            multiplies and their order are untouched, so the results must be
            bit-identical, not merely close.

            The shape matters: the batch puts 300 rows on every expert, which
            the tile ladder resolves to a 128-row tile, so each expert has two
            interior tiles *and* one ragged 44-row tile and a single launch
            exercises both paths. A small-batch case would leave every tile
            ragged and the test would pass without the fast path ever running.
            """
            rows_per_expert = _RAGGED_TILE_ROWS_PER_EXPERT
            case = _build_case(
                _QWEN3_NK[1][1],
                _QWEN3_NK[1][2],
                _QWEN3_E,
                rows_per_expert * _QWEN3_E,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            outs = {}
            for flag in ("0", "1"):
                with _env_override(ARK_MOE_W4A8_PREFILL_FULL_TILE=flag):
                    outs[flag] = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
            assert torch.equal(outs["0"], outs["1"]), (
                "the interior-tile epilogue disagrees with the guarded one: "
                f"max |diff| {(outs['0'].float() - outs['1'].float()).abs().max().item():.6g}"
            )

        def test_act_quant_single_pass_matches(self):
            """Holding the row in registers must be bit-identical to re-reading it.

            The single-pass kernel changes *when* the row is read, not what is
            computed from it: the per-lane reduction is still ``fmax`` over the
            same values into the same four partial maxima (exact and
            order-independent, so any grouping merges to the same bits), the
            sub-group reduce and the reciprocal are untouched, and every
            element goes through the same ``rint``/``clamp``. So the two-pass
            kernel is an exact reference, not an approximate one.

            Both K are checked because the register array is bounded at compile
            time and the rung is chosen from ``K``: at ``VEC = 8`` a lane walks
            ``K / 128`` vectors, so K = 768 (6 vectors) takes the 8-slot rung
            and K = 2048 (16) fills the 16-slot one exactly. An off-by-one in
            the ``s < steps`` guard would either drop a vector from the absmax
            or quantize past the end of the row, and both show up here.
            """
            for nk_label, N, K in _QWEN3_NK:
                case = _build_case(
                    N,
                    K,
                    _QWEN3_E,
                    _PREFILL_BATCHES[0] * _QWEN3_TOPK,
                    _QWEN3_GROUP_SIZE,
                    torch.bfloat16,
                    need_reference=False,
                    need_dequant=False,
                )
                weights_s8, wscales, block = ark.moe_w4a8_prepack(
                    case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
                )
                outs = {}
                for flag in ("0", "1"):
                    with _env_override(ARK_MOE_W4A8_ACT_QUANT_VEC="1", ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS=flag):
                        outs[flag] = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
                assert torch.equal(outs["0"], outs["1"]), (
                    f"{nk_label.strip()} (K={K}): the single-pass activation quantizer disagrees with the "
                    f"two-pass one: max |diff| {(outs['0'].float() - outs['1'].float()).abs().max().item():.6g}"
                )
                case = weights_s8 = wscales = outs = None
                ark.clear_moe_w4a8_prepack_cache()
                ark.moe_w4a8_release_scratch()
                _release_xpu_memory()

        def test_prefill_2d_store_matches_scalar(self):
            """The 2D block store must write exactly what the scalar store wrote.

            Only the store mechanism changes: the scaled value is computed by
            the same convert and the same two multiplies in the same order, and
            the fragment is handed to ``copy`` through the same coordinates the
            scalar path indexes with. The store predicate is gone because the
            block message clips to the ``m x n`` output surface in hardware, so
            the interesting failure is a *ragged* tile writing rows that belong
            to the next expert -- which is silent corruption, not a crash.

            The batch is therefore the one from the interior-tile test: 300
            rows on every expert against the ladder's 128-row tile gives each
            expert two interior tiles and one ragged one, and the experts are
            adjacent in the output, so anything spilling past an expert's last
            row lands in the comparison.
            """
            rows_per_expert = _RAGGED_TILE_ROWS_PER_EXPERT
            case = _build_case(
                _QWEN3_NK[1][1],
                _QWEN3_NK[1][2],
                _QWEN3_E,
                rows_per_expert * _QWEN3_E,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            outs = {}
            for flag in ("0", "1"):
                with _env_override(ARK_MOE_W4A8_PREFILL_STORE_2D=flag):
                    outs[flag] = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
            assert torch.equal(outs["0"], outs["1"]), (
                "the 2D block store disagrees with the scalar store: "
                f"max |diff| {(outs['0'].float() - outs['1'].float()).abs().max().item():.6g}"
            )

        def test_prefill_claim_early_matches(self):
            """Claiming the next tile early must not change which tiles run.

            The persistent kernel hands tiles out with a device-scope
            ``atomicAdd`` on one dword. Moving that claim from after the GEMM
            to before it changes only when the message is issued: the loop
            still performs exactly one claim per tile and still consumes the
            indices in the order the counter produces them, so the two orders
            must agree bit for bit, not merely to an SNR.

            The failure this guards against is a work-stealing bug, and those
            are silent: a tile computed twice or skipped leaves a band of the
            output stale or doubled rather than raising. Two things could
            cause it here -- the claimed index being consumed a tile late, and
            the counter's reset racing the first claims now that they are
            issued within a few instructions of kernel entry (which is why the
            reset moved to the host). Both show up as a mismatch against the
            claim-after-GEMM baseline.

            The batch is the ragged one: 300 rows on every expert against the
            ladder's 128-row tile gives each expert two interior tiles and one
            partial, so the expert boundaries the tile walk has to respect are
            in the comparison, and every expert contributes more tiles than a
            single work-group processes in one pass.
            """
            rows_per_expert = _RAGGED_TILE_ROWS_PER_EXPERT
            case = _build_case(
                _QWEN3_NK[1][1],
                _QWEN3_NK[1][2],
                _QWEN3_E,
                rows_per_expert * _QWEN3_E,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            outs = {}
            for flag in ("0", "1"):
                with _env_override(ARK_MOE_W4A8_PREFILL_CLAIM_EARLY=flag):
                    outs[flag] = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
            assert torch.equal(outs["0"], outs["1"]), (
                "claiming the next tile before the GEMM changed the result: "
                f"max |diff| {(outs['0'].float() - outs['1'].float()).abs().max().item():.6g}, "
                f"{(outs['0'] != outs['1']).sum().item()} of {outs['0'].numel()} elements differ"
            )

        def test_prefill_small_grf_matches(self):
            """The halved register budget must not change the result.

            ``grf_size`` asks the compiler for a register allocation; it does
            not change what the kernel computes. But it is not only a compiler
            hint here: the request also sets how many work-groups are resident,
            and the persistent grid is sized from that number, so the two
            budgets launch *different grids* over the same tiles. That is the
            part worth testing. Tiles are handed out by a device-scope counter
            rather than owned by a work-group, so a grid of a different size
            must still consume every index exactly once.

            The failure mode is silent in the same way the claim-order one is:
            too few work-groups leave the last tiles uncomputed, too many leave
            a band recomputed, and neither raises. Both show up as a mismatch
            against the large-GRF baseline.

            The tile is forced to 128x128 because that is the only rung the
            halved budget applies to -- the ladder's own choice for these
            shapes is 128x256, which declines it and would make this test
            compare a kernel against itself. The batch is the ragged one, so
            every expert contributes two interior tiles and one partial and the
            expert boundaries the tile walk has to respect are in the
            comparison.
            """
            rows_per_expert = _RAGGED_TILE_ROWS_PER_EXPERT
            case = _build_case(
                _QWEN3_NK[1][1],
                _QWEN3_NK[1][2],
                _QWEN3_E,
                rows_per_expert * _QWEN3_E,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            outs = {}
            for flag in ("0", "1"):
                with _env_override(
                    ARK_MOE_W4A8_PREFILL_TILE="128x128",
                    ARK_MOE_W4A8_PREFILL_SMALL_GRF=flag,
                ):
                    outs[flag] = _w4a8(case, weights_s8, wscales, block, "prefill").clone()
            assert torch.equal(outs["0"], outs["1"]), (
                "halving the register budget changed the result: "
                f"max |diff| {(outs['0'].float() - outs['1'].float()).abs().max().item():.6g}, "
                f"{(outs['0'] != outs['1']).sum().item()} of {outs['0'].numel()} elements differ"
            )

        def test_decode_ksplit_matches_legacy(self):
            """The K-split decode mapping must agree with the legacy one.

            Both paths accumulate the same int32 partial sums per re-scale
            block and apply the same scales; only the assignment of K elements
            to lanes differs. That does reorder the *float* accumulation (the
            legacy kernel folds every block in one lane, the K-split kernel
            folds per lane and then reduces across the sub-group), so the two
            are not required to be bit-identical -- but they must agree far
            more closely than either agrees with the fp32 reference. A wrong
            lane mapping, expert offset or block scale would miss by orders of
            magnitude, not by a rounding step.
            """
            case = _build_case(
                _QWEN3_NK[0][1],
                _QWEN3_NK[0][2],
                _QWEN3_E,
                _DECODE_BATCHES[0] * _QWEN3_TOPK,
                _QWEN3_GROUP_SIZE,
                torch.bfloat16,
                need_reference=False,
                need_dequant=False,
            )
            weights_s8, wscales, block = ark.moe_w4a8_prepack(
                case["packed"], case["scales"], group_size=_QWEN3_GROUP_SIZE
            )
            outs = {}
            for flag in ("0", "1"):
                with _env_override(ARK_MOE_W4A8_DECODE_KSPLIT=flag):
                    outs[flag] = _w4a8(case, weights_s8, wscales, block, "decode").clone()
            snr = _snr_db(outs["0"], outs["1"])
            cos = _cosine(outs["0"], outs["1"])
            assert snr >= 40.0, f"K-split decode disagrees with the legacy GEMV: SNR {snr:.2f} dB"
            assert cos >= 0.9999, f"K-split decode disagrees with the legacy GEMV: cosine {cos:.6f}"


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


def _assert_targets(request, phase, rows) -> None:
    """Enforce the perf goals only when ``--enforce-targets`` was passed.

    The goals are device-dependent (they assume a discrete Arc-class GPU), so
    by default the verdict printed by ``_print_targets`` is informational and
    the perf tests stay green anywhere the kernel merely runs. Rows whose
    bandwidth-bound ceiling is below the target are never enforced: no kernel
    change can satisfy the target at that routing.
    """
    if not request.config.getoption("--enforce-targets", default=False):
        return
    is_prefill = phase == "prefill"
    target = _TARGET_PREFILL_TFLOPS if is_prefill else _TARGET_DECODE_GBPS
    for row in rows:
        ceiling = row.get("tflops_ceiling")
        if is_prefill and ceiling is not None and ceiling < target:
            continue
        measured = row["tflops"] if is_prefill else row["gbps"]
        unit = "TFLOPS" if is_prefill else "GB/s"
        label = f"{row['label']} phase={phase} N={row['N']} K={row['K']} tokens={row['tokens']}"
        assert measured > target, f"{label}: {measured:.2f} {unit} is below the {target:g} {unit} target"


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
        "--models",
        default=",".join(_DEFAULT_MODELS),
        help=(
            "Comma-separated shape groups to benchmark, or 'all' "
            f"(available: {', '.join(_MODELS)}; default: {','.join(_DEFAULT_MODELS)})."
        ),
    )
    parser.add_argument(
        "--sweep-configs",
        action="store_true",
        help=(
            "Also sweep the kernel dispatch configurations (decode lane mapping, prefill tile) "
            "and report the fastest numerically-equivalent one per shape."
        ),
    )
    parser.add_argument(
        "--rescale-group-size",
        type=int,
        default=-1,
        help="AUTO_S8 re-scale block size; -1 (default) = one scale per output channel.",
    )
    parser.add_argument("--skip-accuracy", action="store_true", help="Only run the perf sweep.")
    parser.add_argument("--skip-perf", action="store_true", help="Only run the accuracy sweep.")
    parser.add_argument(
        "--compute-bound",
        action="store_true",
        help=(
            f"Also run the compute-bound prefill batch ({_PREFILL_TARGET_ROWS_PER_EXPERT} rows per expert), "
            "the smallest sweep point where the 100 TFLOPS goal is not capped by weight bandwidth."
        ),
    )
    parser.add_argument(
        "--long-seq",
        action="store_true",
        help=(
            f"Also run the long-prompt prefill point ({_PREFILL_LONG_SEQ_LEN} model tokens, one 8K sequence), "
            "where Qwen3-MoE routes 512 rows per expert -- a higher intensity than the compute-bound batch, "
            "and the only routing at which a 256-row tile does not pad. With --sweep-configs the "
            "prefill tile sweep is repeated there."
        ),
    )
    parser.add_argument(
        "--dedup-quant",
        action="store_true",
        help=(
            "Also run the end-to-end up/gate comparison that deduplicates the dynamic quantization: "
            "quantize the batch distinct tokens once and permute int8, instead of permuting 16-bit and "
            "letting the kernel quantize every routed row. Both paths are timed with the permute included."
        ),
    )
    parser.add_argument(
        "--contracts",
        action="store_true",
        help=(
            "Also run the prefill points with the traffic-cutting call contracts (caller-supplied int8 "
            "activations + the fused top-k reduction), and with --sweep-configs the contract A/B sweep. "
            "These change what the call moves, so the printed ceiling and BW@100T follow the contract."
        ),
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help=(
            "Also sweep the prefill mainloop's prefetch depth (1-8) at whichever prefill points are "
            "selected. This is the measurement a K-dependent default would have to rest on: the "
            "qwen3 down-projection has 12 k-tiles where minimax up has 48, so a fixed depth is a very "
            "different fraction of the mainloop on each. Pair with --long-seq for the 8K-prompt routing."
        ),
    )
    parser.add_argument(
        "--claim-early",
        action="store_true",
        help=(
            "Also sweep when the persistent prefill kernel claims its next tile: after the GEMM (the "
            "old order, a device-scope atomic round trip fully exposed between two tiles) or before it, "
            "in flight across the mainloop. Bounded by how much work a tile has to hide it behind, so "
            "pair with --long-seq for the 12-k-tile down projections where it should show first."
        ),
    )
    parser.add_argument(
        "--grf",
        action="store_true",
        help=(
            "Also sweep the prefill register budget: the same 128x128 tile at grf_size 256 and 128, "
            "plus the shipped 128x256 tile for reference. Halving the request doubles how many "
            "work-groups are resident, which the persistent grid is sized from, so this is the "
            "occupancy sweep the small tiles in the ladder have never had -- they were all measured "
            "at the large budget their smaller accumulator was supposed to avoid."
        ),
    )
    parser.add_argument("--iters", type=int, default=ITERS, help=f"Timed iterations per measurement (default {ITERS}).")
    parser.add_argument("--warmup", type=int, default=WARMUP, help=f"Warmup iterations (default {WARMUP}).")
    parser.add_argument(
        "--rounds",
        type=int,
        default=SWEEP_ROUNDS,
        help=f"Rounds the sweeps round-robin over their configurations (default {SWEEP_ROUNDS}). "
        "Each round times ITERS // ROUNDS iterations, so the total is unchanged; raise it when the "
        "reported drift is comparable to the spread being ranked.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    global WARMUP, ITERS, SWEEP_ROUNDS
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if _W4A8_SKIP:
        print(f"[moe-w4a8-perf] cannot run: {_W4A8_SKIP}")
        return 1

    WARMUP = args.warmup
    ITERS = args.iters
    SWEEP_ROUNDS = max(1, args.rounds)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    phases = ("decode", "prefill") if args.phase == "both" else (args.phase,)

    env_block = os.environ.get("ARK_MOE_W4A8_AUTO_S8")
    if env_block is not None:
        print(f"[moe-w4a8-perf] ARK_MOE_W4A8_AUTO_S8={env_block} overrides --rescale-group-size")

    models = "all" if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]

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
            run_perf(phase, batches, dtype=dtype, rescale_group_size=args.rescale_group_size, models=models)
            if phase == "prefill" and args.compute_bound:
                run_perf(
                    phase,
                    None,
                    dtype=dtype,
                    rescale_group_size=args.rescale_group_size,
                    torch_baseline=False,
                    compute_bound=True,
                    models=models,
                )
            if phase == "prefill" and args.long_seq:
                run_perf(
                    phase,
                    _long_seq_batches(),
                    dtype=dtype,
                    rescale_group_size=args.rescale_group_size,
                    torch_baseline=False,
                    models=models,
                )
            if phase == "prefill" and args.contracts:
                run_perf(
                    phase,
                    _long_seq_batches() if args.long_seq else None,
                    dtype=dtype,
                    rescale_group_size=args.rescale_group_size,
                    torch_baseline=False,
                    compute_bound=not args.long_seq,
                    models=models,
                    prequantized=True,
                    fused_reduce=True,
                )
            if phase == "prefill" and args.dedup_quant:
                run_dedup_quant(
                    _long_seq_batches() if args.long_seq else None,
                    dtype=dtype,
                    models=models,
                )
            if args.sweep_configs:
                configs = _PREFILL_TILE_CONFIGS if phase == "prefill" else _DECODE_CONFIGS
                run_config_sweep(phase, configs, dtype=dtype, models=models)
                if phase == "prefill" and args.long_seq:
                    run_config_sweep(phase, configs, dtype=dtype, models=models, batches=_long_seq_batches())
                if phase == "prefill" and args.contracts:
                    run_config_sweep(phase, _PREFILL_CONTRACT_CONFIGS, dtype=dtype, models=models)
            if phase == "prefill" and args.prefetch:
                if not args.long_seq or args.compute_bound:
                    run_config_sweep(phase, _PREFILL_PREFETCH_CONFIGS, dtype=dtype, models=models)
                if args.long_seq:
                    run_config_sweep(
                        phase,
                        _PREFILL_PREFETCH_CONFIGS,
                        dtype=dtype,
                        models=models,
                        batches=_long_seq_batches(),
                    )
            if phase == "prefill" and args.claim_early:
                if not args.long_seq or args.compute_bound:
                    run_config_sweep(phase, _PREFILL_CLAIM_CONFIGS, dtype=dtype, models=models)
                if args.long_seq:
                    run_config_sweep(
                        phase,
                        _PREFILL_CLAIM_CONFIGS,
                        dtype=dtype,
                        models=models,
                        batches=_long_seq_batches(),
                    )
            if phase == "prefill" and args.grf:
                if not args.long_seq or args.compute_bound:
                    run_config_sweep(phase, _PREFILL_GRF_CONFIGS, dtype=dtype, models=models)
                if args.long_seq:
                    run_config_sweep(
                        phase,
                        _PREFILL_GRF_CONFIGS,
                        dtype=dtype,
                        models=models,
                        batches=_long_seq_batches(),
                    )

    if failures:
        print()
        print("ACCURACY FAILURES:")
        for message in failures:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
