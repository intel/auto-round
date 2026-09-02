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

"""Performance comparison: ``ark.moe_gemm_decode`` vs default XPU MoE.

The "default XPU MoE implementation" used as the baseline is the standard
per-expert PyTorch matmul loop (the same approach ``_moe_decode_reference``
uses in ``test_moe.py``). For quantized formats the weights are dequantized
once up-front (outside the timed region), so the baseline measures only the
matmul cost on XPU. This is what models fall back to when no fused decode
kernel is available.

How to run::

    pytest -v -s auto_round_extension/ark/test/test_moe_decode_perf.py

The ``-s`` flag is required to see the printed timing tables.

By default only the smallest shape group (``bs1``) is run so a CI pass
stays short. Pass ``--all-shapes`` to also include ``bs32``::

    pytest -v -s auto_round_extension/ark/test/test_moe_decode_perf.py \
        --all-shapes
"""

import auto_round_kernel
import pytest
import torch

# Reuse the existing pack/dequant helpers from the correctness tests so that
# the benchmarked path matches what the unit tests already validate.
from test_moe import (  # noqa: E402
    _dequant_fp8,
    _dequant_int2_asym,
    _dequant_int2_sym,
    _dequant_int4_asym,
    _dequant_int4_sym,
    _dequant_int8_asym,
    _dequant_int8_sym,
    _pack_fp8,
    _pack_int2_asym,
    _pack_int2_sym,
    _pack_int4_asym,
    _pack_int4_sym,
    _pack_int8_asym,
    _pack_int8_sym,
)

ark = auto_round_kernel


# ---------------------------------------------------------------------------
# Skip reasons.
#
# The original test_moe.py collapses several different failure modes into one
# generic "kernel not built" message which makes it impossible to tell whether
# the build is missing the kernel or whether XPU itself didn't come up. The
# helpers below distinguish those cases so a skipped run is actually
# actionable.
# ---------------------------------------------------------------------------


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _xpu_skip_reason() -> str:
    if not hasattr(torch, "xpu"):
        return "torch has no xpu submodule (need an Intel XPU build of torch)"
    if not torch.xpu.is_available():
        return "torch.xpu.is_available() == False (no XPU device or driver visible)"
    return ""


def _decode_skip_reason() -> str:
    """Return non-empty string if the decode kernel can't be exercised."""
    reason = _xpu_skip_reason()
    if reason:
        return reason
    if ark.xpu_lib is None:
        return (
            "ark.xpu_lib is None -- the XPU extension module "
            "(auto_round_kernel_xpu) failed to import; check that auto_round_kernel "
            "was installed for THIS Python env with XPU support enabled"
        )
    if not hasattr(ark.xpu_lib, "moe_gemm_decode"):
        return (
            "ark.xpu_lib loaded but has no moe_gemm_decode symbol -- "
            "rebuild with ARK_SYCL_TLA=ON to compile the MoE decode GEMV kernel"
        )
    return ""


_DECODE_SKIP = _decode_skip_reason()

# Surface diagnostics on collection so the user always sees why the suite
# would skip, without having to add extra flags.
print(
    "[moe-decode-perf] xpu_available=%s  xpu_lib=%s  has_moe_gemm_decode=%s"
    % (
        _xpu_available(),
        "loaded" if ark.xpu_lib is not None else "None",
        hasattr(ark.xpu_lib, "moe_gemm_decode") if ark.xpu_lib is not None else False,
    )
)
if _DECODE_SKIP:
    print("[moe-decode-perf] suite will SKIP. reason: %s" % _DECODE_SKIP)


# ---------------------------------------------------------------------------
# Timing utilities.
# ---------------------------------------------------------------------------

# Warmup / iteration counts kept modest so the suite is still UT-shaped
# (finishes in seconds) but large enough for stable medians.
WARMUP = 5
ITERS = 30


def _xpu_time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Time ``fn`` on XPU using device events; returns median ms per call."""
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


def _default_moe_decode(activations, dequant_weights, num_tokens_per_expert):
    """Default XPU MoE decode baseline: per-expert torch matmul loop.

    This mirrors the path a model would take when no fused MoE decode kernel
    is available: gather/sort tokens by expert (done by the caller), then
    iterate over experts and do a plain ``A @ W.T`` on each slice.
    """
    total_tokens, _ = activations.shape
    E, N, _ = dequant_weights.shape
    out = torch.empty(total_tokens, N, dtype=activations.dtype, device=activations.device)
    offset = 0
    for e in range(E):
        n_tokens = int(num_tokens_per_expert[e].item())
        if n_tokens == 0:
            continue
        a = activations[offset : offset + n_tokens]
        out[offset : offset + n_tokens] = a @ dequant_weights[e].T
        offset += n_tokens
    return out


# ---------------------------------------------------------------------------
# Shape matrix.
#
# Shapes follow MiniMax-M2 MoE config:
#   hidden_size         = 3072   (K for gate/up-proj, N for down-proj)
#   intermediate_size   = 1536   (N for gate/up-proj, K for down-proj)
#   num_local_experts   = 192
#   num_experts_per_tok = 8      (top-8 routing -> 8 active experts at decode)
#
# Two decode-phase ``tokens_per_expert`` patterns are exercised:
#
#   * ``bs1``  -- batch=1 decode (the classic single-stream case). With
#                 top-8 of 192 exactly eight experts see one token each
#                 and the remaining 184 experts are idle.
#   * ``bs32`` -- batch=32 decode (32 concurrent decoding streams, the
#                 common server-side continuous-batching size). With
#                 top-8 routing this produces 32*8 = 256 routed
#                 expert-token slots spread across the 192 experts;
#                 some experts see >1 token, many see exactly one and a
#                 minority are idle. The distribution below is a fixed
#                 deterministic histogram (64 experts get 2 tokens,
#                 128 get 1 -> sum == 256) so timings are reproducible
#                 across runs and machines.
# ---------------------------------------------------------------------------

# MiniMax-M2 decode, batch=1, top-8 of 192: eight arbitrary experts get 1 token.
_MINIMAX_TPE_BS1 = [0] * 192
for _i in (3, 17, 42, 73, 88, 121, 150, 181):
    _MINIMAX_TPE_BS1[_i] = 1

# MiniMax-M2 decode, batch=32, top-8 of 192. Total routed slots = 32*8 = 256.
# Use a fixed deterministic histogram: 64 experts get 2 tokens, 128 get 1
# (64*2 + 128*1 = 256). The hot-expert indices are striped (every third
# expert) so the active set is spread across the full expert range rather
# than clustered, mirroring the load pattern a real router produces.
_MINIMAX_TPE_BS32 = [1] * 192
for _i in range(0, 192, 3):  # 64 indices: 0, 3, 6, ..., 189
    _MINIMAX_TPE_BS32[_i] = 2
assert sum(_MINIMAX_TPE_BS32) == 256, sum(_MINIMAX_TPE_BS32)

# Backwards-compatible alias (older code/tests referenced ``_MINIMAX_TPE``).
_MINIMAX_TPE = _MINIMAX_TPE_BS1

DECODE_SHAPES = [
    # (label, num_experts, tokens_per_expert, N, K)
    # batch=1 decode (single-stream).
    ("minimax up   bs1 ", 192, list(_MINIMAX_TPE_BS1), 1536, 3072),  # gate/up-proj
    ("minimax down bs1 ", 192, list(_MINIMAX_TPE_BS1), 3072, 1536),  # down-proj
    # batch=32 decode (32 concurrent streams, 32 generated tokens per step).
    ("minimax up   bs32", 192, list(_MINIMAX_TPE_BS32), 1536, 3072),  # gate/up-proj
    ("minimax down bs32", 192, list(_MINIMAX_TPE_BS32), 3072, 1536),  # down-proj
]


@pytest.fixture(autouse=True)
def _maybe_restrict_shapes(request, monkeypatch):
    """Optionally restrict ``DECODE_SHAPES`` to the smallest shape group.

    Controlled by the ``--all-shapes`` pytest CLI flag (registered in this
    directory's ``conftest.py``). Default is off, so only the ``bs1`` rows
    are run and a CI pass stays short. Pass ``--all-shapes`` to also
    include the ``bs32`` rows.

    The filter is applied by monkeypatching this module's ``DECODE_SHAPES``
    for the duration of each test so the ``for label, E, tpe, N, K in
    DECODE_SHAPES`` loop inside every ``test_perf_*`` method sees the
    filtered list without threading the option through each call site.
    """
    if request.config.getoption("--all-shapes", default=False):
        return
    import sys

    module = sys.modules[__name__]
    filtered = [s for s in DECODE_SHAPES if s[0].rstrip().endswith("bs1")]
    monkeypatch.setattr(module, "DECODE_SHAPES", filtered)


def _release_xpu_memory() -> None:
    """Free cached XPU memory and synchronize.

    Called before and after every test (via the autouse cleanup fixture
    below) so allocator pressure from one parametrization does not bleed
    into the next -- mirroring the ``torch.xpu.empty_cache()`` pattern
    used at the end of every XPU test in ``test_matmul.py`` /
    ``test_weightonly.py`` and the ``_release_xpu_memory`` helper in
    ``test_moe_prefill_perf.py``.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()


@pytest.fixture(autouse=True)
def _xpu_cleanup_between_tests():
    """Release XPU allocator cache before and after every test.

    An aborted test (OOM, kernel error, assertion) can leave the XPU
    allocator holding a large working set that then starves the next
    parametrization. Bracketing each test with an ``empty_cache`` call
    isolates parametrizations from one another.
    """
    _release_xpu_memory()
    try:
        yield
    finally:
        _release_xpu_memory()


def _print_header(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print(f"{'shape':<18}{'N':>7}{'K':>7}{'tokens':>8}" f"{'baseline(ms)':>16}{'ark(ms)':>14}{'speedup':>12}")
    print("-" * 110)


def _print_row(label, N, K, total_tokens, base_ms, ark_ms):
    """Print a benchmark row.

    ``speedup`` is ``baseline / ark`` -- a pure matmul-vs-matmul comparison
    against the per-expert ``A @ W.T`` baseline running on already-dequantized
    weights.
    """
    speedup = base_ms / ark_ms if ark_ms > 0 else float("nan")
    print(f"{label:<18}{N:>7}{K:>7}{total_tokens:>8}" f"{base_ms:>16.4f}{ark_ms:>14.4f}{speedup:>11.2f}x")


# ---------------------------------------------------------------------------
# Benchmark cases.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bool(_DECODE_SKIP), reason=_DECODE_SKIP or "ok")
class TestMoEGemmDecodePerf:
    """Median XPU-event timings of ``moe_gemm_decode`` vs per-expert ``A @ W.T``.

    The baseline uses *already-dequantized* weights, so quantized cases only
    pay the matmul cost in the ``baseline(ms)`` column. ``speedup`` is
    ``baseline / ark`` -- a pure matmul-vs-matmul comparison that isolates
    the fused decode kernel's matmul speedup from any per-step dequant
    overhead a quantized-weight pipeline might otherwise pay.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_perf_fp(self, dtype):
        _print_header(f"FP weights ({str(dtype).split('.')[-1]})  -- ark.moe_gemm_decode vs per-expert A @ W.T")
        for label, E, tpe, N, K in DECODE_SHAPES:
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            weights = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_decode(activations, weights, ntpe))
            ark_ms = _xpu_time_ms(lambda: ark.moe_gemm_decode(activations, weights, ntpe, weight_bits=16))
            _print_row(label, N, K, total_tokens, base_ms, ark_ms)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int4(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT4 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_decode vs dequant + per-expert A @ W.T"
        )
        for label, E, tpe, N, K in DECODE_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int4_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int4_asym(packed, scales, zeros, group_size).to(dtype)
            else:
                zeros = None
                packed = _pack_int4_sym(w_float, scales, group_size)
                dequant = _dequant_int4_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_decode(activations, dequant, ntpe))
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_decode(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    zeros=zeros,
                    weight_bits=4,
                    group_size=group_size,
                    asym=asym,
                )
            )
            _print_row(label, N, K, total_tokens, base_ms, ark_ms)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int8(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT8 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_decode vs dequant + per-expert A @ W.T"
        )
        for label, E, tpe, N, K in DECODE_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int8_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int8_asym(packed, scales, zeros, group_size).to(dtype)
            else:
                zeros = None
                packed = _pack_int8_sym(w_float, scales, group_size)
                dequant = _dequant_int8_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_decode(activations, dequant, ntpe))
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_decode(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    zeros=zeros,
                    weight_bits=8,
                    group_size=group_size,
                    asym=asym,
                )
            )
            _print_row(label, N, K, total_tokens, base_ms, ark_ms)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_perf_int2(self, dtype, asym):
        group_size = 128
        kind = "asym" if asym else "sym"
        _print_header(
            f"INT2 {kind} (group_size={group_size}, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_decode vs dequant + per-expert A @ W.T"
        )
        for label, E, tpe, N, K in DECODE_SHAPES:
            if K % group_size != 0 or K % 4 != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            if asym:
                zeros = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
                packed = _pack_int2_asym(w_float, scales, zeros, group_size)
                dequant = _dequant_int2_asym(packed, scales, zeros, group_size).to(dtype)
            else:
                zeros = None
                packed = _pack_int2_sym(w_float, scales, group_size)
                dequant = _dequant_int2_sym(packed, scales, group_size).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_decode(activations, dequant, ntpe))
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_decode(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    zeros=zeros,
                    weight_bits=2,
                    group_size=group_size,
                    asym=asym,
                )
            )
            _print_row(label, N, K, total_tokens, base_ms, ark_ms)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_perf_fp8(self, dtype, fp8_dtype):
        group_size = 128
        _print_header(
            f"FP8 {str(fp8_dtype).split('.')[-1]} (group_size={group_size}, "
            f"act={str(dtype).split('.')[-1]}) -- ark.moe_gemm_decode vs dequant + per-expert A @ W.T"
        )
        for label, E, tpe, N, K in DECODE_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            w_float = (torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
            scales = torch.empty(E, N, K // group_size, dtype=dtype, device="xpu")
            packed = _pack_fp8(w_float, scales, group_size, fp8_dtype)
            dequant = _dequant_fp8(packed, scales, group_size, dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_decode(activations, dequant, ntpe))
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_decode(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    group_size=group_size,
                    asym=False,
                )
            )
            _print_row(label, N, K, total_tokens, base_ms, ark_ms)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_perf_fp8_per_tensor(self, dtype, fp8_dtype):
        """Perf: FP8 per-expert (per-tensor) scale for the decode path.

        The C++ decode kernel does NOT expose a native ``[E]`` per-tensor
        scale API -- ``moe_gemm_decode`` only accepts per-K-group scales
        (``[E, N, K/group_size]``). To exercise the "one FP32 scalar per
        expert" quantisation scheme through the existing decode kernel we
        emulate it by:

        1. Packing each expert's weight tile with a single max-abs FP32
           scalar (``scales_pe.shape == [E]``, same recipe as the prefill
           ``test_perf_fp8_per_tensor`` and the accuracy test
           ``test_accuracy_fp8_per_tensor_dpas``).
        2. Broadcasting that ``[E]`` scalar to a ``[E, N, K/group_size]``
           tensor filled with the same value per expert, which is what
           the decode kernel expects on the wire.

        Semantically this is identical to a per-tensor quantised
        checkpoint (every K-group inside an expert shares one scale) so
        the timings reflect what a real per-expert-scale FP8 checkpoint
        would cost on the existing decode kernel. It does NOT measure a
        different code path from ``test_perf_fp8`` -- the point is to
        confirm the quantisation scheme runs at the same speed as the
        richer per-group scheme, i.e. the extra memory traffic of the
        broadcast tensor is the same as a genuine per-group one.
        """
        group_size = 128
        fp8_finfo_max = torch.finfo(fp8_dtype).max
        _print_header(
            f"FP8 per-expert scale {str(fp8_dtype).split('.')[-1]} "
            f"(scales=[E] fp32 broadcast to K-groups, act={str(dtype).split('.')[-1]}) "
            f"-- ark.moe_gemm_decode vs dequant + per-expert A @ W.T"
        )
        for label, E, tpe, N, K in DECODE_SHAPES:
            if K % group_size != 0:
                continue
            total_tokens = sum(tpe)
            activations = torch.randn(total_tokens, K, dtype=dtype, device="xpu")
            # Per-expert FP32 scale (max-abs of the tile). Note we build the
            # weight in [E, N, K] here to match the decode kernel layout.
            w_float = torch.randn(E, N, K, dtype=torch.float32, device="xpu") * 0.1
            amax = w_float.reshape(E, -1).abs().amax(dim=1).clamp_min(1e-8)
            scales_pe = amax / fp8_finfo_max  # [E] fp32
            packed = (w_float / scales_pe.reshape(E, 1, 1)).to(fp8_dtype)

            # Broadcast the per-expert scalar into the [E, N, K/group_size]
            # layout the decode kernel wire format requires. Every K-group
            # in an expert holds the same value, so the accumulated result
            # matches a genuine per-tensor scale semantically.
            scales = scales_pe.to(dtype).reshape(E, 1, 1).expand(E, N, K // group_size).contiguous()

            # Reference dequant for the baseline: multiply the fp8 bytes by
            # the per-expert scalar and cast to the activation dtype.
            dequant = (packed.to(torch.float32) * scales_pe.reshape(E, 1, 1)).to(dtype)
            ntpe = torch.tensor(tpe, dtype=torch.int32, device="xpu")

            base_ms = _xpu_time_ms(lambda: _default_moe_decode(activations, dequant, ntpe))
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_decode(
                    activations,
                    packed,
                    ntpe,
                    scales=scales,
                    group_size=group_size,
                    asym=False,
                )
            )
            _print_row(label, N, K, total_tokens, base_ms, ark_ms)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
