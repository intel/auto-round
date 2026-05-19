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
    has_moe_gemm_decode,
    is_xpu_available,
)

ark = auto_round_kernel.ARK()


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
# Picked to cover small / medium / large MoE expert FFNs (Mixtral-style
# 4096x14336 down-projection is the upper bound; smaller shapes catch
# launch-overhead-dominated cases). ``tokens_per_expert`` follows the
# expected decode-phase pattern (top-k routing with batch=1: each active
# expert sees one token).
# ---------------------------------------------------------------------------

DECODE_SHAPES = [
    # (label, num_experts, tokens_per_expert, N, K)
    ("small   E=4 ", 4, [1, 0, 1, 1], 1024, 1024),
    ("medium  E=8 ", 8, [1, 1, 0, 1, 1, 0, 1, 1], 2048, 2048),
    ("large   E=8 ", 8, [1, 0, 1, 1, 0, 1, 1, 1], 4096, 4096),
    ("ffn-up  E=8 ", 8, [1, 1, 0, 1, 1, 1, 0, 1], 14336, 4096),
    ("ffn-dn  E=8 ", 8, [1, 1, 0, 1, 1, 1, 0, 1], 4096, 14336),
]


def _print_header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print(
        f"{'shape':<14}{'N':>7}{'K':>7}{'tokens':>8}"
        f"{'baseline(ms)':>16}{'ark(ms)':>14}{'speedup':>12}"
    )
    print("-" * 96)


def _print_row(label, N, K, total_tokens, base_ms, ark_ms):
    speedup = base_ms / ark_ms if ark_ms > 0 else float("nan")
    print(
        f"{label:<14}{N:>7}{K:>7}{total_tokens:>8}"
        f"{base_ms:>16.4f}{ark_ms:>14.4f}{speedup:>11.2f}x"
    )


# ---------------------------------------------------------------------------
# Benchmark cases.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
@pytest.mark.skipif(not has_moe_gemm_decode(), reason="MoE decode GEMV kernel not built (need ARK_SYCL_TLA=ON)")
class TestMoEGemmDecodePerf:
    """Median XPU-event timings of ``moe_gemm_decode`` vs per-expert ``A @ W.T``.

    The baseline uses *already-dequantized* weights, so quantized cases only
    pay the matmul cost in the timed region (no per-iteration dequant). This
    is the most favorable apples-to-apples comparison for the baseline; the
    fused decode kernel must beat that to be worth using.
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
            ark_ms = _xpu_time_ms(
                lambda: ark.moe_gemm_decode(activations, weights, ntpe, weight_bits=16)
            )
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
                    activations, packed, ntpe,
                    scales=scales, zeros=zeros,
                    weight_bits=4, group_size=group_size, asym=asym,
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
                    activations, packed, ntpe,
                    scales=scales, zeros=zeros,
                    weight_bits=8, group_size=group_size, asym=asym,
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
                    activations, packed, ntpe,
                    scales=scales, zeros=zeros,
                    weight_bits=2, group_size=group_size, asym=asym,
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
                    activations, packed, ntpe,
                    scales=scales,
                    group_size=group_size, asym=False,
                )
            )
            _print_row(label, N, K, total_tokens, base_ms, ark_ms)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
