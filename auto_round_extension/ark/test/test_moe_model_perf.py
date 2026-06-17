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

"""Model-level performance test for the unified ``ark.moe`` dispatcher.

This benchmark simulates a realistic MoE LLM generation trace:

  1. A **prefill** step that processes the whole prompt (many tokens/expert).
  2. A handful of **decode** steps that produce one token each (1-2
     tokens/expert after TopK routing).

Each step runs through ``L`` MoE layers; each layer consists of an
``up_proj`` (``K -> N_inter``) and a ``down_proj`` (``N_inter -> K``) MoE
GEMM. This is the standard shape for Mixtral-style models.

We then compare four call strategies for the *same* trace:

  * ``always_prefill``: model code always calls ``moe_gemm_prefill``.
    Represents the simple-but-suboptimal "use the prefill kernel
    everywhere" approach.
  * ``always_decode``:  model code always calls ``moe_gemm_decode``.
    The opposite extreme -- decode kernel for both phases.
  * ``manual_branch``:  model code branches on the known phase (``if
    is_prefill: moe_gemm_prefill else moe_gemm_decode``). Optimal but
    requires two call sites and a phase flag in the model.
  * ``unified_auto``:   single call site: ``ark.moe(..., phase="auto")``.
    The dispatcher picks the right kernel from ``num_tokens_per_expert``;
    pays one tiny host-device sync per call.
  * ``unified_hinted``: single call site: ``ark.moe(..., phase=<known>)``.
    Skips the sync when the caller already knows the phase.

The reported speedup is ``always_prefill_time / strategy_time`` -- i.e.,
how much faster the unified API gets you relative to the naive
single-kernel approach a typical first-pass integration would use.

How to run::

    pytest -v -s auto_round_extension/ark/test/test_moe_model_perf.py
"""

from dataclasses import dataclass
from typing import Callable

import auto_round_kernel
import pytest
import torch

from test_moe import (  # noqa: E402
    _pack_int4_sym,
)

ark = auto_round_kernel


# ---------------------------------------------------------------------------
# Skip reasons (mirror the other perf tests)
# ---------------------------------------------------------------------------


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _skip_reason() -> str:
    if not _xpu_available():
        return "XPU not available"
    if ark.xpu_lib is None:
        return "ark.xpu_lib is None (XPU extension failed to import)"
    for sym in ("moe_gemm_decode", "moe_gemm_prefill"):
        if not hasattr(ark.xpu_lib, sym):
            return f"ark.xpu_lib missing {sym} (need ARK_SYCL_TLA=ON)"
    if not hasattr(ark, "moe"):
        return "ark.moe (unified entry point) not exported by auto_round_kernel"
    return ""


_SKIP = _skip_reason()

print(
    "[moe-model-perf] xpu_available=%s  xpu_lib=%s  has_moe=%s"
    % (
        _xpu_available(),
        "loaded" if ark.xpu_lib is not None else "None",
        hasattr(ark, "moe"),
    )
)
if _SKIP:
    print("[moe-model-perf] suite will SKIP. reason: %s" % _SKIP)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

WARMUP = 3
ITERS = 10


def _xpu_time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
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
# Model + trace definitions
# ---------------------------------------------------------------------------


@dataclass
class MoELayerWeights:
    """Quantized weights for one MoE layer (up_proj + down_proj)."""

    up_packed: torch.Tensor       # [E, N_inter, K // 2]   (INT4)
    up_scales: torch.Tensor       # [E, N_inter, K // group_size]
    down_packed: torch.Tensor     # [E, K, N_inter // 2]
    down_scales: torch.Tensor     # [E, K, N_inter // group_size]


def _build_layers(num_layers, E, K, N_inter, group_size, dtype):
    layers = []
    for _ in range(num_layers):
        # up_proj: [E, N_inter, K]
        up_float = (torch.randn(E, N_inter, K, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        up_scales = torch.empty(E, N_inter, K // group_size, dtype=dtype, device="xpu")
        up_packed = _pack_int4_sym(up_float, up_scales, group_size)

        # down_proj: [E, K, N_inter]
        down_float = (torch.randn(E, K, N_inter, dtype=torch.float32, device="xpu") * 0.1).to(dtype)
        down_scales = torch.empty(E, K, N_inter // group_size, dtype=dtype, device="xpu")
        down_packed = _pack_int4_sym(down_float, down_scales, group_size)

        layers.append(MoELayerWeights(up_packed, up_scales, down_packed, down_scales))
    return layers


@dataclass
class Step:
    """One trace step: activations + per-expert token counts + phase flag."""

    activations: torch.Tensor      # [total_tokens, K]
    ntpe: torch.Tensor             # [E] int32
    is_prefill: bool


def _build_trace(E, K, prompt_tokens, decode_steps, dtype, seed=0):
    """Mixed prefill+decode trace mimicking an LLM generation request."""
    g = torch.Generator(device="xpu")
    g.manual_seed(seed)

    steps = []

    # ---- Prefill step: prompt_tokens routed across E experts (roughly even). ----
    base = prompt_tokens // E
    rem = prompt_tokens - base * E
    tpe = [base + (1 if i < rem else 0) for i in range(E)]
    activations = torch.randn(prompt_tokens, K, dtype=dtype, device="xpu", generator=g)
    steps.append(Step(activations, torch.tensor(tpe, dtype=torch.int32, device="xpu"), is_prefill=True))

    # ---- Decode steps: 1 token, TopK=2 routing -> 2 experts see 1 token each. ----
    # We simulate batch=1 + top_k=2: total_tokens=2, two experts each get 1 token.
    for s in range(decode_steps):
        tpe = [0] * E
        chosen = [(s * 2) % E, (s * 2 + 1) % E]
        if chosen[0] == chosen[1]:
            chosen[1] = (chosen[1] + 1) % E
        tpe[chosen[0]] = 1
        tpe[chosen[1]] = 1
        activations = torch.randn(2, K, dtype=dtype, device="xpu", generator=g)
        steps.append(Step(activations, torch.tensor(tpe, dtype=torch.int32, device="xpu"), is_prefill=False))

    return steps


# ---------------------------------------------------------------------------
# Call strategies (the four "model-side" integration patterns)
# ---------------------------------------------------------------------------


_CallStrategy = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, int], torch.Tensor]


def _strat_always_prefill(activations, packed, ntpe, scales, _zeros, _is_prefill, group_size):
    return ark.moe_gemm_prefill(
        activations, packed, ntpe,
        scales=scales, weight_bits=4, group_size=group_size, asym=False,
    )


def _strat_always_decode(activations, packed, ntpe, scales, _zeros, _is_prefill, group_size):
    return ark.moe_gemm_decode(
        activations, packed, ntpe,
        scales=scales, weight_bits=4, group_size=group_size, asym=False,
    )


def _strat_manual_branch(activations, packed, ntpe, scales, _zeros, is_prefill, group_size):
    if is_prefill:
        return ark.moe_gemm_prefill(
            activations, packed, ntpe,
            scales=scales, weight_bits=4, group_size=group_size, asym=False,
        )
    return ark.moe_gemm_decode(
        activations, packed, ntpe,
        scales=scales, weight_bits=4, group_size=group_size, asym=False,
    )


def _strat_unified_auto(activations, packed, ntpe, scales, _zeros, _is_prefill, group_size):
    return ark.moe(
        activations, packed, ntpe,
        scales=scales, weight_bits=4, group_size=group_size, asym=False,
        phase="auto",
    )


def _strat_unified_hinted(activations, packed, ntpe, scales, _zeros, is_prefill, group_size):
    return ark.moe(
        activations, packed, ntpe,
        scales=scales, weight_bits=4, group_size=group_size, asym=False,
        phase="prefill" if is_prefill else "decode",
    )


_STRATEGIES = [
    ("always_prefill ", _strat_always_prefill),
    ("always_decode  ", _strat_always_decode),
    ("manual_branch  ", _strat_manual_branch),
    ("unified_auto   ", _strat_unified_auto),
    ("unified_hinted ", _strat_unified_hinted),
]


# ---------------------------------------------------------------------------
# End-to-end "model forward over the trace" runner
# ---------------------------------------------------------------------------


def _forward_full_trace(strategy, layers, trace, group_size):
    """Run the strategy across every step and every layer of the trace.

    Each layer is up_proj followed by down_proj. The down_proj input is just
    the up_proj output reshaped to ``[total_tokens, N_inter]`` (we skip the
    SiLU/element-wise ops; we only want to measure the MoE-kernel cost,
    which is what the dispatcher changes).
    """
    for step in trace:
        x = step.activations
        for layer in layers:
            up_out = strategy(
                x, layer.up_packed, step.ntpe, layer.up_scales, None,
                step.is_prefill, group_size,
            )
            # down_proj takes the up output (same total_tokens, dim = N_inter).
            x_down = strategy(
                up_out, layer.down_packed, step.ntpe, layer.down_scales, None,
                step.is_prefill, group_size,
            )
            # Feed down_proj output into the next layer's up_proj.
            x = x_down


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


# Three model presets mimicking common MoE configurations. Each runs the
# whole trace (1 prefill + N decode steps) through L layers.
#
# Shapes are intentionally smaller than the perf-bench shapes used in
# `test_moe_prefill_perf.py` so the full multi-layer forward stays tractable
# (we run the whole trace ITERS times for the median timing).
_MODEL_PRESETS = [
    # (label, num_layers, E, K, N_inter, group_size, prompt_tokens, decode_steps)
    ("mixtral-tiny  L=2  E=8 ", 2, 8, 1024, 2048, 128, 64, 8),
    ("mixtral-small L=4  E=8 ", 4, 8, 2048, 4096, 128, 128, 8),
    ("deepseek-tiny L=2  E=16", 2, 16, 1024, 2048, 128, 96, 8),
]


def _print_header():
    print()
    print("=" * 110)
    print(
        f"Model-level MoE perf: full forward over (prefill + N x decode) for L layers; "
        f"INT4 sym weights"
    )
    print("-" * 110)
    print(
        f"{'preset':<24}{'strategy':<18}{'forward(ms)':>14}{'speedup vs prefill-only':>28}"
    )
    print("-" * 110)


def _print_row(preset, strat_name, ms, speedup):
    print(f"{preset:<24}{strat_name:<18}{ms:>14.4f}{speedup:>26.2f}x")


@pytest.mark.skipif(bool(_SKIP), reason=_SKIP or "ok")
class TestMoEModelPerf:
    """Model-level perf comparison for the unified MoE dispatcher.

    The benchmark validates two things in one go:
      1. *Correctness of the dispatcher*: the perf-comparison loop also
         confirms every strategy produces a runnable forward (any kernel
         error during timing causes a hard failure).
      2. *Speedup vs naive single-kernel integration*: the printed table
         shows how much the unified API saves a model author who would
         otherwise reach for ``always_prefill``.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_model_forward(self, dtype):
        _print_header()
        for preset_label, L, E, K, N_inter, group_size, prompt_tokens, decode_steps in _MODEL_PRESETS:
            layers = _build_layers(L, E, K, N_inter, group_size, dtype)
            trace = _build_trace(E, K, prompt_tokens, decode_steps, dtype)

            # Time each strategy on the same (layers, trace) pair.
            baseline_ms = None
            for strat_name, strat_fn in _STRATEGIES:
                def _run(strat_fn=strat_fn):
                    _forward_full_trace(strat_fn, layers, trace, group_size)

                ms = _xpu_time_ms(_run)
                if baseline_ms is None:
                    baseline_ms = ms  # first row is always always_prefill
                speedup = baseline_ms / ms if ms > 0 else float("nan")
                _print_row(f"{preset_label} {dtype}".strip(), strat_name, ms, speedup)

            print("-" * 110)

            # Free the per-preset workspace so the next preset starts clean.
            ark.clear_moe_prefill_workspace_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
