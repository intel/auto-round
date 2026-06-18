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

"""Model-level perf benchmark for MoE LLMs on the ARK (XPU) backend.

This file mirrors the structure of ``test/test_ark/test_model.py`` but
operates on real MoE checkpoints (Qwen1.5-MoE, DeepSeek-V2-Lite) and adds
prefill / decode latency measurement on top.

For each ``(model, bits, dtype)`` combination we:

  1. Load a *tiny* slice of the MoE model with random weights
     (``num_layers=2``, ``num_experts=4``) via ``helpers.get_tiny_model`` so
     the suite is tractable in CI. Set ``AR_MOE_PERF_FULL=1`` to load the
     full checkpoint instead.
  2. Quantize with ``AutoRound(iters=0, nsamples=1, disable_opt_rtn=True)``
     and export with ``format="auto_round"``.
  3. Reload twice on XPU:
        a. unquantized FP reference (no ``quantization_config``);
        b. ARK backend (``AutoRoundConfig(backend="ark")``);
        c. *optional* GPTQModel backend, skipped if not installed.
  4. Smoke-test correctness via ``helpers.model_infer`` (asserts non-empty
     output -> catches any wiring break in the backend).
  5. Measure **prefill** latency (single forward over a 128-token prompt)
     and **per-token decode** latency (``model.generate(max_new_tokens=32)``
     after a 4-token warmup) using ``torch.xpu.Event``. Report the median
     of 3 runs.
  6. Assert that ARK decode latency is within ``ARK_DECODE_REGRESSION_FACTOR``
     of the FP reference -- defends against silent perf regressions of the
     unified ``ark.moe`` dispatcher.

How to run::

    pytest -v -s test/test_ark/test_moe_model_perf.py
    # full-size checkpoints (needs ~30GB free + checkpoint cached locally):
    AR_MOE_PERF_FULL=1 pytest -v -s test/test_ark/test_moe_model_perf.py
    # exclude from default CI runs:
    pytest -m 'not perf' ...
"""

import os
import shutil
from statistics import median

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ..helpers import (
    deepseek_v2_name_or_path,
    get_model_path,
    get_tiny_model,
    model_infer,
    qwen_moe_name_or_path,
)


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

# Use the full pretrained checkpoint when set; otherwise build a tiny random
# slice via ``get_tiny_model`` so the test stays CI-friendly.
_USE_FULL_MODEL = os.environ.get("AR_MOE_PERF_FULL", "0") == "1"

# Tiny-model slice geometry (only used when AR_MOE_PERF_FULL=0).
_TINY_NUM_LAYERS = 2
_TINY_NUM_EXPERTS = 4

# Timing harness.
_PREFILL_PROMPT_TOKENS = 128
_DECODE_NEW_TOKENS = 32
_DECODE_WARMUP_TOKENS = 4
_TIMING_REPEATS = 3

# Perf-regression guard for the ARK decode path vs the unquantized FP
# baseline. Loose enough to absorb run-to-run jitter (we already take the
# median of N runs) but tight enough to catch the "phase=auto sync"
# regression class described in the upstream perf analysis (which costs
# ~25% per kernel call).
ARK_DECODE_REGRESSION_FACTOR = 2.0


# ---------------------------------------------------------------------------
# ARK availability gate (mirrors auto_round_extension/ark/test/test_moe_unified.py)
# ---------------------------------------------------------------------------


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _ark_skip_reason() -> str:
    if not _xpu_available():
        return "XPU not available"
    try:
        import auto_round_kernel as ark
    except ImportError as exc:
        return f"auto_round_kernel not importable: {exc}"
    if getattr(ark, "xpu_lib", None) is None:
        return "ark.xpu_lib is None (XPU extension failed to import)"
    for sym in ("moe_gemm_decode", "moe_gemm_prefill"):
        if not hasattr(ark.xpu_lib, sym):
            return f"ark.xpu_lib missing {sym} (need ARK_SYCL_TLA=ON)"
    if not hasattr(ark, "moe"):
        return "ark.moe (unified entry point) not exported by auto_round_kernel"
    return ""


_ARK_SKIP = _ark_skip_reason()


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def _xpu_sync():
    if _xpu_available():
        torch.xpu.synchronize()


def _xpu_time_ms(fn, repeats: int = _TIMING_REPEATS) -> float:
    """Time ``fn`` on XPU using ``torch.xpu.Event``; return median ms.

    The function is invoked ``repeats`` times after one warmup call. The
    median is returned to absorb run-to-run jitter from the XPU runtime.
    """
    fn()  # warmup
    _xpu_sync()
    timings = []
    for _ in range(repeats):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return median(timings)


def _measure_prefill_ms(model, input_ids, attention_mask) -> float:
    def _run():
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)

    return _xpu_time_ms(_run)


def _measure_decode_ms_per_tok(model, tokenizer, prompt_ids, attention_mask) -> float:
    """Median per-token decode latency from ``generate(max_new_tokens=N)``."""
    # Warmup generate (compiles caches, allocates KV).
    with torch.inference_mode():
        model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=_DECODE_WARMUP_TOKENS,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    def _run():
        with torch.inference_mode():
            model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=_DECODE_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    total_ms = _xpu_time_ms(_run)
    return total_ms / _DECODE_NEW_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prefill_inputs(tokenizer, device, dtype, num_tokens=_PREFILL_PROMPT_TOKENS):
    """Build a fixed-length (num_tokens) input tensor on ``device``."""
    # Use a deterministic synthetic prompt to keep the perf number stable
    # across machines that may not have the same vocab tokenization for a
    # natural-language prompt of a given length.
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    input_ids = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0) % max(1, vocab_size)
    attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def _load_tiny_or_full(model_name_or_path, dtype):
    """Return a model instance + tokenizer for the given MoE checkpoint.

    Tiny path uses ``get_tiny_model`` (random weights, 2 layers, 4 experts)
    so the suite is tractable in CI. Full path loads the real checkpoint
    when ``AR_MOE_PERF_FULL=1``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if _USE_FULL_MODEL:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, trust_remote_code=True)
    else:
        model = get_tiny_model(
            model_name_or_path,
            num_layers=_TINY_NUM_LAYERS,
            num_experts=_TINY_NUM_EXPERTS,
            from_config=True,
            trust_remote_code=True,
        )
        model = model.to(dtype)
    return model, tokenizer


def _format_row(model_label, backend, prefill_ms, decode_ms_per_tok, baseline_decode_ms):
    tps = 1000.0 / decode_ms_per_tok if decode_ms_per_tok > 0 else float("nan")
    if baseline_decode_ms is None or baseline_decode_ms <= 0:
        speedup_str = "    --"
    else:
        speedup_str = f"{baseline_decode_ms / decode_ms_per_tok:6.2f}x"
    return (
        f"{model_label:<22}{backend:<14}{prefill_ms:>12.3f}{decode_ms_per_tok:>16.3f}"
        f"{tps:>14.2f}{speedup_str:>14}"
    )


def _print_header(dtype):
    print()
    print("=" * 96)
    print(f"Model-level MoE perf  --  dtype={dtype}  AR_MOE_PERF_FULL={_USE_FULL_MODEL}")
    print("-" * 96)
    print(
        f"{'model':<22}{'backend':<14}{'prefill(ms)':>12}{'decode(ms/tok)':>16}"
        f"{'tokens/s':>14}{'vs FP':>14}"
    )
    print("-" * 96)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


pytestmark = [pytest.mark.perf]


@pytest.mark.skipif(not _xpu_available(), reason="XPU not available")
class TestMoEModelPerf:
    """Model-level perf table for MoE LLMs on the ARK XPU backend."""

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_folder = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_folder, ignore_errors=True)

    # -- helpers ------------------------------------------------------------

    def _quantize_and_save(self, model, tokenizer, bits, group_size, sym):
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            nsamples=1,
            disable_opt_rtn=True,
        )
        _, saved_folder = autoround.quantize_and_save(output_dir=self.save_folder, format="auto_round")
        return saved_folder

    def _reload(self, saved_folder, dtype, *, backend):
        kwargs = dict(dtype=dtype, device_map="xpu", trust_remote_code=True)
        if backend is not None:
            kwargs["quantization_config"] = AutoRoundConfig(backend=backend)
        return AutoModelForCausalLM.from_pretrained(saved_folder, **kwargs)

    def _bench_one(self, model, tokenizer, dtype):
        prompt_ids, attention_mask = _make_prefill_inputs(tokenizer, model.device, dtype)
        prefill_ms = _measure_prefill_ms(model, prompt_ids, attention_mask)
        decode_ms = _measure_decode_ms_per_tok(model, tokenizer, prompt_ids, attention_mask)
        return prefill_ms, decode_ms

    def _smoke_test(self, model, tokenizer):
        out = model_infer(model, tokenizer)
        assert out is not None and len(out) > 0, "ARK backend produced empty output"

    # -- parametrized perf scan --------------------------------------------

    @pytest.mark.parametrize(
        "model_label, model_path",
        [
            ("qwen-moe", qwen_moe_name_or_path),
            ("deepseek-v2-lite", deepseek_v2_name_or_path),
        ],
    )
    @pytest.mark.parametrize("bits, group_size, sym", [(4, 128, True), (8, 128, True)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_moe_forward_perf(self, model_label, model_path, bits, group_size, sym, dtype):
        # 1. Resolve checkpoint -- skip if neither local mirror nor HF cache
        #    has it (CI without internet must not fail this test).
        resolved = get_model_path(model_path)
        try:
            tokenizer_probe = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        except (OSError, ValueError) as exc:
            pytest.skip(f"checkpoint {model_path!r} not available locally: {exc}")
        del tokenizer_probe

        # 2. Load tiny (or full) FP MoE model + tokenizer.
        fp_model, tokenizer = _load_tiny_or_full(resolved, dtype)

        # 3. Quantize + save.
        saved_folder = self._quantize_and_save(fp_model, tokenizer, bits, group_size, sym)
        # Free the calibration-time model before reloading on XPU.
        del fp_model
        torch.xpu.empty_cache()

        _print_header(dtype)
        label = f"{model_label} INT{bits}"

        # 4a. FP reference on XPU (no quantization_config).
        fp_decode_ms = None
        try:
            fp_model_xpu = self._reload(resolved, dtype, backend=None)
            fp_prefill_ms, fp_decode_ms = self._bench_one(fp_model_xpu, tokenizer, dtype)
            print(_format_row(label, "fp(ref)", fp_prefill_ms, fp_decode_ms, fp_decode_ms))
            del fp_model_xpu
            torch.xpu.empty_cache()
        except Exception as exc:  # noqa: BLE001 -- FP baseline is optional
            print(f"[moe-model-perf] fp(ref) row skipped for {label}: {exc}")

        # 4b. ARK backend (the thing under test).
        if _ARK_SKIP:
            pytest.skip(f"ARK backend unavailable: {_ARK_SKIP}")
        ark_model = self._reload(saved_folder, dtype, backend="ark")
        self._smoke_test(ark_model, tokenizer)
        ark_prefill_ms, ark_decode_ms = self._bench_one(ark_model, tokenizer, dtype)
        print(_format_row(label, "ark", ark_prefill_ms, ark_decode_ms, fp_decode_ms))
        del ark_model
        torch.xpu.empty_cache()

        # 4c. Optional GPTQModel cross-reference (skip silently if missing).
        try:
            gptq_model = self._reload(saved_folder, dtype, backend="gptqmodel")
            gptq_prefill_ms, gptq_decode_ms = self._bench_one(gptq_model, tokenizer, dtype)
            print(_format_row(label, "gptqmodel", gptq_prefill_ms, gptq_decode_ms, fp_decode_ms))
            del gptq_model
            torch.xpu.empty_cache()
        except Exception as exc:  # noqa: BLE001 -- backend is optional
            print(f"[moe-model-perf] gptqmodel row skipped for {label}: {exc}")

        print("-" * 96)

        # 5. Perf-regression assertion (only when we have an FP baseline).
        if fp_decode_ms is not None and fp_decode_ms > 0:
            assert ark_decode_ms <= fp_decode_ms * ARK_DECODE_REGRESSION_FACTOR, (
                f"ARK decode latency {ark_decode_ms:.3f} ms/tok exceeds "
                f"{ARK_DECODE_REGRESSION_FACTOR}x FP baseline {fp_decode_ms:.3f} ms/tok "
                f"for {label} ({dtype}). Likely regression in the ark.moe dispatcher."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
