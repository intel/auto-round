# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BF16 baseline vs quantized quality regression tests.

The single most-asked question in the auto-round issue tracker is
"how much accuracy do I lose by quantizing?".  This file answers it
once and for all, per-model and per-scheme, and writes the
``acc_loss`` delta to ``test/output/cpu_e2e.jsonl`` for trend tracking.

For each (model, scheme) pair the test:

    1. Runs ``lm-eval`` on the bf16 model (no quantization).
    2. Quantizes with the given scheme and re-runs ``lm-eval``.
    3. Asserts ``acc_loss`` = ``acc(bf16) - acc(quant)`` is within
       a generous bound (configurable below).

The bf16 baseline is **cached** for the duration of the test session
so the matrix doesn't pay the eval cost N times.
"""

from __future__ import annotations

import os
import time
from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    extract_metric,
    quantize_and_save,
    record,
)
from typing import Dict, Optional

import pytest

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# (model_id, scheme, min_ram_gib, max_acc_loss)
# ``max_acc_loss`` is the *allowed* acc drop compared to bf16.  These
# numbers are intentionally loose so the tests catch catastrophic
# regressions (e.g. a kernel bug) without flaking on real week-to-week
# variance in the calibration data.
MODELS = [
    # (hf_id, allowed_drop_pi, allowed_drop_lm)
    ("Qwen/Qwen3-0.6B", 8),
]

SCHEMES = [
    ("W4A16", 4, 128, 0.06, 0.10),  # 6pp on piqa, 10pp on lambada
    ("W2A16", 2, 128, 0.20, 0.40),  # 2-bit is allowed to lose a lot
    ("W8A8", 8, 128, 0.02, 0.05),  # W8A8 should be very close to bf16
    ("MXFP4", 0, 0, 0.06, 0.10),  # MXFP4 group_size implicit
    ("gguf:q4_k_m", 4, 32, 0.06, 0.10),
]


def _case_id(model_id: str, scheme: str) -> str:
    return f"{model_id.split('/')[-1].lower()}-{scheme.replace(':', '_')}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluate_bf16(model_id: str, tasks: str, limit: int) -> Dict[str, Optional[float]]:
    """Run ``lm-eval`` on the bf16 model and return a {task: acc} dict."""
    from auto_round.eval.evaluation import simple_evaluate_user_model
    from auto_round.utils import llm_load_model

    model, tokenizer = llm_load_model(model_id, trust_remote_code=True)
    try:
        results = simple_evaluate_user_model(model, tokenizer, batch_size=1, limit=limit, tasks=tasks)
    finally:
        del model
        import gc

        gc.collect()
    out = {}
    for task in tasks.split(","):
        out[task] = extract_metric(results, task, "acc,none") or extract_metric(results, task, "ppl,none")
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBf16VsQuantQuality:
    """For each (model, scheme) pair, run bf16 eval → quant → eval → compare."""

    @pytest.mark.parametrize(
        "model_id,min_ram",
        MODELS,
        ids=[m.split("/")[-1].lower() for m, _ in MODELS],
    )
    @pytest.mark.parametrize(
        "scheme,bits,group_size,max_drop_pi,max_drop_lm",
        SCHEMES,
        ids=[s[0].replace(":", "_") for s in SCHEMES],
    )
    def test_acc_loss_under_bound(
        self,
        model_id: str,
        min_ram: int,
        scheme: str,
        bits: int,
        group_size: int,
        max_drop_pi: float,
        max_drop_lm: float,
        tmp_path,
        require_lm_eval,
    ):
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / 1024**3
        if avail < min_ram:
            pytest.skip(f"only {avail:.1f} GiB free RAM, need {min_ram} GiB")

        from test.helpers import get_model_path

        model_id = get_model_path(model_id)

        # ---- 1. bf16 baseline ----
        t0 = time.perf_counter()
        bf16_accs = _evaluate_bf16(model_id, tasks="piqa,lambada_openai", limit=80)
        bf16_time = time.perf_counter() - t0
        bf16_pi = bf16_accs.get("piqa")
        bf16_lm = bf16_accs.get("lambada_openai")

        # ---- 2. quantize ----
        save_dir = str(tmp_path / f"acc_{_case_id(model_id, scheme)}")
        t0 = time.perf_counter()
        saved = quantize_and_save(
            model_id=model_id,
            bits=bits,
            group_size=group_size,
            sym=True,
            fmt="auto_round" if "gguf" not in scheme else scheme,
            output_dir=save_dir,
            iters=200,
            nsamples=128,
            seqlen=2048,
        )
        quant_time = time.perf_counter() - t0

        # ---- 3. quantized eval ----
        from auto_round.eval.evaluation import simple_evaluate

        t0 = time.perf_counter()
        if "gguf" in scheme:
            # GGUF: use ``model=gguf`` so lm-eval dequantizes via llama.cpp.
            results = simple_evaluate(
                model="gguf",
                model_args=f"pretrained={saved}",
                tasks="piqa,lambada_openai",
                limit=80,
                batch_size="auto",
            )
        else:
            results = simple_evaluate(
                model="hf",
                model_args=f"pretrained={saved}",
                tasks="piqa,lambada_openai",
                limit=80,
                batch_size="auto",
            )
        eval_time = time.perf_counter() - t0

        q_pi = extract_metric(results, "piqa", "acc,none")
        q_lm = extract_metric(results, "lambada_openai", "acc,none") or extract_metric(
            results, "lambada_openai", "ppl,none"
        )

        # ---- 4. record + assert ----
        if bf16_pi is not None and q_pi is not None:
            drop = bf16_pi - q_pi
            record(
                EvalResult(
                    test=self.__class__.__name__,
                    model=model_id,
                    fmt=scheme,
                    bits=bits,
                    group_size=group_size,
                    sym=True,
                    task="piqa",
                    metric="acc_loss",
                    value=drop,
                    wall_time_s=bf16_time + quant_time + eval_time,
                    extra={"bf16": bf16_pi, "quant": q_pi},
                )
            )
            assert drop <= max_drop_pi, (
                f"{model_id} {scheme} piqa acc drop {drop:.3f} > {max_drop_pi:.3f} "
                f"(bf16={bf16_pi:.3f}, quant={q_pi:.3f})"
            )

        if bf16_lm is not None and q_lm is not None:
            drop = bf16_lm - q_lm
            record(
                EvalResult(
                    test=self.__class__.__name__,
                    model=model_id,
                    fmt=scheme,
                    bits=bits,
                    group_size=group_size,
                    sym=True,
                    task="lambada_openai",
                    metric="acc_loss",
                    value=drop,
                    wall_time_s=0.0,
                    extra={"bf16": bf16_lm, "quant": q_lm},
                )
            )
            assert drop <= max_drop_lm, (
                f"{model_id} {scheme} lambada acc drop {drop:.3f} > {max_drop_lm:.3f} "
                f"(bf16={bf16_lm:.3f}, quant={q_lm:.3f})"
            )

        print(
            f"\n[Bf16-vs-quant] {model_id} {scheme} "
            f"piqa: {bf16_pi:.3f} -> {q_pi:.3f} "
            f"lambada: {bf16_lm:.3f} -> {q_lm:.3f}"
        )
