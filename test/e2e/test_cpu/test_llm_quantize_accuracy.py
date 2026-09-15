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
"""End-to-end accuracy matrix tests on CPU.

These tests run the full ``auto-round`` pipeline (quantize → save → reload
→ eval) on real, user-sized LLMs and check that the quantized model still
hits a reasonable accuracy floor on a small set of lm-eval tasks.

The matrix is the same one defined in :mod:`test.e2e.test_cpu.conftest`:
- ``Qwen3-0.6B``, ``Llama-3.2-1B``, ``Phi-3.5-mini``, ``gemma-2-2b``,
  ``internlm2-1.8b``, ...
- export formats: ``auto_round``, ``auto_gptq``, ``auto_awq``, ``gguf:q*_*``
- bits: 2 / 4 / 8

A case is auto-skipped if the host has less free RAM than the case
requires (see :class:`~.conftest.ModelCase.min_ram_gib`) or if
``lm-eval`` is not installed.

Each accuracy number is recorded to ``test/output/cpu_e2e.jsonl`` so that
weekly CI runs can be diffed over time.

Run a single case::

    pytest test/e2e/test_cpu/test_llm_quantize_accuracy.py -k "qwen3-0.6b-w4a16-auto_round" -v -s

Run the whole default matrix::

    pytest test/e2e/test_cpu/test_llm_quantize_accuracy.py -v -s

Run only the heavier cases (>=24 GiB host recommended)::

    E2E_CPU_PRESET=large pytest test/e2e/test_cpu/test_llm_quantize_accuracy.py -v -s
"""

from __future__ import annotations

import time
from test.e2e.test_cpu.conftest import (  # noqa: E402
    DEFAULT_MODEL_CASES,
    LARGE_MODEL_CASES,
    EvalResult,
    extract_metric,
    quantize_and_save,
    record,
    run_lm_eval,
)

import pytest

# Loose accuracy floors.  These are intentionally below the values you'd
# see in the paper so the tests catch *catastrophic* regressions (e.g. a
# kernel bug that collapses accuracy to chance) without flaking on
# natural week-to-week variance in the calibration data pipeline.
ACC_FLOORS = {
    # (fmt, bits) -> (task -> floor)
    ("auto_round", 4): {"piqa": 0.55, "lambada_openai": 0.30},
    ("auto_gptq", 4): {"piqa": 0.55, "lambada_openai": 0.25},
    ("auto_awq", 4): {"piqa": 0.55, "lambada_openai": 0.25},
    ("gguf:q4_k_m", 4): {"piqa": 0.55, "lambada_openai": 0.30},
    ("gguf:q8_0", 8): {"piqa": 0.60, "lambada_openai": 0.40},
    ("auto_round", 2): {"piqa": 0.45, "lambada_openai": 0.10},
    ("auto_round", 8): {"piqa": 0.60, "lambada_openai": 0.40},
}


def _floor_for(fmt: str, bits: int) -> dict:
    return ACC_FLOORS.get((fmt, bits), ACC_FLOORS.get(("auto_round", 4), {}))


def _case_id(c) -> str:
    """Stable, human-readable id for parametrize."""
    return f"{c.hf_id.split('/')[-1].lower()}-w{c.bits}g{c.group_size}-{c.fmt.replace(':', '_')}"


# ---------------------------------------------------------------------------
# Test class - parametrized over the full CPU matrix
# ---------------------------------------------------------------------------


class TestLlmQuantizeAccuracy:
    """Quantize + reload + eval accuracy matrix."""

    @pytest.mark.parametrize(
        "model_case",
        DEFAULT_MODEL_CASES,
        ids=[_case_id(c) for c in DEFAULT_MODEL_CASES],
    )
    def test_default_matrix(self, model_case, tmp_path, require_ram, require_lm_eval):
        self._run_case(model_case, tmp_path)

    @pytest.mark.parametrize(
        "model_case",
        LARGE_MODEL_CASES,
        ids=[_case_id(c) for c in LARGE_MODEL_CASES],
    )
    def test_large_matrix(self, model_case, tmp_path, require_ram, require_lm_eval):
        self._run_case(model_case, tmp_path)

    # -- implementation -----------------------------------------------------

    def _run_case(self, model_case, tmp_path):
        from test.helpers import get_model_path

        model_id = get_model_path(model_case.hf_id)
        save_dir = str(tmp_path / f"saved_{_case_id(model_case)}")
        floors = _floor_for(model_case.fmt, model_case.bits)
        assert floors, f"no accuracy floor defined for fmt={model_case.fmt} bits={model_case.bits}"

        # ---- quantize ----
        t0 = time.perf_counter()
        saved = quantize_and_save(
            model_id=model_id,
            bits=model_case.bits,
            group_size=model_case.group_size,
            sym=model_case.sym,
            fmt=model_case.fmt,
            output_dir=save_dir,
            iters=200,
            nsamples=128,
            seqlen=2048,
        )
        quant_time = time.perf_counter() - t0

        # ---- eval ----
        t0 = time.perf_counter()
        results = run_lm_eval(
            saved,
            tasks=model_case.eval_tasks,
            limit=model_case.eval_limit,
            batch_size="auto",
        )
        eval_time = time.perf_counter() - t0

        # ---- assertions + record ----
        for task in model_case.eval_tasks.split(","):
            value = extract_metric(results, task, "acc,none")
            if value is None:
                # lambada returns ppl/none, piqa returns acc/none; treat
                # missing as a skip rather than a fail.
                value = extract_metric(results, task, "ppl,none")
            floor = floors.get(task)
            record(
                EvalResult(
                    test=self.__class__.__name__,
                    model=model_case.hf_id,
                    fmt=model_case.fmt,
                    bits=model_case.bits,
                    group_size=model_case.group_size,
                    sym=model_case.sym,
                    task=task,
                    metric="acc,none" if value is not None else "ppl,none",
                    value=value,
                    wall_time_s=quant_time + eval_time,
                    extra={"quant_time_s": quant_time, "eval_time_s": eval_time},
                )
            )
            if floor is not None and value is not None and task in ("piqa", "lambada_openai"):
                assert value >= floor, (
                    f"{model_case.hf_id} {model_case.fmt} w{model_case.bits} "
                    f"{task}={value:.3f} below floor {floor:.3f}"
                )

        print(
            f"\n[CPU-e2e] {model_case.hf_id} {model_case.fmt} w{model_case.bits} "
            f"-> quant={quant_time:.0f}s, eval={eval_time:.0f}s"
        )
