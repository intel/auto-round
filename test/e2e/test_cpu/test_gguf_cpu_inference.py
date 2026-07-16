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
"""End-to-end GGUF export + llama.cpp CPU inference.

The unit tests in :mod:`test.unit.test_cpu.export.test_gguf_format`
already cover *exporting* GGUF, and they load the resulting file with
``transformers.AutoModelForCausalLM`` for sanity checks.  This file
covers the more interesting half of the user journey: the actual
dequantization + forward path used in production, which is
``llama-cpp-python`` (or ``llama.cpp``).

Each test:

    1. Quantizes a real model with ``auto-round --format gguf:q*_*``.
    2. Loads the resulting ``.gguf`` file with ``llama_cpp.Llama``.
    3. Runs a few greedy generations and checks that the output is
       non-garbage and contains an expected keyword.
    4. Measures tokens/s (greedy decode, batch=1) and records it.

If ``llama-cpp-python`` is not installed in the test environment the
tests are auto-skipped via the ``require_llama_cpp`` fixture.
"""

from __future__ import annotations

import os
import time
from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    assert_non_garbage_output,
    quantize_and_save,
    record,
)
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# (hf_id, scheme, expected_keyword)
# ``expected_keyword`` is a loose lower-case substring; we don't try to
# nail the answer to a single string, we just want to make sure the
# model still produces fluent English.
PROMPT = "The capital of France is"


CASES = [
    ("Qwen/Qwen3-0.6B", "gguf:q4_k_m", "paris"),
    ("Qwen/Qwen3-0.6B", "gguf:q5_k_m", "paris"),
    ("Qwen/Qwen3-0.6B", "gguf:q8_0", "paris"),
    ("Qwen/Qwen3-0.6B", "gguf:q2_k", "paris"),  # ultra-low bit - loose floor
    ("meta-llama/Llama-3.2-1B", "gguf:q4_k_m", "paris"),
]

# Higher-quality schemes run on slightly bigger models if the host has RAM.
LARGE_CASES = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "gguf:q4_k_m", "paris"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "gguf:q6_k", "paris"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case_id(model_id: str, scheme: str) -> str:
    return f"{model_id.split('/')[-1].lower()}-{scheme.replace(':', '_')}"


def _build_llamacpp(gguf_path: str, n_ctx: int = 512, n_threads: int = 0):
    """Construct a llama_cpp.Llama.  ``n_threads=0`` means "use all cores"."""
    try:
        from llama_cpp import Llama
    except ImportError as e:
        pytest.skip(f"llama-cpp-python is not installed: {e}")

    return Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads or os.cpu_count() or 4,
        verbose=False,
        logits_all=False,
    )


def _find_gguf(save_dir: str) -> str:
    """Locate the .gguf file produced by ``auto-round --format gguf:*``."""
    matches: List[str] = []
    for root, _, files in os.walk(save_dir):
        for name in files:
            if name.endswith(".gguf"):
                matches.append(os.path.join(root, name))
    assert matches, f"no .gguf file found under {save_dir}"
    # ``auto-round`` names the file after the model dir, e.g.
    # ``Qwen3-0.6B.Q4_K_M.gguf``.  When several are present (sharding)
    # we pick the largest.
    matches.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGgufCpuInference:
    """GGUF export + llama.cpp CPU inference."""

    @pytest.mark.parametrize("model_id,scheme,expected", CASES, ids=[_case_id(m, s) for m, s, _ in CASES])
    def test_quantize_and_generate(self, model_id, scheme, expected, tmp_path, require_llama_cpp):
        from test.helpers import get_model_path

        model_id = get_model_path(model_id)
        save_dir = str(tmp_path / "gguf_out")

        # ---- 1. quantize + export ----
        t0 = time.perf_counter()
        saved = quantize_and_save(
            model_id=model_id,
            bits=4,  # ignored for gguf:*; the scheme carries the bit width
            group_size=32,  # GGUF default
            sym=True,
            fmt=scheme,
            output_dir=save_dir,
            iters=200,
            nsamples=128,
            seqlen=2048,
        )
        quant_time = time.perf_counter() - t0

        gguf_path = _find_gguf(saved)

        # ---- 2. load with llama.cpp ----
        llm = _build_llamacpp(gguf_path)
        try:
            # Warm-up to amortize first-call overhead.
            llm(PROMPT, max_tokens=8, temperature=0.0, echo=False)

            t0 = time.perf_counter()
            out = llm(PROMPT, max_tokens=32, temperature=0.0, echo=False)
            decode_time = time.perf_counter() - t0

            text = out["choices"][0]["text"]
            n_tokens = out["usage"].get("completion_tokens", 0) or len(out["choices"][0].get("logits", [])) or 0
            tok_per_s = (n_tokens / decode_time) if decode_time > 0 and n_tokens else 0.0

            assert_non_garbage_output(text)
            # Loose keyword check; q2_k is allowed to miss it.
            if "q2_k" not in scheme:
                assert expected in text.lower(), f"{model_id} {scheme} did not produce '{expected}': {text!r}"

            record(
                EvalResult(
                    test=self.__class__.__name__,
                    model=model_id,
                    fmt=scheme,
                    bits=4 if "q2_k" in scheme else 0,  # 0 = "depends on scheme"
                    group_size=32,
                    sym=True,
                    task="generate",
                    metric="tok_per_s",
                    value=tok_per_s,
                    wall_time_s=quant_time + decode_time,
                    extra={"quant_time_s": quant_time, "decode_time_s": decode_time, "tokens": n_tokens},
                )
            )

            print(
                f"\n[GGUF-cpu] {model_id} {scheme} -> "
                f"{tok_per_s:.1f} tok/s ({n_tokens} tokens in {decode_time:.1f}s)"
            )
        finally:
            try:
                llm.close()
            except Exception:
                pass


class TestGgufCpuLarge:
    """Heavier 1.5B-class cases; skipped on hosts with <16 GiB free RAM."""

    @pytest.mark.parametrize(
        "model_id,scheme,expected",
        LARGE_CASES,
        ids=[_case_id(m, s) for m, s, _ in LARGE_CASES],
    )
    def test_quantize_and_generate(self, model_id, scheme, expected, tmp_path, require_llama_cpp, require_ram):
        # Reuse the smaller case's logic.  ``require_ram`` reads
        # ``model_case.min_ram_gib`` from a per-case constant below.
        from test.helpers import get_model_path

        model_id = get_model_path(model_id)
        save_dir = str(tmp_path / "gguf_out")

        saved = quantize_and_save(
            model_id=model_id,
            bits=4,
            group_size=32,
            sym=True,
            fmt=scheme,
            output_dir=save_dir,
            iters=200,
            nsamples=128,
            seqlen=2048,
        )
        gguf_path = _find_gguf(saved)
        llm = _build_llamacpp(gguf_path)
        try:
            llm(PROMPT, max_tokens=8, temperature=0.0, echo=False)
            t0 = time.perf_counter()
            out = llm(PROMPT, max_tokens=32, temperature=0.0, echo=False)
            decode_time = time.perf_counter() - t0
            text = out["choices"][0]["text"]
            n_tokens = out["usage"].get("completion_tokens", 0)
            tok_per_s = (n_tokens / decode_time) if decode_time > 0 and n_tokens else 0.0

            assert_non_garbage_output(text)
            assert expected in text.lower(), f"{model_id} {scheme} did not produce '{expected}': {text!r}"

            record(
                EvalResult(
                    test=self.__class__.__name__,
                    model=model_id,
                    fmt=scheme,
                    bits=4,
                    group_size=32,
                    sym=True,
                    task="generate",
                    metric="tok_per_s",
                    value=tok_per_s,
                    wall_time_s=decode_time,
                    extra={"tokens": n_tokens},
                )
            )
            print(f"\n[GGUF-cpu-large] {model_id} {scheme} -> {tok_per_s:.1f} tok/s")
        finally:
            try:
                llm.close()
            except Exception:
                pass
