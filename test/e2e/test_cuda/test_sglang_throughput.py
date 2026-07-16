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
"""End-to-end throughput & latency tests for the **SGLang** inference engine.

These tests mirror :mod:`test_vllm_throughput` but drive the SGLang
``Engine`` instead.  SGLang has slightly different defaults
(``mem_fraction_static`` instead of ``gpu_memory_utilization``) and
exposes its own ``token_usage`` and ``decode throughput`` fields, which
we record alongside the wall-clock measurement.

The test pipeline is identical:

    1. ``auto-round`` Python API → quantized checkpoint.
    2. ``sgl.Engine`` loads the checkpoint.
    3. Warm up, then run a fixed prompt batch.
    4. Record output tokens/s, decode-only tokens/s and TTFT.

Run a single test locally::

    pytest test/e2e/test_cuda/test_sglang_throughput.py::TestSglangThroughput::test_quantize_and_serve \\
           --e2e-model-preset=default -v -s
"""

import json
import os
import time
from test.e2e.test_cuda.conftest import (  # noqa: E402
    BenchResult,
    free_cuda,
    make_bench_prompts,
    quantize_and_save,
)
from typing import List

import pytest
import torch

# ---------------------------------------------------------------------------
# Output sink
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_THIS_DIR, "..", "..", "output")
_OUTPUT_FILE = os.path.normpath(os.path.join(_OUTPUT_DIR, "sglang_throughput.jsonl"))


def _record(result: BenchResult) -> None:
    """Append a benchmark result to a JSONL file for trend tracking."""
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(_OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.__dict__) + "\n")


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

# sglang 0.2+ has a known multiprocessing.resource_tracker bug on Linux that
# manifests as ``[Errno 10] No child processes`` during teardown. We patch
# ``ResourceTracker._stop`` to swallow that exception, mirroring the
# workaround in ``test/integration/test_cuda/test_sglang.py``.
import gc  # noqa: E402
import multiprocessing.resource_tracker  # noqa: E402

_orig_stop = multiprocessing.resource_tracker.ResourceTracker._stop


def _patched_stop(self, *args, _orig=_orig_stop, **kwargs):
    if _orig is not None:
        try:
            _orig(self, *args, **kwargs)
        except ChildProcessError:
            pass


multiprocessing.resource_tracker.ResourceTracker._stop = _patched_stop

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="SGLang throughput tests require a CUDA GPU",
    ),
]


# ---------------------------------------------------------------------------
# SGLang wrapper
# ---------------------------------------------------------------------------


def _build_sglang_engine(model_path: str, mem_fraction_static: float, context_len: int):
    """Construct a sglang engine configured for AutoRound-quantized checkpoints.

    ``disable_piecewise_cuda_graph=True`` and a small ``cuda_graph_bs`` list
    avoid the gptq_marlin_repack JIT kernel tripping over Blackwell SM 12.x
    when CUDA < 12.9 (the same constraint as the vLLM tests).
    """
    try:
        import sglang as sgl
    except ImportError as e:
        pytest.skip(f"sglang is not installed: {e}")

    return sgl.Engine(
        model_path=model_path,
        mem_fraction_static=mem_fraction_static,
        context_length=context_len,
        # Keep cuda-graphs conservative – AutoRound-int4 checkpoints don't
        # benefit from large captured graphs on small workloads.
        disable_piecewise_cuda_graph=True,
        cuda_graph_bs=[1, 2, 4],
    )


def _run_sglang_benchmark(
    model_path: str,
    max_new_tokens: int = 128,
    num_prompts: int = 8,
    mem_fraction_static: float = 0.7,
    context_len: int = 2048,
    warmup: int = 1,
) -> BenchResult:
    """End-to-end benchmark: load with SGLang, warm up, then time a prompt batch."""
    from test.helpers import get_model_path

    model_path = get_model_path(model_path) if "/" in model_path else model_path

    llm = _build_sglang_engine(model_path, mem_fraction_static=mem_fraction_static, context_len=context_len)
    try:
        # SGLang's tokenizer is the HuggingFace tokenizer.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompts = make_bench_prompts(tokenizer, num_prompts=num_prompts, target_input_tokens=64)

        sampling_params = {
            "temperature": 0.0,  # greedy ⇒ deterministic output length
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        }

        # --- warm-up ---
        for _ in range(max(0, warmup)):
            llm.generate(prompts[:1], sampling_params)

        # --- timed run ---
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        total_time = time.perf_counter() - t0

        n_out_tokens = sum(len(o.get("meta_info", {}).get("output_ids", []) or []) for o in outputs)
        gen_tokens_per_s = n_out_tokens / max(total_time, 1e-6)

        # SGLang's per-request meta_info exposes per-request decode throughput
        # and TTFT; we average over the batch for a single number.
        decode_tps = []
        ttfts = []
        for o in outputs:
            info = o.get("meta_info", {}) or {}
            # ``completion_tokens`` and ``e2e_latency`` are always present.
            comp = info.get("completion_tokens")
            lat = info.get("e2e_latency")
            if comp and lat and lat > 0:
                decode_tps.append(comp / lat)
            # TTFT is reported in ``prefill_latency`` for batch_size=1.
            ttft = info.get("prefill_latency")
            if ttft and ttft > 0:
                ttfts.append(ttft)

        gen_tokens_per_s_avg = sum(decode_tps) / len(decode_tps) if decode_tps else None
        ttft_s = sum(ttfts) / len(ttfts) if ttfts else None

        return BenchResult(
            engine="sglang",
            model=os.path.basename(model_path.rstrip("/")),
            fmt=os.environ.get("_AR_E2E_FMT", "auto_round"),
            bits=int(os.environ.get("_AR_E2E_BITS", "4")),
            group_size=int(os.environ.get("_AR_E2E_GS", "128")),
            num_prompts=num_prompts,
            max_new_tokens=max_new_tokens,
            total_time_s=total_time,
            output_tokens_per_s=n_out_tokens / max(total_time, 1e-6),
            gen_tokens_per_s=gen_tokens_per_s_avg,
            ttft_s=ttft_s,
            sample_output=outputs[0]["text"],
        )
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSglangThroughput:
    """Quantize-with-autoround + serve-with-sglang end-to-end suite."""

    @pytest.mark.parametrize(
        "model_case",
        [
            pytest.param(
                __import__("test.e2e.test_cuda.conftest", fromlist=["ModelCase"]).ModelCase(
                    "Qwen/Qwen3-1.7B", 4, 128, True, "auto_round", min_gpu_gib=10
                ),
                id="qwen3-1.7b-w4a16-auto_round",
            ),
            pytest.param(
                __import__("test.e2e.test_cuda.conftest", fromlist=["ModelCase"]).ModelCase(
                    "Qwen/Qwen3-1.7B", 4, 128, True, "auto_gptq", min_gpu_gib=10
                ),
                id="qwen3-1.7b-w4a16-auto_gptq",
            ),
            pytest.param(
                __import__("test.e2e.test_cuda.conftest", fromlist=["ModelCase"]).ModelCase(
                    "Qwen/Qwen3-1.7B", 4, 128, True, "auto_awq", min_gpu_gib=10
                ),
                id="qwen3-1.7b-w4a16-auto_awq",
            ),
        ],
    )
    def test_quantize_and_serve(self, model_case, tmp_path, require_gpu_memory):
        save_dir = str(tmp_path / f"saved_{model_case.fmt}_w{model_case.bits}")

        os.environ["_AR_E2E_FMT"] = model_case.fmt
        os.environ["_AR_E2E_BITS"] = str(model_case.bits)
        os.environ["_AR_E2E_GS"] = str(model_case.group_size)

        saved = quantize_and_save(
            model_id=model_case.hf_id,
            bits=model_case.bits,
            group_size=model_case.group_size,
            sym=model_case.sym,
            fmt=model_case.fmt,
            output_dir=save_dir,
            iters=200,
            nsamples=128,
            seqlen=2048,
        )

        result = _run_sglang_benchmark(saved, max_new_tokens=64, num_prompts=4, mem_fraction_static=0.7)
        _record(result)

        assert result.sample_output.strip(), "SGLang produced empty output"
        assert "!!!" not in result.sample_output, "SGLang produced garbage output"
        assert result.output_tokens_per_s > 0
        assert (
            result.output_tokens_per_s >= 1.0
        ), f"SGLang throughput suspiciously low: {result.output_tokens_per_s:.2f} tok/s"

        print(
            f"\n[SGLang] {model_case.hf_id} {model_case.fmt} w{model_case.bits} "
            f"-> {result.output_tokens_per_s:.1f} tok/s "
            f"(decode-only: {result.gen_tokens_per_s}, ttft: {result.ttft_s})"
        )


class TestSglangLarge:
    """Heavier cases that need >=24 GiB; skipped on smaller GPUs."""

    @pytest.mark.parametrize(
        "model_case",
        [
            pytest.param(
                __import__("test.e2e.test_cuda.conftest", fromlist=["ModelCase"]).ModelCase(
                    "Qwen/Qwen2.5-7B-Instruct", 4, 128, True, "auto_round", min_gpu_gib=18
                ),
                id="qwen2.5-7b-w4a16-auto_round",
            ),
            pytest.param(
                __import__("test.e2e.test_cuda.conftest", fromlist=["ModelCase"]).ModelCase(
                    "meta-llama/Llama-3.2-3B-Instruct", 4, 128, True, "auto_round", min_gpu_gib=12
                ),
                id="llama-3.2-3b-w4a16-auto_round",
            ),
        ],
    )
    def test_quantize_and_serve(self, model_case, tmp_path, require_gpu_memory):
        save_dir = str(tmp_path / f"saved_{model_case.fmt}_w{model_case.bits}")

        os.environ["_AR_E2E_FMT"] = model_case.fmt
        os.environ["_AR_E2E_BITS"] = str(model_case.bits)
        os.environ["_AR_E2E_GS"] = str(model_case.group_size)

        saved = quantize_and_save(
            model_id=model_case.hf_id,
            bits=model_case.bits,
            group_size=model_case.group_size,
            sym=model_case.sym,
            fmt=model_case.fmt,
            output_dir=save_dir,
            iters=200,
            nsamples=128,
            seqlen=2048,
        )

        result = _run_sglang_benchmark(saved, max_new_tokens=64, num_prompts=4, mem_fraction_static=0.75)
        _record(result)

        assert result.sample_output.strip()
        assert "!!!" not in result.sample_output
        assert result.output_tokens_per_s >= 1.0

        print(
            f"\n[SGLang-large] {model_case.hf_id} {model_case.fmt} w{model_case.bits} "
            f"-> {result.output_tokens_per_s:.1f} tok/s"
        )


# ---------------------------------------------------------------------------
# CLI / mixed-format regression
# ---------------------------------------------------------------------------


def test_sglang_awq_format_via_cli(require_cuda):
    """``auto-round --format auto_round:auto_awq`` → SGLang load.

    This mirrors the existing ``test_ar_format_sglang`` in the
    integration suite, but is parameterised over a real model on GPU so
    that the same code path can be re-checked in the e2e pipeline.
    """
    import sys
    import tempfile
    from test.helpers import get_model_path

    model = get_model_path("Qwen/Qwen3-0.6B")
    with tempfile.TemporaryDirectory() as out:
        cmd = (
            f"{sys.executable} -m auto_round --model {model} "
            f"--scheme W4A16 --iters 0 --disable_opt_rtn --format auto_round:auto_awq "
            f"--output_dir {out}"
        )
        rc = os.system(cmd)
        assert rc == 0, f"awq-format quant via CLI failed (rc={rc})"

        # SGLang will JIT-compile the awq kernels during the first generate
        # call, so we expect the first request to be slow but the second
        # to be representative.
        llm = _build_sglang_engine(out, mem_fraction_static=0.5, context_len=1024)
        try:
            outputs = llm.generate(["Hello, my name is"], {"max_new_tokens": 16, "temperature": 0.0})
            text = outputs[0]["text"]
            assert text.strip() and "!!!" not in text
        finally:
            try:
                llm.shutdown()
            except Exception:
                pass
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
