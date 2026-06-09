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
"""End-to-end throughput & latency tests for the **vLLM** inference engine.

Each test in this file exercises the full pipeline that a real user
would follow:

    1. Quantize a real LLM with ``auto-round`` (Python API).
    2. Save the checkpoint in a deployment-ready format
       (``auto_round``, ``auto_gptq``, ``auto_awq``, ``gguf``, ...).
    3. Load the checkpoint with the official ``vllm.LLM`` engine.
    4. Run a fixed prompt batch and measure wall-clock throughput
       (output tokens/s) and decode-only throughput.

The numbers are recorded in a JSON line at the end of each test under
``test/e2e/output/vllm_throughput.jsonl`` so weekly CI runs can be
diffed over time.

These tests are slow (a single 1.7B W4A16 case takes ~5 min on an
A100) and are expected to be scheduled weekly, not on every PR.

Run a single test locally::

    pytest test/e2e/test_cuda/test_vllm_throughput.py::TestVllmThroughput::test_quantize_and_serve \\
           --e2e-model-preset=default -v -s

Run the full (large) matrix::

    pytest test/e2e/test_cuda/test_vllm_throughput.py \\
           --e2e-model-preset=all -v -s
"""

import json
import os
import time
from typing import List

import pytest
import torch

from test.e2e.test_cuda.conftest import (  # noqa: E402
    BenchResult,
    free_cuda,
    make_bench_prompts,
    quantize_and_save,
)


# ---------------------------------------------------------------------------
# Output sink
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_THIS_DIR, "..", "..", "output")
_OUTPUT_FILE = os.path.normpath(os.path.join(_OUTPUT_DIR, "vllm_throughput.jsonl"))


def _record(result: BenchResult) -> None:
    """Append a benchmark result to a JSONL file for trend tracking."""
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(_OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.__dict__) + "\n")


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="vLLM throughput tests require a CUDA GPU",
    ),
    # The "low" preset fits on a 24 GiB card; the "large" preset needs ~40+ GiB.
]


# ---------------------------------------------------------------------------
# vLLM-engine wrapper
# ---------------------------------------------------------------------------


def _build_vllm_engine(model_path: str, max_model_len: int, gpu_mem_util: float):
    """Construct a vLLM engine configured for AutoRound-quantized checkpoints."""
    try:
        from vllm import LLM
        from vllm.platforms import current_platform
    except ImportError as e:
        pytest.skip(f"vllm is not installed: {e}")

    if not (current_platform.is_cpu() or current_platform.is_xpu() or current_platform.is_cuda()):
        pytest.skip("vLLM tests only run on CPU/XPU/CUDA")

    # ``auto-round`` is registered as a vLLM plugin via entrypoints.  We still
    # pass ``quantization="auto-round"`` explicitly to make the dependency
    # obvious in CI logs and to be future-proof against entrypoint changes.
    return LLM(
        model=model_path,
        quantization="auto-round",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        dtype="auto",
        enforce_eager=False,
    )


def _run_vllm_benchmark(
    model_path: str,
    max_new_tokens: int = 128,
    num_prompts: int = 8,
    gpu_mem_util: float = 0.85,
    max_model_len: int = 2048,
    warmup: int = 1,
) -> BenchResult:
    """End-to-end benchmark: load with vLLM, warm up, then time a prompt batch."""
    try:
        from vllm import SamplingParams
    except ImportError as e:
        pytest.skip(f"vllm is not installed: {e}")

    from test.helpers import get_model_path

    model_path = get_model_path(model_path) if "/" in model_path else model_path

    llm = _build_vllm_engine(model_path, max_model_len=max_model_len, gpu_mem_util=gpu_mem_util)
    try:
        tokenizer = llm.get_tokenizer()

        prompts = make_bench_prompts(tokenizer, num_prompts=num_prompts, target_input_tokens=64)
        sampling = SamplingParams(
            temperature=0.0,  # greedy ⇒ deterministic output length
            top_p=1.0,
            max_tokens=max_new_tokens,
        )

        # --- warm-up ---
        for _ in range(max(0, warmup)):
            llm.generate(prompts[:1], sampling)

        # --- timed run ---
        # Some vLLM versions populate metrics.* after generate; we use both
        # wall-clock time and vLLM's own counters for a robust measurement.
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        total_time = time.perf_counter() - t0

        # Aggregate output token counts.
        n_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        gen_tokens_per_s = n_out_tokens / max(total_time, 1e-6)

        # Try to pull vLLM's own decode-time stats for a second opinion.
        gen_tokens_per_s_vllm: float | None = None
        ttft_s: float | None = None
        try:
            metrics = llm.aggregate_metrics()  # vllm >= 0.6
            stats = getattr(metrics, "stats", None) or {}
            # vLLM reports ``prompt_tokens`` and ``generation_tokens``.
            gen_tok = float(stats.get("generation_tokens", 0.0))
            gen_time = float(stats.get("gen_time", 0.0))  # seconds, decode only
            prompt_tok = float(stats.get("prompt_tokens", 0.0))
            prompt_time = float(stats.get("prompt_time", 0.0))
            if gen_time > 0 and gen_tok > 0:
                gen_tokens_per_s_vllm = gen_tok / gen_time
            if prompt_time > 0 and prompt_tok > 0:
                ttft_s = prompt_time / max(1, len(prompts))
        except Exception:
            pass

        return BenchResult(
            engine="vllm",
            model=os.path.basename(model_path.rstrip("/")),
            fmt=os.environ.get("_AR_E2E_FMT", "auto_round"),
            bits=int(os.environ.get("_AR_E2E_BITS", "4")),
            group_size=int(os.environ.get("_AR_E2E_GS", "128")),
            num_prompts=num_prompts,
            max_new_tokens=max_new_tokens,
            total_time_s=total_time,
            output_tokens_per_s=n_out_tokens / max(total_time, 1e-6),
            gen_tokens_per_s=gen_tokens_per_s_vllm,
            ttft_s=ttft_s,
            sample_output=outputs[0].outputs[0].text,
        )
    finally:
        # vLLM 0.6+ supports .shutdown(); older versions fall back to del+gc.
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        del llm
        free_cuda()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVllmThroughput:
    """Quantize-with-autoround + serve-with-vllm end-to-end suite."""

    @pytest.mark.parametrize(
        "model_case",
        [
            # (hf_id, bits, group_size, sym, fmt, min_gpu_gib)
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
        """Quantize → save → load with vLLM → measure throughput."""
        save_dir = str(tmp_path / f"saved_{model_case.fmt}_w{model_case.bits}")

        # Quantize with the python API (mirrors `auto-round --format ...`).
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

        # Serve + benchmark.
        result = _run_vllm_benchmark(saved, max_new_tokens=64, num_prompts=4, gpu_mem_util=0.8)
        _record(result)

        # Sanity: the model produced non-empty output and didn't blow up.
        assert result.sample_output.strip(), "vLLM produced empty output"
        assert "!!!" not in result.sample_output, "vLLM produced garbage output"
        assert result.output_tokens_per_s > 0
        # Decoding throughput is bounded below by 1 tok/s on any modern GPU.
        # The number is intentionally loose – this is a regression check, not
        # a perf gate.
        assert result.output_tokens_per_s >= 1.0, (
            f"vLLM throughput suspiciously low: {result.output_tokens_per_s:.2f} tok/s"
        )

        print(
            f"\n[vLLM] {model_case.hf_id} {model_case.fmt} w{model_case.bits} "
            f"-> {result.output_tokens_per_s:.1f} tok/s "
            f"(decode-only: {result.gen_tokens_per_s}, ttft: {result.ttft_s})"
        )


class TestVllmLarge:
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

        result = _run_vllm_benchmark(saved, max_new_tokens=64, num_prompts=4, gpu_mem_util=0.85)
        _record(result)

        assert result.sample_output.strip()
        assert "!!!" not in result.sample_output
        assert result.output_tokens_per_s >= 1.0

        print(
            f"\n[vLLM-large] {model_case.hf_id} {model_case.fmt} w{model_case.bits} "
            f"-> {result.output_tokens_per_s:.1f} tok/s"
        )


def test_vllm_offline_eval_backend(tmp_path, require_cuda):
    """Make sure ``--eval_backend vllm`` in the CLI still works.

    This complements the throughput tests above: rather than driving vLLM
    directly from Python, it spawns the full ``auto-round`` CLI exactly
    as an end user would.  It is also a useful canary for regressions in
    the CLI plumbing.
    """
    import sys

    from test.helpers import get_model_path

    model = get_model_path("Qwen/Qwen3-0.6B")
    output_dir = str(tmp_path / "cli_saved")
    cmd = (
        f"{sys.executable} -m auto_round --model {model} --scheme W4A16 --iters 0 "
        f"--disable_opt_rtn --tasks lambada_openai --eval_backend vllm --limit 4 "
        f"--eval_bs 4 --output_dir {output_dir} "
        f"--vllm_args tensor_parallel_size=1,gpu_memory_utilization=0.5,max_model_len=1024"
    )
    env = os.environ.copy()
    env["VLLM_SKIP_WARMUP"] = "true"
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    rc = os.system(cmd)
    assert rc == 0, f"`auto-round --eval_backend vllm` failed (rc={rc})"
