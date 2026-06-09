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
"""Shared fixtures and helpers for the e2e CUDA throughput tests.

These tests exercise the full pipeline:
    1. Quantize a real, medium-sized LLM with auto-round.
    2. Save the quantized checkpoint in a deployment-ready format
       (auto_round, auto_gptq, auto_awq, gguf, llm_compressor).
    3. Load the checkpoint with the target inference engine
       (vLLM or SGLang) on a CUDA GPU.
    4. Measure end-to-end throughput and latency.

They are intentionally slow (a single Qwen2.5-7B W4A16 quantize + vLLM
load + warmup + benchmark takes several minutes) and are designed to
run in a weekly scheduled CI job, not on every PR.
"""

import gc
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import torch

# Make sure the repo root is importable so `from test.helpers import ...`
# works when pytest is invoked from the repo root with a relative path.
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# vLLM and sglang both fork-spawn workers, so spawn is the safest default.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Skip the slow internal warmup step so the timed portion of the test is
# dominated by the user's workload, not by vLLM's calibration pass.
os.environ.setdefault("VLLM_SKIP_WARMUP", "true")


# ---------------------------------------------------------------------------
# pytest configuration: register the e2e marker
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end test (slow, runs real models on real GPU, scheduled weekly)",
    )


# ---------------------------------------------------------------------------
# Environment gates
# ---------------------------------------------------------------------------


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _is_sm12_with_old_cuda() -> bool:
    """SM 12.x (Blackwell) + CUDA < 12.9 breaks the gptq_marlin JIT kernel."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major < 12:
            return False
        cuda_ver = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
        return cuda_ver < (12, 9)
    except Exception:
        return False


def _gpu_free_gib() -> float:
    if not _has_cuda():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1024**3


# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------


@dataclass
class ModelCase:
    """A single (model, scheme, format) e2e case."""

    hf_id: str
    bits: int
    group_size: int
    sym: bool
    fmt: str
    # Lower bound on required free GPU memory in GiB; case is skipped if
    # the GPU is smaller than this.  Keeps the same test file usable on
    # A100-40G, A100-80G and H100.
    min_gpu_gib: int = 16
    # Extra args passed to AutoRound.quantize_and_save (e.g. "low_cpu_mem_usage").
    extra_quant_kwargs: dict = field(default_factory=dict)


# A pragmatic matrix that covers (a) the default W4A16 path that almost
# every user runs, (b) the W2A16 low-memory path, (c) the activation
# quant path and (d) the GPTQ/AWQ back-compat paths.  All models are
# small enough to fit on a single 24 GiB GPU at W4A16 with offloading.
DEFAULT_MODEL_CASES: List[ModelCase] = [
    ModelCase("Qwen/Qwen3-1.7B", 4, 128, True, "auto_round", min_gpu_gib=10),
    ModelCase("Qwen/Qwen3-1.7B", 4, 128, True, "auto_gptq", min_gpu_gib=10),
    ModelCase("Qwen/Qwen3-1.7B", 4, 128, True, "auto_awq", min_gpu_gib=10),
    ModelCase("Qwen/Qwen3-1.7B", 2, 128, True, "auto_round", min_gpu_gib=10),
    ModelCase("Qwen/Qwen3-1.7B", 8, 128, True, "auto_round", min_gpu_gib=10),
]

# A more demanding matrix for nightly runs (8 GiB / 7B-class).  These
# require ~16 GiB free at fp16 master + W4A16 weights.
LARGE_MODEL_CASES: List[ModelCase] = [
    ModelCase("Qwen/Qwen2.5-7B-Instruct", 4, 128, True, "auto_round", min_gpu_gib=18),
    ModelCase("Qwen/Qwen2.5-7B-Instruct", 4, 128, True, "auto_gptq", min_gpu_gib=18),
    ModelCase("meta-llama/Llama-3.2-3B-Instruct", 4, 128, True, "auto_round", min_gpu_gib=12),
]


def pytest_addoption(parser):
    parser.addoption(
        "--e2e-model-preset",
        action="store",
        default="default",
        choices=["default", "large", "all"],
        help="Model matrix for throughput tests. 'large'/'all' need >=24 GiB GPU.",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_matrix(request) -> List[ModelCase]:
    preset = request.config.getoption("--e2e-model-preset")
    if preset == "default":
        return DEFAULT_MODEL_CASES
    if preset == "large":
        return LARGE_MODEL_CASES
    return DEFAULT_MODEL_CASES + LARGE_MODEL_CASES


@pytest.fixture
def require_cuda():
    if not _has_cuda():
        pytest.skip("CUDA is not available on this host")
    if _is_sm12_with_old_cuda():
        pytest.skip(
            "SM 12.x (Blackwell) requires CUDA >= 12.9 for gptq_marlin JIT kernels "
            f"(installed: CUDA {torch.version.cuda})"
        )


@pytest.fixture
def require_gpu_memory(model_case: ModelCase):
    if not _has_cuda():
        pytest.skip("CUDA is not available on this host")
    free_gib = _gpu_free_gib()
    if free_gib < model_case.min_gpu_gib:
        pytest.skip(
            f"Skipping {model_case.hf_id} {model_case.fmt} w{model_case.bits}: "
            f"only {free_gib:.1f} GiB free, need {model_case.min_gpu_gib} GiB"
        )


@pytest.fixture
def model_case(request) -> ModelCase:
    return request.param


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def quantize_and_save(
    model_id: str,
    bits: int,
    group_size: int,
    sym: bool,
    fmt: str,
    output_dir: str,
    iters: int = 200,
    nsamples: int = 128,
    seqlen: int = 2048,
    extra_kwargs: Optional[dict] = None,
):
    """Run the full AutoRound pipeline and return the saved checkpoint dir.

    This is the canonical entry point used by both the vLLM and SGLang
    throughput tests.  The CLI equivalent is:

        auto-round --model {model_id} --bits {bits} --group_size {group_size} \\
                   --sym --format {fmt} --output_dir {output_dir} \\
                   --iters {iters} --nsamples {nsamples} --seqlen {seqlen}
    """
    from auto_round import AutoRound  # local import: heavy module

    shutil.rmtree(output_dir, ignore_errors=True)
    ar = AutoRound(
        model=model_id,
        bits=bits,
        group_size=group_size,
        sym=sym,
        iters=iters,
        nsamples=nsamples,
        seqlen=seqlen,
        **(extra_kwargs or {}),
    )
    _, saved_dir = ar.quantize_and_save(output_dir=output_dir, format=fmt, inplace=False)
    return saved_dir


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Outcome of a single inference benchmark run."""

    engine: str
    model: str
    fmt: str
    bits: int
    group_size: int
    num_prompts: int
    max_new_tokens: int
    # Wall-clock seconds from the first generate() call to the last token.
    total_time_s: float
    # End-to-end output tokens per second (prompt processing + decoding).
    output_tokens_per_s: float
    # Decoding-only tokens per second (excludes prompt eval), if measurable.
    gen_tokens_per_s: Optional[float]
    # Time-to-first-token seconds (mean over the batch, if available).
    ttft_s: Optional[float]
    # Generated text for the first prompt; useful for sanity checks.
    sample_output: str


def _standard_prompts() -> List[str]:
    """A fixed prompt list so different runs are comparable."""
    return [
        "The capital of France is",
        "Briefly explain the difference between quantization and pruning in ML:",
        "Write a short Python function that reverses a linked list:",
        "Summarize the plot of 'The Great Gatsby' in two sentences:",
        "What is the derivative of x^3 with respect to x?",
        "List three benefits of regular physical exercise:",
        "Translate 'Good morning, how are you?' into Japanese:",
        "Explain the difference between TCP and UDP in one paragraph:",
    ]


def make_bench_prompts(tokenizer, num_prompts: int, target_input_tokens: int = 64) -> List[str]:
    """Pad each base prompt with lorem-style text to ~target_input_tokens.

    The result is a list of prompts whose prompt-eval cost is similar
    across runs, which makes throughput numbers reproducible.
    """
    base = _standard_prompts()
    pad = (
        " Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    )
    out: List[str] = []
    i = 0
    while len(out) < num_prompts:
        prompt = base[i % len(base)] + pad * 4
        # Trim/pad to the target token count so all prompts cost the same.
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > target_input_tokens:
            ids = ids[:target_input_tokens]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
        out.append(prompt)
        i += 1
    return out


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
