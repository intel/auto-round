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
"""Shared fixtures and helpers for the e2e CPU tests.

These tests cover the full quantization + serialization + (local) inference
loop for real, "user-sized" models.  They do *not* require a GPU; in fact
they intentionally exercise the CPU path because that is what most CI
hosts and a large slice of the user base actually run.

The fixtures here are deliberately similar to (but independent of) the
ones in :mod:`test.e2e.test_cuda.conftest` because the helper APIs that
are convenient for vLLM/SGLang benchmarks (gpu memory probes, etc.)
are not the right fit for CPU-only scenarios.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import pytest

# Ensure the repo root is importable so `from test.helpers import ...` works.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end test (slow, real models, runs on weekly CI)",
    )


# ---------------------------------------------------------------------------
# Memory gate
# ---------------------------------------------------------------------------


def _host_mem_gib() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        # Fall back to a generous 64 GiB if psutil is missing.
        return 64.0


def _host_mem_avail_gib() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return 64.0


def pytest_addoption(parser):
    parser.addoption(
        "--e2e-cpu-mem-gib",
        action="store",
        default=None,
        type=int,
        help=(
            "Override the host-RAM gate (GiB) used by e2e CPU tests. "
            "By default tests skip when the case needs more RAM than the host has."
        ),
    )


# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------


@dataclass
class ModelCase:
    """A single (model, scheme, format) CPU e2e case.

    Attributes
    ----------
    hf_id:
        HuggingFace model id (or local path resolved by ``get_model_path``).
    bits:
        Weight bits.  Set to 16 for a bf16 baseline.
    group_size:
        Quantization group size; -1 means per-channel.
    sym:
        Symmetric vs asymmetric quantization.
    fmt:
        Export format - one of the strings accepted by
        ``AutoRound.quantize_and_save``: ``auto_round``, ``auto_gptq``,
        ``auto_awq``, ``llm_compressor``, ``gguf:q*_*``, ``fake``, ...
    min_ram_gib:
        Required free RAM in GiB; the case is skipped when the host has
        less than this.  Calibrated empirically for full-precision
        quantize (iters=200, nsamples=128) on an 8-core x86 CPU.
    skip_eval:
        Set True for cases that are pure smoke-tests (e.g. "did the
        checkpoint load"); these skip the ``--eval`` step.
    eval_tasks:
        lm-eval tasks to run after quantization.  Defaults to a small
        set of fast tasks that exercise different capabilities.
    eval_limit:
        Sample limit for the eval; small enough that the entire matrix
        finishes inside the weekly CI window.
    """

    hf_id: str
    bits: int
    group_size: int
    sym: bool
    fmt: str
    min_ram_gib: int = 16
    skip_eval: bool = False
    eval_tasks: str = "lambada_openai,piqa"
    eval_limit: int = 100
    extra_quant_kwargs: dict = field(default_factory=dict)


# Default CPU matrix: real, small (≤1.5B) LLMs that finish quantize+eval
# in a few minutes on a 32 GiB host.  These are the models most likely
# to actually run on CPU in production.
DEFAULT_MODEL_CASES: List[ModelCase] = [
    # Qwen family - small + well supported across all formats.
    ModelCase("Qwen/Qwen3-0.6B", 4, 128, True, "auto_round", min_ram_gib=8, eval_limit=80),
    ModelCase("Qwen/Qwen3-0.6B", 4, 128, True, "auto_gptq", min_ram_gib=8, eval_limit=80),
    ModelCase("Qwen/Qwen3-0.6B", 4, 128, True, "auto_awq", min_ram_gib=8, eval_limit=80),
    ModelCase("Qwen/Qwen3-0.6B", 2, 128, True, "auto_round", min_ram_gib=8, eval_limit=80),
    ModelCase("Qwen/Qwen3-0.6B", 8, 128, True, "auto_round", min_ram_gib=8, eval_limit=80),
    # GGUF - special path, exercises the same export.
    ModelCase("Qwen/Qwen3-0.6B", 4, 32, True, "gguf:q4_k_m", min_ram_gib=8, eval_limit=80),
    ModelCase("Qwen/Qwen3-0.6B", 8, 32, True, "gguf:q8_0", min_ram_gib=8, eval_limit=80),
    # Llama 3.2 1B - small, gated but commonly available locally.
    ModelCase("meta-llama/Llama-3.2-1B", 4, 128, True, "auto_round", min_ram_gib=10, eval_limit=80),
    ModelCase("meta-llama/Llama-3.2-1B", 4, 128, True, "auto_gptq", min_ram_gib=10, eval_limit=80),
    # Phi family.
    ModelCase("microsoft/Phi-3.5-mini-instruct", 4, 128, True, "auto_round", min_ram_gib=18, eval_limit=50),
    # gemma 2 2b - frequently used for accuracy benchmarks.
    ModelCase("google/gemma-2-2b", 4, 128, True, "auto_round", min_ram_gib=18, eval_limit=50),
    # InternLM 1.8B - extra coverage for non-Qwen/Llama architectures.
    ModelCase("internlm/internlm2-chat-1_8b", 4, 128, True, "auto_round", min_ram_gib=12, eval_limit=50),
]

# Heavier cases - 1.5B-2B; need ~24 GiB free RAM and longer wall-clock.
LARGE_MODEL_CASES: List[ModelCase] = [
    ModelCase("Qwen/Qwen2.5-1.5B-Instruct", 4, 128, True, "auto_round", min_ram_gib=14, eval_limit=80),
    ModelCase("Qwen/Qwen2.5-1.5B-Instruct", 4, 128, True, "auto_gptq", min_ram_gib=14, eval_limit=80),
    ModelCase("Qwen/Qwen2.5-1.5B-Instruct", 4, 128, True, "auto_awq", min_ram_gib=14, eval_limit=80),
    ModelCase("Qwen/Qwen2.5-1.5B-Instruct", 4, 128, True, "gguf:q4_k_m", min_ram_gib=14, eval_limit=80),
    ModelCase("meta-llama/Llama-3.2-3B-Instruct", 4, 128, True, "auto_round", min_ram_gib=20, eval_limit=50),
]


def _resolve_mem_override(request) -> Optional[int]:
    return request.config.getoption("--e2e-cpu-mem-gib") if request else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_matrix(request) -> List[ModelCase]:
    preset = os.environ.get("E2E_CPU_PRESET", "default")
    if preset == "default":
        return DEFAULT_MODEL_CASES
    if preset == "large":
        return LARGE_MODEL_CASES
    return DEFAULT_MODEL_CASES + LARGE_MODEL_CASES


@pytest.fixture
def model_case(request) -> ModelCase:
    """Parametrize helper - tests use ``@pytest.mark.parametrize("model_case", [...])``."""
    return request.param


@pytest.fixture
def require_ram(model_case: ModelCase, request):
    override = _resolve_mem_override(request)
    if override is not None:
        if override < model_case.min_ram_gib:
            pytest.skip(f"--e2e-cpu-mem-gib={override} < required {model_case.min_ram_gib} GiB for {model_case.hf_id}")
        return
    avail = _host_mem_avail_gib()
    if avail < model_case.min_ram_gib:
        pytest.skip(f"Skipping {model_case.hf_id}: only {avail:.1f} GiB free, need {model_case.min_ram_gib} GiB")


@pytest.fixture
def require_lm_eval():
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        pytest.skip("lm-eval is not installed (`pip install 'lm-eval>=0.4.2'`) to run accuracy tests")


@pytest.fixture
def require_llama_cpp():
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        pytest.skip("llama-cpp-python is not installed (`pip install llama-cpp-python`) to run GGUF CPU tests")


@pytest.fixture
def require_diffusers():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        pytest.skip("diffusers is not installed (`pip install diffusers`) to run diffusion tests")


@pytest.fixture
def require_transformers_vlm():
    """Some VLMs need a recent transformers version; skip if too old."""
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version("4.45.0"):
        pytest.skip(f"transformers>={4.45}.0 required for VLM tests (have {transformers.__version__})")


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """One accuracy / sanity measurement, appended to a JSONL file."""

    test: str
    model: str
    fmt: str
    bits: int
    group_size: int
    sym: bool
    task: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    extra: dict = field(default_factory=dict)
    wall_time_s: float = 0.0


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_THIS_DIR, "..", "..", "output")
_OUTPUT_FILE = os.path.normpath(os.path.join(_OUTPUT_DIR, "cpu_e2e.jsonl"))


def record(result: EvalResult) -> None:
    """Append a result to ``test/output/cpu_e2e.jsonl`` for trend tracking."""
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(_OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")


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
    scheme: Optional[str] = None,
):
    """Quantize a model with the Python API and save it.

    This mirrors the CLI:

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
        scheme=scheme,
        **(extra_kwargs or {}),
    )
    _, saved_dir = ar.quantize_and_save(output_dir=output_dir, format=fmt, inplace=False)
    return saved_dir


def _try_getattr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def run_lm_eval(
    saved_dir: str,
    tasks: str = "lambada_openai,piqa",
    limit: int = 100,
    batch_size: str = "auto",
    model_type: str = "hf",
    extra_model_args: Optional[dict] = None,
):
    """Run ``lm-eval`` over a saved checkpoint.

    Returns the dict returned by ``lm_eval.simple_evaluate``.
    """
    from auto_round.eval.evaluation import simple_evaluate

    if model_type == "hf":
        model_args = f"pretrained={saved_dir}"
        if extra_model_args:
            model_args = model_args + "," + ",".join(f"{k}={v}" for k, v in extra_model_args.items())
    else:
        model_args = extra_model_args or {}

    return simple_evaluate(
        model=model_type,
        model_args=model_args,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
    )


def extract_metric(results: dict, task: str, metric: str = "acc,none") -> Optional[float]:
    """Pull a single metric out of the lm-eval results dict (may be missing)."""
    try:
        return float(results["results"][task][metric])
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def run_cli(argv: List[str], env: Optional[dict] = None, timeout: int = 60 * 60) -> int:
    """Spawn ``python -m auto_round <argv>`` and return the exit code.

    Used by the CLI e2e tests; intentionally goes through the actual
    entry-point to catch regressions in argparse / import-time wiring.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    cmd = [sys.executable, "-m", "auto_round", *argv]
    try:
        return subprocess.call(cmd, env=full_env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def assert_non_garbage_output(text: str) -> None:
    """Common regression guard: the generated text must be non-empty and
    must not contain the classic "all-token-quantization-collapse" marker.
    """
    assert text and text.strip(), "model produced empty output"
    assert "!!!" not in text, f"model produced garbage output: {text!r}"


# ---------------------------------------------------------------------------
# Final cleanup hook
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _gc_between_tests():
    """Keep RSS bounded across the long e2e run."""
    yield
    gc.collect()
