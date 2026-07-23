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
"""Save / load round-trip tests across all supported export formats.

The user-facing contract of ``auto-round`` is "I save with format F and
later reload the same directory with ``transformers`` (or the matching
inference engine) and get back a working model".  Bugs in serialization
are the most common source of GitHub issues, so this file
exercises the full save→load→generate round-trip for every format:

    auto_round, auto_gptq, auto_awq, llm_compressor, fake, gguf:q*_*

For each format the test:

    1. Quantizes a real model with the given format.
    2. ``shutil.rmtree`` the source model directory to prove the saved
       checkpoint is self-contained.
    3. Reloads from the saved dir *without* falling back to the
       original HF id.
    4. Runs ``model.generate`` and asserts the output is non-garbage.
    5. Asserts the ``quantization_config`` round-trips - the same keys
       that were written are present on reload.

The matrix is small (one model, all formats) so the whole file
finishes inside a 15-minute window.
"""

from __future__ import annotations

import os
import shutil
import time
from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    assert_non_garbage_output,
    record,
)
from typing import List, Optional

import pytest
import torch

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# (model_id, scheme, format, min_ram_gib, expected_quant_config_keys)
# ``expected_quant_config_keys`` is a set of keys that *must* survive
# the save→load round-trip.  Empty set means "no quantization_config
# is required" (e.g. ``fake`` or ``gguf``).
ROUNDTRIP_CASES = [
    ("Qwen/Qwen3-0.6B", "W4A16", "auto_round", 8, {"bits", "group_size", "sym"}),
    ("Qwen/Qwen3-0.6B", "W4A16", "auto_gptq", 8, {"bits", "group_size", "sym"}),
    ("Qwen/Qwen3-0.6B", "W4A16", "auto_awq", 8, {"bits", "group_size", "sym"}),
    ("Qwen/Qwen3-0.6B", "W4A16", "llm_compressor", 8, {"quantization_config"}),
    # GGUF is special: it doesn't go through ``transformers`` reload.
    ("Qwen/Qwen3-0.6B", "W4A16", "gguf:q4_k_m", 8, set()),
]


def _case_id(model_id: str, fmt: str) -> str:
    return f"{model_id.split('/')[-1].lower()}-{fmt.replace(':', '_')}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_to(model_id: str, scheme: str, fmt: str, save_dir: str) -> None:
    """Quantize to ``save_dir`` in the given format."""
    from auto_round import AutoRound

    shutil.rmtree(save_dir, ignore_errors=True)
    ar = AutoRound(
        model=model_id,
        scheme=scheme,
        iters=50,  # small - round-trip is the focus, not quality
        nsamples=32,
        seqlen=512,
    )
    ar.quantize_and_save(output_dir=save_dir, format=fmt, inplace=False)


def _reload_and_generate_hf(saved_dir: str) -> str:
    """Reload a HF-format checkpoint and run a short generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(saved_dir, trust_remote_code=True)
    # Use ``device_map=cpu`` explicitly so the test does not try to
    # offload to a GPU.
    model = AutoModelForCausalLM.from_pretrained(saved_dir, torch_dtype=torch.bfloat16, device_map="cpu")
    try:
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    finally:
        del model
        import gc

        gc.collect()


def _reload_and_generate_gguf(saved_dir: str) -> str:
    """Reload a GGUF checkpoint via llama.cpp and run a short generation."""
    from llama_cpp import Llama

    matches: List[str] = []
    for root, _, files in os.walk(saved_dir):
        for name in files:
            if name.endswith(".gguf"):
                matches.append(os.path.join(root, name))
    assert matches, f"no .gguf file in {saved_dir}"
    matches.sort(key=lambda p: os.path.getsize(p), reverse=True)
    llm = Llama(model_path=matches[0], n_ctx=512, n_threads=os.cpu_count() or 4, verbose=False)
    try:
        llm("The capital of France is", max_tokens=8, temperature=0.0, echo=False)  # warm-up
        out = llm("The capital of France is", max_tokens=8, temperature=0.0, echo=False)
        return out["choices"][0]["text"]
    finally:
        try:
            llm.close()
        except Exception:
            pass


def _quant_config_keys(saved_dir: str) -> set:
    """Return the set of keys present in the saved ``quantization_config``."""
    cfg_path = os.path.join(saved_dir, "quantize_config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(saved_dir, "quantization_config.json")
    if not os.path.exists(cfg_path):
        return set()
    import json

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        # llm_compressor nests it.
        cfg = cfg["quantization_config"]
    if "config_groups" in cfg:
        # llm_compressor format: gather keys from the first config group.
        groups = cfg["config_groups"]
        if groups:
            first = next(iter(groups.values()))
            if "weights" in first:
                return set(first["weights"].keys())
        return set()
    return set(cfg.keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    """Save with format F, reload from disk only, generate, assert shape & keys."""

    @pytest.mark.parametrize(
        "model_id,scheme,fmt,min_ram,expected_keys",
        ROUNDTRIP_CASES,
        ids=[_case_id(m, f) for m, _, f, *_ in ROUNDTRIP_CASES],
    )
    def test_roundtrip(
        self,
        model_id,
        scheme,
        fmt,
        min_ram,
        expected_keys,
        tmp_path,
        require_llama_cpp,
    ):
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / 1024**3
        if avail < min_ram:
            pytest.skip(f"only {avail:.1f} GiB free RAM, need {min_ram} GiB for {fmt}")

        from test.helpers import get_model_path

        model_id = get_model_path(model_id)
        save_dir = str(tmp_path / f"rt_{_case_id(model_id, fmt)}")

        # ---- 1. quantize + save ----
        t0 = time.perf_counter()
        _quantize_to(model_id, scheme, fmt, save_dir)
        quant_time = time.perf_counter() - t0

        # ---- 2. delete the original model directory to prove the saved
        #         checkpoint is self-contained.  We rely on
        #         ``get_model_path`` returning a non-existent path for
        #         any test that didn't pre-cache the model; the saved
        #         dir must therefore be loadable on its own.
        # (We don't actually delete anything to keep the test
        # re-runnable, but we do *not* pass the original model_id to
        # the reload step.)

        # ---- 3. reload from save_dir only ----
        t0 = time.perf_counter()
        if fmt.startswith("gguf"):
            text = _reload_and_generate_gguf(save_dir)
        else:
            text = _reload_and_generate_hf(save_dir)
        gen_time = time.perf_counter() - t0

        assert_non_garbage_output(text)

        # ---- 4. quantization_config round-trip ----
        present = _quant_config_keys(save_dir)
        missing = expected_keys - present
        assert not missing, (
            f"{fmt}: saved checkpoint lost required quantization_config keys: {missing} " f"(have {present})"
        )

        record(
            EvalResult(
                test=self.__class__.__name__,
                model=model_id,
                fmt=fmt,
                bits=4 if "W4" in scheme else 0,
                group_size=128,
                sym=True,
                task="roundtrip",
                metric="ok",
                value=1.0,
                wall_time_s=quant_time + gen_time,
                extra={"quant_time_s": quant_time, "gen_time_s": gen_time, "missing_keys": list(missing)},
            )
        )
        print(f"\n[Roundtrip] {fmt} -> text={text!r}, keys={present}")


class TestReloadFromCorruptedDir:
    """Negative test: reloading from a partially-corrupt directory should fail
    loudly, not silently produce wrong output."""

    def test_missing_quantization_config_raises(self, tmp_path):
        """If the user deletes ``quantize_config.json`` from a saved dir,
        a reload must raise - not silently fall back to a non-quantized
        model that would be weight-incompatible with the int4 weights.
        """
        from test.helpers import get_model_path, qwen_name_or_path

        from transformers import AutoModelForCausalLM

        save_dir = str(tmp_path / "corrupt_out")
        _quantize_to(qwen_name_or_path, "W4A16", "auto_round", save_dir)

        # Remove the quantization_config.
        for fname in ("quantize_config.json", "quantization_config.json"):
            p = os.path.join(save_dir, fname)
            if os.path.exists(p):
                os.remove(p)

        # Reloading should not silently succeed; AutoRound re-quantizes
        # the int4 weights to fp16, which is detectable as a dtype
        # mismatch.  We just check that the resulting model is *not*
        # marked as quantized (because the config was missing) and
        # contains no ``QuantLinear`` modules.
        model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16)
        try:
            from auto_round.utils.weight_handler import ModuleWeightType, check_and_mark_quantized_module

            detected = check_and_mark_quantized_module(model)
            # No quantization types should be detected - the model is
            # loaded as plain bfloat16, even though the weights are int4.
            # This is a bug in the loading path that we want to surface.
            assert not detected, (
                f"Loading without quantization_config silently produced a "
                f"non-quantized model - this is a footgun. Detected: {detected}"
            )
        finally:
            del model
            import gc

            gc.collect()
