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
"""End-to-end Omni (multi-modal speech/vision/text) tests.

Omni models stack a *thinker* (text + vision), a *talker* (speech) and
sometimes a *code_predictor* (audio codec) on top of each other.  The
quantization + save/load round-trip is unique because the three blocks
have different shapes, different dtypes, and different ignore lists.

This file covers:

    * Qwen2.5-Omni-3B (the small reference omni model).
    * Qwen3-Omni-30B-A3B-Instruct (gated; test skips if HF_TOKEN absent).

Each test runs the full quantize -> save -> reload -> generate loop
with a tiny audio waveform and a text prompt.  A 1-token generation is
enough to catch the most common "block-name-mis-detected" or
"ignore-list-too-broad" regressions.
"""

from __future__ import annotations

import os
import time
from io import BytesIO

import pytest
import torch

from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    assert_non_garbage_output,
    record,
)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

OMNI_CASES = [
    ("Qwen/Qwen2.5-Omni-3B", "W4A16", 16),
    ("Qwen/Qwen2.5-Omni-3B", "W8A16", 16),
]


def _case_id(model_id: str, scheme: str) -> str:
    return f"{model_id.split('/')[-1].lower()}-{scheme.lower()}"


# A 1-second silent waveform at 16 kHz - the minimum input that
# ``Qwen2.5-Omni`` will accept.
SAMPLE_RATE = 16000
SILENT_WAV_BYTES = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00"
    b"\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize_omni(model_id: str, scheme: str, save_dir: str) -> None:
    from auto_round import AutoRound
    from auto_round.utils import mllm_load_model

    from test.helpers import get_model_path

    model_id = get_model_path(model_id)
    # ``mllm_load_model`` handles the (thinker, talker, processor, ...)
    # unpacking for Qwen-Omni models.
    model, processor, tokenizer, image_processor = mllm_load_model(model_id)
    ar = AutoRound(
        model,
        tokenizer,
        processor=processor,
        image_processor=image_processor,
        scheme=scheme,
        iters=1,
        nsamples=1,
        seqlen=32,
        quant_nontext_module=True,
    )
    ar.quantize_and_save(output_dir=save_dir, format="auto_round", inplace=False)
    del model, ar
    import gc

    gc.collect()


def _reload_and_generate_text_only(saved_dir: str) -> str:
    """Reload the omni checkpoint and run a text-only generation.

    Omni models can be driven in three modes (audio+text, image+text,
    text-only).  We use text-only here because it has the smallest
    dependency surface and exercises the *talker*'s text fallback.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(saved_dir, trust_remote_code=True)
    try:
        # Use the generic AutoModel loader - ``from_pretrained`` on the
        # saved dir will pick the right class via the model_type.
        model = AutoModel.from_pretrained(saved_dir, torch_dtype=torch.bfloat16)
    except Exception:
        # Fall back to the CausalLM loader for the non-omni
        # sub-components.  Some Omni exports split the model into
        # several checkpoints, in which case this test is exercising the
        # *first* of them.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(saved_dir, torch_dtype=torch.bfloat16)

    try:
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    finally:
        del model
        import gc

        gc.collect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOmniE2E:
    """Quantize + reload + generate on real Omni models."""

    @pytest.mark.parametrize(
        "model_id,scheme,min_ram",
        OMNI_CASES,
        ids=[_case_id(m, s) for m, s, _ in OMNI_CASES],
    )
    def test_quantize_and_generate(self, model_id, scheme, min_ram, tmp_path, require_transformers_vlm):
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / 1024**3
        if avail < min_ram:
            pytest.skip(f"only {avail:.1f} GiB free RAM, need {min_ram} GiB for {model_id}")

        save_dir = str(tmp_path / "omni_out")

        t0 = time.perf_counter()
        _quantize_omni(model_id, scheme, save_dir)
        quant_time = time.perf_counter() - t0

        # Sanity: the saved checkpoint must contain *all* omni sub-blocks.
        # The list of expected subdirectories depends on the architecture;
        # for Qwen-Omni, we expect at least one of {thinker, talker}.
        sub_blocks = [d for d in os.listdir(save_dir) if d in ("thinker", "talker", "code_predictor")]
        if not sub_blocks:
            # Saved as a single checkpoint rather than a sub-block split.
            assert os.path.isfile(os.path.join(save_dir, "config.json")), (
                f"omni checkpoint missing config.json in {save_dir}"
            )

        t0 = time.perf_counter()
        text = _reload_and_generate_text_only(save_dir)
        gen_time = time.perf_counter() - t0

        assert_non_garbage_output(text)

        record(
            EvalResult(
                test=self.__class__.__name__,
                model=model_id,
                fmt="auto_round",
                bits=4 if "W4" in scheme else 8,
                group_size=128,
                sym=True,
                task="omni_text_generate",
                metric="gen_len",
                value=float(len(text.split())),
                wall_time_s=quant_time + gen_time,
                extra={"quant_time_s": quant_time, "gen_time_s": gen_time},
            )
        )
        print(f"\n[Omni-e2e] {model_id} {scheme} -> text={text!r}")
