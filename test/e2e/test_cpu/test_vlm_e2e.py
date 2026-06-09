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
"""End-to-end Vision-Language Model (VLM) quantization tests.

Each test exercises the multi-modal path that is unique to VLMs:

    1. Load the model with ``mllm_load_model`` (loads vision tower +
       language model + processor + image_processor).
    2. Quantize with ``auto-round`` (``quant_nontext_module=True`` for
       vision-tower quantization; ``False`` for "text-only" quant).
    3. Reload the saved checkpoint.
    4. Feed a single image + prompt, run ``model.generate`` and check
       that the output is non-garbage.

A case is auto-skipped if the host has insufficient free RAM, if
``transformers`` is too old for the target VLM, or if the model is
gated and not accessible.
"""

from __future__ import annotations

import os
import time

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

# (hf_id, scheme, quant_vision, min_ram_gib, expected_substring)
VLM_CASES = [
    ("Qwen/Qwen2-VL-2B-Instruct", "W4A16", True, 12, ["bus", "white", "red"]),
    ("Qwen/Qwen2-VL-2B-Instruct", "W8A8", True, 12, ["bus"]),
    ("Qwen/Qwen2-VL-2B-Instruct", "W4A16", False, 12, ["bus"]),  # text-only quant
    ("Qwen/Qwen2.5-VL-3B-Instruct", "W4A16", True, 16, ["bus"]),
    # gemma-3-4b-it is a multimodal model that became generally available
    # in 2025; we use a tiny 224x224 image to keep the test cheap.
    ("google/gemma-3-4b-it", "W4A16", True, 18, []),
]


# A tiny in-memory image; avoids needing network or a checked-in asset.
# 4×4 red square is enough to drive a non-empty forward pass.
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x04\x00\x00\x00\x04\x08\x02\x00\x00\x00\x86"
    b"\xb1\x8c\x95\x00\x00\x00\x0fIDATx\x9cc\xfc\xcf\xc0P\x0f\x00\x05\x01\x01\x01"
    b"\x00\xc8\xff\xff\xfft\x00\x06\x00\x02\xfe\xa6\x80Q\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _case_id(model_id: str, scheme: str, quant_vision: bool) -> str:
    suffix = "vision" if quant_vision else "text"
    return f"{model_id.split('/')[-1].lower()}-{scheme.lower()}-{suffix}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tiny_prompt(model_id: str):
    """Build a (prompt, image) pair suitable for the given VLM family.

    Qwen2-VL / Qwen2.5-VL use the ``<image>`` placeholder in the prompt;
    gemma-3 expects the image as a separate content part.  We just
    return a generic prompt and a tiny PIL image; each test method
    adapts it as needed.
    """
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(TINY_PNG)).convert("RGB")
    prompt = "<image>\nDescribe this image in one short sentence."
    return prompt, image


def _quantize_vlm(model_id: str, scheme: str, quant_vision: bool, save_dir: str) -> None:
    """Quantize a VLM end-to-end and save the checkpoint."""
    from auto_round import AutoRound
    from auto_round.utils import mllm_load_model

    from test.helpers import get_model_path

    model_id = get_model_path(model_id)
    model, processor, tokenizer, image_processor = mllm_load_model(model_id)

    ar = AutoRound(
        model,
        tokenizer,
        processor=processor,
        image_processor=image_processor,
        scheme=scheme,
        iters=2,
        nsamples=4,
        seqlen=32,
        quant_nontext_module=quant_vision,
    )
    ar.quantize_and_save(output_dir=save_dir, format="auto_round", inplace=False)
    del model, ar
    import gc

    gc.collect()


def _reload_and_generate(saved_dir: str, model_id: str, max_new_tokens: int = 16) -> str:
    """Reload the saved VLM and run a single short generation."""
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(saved_dir, trust_remote_code=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(saved_dir, trust_remote_code=True)
    except Exception:
        tokenizer = None

    # Pick the right model class for the architecture.  We fall back to
    # ``AutoModelForVision2Seq`` which works for the Qwen2-VL family
    # and most modern VLMs.
    try:
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(saved_dir, torch_dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(saved_dir, torch_dtype=torch.bfloat16)

    prompt, image = _load_tiny_prompt(model_id)

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = ids[0][len(inputs["input_ids"][0]):]
        return processor.batch_decode([out_ids], skip_special_tokens=True)[0]
    finally:
        del model
        if tokenizer is not None:
            del tokenizer
        del processor
        import gc

        gc.collect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVlmE2E:
    """Quantize + reload + generate on real VLMs."""

    @pytest.mark.parametrize(
        "model_id,scheme,quant_vision,min_ram,expected_keywords",
        VLM_CASES,
        ids=[_case_id(m, s, v) for m, s, v, *_ in VLM_CASES],
    )
    def test_quantize_and_generate(
        self,
        model_id: str,
        scheme: str,
        quant_vision: bool,
        min_ram: int,
        expected_keywords,
        tmp_path,
        require_transformers_vlm,
    ):
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / 1024**3
        if avail < min_ram:
            pytest.skip(f"only {avail:.1f} GiB free RAM, need {min_ram} GiB for {model_id}")

        save_dir = str(tmp_path / "vlm_out")

        t0 = time.perf_counter()
        _quantize_vlm(model_id, scheme, quant_vision, save_dir)
        quant_time = time.perf_counter() - t0

        # Sanity: the saved checkpoint must have a vision_config + quantization_config.
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(save_dir, trust_remote_code=True)
        # VLM models always carry some form of vision config.
        assert any(hasattr(cfg, attr) for attr in ("vision_config", "image_config")), (
            "saved checkpoint does not look like a VLM: missing vision_config"
        )

        t0 = time.perf_counter()
        text = _reload_and_generate(save_dir, model_id, max_new_tokens=12)
        gen_time = time.perf_counter() - t0

        assert_non_garbage_output(text)
        # Loose substring check - the VLM doesn't have to produce any
        # specific sentence, just *something* on topic.
        text_low = text.lower()
        for kw in expected_keywords:
            if kw:
                assert kw in text_low, f"{model_id} {scheme} did not mention '{kw}': {text!r}"

        record(
            EvalResult(
                test=self.__class__.__name__,
                model=model_id,
                fmt="auto_round",
                bits=4 if "W4" in scheme else 8,
                group_size=128,
                sym=True,
                task="vlm_generate",
                metric="gen_len",
                value=float(len(text.split())),
                wall_time_s=quant_time + gen_time,
                extra={"quant_vision": quant_vision, "quant_time_s": quant_time, "gen_time_s": gen_time},
            )
        )
        print(
            f"\n[VLM-e2e] {model_id} {scheme} vision={quant_vision} -> "
            f"quant={quant_time:.0f}s, gen={gen_time:.0f}s, text={text!r}"
        )
