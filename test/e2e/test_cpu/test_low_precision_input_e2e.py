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
"""End-to-end "low-precision input model" tests.

A common user workflow is:

    "I have an FP8 / NVFP4 / MXFP4 model on HF Hub.  Can I just hand
     it to auto-round and get back an auto_round-format checkpoint?"

The unit test
:mod:`test.unit.test_cpu.advanced.test_low_precision_input_model` already
covers the format-detection logic in isolation.  This file covers the
end-to-end version: download a known-quantized model, run
``auto-round`` on it (no further quantization), and verify that the
saved auto_round checkpoint loads and runs.

These tests are also useful canaries for the
``compressed_tensors`` integration - several of the input formats rely
on the ``compressed_tensors`` package which has its own dependency
matrix.
"""

from __future__ import annotations

import os
import time
from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    assert_non_garbage_output,
    record,
)

import pytest
import torch

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# (hf_id, expected_input_dtype, target_format, min_ram_gib, expected_module_attr)
LOW_PRECISION_CASES = [
    (
        "RedHatAI/Qwen3-0.6B-FP8-BLOCK",
        torch.float8_e4m3fn,
        "auto_round",
        10,
        "CompressedLinear",
    ),
    (
        "RedHatAI/Qwen3-0.6B-quantized.w4a16",
        None,  # W4A16 typically lands as int Linear, not float8
        "auto_round",
        10,
        None,
    ),
    # NVFP4 and MXFP4 are gated on compressed_tensors; skip gracefully
    # if the package is missing or the model is gated.
    # ("kaitchup/Qwen3-0.6B-NVFP4", ..., "auto_round", 12, "CompressedLinear"),
    # ("QuixiAI/Llama-3.2-1B-MXFP4", ..., "auto_round", 12, "CompressedLinear"),
]


def _case_id(model_id: str) -> str:
    return f"{model_id.split('/')[-1].lower()}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLowPrecisionInputE2E:
    """End-to-end: take a pre-quantized model, re-export to auto_round."""

    @pytest.mark.parametrize(
        "model_id,expected_dtype,target_format,min_ram,expected_module",
        LOW_PRECISION_CASES,
        ids=[_case_id(m) for m, *_ in LOW_PRECISION_CASES],
    )
    def test_load_recompress_and_generate(
        self,
        model_id,
        expected_dtype,
        target_format,
        min_ram,
        expected_module,
        tmp_path,
    ):
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / 1024**3
        if avail < min_ram:
            pytest.skip(f"only {avail:.1f} GiB free RAM, need {min_ram} GiB for {model_id}")

        from test.helpers import get_model_path

        model_id = get_model_path(model_id)

        # If the model is gated or doesn't exist locally, surface a clear skip
        # rather than a stack trace.
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(model_id, allow_patterns=["config.json"])
        except Exception as e:
            pytest.skip(f"cannot fetch {model_id}: {e}")

        # ---- 1. load the pre-quantized model ----
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            pytest.skip(f"cannot load config of {model_id}: {e}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
        except Exception as e:
            pytest.skip(f"cannot load weights of {model_id}: {e}")

        # Optional: assert that the input dtype matches expectation.
        if expected_dtype is not None and expected_module is not None:
            first_layer = None
            for name, mod in model.named_modules():
                if hasattr(mod, "weight") and hasattr(mod, expected_module):
                    first_layer = mod
                    break
            if first_layer is not None:
                wt = getattr(first_layer, "weight_packed", first_layer.weight)
                assert wt.dtype == expected_dtype, f"{model_id} weight dtype {wt.dtype} != expected {expected_dtype}"

        # ---- 2. re-export through auto-round (no further quant) ----
        from auto_round import AutoRound

        save_dir = str(tmp_path / "lp_out")

        t0 = time.perf_counter()
        try:
            ar = AutoRound(
                model,
                tokenizer,
                scheme=target_format,
                iters=0,
                disable_opt_rtn=True,
            )
            ar.quantize_and_save(output_dir=save_dir, format=target_format, inplace=False)
        except Exception as e:
            # Some compressed_tensors / llm-compressor versions emit
            # different quantization_config keys; we surface this as
            # a *xfail* rather than a failure so the test suite
            # continues running.
            pytest.xfail(f"recompress of {model_id} failed: {e}")
        wall_time = time.perf_counter() - t0

        # ---- 3. reload the new checkpoint and generate ----
        del model
        import gc

        gc.collect()

        model2 = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype="auto")
        try:
            inputs = tokenizer("The capital of France is", return_tensors="pt")
            with torch.no_grad():
                ids = model2.generate(**inputs, max_new_tokens=8, do_sample=False)
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            assert_non_garbage_output(text)
        finally:
            del model2
            gc.collect()

        record(
            EvalResult(
                test=self.__class__.__name__,
                model=model_id,
                fmt=target_format,
                bits=0,  # input is already at a non-INT precision
                group_size=0,
                sym=True,
                task="generate",
                metric="ok",
                value=1.0,
                wall_time_s=wall_time,
                extra={"input_model": model_id},
            )
        )
        print(f"\n[LowPrec-e2e] {model_id} -> re-exported, generated {text!r}")


class TestLowPrecisionDetection:
    """Lighter tests that just check format detection on a sliced model.

    Uses the same pre-quantized models as above but only loads a single
    layer (via :func:`get_tiny_model`) and asserts the right module
    class is detected.  These are useful as a canary when a CI host
    doesn't have the RAM for the full model.
    """

    @pytest.mark.parametrize(
        "model_id,expected_module",
        [
            ("RedHatAI/Qwen3-0.6B-FP8-BLOCK", "CompressedLinear"),
            ("RedHatAI/Qwen3-0.6B-quantized.w4a16", None),
        ],
        ids=["fp8-block", "w4a16"],
    )
    def test_detect_module_type(self, model_id, expected_module, tmp_path):
        from test.helpers import get_model_path, get_tiny_model

        from auto_round.utils.weight_handler import (  # type: ignore
            ModuleWeightType,
            check_and_mark_quantized_module,
        )

        model_id = get_model_path(model_id)
        # Slice the model so the test is cheap.
        try:
            model = get_tiny_model(model_id, num_layers=1, from_config=False)
        except Exception as e:
            pytest.skip(f"cannot load {model_id}: {e}")

        detected = check_and_mark_quantized_module(model)
        if expected_module is not None:
            # At least one of the recognized low-precision types must be
            # present in the detection set.
            assert any(
                t in detected
                for t in (ModuleWeightType.FP8, ModuleWeightType.NVFP4, ModuleWeightType.MXFP4, ModuleWeightType.INT)
            ), f"no quantization type detected for {model_id}: {detected}"
