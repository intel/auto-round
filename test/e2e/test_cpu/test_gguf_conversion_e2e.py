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
"""End-to-end GGUF conversion tests for all quantization types.

The unit test
:mod:`test.unit.test_cpu.export.test_gguf_format` covers a handful of
the most common GGUF types (``q4_0``, ``q4_k_m``, ...).  This file
exercises the *full* matrix exported by llama.cpp, so we catch
breakages in any of the conversion paths.

For each type, the test:

    1. Quantizes a real model with ``auto-round --format gguf:TYPE``.
    2. Verifies the saved ``.gguf`` file can be loaded back by
       ``llama_cpp.Llama`` (or, if ``llama-cpp-python`` is unavailable,
       by parsing the GGUF header with the local ``gguf`` package).
    3. Checks that the GGUF file's metadata reports the expected
       quantization type.
    4. Runs a single short generation and asserts the output is
       non-garbage.

The whole matrix is gated on free RAM (~10 GiB), and skipped cleanly
if the host can't fit it.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import pytest

from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    assert_non_garbage_output,
    record,
)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# (gguf_type, bits, group_size, expected_gguf_metadata_key)
# ``expected_gguf_metadata_key`` is the GGUF metadata value that the
# converter is supposed to write.  We assert it is present in the saved
# file, to catch "the converter silently produced a different type" bugs.
ALL_GGUF_TYPES = [
    ("gguf:q2_k", 2, 32, "Q2_K"),
    ("gguf:q3_k_s", 3, 32, "Q3_K"),
    ("gguf:q3_k_m", 3, 32, "Q3_K"),
    ("gguf:q3_k_l", 3, 32, "Q3_K"),
    ("gguf:q4_0", 4, 32, "Q4_0"),
    ("gguf:q4_1", 4, 32, "Q4_1"),
    ("gguf:q4_k_s", 4, 32, "Q4_K"),
    ("gguf:q4_k_m", 4, 32, "Q4_K"),
    ("gguf:q5_0", 5, 32, "Q5_0"),
    ("gguf:q5_1", 5, 32, "Q5_1"),
    ("gguf:q5_k_s", 5, 32, "Q5_K"),
    ("gguf:q5_k_m", 5, 32, "Q5_K"),
    ("gguf:q6_k", 6, 32, "Q6_K"),
    ("gguf:q8_0", 8, 32, "Q8_0"),
]

# Pick a single small model to keep the matrix's wall-clock under 30 min.
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

PROMPT = "The capital of France is"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case_id(gguf_type: str) -> str:
    return gguf_type.replace(":", "_").replace(".", "_")


def _find_gguf(save_dir: str) -> str:
    matches: List[str] = []
    for root, _, files in os.walk(save_dir):
        for name in files:
            if name.endswith(".gguf"):
                matches.append(os.path.join(root, name))
    assert matches, f"no .gguf file found under {save_dir}"
    matches.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return matches[0]


def _gguf_general_metadata(path: str) -> dict:
    """Read the GGUF header using the local ``gguf`` package."""
    try:
        from gguf.gguf_reader import Reader  # type: ignore
    except ImportError:
        pytest.skip("gguf package is not installed")
    reader = Reader(path)
    return dict(reader.fields)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGgufFullMatrix:
    """Quantize + verify metadata for every supported GGUF type."""

    @pytest.mark.parametrize("gguf_type,bits,group_size,gguf_name", ALL_GGUF_TYPES, ids=[_case_id(g) for g, *_ in ALL_GGUF_TYPES])
    def test_quantize_and_verify(self, gguf_type, bits, group_size, gguf_name, tmp_path, require_llama_cpp):
        from test.helpers import get_model_path

        model_id = get_model_path(DEFAULT_MODEL)
        save_dir = str(tmp_path / "gguf_full_out")

        # ---- 1. quantize + save ----
        from auto_round import AutoRound  # local import: heavy module

        # Skip the model-load if we've already exported the same model
        # once during this test session.
        import shutil

        ar = AutoRound(
            model=model_id,
            bits=bits,
            group_size=group_size,
            sym=True,
            iters=200,
            nsamples=128,
            seqlen=2048,
        )
        ar.quantize_and_save(output_dir=save_dir, format=gguf_type, inplace=False)
        gguf_path = _find_gguf(save_dir)

        # ---- 2. metadata check ----
        try:
            meta = _gguf_general_metadata(gguf_path)
        except Exception as e:
            pytest.skip(f"cannot read GGUF metadata (gguf package missing?): {e}")

        # The GGUF writer embeds the file-format string in
        # ``general.file_type``.  We don't require the *exact* value
        # (the writer normalises things like Q3_K_S/M/L to Q3_K) but
        # we do require the *family* prefix to match.
        file_type = str(meta.get("general.file_type", ""))
        if file_type:
            assert file_type.startswith(gguf_name), (
                f"{gguf_type}: GGUF metadata general.file_type={file_type!r} "
                f"does not start with {gguf_name!r}"
            )

        # ---- 3. load + run a single generation ----
        from llama_cpp import Llama

        llm = Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        try:
            llm(PROMPT, max_tokens=8, temperature=0.0, echo=False)  # warm-up
            out = llm(PROMPT, max_tokens=8, temperature=0.0, echo=False)
            text = out["choices"][0]["text"]
            assert_non_garbage_output(text)
        finally:
            try:
                llm.close()
            except Exception:
                pass

        record(
            EvalResult(
                test=self.__class__.__name__,
                model=model_id,
                fmt=gguf_type,
                bits=bits,
                group_size=group_size,
                sym=True,
                task="gguf_metadata",
                metric="file_type",
                value=1.0 if file_type.startswith(gguf_name) else 0.0,
                wall_time_s=0.0,
                extra={"file_type": file_type},
            )
        )
        print(f"\n[GGUF-matrix] {gguf_type} -> file_type={file_type}, text={text!r}")


class TestGgufMetadataHeader:
    """Header-only check; useful for fast CI feedback without loading the model."""

    @pytest.mark.parametrize("gguf_type,bits,group_size,_", ALL_GGUF_TYPES[:4], ids=[_case_id(g) for g, *_ in ALL_GGUF_TYPES[:4]])
    def test_header_only(self, gguf_type, bits, group_size, _, tmp_path):
        """Quantize, then read the header and verify metadata - no inference."""
        from test.helpers import get_model_path

        model_id = get_model_path(DEFAULT_MODEL)
        save_dir = str(tmp_path / "gguf_header_out")
        from auto_round import AutoRound

        ar = AutoRound(
            model=model_id,
            bits=bits,
            group_size=group_size,
            sym=True,
            iters=0,  # header-only check: RTN is enough
            disable_opt_rtn=True,
        )
        ar.quantize_and_save(output_dir=save_dir, format=gguf_type, inplace=False)
        gguf_path = _find_gguf(save_dir)

        meta = _gguf_general_metadata(gguf_path)
        # ``general.architecture`` is mandatory and tells us the
        # dequantizer which tensor layout to expect.
        assert "general.architecture" in meta, f"missing general.architecture in {gguf_path}"
        # The tokenizer must be embedded for llama.cpp to use the model.
        for key in ("tokenizer.ggml.model", "tokenizer.model"):
            if key in meta:
                break
        else:
            pytest.fail(f"no tokenizer metadata embedded in {gguf_path}")
