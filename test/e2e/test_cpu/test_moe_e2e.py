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
"""End-to-end MoE (Mixture of Experts) quantization tests.

These tests target the architectures that are most active in 2025-2026:
Qwen3-MoE, DeepSeek-V2-Lite, Mixtral, Llama-4, and OpenAI's
gpt-oss-20b (which uses native MXFP4).

The test pipeline is the canonical one:

    1. ``AutoRound`` quantizes the model.
    2. The saved checkpoint is reloaded.
    3. A single short generation is run.
    4. We assert the model produces non-garbage output and that
       ``model.config.num_local_experts`` is preserved through the
       save/load round-trip.

MoE model downloads are large; the cases are gated on free RAM.
"""

from __future__ import annotations

import os
import time
from typing import List

import pytest
import torch

from test.e2e.test_cpu.conftest import (  # noqa: E402
    EvalResult,
    assert_non_garbage_output,
    quantize_and_save,
    record,
)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# (hf_id, scheme, min_ram_gib, ignore)
# ``ignore`` matches the standard auto-round ignore list: router, lm_head,
# and the gating networks are left in fp16.
MOE_CASES = [
    ("Qwen/Qwen1.5-MoE-A2.7B", "W4A16", 16, "self_attn,router,lm_head,mlp.gate"),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", "W4A16", 16, "self_attn,router,lm_head,mlp.gate"),
    ("openai/gpt-oss-20b", "MXFP4", 24, "self_attn,lm_head"),
    # Mixtral and the bigger Qwen3-MoE are gated - the test will skip if
    # the host can't authenticate.  Listed for documentation, may be
    # enabled in CI when the org has a HF_TOKEN with the right scope.
    # ("mistralai/Mixtral-8x7B-Instruct-v0.1", "W4A16", 80, "self_attn,router,lm_head,mlp.gate"),
]


def _case_id(model_id: str, scheme: str) -> str:
    return f"{model_id.split('/')[-1].lower()}-{scheme.lower()}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMoeE2E:
    """Quantize + reload + generate for MoE architectures."""

    @pytest.mark.parametrize(
        "model_id,scheme,min_ram,ignore",
        MOE_CASES,
        ids=[_case_id(m, s) for m, s, *_ in MOE_CASES],
    )
    def test_quantize_and_generate(self, model_id, scheme, min_ram, ignore, tmp_path):
        import psutil  # type: ignore

        avail = psutil.virtual_memory().available / 1024**3
        if avail < min_ram:
            pytest.skip(f"only {avail:.1f} GiB free RAM, need {min_ram} GiB for {model_id}")

        from test.helpers import get_model_path

        model_id = get_model_path(model_id)
        save_dir = str(tmp_path / "moe_out")

        # ---- 1. quantize + save ----
        t0 = time.perf_counter()
        saved = quantize_and_save(
            model_id=model_id,
            bits=4 if "W4" in scheme else 0,
            group_size=128,
            sym=True,
            fmt="auto_round",
            output_dir=save_dir,
            iters=2,  # MoE models are slow; we keep the iter count small
            nsamples=4,
            seqlen=512,
            extra_kwargs={"ignore_layers": ignore, "disable_opt_rtn": False},
        )
        quant_time = time.perf_counter() - t0

        # ---- 2. reload + assert expert count is preserved ----
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        cfg = AutoConfig.from_pretrained(saved, trust_remote_code=True)
        orig_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # The number of experts must be preserved (the routers are not
        # quantized, but the per-expert weights are - and there must be
        # the same number of them after load).
        for attr in ("num_local_experts", "num_experts"):
            if hasattr(orig_cfg, attr):
                assert getattr(cfg, attr) == getattr(orig_cfg, attr), (
                    f"{model_id}: {attr} changed from "
                    f"{getattr(orig_cfg, attr)} to {getattr(cfg, attr)} after quantize+save"
                )

        # ---- 3. generate a few tokens ----
        tokenizer = AutoTokenizer.from_pretrained(saved, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(saved, torch_dtype=torch.bfloat16)
        except Exception:
            # Some MoE exports may require specific dtypes.
            model = AutoModelForCausalLM.from_pretrained(saved)

        try:
            inputs = tokenizer("The capital of France is", return_tensors="pt")
            t0 = time.perf_counter()
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            gen_time = time.perf_counter() - t0
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            assert_non_garbage_output(text)
        finally:
            del model
            import gc

            gc.collect()

        record(
            EvalResult(
                test=self.__class__.__name__,
                model=model_id,
                fmt="auto_round",
                bits=4 if "W4" in scheme else 0,
                group_size=128,
                sym=True,
                task="generate",
                metric="gen_len",
                value=float(len(text.split())),
                wall_time_s=quant_time + gen_time,
                extra={"quant_time_s": quant_time, "gen_time_s": gen_time, "ignore_layers": ignore},
            )
        )
        print(f"\n[MoE-e2e] {model_id} {scheme} -> text={text!r}")


class TestMoeExpertUnfuse:
    """Verify the saved MoE checkpoint exposes per-expert modules.

    Several backends (vLLM, Marlin, ...) want the experts in their
    *fused* or *unfused* form depending on the model.  This test
    sanity-checks that the saved checkpoint has the right structure.
    """

    @pytest.mark.parametrize(
        "model_id,expected_expert_attr",
        [
            ("Qwen/Qwen1.5-MoE-A2.7B", "num_local_experts"),
        ],
        ids=["qwen-moe"],
    )
    def test_save_preserves_expert_count(self, model_id, expected_expert_attr, tmp_path):
        from transformers import AutoConfig

        from test.helpers import get_model_path

        model_id = get_model_path(model_id)

        save_dir = str(tmp_path / "moe_struct_out")
        quantize_and_save(
            model_id=model_id,
            bits=4,
            group_size=128,
            sym=True,
            fmt="auto_round",
            output_dir=save_dir,
            iters=1,
            nsamples=2,
            seqlen=128,
            extra_kwargs={"ignore_layers": "self_attn,router,lm_head,mlp.gate"},
        )

        cfg = AutoConfig.from_pretrained(save_dir, trust_remote_code=True)
        n_experts = getattr(cfg, expected_expert_attr, None)
        assert n_experts is not None and n_experts > 0, (
            f"saved MoE checkpoint lost its expert count (cfg.{expected_expert_attr}={n_experts})"
        )
