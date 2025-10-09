# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest test/test_cuda/test_vllm.py`.
"""

import os
import shutil
import subprocess

import pytest
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

MODELS = [
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  ##auto_round:auto_gptq
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",  ##auto_round:auto_awq
]


# @pytest.mark.skipif(
#     not current_platform.is_cpu() and not current_platform.is_xpu() and not current_platform.is_cuda(),
#     reason="only supports CPU/XPU/CUDA backend.",
# )
# @pytest.mark.parametrize("model", MODELS)
# def test_auto_round(model):
#     # Sample prompts.
#     prompts = [
#         "The capital of France is",
#         "The future of AI is",
#     ]
#     # Create a sampling params object.
#     sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
#     # Create an LLM.
#     QUANTIZATION = "auto-round"
#     llm = LLM(model=model, quantization=QUANTIZATION, trust_remote_code=True, tensor_parallel_size=1)
#     # Generate texts from the prompts.
#     # The output is a list of RequestOutput objects
#     # that contain the prompt, generated text, and other information.
#     outputs = llm.generate(prompts, sampling_params)
#     # Print the outputs.
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         if "France" in prompt:
#             assert "Paris" in generated_text
#
#
# @pytest.mark.parametrize("model", MODELS)
# def test_vllm_lm_eval(model):
#     if shutil.which("auto-round") is None:
#         pytest.skip("auto-round CLI not available")
#
#     env = os.environ.copy()
#     env["VLLM_SKIP_WARMUP"] = "true"
#     env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
#
#     cmd = [
#         "auto-round",
#         "--model",
#         model,
#         "--eval",
#         "--tasks",
#         "lambada_openai",
#         "--eval_bs",
#         "8",
#         "--limit",
#         "10",
#         "--vllm",
#     ]
#
#     proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#     assert proc.returncode == 0, f"auto-round failed (rc={proc.returncode}):\n{proc.stdout}"
