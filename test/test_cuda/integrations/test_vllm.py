# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest test/test_cuda/test_vllm.py`.
"""

import os
import shutil

import pytest
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

from auto_round import AutoRound

from ...helpers import get_model_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = [
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  ##auto_round:auto_gptq
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",  ##auto_round:auto_awq
]


@pytest.mark.skipif(
    not current_platform.is_cpu() and not current_platform.is_xpu() and not current_platform.is_cuda(),
    reason="only supports CPU/XPU/CUDA backend.",
)
@pytest.mark.parametrize("model", MODELS)
def test_auto_round(model):
    # offline inference loading test
    prompts = [
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Create an LLM.
    QUANTIZATION = "auto-round"
    llm = LLM(
        model=model,
        quantization=QUANTIZATION,
        trust_remote_code=True,
        tensor_parallel_size=1,
        allow_deprecated_quantization=True,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if "France" in prompt:
            assert "Paris" in generated_text


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="only supports CUDA backend.",
)
def test_auto_round_awq_format_vllm():
    # quantization and inference test
    model_path = get_model_path("facebook/opt-125m")
    save_dir = "./saved"

    autoround = AutoRound(
        model=model_path,
        scheme="W4A16",
        iters=1,
        seqlen=2,
    )
    autoround.quantize_and_save(output_dir=save_dir, format="auto_round:auto_awq")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm = LLM(
        model=save_dir,
        # quantization="awq",
        trust_remote_code=True,
        tensor_parallel_size=1,
        # allow_deprecated_quantization=True,
        # dtype="auto",
    )
    outputs = llm.generate(["The capital of France is"], sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(generated_text)
    assert len(generated_text.strip()) > 0, "vLLM AWQ inference produced empty output"
    shutil.rmtree(save_dir, ignore_errors=True)
