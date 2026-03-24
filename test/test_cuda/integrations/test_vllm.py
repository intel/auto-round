# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest test/test_cuda/test_vllm.py`.
"""

import os
import shutil
import sys

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


def test_mixed_autoround_format_vllm(tiny_opt_model_path, dataloader, tmp_path):
    layer_config = {
        "self_attn": {"bits": 8},
        "lm_head": {"bits": 16},
    }
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        iters=0,
        disable_opt_rtn=True,
        layer_config=layer_config,
    )
    autoround.quantize()
    quantized_model_path = str(tmp_path / "saved")
    autoround.save_quantized(output_dir=quantized_model_path, format="auto_round")

    # verify loading.
    llm = LLM(
        model=quantized_model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        allow_deprecated_quantization=True,
    )


def test_mixed_llmcompressor_format_vllm(tiny_opt_model_path, dataloader, tmp_path):
    layer_config = {
        "self_attn": {"bits": 16, "act_bits": 16},
        "lm_head": {"bits": 16, "act_bits": 16},
        "fc1": {"bits": 16, "act_bits": 16},
    }
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="NVFP4",
        iters=0,
        disable_opt_rtn=True,
        layer_config=layer_config,
    )
    quantized_model_path = str(tmp_path / "saved")
    autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")

    # verify loading.
    llm = LLM(
        model=quantized_model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        allow_deprecated_quantization=True,
    )


# ================ Test Evaluation function ===============


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/nvidia-smi") and not os.path.exists("/usr/local/cuda"), reason="CUDA not available"
)
class TestVllmEvaluation:
    """Test VLLM backend evaluation functionality."""

    def test_vllm_backend_with_custom_args(self, tiny_opt_model_path):
        """Test vllm backend evaluation with custom vllm_args parameter."""
        python_path = sys.executable

        os.environ["VLLM_SKIP_WARMUP"] = "true"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Test with custom vllm_args
        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --eval --tasks lambada_openai --eval_bs 128 --eval_backend vllm --limit 10 --vllm_args tensor_parallel_size=1,gpu_memory_utilization=0.2,max_model_len=1024"

        ret = os.system(cmd)

        assert ret == 0, f"vllm evaluation with custom args failed (rc={ret})"

    def test_vllm_backend_with_quantization_iters_0(self, tiny_opt_model_path):
        """Test vllm evaluation with iters=0 (quantization without fine-tuning)."""
        python_path = sys.executable

        os.environ["VLLM_SKIP_WARMUP"] = "true"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --iters 0 --disable_opt_rtn --tasks lambada_openai --eval_bs 8 --eval_backend vllm --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"vllm evaluation with iters=0 failed (rc={ret})"


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
