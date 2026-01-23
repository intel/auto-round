# Copyright (c) 2025 Intel Corporation
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

"""
GPU tests for VLLM evaluation functionality.
Tests the eval_with_vllm function and custom vllm_args parameter parsing.
Validates accuracy thresholds for quantized models.

Run with: pytest test/test_cuda/integrations/test_vllm_eval.py -v
"""

import os
import sys

import pytest

# Test models for vllm evaluation
VLLM_EVAL_MODELS = [
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  # auto_round:auto_gptq format
]


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/nvidia-smi") and not os.path.exists("/usr/local/cuda"), reason="CUDA not available"
)
class TestVllmEvaluation:
    """Test VLLM backend evaluation functionality."""

    @pytest.mark.parametrize("model", VLLM_EVAL_MODELS)
    def test_vllm_backend_with_custom_args(self, model):
        """Test vllm backend evaluation with custom vllm_args parameter."""
        python_path = sys.executable

        os.environ["VLLM_SKIP_WARMUP"] = "true"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Test with custom vllm_args
        cmd = f"{python_path} -m auto_round --model {model} --eval --tasks lambada_openai --eval_bs 8 --eval_backend vllm --limit 10 --vllm_args tensor_parallel_size=1,gpu_memory_utilization=0.8,max_model_len=2048"

        ret = os.system(cmd)

        assert ret == 0, f"vllm evaluation with custom args failed (rc={ret})"

    def test_vllm_backend_with_quantization_iters_0(self, tiny_opt_model_path):
        """Test vllm evaluation with iters=0 (quantization without fine-tuning)."""
        python_path = sys.executable

        os.environ["VLLM_SKIP_WARMUP"] = "true"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --iters 0 --tasks lambada_openai --eval_bs 8 --eval_backend vllm --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"vllm evaluation with iters=0 failed (rc={ret})"


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/nvidia-smi") and not os.path.exists("/usr/local/cuda"), reason="CUDA not available"
)
class TestHFEvaluation:
    """Test different evaluation modes: --eval and --eval_backend."""

    @pytest.mark.parametrize("model", VLLM_EVAL_MODELS)
    def test_eval_mode_hf_backend(self, model):
        """Test --eval flag: evaluate model without quantization (HF backend default)."""
        python_path = sys.executable

        cmd = f"{python_path} -m auto_round --model {model} --eval --tasks lambada_openai --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"HF backend evaluation failed (rc={ret})"

    @pytest.mark.parametrize("model", VLLM_EVAL_MODELS)
    def test_eval_mode_task_by_task(self, model):
        """Test --eval with --eval_task_by_task flag (HF backend)."""
        python_path = sys.executable

        cmd = f"{python_path} -m auto_round --model {model} --eval --eval_task_by_task --tasks lambada_openai,piqa --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"HF backend task-by-task evaluation failed (rc={ret})"

    def test_iters_0_hf_backend(self, tiny_opt_model_path):
        """Test quantization with iters=0 and HF backend evaluation."""
        python_path = sys.executable

        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --iters 0 --tasks lambada_openai --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"HF backend with iters=0 failed (rc={ret})"

    def test_iters_0_task_by_task(self, tiny_opt_model_path):
        """Test quantization with iters=0 and task-by-task evaluation."""
        python_path = sys.executable

        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --iters 0 --eval_task_by_task --tasks lambada_openai,piqa --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"Task-by-task with iters=0 failed (rc={ret})"
