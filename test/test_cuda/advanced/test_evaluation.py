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

Run with: pytest test/test_cuda/advanced/test_evaluation.py -v
"""

import os
import sys

import pytest

from ...helpers import opt_name_or_path


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/nvidia-smi") and not os.path.exists("/usr/local/cuda"), reason="CUDA not available"
)
class TestHFEvaluation:
    """Test different evaluation modes: --eval and --eval_backend."""

    def test_eval_mode_hf_backend(self, tiny_opt_model_path):
        """Test --eval flag: evaluate model without quantization (HF backend default)."""
        python_path = sys.executable

        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --eval --tasks lambada_openai --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"HF backend evaluation failed (rc={ret})"

    @pytest.mark.skip_ci(reason="The evaluation is time-consuming")
    def test_iters_0_hf_backend(self, tiny_opt_model_path):
        """Test quantization with iters=0 and HF backend evaluation."""
        python_path = sys.executable

        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --iters 0 --disable_opt_rtn --tasks lambada_openai --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"HF backend with iters=0 failed (rc={ret})"

    @pytest.mark.skip_ci(reason="The evaluation is time-consuming")
    def test_iters_0_task_by_task(self, tiny_opt_model_path):
        """Test quantization with iters=0 and task-by-task evaluation."""
        python_path = sys.executable

        cmd = f"{python_path} -m auto_round --model {tiny_opt_model_path} --iters 0 --disable_opt_rtn --eval_task_by_task --tasks lambada_openai,piqa --limit 10"

        ret = os.system(cmd)

        assert ret == 0, f"Task-by-task with iters=0 failed (rc={ret})"
