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

import os
import sys
from test.helpers import opt_name_or_path

import pytest


def _run_auto_round_cli(monkeypatch, cmd):
    """Run the auto_round CLI in-process by patching ``sys.argv``.

    ``cmd`` contains only the CLI arguments (everything after ``-m auto_round``).
    This replaces the previous ``os.system('python -m auto_round ...')`` calls so
    the code runs in the test process and is measured by coverage.
    """
    from auto_round.cli.main import run

    monkeypatch.setattr(sys, "argv", ["auto_round", *cmd.split()])
    try:
        run()
    except SystemExit as exc:  # argparse help/errors exit; only 0/None is success
        assert exc.code in (0, None), f"cmd line test fail, exit code {exc.code}"


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/nvidia-smi") and not os.path.exists("/usr/local/cuda"), reason="CUDA not available"
)
class TestHFEvaluation:
    """Test different evaluation modes: --eval and --eval_backend."""

    def test_eval_mode_hf_backend(self, monkeypatch, tiny_opt_model_path):
        """Test --eval flag: evaluate model without quantization (HF backend default)."""
        _run_auto_round_cli(
            monkeypatch,
            f"--model {tiny_opt_model_path} --eval --tasks lambada_openai --limit 10",
        )

    @pytest.mark.skip_ci(reason="The evaluation is time-consuming")
    def test_iters_0_hf_backend(self, monkeypatch, tiny_opt_model_path):
        """Test quantization with iters=0 and HF backend evaluation."""
        _run_auto_round_cli(
            monkeypatch,
            f"--model {tiny_opt_model_path} --iters 0 --disable_opt_rtn --tasks lambada_openai --limit 10",
        )

    @pytest.mark.skip_ci(reason="The evaluation is time-consuming")
    def test_iters_0_task_by_task(self, monkeypatch, tiny_opt_model_path):
        """Test quantization with iters=0 and task-by-task evaluation."""
        _run_auto_round_cli(
            monkeypatch,
            f"--model {tiny_opt_model_path} --iters 0 --disable_opt_rtn "
            f"--eval_task_by_task --tasks lambada_openai,piqa --limit 10",
        )
