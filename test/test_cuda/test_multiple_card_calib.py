import os
import re
import shutil
import sys

import pytest

from auto_round.testing_utils import multi_card


def get_accuracy(data):
    match = re.search(r"\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|", data)

    if match:
        accuracy = float(match.group(1))
        return accuracy
    else:
        return 0.0


class TestAutoRound:
    save_dir = "./saved"
    tasks = "lambada_openai"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @multi_card
    def test_multiple_card_calib(self):
        python_path = sys.executable

        ##test llm script
        res = os.system(
            f"PYTHONPATH='../..:$PYTHONPATH' {python_path} -m auto_round --model /models/Meta-Llama-3.1-8B-Instruct --devices '0,1' --quant_lm_head --iters 1 --nsamples 1 --output_dir None"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    @multi_card
    def test_multiple_card_nvfp4(self):
        python_path = sys.executable

        ##test llm script
        res = os.system(
            f"PYTHONPATH='../..:$PYTHONPATH' {python_path} -m auto_round --model facebook/opt-125m  --scheme NVFP4 --devices '0,1' --iters 1 --nsamples 1 --enable_torch_compile --low_gpu_mem_usage"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
