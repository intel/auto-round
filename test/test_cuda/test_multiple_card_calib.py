import os
import re
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

from auto_round.testing_utils import multi_card


def get_accuracy(data):
    match = re.search(r"\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|", data)

    if match:
        accuracy = float(match.group(1))
        return accuracy
    else:
        return 0.0


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @multi_card
    def test_multiple_card_calib(self):
        python_path = sys.executable

        ##test llm script
        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model /models/Meta-Llama-3.1-8B-Instruct --devices '0,1' --quant_lm_head --iters 1 --nsamples 1 --output_dir None"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"


if __name__ == "__main__":
    unittest.main()
