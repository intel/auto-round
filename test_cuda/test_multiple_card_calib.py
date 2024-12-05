import copy
import shutil
import sys
import unittest
import re

sys.path.insert(0, "..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate
from lm_eval.utils import make_table  # pylint: disable=E0401
import os

def get_accuracy(data):
    match = re.search(r'\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|', data)

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

    def test_multiple_card_calib(self):
        python_path = sys.executable

        ##test llm script
        res = os.system(
            f"cd .. && {python_path} -m auto_round --model /models/Meta-Llama-3.1-8B-Instruct --devices '0,1' --quant_lm_head --disable_eval --iters 1 --nsamples 1 --output_dir None")


if __name__ == "__main__":
    unittest.main()


