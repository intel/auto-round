import copy
import re
import shutil
import sys
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "../..")
import torch
import transformers
from lm_eval.utils import make_table  # pylint: disable=E0401

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate, simple_evaluate_user_model
from auto_round.testing_utils import require_autogptq, require_greater_than_050, require_greater_than_051


def get_accuracy(data):
    match = re.search(r"\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|", data)

    if match:
        accuracy = float(match.group(1))
        return accuracy
    else:
        return 0.0


class TestCustomizedData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_mixed_attention_mask(self):
        model_name = "/models/Qwen3-0.6B"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 多个 prompts
        texts = [
            "There is a girl who likes adventure,",
            "Tell me a story about a brave robot,",
            "Explain why the sky is blue,",
        ]

        # 批处理输入（padding 到同长度）
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=9, return_tensors="pt").to(model.device)

        inputs = inputs["input_ids"]
        inputs = inputs.split(dim=0, split_size=1)
        ar = AutoRound(model_name, dataset=inputs, seqlen=9)
        ar.quantize()
