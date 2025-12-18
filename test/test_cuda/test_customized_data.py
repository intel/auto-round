from transformers import AutoModelForCausalLM, AutoTokenizer

import copy
import re
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig


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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 多个 prompts
        texts = [
            "There is a girl who likes adventure,",
            "Tell me a story about a brave robot,",
            "Explain why the sky is blue,"
        ]

        # 批处理输入（padding 到同长度）
        inputs = tokenizer(
            texts,
            padding=True, truncation=True,
            max_length=9,
            return_tensors="pt"
        ).to(model.device)
        inputs = inputs["input_ids"]
        inputs = inputs.split(dim=0, split_size=1)
        ar = AutoRound(model_name, dataset=inputs, seqlen=9)
        ar.quantize()



