import copy
import shutil
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ..helpers import model_infer


class TestQuantizationConv1d:
    @classmethod
    def setup_class(self):
        self.model_name = "/tf_dataset/auto_round/models/MBZUAI/LaMini-GPT-124M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_quant(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )

        autoround.quantize()
        autoround.save_quantized("./saved")

        model = AutoModelForCausalLM.from_pretrained("./saved", device_map="cpu", trust_remote_code=True)
        model_infer(model, self.tokenizer)
