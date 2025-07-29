import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class TestLLMC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "stas/tiny-random-llama-2"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_llmcompressor_w8a8(self):
        bits, group_size, sym, act_bits = 8, -1, True, 8
        ## quantize the model
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            act_bits=act_bits,
            iters=0,
        )
        autoround.quantize()
        autoround.save_quantized("./saved", format="llmcompressor", inplace=True)


if __name__ == "__main__":
    unittest.main()
