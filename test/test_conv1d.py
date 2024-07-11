import copy
import shutil
import sys
import unittest

sys.path.insert(0, "..")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestQuantizationConv1d(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "MBZUAI/LaMini-GPT-124M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)


    def test_quant(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
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
        try:
            import auto_gptq
        except:
            return
        autoround.save_quantized("./saved")


if __name__ == "__main__":
    unittest.main()
