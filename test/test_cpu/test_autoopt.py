import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRoundAdam


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_Adam(self):
        bits, group_size, sym = 4, 128, False
        from auto_round.utils import get_block_names

        llm_block_names = get_block_names(self.model, quant_vision=True)
        bits, group_size, sym, batch_size = 4, 128, False, 20
        adamround = AutoRoundAdam(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            batch_size=batch_size,
            dataset=self.llm_dataloader,
            to_quant_block_names=llm_block_names,
        )
        adamround.quantize()


if __name__ == "__main__":
    unittest.main()
