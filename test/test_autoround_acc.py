
import copy
import shutil
import sys
import unittest
sys.path.insert(0, "..")
import torch
import transformers
from math import isclose
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound  # pylint: disable=E0401
from auto_round.export.export_to_itrex.export import pack_model  # pylint: disable=E0401

class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)
            

class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_default_acc(self):
        bits, group_size, sym = 4, 128, True
        inp = torch.ones([1, 10], dtype=torch.long)
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            device="cpu",
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader
        )
        autoround.quantize()
        out0 = self.model(inp)
        print(f"out0 = {float(out0[0][0][0][0])}")
        
        model_tmp = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32, trust_remote_code=True)
        autoround_1 = AutoRound(
            model_tmp,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            device="cpu",
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader
        )
        autoround_1.quantize()
        out1 = model_tmp(inp)
        
        assert out0[0].equal(out1[0])
        self.assertTrue(isclose(float(out0[0][0][0][0]), -0.021002087742090225, rel_tol=1e-04))


if __name__ == "__main__":
    unittest.main()

