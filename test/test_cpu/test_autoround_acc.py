import copy
import shutil
import sys
import unittest

from auto_round.eval.evaluation import simple_evaluate

sys.path.insert(0, "../..")
from math import isclose

import torch
import transformers
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
        self.llm_dataloader = LLMDataLoader()
        self.save_dir = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_default_acc(self):
        model_name = "/tf_dataset/auto_round/models/hf-internal-testing/tiny-random-GPTJForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        inp = torch.ones([1, 10], dtype=torch.long)
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            device="cpu",
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        out0 = model(inp)
        print(f"out0 = {float(out0[0][0][0][0])}")

        model_tmp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
        autoround_1 = AutoRound(
            model_tmp,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            device="cpu",
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround_1.quantize()
        out1 = model_tmp(inp)

        assert out0[0].equal(out1[0])
        self.assertTrue(isclose(float(out0[0][0][0][0]), -0.021002087742090225, rel_tol=5e-04))

    def test_3bits_asym_autoround(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"

        bits, sym = 3, False
        autoround = AutoRound(model_name, bits=bits, sym=sym, iters=0)
        autoround.quantize_and_save(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        # res = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto", limit=10)

        # accuracy = res["results"]["lambada_openai"]["acc,none"]
        # print(f"accuracy = {accuracy}")
        # assert accuracy > 0.15
        shutil.rmtree(self.save_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
