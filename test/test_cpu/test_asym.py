import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch
from _test_helpers import model_infer
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.utils import get_module


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(3):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRoundAsym(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # self.model_name = "/models/opt-125m"
        self.model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_asym_group_size(self):
        model_name = self.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for group_size in [32, 64, 128]:
            bits, sym = 4, False
            ar = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1)
            ar.quantize_and_save(format="auto_round", output_dir=self.save_folder)

            # TODO when ark is ready, uncomment the following lines to do inference test

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_folder,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_folder)

    def test_asym_bits(self):
        model_name = self.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for bits in [2, 8]:
            group_size, sym = 128, False
            ar = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1)
            ar.quantize_and_save(format="auto_round", output_dir=self.save_folder)

            # TODO when ark is ready, uncomment the following lines to do inference test

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_folder,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_folder)

    # use parameters later
    def test_asym_format(self):
        model_name = self.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for format in ["auto_round", "auto_round:auto_gptq", "auto_round:gptqmodel"]:
            bits, group_size, sym = 4, 128, False
            ar = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1)
            # TODO when ark is ready, uncomment the following lines to do inference test
            ar.quantize_and_save(format=format, output_dir=self.save_folder)

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_folder,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_folder)
