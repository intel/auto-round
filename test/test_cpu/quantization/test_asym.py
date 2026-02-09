import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path


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
        self.model_name = get_model_path("facebook/opt-125m")
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_asym_group_size(self):
        model_name = self.model_name
        for group_size in [32, 64, 128]:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_asym_bits(self):
        model_name = self.model_name
        for bits in [2, 8]:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            shutil.rmtree(self.save_folder, ignore_errors=True)

    # use parameters later
    def test_asym_format(self):
        model_name = self.model_name

        for format in ["auto_round", "auto_round:auto_gptq", "auto_round:gptqmodel"]:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            shutil.rmtree(self.save_folder, ignore_errors=True)
