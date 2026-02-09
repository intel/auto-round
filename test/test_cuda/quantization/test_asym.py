import copy
import shutil
import sys
import unittest

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(3):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRoundAsym:
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_asym_group_size(self, tiny_opt_model_path):
        for group_size in [32, 64, 128]:
            bits, sym = 4, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

            # TODO when ark is ready, uncomment the following lines to do inference test

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_dir,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_asym_bits(self, tiny_opt_model_path):
        for bits in [2, 3, 8]:
            group_size, sym = 128, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

            # TODO when ark is ready, uncomment the following lines to do inference test

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_dir,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_dir, ignore_errors=True)

    # use parameters later
    def test_asym_format(self, tiny_opt_model_path):
        for format in ["auto_round", "auto_round:auto_gptq", "auto_round:gptqmodel"]:
            bits, group_size, sym = 4, 128, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1
            )
            # TODO when ark is ready, uncomment the following lines to do inference test
            ar.quantize_and_save(format=format, output_dir=self.save_dir)

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_dir,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_asym_group_size_with_tuning(self, tiny_opt_model_path):
        for group_size in [32, 64, 128]:
            bits, sym = 4, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

            # TODO when ark is ready, uncomment the following lines to do inference test

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_dir,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_asym_bits_with_tuning(self, tiny_opt_model_path):
        for bits in [2, 3, 8]:
            group_size, sym = 128, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

            # TODO when ark is ready, uncomment the following lines to do inference test

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_dir,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_dir, ignore_errors=True)

    # use parameters later
    def test_asym_format_with_tuning(self, tiny_opt_model_path):
        for format in ["auto_round", "auto_round:auto_gptq", "auto_round:gptqmodel"]:
            bits, group_size, sym = 4, 128, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, nsamples=1
            )
            # TODO when ark is ready, uncomment the following lines to do inference test
            ar.quantize_and_save(format=format, output_dir=self.save_dir)

            # model = AutoModelForCausalLM.from_pretrained(
            #     self.save_dir,
            #     torch_dtype="auto",
            #     device_map="auto",
            # )

            # tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            # model_infer(model, tokenizer)
            shutil.rmtree(self.save_dir, ignore_errors=True)
