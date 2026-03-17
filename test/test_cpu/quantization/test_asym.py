import copy
import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path, model_infer


class TestAutoRoundAsym:
    save_folder = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_asym_group_size(self, tiny_opt_model_path):
        for group_size in [32, 64, 128]:
            bits, sym = 4, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_folder)

            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder,
                torch_dtype="auto",
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            model_infer(model, tokenizer)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_asym_bits(self, tiny_opt_model_path):
        for bits in [2, 8]:
            group_size, sym = 128, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_folder)

            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder,
                torch_dtype="auto",
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            model_infer(model, tokenizer)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    # use parameters later
    def test_asym_format(self, tiny_opt_model_path):
        for format in ["auto_round", "auto_round:auto_gptq", "auto_round:gptqmodel"]:
            bits, group_size, sym = 4, 128, False
            ar = AutoRound(
                tiny_opt_model_path,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=0,
                seqlen=2,
                nsamples=1,
                disable_opt_rtn=True,
            )
            ar.quantize_and_save(format=format, output_dir=self.save_folder)

            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder,
                torch_dtype="auto",
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            model_infer(model, tokenizer)
            shutil.rmtree(self.save_folder, ignore_errors=True)
