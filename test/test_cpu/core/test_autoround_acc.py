import copy
import shutil
from math import isclose

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound  # pylint: disable=E0401

from ...helpers import gptj_name_or_path


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.save_dir = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_default_acc(self, dataloader):
        model_name = gptj_name_or_path
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
            dataset=dataloader,
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
            dataset=dataloader,
        )
        autoround_1.quantize()
        out1 = model_tmp(inp)

        assert out0[0].equal(out1[0])
        assert isclose(float(out0[0][0][0][0]), -0.021002087742090225, rel_tol=5e-04)

    def test_3bits_asym_autoround(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path

        bits, sym = 3, False
        autoround = AutoRound(model_name, bits=bits, sym=sym, iters=0)
        autoround.quantize_and_save(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        shutil.rmtree(self.save_dir, ignore_errors=True)
