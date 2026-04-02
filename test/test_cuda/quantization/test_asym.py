import copy
import shutil
import sys
import unittest

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path, model_infer


class TestAutoRoundAsym:

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_asym_group_size_with_tuning(self, group_size, tiny_opt_model_path):
        bits, sym = 4, False
        ar = AutoRound(tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, nsamples=1)
        _, saved_folders = ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(
            saved_folders[0],
            torch_dtype="auto",
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(saved_folders[0])
        model_infer(model, tokenizer)

    @pytest.mark.skip_ci(reason="Not necessary since it's covered by backend tests")  # skip this test in CI
    @pytest.mark.parametrize("bits", [2, 3, 8])
    def test_asym_bits_with_tuning(self, bits, tiny_opt_model_path):
        group_size, sym = 128, False
        ar = AutoRound(tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, nsamples=1)
        _, saved_folders = ar.quantize_and_save(format="auto_round", output_dir=self.save_dir)

        model = AutoModelForCausalLM.from_pretrained(
            saved_folders[0],
            torch_dtype="auto",
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(saved_folders[0])
        model_infer(model, tokenizer)

    @pytest.mark.skip_ci(reason="Not necessary since it's covered by backend tests")  # skip this test in CI
    @pytest.mark.parametrize("format", ["auto_round", "auto_round:auto_gptq", "auto_round:gptqmodel"])
    def test_asym_format_with_tuning(self, format, tiny_opt_model_path):
        bits, group_size, sym = 4, 128, False
        ar = AutoRound(tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, nsamples=1)
        _, saved_folders = ar.quantize_and_save(format=format, output_dir=self.save_dir)

        if format == "auto_round:auto_gptq":
            # Cannot load correctly, skip auto_gptq since it's deprecated.
            return

        model = AutoModelForCausalLM.from_pretrained(
            saved_folders[0],
            torch_dtype="auto",
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(saved_folders[0])
        model_infer(model, tokenizer)
