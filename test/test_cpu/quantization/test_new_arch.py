import copy
import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path


class TestAutoRound:
    # def test_calib(self, tiny_opt_model_path):
    #     from auto_round.compressors_new import Compressor
    #     from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    #     config = SignRoundConfig(scheme="W4A16", iters=200, lr=0.005, bits=2, group_size=32)
    #     compressor = Compressor(config, tiny_opt_model_path, format="auto_round")
    #     compressor.quantize_and_save()

    def test_opt_rtn(self, tiny_opt_model_path):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.compressors_new import Compressor

        config = RTNConfig(scheme="W4A16", bits=2, group_size=32)
        compressor = Compressor(config, tiny_opt_model_path, format="auto_round")
        compressor.quantize_and_save()

        ar = AutoRound(tiny_opt_model_path, bits=2, group_size=32, iters=0)
        ar.quantize_and_save()

    # def test_rtn(self, tiny_opt_model_path):
    #     from auto_round.compressors_new import Compressor
    #     from auto_round.algorithms.quantization.rtn.config import RTNConfig
    #     config = RTNConfig(scheme="W4A16", bits=2, group_size=32, disable_opt_rtn=True)
    #     compressor = Compressor(config, tiny_opt_model_path, format="auto_round")
    #     compressor.quantize_and_save()
