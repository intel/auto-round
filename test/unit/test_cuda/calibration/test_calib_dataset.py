import json
import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class TestLocalCalibDataset:
    def test_combine_dataset(self, tiny_opt_model_path):
        dataset = "NeelNanda/pile-10k" + ",BAAI/CCI3-HQ" + ",madao33/new-title-chinese"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=2, seqlen=128, dataset=dataset
        )
        autoround.quantize()
