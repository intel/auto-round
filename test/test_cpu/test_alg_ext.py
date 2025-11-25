import copy
import shutil
import sys
import unittest

from parameterized import parameterized

sys.path.insert(0, "../..")

from auto_round import AutoRound


class TestAlgExt(unittest.TestCase):
    def test_alg_ext(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        ar = AutoRound(model_name, scheme="W2A16", iters=1, nsamples=1, enable_alg_ext=True)
        ar.quantize()

        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B"
        ar = AutoRound(model_name, scheme="gguf:q4_k_s", iters=1, nsamples=1, enable_alg_ext=True)
        ar.quantize()

    def test_alg_ext_import(self):
        from auto_round.alg_ext import wrapper_autoround

    def test_all_support_dtype(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        for scheme in ["MXFP4", "NVFP4", "W2A16G64"]:
            ar = AutoRound(
                model_name, scheme="W2A16", iters=1, nsamples=1, enable_alg_ext=True, enable_torch_compile=True
            )
            ar.quantize()
