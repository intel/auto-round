import copy
import re
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

from auto_round import AutoRound, AutoRoundConfig, AutoScheme

class TestAutoScheme(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_auto_scheme(self):
        model_name = "facebook/opt-125m"
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=1, nsamples=1)
        ar.quantize_and_save(self.save_dir)
