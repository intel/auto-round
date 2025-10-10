import copy
import re
import shutil
import sys
import unittest

from auto_round.auto_schemes.utils import compute_avg_bits_for_model
from auto_round.eval.evaluation import simple_evaluate

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

    def test_avg_bits(self):
        model_name = "facebook/opt-125m"
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, format="fake")
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert 2.9 < avg_bits <= 3.0

    def test_auto_scheme_export(self):
        model_name = "facebook/opt-125m"
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
        ar = AutoRound(model=model_name, scheme=scheme)
        ar.quantize_and_save(self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.25)
