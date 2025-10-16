import copy
import re
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
from auto_round import AutoRound, AutoRoundConfig, AutoScheme
from auto_round.auto_scheme.utils import compute_avg_bits_for_model
from auto_round.eval.evaluation import simple_evaluate
from auto_round.testing_utils import multi_card
from auto_round.utils import get_module


class TestAutoScheme(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)


    def test_auto_scheme_export(self):
        model_name = "/models/opt-125m"
        scheme = AutoScheme(avg_bits=2, options=("W2A16"),nsamples=1,ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme)
        ar.quantize_and_save(self.save_dir)
        shutil.rmtree(self.save_dir, ignore_errors=True)