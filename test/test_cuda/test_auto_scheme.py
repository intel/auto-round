import copy
import re
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from lm_eval.utils import make_table  # pylint: disable=E0401
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig, AutoScheme
from auto_round.eval.evaluation import simple_evaluate, simple_evaluate_user_model
from auto_round.testing_utils import require_autogptq, require_greater_than_050, require_greater_than_051


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
        scheme = AutoScheme(target_bits=3, options=("W2A16", "W4A16", "BF16"))
        ar = AutoRound(model_name=model_name, scheme=scheme)
        ar.quantize_and_save(self.save_dir)
