import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model


class TestAlgExt(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_2bits(self):
        model_name = "/models/opt-125m"
        ar = AutoRound(model=model_name, bits=2, group_size=64, enable_alg_ext=True)
        ar.quantize_and_save(self.save_folder)
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=64, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        # wo alg ext 0.2084, with 0.2364
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.22)
        shutil.rmtree(self.save_folder, ignore_errors=True)
