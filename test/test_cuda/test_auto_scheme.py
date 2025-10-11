import copy
import re
import shutil
import sys
import unittest
sys.path.insert(0, "../..")

from auto_round.auto_schemes.utils import compute_avg_bits_for_model
from auto_round.eval.evaluation import simple_evaluate
from auto_round.utils import get_module


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
        model_name = "/models/opt-125m"
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
        user_layer_config = {"model.decoder.layers.10.fc1":{"bits":8,"group_size":32, "sym":False}}
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1,layer_config=user_layer_config)
        model, layer_config = ar.quantize()
        self.assertEqual(layer_config["model.decoder.layers.10.fc1"]["bits"], 8)
        self.assertEqual(layer_config["model.decoder.layers.10.fc1"]["sym"], False)
        self.assertEqual(layer_config["model.decoder.layers.10.fc1"]["group_size"], 32)
        layer = get_module(model, "model.decoder.layers.10.fc1")
        self.assertEqual(layer.bits, 8)
        self.assertEqual(layer.sym, False)
        self.assertEqual(layer.group_size,32)
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert 2.9 < avg_bits <= 3.0

    def test_lm_head_and_mix_dtype(self):
        model_name = "/models/Qwen3-8B"
        target_bits = 8.192
        scheme = AutoScheme(avg_bits=target_bits, options=( "MXFP4", "W8A16"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        model, layer_config = ar.quantize()
        # self.assertLessEqual(layer_config["lm_head"]["bits"], 8)
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits-0.1 < avg_bits <= target_bits

    def test_auto_scheme_export(self):
        model_name = "facebook/opt-125m"
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
        ar = AutoRound(model=model_name, scheme=scheme)
        ar.quantize_and_save(self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.25)
