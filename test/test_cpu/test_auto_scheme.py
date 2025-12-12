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

    def test_auto_scheme_export(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        scheme = AutoScheme(avg_bits=2, options=("W2A16"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)
        shutil.rmtree(self.save_dir, ignore_errors=True)

        scheme = AutoScheme(avg_bits=4, options=("mxfp4"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)
        shutil.rmtree(self.save_dir, ignore_errors=True)

        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B"
        scheme = AutoScheme(avg_bits=3, options=("gguf:q2_k_s,gguf:q4_k_s"), nsamples=1, ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        ar.quantize_and_save(self.save_dir)
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_layer_config(self):
        from auto_round.auto_scheme.utils import compute_avg_bits_for_model
        from auto_round.utils import get_module

        target_bits = 3.0
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "BF16"))
        user_layer_config = {"model.decoder.layers.10.fc1": {"bits": 8, "group_size": 32, "sym": False}}
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, layer_config=user_layer_config)
        model, layer_config = ar.quantize()
        self.assertEqual(layer_config["model.decoder.layers.10.fc1"]["bits"], 8)
        self.assertEqual(layer_config["model.decoder.layers.10.fc1"]["sym"], False)
        self.assertEqual(layer_config["model.decoder.layers.10.fc1"]["group_size"], 32)
        layer = get_module(model, "model.decoder.layers.10.fc1")
        self.assertEqual(layer.bits, 8)
        self.assertEqual(layer.sym, False)
        self.assertEqual(layer.group_size, 32)
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3


if __name__ == "__main__":
    unittest.main()
