import copy
import re
import shutil
import sys
import unittest

from auto_round.testing_utils import multi_card

sys.path.insert(0, "../..")

from auto_round import AutoRound, AutoRoundConfig, AutoScheme
from auto_round.auto_schemes.utils import compute_avg_bits_for_model
from auto_round.eval.evaluation import simple_evaluate
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

    def test_gguf_export(self):
        model_name = "/models/Qwen3-0.6B"
        target_bits = 3
        scheme = AutoScheme(avg_bits=target_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M"), ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0)
        ar.quantize_and_save(self.save_dir, format="gguf:q2_k_s")
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_gguf(self):
        model_name = "/models/Qwen3-8B"
        target_bits = 3
        scheme = AutoScheme(avg_bits=target_bits, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M"), ignore_scale_zp_bits=True)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, disable_opt_rtn=True)
        model, layer_config = ar.quantize()
        # self.assertLessEqual(layer_config["lm_head"]["bits"], 8)
        avg_bits, _ = compute_avg_bits_for_model(model, ignore_scale_zp_bits=True)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_shared_layers(self):
        model_name = "/models/opt-125m"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_name)
        shared_layers = [
            ["*.self_attn.k_proj", "v_proj", "q_proj", "out_proj"],
            ("model.decoder.layers.6.fc1", "model.decoder.layers.6.fc2"),
            ("fc1", "fc2"),
        ]
        from auto_round.auto_schemes.utils import parse_shared_layers

        res = parse_shared_layers(model, shared_layers)
        self.assertEqual(len(res), 24)
        assert [
            "model.decoder.layers.2.self_attn.out_proj",
            "model.decoder.layers.2.self_attn.q_proj",
            "model.decoder.layers.2.self_attn.v_proj",
        ] in res
        assert ["model.decoder.layers.6.fc1", "model.decoder.layers.6.fc2"] in res
        assert ["model.decoder.layers.7.fc1", "model.decoder.layers.7.fc2"] in res
        target_bits = 5.0
        scheme = AutoScheme(avg_bits=target_bits, options=("W4A16", "MXFP8"), shared_layers=shared_layers)
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        for names in res:
            bits = []
            for name in names:
                module = get_module(model, name)
                if hasattr(module, "orig_layer"):
                    bits.append(module.orig_layer.bits)
                else:
                    bits.append(module.bits)
            bits = set(bits)
            self.assertEqual(len(bits), 1)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    #
    @multi_card
    def test_multi_card(self):
        model_name = "/models/Qwen3-8B"
        target_bits = 5.254
        # for device_map in ["auto", "0,1", "0", None]:
        scheme = AutoScheme(avg_bits=target_bits, options=("NVFP4"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    @multi_card
    def test_dict_device_map(self):  # TODO rtn mode has bug
        model_name = "/models/Qwen3-8B"
        target_bits = 8.755
        device_map = {"up_proj": 0, "down_proj": 1}

        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP8"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, device_map=device_map)
        model, layer_config = ar.quantize()
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_min_target_bits(self):
        model_name = "/models/opt-125m"
        target_bits = 4.644
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "W8A16"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        model, layer_config = ar.quantize()
        # self.assertLessEqual(layer_config["lm_head"]["bits"], 8)
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    #
    def test_max_target_bits(self):
        model_name = "/models/opt-125m"
        target_bits = 8.211
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "W8A16"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1)
        model, layer_config = ar.quantize()
        # self.assertLessEqual(layer_config["lm_head"]["bits"], 8)
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_patch_scheme(self):
        model_name = "/models/opt-125m"
        target_bits = 5
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "W8A16"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, group_size=32)
        model, layer_config = ar.quantize()
        for n, m in model.named_modules():
            if hasattr(m, "group_size"):
                self.assertEqual(m.group_size, 32)
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_layer_config(self):
        target_bits = 3.0
        model_name = "/models/opt-125m"
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

    def test_lm_head_and_mix_dtype(self):
        model_name = "/models/Qwen3-8B"
        target_bits = 6
        scheme = AutoScheme(avg_bits=target_bits, options=("MXFP4", "W8A16"))
        ar = AutoRound(model=model_name, scheme=scheme, iters=0, nsamples=1, quant_lm_head=True)
        model, layer_config = ar.quantize()
        self.assertLessEqual(layer_config["lm_head"]["bits"], 8)
        avg_bits, _ = compute_avg_bits_for_model(model)
        print(avg_bits)
        assert target_bits - 0.1 < avg_bits <= target_bits + 1e-3

    def test_auto_scheme_export(self):
        model_name = "/models/opt-125m"
        scheme = AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "W8A16", "BF16"))
        ar = AutoRound(model=model_name, scheme=scheme)
        ar.quantize_and_save(self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.25)
        shutil.rmtree(self.save_dir, ignore_errors=True)
