import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class TestLLMC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "/tf_dataset/auto_round/models/stas/tiny-random-llama-2"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_llmcompressor_w8a8(self):
        bits, group_size, sym, act_bits = 8, -1, True, 8
        ## quantize the model
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            act_bits=act_bits,
            seqlen=8,
            nsamples=2,
            iters=0,
        )
        autoround.quantize()
        autoround.save_quantized("./saved", format="llm_compressor", inplace=True)

    def test_llmcompressor_fp8(self):
        ## quantize the model
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            nsamples=2,
            iters=0,
        )
        autoround.quantize_and_save("./saved", format="llm_compressor")
        # from vllm import LLM
        # model = LLM("./saved")
        # result = model.generate("Hello my name is")
        # print(result)

        import json

        config = json.load(open("./saved/config.json"))
        self.assertIn("group_0", config["quantization_config"]["config_groups"])
        self.assertEqual(config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"], 8)
        self.assertEqual(config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"], "channel")
        self.assertEqual(config["quantization_config"]["quant_method"], "compressed-tensors")

    def test_autoround_llmcompressor_fp8(self):
        ## quantize the model
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            group_size=0,
            nsamples=2,
            iters=0,
        )
        autoround.quantize_and_save("./saved", format="auto_round:llm_compressor")

        import json

        config = json.load(open("./saved/config.json"))
        self.assertIn("group_0", config["quantization_config"]["config_groups"])
        self.assertEqual(config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"], 8)
        self.assertEqual(config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"], "tensor")
        self.assertEqual(
            config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["strategy"], "tensor"
        )
        self.assertEqual(config["quantization_config"]["quant_method"], "compressed-tensors")


if __name__ == "__main__":
    unittest.main()
