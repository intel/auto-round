import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path, opt_name_or_path


class TestLLMC:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("stas/tiny-random-llama-2")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    # remove since w8a8 not in llmcompressor format supported schemes
    # def test_llmcompressor_w8a8(self):
    #     bits, group_size, sym, act_bits = 8, -1, True, 8
    #     ## quantize the model
    #     autoround = AutoRound(
    #         self.model,
    #         self.tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         act_bits=act_bits,
    #         seqlen=8,
    #         nsamples=2,
    #         iters=0,
    #     )
    #     autoround.quantize()
    #     autoround.save_quantized("./saved", format="llm_compressor", inplace=True)

    def test_llmcompressor_fp8(self):
        ## quantize the model
        model_name = opt_name_or_path
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            nsamples=2,
            iters=0,
        )
        _, quantized_model_path = autoround.quantize_and_save("./saved", format="llm_compressor")

        # from vllm import LLM
        # model = LLM("./saved")
        # result = model.generate("Hello my name is")
        # print(result)

        import json

        config = json.load(open(f"{quantized_model_path}/config.json"))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "channel"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"

    def test_autoround_llmcompressor_fp8(self):
        ## quantize the model
        model_name = opt_name_or_path
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            group_size=0,
            nsamples=2,
            iters=0,
        )
        _, quantized_model_path = autoround.quantize_and_save("./saved", format="auto_round:llm_compressor")

        import json

        config = json.load(open(f"{quantized_model_path}/config.json"))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "tensor"
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["strategy"] == "tensor"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"
