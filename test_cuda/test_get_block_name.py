import copy
import shutil
import sys
import unittest

sys.path.insert(0, "..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoModelForVision2Seq
from auto_round.utils import get_block_names
from auto_round import AutoRound


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass


    @classmethod
    def tearDownClass(self):
        shutil.rmtree("runs", ignore_errors=True)

    def check_block_names(self,block_names, prefixs=[],n_layers=[]):
        for i, block_name in enumerate(block_names):
            prefix = prefixs[i]
            n_layer = n_layers[i]
            expected_block_names = [prefix+"."+str(i) for i in range(n_layer)]
            assert block_name == expected_block_names


    def test_opt_125m(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.decoder.layers"], [12])



    def test_Qwen(self):
        model_name = "/models/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [28])

    def test_phi4(self):
        model_name = "/models/phi-4"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [40])

    def test_llama3(self):
        model_name = "/models/Meta-Llama-3.1-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [32])


    def test_mixtral(self):
        model_name = "/models/Mixtral-8x7B-Instruct-v0.1"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [32])



    def test_falcon(self):
        model_name = "/models/Falcon3-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [28])


    def test_orca(self):
        model_name = "/models/Orca-2-7b"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [32])

    def test_OLMo(self):
        model_name = "/models/OLMo-2-1124-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [32])


    def test_Qwen2VL(self):
        model_name = "/models/Qwen2-VL-2B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.layers"], [28])

        block_names = get_block_names(model,quant_vision=True)
        self.check_block_names(block_names, ["visual.blocks","model.layers"], [32,28])


    def test_Llama32(self):
        model_name = "/models/Llama-3.2-11B-Vision-Instruct"
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["language_model.model.layers"], [40])

        block_names = get_block_names(model,quant_vision=True)
        self.check_block_names(block_names, ["vision_model.transformer.layers","vision_model.global_transformer.layers","language_model.model.layers"], [32,8,40])


    def test_SmolVLM(self):
        model_name = "/models/SmolVLM-Instruct"
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        block_names = get_block_names(model)
        self.check_block_names(block_names,["model.text_model.layers"], [24])

        block_names = get_block_names(model,quant_vision=True)
        self.check_block_names(block_names, ["model.vision_model.encoder.layers","model.text_model.layers"], [27,24])



if __name__ == "__main__":
    unittest.main()