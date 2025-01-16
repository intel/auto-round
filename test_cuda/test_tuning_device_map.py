import shutil
import sys
import unittest
sys.path.insert(0, "..")
from auto_round import AutoRound


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
    #
    # def test_device_map(self):
    #     model_name = "/models/Qwen2-0.5B-Instruct"
    #     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     device_map = {".*q_proj": "0", ".*k_proj": "cuda:0", "v_proj": 1, ".*up_proj": "cpu"}
    #     autoround = AutoRound(model, tokenizer,iters=2,device_map_for_block=device_map)
    #     autoround.quantize()


    def test_device_map_str(self):
        model_name = "/models/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = '.*q_proj:0,.*k_proj:cuda:0,v_proj:1,.*up_proj:cpu'
        autoround = AutoRound(model, tokenizer,iters=2,device_map_for_block=device_map)
        autoround.quantize()
