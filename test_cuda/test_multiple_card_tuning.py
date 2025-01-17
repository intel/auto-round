import shutil
import sys
import unittest
from auto_round.eval.evaluation import simple_evaluate
from lm_eval.utils import make_table  # pylint: disable=E0401

sys.path.insert(0, "..")
from auto_round import AutoRound

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def get_accuracy(data):
    match = re.search(r'\|acc\s+\|[↑↓]\s+\|\s+([\d.]+)\|', data)

    if match:
        accuracy = float(match.group(1))
        return accuracy
    else:
        return 0.0


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_device_map(self):
        model_name = "/models/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = {".*q_proj": '0', ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "cpu"}
        autoround = AutoRound(model, tokenizer, iters=2, device_map=device_map, nsamples=7,seqlen=32)
        autoround.quantize()

    def test_device_map_str(self):
        model_name = "/models/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = '.*q_proj:0,.*k_proj:cuda:0,v_proj:1,.*up_proj:1'
        autoround = AutoRound(model, tokenizer,device_map=device_map)
        autoround.quantize()
        autoround.save_quantized(self.save_dir, format="auto_round", inplace=False)
        model_args = f"pretrained={self.save_dir}"
        res = simple_evaluate(model="hf", model_args=model_args,
                              tasks=self.tasks,
                              batch_size="auto")
        res = make_table(res)
        accuracy = get_accuracy(res)
        print(accuracy)
        assert accuracy > 0.45 ##0.4786
        shutil.rmtree("./saved", ignore_errors=True)

    def test_layer_norm(self):
        model_name = "/models/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = {"norm": "cuda:1"}
        autoround = AutoRound(model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32,
                              enable_norm_bias_tuning=True)
        autoround.quantize()


    def test_rms_norm(self):
        model_name = "/models/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = {"norm": "cuda:1"}
        autoround = AutoRound(model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32,
                              enable_norm_bias_tuning=True)
        autoround.quantize()

    def test_act_quantization(self):
        model_name = "/models/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = {".*q_proj": '0', ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "1"}
        autoround = AutoRound(model, tokenizer, iters=2, device_map=device_map, nsamples=7,seqlen=32,act_bits=4,act_dynamic=False)
        autoround.quantize()

    def test_lm_head(self):
        model_name = "/models/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = {".*q_proj": '0', ".*k_proj": "cuda:1", "v_proj": 1, ".*up_proj": "1","lm_head":1}
        layer_config={"lm_head": {"bits": 4}}
        autoround = AutoRound(model, tokenizer, iters=2, device_map=device_map, nsamples=7, seqlen=32,
                              enable_norm_bias_tuning=True,layer_config=layer_config)
        autoround.quantize()


