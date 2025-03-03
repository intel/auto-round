import os
import shutil
import sys
import unittest

sys.path.insert(0, "..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf_format(self):
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            nsamples=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"    
        autoround.save_quantized(output_dir=quantized_model_path, format="gguf:q4_1")

        from llama_cpp import Llama
        llm = Llama("saved/Qwen2.5-0.5B-Instruct-Q4_1.gguf", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
    
        save_dir = os.path.join(os.path.dirname(__file__), "saved")
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        res = os.system(
            f"cd .. && {sys.executable} -m auto_round --model {model_path} --iter 2 "
            f"--output_dir {save_dir} --nsample 2 --format gguf:q4_0 --device 0"
        )
        print(save_dir)
        self.assertFalse(res > 0 or res == -1, msg="qwen2 tuning fail")
        
        from llama_cpp import Llama
        llm = Llama("saved/Qwen2.5-0.5B-Instruct-w4g32-gguf-q4-0/Qwen2.5-0.5B-Instruct-Q4_0.gguf", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
    
    
if __name__ == "__main__":
    unittest.main()