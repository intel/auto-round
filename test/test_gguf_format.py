import os
import sys
import unittest
import shutil
sys.path.insert(0, "..")

import torch
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
        self.model_name = "meta-llama/Llama-3.2-1B"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.model.model.layers = self.model.model.layers[:3]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()
    
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
    
    def test_q2_k_export(self):
        bits, group_size, sym = 2, 16, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=1,
            dataset=self.llm_dataloader,
            data_type="int_asym_dq"
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q2_k_s")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file="Llama-3.2-1B-445M-Q2_K_S.gguf", device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_basic_usage(self):
        python_path = sys.executable
        res = os.system(
            f"cd .. && {python_path} -m auto_round --model {self.model_name} --eval_task_by_task"
            f" --tasks piqa,openbookqa --bs 16 --iters 1 --nsamples 1 --format fake,gguf:q4_k_s"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)
if __name__ == "__main__":
    unittest.main()
