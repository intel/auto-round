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


class TestGGUF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()
    
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
    
    def test_q2_k_export(self):
        bits, group_size, sym = 2, 16, False
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        autoround = AutoRound(
            model,
            tokenizer,
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
        gguf_file = "Qwen2.5-1.5B-Instruct-1.5B-Q2_K_S.gguf"
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        result = self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0])
        print(result)

        from auto_round.eval.evaluation import simple_evaluate_user_model
        result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="piqa")
        self.assertGreater(result['results']['piqa']['acc,none'], 0.45)
        
        shutil.rmtree("./saved", ignore_errors=True)

    def test_basic_usage(self):
        python_path = sys.executable
        res = os.system(
            f"cd .. && {python_path} -m auto_round --model {self.model_name} --eval_task_by_task"
            f" --tasks piqa,openbookqa --bs 16 --iters 1 --nsamples 1 --format fake,gguf:q4_0"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_q4_0(self):
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            data_type="int"
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q4_0")
        gguf_file = "Qwen2.5-0.5B-Instruct-494M-Q4_0.gguf"
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        from auto_round.eval.evaluation import simple_evaluate_user_model
        result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="piqa")
        self.assertGreater(result['results']['piqa']['acc,none'], 0.55)
        shutil.rmtree("./saved", ignore_errors=True)
    
    def test_q4_1(self):
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            data_type="int"
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q4_1")
        gguf_file = "Qwen2.5-0.5B-Instruct-494M-Q4_1.gguf"
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        from auto_round.eval.evaluation import simple_evaluate_user_model
        result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="piqa")
        self.assertGreater(result['results']['piqa']['acc,none'], 0.55)
        shutil.rmtree("./saved", ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
