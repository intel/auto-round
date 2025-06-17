import os
import sys
import unittest
import shutil
sys.path.insert(0, "../..")

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

    def test_basic_usage(self):
        python_path = sys.executable
        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model {self.model_name} "
            f" --bs 16 --iters 1 --nsamples 1 --format fake,gguf:q4_0"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

        res = os.system(
            f"cd ../.. && {python_path} -m auto_round --model {self.model_name}"
            f" --bs 16 --iters 1 --nsamples 1 --format fake,gguf:q4_0"
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
        quantized_model_path = "./saved"

        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q4_0")
        gguf_file = "Qwen2.5-0.5B-Instruct-494M-Q4_0.gguf"
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        # from auto_round.eval.evaluation import simple_evaluate_user_model
        # result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="openbookqa", eval_model_dtype="bf16")
        # # 0.246
        # self.assertGreater(result['results']['openbookqa']['acc,none'], 0.23)
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
        quantized_model_path = "./saved"

        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q4_1")
        gguf_file = "Qwen2.5-0.5B-Instruct-494M-Q4_1.gguf"
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        # from auto_round.eval.evaluation import simple_evaluate_user_model
        # result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="openbookqa", eval_model_dtype="bf16")
        # # 0.23
        # self.assertGreater(result['results']['openbookqa']['acc,none'], 0.22)
        shutil.rmtree("./saved", ignore_errors=True)

    def test_func(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            # bits=bits,
            # group_size=group_size,
            # sym=sym,
            iters=1,
            # data_type="int"
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_1")
        self.assertTrue(autoround.group_size == 32)
        self.assertFalse(autoround.sym)
        gguf_file = os.listdir("saved")[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=3,
            group_size=16,
            sym=True,
            iters=1,
            data_type="int_sym_dq",
            super_group_size=16,
            super_bits=6
        )
        quantized_model_path = "./saved"
        # autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_k_s")
        # from auto_round.eval.evaluation import simple_evaluate_user_model
        # gguf_file = os.listdir("saved")[0]
        # model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        # result = simple_evaluate_user_model(model, self.tokenizer, batch_size=16, tasks="lambada_openai", eval_model_dtype="bf16")
        # self.assertGreater(result['results']['lambada_openai']['acc,none'], 0.5)
        shutil.rmtree("./saved", ignore_errors=True)

    def test_q5_k(self):
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=5,
            group_size=32,
            sym=False,
            iters=1,
            data_type="int_asym_dq",
            super_group_size=8,
            super_bits=6
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_k_s")
        gguf_file = os.listdir("saved")[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_q6_k(self):
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=6,
            group_size=16,
            sym=True,
            iters=1,
            data_type="int_sym_dq",
            super_group_size=16,
            super_bits=8
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_k")
        gguf_file = os.listdir("saved")[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_gguf_baseline(self):
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=3,
            group_size=16,
            sym=True,
            iters=0,
            data_type="rtn_int_sym_dq",
            super_group_size=16,
            super_bits=6,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="fake")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=5,
            group_size=32,
            sym=True,
            iters=0,
            data_type="int_asym_dq",
            super_group_size=8,
            super_bits=6,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q5_k_s,fake")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path + "/fake", device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

