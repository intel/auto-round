import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_small_model_rtn_generation(self):
        model_name = "/models/Qwen3-0.6B-FP8"
        ar = AutoRound(model=model_name, iters=0)
        ar.quantize_and_save(output_dir=self.save_dir)
        model = AutoModelForCausalLM.from_pretrained(self.save_dir, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_gguf_imatrix(self):
        model_name = "/models/Qwen3-0.6B-FP8"
        ar = AutoRound(model=model_name, iters=0)
        ar.quantize_and_save(format="gguf:q2_k_s", output_dir=self.save_dir)
        # from llama_cpp import Llama
        #
        # gguf_file = os.listdir("saved/Qwen3-0.6B-FP8/-gguf")[0]
        # llm = Llama(f"saved/Qwen2.5-0.5B-Instruct-gguf/{gguf_file}", n_gpu_layers=-1)
        # output = llm("There is a girl who likes adventure,", max_tokens=32)
        # print(output)
        # shutil.rmtree("./saved", ignore_errors=True)
        # model = AutoModelForCausalLM.from_pretrained(self.save_dir, torch_dtype="auto", trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        # text = "There is a girl who likes adventure,"
        # inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_small_model_rtn(self):
        model_name = "/models/Qwen3-0.6B-FP8"
        ar = AutoRound(model=model_name, iters=0)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.25)

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_small_model_iters1(self):
        model_name = "/models/Qwen3-0.6B-FP8"
        ar = AutoRound(model=model_name, iters=1)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.25)

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_medium_model_rtn(self):
        model_name = "/models/Qwen3-8B-FP8"
        ar = AutoRound(model=model_name, iters=0)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.55)

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_medium_model_rtn_with_lm_head(self):
        model_name = "/models/Qwen3-8B-FP8"
        layer_config = {"lm_head": {"bits": 4}}
        ar = AutoRound(model=model_name, iters=0, layer_config=layer_config)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.55)

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_fp8_model_gguf(self):
        from llama_cpp import Llama

        model_name = "Qwen/Qwen3-0.6B-FP8"

        ar = AutoRound(model=model_name, iters=0)
        ar.quantize_and_save(output_dir=self.save_dir, format="gguf:q4_0")
        for file in os.listdir(self.save_dir):
            if file.endswith(".gguf"):
                gguf_file = file
        llm = Llama(f"saved/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
        shutil.rmtree(self.save_dir, ignore_errors=True)

        ar = AutoRound(model=model_name, iters=1)
        ar.quantize_and_save(output_dir=self.save_dir, format="gguf:q3_k_s")
        for file in os.listdir(self.save_dir):
            if file.endswith(".gguf"):
                gguf_file = file
        llm = Llama(f"saved/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
        shutil.rmtree(self.save_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
