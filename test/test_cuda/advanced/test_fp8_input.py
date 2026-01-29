import os
import shutil

import pytest
import torch
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate
from auto_round.utils import llm_load_model

from ...helpers import get_model_path, get_tiny_model, transformers_version


class TestAutoRound:
    save_dir = "./saved"

    def tiny_fp8_model(self):
        model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
        model, tokenizer = llm_load_model(model_name)
        model.model.layers = model.model.layers[:3]
        return model, tokenizer

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_small_model_rtn_generation(self):
        model, tokenizer = self.tiny_fp8_model()
        ar = AutoRound(model=model, tokenizer=tokenizer, iters=0)
        ar.quantize_and_save(output_dir=self.save_dir)
        model = AutoModelForCausalLM.from_pretrained(self.save_dir, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0.0"),
        reason="GGUF format saving and loading failed in transformers v5, \
            https://github.com/huggingface/transformers/issues/43482",
    )
    def test_gguf_imatrix(self):
        model, tokenizer = self.tiny_fp8_model()
        ar = AutoRound(model=model, tokenizer=tokenizer, iters=0)
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
        model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
        ar = AutoRound(model=model_name, iters=0)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.25

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_small_model_iters1(self):
        model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
        ar = AutoRound(model=model_name, iters=1)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.25

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_medium_model_rtn(self):
        model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
        ar = AutoRound(model=model_name, iters=0)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.33

        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_medium_model_rtn_with_lm_head(self):
        model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
        layer_config = {"lm_head": {"bits": 4}}
        ar = AutoRound(model=model_name, iters=0, layer_config=layer_config)
        _, folder = ar.quantize_and_save(output_dir=self.save_dir)
        model_args = f"pretrained={self.save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks="lambada_openai", batch_size="auto")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.33

        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0.0"),
        reason="GGUF format saving and loading failed in transformers v5, \
            https://github.com/huggingface/transformers/issues/43482",
    )
    def test_fp8_model_gguf(self):
        from llama_cpp import Llama

        model, tokenizer = self.tiny_fp8_model()
        ar = AutoRound(model=model, tokenizer=tokenizer, iters=0)
        ar.quantize_and_save(output_dir=self.save_dir, format="gguf:q4_0")
        for file in os.listdir(self.save_dir):
            if file.endswith(".gguf"):
                gguf_file = file
        llm = Llama(f"saved/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
        shutil.rmtree(self.save_dir, ignore_errors=True)

        model, tokenizer = self.tiny_fp8_model()
        ar = AutoRound(model=model, tokenizer=tokenizer, iters=1)
        ar.quantize_and_save(output_dir=self.save_dir, format="gguf:q3_k_s")
        for file in os.listdir(self.save_dir):
            if file.endswith(".gguf"):
                gguf_file = file
        llm = Llama(f"saved/{gguf_file}", n_gpu_layers=-1)
        output = llm("There is a girl who likes adventure,", max_tokens=32)
        print(output)
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0.0"),
        reason="We need this patch for fp8 model loading without dequantization."
        "https://github.com/intel/auto-round/blob/72e1cecb4a984db101e26700618266115029b9ac/test/test_cuda/quantization/test_mxfp_nvfp.py#L19C5-L19C25",
    )
    def test_diff_datatype(self):
        for scheme in ["NVFP4", "MXFP4"]:
            model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
            for iters in [0, 1]:
                print(f"Testing scheme: {scheme}, iters: {iters}")
                ar = AutoRound(model_name, iters=iters, scheme=scheme)
                ar.quantize_and_save(output_dir=self.save_dir)
                shutil.rmtree(self.save_dir, ignore_errors=True)
