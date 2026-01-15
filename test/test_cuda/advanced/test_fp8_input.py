import os
import shutil

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate
from auto_round.utils import llm_load_model

from ...helpers import get_model_path, get_tiny_model


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

    def test_diff_datatype(self):
        for scheme in ["NVFP4", "MXFP4"]:
            model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
            for iters in [0, 1]:
                print(f"Testing scheme: {scheme}, iters: {iters}")
                ar = AutoRound(model_name, iters=iters, scheme=scheme)
                ar.quantize_and_save(output_dir=self.save_dir)
                shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_fp8_model_with_ignore_layers(self):
        """Test that FP8 layers specified in ignore_layers remain in FP8 format."""
        from auto_round.compressors.config import SchemeExtraConfig
        from auto_round.utils import is_fp8_linear
        
        model_name = get_model_path("qwen/Qwen3-0.6B-FP8")
        model, tokenizer = llm_load_model(model_name)
        
        # Verify model is FP8
        assert hasattr(model, "is_fp8") and model.is_fp8, "Model should be marked as FP8"
        
        # Check that some layers are FP8 before quantization
        fp8_layers_before = []
        for n, m in model.named_modules():
            if is_fp8_linear(m):
                fp8_layers_before.append(n)
        print(f"FP8 layers before quantization: {len(fp8_layers_before)}")
        assert len(fp8_layers_before) > 0, "Model should have FP8 layers"
        
        # Quantize with ignore_layers containing "attn" - these should remain in FP8
        scheme_config = SchemeExtraConfig(ignore_layers="attn")
        from auto_round.compressors.config import ExtraConfig
        extra_config = ExtraConfig()
        extra_config.scheme_config = scheme_config
        
        ar = AutoRound(
            model=model, 
            tokenizer=tokenizer, 
            iters=0, 
            extra_config=extra_config
        )
        quantized_model, layer_config = ar.quantize()
        
        # Check that layers with "attn" in the name have skip_quantization flag
        attn_layers_with_skip = []
        for layer_name, cfg in layer_config.items():
            if "attn" in layer_name and cfg.get("skip_quantization", False):
                attn_layers_with_skip.append(layer_name)
        
        print(f"Attention layers marked with skip_quantization: {len(attn_layers_with_skip)}")
        assert len(attn_layers_with_skip) > 0, "Attention layers should be marked with skip_quantization"
        
        # Check that FP8 layers with "attn" still exist as FP8Linear after quantization
        fp8_attn_layers_after = []
        for n, m in quantized_model.named_modules():
            if "attn" in n and is_fp8_linear(m):
                fp8_attn_layers_after.append(n)
        
        print(f"FP8 attention layers after quantization: {len(fp8_attn_layers_after)}")
        assert len(fp8_attn_layers_after) > 0, "Attention layers should remain in FP8 format"
        
        print(f"✓ Test passed: {len(fp8_attn_layers_after)} FP8 attention layers preserved")
