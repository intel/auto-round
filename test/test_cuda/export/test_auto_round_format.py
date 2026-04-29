import copy
import shutil

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...envs import (
    require_autogptq,
    require_awq,
    require_gptqmodel,
    require_greater_than_050,
    require_ipex,
)
from ...helpers import eval_generated_prompt, evaluate_accuracy, get_model_path, is_cuda_support_fp8


class TestAutoRound:

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_greater_than_050
    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    @pytest.mark.parametrize("group_size", [32, 128])
    @pytest.mark.parametrize("is_sym", [True, False])
    def test_autoround_format(self, tiny_opt_model_path, bits, group_size, is_sym):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=is_sym,
            iters=0,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir

        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

        # Verify loading
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module), "Loaded model is not an instance of torch.nn.Module"

    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    @require_autogptq
    def test_mixed_precision(self):
        model_name = get_model_path("facebook/opt-125m")
        layer_config = {}

        layer_config["model.decoder.layers.0.self_attn.k_proj"] = {"bits": 8}
        layer_config["model.decoder.layers.2.self_attn.q_proj"] = {
            "bits": 3,
            "group_size": 64,
        }  ## 3bits when using asym will have some issue
        layer_config["model.decoder.layers.6.self_attn.out_proj"] = {"bits": 2, "group_size": 32}
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(model_name, bits=bits, group_size=group_size, sym=sym, layer_config=layer_config)
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        eval_generated_prompt(quantized_model_path)
        evaluate_accuracy(quantized_model_path, threshold=0.32, batch_size=16)

    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    @require_gptqmodel
    def test_awq_backend(self):
        model_name = get_model_path("facebook/opt-125m")
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            iters=1,
            nsamples=1,
            sym=sym,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:auto_awq"
        )

        quantization_config = AutoRoundConfig(backend="auto")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            quantization_config=quantization_config,
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        eval_generated_prompt(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.18, batch_size=16)
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            quantization_config=quantization_config,
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        eval_generated_prompt(model, tokenizer)

    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    @require_greater_than_050
    def test_tritonv2_bf16(self):
        model_name = get_model_path("OPEA/Meta-Llama-3.1-8B-Instruct-int4-sym-inc")
        quantization_config = AutoRoundConfig(backend="tritonv2")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda:0", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        eval_generated_prompt(model, tokenizer)
        torch.cuda.empty_cache()

    @pytest.mark.skip_ci(reason="IPEX is deprecated.")
    @require_ipex
    def test_autoround_gptq_sym_format(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir

        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path)

        from transformers import AutoRoundConfig

        quantization_config = AutoRoundConfig(backend="ipex")

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", trust_remote_code=True, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", trust_remote_code=True, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

    @pytest.mark.skip_ci(reason="IPEX is deprecated.")
    @require_awq
    @require_ipex
    def test_autoround_awq_sym_format(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir

        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:auto_awq"
        )

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

    def test_fp8_block_fp8_format(self):
        model_name = "Qwen/Qwen3-0.6B"

        scheme = "FP8_BLOCK"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=2,
            seqlen=2,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="fp8")
        tmp_layer = compressed_model.model.layers[1].self_attn.q_proj
        assert hasattr(tmp_layer, "weight_scale_inv")
        assert tmp_layer.weight.dtype is torch.float8_e4m3fn
        assert list(tmp_layer.weight_scale_inv.shape) == [16, 8]
        assert compressed_model.config.quantization_config["quant_method"] == "fp8"
        assert compressed_model.config.quantization_config["weight_block_size"] == (128, 128)
        if is_cuda_support_fp8():
            eval_generated_prompt(quantized_model_path, device="cuda:0")
