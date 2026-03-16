import copy
import shutil

import pytest
import torch
import transformers
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_awq, require_optimum, require_package_version_ut
from ...helpers import get_model_path, get_tiny_model, transformers_version


class TestAutoRound:
    save_dir = "./saved"

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

    @require_optimum
    def test_autogptq_format(self, tiny_opt_model_path):
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            disable_opt_rtn=True,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_gptq")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        assert (
            res == "</s>There is a girl who likes adventure, she is a good friend of mine, she is a good friend of"
            " mine, she is a good friend of mine, she is a good friend of mine, she is a good friend of mine, "
            "she is a good friend of mine, she is"
        )
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autogptq_format_qsave_ignore_layers(self, tiny_opt_model_path):
        model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path)

        layer_config = {}
        for n, m in model.named_modules():
            if "q_proj" in n:
                layer_config[n] = {"bits": 16}

        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            disable_opt_rtn=True,
            layer_config=layer_config,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", trust_remote_code=True, quantization_config=AutoRoundConfig()
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        g_tokens = model.generate(**inputs, max_new_tokens=50)[0]
        res = tokenizer.decode(g_tokens)
        print(res)
        ##print(res)
        shutil.rmtree("./saved", ignore_errors=True)
