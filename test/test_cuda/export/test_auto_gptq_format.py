import copy
import shutil

import pytest
import torch
import transformers
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_awq, require_optimum, require_package_version_ut
from ...helpers import eval_generated_prompt, get_model_path, get_tiny_model, transformers_version


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
        assert model is not None, "Loaded model should not be None."
        shutil.rmtree("./saved", ignore_errors=True)

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")  # skip this test in CI
    def test_autogptq_format_qsave_ignore_layers(self):
        model = AutoModelForCausalLM.from_pretrained(get_model_path("facebook/opt-125m"))

        layer_config = {}
        for n, m in model.named_modules():
            if "q_proj" in n:
                layer_config[n] = {"bits": 16}

        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            get_model_path("facebook/opt-125m"),
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
        eval_generated_prompt(model, tokenizer)

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", trust_remote_code=True, quantization_config=AutoRoundConfig()
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        eval_generated_prompt(model, tokenizer)
        shutil.rmtree("./saved", ignore_errors=True)
