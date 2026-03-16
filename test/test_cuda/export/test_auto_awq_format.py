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

    @require_awq
    # @require_package_version_ut("transformers", "<4.57.0")
    def test_autoawq_format(self, tiny_opt_model_path):
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

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_awq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        assert res == (
            "</s>There is a girl who likes adventure, but she's not a very good one.\nI don't"
            " know why you're getting downvoted. I think you're just being a dick.\nI'm not a dick, "
            "I just think it's funny that people are downvoting"
        )
        shutil.rmtree("./saved", ignore_errors=True)

    @require_optimum
    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_autoawq_format_fp_qsave_layers(self, tiny_opt_model_path):
        layer_config = {
            "model.decoder.layers.0.self_attn.k_proj": {"bits": 16},
        }
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
        quantized_model_path = "./saved/test_export"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=AutoRoundConfig()
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)

        shutil.rmtree("./saved", ignore_errors=True)
