import copy
import os
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

    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
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
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_awq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        assert model is not None, "Loaded model should not be None."

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    @require_optimum
    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_autoawq_format_fp_qsave_layers(self):
        layer_config = {
            "model.decoder.layers.0.self_attn.k_proj": {"bits": 16},
        }
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
        quantized_model_path = os.path.join(self.save_dir, "test_export")
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")

        # test loading with AutoRoundConfig
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=AutoRoundConfig()
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        eval_generated_prompt(model, tokenizer)

        # test loading without quantization_config
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        eval_generated_prompt(model, tokenizer)
