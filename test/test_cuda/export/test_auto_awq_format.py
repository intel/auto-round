import copy
import os
import shutil

import pytest
import torch
import transformers
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_gptqmodel
from ...helpers import eval_generated_prompt, generate_prompt, get_model_path


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

    @require_gptqmodel
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
    @require_gptqmodel
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
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")

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

    @require_gptqmodel
    def test_fallback_regex_for_awq_format(self, tiny_opt_model_path, dataloader):
        layer_config = {
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "self.save_dir"
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        generate_prompt(model, tokenizer)
        shutil.rmtree(quantized_model_path, ignore_errors=True)
