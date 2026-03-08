import copy
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class TestAutoRoundAct:
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

    def test_mx_fp4(self, tiny_opt_model, opt_tokenizer, dataloader):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model,
            opt_tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            act_bits=4,
            data_type="mx_fp",
        )
        autoround.quantize()

    def test_wint4fp8_dynamic(self, tiny_opt_model, opt_tokenizer, dataloader):
        bits, group_size = 4, 128
        autoround = AutoRound(
            tiny_opt_model,
            opt_tokenizer,
            bits=bits,
            group_size=group_size,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            act_bits=8,
            data_type="fp8",
            act_data_type="fp8",
        )
        autoround.quantize()

    def test_wint4fp8_static(self, tiny_opt_model, opt_tokenizer, dataloader):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model,
            opt_tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            act_bits=8,
            data_type="fp8_to_int_sym",
            act_dynamic=False,
            act_data_type="fp8",
        )
        autoround.quantize()

    @pytest.mark.parametrize("act_group_size", [-1, 128])
    def test_wfp8afp8_static(self, act_group_size, tiny_opt_model, opt_tokenizer, dataloader):
        from auto_round.wrapper import WrapperWALayer

        autoround = AutoRound(
            tiny_opt_model,
            opt_tokenizer,
            group_size=128,
            act_group_size=act_group_size,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            data_type="fp8",
            act_dynamic=False,
            act_data_type="fp8",
        )
        autoround.quantize()

        k_proj = autoround.model.model.decoder.layers[1].self_attn.k_proj
        assert isinstance(k_proj, WrapperWALayer), "k_proj should be WrapperWALayer"
        if act_group_size == -1:
            assert k_proj.orig_layer.act_scale.shape[0] == 20, "act_scale shape[0] should be 20"
            assert k_proj.orig_layer.act_max.shape[0] == 20, "act_max shape[0] should be 20"
        else:
            assert k_proj.orig_layer.act_scale.shape[0] == int(2 * 10 * 768 / 128), "act_scale shape[0] is incorrect"
            assert k_proj.orig_layer.act_max.shape[0] == int(2 * 10 * 768 / 128), "act_max shape[0] is incorrect"

    def test_act_config_MXFP4_saving(self, tiny_opt_model_path, dataloader):
        scheme = "MXFP4"
        layer_config = {"lm_head": {"act_bits": 8, "bits": 8}, "k_proj": {"act_bits": 8, "bits": 8}}
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme=scheme,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        assert "lm_head" not in model.config.quantization_config.extra_config

        # check inblock layer config values
        kproj_config = model.config.quantization_config.extra_config["model.decoder.layers.1.self_attn.k_proj"]
        assert "act_bits" in kproj_config.keys() and kproj_config["act_bits"] == 8
        assert "bits" in kproj_config.keys() and kproj_config["bits"] == 8
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_act_config_NVFP4_saving(self, tiny_opt_model_path, dataloader):
        scheme = "NVFP4"
        layer_config = {"k_proj": {"act_bits": 16, "bits": 16}}
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme=scheme,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        kproj_config = model.config.quantization_config.extra_config["model.decoder.layers.1.self_attn.k_proj"]
        assert "act_bits" in kproj_config.keys() and kproj_config["act_bits"] == 16
        assert "bits" in kproj_config.keys() and kproj_config["bits"] == 16
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_WOQ_config_INT_saving(self, tiny_opt_model_path, dataloader):
        scheme = "W4A16"
        layer_config = {"k_proj": {"bits": 8}}
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme=scheme,
            iters=2,
            seqlen=2,
            sym=False,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        extra_config = model.config.quantization_config.extra_config

        # check inblock layer config values
        kproj_config = extra_config["model.decoder.layers.1.self_attn.k_proj"]
        assert "bits" in kproj_config.keys() and kproj_config["bits"] == 8
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_act_config_FP8_saving(self, tiny_opt_model_path, dataloader):
        scheme = "FP8_STATIC"
        layer_config = {
            "lm_head": {"act_bits": 8, "bits": 8},
            # check fp8 woq config
            "k_proj": {
                "bits": 8,
                "group_size": 0,
                "data_type": "fp",
                "act_bits": 16,
                "act_data_type": "fp",
            },
        }
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme=scheme,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        from transformers import AutoConfig

        extra_config = AutoConfig.from_pretrained(quantized_model_path).quantization_config["extra_config"]
        assert "lm_head" not in extra_config

        # check inblock layer config values
        kproj_config = extra_config["model.decoder.layers.0.self_attn.k_proj"]
        assert "act_bits" in kproj_config.keys() and kproj_config["act_bits"] == 16
        assert "group_size" in kproj_config.keys() and kproj_config["group_size"] == 0
        shutil.rmtree(quantized_model_path, ignore_errors=True)
