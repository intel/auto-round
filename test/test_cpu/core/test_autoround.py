import copy
import shutil

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils import get_module

from ...helpers import (
    evaluate_accuracy,
    get_model_path,
    model_infer,
    opt_name_or_path,
    qwen_name_or_path,
    transformers_version,
)


class TestAutoRound:

    @classmethod
    def setup_class(self):
        model_name = opt_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_folder = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_bits_setting(self, tiny_opt_model_path):
        layer_config = {"model.decoder.layers.0.self_attn.k_proj": {"data_type": "mx_fp8", "group_size": 32}}
        autoround = AutoRound(tiny_opt_model_path, iters=2, seqlen=2, nsamples=1, layer_config=layer_config)
        autoround.quantize()
        module = get_module(autoround.model, "model.decoder.layers.0.self_attn.k_proj")
        if module.bits != 8:
            raise ValueError(f"Expected bits to be 8, but got {module.bits}")

    def test_layer_config(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        layer_config = {"self_attn": {"bits": 4, "data_type": "nv_fp", "act_bits": 16, "group_size": 16}}
        autoround = AutoRound(
            model_name,
            self.tokenizer,
            scheme="NVFP4",
            iters=0,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
            amp=False,
        )
        autoround.quantize_and_save(self.save_folder, inplace=False, format="fake")

    def test_remove_whole_block(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        layer_config = {
            "model.decoder.layers.0.self_attn.k_proj": {"bits": 32},
            "model.decoder.layers.0.self_attn.v_proj": {"bits": 32},
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 32},
            "model.decoder.layers.0.self_attn.out_proj": {"bits": 32},
            "model.decoder.layers.0.fc1": {"bits": 32},
            "model.decoder.layers.0.fc2": {"bits": 32},
        }
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_consecutive_quant(self, tiny_opt_model_path, tiny_phi2_model_path, dataloader):
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()

        autoround = AutoRound(
            tiny_phi2_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_mx_fp4(self, dataloader):
        model_name = opt_name_or_path
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            act_bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            nsamples=2,
            seqlen=128,
            data_type="mx_fp4",
            act_data_type="mx_fp_rceil",
        )
        model, _ = autoround.quantize()
        evaluate_accuracy(model, self.tokenizer, threshold=0.3, batch_size="auto:8", limit=32)

    def test_nv_fp4(self, dataloader):
        model_name = opt_name_or_path
        bits, group_size, sym = 4, 16, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            data_type="nv_fp4",
        )
        model, _ = autoround.quantize()
        evaluate_accuracy(model, self.tokenizer, threshold=0.35, batch_size="auto:8", limit=32)

    def test_w4g1(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=dataloader,
        )
        autoround.quantize()

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_g128(self, bits, dataloader):
        model_name = opt_name_or_path
        group_size, sym = 128, True
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=10,
            dataset=dataloader,
        )
        model, _ = autoround.quantize()
        if bits == 3:
            evaluate_accuracy(model, self.tokenizer, threshold=0.15, batch_size="auto:8", limit=32)
        elif bits == 4:
            evaluate_accuracy(model, self.tokenizer, threshold=0.3, batch_size="auto:8", limit=32)

    def test_disable_quanted_input(self, dataloader):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_quanted_input=False,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_enable_norm_bias_tuning_qwen3(self, tiny_qwen_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        model_name = tiny_qwen_model_path
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_norm_bias_tuning=True,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_enable_norm_bias_tuning(self, dataloader):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_quanted_input=False,
            enable_norm_bias_tuning=True,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_disable_minmax_tuning(self, dataloader):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            dataset=dataloader,
        )
        autoround.quantize()

    #
    def test_signround(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            enable_quanted_input=False,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_lm_head_layer_config_way(self, tiny_untied_qwen_model_path, dataloader):
        bits, group_size, sym = 4, -1, False
        layer_config = {"lm_head": {"data_type": "int"}}
        autoround = AutoRound(
            tiny_untied_qwen_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=10,
            dataset=dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_wa_quant(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        bits, group_size, sym, act_bits = 4, 128, False, 4
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            act_bits=act_bits,
        )
        autoround.quantize()

    def test_auto_device_map(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, False
        model_name = tiny_opt_model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True, device_map="auto"
        )
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_device_map_dict(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, False
        device_map = {".*": "cpu"}
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            device_map=device_map,
        )
        autoround.quantize()

        # test model_name
        model_name = tiny_opt_model_path
        autoround = AutoRound(
            model_name,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            device_map=device_map,
        )
        autoround.quantize()

    def test_fp32(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, False
        model_name = tiny_opt_model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
        )
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            amp=False,
        )
        autoround.quantize()

    def test_tensor_reshape(self, dataloader):
        bits, group_size, sym = 4, 100, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_rtn(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=0, nsamples=1)
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)

    def test_embed_quant(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        model_name = tiny_opt_model_path
        layer_config = {
            "model.decoder.embed_tokens": {"bits": 4},
        }
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            nsamples=3,
            dataset=dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_fallback_layers(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        model_name = tiny_opt_model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
        )
        layer_config = {
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 16},
            "model.decoder.layers.1.self_attn.k_proj": {"bits": 16},
            "model.decoder.embed_tokens": {"bits": 16},
        }
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()
        quantized_model_path = self.save_folder

        _, quantized_model_path = autoround.save_quantized(
            output_dir=quantized_model_path, format="auto_round", inplace=True, return_folders=True
        )

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=1)[0])
