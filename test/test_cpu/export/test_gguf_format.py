import os
import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path, get_tiny_model, save_tiny_model

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


class TestGGUF:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_basic_usage(self, tiny_gemma_model_path, tiny_qwen_model_path):
        python_path = sys.executable
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_gemma_model_path} "
            f" --bs 16 --iters 0 --nsamples 1 --format gguf:q4_k_m"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {tiny_qwen_model_path}"
            f" --bs 16 --iters 1 --nsamples 1 --format fake,gguf:q4_0"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_q4_0(self):
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            self.model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            data_type="int",
            nsamples=1,
            seqlen=8,
        )
        quantized_model_path = "./saved"

        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q4_0")
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

        shutil.rmtree("./saved", ignore_errors=True)

    def test_func(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model_name,
            iters=1,
            nsamples=1,
            seqlen=10,
            # data_type="int"
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="gguf:q*_1")
        assert autoround.group_size == 32
        assert not autoround.sym
        gguf_file = os.listdir("saved")[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_gguf_baseline(self):
        model_name = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=3,
            group_size=16,
            sym=True,
            iters=0,
            nsamples=8,
            seqlen=2,
            data_type="rtn_int_sym_dq",
            super_group_size=16,
            super_bits=6,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="fake")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_q4_k_m(self, dataloader):
        model_name = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            "lm_head": {
                "bits": 4,
                "group_size": 32,
                "sym": False,
                "data_type": "int_asym_dq",
                "super_bits": 6,
                "super_group_size": 8,
            },
            "model.embed_tokens": {"bits": 6, "group_size": 32, "super_bits": 6, "super_group_size": 8},
            "model.layers.12.mlp.gate_proj": {"bits": 3},
            "model.layers.10.mlp.gate_proj": {"bits": 8},
        }
        autoround = AutoRound(
            model,
            tokenizer,
            layer_config=layer_config,
            iters=0,
            seqlen=1,
            nsamples=8,
            dataset=dataloader,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m,fake")
        assert autoround.layer_config["model.layers.11.self_attn.v_proj"]["super_group_size"] == 16
        assert autoround.layer_config["model.layers.11.self_attn.v_proj"]["data_type"] == "int_sym_dq"
        assert autoround.layer_config["model.layers.7.self_attn.v_proj"]["data_type"] == "int_asym_dq"
        assert autoround.model.model.layers[0].self_attn.v_proj.bits == 6
        assert autoround.model.model.layers[12].self_attn.v_proj.bits == 4
        assert autoround.model.model.embed_tokens.bits == 6
        assert autoround.model.model.embed_tokens.group_size == 16
        assert autoround.model.model.layers[12].mlp.gate_proj.bits == 3
        assert autoround.model.model.layers[10].mlp.gate_proj.bits == 8
        assert autoround.layer_config["model.layers.10.mlp.gate_proj"]["mostly"] == "gguf:q8_0"
        shutil.rmtree("./saved", ignore_errors=True)

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(model, tokenizer, iters=0, nsamples=1, seqlen=128, disable_opt_rtn=False)
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m,fake")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_all_format(self, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
        python_path = sys.executable
        # for gguf_format in ["gguf:q4_0", "gguf:q4_1", "gguf:q4_k_m", "gguf:q6_k"]:
        for gguf_format in ["gguf:q4_k_m"]:
            res = os.system(
                f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name} "
                f" --bs 16 --iters 1 --nsamples 1 --seqlen 16 --format {gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

            res = os.system(
                f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
                f" --bs 16 --iters 0 --nsamples 1 --seqlen 16 --format fake,{gguf_format}"
            )
            if res > 0 or res == -1:
                assert False, "cmd line test fail, please have a check"
            shutil.rmtree("../../tmp_autoround", ignore_errors=True)

        # test mixed q2_k_s
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
            f" --bs 16 --iters 0 --nsamples 1 --seqlen 16 --scheme GGUF:Q2_K_MIXED"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)

    def test_vlm_gguf(self):
        from ...helpers import save_tiny_model

        model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
        tiny_model_path = save_tiny_model(model_name, "./tmp/tiny_qwen_vl_model_path", num_layers=3, is_mllm=True)
        from auto_round import AutoRoundMLLM

        autoround = AutoRoundMLLM(
            tiny_model_path,
            iters=0,
            nsamples=8,
            disable_opt_rtn=True,
            quant_nontext_module=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        assert "mmproj-model.gguf" in os.listdir("./saved")
        for file_name in os.listdir(quantized_model_path):
            file_size = os.path.getsize(os.path.join(quantized_model_path, file_name)) / 1024**2
            if file_name == "mmproj-model.gguf":
                assert abs(file_size - 56) < 5.0
            else:
                assert abs(file_size - 264) < 5.0
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree(tiny_model_path, ignore_errors=True)

    def test_vlm_gguf_wo_quant_nontext_module(self):
        from ...helpers import save_tiny_model

        model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
        tiny_model_path = save_tiny_model(model_name, "./tmp/tiny_qwen_vl_model_path", num_layers=3, is_mllm=True)
        from auto_round import AutoRoundMLLM

        autoround = AutoRoundMLLM(
            tiny_model_path,
            iters=0,
            nsamples=8,
            disable_opt_rtn=True,
            quant_nontext_module=False,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        assert "mmproj-model.gguf" in os.listdir("./saved")
        for file_name in os.listdir(quantized_model_path):
            file_size = os.path.getsize(os.path.join(quantized_model_path, file_name)) / 1024**2
            if file_name == "mmproj-model.gguf":
                assert abs(file_size - 361) < 5.0
            else:
                assert abs(file_size - 264) < 5.0
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree(tiny_model_path, ignore_errors=True)

    def test_qtype_setting(self):
        # Qwen2.5-0.5B-Instruct no output, token_embed q6_k fallbakc to q8_0 336M
        # Qwen3-0.6B output q6_k, token_embed q4_0  448M
        # Qwen3-8B output q6_k, token_embed q4_0 4.5G
        # Llama-3.2-1B-Instruct o output, token_embed q6_k 736M
        from auto_round.compressors.utils import set_layer_config
        from auto_round.export.export_to_gguf.config import ModelType

        model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert ar.layer_config["model.embed_tokens"]["bits"] == 8
        assert "lm_head" not in ar.layer_config

        model_name = "Qwen/Qwen3-0.6B"
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert ar.layer_config["model.embed_tokens"]["bits"] == 4
        assert ar.layer_config["lm_head"]["bits"] == 6 and ar.layer_config["lm_head"]["super_bits"] == 8

        layer_config = {
            "model.embed_tokens": {"bits": 6, "super_bits": 8},
            "lm_head": {"bits": 4},
        }
        ar = AutoRound(model=model_name, scheme="gguf:q4_0", iters=0, layer_config=layer_config)
        ar.formats = ["gguf:q4_0"]
        ar.layer_config, _, _ = set_layer_config(
            ar.model,
            ar.layer_config,
            ar.scheme,
            ar.scale_dtype,
            ar.supported_types,
            ar.inner_supported_types,
            ar.quant_block_list,
            ar.ignore_layers,
            ar.quant_lm_head,
            enable_gguf_official_mixed=True,
            is_mllm=ar.mllm,
        )
        assert (
            ar.layer_config["lm_head"]["bits"] == 4
            and ar.layer_config["model.embed_tokens"]["bits"] == 6
            and ar.layer_config["model.embed_tokens"]["super_bits"] == 8
        )

    def test_q2k_mixed(self):
        model_name = get_model_path("Qwen/Qwen1.5-MoE-A2.7B")
        saved_tiny_model_path = save_tiny_model(
            model_name,
            "./tmp/tiny_qwen_model_path",
            num_layers=3,
            is_mllm=False,
        )
        autoround = AutoRound(
            saved_tiny_model_path,
            iters=0,
            nsamples=1,
            seqlen=16,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q2_k_mixed")
        gguf_file = os.listdir(quantized_model_path)[0]
        file_size = os.path.getsize(os.path.join(quantized_model_path, gguf_file)) / 1024**2
        assert abs(file_size - 1362) < 5.0
        from gguf.gguf_reader import GGUFReader

        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        assert gguf_model.get_tensor(2).name == "blk.0.attn_k.weight"
        assert gguf_model.get_tensor(2).tensor_type.name == "Q4_K"
        assert gguf_model.get_tensor(10).name == "blk.0.ffn_up_exps.weight"
        assert gguf_model.get_tensor(10).tensor_type.name == "Q2_K"

        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree(saved_tiny_model_path, ignore_errors=True)
