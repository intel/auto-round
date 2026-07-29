import os
import shutil
import sys

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig

from ...helpers import eval_generated_prompt, get_model_path, get_tiny_model, save_tiny_model

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


class TestGGUF:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_q4_0(self, tiny_qwen_model_path):
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            tiny_qwen_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            data_type="int",
            nsamples=1,
            seqlen=8,
        )
        quantized_model_path = self.save_dir

        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, inplace=False, format="gguf:q4_0"
        )
        gguf_file = os.listdir(quantized_model_path)[0]
        assert gguf_file.endswith(".gguf"), "Saved file is not in gguf format"
        # Accuracy test is covered in test_cuda/export/test_gguf_format.py::TestAutoRound::test_q4_0_accuracy

    def test_q2_k_s_routes_data_driven(self, tiny_qwen_model_path):
        autoround = AutoRound(
            tiny_qwen_model_path,
            scheme="gguf:q2_k_s",
            iters=0,
            nsamples=1,
            seqlen=8,
        )

        assert type(autoround).__name__ == "CompressionOrchestrator"
        assert isinstance(autoround.quantize_config, OptimizedRTNConfig)

    def test_func(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model_name,
            iters=0,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, inplace=False, format="gguf:q*_1"
        )
        assert autoround.group_size == 32
        assert not autoround.sym
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        eval_generated_prompt(model, self.tokenizer)

    def test_q4_k_m(self, dataloader, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
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
            "model.layers.1.mlp.gate_proj": {"bits": 3},
            "model.layers.0.mlp.gate_proj": {"bits": 8},
        }
        autoround = AutoRound(
            model_name,
            layer_config=layer_config,
            iters=0,
            seqlen=16,
            nsamples=8,
            dataset=dataloader,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="gguf:q4_k_m,fake"
        )
        assert autoround.layer_config["model.layers.1.self_attn.v_proj"]["super_group_size"] == 16
        assert autoround.layer_config["model.layers.1.self_attn.v_proj"]["data_type"] == "int_sym_dq"
        assert autoround.layer_config["model.layers.0.self_attn.v_proj"]["data_type"] == "int_asym_dq"
        assert autoround.model.model.layers[0].self_attn.v_proj.bits == 4
        assert autoround.model.model.layers[1].self_attn.v_proj.bits == 6
        assert autoround.model.model.embed_tokens.bits == 6
        assert autoround.model.model.embed_tokens.group_size == 16
        assert autoround.model.model.layers[1].mlp.gate_proj.bits == 3
        assert autoround.model.model.layers[0].mlp.gate_proj.bits == 8
        assert autoround.layer_config["model.layers.0.mlp.gate_proj"]["mostly"] == "gguf:q8_0"

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

        # test q2_k_mixed with iters=0 (RTN) on non-MoE model — should still work
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
            f" --bs 16 --iters 0 --disable_opt_rtn --nsamples 1 --seqlen 16 --scheme GGUF:Q2_K_MIXED"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)
        # test q2_k_mixed with iters=1 on non-MoE model — should fallback to q4_k_m
        res = os.system(
            f"PYTHONPATH='{AUTO_ROUND_PATH}:$PYTHONPATH' {python_path} -m auto_round --model {model_name}"
            f" --bs 16 --iters 1 --nsamples 1 --seqlen 16 --format gguf:q2_k_mixed"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"
        shutil.rmtree("../../tmp_autoround", ignore_errors=True)
