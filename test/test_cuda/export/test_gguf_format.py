import os
import shutil
import sys

import pytest
import torch
import transformers
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_gguf
from ...helpers import eval_generated_prompt, evaluate_accuracy, generate_prompt, get_model_path, save_tiny_model

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


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

    @require_gguf
    def test_gguf_format(self, tiny_qwen_model_path, dataloader):
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            tiny_qwen_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            nsamples=2,
            dataset=dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir
        autoround.save_quantized(output_dir=quantized_model_path, format="gguf:q4_1")

    @require_gguf
    def test_q4_0_accuracy(self):
        model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            model_name, bits=bits, group_size=group_size, sym=sym, iters=0, data_type="int", disable_opt_rtn=True
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q4_0")

        gguf_file = os.listdir(quantized_model_path)[0]

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        eval_generated_prompt(model, autoround.tokenizer)

        evaluate_accuracy(model, autoround.tokenizer, threshold=0.54, batch_size=16, task="piqa")

    @pytest.mark.skip_ci(reason="Not necessary to test all options in CI")
    @require_gguf
    def test_q2_k_export(self, dataloader):
        bits, group_size, sym = 2, 16, False
        model_path = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        tiny_model_path = "./tmp/tmp_tiny_qwen_model_path"
        save_tiny_model(model_path, tiny_model_path, num_layers=2)
        autoround = AutoRound(
            model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=1,
            dataset=dataloader,
            data_type="int_asym_dq",
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="gguf:q2_k_s")
        gguf_file = os.listdir(quantized_model_path)[0]
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, gguf_file=gguf_file, device_map="auto")
        result = generate_prompt(model, autoround.tokenizer)
        print(result)
        shutil.rmtree(tiny_model_path, ignore_errors=True)

    @pytest.mark.skip_ci(reason="Not necessary to test all options in CI")
    @require_gguf
    def test_all_format(self):
        for model_name in ["Qwen/Qwen3-8B", "meta-llama/Llama-3.2-3B"]:
            for gguf_format in ["gguf:q5_0", "gguf:q5_1", "gguf:q3_k_m", "gguf:q5_k_m", "gguf:q6_k", "gguf:q8_0"]:
                model_path = get_model_path(model_name)
                tiny_model_path = "tmp_tiny_model"
                tiny_model_path = save_tiny_model(model_path, tiny_model_path, num_layers=2)
                ar = AutoRound(tiny_model_path, scheme=gguf_format, iters=0, nsamples=1, seqlen=16)
                ar.quantize_and_save(output_dir=self.save_dir, format=gguf_format)

                ar = AutoRound(tiny_model_path, scheme=gguf_format, iters=1, nsamples=1, seqlen=16)
                ar.quantize_and_save(output_dir=self.save_dir, format=gguf_format)

                shutil.rmtree(tiny_model_path, ignore_errors=True)

    @pytest.mark.skip_ci(reason="Not necessary to test special models in CI")
    @require_gguf
    def test_special_model(self):
        from ...helpers import save_tiny_model

        model_name = get_model_path("ibm-granite/granite-4.0-h-tiny")
        tiny_model_path = save_tiny_model(model_name, "tiny_model_path", num_layers=2)
        from auto_round import AutoRound

        autoround = AutoRound(
            tiny_model_path,
            iters=0,
            nsamples=8,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_0")
        file_name = os.listdir(quantized_model_path)[0]
        file_size = os.path.getsize(os.path.join(quantized_model_path, file_name)) / 1024**2
        assert abs(file_size - 307) < 5.0
        shutil.rmtree(tiny_model_path, ignore_errors=True)

    @require_gguf
    def test_vlm_gguf(self):
        from huggingface_hub import hf_hub_download

        from ...helpers import save_tiny_model

        model_name = "google/gemma-3-4b-it"
        tiny_model_path = save_tiny_model(model_name, "tiny_model_path", num_layers=3, is_mllm=True, use_fast=False, use_config=True)
        # Needs tokenizer.model for gguf
        # New transformers won't download it even with use_fast=False
        file_path = hf_hub_download(repo_id=model_name, filename="tokenizer.model", local_dir=tiny_model_path)

        from auto_round import AutoRound

        autoround = AutoRound(
            tiny_model_path,
            device="auto",
            nsamples=32,
            iters=0,
            disable_opt_rtn=True,
            quant_nontext_module=True,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q4_k_m")
        assert "mmproj-model.gguf" in os.listdir(self.save_dir)
        for file in os.listdir(self.save_dir):
            print(f"{file}: {os.path.getsize(os.path.join(self.save_dir, file)) / 1024**2} MB")
            file_size = os.path.getsize(os.path.join(self.save_dir, file)) / 1024**2
            if "mmproj-model.gguf" in file:
                assert abs(file_size - 75) < 5.0
            else:
                assert abs(file_size - 690) < 5.0

        shutil.rmtree(tiny_model_path, ignore_errors=True)

    @pytest.mark.skip_ci(reason="Not necessary to test all options in CI")
    def test_q2k_mixed(self):
        model_path = get_model_path("miromind-ai/MiroThinker-v1.5-30B")
        saved_tiny_model_path = save_tiny_model(
            model_path,
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
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q2_k_mixed")
        gguf_file = os.listdir(quantized_model_path)[0]
        file_size = os.path.getsize(os.path.join(quantized_model_path, gguf_file)) / 1024**2
        assert abs(file_size - 1236) < 5.0
        from gguf.gguf_reader import GGUFReader

        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        assert gguf_model.get_tensor(2).name == "blk.0.attn_v.weight"
        assert gguf_model.get_tensor(2).tensor_type.name == "Q4_K"
        assert gguf_model.get_tensor(9).name == "blk.0.ffn_up_exps.weight"
        assert gguf_model.get_tensor(9).tensor_type.name == "Q2_K"

        shutil.rmtree(saved_tiny_model_path, ignore_errors=True)

    @require_gguf
    def test_q2_k_s_ffn_down_q4k(self):
        """Verify blk.0.ffn_down.weight is Q4_K in gguf:q2_k_s format.
        Blocks where i_layer < n_layer/8 should use Q4_K instead of Q2_K for ffn_down."""
        from gguf.gguf_reader import GGUFReader

        model_path = get_model_path("Qwen/Qwen3-1.7B")
        tiny_model_path = "./tmp/tiny_qwen3_1b"
        save_tiny_model(model_path, tiny_model_path, num_layers=8)
        autoround = AutoRound(
            tiny_model_path,
            iters=0,
            nsamples=1,
            seqlen=16,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q2_k_s")
        gguf_file = os.listdir(quantized_model_path)[0]
        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        ffn_down_type = None
        for tensor in gguf_model.tensors:
            if tensor.name == "blk.0.ffn_down.weight":
                ffn_down_type = tensor.tensor_type.name
                break
        assert ffn_down_type is not None, "blk.0.ffn_down.weight not found in GGUF file"
        assert ffn_down_type == "Q4_K", f"Expected Q4_K for blk.0.ffn_down.weight but got {ffn_down_type}"
        shutil.rmtree(tiny_model_path, ignore_errors=True)

    @pytest.mark.skip_ci(reason="Only tiny model is suggested for CI")
    def test_gguf_baseline(self):
        model_name = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        autoround = AutoRound(
            model_name,
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
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="fake")
        eval_generated_prompt(quantized_model_path)
