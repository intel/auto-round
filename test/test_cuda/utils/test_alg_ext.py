import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model

from ...helpers import get_model_path

AUTO_ROUND_PATH = __file__.split("/")
AUTO_ROUND_PATH = "/".join(AUTO_ROUND_PATH[: AUTO_ROUND_PATH.index("test")])


class TestAlgExt:
    save_folder = "./saved"

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

    def test_2bits(self):
        model_name = get_model_path("facebook/opt-125m")
        ar = AutoRound(model=model_name, bits=2, group_size=64, enable_alg_ext=True)
        ar.quantize_and_save(self.save_folder)
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=64, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        # wo alg ext 0.2078, with 0.2371
        assert result["results"]["lambada_openai"]["acc,none"] > 0.22
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_cli(self, tiny_opt_model_path):
        import os

        python_path = sys.executable

        res = os.system(
            f"PYTHONPATH='AUTO_ROUND_PATH:$PYTHONPATH' CUDA_VISIBLE_DEVICES=0 {python_path} -m auto_round --model {tiny_opt_model_path} --iters 1 --device auto --enable_alg_ext --avg_bits 2 --options=W2A16,W4A16 --ignore_scale_zp_bits --nsamples 1 --seqlen 32"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"PYTHONPATH='AUTO_ROUND_PATH:$PYTHONPATH' CUDA_VISIBLE_DEVICES=0 {python_path} -m auto_round --model {tiny_opt_model_path} --iters 1 --device auto --enable_alg_ext --avg_bits 5.5 --options=mxfp4,mxfp8 --ignore_scale_zp_bits --enable_torch_compile --nsamples 1 --seqlen 32"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_all_support_dtype(self, tiny_qwen_model_path):
        from auto_round.auto_scheme import AutoScheme

        for scheme in ["MXFP4", "NVFP4", "W2A16G64", "gguf:q2_k_s,gguf:q4_k_s"]:
            avg_bits = 2 if scheme == "W2A16G64" else 4
            scheme = AutoScheme(options=scheme, avg_bits=avg_bits, ignore_scale_zp_bits=True)
            ar = AutoRound(
                tiny_qwen_model_path,
                scheme=scheme,
                iters=1,
                nsamples=1,
                seqlen=32,
                enable_alg_ext=True,
                enable_torch_compile=True,
            )
            ar.quantize()
