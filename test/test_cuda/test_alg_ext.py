import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model


class TestAlgExt(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_2bits(self):
        model_name = "/models/opt-125m"
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
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.22)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_cli(self):
        import os

        model_name = "/models/opt-125m"
        python_path = sys.executable

        res = os.system(
            f"cd ../.. && CUDA_VISIBLE_DEVICES=0 {python_path} -m auto_round --model {model_name} --iters 1 --device auto --enable_alg_ext --avg_bits 2 --options=W2A16,W4A16 --ignore_scale_zp_bits"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

        res = os.system(
            f"cd ../.. && CUDA_VISIBLE_DEVICES=0 {python_path} -m auto_round --model {model_name} --iters 1 --device auto --enable_alg_ext --avg_bits 5.5 --options=mxfp4,mxfp8 --ignore_scale_zp_bits --enable_torch_compile"
        )
        if res > 0 or res == -1:
            assert False, "cmd line test fail, please have a check"

    def test_all_support_dtype(self):
        from auto_round.auto_scheme import AutoScheme

        model_name = "/models/Qwen3-0.6B"
        for scheme in ["MXFP4", "NVFP4", "W2A16G64", "gguf:q4_k_s"]:
            avg_bits = 2 if scheme == "W2A16G64" else 4
            scheme = AutoScheme(options=[scheme], avg_bits=avg_bits, ignore_scale_zp_bits=True)
            ar = AutoRound(
                model_name, scheme=scheme, iters=1, nsamples=1, enable_alg_ext=True, enable_torch_compile=True
            )
            ar.quantize()


if __name__ == "__main__":
    unittest.main()
