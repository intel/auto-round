import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ....helpers import get_model_path, save_tiny_model


class TestGGUFVLM:

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_vlm_gguf(self):
        model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
        tiny_model_path = save_tiny_model(model_name, "./tmp/tiny_qwen_vl_model_path", num_layers=3, is_mllm=True)
        from auto_round import AutoRoundMLLM

        autoround = AutoRoundMLLM(
            tiny_model_path,
            iters=0,
            nsamples=8,
            disable_opt_rtn=True,
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
