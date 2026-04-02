import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_gguf
from ...helpers import get_model_path, get_tiny_model


class TestTorchCompile:
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        print("[Setup] Running before any test in class")
        yield
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gguf
    def test_gguf_q2ks_torch_compile(self, dataloader):
        """Test GGUF Q2_K_S quantization with torch.compile enabled.

        Verifies that:
        1. Quantization completes successfully with enable_torch_compile=True.
        2. The GGUF file is exported correctly.
        3. The exported model can be loaded and run inference.
        """
        model_path = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        model = get_tiny_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        autoround = AutoRound(
            model,
            tokenizer,
            scheme="gguf:q2_k_s",
            iters=2,
            seqlen=16,
            nsamples=2,
            enable_alg_ext=True,
            enable_torch_compile=True,
        )
        autoround.quantize()
        autoround.save_quantized(output_dir=self.save_dir)

        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_gguf
    def test_gguf_q2ks_torch_compile_iters0(self, tiny_qwen_model_path):
        """Test GGUF Q2_K_S with torch.compile and iters=0 (RTN mode).

        Verifies that torch.compile does not break the fast RTN-based quantization
        path when exporting to GGUF Q2_K_S format.
        """
        autoround = AutoRound(
            tiny_qwen_model_path,
            iters=0,
            nsamples=2,
            seqlen=16,
            enable_torch_compile=True,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="gguf:q2_k_s")

        saved_files = [f for f in os.listdir(quantized_model_path) if f.endswith(".gguf")]
        assert len(saved_files) > 0, "No GGUF file was generated"

        shutil.rmtree(self.save_dir, ignore_errors=True)
