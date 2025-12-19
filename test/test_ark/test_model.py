import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.testing_utils import require_autogptq, require_gptqmodel

from ..helpers import model_infer


class TestAutoRoundTorchBackend:

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

    def test_torch_4bits_sym_cpu(self, opt_model, opt_tokenizer, dataloader):
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            opt_model,
            opt_tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:gptqmodel")

        quantization_config = AutoRoundConfig(backend="ark")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=torch.float16, device_map="cpu", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=32, tasks="lambada_openai", limit=1000)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.28

        shutil.rmtree("./saved", ignore_errors=True)

    def test_torch_4bits_sym_xpu(self, opt_model, opt_tokenizer, dataloader):
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            opt_model,
            opt_tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")  ##will convert to gptq model

        quantization_config = AutoRoundConfig(backend="ark")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=torch.float16, device_map="xpu", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=32, tasks="lambada_openai", limit=1000)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.28
        torch.xpu.empty_cache()
        shutil.rmtree(self.save_folder, ignore_errors=True)
