import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.testing_utils import require_autogptq, require_gptqmodel, require_package_version_ut

from ...helpers import get_model_path, model_infer


class TestAutoRoundexllamaBackend:
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

    @require_gptqmodel
    def test_gptqmodel_exllmav2_4bits_asym(self, dataloader):
        model_path = get_model_path("facebook/opt-125m")
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, dataset=dataloader
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:gptqmodel")

        quantization_config = AutoRoundConfig(backend="gptqmodel:exllamav2")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_dir, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.35
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            self.save_dir, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.35
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)

    @require_autogptq
    @require_package_version_ut("torch", "<2.6.0")
    def test_gptq_exllamav2_4bits_sym(self, dataloader):
        model_path = get_model_path("facebook/opt-125m")
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")  ##will convert to gptq model

        quantization_config = AutoRoundConfig(backend="gptq:exllamav2")  ## or exllamav2
        model = AutoModelForCausalLM.from_pretrained(
            self.save_dir, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.27
        torch.cuda.empty_cache()
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_autogptq
    @require_package_version_ut("torch", "<2.6.0")
    def test_gptq_exllamav2_4bits_sym_group_size(self):
        model_path = get_model_path("facebook/opt-125m")
        for group_size in [-1, 32, 64, 128, 256, 1024]:  ## 384, 768 has accuracy issue
            print(f"!!!!!!!!!!!!!!!!!{group_size}!!!!!!!!!!!!!!!!!")
            bits, group_size, sym = 4, group_size, True
            autoround = AutoRound(
                model_path,
                bits=bits,
                iters=1,
                nsamples=1,
                group_size=group_size,
                sym=sym,
            )
            quantized_model_path = self.save_dir
            autoround.quantize_and_save(
                output_dir=quantized_model_path, format="auto_round"
            )  ##will convert to gptq model

            quantization_config = AutoRoundConfig(backend="gptq:exllamav2")  ## or exllamav2
            model = AutoModelForCausalLM.from_pretrained(
                self.save_dir, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            model_infer(model, tokenizer)
            result = simple_evaluate_user_model(model, tokenizer, batch_size=64, tasks="lambada_openai")
            print(result["results"]["lambada_openai"]["acc,none"])
            assert result["results"]["lambada_openai"]["acc,none"] > 0.15
            torch.cuda.empty_cache()
            shutil.rmtree(self.save_dir, ignore_errors=True)
