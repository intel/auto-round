import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_autogptq, require_gptqmodel, require_package_version_ut
from ...helpers import eval_generated_prompt, evaluate_accuracy, generate_prompt, get_model_path, model_infer


class TestAutoRoundexllamaBackend:

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("runs", ignore_errors=True)

    # keep one CI test for exllamav2 backend, since it's the only backend supporting 4bits asym quantization.
    # @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    # @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    @require_gptqmodel
    def test_gptqmodel_exllmav2_4bits_asym(self, dataloader):
        model_path = get_model_path("facebook/opt-125m")
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model_path, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, dataset=dataloader
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:gptqmodel"
        )

        quantization_config = AutoRoundConfig(backend="gptqmodel:exllamav2")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.35, batch_size=16)
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.35, batch_size=16)
        torch.cuda.empty_cache()

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
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
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.27, batch_size=16)
        torch.cuda.empty_cache()

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    @require_autogptq
    @require_package_version_ut("torch", "<2.6.0")
    @pytest.mark.parametrize("group_size", [-1, 32, 64, 128, 256, 1024])
    def test_gptq_exllamav2_4bits_sym_group_size(self, group_size):
        model_path = get_model_path("facebook/opt-125m")
        print(f"!!!!!!!!!!!!!!!!!{group_size}!!!!!!!!!!!!!!!!!")
        autoround = AutoRound(
            model_path,
            bits=4,
            iters=0,
            disable_opt_rtn=True,
            group_size=group_size,
            sym=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round"
        )  ##will convert to gptq model

        quantization_config = AutoRoundConfig(backend="gptq:exllamav2")  ## or exllamav2
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.15, batch_size=64)
        torch.cuda.empty_cache()

    @require_gptqmodel
    def test_gptqmodel_awq_exllamav2_4bits_asym(self, dataloader):
        """Test AWQ quantization with gptqmodel:awq_exllamav2 backend (bfloat16 inference)."""
        model_path = get_model_path("facebook/opt-125m")
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:auto_awq"
        )
        quantization_config = AutoRoundConfig(backend="gptqmodel:awq_exllamav2")
        # test awq bfloat16 inference
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )
        assert model.dtype == torch.bfloat16, f"Expected model dtype bfloat16, got {model.dtype}"

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        # Inference generation check
        eval_generated_prompt(model, tokenizer)
        # Accuracy check
        evaluate_accuracy(model, tokenizer, threshold=0.35, batch_size=16)
        torch.cuda.empty_cache()

    @require_gptqmodel
    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    def test_gptqmodel_awq_exllamav2_4bits_sym(self, dataloader):
        """Test AWQ quantization with gptqmodel:awq_exllamav2 backend (bfloat16 inference, symmetric)."""
        model_path = get_model_path("facebook/opt-125m")
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:auto_awq"
        )

        quantization_config = AutoRoundConfig(backend="gptqmodel:awq_exllamav2")
        model = AutoModelForCausalLM.from_pretrained(  # test awq bfloat16 inference
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        # Inference generation check
        eval_generated_prompt(model, tokenizer)
        # Accuracy check
        evaluate_accuracy(model, tokenizer, threshold=0.2, batch_size=16)
        torch.cuda.empty_cache()
