import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_autogptq, require_gptqmodel
from ...helpers import evaluate_accuracy, generate_prompt, get_model_path, model_infer


class TestAutoRoundTorchBackend:

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

    # Keep one CI test for torch backend and skip others to save time.
    # @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    # @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    def test_torch_4bits_asym(self, dataloader):
        model_path = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:gptqmodel")

        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.35, batch_size=16)
        torch.cuda.empty_cache()

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    @pytest.mark.skip_ci(reason="Time-consuming; Accuracy evaluation")
    def test_torch_4bits_sym(self, dataloader):
        model_path = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")  ##will convert to gptq model

        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.28, batch_size=16)
        torch.cuda.empty_cache()

    def test_autoround_3bit_asym_torch_format(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 3, 128, False
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round:gptqmodel")

        device = "auto"  ##cpu, hpu, cuda
        from transformers import AutoRoundConfig

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))

    @pytest.mark.skip_ci(reason="Not necessary to test both symmetric and asymmetric for 3-bit quantization in CI")
    def test_autoround_3bit_sym_torch_format(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

        device = "auto"  ##cpu, hpu, cuda
        from transformers import AutoRoundConfig

        quantization_config = AutoRoundConfig(backend=device)
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=device, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @require_gptqmodel
    @pytest.mark.skip_ci(reason="Not necessary to test low priority backend in CI")
    def test_gptqmodel_awq_torch_4bits_group_size_16(self, dataloader):
        """Test AWQ quantization with gptqmodel:awq_torch backend (group_size=16, float16)."""
        model_path = get_model_path("facebook/opt-125m")
        bits, group_size, sym = 4, 16, True
        autoround = AutoRound(
            model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            disable_opt_rtn=True,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")

        quantization_config = AutoRoundConfig(backend="gptqmodel:awq_torch")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_dir, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        # Inference generation check
        output = model_infer(model, tokenizer)
        assert isinstance(output, str) and len(output.strip()) > 0, "Model failed to generate non-empty output"
        generated = generate_prompt(model, tokenizer, "There is a girl who likes adventure,")
        assert len(generated) > len("There is a girl who likes adventure,"), "Generation did not produce new tokens"
        # Accuracy check
        evaluate_accuracy(model, tokenizer, threshold=0.2, batch_size=16)
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)
