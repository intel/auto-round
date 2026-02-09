import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...helpers import evaluate_accuracy, get_model_path, model_infer


class TestAutoRoundMarlinBackend:
    save_dir = "./saved"
    model_name = get_model_path("facebook/opt-125m")

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

    def test_marlin_group_size(self, dataloader):
        for group_size in [-1, 64]:
            print(f"{group_size}!!!!!!!!!!!!!!!!!")
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            bits, group_size, sym = 4, group_size, True
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
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

            quantization_config = AutoRoundConfig(backend="marlin")
            model = AutoModelForCausalLM.from_pretrained(
                self.save_dir, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            model_infer(model, tokenizer)
            evaluate_accuracy(model, tokenizer, threshold=0.14, batch_size=16)

        for group_size in [32, 128]:
            print(f"{group_size}!!!!!!!!!!!!!!!!!")
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            bits, group_size, sym = 4, group_size, True
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
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            quantization_config = AutoRoundConfig(backend="marlin")
            model = AutoModelForCausalLM.from_pretrained(
                self.save_dir, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            model_infer(model, tokenizer)
            evaluate_accuracy(model, tokenizer, threshold=0.14, batch_size=16)

    def test_marlin_4bits_sym_with_zp_m_1(self, dataloader):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
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
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

        quantization_config = AutoRoundConfig(backend="marlin")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_dir, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.27, batch_size=16)
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            self.save_dir, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        model_infer(model, tokenizer)
        evaluate_accuracy(model, tokenizer, threshold=0.27, batch_size=16)
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)

    # def test_marlin_4bits_sym(self):
    #     model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    #     bits, group_size, sym = 4, 128, True
    #     autoround = AutoRound(
    #         model,
    #         tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         iters=1,
    #         seqlen=2,
    #         dataset=dataloader,
    #     )
    #     quantized_model_path = self.save_dir
    #     autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
    #
    #     quantization_config = AutoRoundConfig(backend="marlin")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         self.save_dir,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #         quantization_config=quantization_config
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
    #     model_infer(model, tokenizer)
    #     result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
    #     print(result['results']['lambada_openai']['acc,none'])
    #     assert result['results']['lambada_openai']['acc,none'] > 0.27
    #     torch.cuda.empty_cache()
    #
    #     model = AutoModelForCausalLM.from_pretrained(
    #         self.save_dir,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         quantization_config=quantization_config
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
    #     model_infer(model, tokenizer)
    #     result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
    #     print(result['results']['lambada_openai']['acc,none'])
    #     assert result['results']['lambada_openai']['acc,none'] > 0.27
    #     torch.cuda.empty_cache()
    #     shutil.rmtree("./saved", ignore_errors=True)
