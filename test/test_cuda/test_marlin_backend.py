import shutil
import sys
import unittest

import pytest

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRoundMarlinBackend(unittest.TestCase):

    def test_marlin_group_size(self):
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
                dataset=self.llm_dataloader,
            )
            quantized_model_path = self.save_folder
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

            quantization_config = AutoRoundConfig(backend="marlin")
            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            self.model_infer(model, tokenizer)
            result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
            print(result["results"]["lambada_openai"]["acc,none"])
            self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.14)

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
                dataset=self.llm_dataloader,
            )
            quantized_model_path = self.save_folder
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            quantization_config = AutoRoundConfig(backend="marlin")
            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            self.model_infer(model, tokenizer)
            result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
            print(result["results"]["lambada_openai"]["acc,none"])
            self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.14)

    @classmethod
    def setUpClass(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"
        self.llm_dataloader = LLMDataLoader()

    def model_infer(self, model, tokenizer):
        prompts = [
            "Hello,my name is",
            # "The president of the United States is",
            # "The capital of France is",
            # "The future of AI is",
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            do_sample=False,  ## change this to follow official usage
            max_new_tokens=5,
        )
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]

        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, prompt in enumerate(prompts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {decoded_outputs[i]}")
            print("-" * 50)
        return decoded_outputs[0]

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_marlin_4bits_sym_with_zp_m_1(self):
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
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

        quantization_config = AutoRoundConfig(backend="marlin")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        self.model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.27)
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        self.model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.27)
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
    #         dataset=self.llm_dataloader,
    #     )
    #     quantized_model_path = self.save_folder
    #     autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
    #
    #     quantization_config = AutoRoundConfig(backend="marlin")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         self.save_folder,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #         quantization_config=quantization_config
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
    #     self.model_infer(model, tokenizer)
    #     result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
    #     print(result['results']['lambada_openai']['acc,none'])
    #     self.assertGreater(result['results']['lambada_openai']['acc,none'], 0.27)
    #     torch.cuda.empty_cache()
    #
    #     model = AutoModelForCausalLM.from_pretrained(
    #         self.save_folder,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         quantization_config=quantization_config
    #     )
    #
    #     tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
    #     self.model_infer(model, tokenizer)
    #     result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
    #     print(result['results']['lambada_openai']['acc,none'])
    #     self.assertGreater(result['results']['lambada_openai']['acc,none'], 0.27)
    #     torch.cuda.empty_cache()
    #     shutil.rmtree("./saved", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
