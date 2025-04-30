import copy
import shutil
import sys
import unittest

sys.path.insert(0, "..")
from auto_round.eval.evaluation import simple_evaluate_user_model

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round import AutoRoundConfig


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "facebook/opt-125m"

        self.llm_dataloader = LLMDataLoader()
        self.save_folder = "./saved"

    def model_infer(self, model, tokenizer):
        prompts = [
            "Hello,my name is",
            # "The president of the United States is",
            # "The capital of France is",
            # "The future of AI is",
        ]

        ##texts = []
        # for prompt in prompts:
        #     messages = [
        #         {"role": "user", "content": prompt}
        #     ]
        #     text = tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        #     texts.append(text)

        inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

        outputs = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            do_sample=False,  ## change this to follow official usage
            max_new_tokens=5
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)
        ]

        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, prompt in enumerate(prompts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {decoded_outputs[i]}")
            print("-" * 50)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autoround_asym(self):
        for bits in [2, 4, 8]:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            bits, group_size, sym = bits, 128, False
            autoround = AutoRound(
                model,
                tokenizer,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
            )
            quantized_model_path = self.save_folder

            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto",
                                                         trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
            print(res)
            assert ("!!!" not in res)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_mixed_precision(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        layer_config = {}

        layer_config["model.decoder.layers.0.self_attn.k_proj"] = {"bits": 8}
        layer_config["model.decoder.layers.2.self_attn.q_proj"] = {"bits": 3,
                                                                   "group_size": 64}  ## 3bits when using asym will have some issue
        layer_config["model.decoder.layers.6.self_attn.out_proj"] = {"bits": 2, "group_size": 32}
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            layer_config=layer_config
        )
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        quantization_config = AutoRoundConfig(backend="auto")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        self.model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result['results']['lambada_openai']['acc,none'])
        self.assertGreater(result['results']['lambada_openai']['acc,none'], 0.32)

    def test_awq_backend(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            iters=1,
            nsamples=1,
            sym=sym,
        )
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")

        quantization_config = AutoRoundConfig(backend="auto")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        self.model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result['results']['lambada_openai']['acc,none'])
        self.assertGreater(result['results']['lambada_openai']['acc,none'], 0.18)
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        self.model_infer(model, tokenizer)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_tritonv2_bf16(self):
        model_name = "/data5/wenhuach/Meta-Llama-3.1-8B-Instruct-int4-sym-inc"
        quantization_config = AutoRoundConfig(backend="tritonv2")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_infer(model, tokenizer)

        torch.cuda.empty_cache()

    def test_autoround_gptq_sym_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"

        autoround.quantize_and_save(output_dir=quantized_model_path)

        from auto_round import AutoRoundConfig
        quantization_config = AutoRoundConfig(backend="ipex_gptq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True,
                                                     quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True,
                                                     quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)

        shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_awq_sym_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"

        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)

        shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_sym(self):
        for bits in [2, 3, 4, 8]:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            bits, group_size, sym = bits, 128, True
            autoround = AutoRound(
                model,
                tokenizer,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
            )
            quantized_model_path = "./saved"

            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto",
                                                         trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
            print(res)
            assert ("!!!" not in res)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_load_gptq_model_3bits(self):
        model_name = "LucasSantiago257/gemma-2b-2bits-gptq"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True,
                                                     device_map="auto",
                                                     quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model_infer(model, tokenizer)

    def test_autoround_asym(self):
        for bits in [2, 3, 4, 8]:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            bits, group_size, sym = bits, 128, False
            autoround = AutoRound(
                model,
                tokenizer,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
            )
            quantized_model_path = self.save_folder

            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0",
                                                         trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
            print(res)
            assert ("!!!" not in res)
            shutil.rmtree(self.save_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
