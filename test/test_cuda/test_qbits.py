import shutil
import sys
import unittest

sys.path.insert(0, "../..")

from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.testing_utils import require_gptqmodel, require_itrex


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"

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

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("runs", ignore_errors=True)

    ## require torch 2.6
    @require_itrex
    def test_load_gptq_model_8bits(self):
        model_name = "acloudfan/opt-125m-gptq-8bit"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
            device_map="cpu",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model_infer(model, tokenizer)

    @require_itrex
    def test_load_gptq_model_2bits(self):
        model_name = "LucasSantiago257/gemma-2b-2bits-gptq"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
            device_map="cpu",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model_infer(model, tokenizer)

    @require_itrex
    def test_mixed_precision(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        layer_config = {}

        layer_config["model.decoder.layers.0.self_attn.k_proj"] = {"bits": 8}
        layer_config["model.decoder.layers.6.self_attn.out_proj"] = {"bits": 2, "group_size": 32}
        bits, group_size, sym = 4, 128, True
        import torch

        from auto_round import AutoRound

        autoround = AutoRound(
            model, tokenizer, bits=bits, group_size=group_size, iters=1, nsamples=1, sym=sym, layer_config=layer_config
        )
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder,
            dtype=torch.float16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @require_gptqmodel
    def test_autoround_sym(self):
        for bits in [4]:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            bits, group_size, sym = bits, 128, True
            autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2, seqlen=2)
            quantized_model_path = "./saved"

            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="auto", trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
            print(res)
            assert "!!!" not in res
            shutil.rmtree(self.save_folder, ignore_errors=True)
