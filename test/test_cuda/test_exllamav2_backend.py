import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.testing_utils import require_autogptq, require_gptqmodel, require_package_version_ut

from ..helpers import model_infer


class TestAutoRoundexllamaBackend:

    @classmethod
    def setup_class(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gptqmodel
    def test_gptqmodel_exllmav2_4bits_asym(self, dataloader):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=1, seqlen=2, dataset=dataloader
        )
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:gptqmodel")

        quantization_config = AutoRoundConfig(backend="gptqmodel:exllamav2")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.35)
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.35)
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)

    @require_autogptq
    @require_package_version_ut("torch", "<2.6.0")
    def test_gptq_exllamav2_4bits_sym(self, dataloader):
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
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")  ##will convert to gptq model

        quantization_config = AutoRoundConfig(backend="gptq:exllamav2")  ## or exllamav2
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.27)
        torch.cuda.empty_cache()
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @require_autogptq
    @require_package_version_ut("torch", "<2.6.0")
    def test_gptq_exllamav2_4bits_sym_group_size(self):
        for group_size in [-1, 32, 64, 128, 256, 1024]:  ## 384, 768 has accuracy issue
            print(f"!!!!!!!!!!!!!!!!!{group_size}!!!!!!!!!!!!!!!!!")
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            bits, group_size, sym = 4, group_size, True
            autoround = AutoRound(
                model,
                tokenizer,
                bits=bits,
                iters=1,
                nsamples=1,
                group_size=group_size,
                sym=sym,
            )
            quantized_model_path = self.save_folder
            autoround.quantize_and_save(
                output_dir=quantized_model_path, format="auto_round"
            )  ##will convert to gptq model

            quantization_config = AutoRoundConfig(backend="gptq:exllamav2")  ## or exllamav2
            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            model_infer(model, tokenizer)
            result = simple_evaluate_user_model(model, tokenizer, batch_size=64, tasks="lambada_openai")
            print(result["results"]["lambada_openai"]["acc,none"])
            self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.15)
            torch.cuda.empty_cache()
            shutil.rmtree(self.save_folder, ignore_errors=True)
