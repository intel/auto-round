import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.testing_utils import require_autogptq, require_gptqmodel

from ...helpers import get_model_path, model_infer


class TestAutoRoundTorchBackend:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("facebook/opt-125m")
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_torch_4bits_asym(self, dataloader):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:gptqmodel"
        )

        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=torch.float16, device_map="cpu", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai", limit=10)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.35
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder, dtype=torch.bfloat16, device_map="cpu", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai", limit=10)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.35
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)

    def test_torch_4bits_sym(self, dataloader):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 32, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=0,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round"
        )  ##will convert to gptq model

        quantization_config = AutoRoundConfig(backend="torch")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=32, tasks="lambada_openai", limit=1000)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.28
        torch.cuda.empty_cache()
        shutil.rmtree(quantized_model_path, ignore_errors=True)
