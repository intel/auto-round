import copy
import shutil

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ..helpers import get_model_path


class TestAutoRoundXPU:
    @classmethod
    def setup_class(self):
        self.device = "xpu"
        pass

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        pass

    def test_gptq_format(self, dataloader):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True

        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path)
        quantized_model_path = quantized_model_path[0]

        quantization_config = AutoRoundConfig(backend="auto")

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res

    def test_awq_format(self, dataloader):
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True, device_map=self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = "./saved"
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:auto_awq"
        )
        quantized_model_path = quantized_model_path[0]

        quantization_config = AutoRoundConfig(backend="auto")
        # device_map="auto" doesn't work, must use "xpu"
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=self.device, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert "!!!" not in res
