import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ....helpers import get_model_path


class TestGGUFBaseline:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf_baseline(self):
        model_name = get_model_path("Qwen/Qwen2.5-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=3,
            group_size=16,
            sym=True,
            iters=0,
            nsamples=8,
            seqlen=2,
            data_type="rtn_int_sym_dq",
            super_group_size=16,
            super_bits=6,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="fake")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        text = "There is a girl who likes adventure,"
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        print(self.tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        shutil.rmtree("./saved", ignore_errors=True)
