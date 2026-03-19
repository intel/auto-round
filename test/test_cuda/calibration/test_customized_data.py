import copy
import re
import shutil
import sys

import pytest
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path


class TestCustomizedData:

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    def test_list_batch_encoding(self, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        texts = [
            "There is a girl who likes adventure,",
            "Tell me a story about a brave robot,",
            "Explain why the sky is blue,",
        ]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=9, return_tensors="pt")

        ar = AutoRound(model_name, dataset=[inputs], seqlen=9)
        ar.quantize()

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    def test_mixed_attention_mask(self):
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        texts = [
            "There is a girl who likes adventure,",
            "Tell me a story about a brave robot,",
            "Explain why the sky is blue,",
        ]

        inputs = tokenizer(texts, padding=True, truncation=True, max_length=9, return_tensors="pt")

        inputs = inputs["input_ids"]
        inputs = inputs.split(dim=0, split_size=1)
        ar = AutoRound(model_name, dataset=inputs, seqlen=9)
        ar.quantize()

    @pytest.mark.skip_ci(reason="Only tiny model is suggested")
    def test_batch_encoding(self):
        model_name = get_model_path("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        texts = [
            "There is a girl who likes adventure,",
            "Tell me a story about a brave robot,",
            "Explain why the sky is blue,",
        ]

        inputs = tokenizer(texts, padding=True, truncation=True, max_length=9, return_tensors="pt").to(model.device)

        ar = AutoRound(model_name, dataset=inputs, seqlen=9)
        ar.quantize()
