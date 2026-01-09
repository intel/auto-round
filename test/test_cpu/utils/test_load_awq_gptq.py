import shutil

import pytest
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from ...helpers import get_model_path, model_infer


class TestAutoRound:

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_load_gptq_no_dummy_gidx_model(self):
        model_name = get_model_path("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1")
        quantization_config = AutoRoundConfig()
        with pytest.raises(NotImplementedError):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="cpu",
                quantization_config=quantization_config,
            )

    def test_load_awq(self):
        model_name = get_model_path("casperhansen/opt-125m-awq")
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map="cpu",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_infer(model, tokenizer)
