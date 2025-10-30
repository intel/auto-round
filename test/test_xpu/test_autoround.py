import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoRoundConfig


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(3):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRoundXPU(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        pass

    def test_gptq_format(self):
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", trust_remote_code=True, device_map="auto"
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
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path)

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

    def test_awq_format(self):
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", trust_remote_code=True, device_map="xpu"
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
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")

        quantized_model_path = "./saved"
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


if __name__ == "__main__":
    unittest.main()
