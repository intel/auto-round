import os
import shutil
import sys
import unittest

from parameterized import parameterized

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import require_gptqmodel


def _get_folder_size(path: str) -> float:
    """Return folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # convert to GB


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gptqmodel
    def test_mixed_gptqmodel(self):
        bits, sym, group_size = 4, True, 128
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        layer_config = {
            "k_proj": {"bits": 8},
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
        from gptqmodel import GPTQModel

        model = GPTQModel.load(quantized_model_path)
        assert model.model.model.decoder.layers[0].self_attn.k_proj.bits == 8
        assert model.model.model.decoder.layers[0].self_attn.q_proj.bits == 4
        result = model.generate("Uncovering deep insights begins with")[0]  # tokens
        assert "!!!" not in model.tokenizer.decode(result)  # string output
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_autoround_format(self):
        bits, sym, group_size = 4, True, 128
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        layer_config = {
            "k_proj": {"bits": 8},
            "q_proj": {"bits": 3},
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        assert model.model.decoder.layers[0].self_attn.k_proj.bits == 8
        assert model.model.decoder.layers[0].self_attn.q_proj.bits == 3
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
