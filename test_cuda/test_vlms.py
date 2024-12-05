import copy
import shutil
import sys
import unittest
import re

sys.path.insert(0, "..")
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.save_dir = "./saved"
        self.tasks = "lambada_openai"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @unittest.skipIf(torch.cuda.is_available() is False, "Skipping because no cuda")
    def test_vision_generation(self):
        quantized_model_path = "OPEA/Phi-3.5-vision-instruct-qvision-int4-sym-inc"
        from auto_round import AutoRoundConfig
        device = "auto"  ##cpu, hpu, cuda
        quantization_config = AutoRoundConfig(
            backend=device
        )
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, trust_remote_code=True,
                                                     device_map=device, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert (
                res == """<s> There is a girl who likes adventure, and she is looking for a partner to go on a treasure hunt. She has found a map that leads to a hidden treasure, but she needs a partner to help her decipher the clues and find the treasure. You""")

if __name__ == "__main__":
    unittest.main()