import os
import shutil
import sys
import unittest

sys.path.insert(0, "..")
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestLocalCalibDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        json_data = [{"text": "awefdsfsddfd"}, {"text": "fdfdfsdfdfdfd"}, {"text": "dfdsfsdfdfdfdf"}]
        os.makedirs("./saved", exist_ok=True)
        self.json_file = "./saved/tmp.json"
        with open(self.json_file, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        self.text_file = "./saved/tmp.txt"
        txt_data = ["awefdsfsddfd", "fdfdfsdfdfdfd", "dfdsfsdfdfdfdf"]
        with open(self.text_file, "w") as text_file:
            for data in txt_data:
                text_file.write(data + "\n")

        model_name = "facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def test_json(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=5,
            dataset=self.json_file,
        )
        autoround.quantize()

    def test_txt(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=5,
            dataset=self.text_file,
        )
        autoround.quantize()

    def test_combine_dataset(self):
        dataset = self.text_file + "," + "NeelNanda/pile-10k" + "," + "madao33/new-title-chinese" + "," + "mbpp"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2, seqlen=128, dataset=dataset
        )
        autoround.quantize()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
