import json
import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class TestLocalCalibDataset:
    @classmethod
    def setup_class(self):
        json_data = [{"text": "awefdsfsddfd"}, {"text": "fdfdfsdfdfdfd"}, {"text": "dfdsfsdfdfdfdf"}]
        os.makedirs("./saved", exist_ok=True)
        self.json_file = "./saved/tmp.json"
        with open(self.json_file, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        jsonl_data = [{"text": "哈哈，開心點"}, {"text": "hello world"}]
        os.makedirs("./saved", exist_ok=True)
        self.jsonl_file = "./saved/tmp.jsonl"
        with open(self.jsonl_file, "w") as jsonl_file:
            for item in jsonl_data:
                json.dump(item, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")

        model_name = "facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def test_combine_dataset(self):
        dataset = "NeelNanda/pile-10k" + ",BAAI/CCI3-HQ" + ",madao33/new-title-chinese"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2, seqlen=128, dataset=dataset
        )
        autoround.quantize()
