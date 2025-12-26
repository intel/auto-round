import json
import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ..helpers import get_model_path, opt_name_or_path


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

        model_name = opt_name_or_path
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

    def test_jsonl(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=4,
            dataset=self.jsonl_file,
        )
        autoround.quantize()

    def test_apply_chat_template(self):
        model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dataset = "NeelNanda/pile-10k:apply_chat_template:system_prompt=''"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=128,
            dataset=dataset,
            nsamples=1,
        )
        autoround.quantize()

    def test_combine_dataset(self):
        dataset = "NeelNanda/pile-10k" + "," + "madao33/new-title-chinese" + "," + "mbpp"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=128,
            dataset=dataset,
            nsamples=1,
        )
        autoround.quantize()

    def test_combine_dataset2(self):
        dataset = "NeelNanda/pile-10k:num=256,mbpp:num=256"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=128,
            dataset=dataset,
            nsamples=1,
        )
        autoround.quantize()

    # def test_pile_val_backup_dataset(self):
    #     dataset = "swift/pile-val-backup"
    #     bits, group_size, sym = 4, 128, True
    #     autoround = AutoRound(
    #         self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2, seqlen=128, dataset=dataset
    #     )
    #     autoround.quantize()

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
