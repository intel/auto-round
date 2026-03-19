import json
import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import get_model_path, opt_name_or_path


class TestLocalCalibDataset:
    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        os.makedirs(self.save_dir, exist_ok=True)

        json_data = [{"text": "awefdsfsddfd"}, {"text": "fdfdfsdfdfdfd"}, {"text": "dfdsfsdfdfdfdf"}]
        self.json_file = os.path.join(self.save_dir, "tmp.json")
        with open(self.json_file, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        jsonl_data = [{"text": "哈哈，開心點"}, {"text": "hello world"}]
        self.jsonl_file = os.path.join(self.save_dir, "tmp.jsonl")
        with open(self.jsonl_file, "w") as jsonl_file:
            for item in jsonl_data:
                json.dump(item, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")

        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    def test_json(self, tiny_opt_model_path):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=5,
            dataset=self.json_file,
        )
        autoround.quantize()

    def test_jsonl(self, tiny_opt_model_path):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=4,
            dataset=self.jsonl_file,
        )
        autoround.quantize()

    def test_apply_chat_template(self, tiny_qwen_model_path):
        model_name = tiny_qwen_model_path
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

    def test_combine_dataset(self, tiny_qwen_model_path):
        dataset = "NeelNanda/pile-10k" + "," + "madao33/new-title-chinese" + "," + "mbpp"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_qwen_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=128,
            dataset=dataset,
            nsamples=1,
        )
        autoround.quantize()

    def test_combine_dataset2(self, tiny_opt_model_path):
        dataset = "NeelNanda/pile-10k:num=256,mbpp:num=256"
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=128,
            dataset=dataset,
            nsamples=1,
        )
        autoround.quantize()
