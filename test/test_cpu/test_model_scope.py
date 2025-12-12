import copy
import os
import shutil
import sys
import unittest

sys.path.insert(0, "../..")

import torch

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(3):
            yield torch.ones([1, 10], dtype=torch.long)


class TestModelScope(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.saved_path = "./saved"
        self.dataset = LLMDataLoader()

        self.source_path, self.cache_path = "/tf_dataset/auto_round/modelscope", "/home/hostuser/.cache/modelscope"
        if os.path.exists(self.source_path):
            if not os.path.exists("/home/hostuser/.cache"):
                os.makedirs("/home/hostuser/.cache")
            shutil.copytree(self.source_path, self.cache_path, dirs_exist_ok=True)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path, ignore_errors=True)

        return super().tearDownClass()

    def test_llm(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        autoround = AutoRound(
            model_name, platform="model_scope", scheme="w4a16", iters=0, seqlen=2, dataset=self.dataset
        )
        autoround.quantize_and_save()

    def test_mllm(self):
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        autoround = AutoRound(
            model_name, platform="model_scope", scheme="w4a16", iters=0, seqlen=2, dataset=self.dataset, batch_size=2
        )
        autoround.quantize_and_save(self.saved_path)


if __name__ == "__main__":
    unittest.main()
