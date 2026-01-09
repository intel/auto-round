import copy
import os
import shutil

import pytest
import torch

from auto_round import AutoRound

from ...helpers import get_model_path


class TestModelScope:
    @classmethod
    def setup_class(self):
        self.saved_path = "./saved"

        self.source_path, self.cache_path = "/tf_dataset/auto_round/modelscope", "/home/hostuser/.cache/modelscope"
        if os.path.exists(self.source_path):
            if not os.path.exists("/home/hostuser/.cache"):
                os.makedirs("/home/hostuser/.cache")
            shutil.copytree(self.source_path, self.cache_path, dirs_exist_ok=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path, ignore_errors=True)

    def test_llm(self, dataloader):
        model_name = get_model_path("Qwen/Qwen2.5-0.5B-Instruct")
        autoround = AutoRound(model_name, platform="model_scope", scheme="w4a16", iters=0, seqlen=2, dataset=dataloader)
        autoround.quantize_and_save()

    def test_mllm(self, dataloader):
        model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
        autoround = AutoRound(
            model_name, platform="model_scope", scheme="w4a16", iters=0, seqlen=2, dataset=dataloader, batch_size=2
        )
        autoround.quantize_and_save(self.saved_path)
