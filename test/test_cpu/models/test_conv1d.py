import copy
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...helpers import lamini_name_or_path, model_infer


class TestQuantizationConv1d:
    @classmethod
    def setup_class(self):
        self.model_name = lamini_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")

    def test_quant(self, dataloader):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )

        autoround.quantize()
        _, quantized_model_path = autoround.save_quantized(self.save_dir, return_folders=True)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True)
        model_infer(model, self.tokenizer)


def test_find_layers_from_config_lamini():
    from auto_round.utils.model import find_layers_from_config

    res = find_layers_from_config(lamini_name_or_path, class_names="Conv1d")
    print(res)
    assert "Conv1D" in res, "Conv1D should be detected in the model config"
    assert "h.0.attn.c_attn" in res["Conv1D"], "Conv1D should be detected in the model config with correct prefix"
    assert len(res["Conv1D"]) == 48, "At least one Conv1D layer should be detected in the model config"
