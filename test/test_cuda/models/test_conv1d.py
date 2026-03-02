import copy
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ...envs import require_gptqmodel
from ...helpers import get_model_path, get_tiny_model, model_infer


class TestQuantizationConv1d:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gptqmodel
    def test_quant(self, dataloader):
        model_name = get_model_path("MBZUAI/LaMini-GPT-124M")
        model = get_tiny_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        from transformers import AutoRoundConfig

        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )

        autoround.quantize()
        autoround.save_quantized("./saved")

        model = AutoModelForCausalLM.from_pretrained("./saved", device_map="cuda", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("./saved", trust_remote_code=True)
        model_infer(model, tokenizer)
