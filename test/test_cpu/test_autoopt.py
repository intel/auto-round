import copy
import shutil

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRoundAdam


class TestAutoRound:

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

    def test_Adam(self, tiny_opt_model, tokenizer, dataloader):
        bits, group_size, sym = 4, 128, False
        from auto_round.utils import get_block_names

        llm_block_names = get_block_names(tiny_opt_model, quant_vision=True)
        bits, group_size, sym, batch_size = 4, 128, False, 20
        adamround = AutoRoundAdam(
            tiny_opt_model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            batch_size=batch_size,
            dataset=dataloader,
            to_quant_block_names=llm_block_names,
        )
        adamround.quantize()
