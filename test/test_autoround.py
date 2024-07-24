import copy
import shutil
import sys
import unittest

sys.path.insert(0, ".")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        
    def test_block(self):
        
        def feak_input():
            input_ids = torch.randn((2, 10, 768))
            positional_inputs = []
            attention_mask = torch.randn((2, 1, 10, 10))
            input_others = {"positional_inputs":positional_inputs, "attention_mask": attention_mask}
            return input_ids, input_others

        block = self.model.model.decoder.layers[0]
        block.dtype = next(block.parameters()).dtype
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            block,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            dataloader=self.llm_dataloader,
            use_quant_input=False, # disable it for now
            n_samples=2, # double-check it
            amp=False,
        )
        input_ids, input_others = feak_input()
        _, output = autoround.quant_block(block, input_ids=input_ids, input_others=input_others)

    def test_default(self):
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()
        if torch.cuda.is_available():
            autoround.save_quantized(output_dir="./saved", inplace=False)
        autoround.save_quantized(output_dir="./saved", inplace=False, format="itrex")

    def test_sym(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w4g1(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w3g128(self):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w2g128(self):
        bits, group_size, sym = 2, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()

    def test_disable_use_quant_input(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            use_quant_input=False,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()

    def test_disable_minmax_tuning(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            enable_minmax_tuning=False,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()

    def test_signround(self):
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            enable_minmax_tuning=False,
            use_quant_input=False,
            dataloader=self.llm_dataloader,
        )
        autoround.quantize()


if __name__ == "__main__":
    unittest.main()
