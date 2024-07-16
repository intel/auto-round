import shutil
import sys
import os
import unittest
sys.path.insert(0, "..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round.low_cpu_mem.utils import (
    load_model_with_hooks,
    load_empty_model,
    get_layers_before_block,
    layer_wise_load,
    layer_wise_save,
    )

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)

class TestLowCPUMem(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "facebook/opt-125m"
        self.saved_path = './test_tmp_saved'
        self.ori_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = load_model_with_hooks(self.model_name, AutoModelForCausalLM, saved_path=self.saved_path, device='cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.saved_path, ignore_errors=True)

    def test_default(self):
        self.assertTrue(self.model.device.type, 'meta')

        layers = get_layers_before_block(self.model)
        self.assertEqual(layers[0][0], 'model.decoder.embed_tokens')

        # test get_weight bias
        self.assertTrue(torch.equal(
            self.model.model.decoder.layers[0].self_attn.k_proj.get_weight(),
            self.ori_model.model.decoder.layers[0].self_attn.k_proj.weight,
        ))
        self.assertTrue(torch.equal(
            self.model.model.decoder.layers[0].self_attn.k_proj.get_bias(),
            self.ori_model.model.decoder.layers[0].self_attn.k_proj.bias,
        ))

        # test hooks
        text = ["Hello, my dog is cute"]
        input = self.tokenizer(text)
        for key in input:
            input[key] = torch.tensor(input[key])
        ori_output = self.ori_model(**input)
        output = self.model(**input)
        self.assertTrue(torch.equal(ori_output[0], output[0]))

        # test save and load
        layer_wise_save(self.model, self.saved_path)
        state_dict = layer_wise_load(self.saved_path)
        self.assertTrue(torch.equal(
            state_dict['lm_head.weight'],
            self.ori_model.lm_head.weight
        ))

        # test layer-wise auto_round
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            device='cpu',
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

        # test block-wise auto_round
        self.model = load_empty_model(self.model_name, AutoModelForCausalLM, saved_path=self.saved_path, device='cpu')
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            device='cpu',
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

if __name__ == "__main__":
    unittest.main()