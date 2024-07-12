import shutil
import sys
import os
import unittest
sys.path.insert(0, '..')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round.layer_wise.utils import (
    load_model_with_hooks,
    get_layers_before_block,
    layer_wise_load,
    layer_wise_save,
    )


class TestLowCPUMem(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"
        self.saved_path = './test_tmp_saved'
        self.ori_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model = load_model_with_hooks(model_name, AutoModelForCausalLM, saved_path=self.saved_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
        ouptut = self.model(**input)
        self.assertTrue(torch.equal(ori_output[0], ouptut[0]))

        # test save and load
        layer_wise_save(self.model, self.saved_path)
        state_dict = layer_wise_load(self.saved_path)
        self.assertTrue(torch.equal(
            state_dict['lm_head.weight'],
            self.ori_model.lm_head.weight
        ))


if __name__ == "__main__":
    unittest.main()