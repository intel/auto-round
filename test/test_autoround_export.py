import copy
import shutil
import sys
import unittest

sys.path.insert(0, ".")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class SimpleDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.randn([1, 30])


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoroundExport(unittest.TestCase):
    approach = "weight_only"

    @classmethod
    def setUpClass(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )
        self.gptj_no_jit = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.llm_dataloader = LLMDataLoader()
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autoround_int_quant(self):
        model = copy.deepcopy(self.gptj)
        out1 = model(self.lm_input)
        round = AutoRound
        optq_1 = round(model, self.tokenizer, n_samples=20, amp=False, seqlen=10, iters=10)
        q_model, weight_config1 = optq_1.quantize()
        from auto_round.export.export_to_itrex import pack_model

        compressed_model = pack_model(model=q_model, weight_config=weight_config1)
        out2 = model(self.lm_input)
        out3 = q_model(self.lm_input)
        out4 = compressed_model(self.lm_input)
        self.assertTrue(torch.all(torch.isclose(out1[0], out2[0], atol=1e-1)))
        self.assertFalse(torch.all(out1[0] == out2[0]))
        self.assertTrue(torch.all(out2[0] == out3[0]))
        self.assertTrue(torch.all(torch.isclose(out3[0], out4[0], atol=1e-5)))
        self.assertTrue("transformer.h.0.attn.k_proj.qzeros" in compressed_model.state_dict().keys())

        model = copy.deepcopy(self.gptj)
        out6 = model(self.lm_input)
        optq_2 = round(model, self.tokenizer, device="cpu", n_samples=20, seqlen=10)
        q_model, weight_config2 = optq_2.quantize()
        compressed_model = pack_model(model=q_model, weight_config=weight_config2, inplace=False)
        out4 = q_model(self.lm_input)
        out5 = compressed_model(self.lm_input)
        self.assertTrue(torch.all(out1[0] == out6[0]))
        self.assertTrue(torch.all(torch.isclose(out4[0], out5[0], atol=1e-5)))

    def test_config(self):
        from auto_round.export import QuantConfig

        config = QuantConfig.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")
        config.save_pretrained("quantization_config_dir")
        loaded_config = QuantConfig.from_pretrained("quantization_config_dir")
        self.assertEqual(config.group_size, loaded_config.group_size)
        self.assertEqual(config.desc_act, loaded_config.desc_act)
        self.assertEqual(config.bits, loaded_config.bits)
        self.assertEqual(config.sym, loaded_config.sym)


if __name__ == "__main__":
    unittest.main()
