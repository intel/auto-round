import copy
import shutil
import sys
import unittest

sys.path.insert(0, ".")
import torch
import transformers
import torch
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


# class TestPytorchWeightOnlyAdaptor(unittest.TestCase):
#     approach = "weight_only"
#
#     @classmethod
#     def setUpClass(self):
#         self.dataloader = SimpleDataLoader()
#         self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
#             "hf-internal-testing/tiny-random-GPTJForCausalLM",
#             torchscript=True,
#         )
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(
#             "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
#         )
#         self.gptj_no_jit = transformers.AutoModelForCausalLM.from_pretrained(
#             "hf-internal-testing/tiny-random-GPTJForCausalLM",
#         )
#         self.llm_dataloader = LLMDataLoader()
#         self.lm_input = torch.ones([1, 10], dtype=torch.long)
#
#     @classmethod
#     def tearDownClass(self):
#         shutil.rmtree("./saved", ignore_errors=True)
#         shutil.rmtree("runs", ignore_errors=True)
#
#     def test_autoround_int_quant(self):
#         model = copy.deepcopy(self.gptj)
#         device = "cpu"
#         model = model
#         out1 = model(self.lm_input)
#         round = AutoRound
#         optq_1 = round(
#             model, self.tokenizer, n_samples=20, device=device, amp=False, seqlen=10, iters=10, scale_dtype="fp32"
#         )
#         q_model, weight_config1 = optq_1.quantize()
#         q_model = q_model
#         from auto_round.export.export_to_itrex import compress_model
#
#         compressed_model, _ = compress_model(q_model, weight_config1)
#         q_model = q_model
#         model = model
#         out2 = model(self.lm_input)
#         out3 = q_model(self.lm_input)
#         out4 = compressed_model(self.lm_input)
#         self.assertTrue(torch.all(torch.isclose(out1[0], out2[0], atol=1e-1)))
#         self.assertFalse(torch.all(out1[0] == out2[0]))
#         self.assertTrue(torch.all(out2[0] == out3[0]))
#         self.assertTrue(torch.all(torch.isclose(out3[0], out4[0], atol=1e-5)))
#         self.assertTrue("transformer.h.0.attn.k_proj.qzeros" in compressed_model.state_dict().keys())
#
#         # model = copy.deepcopy(self.gptj)
#         # out6 = model(self.lm_input)
#         # optq_2 = round(model, self.tokenizer, n_samples=20, amp=False, seqlen=10)
#         # q_model, weight_config2 = optq_2.quantize()
#         # out4 = q_model(self.lm_input)
#         # out5 = model(self.lm_input)
#
#         # self.assertTrue(torch.all(out1[0] == out6[0]))
#         # self.assertTrue(torch.all(out4[0] == out5[0]))
#         # self.assertTrue(torch.all(torch.isclose(out6[0], out5[0], atol=1e-1)))
#
#     def test_config(self):
#         from auto_round import QuantConfig
#
#         config = QuantConfig.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")
#         config.save_pretrained("quantization_config_dir")
#         loaded_config = QuantConfig.from_pretrained("quantization_config_dir")
#         self.assertEqual(config.group_size, loaded_config.group_size)
#         self.assertEqual(config.true_sequential, loaded_config.true_sequential)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_default(self):
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              dataloader=self.llm_dataloader)
        autoround.quantize()
        if torch.cuda.is_available():
            autoround.save_quantized(output_dir="./saved", inplace=False)
        autoround.save_quantized(output_dir="./saved", inplace=False, format="itrex")

    def test_sym(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              dataloader=self.llm_dataloader)
        autoround.quantize()

    def test_w4g1(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              dataloader=self.llm_dataloader)
        autoround.quantize()

    def test_w3g128(self):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              dataloader=self.llm_dataloader)
        autoround.quantize()

    def test_w2g128(self):
        bits, group_size, sym = 2, 128, True
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              dataloader=self.llm_dataloader)
        autoround.quantize()

    def test_disable_use_quant_input(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              use_quant_input=False,
                              dataloader=self.llm_dataloader)
        autoround.quantize()

    def test_disable_minmax_tuning(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              enable_minmax_tuning=False,
                              dataloader=self.llm_dataloader)
        autoround.quantize()

    def test_signround(self):
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(self.model, self.tokenizer, bits=bits, group_size=group_size, sym=sym, iters=2,
                              enable_minmax_tuning=False, use_quant_input=False,
                              dataloader=self.llm_dataloader)
        autoround.quantize()


if __name__ == "__main__":
    unittest.main()
