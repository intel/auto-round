import copy
import shutil
import sys
import unittest

sys.path.insert(0, "..")
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

    def test_autogptq_format(self):
        if not torch.cuda.is_available():
            return
        try:
            import auto_gptq
        except:
            return
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_gptq")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    # def test_autogptq_marlin_format(self):
    #
    #     if not torch.cuda.is_available():
    #         return
    #     try:
    #         import auto_gptq
    #     except:
    #         return
    #     bits, group_size, sym = 4, 128, True
    #     autoround = AutoRound(
    #         self.model,
    #         self.tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         iters=2,
    #         seqlen=2,
    #         dataset=self.llm_dataloader,
    #     )
    #     autoround.quantize()
    #     quantized_model_path = "./saved"
    #
    #     autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_gptq")
    #
    #     from auto_gptq import AutoGPTQForCausalLM
    #     model = AutoGPTQForCausalLM.from_quantized(quantized_model_path, device_map="auto", use_safetensors=True,
    #                                                use_marlin=True)
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
    #     shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_format(self):
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")
        try:
            import intel_extension_for_transformers
        except:
            return

        from auto_round.auto_quantizer import AutoHfQuantizer
        device = "auto"  ##cpu, hpu, cuda
        from auto_round import AutoRoundConfig
        quantization_config = AutoRoundConfig(
            backend=device
        )
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree("./saved", ignore_errors=True)


    def test_autoround_awq_format(self):
        try:
            import awq
        except:
            return
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round:awq")

        from auto_round.auto_quantizer import AutoHfQuantizer
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree("./saved", ignore_errors=True)


    def test_autoawq_format(self):
        try:
            import awq
        except:
            return
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, \
            format="auto_awq", model_path="facebook/opt-125m")

        from auto_round.auto_quantizer import AutoHfQuantizer
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    # def test_autoround_marlin_format(self):
    #     if not torch.cuda.is_available():
    #         return
    #     try:
    #         import auto_gptq
    #     except:
    #         return
    #     bits, group_size, sym = 4, 128, True
    #     autoround = AutoRound(
    #         self.model,
    #         self.tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         iters=2,
    #         seqlen=2,
    #         dataset=self.llm_dataloader,
    #     )
    #     autoround.quantize()
    #     quantized_model_path = "./saved"
    #
    #     autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round:marlin")
    #
    #     from auto_round.auto_quantizer import AutoHfQuantizer
    #     model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
    #     tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
    #     shutil.rmtree("./saved", ignore_errors=True)
    #
