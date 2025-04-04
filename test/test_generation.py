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


class TestAutoRoundFormatGeneration(unittest.TestCase):
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

    @unittest.skipIf(torch.cuda.is_available() is False, "Skipping because no cuda")
    def test_llm_generation_sym_gpu_gptq(self):
        bits = 4
        group_size = 32
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=True,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_round:gptq",inplace=False)
        device = "auto"  ##cpu, hpu, cuda
        from auto_round import AutoRoundConfig
        quantization_config = AutoRoundConfig(
            backend=device
        )

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                                     device_map=device, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)

    @unittest.skipIf(torch.cuda.is_available() is False, "Skipping because no cuda")
    def test_llm_generation_asym_gpu_awq(self):
        bits = 4
        group_size = 32
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=True,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_round:awq",inplace=False)
        device = "auto"  ##cpu, hpu, cuda
        from auto_round import AutoRoundConfig
        quantization_config = AutoRoundConfig(
            backend=device
        )

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                                     device_map=device, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)
        
    def test_llm_generation_asym_qbits(self):
        try:
            import intel_extension_for_transformers
        except:
            return
        bits = 4
        group_size = 32
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=True,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_round",inplace=False)
        device = "cpu"  ##cpu, hpu, cuda
        from auto_round import AutoRoundConfig
        quantization_config = AutoRoundConfig(
            backend="cpu:auto_round:qbits_zp"
        )

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                                     device_map="cpu", quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)
    
    def test_force_to_autoround_format(self):
        bits = 4
        group_size = 128
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=True,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"
        autoround.save_quantized(output_dir=quantized_model_path, format="auto_awq", inplace=False)
        device = "auto"  ##cpu, hpu, cuda, auto
        from auto_round import AutoRoundConfig
        quantization_config = AutoRoundConfig(
            # backend="auto_round:ipex_awq",
            backend="auto"
        )
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device,
                                                     quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert ("!!!" not in res)
        
