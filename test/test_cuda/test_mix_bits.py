import os
import shutil
import sys
import unittest

from parameterized import parameterized

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer
from auto_round.testing_utils import require_gptqmodel

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
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
    
    @require_gptqmodel
    def test_mixed_gptqmodel(self):
        bits, sym, group_size = 4, True, 128
        model_name = "facebook/opt-125m"
        layer_config = {
            "k_proj": {"bits": 8},
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
        from gptqmodel import GPTQModel
        model = GPTQModel.load(quantized_model_path)
        assert (model.model.model.decoder.layers[0].self_attn.k_proj.bits == 8)
        assert (model.model.model.decoder.layers[0].self_attn.q_proj.bits == 4)
        result = model.generate("Uncovering deep insights begins with")[0] # tokens
        assert("!!!" not in model.tokenizer.decode(result)) # string output
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_autoround_format(self):
        bits, sym, group_size = 4, True, 128
        model_name = "facebook/opt-125m"
        layer_config = {
            "k_proj": {"bits": 8},
            "q_proj": {"bits": 3},
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        assert (model.model.decoder.layers[0].self_attn.k_proj.bits == 8)
        assert (model.model.decoder.layers[0].self_attn.q_proj.bits == 3)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_autoround_format_vllm(self):
        layer_config = {
            "self_attn": {"bits": 8},
            "lm_head": {"bits": 16},
        }
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir
        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

        from vllm import LLM, SamplingParams
        # Sample prompts.
        prompts = [
            "The capital of France is",
            "The future of AI is",
        ]
        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        # Create an LLM.
        QUANTIZATION = "auto-round" #quantized_model_path
        llm = LLM(model=quantized_model_path, quantization=QUANTIZATION, trust_remote_code=True, tensor_parallel_size=1)
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # if "France" in prompt:
            assert "!!!" not in generated_text
            print(f"{prompt}: {generated_text}")
        shutil.rmtree(quantized_model_path, ignore_errors=True)


    def test_mixed_llmcompressor_format_vllm(self):
        model_name = "facebook/opt-125m"
        layer_config = {
            "self_attn": {"bits": 16, "act_bits": 16, "data_type": "float"},
            "lm_head": {"bits": 16, "act_bits": 16, "data_type": "float"},
            "fc1": {"bits": 16, "act_bits": 16, "data_type": "float", },
        }
        autoround = AutoRound(
            model_name,
            scheme="NVFP4",
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        compressed,_ = autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="llm_compressor")
        from vllm import LLM, SamplingParams
        # Sample prompts.
        prompts = [
            "The capital of France is",
            "The future of AI is",
        ]
        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        # Create an LLM.
        QUANTIZATION = "auto-round" #quantized_model_path
        llm = LLM(model=quantized_model_path, trust_remote_code=True, tensor_parallel_size=1)
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"{prompt}: {generated_text}")
            assert "!!!" not in generated_text
        shutil.rmtree(quantized_model_path, ignore_errors=True)



if __name__ == "__main__":
    unittest.main()

