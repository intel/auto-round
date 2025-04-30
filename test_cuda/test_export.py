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
        self.model_name = "facebook/opt-125m"

        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autogptq_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
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
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        assert (res == "</s>There is a girl who likes adventure, she is a good friend of mine, she is a good friend of"
                       " mine, she is a good friend of mine, she is a good friend of mine, she is a good friend of mine, "
                       "she is a good friend of mine, she is")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autogptq_format_fp_layers(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        layer_config = {}
        for n, m in model.named_modules():
            if "q_proj" in n:
                layer_config[n] = {"bits": 16}

        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config
        )
        autoround.quantize()
        quantized_model_path = "./saved"
        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_gptq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        assert res == ("</s>There is a girl who likes adventure, she is a great artist, she is a great artist, "
                       "she is a great artist, she is a great artist, she is a great artist,"
                       " she is a great artist, she is a great artist, she is a great artist, she is")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autogptq_format_qsave_fp_layers(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        layer_config = {}
        for n, m in model.named_modules():
            if "q_proj" in n:
                layer_config[n] = {"bits": 16}

        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        assert (
                res == "</s>There is a girl who likes adventure, she is a great artist, she is a great artist,"
                       " she is a great artist, she is a great artist, she is a great artist, "
                       "she is a great artist, she is a great artist, she is a great artist, she is")
        from auto_round import AutoRoundConfig

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True,
                                                     quantization_config=AutoRoundConfig())
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        g_tokens = model.generate(**inputs, max_new_tokens=50)[0]
        res = tokenizer.decode(g_tokens)
        assert (
                res == "</s>There is a girl who likes adventure, she is a great artist, she is a great artist,"
                       " she is a great artist, she is a great artist, she is a great artist, she is a great "
                       "artist, she is a great artist, she is a great artist, she is")
        ##print(res)
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, \
                                 format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        assert res == ("</s>There is a girl who likes adventure, she is a great artist, she is a great artist,"
                       " she is a great artist, she is a great artist, she is a great artist, "
                       "she is a great artist, she is a great artist, she is a great artist, she is")
        shutil.rmtree("./saved", ignore_errors=True)


    #
    def test_autoawq_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, \
                                 format="auto_awq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        assert res == ("</s>There is a girl who likes adventure, but she's not a very good one.\nI don't"
                       " know why you're getting downvoted. I think you're just being a dick.\nI'm not a dick, "
                       "I just think it's funny that people are downvoting")
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autoawq_format_fp_qsave_layers(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        layer_config = {"model.decoder.layers.0.self_attn.k_proj": {"bits": 16},
                        "model.decoder.layers.9.self_attn.v_proj": {"bits": 16}, }
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config
        )
        quantized_model_path = "/data5/wenhuach/test_export"
        autoround.qsave(output_dir=quantized_model_path,
                        format="auto_awq")
        from auto_round import AutoRoundConfig
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto",
                                                     quantization_config=AutoRoundConfig())
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)

        shutil.rmtree("./saved", ignore_errors=True)
