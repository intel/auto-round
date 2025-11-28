import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import require_awq, require_optimum, require_package_version_ut


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
        self.save_dir = "./saved"
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_optimum
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
        assert (
            res == "</s>There is a girl who likes adventure, she is a good friend of mine, she is a good friend of"
            " mine, she is a good friend of mine, she is a good friend of mine, she is a good friend of mine, "
            "she is a good friend of mine, she is"
        )
        shutil.rmtree("./saved", ignore_errors=True)

    @require_optimum
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
            layer_config=layer_config,
        )
        autoround.quantize()
        quantized_model_path = "./saved"
        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_gptq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        # Affected greatly by the transformers version
        # assert res == (
        #     "</s>There is a girl who likes adventure, there there there there there there there there there there "
        #     "there there there there there there there there there there there there there there there there there "
        #     "there there there there there there there there there there there there there there there there there "
        #     "there there there there there there")
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
            layer_config=layer_config,
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
            "she is a great artist, she is a great artist, she is a great artist, she is"
        )
        from auto_round import AutoRoundConfig

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", trust_remote_code=True, quantization_config=AutoRoundConfig()
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        g_tokens = model.generate(**inputs, max_new_tokens=50)[0]
        res = tokenizer.decode(g_tokens)
        assert (
            res == "</s>There is a girl who likes adventure, she is a great artist, she is a great artist,"
            " she is a great artist, she is a great artist, she is a great artist, she is a great "
            "artist, she is a great artist, she is a great artist, she is"
        )
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

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        print(res)
        # Affected greatly by the transformers version
        # assert res == ("</s>There is a girl who likes adventure, she is a great artist, she is a great artist,"
        #                " she is a great artist, she is a great artist, she is a great artist, "
        #                "she is a great artist, she is a great artist, she is a great artist, she is")
        shutil.rmtree("./saved", ignore_errors=True)

    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
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

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_awq")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0])
        assert res == (
            "</s>There is a girl who likes adventure, but she's not a very good one.\nI don't"
            " know why you're getting downvoted. I think you're just being a dick.\nI'm not a dick, "
            "I just think it's funny that people are downvoting"
        )
        shutil.rmtree("./saved", ignore_errors=True)

    @require_optimum
    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_autoawq_format_fp_qsave_layers(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        layer_config = {
            "model.decoder.layers.0.self_attn.k_proj": {"bits": 16},
            "model.decoder.layers.9.self_attn.v_proj": {"bits": 16},
        }
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
            layer_config=layer_config,
        )
        quantized_model_path = "./saved/test_export"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        from auto_round import AutoRoundConfig

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=AutoRoundConfig()
        )
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

    def test_autoround_3bit_asym_torch_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 3, 128, False
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

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round:gptqmodel")

        device = "auto"  ##cpu, hpu, cuda
        from auto_round import AutoRoundConfig

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_autoround_3bit_sym_torch_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 3, 128, True
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

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

        device = "auto"  ##cpu, hpu, cuda
        from auto_round import AutoRoundConfig

        quantization_config = AutoRoundConfig(backend=device)
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map=device, quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_awq_lmhead_export(self):
        bits, sym, group_size = 4, False, 128
        model_name = "/models/phi-2"
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
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
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        lm_head = compressed_model.lm_head
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        assert isinstance(lm_head, WQLinear_GEMM), "Illegal GPTQ quantization for lm_head layer"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_gptq_lmhead_export(self):
        bits, sym, group_size = 4, True, 128
        model_name = "/models/phi-2"
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
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
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
        lm_head = compressed_model.lm_head
        assert hasattr(lm_head, "bits") and lm_head.bits == 4, "Illegal GPTQ quantization for lm_head layer"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])
        print(res)
        shutil.rmtree(quantized_model_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
