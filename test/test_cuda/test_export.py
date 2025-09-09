import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import require_awq, require_optimum


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

    def test_fp8input_mxfp4_llmcompressor_format(self):
        model_name = "/models/Qwen3-0.6B-FP8"
        scheme = "mxfp4"
        ar = AutoRound(
            model=model_name,
            iters=2,
            seqlen=2,
            scheme=scheme,
            dataset=self.llm_dataloader,
        )
        compressed_model, _ = ar.quantize_and_save(output_dir=self.save_dir, format="llm_compressor")
        tmp_layer = compressed_model.model.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_packed")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 2048
        ), "Illegal MXFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(self.save_dir, trust_remote_code=True).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4
        ), f"Invalid MXFP4 quantization configuration: {quantization_config}"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_nvfp4_llmcompressor_format(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        scheme = "nvfp4"
        autoround = AutoRound(
            model,
            tokenizer,
            scheme=scheme,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "nvfp4-pack-quantized"
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 4
        ), f"Invalid NVFP4 quantization configuration: {quantization_config}"
        shutil.rmtree("./saved", ignore_errors=True)
        # from vllm import LLM, SamplingParams
        # prompts = [
        #     "The capital of France is",
        #     "The future of AI is",
        # ]
        ## Create a sampling params object.
        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)
        # QUANTIZATION = "compressed-tensors"
        # llm = LLM(model=quantized_model_path,
        #         #   quantization=QUANTIZATION,
        #           trust_remote_code=True,
        #           tensor_parallel_size=1,
        #           enforce_eager=True,
        #           gpu_memory_utilization=0.7,
        # )
        # outputs = llm.generate(prompts, sampling_params)
        # # Print the outputs.
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     if "France" in prompt:
        #         assert "Paris" in generated_text


if __name__ == "__main__":
    unittest.main()
