import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import (
    require_awq,
    require_gptqmodel,
    require_package_version_ut,
)

from ...helpers import get_model_path


class TestAutoRound:
    save_dir = "./saved"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gptqmodel
    def test_mixed_gptqmodel(self, tiny_opt_model_path, dataloader):
        scheme = "W4A16"
        layer_config = {
            "k_proj": {"bits": 8},  # part name
            "lm_head": {"bits": 4},  # set lm_head quant
            "fc1": {"bits": 16},
            "model.decoder.layers.0.self_attn.v_proj": {"bits": 16},
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 8},  # full name
        }
        autoround = AutoRound(
            model=tiny_opt_model_path,
            scheme=scheme,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
        from gptqmodel import GPTQModel

        model = GPTQModel.load(quantized_model_path)
        assert model.model.model.decoder.layers[0].self_attn.k_proj.bits == 8
        assert model.model.model.decoder.layers[0].self_attn.q_proj.bits == 8
        assert model.model.model.decoder.layers[1].self_attn.v_proj.bits == 4
        res = model.tokenizer.decode(model.generate("Uncovering deep insights begins with")[0])
        print(res)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_gptqmodel_convert_to_ar(self, tiny_opt_model_path, dataloader):
        layer_config = {
            "k_proj": {"bits": 8},  # part name
            "lm_head": {"bits": 4},  # set lm_head quant
            "model.decoder.layers.0.self_attn.v_proj": {"bits": 16},
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 8},  # full name
        }
        autoround = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
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

    def test_mixed_autoround_format(self, tiny_opt_model_path, dataloader):
        layer_config = {
            "k_proj": {"bits": 8},
            "q_proj": {"bits": 3},
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "self.save_dir"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        config_file = Path(quantized_model_path) / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        quant_config = config.get("quantization_config", {})
        extra_config = quant_config.get("extra_config", {})
        # check extra_config only saved attributes differing from Scheme values
        assert "act_bits" not in extra_config[".*fc1.*"].keys()  ## TODO refine this assert
        assert "group_size" not in extra_config[".*fc1.*"].keys()
        assert "act_bits" not in extra_config["model.decoder.layers.0.self_attn.k_proj"].keys()
        assert "group_size" not in extra_config["model.decoder.layers.0.self_attn.k_proj"].keys()
        assert "group_size" not in extra_config["model.decoder.layers.1.self_attn.q_proj"].keys()
        assert "bits" in extra_config["model.decoder.layers.1.self_attn.q_proj"].keys()
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        assert model.model.decoder.layers[0].self_attn.k_proj.bits == 8
        assert model.model.decoder.layers[0].self_attn.q_proj.bits == 3
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    @require_awq
    @require_package_version_ut("transformers", "<4.57.0")
    def test_fallback_regex_for_awq_format(self, tiny_opt_model_path, dataloader):
        layer_config = {
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "self.save_dir"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_ar_format_part_name_hf_loading(self, tiny_opt_model_path, dataloader):
        layer_config = {
            "k_proj": {"bits": 8},  # part name
            "lm_head": {"bits": 16},  # full name
            ".*fc1.*": {"bits": 16},  # standard regex
        }
        autoround = AutoRound(
            model=tiny_opt_model_path,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "self.save_dir"
        autoround.quantize()
        autoround.save_quantized(output_dir=quantized_model_path, format="auto_round")
        # remove old extra_config(which contains full name layer configs), only test regex config loading
        new_extra_config = {
            ".*fc1.*": {  # standard regex
                "bits": 16,
            },
            "k_proj": {  # part name
                "bits": 8,
            },
        }
        config_file = Path(quantized_model_path) / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        quant_config = config.get("quantization_config", {})
        old_extra_config = quant_config.get("extra_config", {})
        # check extra_config only saved attributes differing from Scheme values
        assert "sym" not in old_extra_config[".*fc1.*"].keys()
        assert "act_dynamic" not in old_extra_config[".*fc1.*"].keys()
        assert "group_size" not in old_extra_config[".*fc1.*"].keys()
        quant_config["extra_config"] = new_extra_config
        config["quantization_config"] = quant_config
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        assert model.model.decoder.layers[0].self_attn.k_proj.bits == 8
        assert model.model.decoder.layers[0].self_attn.q_proj.bits == 4
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_MXFP_autoround_format_loading(self, dataloader):
        layer_config = {
            "k_proj": {"bits": 8, "act_bits": 8},
            "lm_head": {"bits": 16, "act_bits": 16},
            "fc1": {"bits": 8, "act_bits": 8},
        }
        model_path = get_model_path("facebook/opt-125m")
        autoround = AutoRound(
            model_path,
            scheme="MXFP4",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        from ...helpers import evaluate_accuracy

        evaluate_accuracy(model, tokenizer, threshold=0.32, batch_size=16)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_autoround_format_vllm(self, tiny_opt_model_path, dataloader):
        layer_config = {
            "self_attn": {"bits": 8},
            "lm_head": {"bits": 16},
        }
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
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
        QUANTIZATION = "auto-round"  # quantized_model_path
        llm = LLM(
            model=quantized_model_path,
            quantization=QUANTIZATION,
            trust_remote_code=True,
            tensor_parallel_size=1,
            allow_deprecated_quantization=True,
        )
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # if "France" in prompt:
            assert "!!!" not in generated_text
            print(f"{prompt}: {generated_text}")
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_llmcompressor_format_vllm(self, tiny_opt_model_path, dataloader):
        layer_config = {
            "self_attn": {"bits": 16, "act_bits": 16},
            "lm_head": {"bits": 16, "act_bits": 16},
            "fc1": {"bits": 16, "act_bits": 16},
        }
        autoround = AutoRound(
            tiny_opt_model_path,
            scheme="NVFP4",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        compressed, _ = autoround.quantize_and_save(
            output_dir=quantized_model_path, inplace=False, format="llm_compressor"
        )
        from vllm import LLM, SamplingParams

        # Sample prompts.
        prompts = [
            "The capital of France is",
            "The future of AI is",
        ]
        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        # Create an LLM.
        llm = LLM(
            model=quantized_model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            allow_deprecated_quantization=True,
        )
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"{prompt}: {generated_text}")
            assert "!!!" not in generated_text
        shutil.rmtree(quantized_model_path, ignore_errors=True)
