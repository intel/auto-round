import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import require_gptqmodel

from ...helpers import opt_name_or_path


def _get_folder_size(path: str) -> float:
    """Return folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # convert to GB


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.model_name = opt_name_or_path
        self.save_dir = ".saved/"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_gptqmodel
    def test_mixed_gptqmodel(self, dataloader):
        layer_config = {
            "k_proj": {"bits": 8},  # part name
            "lm_head": {"bits": 4},  # set lm_head quant
            "fc1": {"bits": 16},
            "model.decoder.layers.0.self_attn.v_proj": {"bits": 16},
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 8},  # full name
        }
        autoround = AutoRound(
            model=self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
        # test original GPTQModel inference
        from gptqmodel import GPTQModel

        model = GPTQModel.load(quantized_model_path)
        assert model.model.model.decoder.layers[0].self_attn.k_proj.bits == 8
        assert model.model.model.decoder.layers[0].self_attn.q_proj.bits == 8
        assert model.model.model.decoder.layers[1].self_attn.v_proj.bits == 4
        result = model.generate("Uncovering deep insights begins with")[0]  # tokens
        assert "!!!" not in model.tokenizer.decode(result)  # string output
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_gptqmodel_convert_to_ar(self, dataloader):
        layer_config = {
            "k_proj": {"bits": 8},  # part name
            "lm_head": {"bits": 4},  # set lm_head quant
            "fc1": {"bits": 16},
            "model.decoder.layers.0.self_attn.v_proj": {"bits": 16},
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 8},  # full name
        }
        autoround = AutoRound(
            model=self.model_name,
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
            quantized_model_path, device_map="cpu", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])
        print(res)
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_autoround_format(self, dataloader):
        layer_config = {
            "k_proj": {"bits": 8},
            "q_proj": {"bits": 3},
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "./saved"
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
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        assert model.model.decoder.layers[0].self_attn.k_proj.bits == 8
        assert model.model.decoder.layers[0].self_attn.q_proj.bits == 3
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_fallback_regex_for_awq_format(self, dataloader):
        layer_config = {
            "lm_head": {"bits": 16},
            "fc1": {"bits": 16},
        }
        autoround = AutoRound(
            model=self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mixed_ar_format_part_name_hf_loading(self, dataloader):
        layer_config = {
            "k_proj": {"bits": 8},  # part name
            "lm_head": {"bits": 16},  # full name
            ".*fc1.*": {"bits": 16},  # standard regex
        }
        autoround = AutoRound(
            model=self.model_name,
            scheme="W4A16",
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "./saved"
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
        assert "act_bits" not in old_extra_config[".*fc1.*"].keys()
        assert "group_size" not in old_extra_config[".*fc1.*"].keys()
        quant_config["extra_config"] = new_extra_config
        config["quantization_config"] = quant_config
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
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
        autoround = AutoRound(
            self.model_name,
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
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        from auto_round.eval.evaluation import simple_evaluate_user_model

        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai", limit=10)
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.14
        shutil.rmtree(quantized_model_path, ignore_errors=True)

