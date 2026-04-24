import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.export.export_to_llmcompressor import export_to_fp as llmc_fp_export
from auto_round.export.export_to_llmcompressor import export_to_static_fp as llmc_static_fp_export

from ...helpers import forbid_threaded_packing, get_model_path, opt_name_or_path


class TestLLMC:
    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("stas/tiny-random-llama-2")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    # remove since w8a8 not in llmcompressor format supported schemes
    # def test_llmcompressor_w8a8(self):
    #     bits, group_size, sym, act_bits = 8, -1, True, 8
    #     ## quantize the model
    #     autoround = AutoRound(
    #         self.model,
    #         self.tokenizer,
    #         bits=bits,
    #         group_size=group_size,
    #         sym=sym,
    #         act_bits=act_bits,
    #         seqlen=8,
    #         nsamples=2,
    #         iters=0,
    #     )
    #     autoround.quantize()
    #     autoround.save_quantized("./saved", format="llm_compressor", inplace=True)

    def test_llmcompressor_fp8(self):
        ## quantize the model
        model_name = opt_name_or_path
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            nsamples=2,
            iters=0,
        )
        autoround.quantize_and_save(self.save_dir, format="llm_compressor")
        # from vllm import LLM
        # model = LLM(self.save_dir)
        # result = model.generate("Hello my name is")
        # print(result)

        import json

        from safetensors import safe_open

        config = json.load(open(os.path.join(self.save_dir, "config.json")))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "channel"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"

        f = safe_open(os.path.join(self.save_dir, "model.safetensors"), framework="pt")
        assert len(f.get_tensor("model.decoder.layers.0.fc1.weight_scale").shape) == 2

    def test_autoround_llmcompressor_fp8(self):
        ## quantize the model
        model_name = opt_name_or_path
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            group_size=0,
            nsamples=2,
            iters=0,
        )
        autoround.quantize_and_save(self.save_dir, format="auto_round:llm_compressor")

        import json

        config = json.load(open(os.path.join(self.save_dir, "config.json")))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "tensor"
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["strategy"] == "tensor"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"


def test_llmcompressor_static_fp_export_packs_serially(tiny_opt_model_path, tmp_path, monkeypatch):
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="FP8_STATIC",
        seqlen=8,
        nsamples=2,
        iters=0,
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, llmc_static_fp_export)
    autoround.save_quantized(tmp_path, format="llm_compressor")
    assert os.path.exists(os.path.join(tmp_path, "config.json"))


def test_llmcompressor_mxfp8_export_packs_serially(tmp_path, monkeypatch):
    autoround = AutoRound(
        model=opt_name_or_path,
        iters=0,
        disable_opt_rtn=True,
        scheme="mxfp8",
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, llmc_fp_export)
    compressed_model = autoround.save_quantized(output_dir=tmp_path, format="llm_compressor")
    tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
    assert hasattr(tmp_layer, "weight_scale")
