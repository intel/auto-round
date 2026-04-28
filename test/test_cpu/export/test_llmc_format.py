import os
import shutil
import json

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, AutoScheme
from auto_round.export.export_to_llmcompressor import export_to_fp as llmc_fp_export
from auto_round.export.export_to_llmcompressor import export_to_static_fp as llmc_static_fp_export

from ...envs import is_compressed_tensors_available
from ...helpers import forbid_threaded_packing, get_model_path, opt_name_or_path

pytestmark = pytest.mark.skipif(not is_compressed_tensors_available(), reason="test requires compressed-tensors")


class TestLLMC:

    @classmethod
    def setup_class(self):
        self.model_name = get_model_path("stas/tiny-random-llama-2")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

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

    def test_llmcompressor_fp8(self, tmp_path):
        ## quantize the model
        model_name = opt_name_or_path
        autoround = AutoRound(
            model_name,
            scheme="FP8_STATIC",
            seqlen=8,
            nsamples=2,
            iters=0,
        )
        _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="llm_compressor")
        # from vllm import LLM
        # model = LLM(tmp_path)
        # result = model.generate("Hello my name is")
        # print(result)

        import json

        from safetensors import safe_open

        config = json.load(open(os.path.join(quantized_model_path, "config.json")))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "channel"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"

        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert len(f.get_tensor("model.decoder.layers.0.fc1.weight_scale").shape) == 2

    def test_autoround_llmcompressor_fp8(self, tmp_path):
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
        _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="auto_round:llm_compressor")

        import json

        config = json.load(open(os.path.join(quantized_model_path, "config.json")))
        assert "group_0" in config["quantization_config"]["config_groups"]
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
        assert config["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] == "tensor"
        assert config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["strategy"] == "tensor"
        assert config["quantization_config"]["quant_method"] == "compressed-tensors"

    def test_mxfp8_llmcompressor_format(self, tiny_opt_model_path, tmp_path):
        scheme = "mxfp8"
        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme=scheme,
        )
        compressed_model, tmp_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")
        tmp_layer = compressed_model.model.decoder.layers[1].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal MXFP8 packing name or data_type or shape"
        quantization_config = transformers.AutoConfig.from_pretrained(
            tmp_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "mxfp8-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["weights"]["group_size"] == 32
            and quantization_config["config_groups"]["group_0"]["weights"]["scale_dtype"] == "torch.uint8"
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["input_activations"]["group_size"] == 32
            and quantization_config["config_groups"]["group_0"]["input_activations"]["scale_dtype"] == "torch.uint8"
            and quantization_config["ignore"] == ["lm_head"]
        ), f"Invalid MXFP8 quantization configuration: {quantization_config}"

    def test_mixed_precision_llmcompressor_format(self, tiny_opt_model_path, tmp_path):
        scheme = AutoScheme(
            avg_bits=7,
            options=("MXFP4", "MXFP8"),
            shared_layers=["q_proj", "k_proj", "v_proj"],
        )
        ar = AutoRound(
            model=tiny_opt_model_path,
            iters=0,
            disable_opt_rtn=True,
            scheme=scheme,
        )
        _, tmp_path = ar.quantize_and_save(output_dir=tmp_path, format="llm_compressor")
        model = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype="auto", trust_remote_code=True)
        op = model.model.decoder.layers[0].fc1
        if op.quantization_scheme.targets != ["Linear"]:
            assert (
                op.quantization_scheme.weights.num_bits == 8
                and op.quantization_scheme.input_activations.num_bits == 8
                and op.quantization_scheme.weights.group_size == 32
                and op.quantization_scheme.input_activations.group_size == 32
                and op.quantization_scheme.weights.scale_dtype == torch.uint8
                and op.quantization_scheme.input_activations.scale_dtype == torch.uint8
            ), "Illegal MXFP4 packing name or data_type or shape"
        quantization_config = model.config.quantization_config.to_dict()
        assert (
            quantization_config["format"] == "mixed-precision"
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
            and quantization_config["config_groups"]["group_0"]["format"] == "mxfp8-quantized"
            and quantization_config["config_groups"]["group_1"]["weights"]["num_bits"] == 4
            and quantization_config["config_groups"]["group_1"]["input_activations"]["num_bits"] == 4
            and quantization_config["config_groups"]["group_1"]["format"] == "mxfp4-pack-quantized"
            and quantization_config["ignore"] == ["lm_head"]
        ), f"Invalid mixed precision quantization configuration: {quantization_config}"


def test_llmcompressor_static_fp_export_packs_serially(tiny_opt_model_path, dataloader, tmp_path, monkeypatch):
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="FP8_STATIC",
        seqlen=8,
        nsamples=2,
        iters=0,
        dataset=dataloader,
    )
    autoround.quantize()
    forbid_threaded_packing(monkeypatch, llmc_static_fp_export)
    autoround.save_quantized(tmp_path, format="llm_compressor")
    assert os.path.exists(os.path.join(tmp_path, "config.json"))


def test_llmcompressor_static_fp8_kv_config(tiny_opt_model_path, dataloader, tmp_path):
    autoround = AutoRound(
        tiny_opt_model_path,
        scheme="FP8_STATIC",
        seqlen=8,
        nsamples=2,
        iters=0,
        dataset=dataloader,
        static_kv_dtype="fp8",
    )
    _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="llm_compressor")

    with open(os.path.join(quantized_model_path, "config.json")) as f:
        config = json.load(f)
    kv_cache_scheme = config["quantization_config"]["kv_cache_scheme"]
    assert kv_cache_scheme is not None
    assert kv_cache_scheme["num_bits"] == 8
    assert kv_cache_scheme["type"] == "float"
    assert kv_cache_scheme["strategy"] == "tensor"
    assert kv_cache_scheme["dynamic"] is False
    assert kv_cache_scheme["symmetric"] is True


def test_llmcompressor_static_fp8_attention_config(dataloader, tmp_path):
    model_name = get_model_path("stas/tiny-random-llama-2")
    autoround = AutoRound(
        model_name,
        scheme="FP8_STATIC",
        seqlen=8,
        nsamples=2,
        iters=0,
        dataset=dataloader,
        static_attention_dtype="fp8",
    )
    _, quantized_model_path = autoround.quantize_and_save(tmp_path, format="llm_compressor")

    with open(os.path.join(quantized_model_path, "config.json")) as f:
        saved_config = json.load(f)
    saved_groups = saved_config["quantization_config"]["config_groups"]
    attention_group = None
    for group in saved_groups.values():
        if "Linear" not in group["targets"]:
            attention_group = group
            break

    assert attention_group is not None
    assert attention_group["weights"] is None
    assert attention_group["input_activations"]["num_bits"] == 8
    assert attention_group["input_activations"]["type"] == "float"
    assert attention_group["input_activations"]["strategy"] == "tensor"
    assert attention_group["input_activations"]["dynamic"] is False
    assert attention_group["input_activations"]["symmetric"] is True
    assert saved_config["quantization_config"]["kv_cache_scheme"] is not None

    model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", trust_remote_code=True)
    quantization_config = model.config.quantization_config
    config = quantization_config.to_dict() if hasattr(quantization_config, "to_dict") else quantization_config
    config_groups = config["config_groups"]

    assert "group_0" in config_groups
    attention_group = None
    for group in config_groups.values():
        targets = group["targets"]
        if "Linear" not in targets:
            attention_group = group
            break

    assert attention_group is not None
    assert attention_group["targets"] == [model.model.layers[0].self_attn.__class__.__name__]
    assert attention_group["weights"] is None
    assert attention_group["input_activations"]["num_bits"] == 8
    assert attention_group["input_activations"]["type"] == "float"
    assert attention_group["input_activations"]["strategy"] == "tensor"
    assert attention_group["input_activations"]["dynamic"] is False
    assert attention_group["input_activations"]["symmetric"] is True
    assert getattr(model.model.layers[0].self_attn, "q_scale", None) is not None
    assert config["kv_cache_scheme"] is not None


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
