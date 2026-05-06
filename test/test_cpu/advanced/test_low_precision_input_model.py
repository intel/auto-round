import pytest
import torch
import torch.nn as nn
import transformers
from packaging import version

from auto_round import AutoRound
from auto_round.modeling.fused_moe import materialize_model_
from auto_round.special_model_handler import update_module
from auto_round.utils.weight_handler import (
    ModuleWeightType,
    check_and_mark_quantized_module,
    convert_module_to_hp_if_necessary,
    prepare_module_for_shard_write_if_necessary,
)

from ...helpers import get_model_path, get_tiny_model, transformers_version


MXFP4_SCALE_DTYPE = getattr(torch, "float8_e8m0fnu", None)
FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


class StandaloneMXFP4Linear(nn.Module):
    def __init__(self, in_features=64, out_features=2, has_bias=True):
        super().__init__()
        if MXFP4_SCALE_DTYPE is None:
            raise RuntimeError("torch.float8_e8m0fnu is required for StandaloneMXFP4Linear tests")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scale", torch.empty(out_features, in_features // 32, dtype=MXFP4_SCALE_DTYPE))
        if has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)


class TinyStandaloneMXFP4Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = build_standalone_mxfp4_layer()


def build_standalone_mxfp4_layer(in_features=64, out_features=2, has_bias=True, packed_dtype=torch.uint8):
    layer = StandaloneMXFP4Linear(in_features=in_features, out_features=out_features, has_bias=has_bias)
    row_offsets = torch.arange(out_features, dtype=torch.uint8).unsqueeze(1)
    col_offsets = torch.arange(in_features, dtype=torch.uint8).unsqueeze(0)
    unpacked_indices = (row_offsets + col_offsets) % 16
    packed_weight = (unpacked_indices[:, 0::2] | (unpacked_indices[:, 1::2] << 4)).contiguous()
    scale_cols = in_features // 32
    scale_bytes = (127 + ((torch.arange(out_features).unsqueeze(1) + torch.arange(scale_cols).unsqueeze(0)) % 2)).to(
        torch.uint8
    )
    layer.weight = packed_weight.view(packed_dtype) if packed_dtype != torch.uint8 else packed_weight
    layer.scale.copy_(scale_bytes.view(MXFP4_SCALE_DTYPE))
    if layer.bias is not None:
        layer.bias.data.copy_(torch.linspace(-0.5, 0.5, out_features, dtype=torch.float32))
    return layer


def dequantize_standalone_mxfp4_weight(layer: StandaloneMXFP4Linear) -> torch.Tensor:
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)
    packed = layer.weight.view(torch.uint8)
    low = (packed & 0xF).long()
    high = ((packed >> 4) & 0xF).long()
    unpacked = torch.stack([lut[low], lut[high]], dim=-1).reshape(layer.out_features, layer.in_features)

    scale_bytes = layer.scale.view(torch.uint8).to(torch.int16)
    scales = torch.pow(torch.full(scale_bytes.shape, 2.0), (scale_bytes - 127).to(torch.float32))
    return (unpacked.reshape(layer.out_features, -1, 32) * scales.unsqueeze(-1)).reshape(
        layer.out_features, layer.in_features
    )


class TestCompressedTensor:
    nvfp4_model_path = "kaitchup/Qwen3-0.6B-NVFP4"
    mxfp4_model_path = "QuixiAI/Llama-3.2-1B-MXFP4"
    fp8_block_model_path = "RedHatAI/Qwen3-0.6B-FP8-BLOCK"
    w4a16_model_path = "RedHatAI/Qwen3-0.6B-quantized.w4a16"

    def test_fp8_block(self):
        model = get_tiny_model(get_model_path(self.fp8_block_model_path))
        assert (
            model.model.layers[0].mlp.up_proj.weight.dtype == torch.float8_e4m3fn
        ), "Original weight is not in FP8 format"
        assert hasattr(
            model.model.layers[0].mlp.up_proj, "quantization_scheme"
        ), "Model does not contain CompressedLinear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.FP8 in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert (
            model.model.layers[0].mlp.up_proj.weight.dtype == torch.bfloat16
        ), "CompressedLinear layer was not converted to Linear"

    @pytest.mark.skip(
        reason="NVFP4 models are currently not supported due to issues with the compressed_tensors library. See https://github.com/vllm-project/compressed-tensors/issues/642"
    )
    def test_nvfp4(self):
        model = get_tiny_model(get_model_path(self.nvfp4_model_path))
        assert (
            model.model.layers[0].mlp.up_proj.weight_packed.dtype == torch.uint8
        ), "Original weight is not in FP8 format"
        assert hasattr(
            model.model.layers[0].mlp.up_proj, "quantization_scheme"
        ), "Model does not contain CompressedLinear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.NVFP4 in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert (
            model.model.layers[0].mlp.up_proj.weight.dtype == torch.bfloat16
        ), "CompressedLinear layer was not converted to Linear"

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0.0"),
        reason="Compressed-tensor is not compatible with transformers 5.0.0 and above. See https://github.com/vllm-project/compressed-tensors/issues/651",
    )
    def test_mxfp4(self):
        model = get_tiny_model(get_model_path(self.mxfp4_model_path))
        assert (
            model.model.layers[0].mlp.up_proj.weight_packed.dtype == torch.uint8
        ), "Original weight is not in FP8 format"
        assert hasattr(
            model.model.layers[0].mlp.up_proj, "quantization_scheme"
        ), "Model does not contain CompressedLinear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.MXFP4 in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert (
            model.model.layers[0].mlp.up_proj.weight.dtype == torch.bfloat16
        ), "CompressedLinear layer was not converted to Linear"

    def test_w4a16(self):
        model = get_tiny_model(get_model_path(self.w4a16_model_path))
        assert (
            model.model.layers[0].mlp.up_proj.weight_packed.dtype == torch.int32
        ), "Original weight is not in INT4 format"
        assert hasattr(
            model.model.layers[0].mlp.up_proj, "quantization_scheme"
        ), "Model does not contain CompressedLinear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.WOQ in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert (
            model.model.layers[0].mlp.up_proj.weight.dtype == torch.bfloat16
        ), "CompressedLinear layer was not converted to Linear"

    def test_w4a16_to_mxfp4(self, tmp_path):
        model = get_tiny_model(get_model_path(self.w4a16_model_path))
        model.config.name_or_path = None  # Clear the name_or_path to avoid MTP copying issues
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.w4a16_model_path)
        ar = AutoRound(
            model,
            tokenizer=tokenizer,
            scheme="MXFP4",
            iters=2,
            nsamples=2,
        )
        _, quantized_model_path = ar.quantize_and_save(tmp_path, format="llm_compressor")
        model = transformers.AutoModelForCausalLM.from_pretrained(quantized_model_path)
        assert model, "Failed to load the quantized model"


@pytest.mark.skipif(MXFP4_SCALE_DTYPE is None, reason="torch.float8_e8m0fnu is required for standalone MXFP4 tests")
class TestStandaloneMXFP4Input:
    @pytest.mark.parametrize("packed_dtype", [torch.uint8, torch.int8])
    def test_detects_standalone_mxfp4_layer(self, packed_dtype):
        model = TinyStandaloneMXFP4Model()
        model.proj = build_standalone_mxfp4_layer(packed_dtype=packed_dtype)
        detected_types = check_and_mark_quantized_module(model)

        assert ModuleWeightType.MXFP4 in detected_types
        assert model.proj.quantized_weight_type == ModuleWeightType.MXFP4

    @pytest.mark.parametrize("packed_dtype", [torch.uint8, torch.int8])
    def test_converts_standalone_mxfp4_layer_to_dense_linear(self, packed_dtype):
        model = TinyStandaloneMXFP4Model()
        model.proj = build_standalone_mxfp4_layer(packed_dtype=packed_dtype)
        expected_weight = dequantize_standalone_mxfp4_weight(model.proj).to(torch.bfloat16)
        expected_bias = model.proj.bias.detach().to(torch.bfloat16)

        check_and_mark_quantized_module(model)
        model = convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16)

        assert type(model.proj) is nn.Linear
        assert model.proj.weight.dtype == torch.bfloat16
        torch.testing.assert_close(model.proj.weight, expected_weight)
        torch.testing.assert_close(model.proj.bias, expected_bias)

    def test_ignores_stale_quantized_weight_type_after_replacement(self):
        model = nn.Sequential(nn.Linear(8, 4))
        model[0].quantized_weight_type = ModuleWeightType.MXFP4

        model = convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16)

        assert type(model[0]) is nn.Linear
        assert model[0].quantized_weight_type is None

    def test_ignores_meta_standalone_mxfp4_shell_during_cleanup(self):
        model = TinyStandaloneMXFP4Model()

        check_and_mark_quantized_module(model)
        model.proj.to("meta")
        model = convert_module_to_hp_if_necessary(model, dtype=torch.bfloat16)

        assert model.proj.weight.device.type == "meta"
        assert model.proj.quantized_weight_type is None

    def test_prepares_standalone_mxfp4_for_shard_write(self):
        model = TinyStandaloneMXFP4Model()
        expected_weight = dequantize_standalone_mxfp4_weight(model.proj).to(torch.bfloat16)
        expected_bias = model.proj.bias.detach().to(torch.bfloat16)

        check_and_mark_quantized_module(model)
        prepared = prepare_module_for_shard_write_if_necessary(model, "proj", dtype=torch.bfloat16, device="cpu")

        assert type(prepared) is nn.Linear
        assert type(model.proj) is nn.Linear
        torch.testing.assert_close(model.proj.weight, expected_weight)
        torch.testing.assert_close(model.proj.bias, expected_bias)

    def test_replaces_standalone_deepseek_v4_experts_with_quantizable_linears(self):
        class DummyConfig:
            model_type = "deepseek_v4"

        class StandaloneDeepseekV4Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.act_fn = nn.SiLU()
                self.limit = 5.0
                self.w1 = build_standalone_mxfp4_layer(in_features=64, out_features=32, has_bias=False)
                self.w3 = build_standalone_mxfp4_layer(in_features=64, out_features=32, has_bias=False)
                self.w2 = build_standalone_mxfp4_layer(in_features=32, out_features=64, has_bias=False)

        class StandaloneDeepseekV4Experts(nn.ModuleList):
            def __init__(self):
                super().__init__([StandaloneDeepseekV4Expert() for _ in range(2)])

        class TinyStandaloneDeepseekV4Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = DummyConfig()
                self.experts = StandaloneDeepseekV4Experts()

        model = TinyStandaloneDeepseekV4Model()
        expected_gate_weight = dequantize_standalone_mxfp4_weight(model.experts[0].w1).to(torch.bfloat16)
        expected_up_weight = dequantize_standalone_mxfp4_weight(model.experts[0].w3).to(torch.bfloat16)
        expected_down_weight = dequantize_standalone_mxfp4_weight(model.experts[0].w2).to(torch.bfloat16)

        model = update_module(model, cleanup_original=False)
        materialize_model_(model)

        expert = model.experts[0]
        assert type(expert.gate_proj) is nn.Linear
        assert type(expert.up_proj) is nn.Linear
        assert type(expert.down_proj) is nn.Linear
        torch.testing.assert_close(expert.gate_proj.weight, expected_gate_weight)
        torch.testing.assert_close(expert.up_proj.weight, expected_up_weight)
        torch.testing.assert_close(expert.down_proj.weight, expected_down_weight)

        state_keys = set(model.state_dict().keys())
        assert "experts.0.gate_proj.weight" in state_keys
        assert "experts.0.up_proj.weight" in state_keys
        assert "experts.0.down_proj.weight" in state_keys
        assert "experts.0.w1.weight" not in state_keys
