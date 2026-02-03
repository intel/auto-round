import torch
from compressed_tensors.linear.compressed_linear import CompressedLinear
from transformers.integrations import FP8Linear

from auto_round.utils.weight_handler import (
    ModuleWeightType,
    check_and_mark_quantized_module,
    convert_module_to_hp_if_necessary,
)

from ...helpers import get_model_path, get_tiny_model


class TestCompresseTensor:
    nvfp4_model_path = "kaitchup/Qwen3-0.6B-NVFP4"
    mxfp4_model_path = "QuixiAI/Llama-3.2-1B-MXFP4"
    fp8_block_model_path = "RedHatAI/Qwen3-0.6B-FP8-BLOCK"

    def test_fp8_block(self):
        model = get_tiny_model(get_model_path(self.fp8_block_model_path))
        assert isinstance(
            model.model.layers[0].mlp.up_proj, CompressedLinear
        ), "Model does not contain CompressedLinear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.FP8 in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert isinstance(
            model.model.layers[0].mlp.up_proj, torch.nn.Linear
        ), "CompressedLinear layer was not converted to Linear"

    def test_nvfp4(self):
        model = get_tiny_model(get_model_path(self.nvfp4_model_path))
        assert isinstance(
            model.model.layers[0].mlp.up_proj, CompressedLinear
        ), "Model does not contain CompressedLinear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.NVFP4 in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert isinstance(
            model.model.layers[0].mlp.up_proj, torch.nn.Linear
        ), "CompressedLinear layer was not converted to Linear"

    def test_mxfp4(self):
        model = get_tiny_model(get_model_path(self.mxfp4_model_path))
        assert isinstance(
            model.model.layers[0].mlp.up_proj, CompressedLinear
        ), "Model does not contain CompressedLinear layers"
        detected_types = check_and_mark_quantized_module(model)
        assert ModuleWeightType.MXFP4 in detected_types
        model = convert_module_to_hp_if_necessary(model)
        assert isinstance(
            model.model.layers[0].mlp.up_proj, torch.nn.Linear
        ), "CompressedLinear layer was not converted to Linear"
