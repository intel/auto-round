# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

import torch
import torch.nn as nn
import transformers.quantizers as qz

# import transformers.integrations.finegrained_fp8 as fgfp8
import transformers.quantizers.auto as qa
import transformers.quantizers.quantizer_finegrained_fp8 as qf8

# from transformers.utils.import_utils import is_grouped_mm_available
from transformers import PreTrainedModel

# from transformers.models.deepseek_v2.modeling_deepseek_v2 import ACT2FN
# from transformers.models.deepseek_v2.modular_deepseek_v2 import  DeepseekV2Config, DeepseekV2DecoderLayer, DeepseekV2Attention, DeepseekV2PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.quantizers import quantizer_finegrained_fp8
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import auto_docstring

logger = logging.getLogger(__name__)


from accelerate import init_empty_weights

# from auto_round.utils.model import dequant_block_fp8_weight


def _pad_weight(weight: torch.Tensor, block_size: list) -> tuple[torch.Tensor, int, int]:
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode="constant", value=0)
    return padded_weight, M, N  # Return original dimensions for unpadding


def _unpad_weight(weight: torch.Tensor, original_M: int, original_N: int, keep_first_dim: bool = False) -> torch.Tensor:
    """Removes padding from the matrix to restore its original shape."""
    if (weight.shape[-2] == original_M) and (weight.shape[-1] == original_N):
        return weight
    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]


def pad_block_fp8_weight_naive(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_size: list
) -> tuple[torch.Tensor, int, int]:
    assert len(block_size) == 2

    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = _pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n

    return weight, orig_M, orig_N


def dequant_block_fp8_weight(weight: torch.Tensor, weight_scale: torch.Tensor, block_size: list) -> torch.Tensor:
    dtype = torch.bfloat16
    if weight_scale is None:
        return weight
    assert len(block_size) == 2

    weight, orig_M, orig_N = pad_block_fp8_weight_naive(weight, weight_scale, block_size)

    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    dequant_weight = _unpad_weight(dequant_weight, orig_M, orig_N, keep_first_dim=keep_first_dim)

    return dequant_weight


def _replace_with_fp8_linear(
    model,
    tp_plan=None,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """Replace Linear layers with FP8Linear."""
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in (modules_to_not_convert or []):
            current_key_name_str = ".".join(current_key_name)
            if not any(key in current_key_name_str for key in (modules_to_not_convert or [])):
                with init_empty_weights():
                    # print(f"replacing module at {name} with FP8Linear")
                    # breakpoint()
                    model._modules[name] = FP8Linear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        activation_scheme=quantization_config.activation_scheme,
                        block_size=quantization_config.weight_block_size,
                    )
                    has_been_replaced = True
            # when changing a layer the TP PLAN for that layer should be updated. TODO

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_fp8_linear(
                module,
                tp_plan,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )

        current_key_name.pop(-1)

    return model, has_been_replaced


def replace_with_fp8_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
):
    """Helper function to replace model layers with FP8 versions."""
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_fp8_linear(
        model,
        tp_plan=model._tp_plan,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )

    return model


class OOTFineGrainedFP8HfQuantizer(qf8.FineGrainedFP8HfQuantizer):
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        # from ..integrations.finegrained_fp8 import replace_with_fp8_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

        model.config.quantization_config = self.quantization_config

    def get_weight_conversions(self):
        return []

    def validate_environment(self, *args, **kwargs):
        return True

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        return []

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        # from ..integrations.finegrained_fp8 import FP8Linear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, FP8Linear):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False



class FP8Linear(nn.Linear):
    dtype = torch.float8_e4m3fn

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        block_size: Optional[tuple[int, int]] = None,
        device=None,
        activation_scheme="dynamic",
    ):
        super().__init__(in_features, out_features)

        # If block size is None, it means that we are doing per-tensor quantization
        self.block_size = block_size
        self.activation_scheme = activation_scheme

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=FP8Linear.dtype, device=device))

        if self.block_size is None:
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = (out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (in_features + self.block_size[1] - 1) // self.block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )

        if self.activation_scheme == "static":
            self.activation_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dequant_weight = dequant_block_fp8_weight(
            self.weight,
            self.weight_scale_inv,
            block_size=self.block_size,
        )
        dequant_weight = dequant_weight.to(input.dtype)
        out = torch.nn.functional.linear(input, dequant_weight, self.bias)
        return out.to(input.dtype)


def patch_fp8_quantizer():
    qf8.FineGrainedFP8HfQuantizer = OOTFineGrainedFP8HfQuantizer
    # Patch common aliases too (if re-exported)
    if hasattr(qz, "FineGrainedFP8HfQuantizer"):
        qz.FineGrainedFP8HfQuantizer = OOTFineGrainedFP8HfQuantizer
    # Ensure Auto dispatcher uses our override
    qa.AUTO_QUANTIZER_MAPPING["fp8"] = OOTFineGrainedFP8HfQuantizer

