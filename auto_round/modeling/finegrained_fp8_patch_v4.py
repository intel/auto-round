# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from transformers.utils import is_accelerate_available, is_torch_available, logging

if is_torch_available():
    import torch
    import torch.nn as nn

    # import triton
    # import triton.language as tl
    from torch.nn import functional as F

if is_accelerate_available():
    from accelerate import init_empty_weights


logger = logging.get_logger(__name__)


logger = logging.get_logger(__name__)


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


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
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=FP8Linear.dtype, device=device))

        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size[0] - 1) // block_size[0]
            scale_in_features = (in_features + block_size[1] - 1) // block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("weight_scale_inv", None)

        self.block_size = block_size

        self.activation_scheme = activation_scheme

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)


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
