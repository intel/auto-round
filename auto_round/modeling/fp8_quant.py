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

from auto_round.utils import is_transformers_version_greater_or_equal_5
from auto_round.utils import logger as auto_round_logger


# Patching replace_with_fp8_linear to disable expert replacement
# https://github.com/huggingface/transformers/blob/78bb85146c59258a0710c8d08311d98d52303c38/src/transformers/integrations/finegrained_fp8.py#L720
def oot_replace_with_fp8_linear(
    model, modules_to_not_convert: list[str] | None = None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `FP8Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `None`):
            Names of the modules to not convert. In practice we keep the `lm_head` in full precision for numerical stability reasons.
        quantization_config (`FbgemmFp8Config`):
            The quantization config object that contains the quantization parameters.
        pre_quantized (`book`, defaults to `False`):
            Whether the model is pre-quantized or not
    """
    import torch
    from transformers.integrations.finegrained_fp8 import (
        FP8Linear,
        logger,
        should_convert_module,
    )

    if quantization_config.dequantize:
        return model

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        # we need this to correctly materialize the weights during quantization
        module_kwargs = {} if pre_quantized else {"dtype": None}
        new_module = None
        with torch.device("meta"):
            # Note: Disable replacing experts, as we do not want concatenated experts
            # if module_name.endswith(".experts"):
            #     new_module = FP8Expert(
            #         config=model.config, block_size=quantization_config.weight_block_size, **module_kwargs
            #     )
            # elif isinstance(module, nn.Linear):
            if isinstance(module, torch.nn.Linear):
                new_module = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    activation_scheme=quantization_config.activation_scheme,
                    block_size=quantization_config.weight_block_size,
                    **module_kwargs,
                )
            if new_module is not None:
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )
    return model


def apply_fp8_expert_replacement_patch():
    if is_transformers_version_greater_or_equal_5():
        import transformers.integrations.finegrained_fp8 as transformers_fp8

        transformers_fp8.replace_with_fp8_linear = oot_replace_with_fp8_linear
        auto_round_logger.debug("Applied FP8 expert replacement patch to transformers.")


apply_fp8_expert_replacement_patch()
