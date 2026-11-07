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

import torch

from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    extract_block_names_to_str,
    get_block_names,
)
from auto_round.utils.model import get_layer_names_in_block


def _is_supported_layer(module: torch.nn.Module) -> bool:
    """Check if module is a supported quantizable layer type."""
    return type(module) in SUPPORTED_LAYER_TYPES or module.__class__.__name__ in INNER_SUPPORTED_LAYER_TYPES


def _collect_modules_to_not_convert(
    model: torch.nn.Module,
    layer_config: dict,
    regex_config: dict,
    to_quant_block_names: list | None = None,
) -> list:
    """Collect all module names that should not be converted (not quantized).

    Args:
        model: The model to scan
        layer_config: Configuration dict for layers
        regex_config: Regex-based configuration for layers
        to_quant_block_names: List of block names to quantize (for MLLM models)

    Returns:
        List of module names to not convert
    """
    modules_to_not_convert = set()
    all_blocks = get_block_names(model, quant_vision=True)
    # 1. add non-quantized block directly
    if to_quant_block_names:
        all_block_names = extract_block_names_to_str(all_blocks).split(",")
        to_quant_set = set(
            to_quant_block_names.split(",") if isinstance(to_quant_block_names, str) else to_quant_block_names
        )
        non_quant_blocks = set(all_block_names) - to_quant_set
        modules_to_not_convert.update(non_quant_blocks)

    layers_in_blocks = set(get_layer_names_in_block(model, quant_block_list=all_blocks))

    # 2. Collect non-quantized layers from layer_config
    layers_from_block_patterns = set()
    for layer_name, layer_cfg in layer_config.items():
        if not check_to_quantized(layer_cfg) and not any(name in layer_name for name in modules_to_not_convert):
            layers_from_block_patterns.add(layer_name)
    modules_to_not_convert.update(layers_from_block_patterns)

    # 3. Scan full model for supported layers not in layer_config and not in blocks
    for module_name, module in model.named_modules():
        if _is_supported_layer(module):
            # If this layer is not in layer_config, it wasn't quantized
            if module_name not in layer_config and module_name not in layers_in_blocks:
                # Standalone layer outside blocks
                modules_to_not_convert.add(module_name)

    # 4. Add high-precision layers from regex_config (bits > 8)
    for regex_name, regex_cfg in regex_config.items():
        bits = regex_cfg.get("bits")
        if bits and int(bits) > 8:
            modules_to_not_convert.add(regex_name)

    return list(modules_to_not_convert)
