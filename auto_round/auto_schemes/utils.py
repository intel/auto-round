# Copyright (c) 2025 Intel Corporation
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
import re
from dataclasses import asdict, fields
from typing import Iterable, Union

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from auto_round.low_cpu_mem import get_module
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    get_block_names,
    get_layer_features,
    is_hpex_available,
)


def apply_quant_scheme(
    model: torch.nn.Module,
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    scheme: Union[str, dict],  # TODO add scale_dtype
) -> None:
    """Apply a quantization scheme to each quantized layer.

    Args:
        model: The model whose layers are to be updated.
        scheme: The scheme preset name or dictionary to apply.
        quant_layer_names: Iterable of layer names to quantize.
        fixed_layer_scheme: Dictionary of fixed per-layer quantization schemes.
    """
    for name in quant_layer_names:
        layer_scheme = fixed_layer_scheme.get(name, scheme)
        if isinstance(layer_scheme, str):
            layer_scheme = asdict(preset_name_to_scheme(layer_scheme))

        module = get_module(model, name)
        for key, value in layer_scheme.items():
            setattr(module, key, value)


def remove_quant_scheme(
    model: torch.nn.Module,
) -> None:
    """Remove attributes corresponding to the applied quantization scheme.

    Args:
        model: The model whose layers are to be cleared.
    """
    scheme_keys = [f.name for f in fields(QuantizationScheme)] + ["scale_dtype"]
    for n, m in model.named_modules():
        for key in scheme_keys:
            if hasattr(m, key):
                delattr(m, key)


def compute_avg_bits_for_scheme(
    model: torch.nn.Module,
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    scheme: Union[str, dict, None] = None,
    ignore_scale_zp_bits: bool = False,
) -> tuple[float, float]:
    """Compute the average and total bit usage for the given quantization scheme.

    Args:
        model: The model to analyze.
        quant_layer_names: Iterable of layer names to include.
        fixed_layer_scheme: Dictionary of fixed per-layer quantization schemes.
        scheme: Optional scheme to temporarily apply before measuring.
        ignore_scale_zp_bits: If True, ignores overhead from scale and zero-points.

    Returns:
        A tuple (avg_bits, total_quantized_bits):
            avg_bits: Average bitwidth per parameter.
            total_quantized_bits: Total quantized bit count.
    """
    if scheme is not None:
        apply_quant_scheme(model, quant_layer_names, fixed_layer_scheme, scheme)

    total_params = 0
    total_quantized_bits = 0

    for name in quant_layer_names:
        module = get_module(model, name)
        # if isinstance(module,torch.nn.Embedding):
        #     continue
        if not hasattr(module, "weight"):
            continue
        total_params += module.weight.numel()
        layer_bits, _ = compute_layer_bits(module, ignore_scale_zp_bits)
        total_quantized_bits += layer_bits

    avg_bits = float(total_quantized_bits) / total_params

    if scheme is not None:
        remove_quant_scheme(model)

    return avg_bits, total_quantized_bits


def compute_avg_bits_for_model(model: torch.nn.Module, ignore_scale_zp_bits: bool = False):
    """Compute the average and total bit usage for the entire model.

    Args:
        model: The model to analyze.
        ignore_scale_zp_bits: If True, ignores overhead from scale and zero-points.
        if scheme is not None:
        apply_quant_scheme(model, quant_layer_names, fixed_layer_scheme, scheme)
    """

    total_params = 0
    total_quantized_bits = 0

    for n, module in model.named_modules():
        if not hasattr(module, "bits"):
            continue
        if not hasattr(module, "weight"):
            continue
        # if isinstance(module,torch.nn.Embedding): # Tricky setting for Embedding
        #     continue
        total_params += module.weight.numel()
        layer_bits, _ = compute_layer_bits(module, ignore_scale_zp_bits)
        total_quantized_bits += layer_bits

    avg_bits = float(total_quantized_bits) / total_params

    return avg_bits, total_quantized_bits


def compute_layer_bits(
    layer: torch.nn.Module,
    ignore_scale_zp_bits: bool = False,
) -> tuple[int, float]:
    """Compute total and average bitwidth for a single quantized layer.

    Args:
        layer: A PyTorch layer with quantization attributes.
        ignore_scale_zp_bits: Whether to ignore scale/zero-point overhead.

    Returns:
        A tuple (total_bits, avg_bits) representing bit usage.
    """
    weight = layer.weight
    n_param = weight.numel()
    weight_bits = getattr(layer, "bits", 16)
    group_size = getattr(layer, "group_size", 128)
    super_group_size = getattr(layer, "super_group_size", None)
    super_weight_bits = getattr(layer, "super_bits", None)

    # Unquantized layer or ignoring scale/zp overhead
    if weight_bits >= 16 or ignore_scale_zp_bits:
        if super_weight_bits is not None:  # reset gguf 16 bits to 32 bits, TODO gguf q4_0, q4_1 have bug (wenhua)
            if weight_bits >= 16:
                return 32 * n_param, 32

        return weight_bits * n_param, weight_bits if weight_bits<16 else 16

    in_features, out_features = get_layer_features(layer)

    # Determine number of groups based on group size
    if group_size > 0:
        n_group = out_features * (in_features + group_size - 1) // group_size
    elif group_size == 0:
        n_group = 1
    elif group_size == -1:
        n_group = out_features
    else:
        raise ValueError(f"Invalid group_size {group_size}")

    # Compute auxiliary bits (scales, zero-points, or double quantization)
    aux_total_bits = 0
    if not super_group_size:
        scale_bits = 16
        zp_bits = weight_bits
        aux_total_bits = n_group * (scale_bits + zp_bits)
    else:
        aux_total_bits += n_group * super_weight_bits * 2
        n_super_group = (n_group + super_group_size - 1) // super_group_size
        aux_total_bits += n_super_group * 32 * 2  # 32-bit scale and min_v

    total_bits = weight_bits * n_param + aux_total_bits
    avg_bits = total_bits / n_param
    return total_bits, avg_bits

def parse_all_available_device(device_map: Union[str, torch.device, int, dict, None] = None) -> list:
    """
    Parse the device map and return a list of all available devices.

    Supported input formats:
        - None: Automatically detect all available devices
        - int: A single device index (e.g., 0)
        - str: Examples:
            "cpu"
            "cuda:0,cuda:1"
            "0,1" (numeric device indices)
        - dict: Extract all device values from the dictionary
        - torch.device: e.g. torch.device("cuda:0")

    Returns:
        list[str]: Normalized device names, e.g., ["cuda:0", "cuda:1"] or ["cpu"]
    """

    # === Step 1. Detect available device types ===
    device_types = []
    if torch.cuda.is_available():
        device_types.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_types.append("xpu")
    if hasattr(torch, "hpu") and is_hpex_available():
        device_types.append("hpu")

    # Always include CPU as a fallback
    if not device_types:
        device_types = ["cpu"]

    # === Step 2. Parse different input formats ===
    if device_map is None:
        # Automatically detect one available device
        if "cuda" in device_types:
            return ["cuda:0"]
        elif "xpu" in device_types:
            return ["xpu:0"]
        elif "hpu" in device_types:
            return ["hpu:0"]
        else:
            return ["cpu"]

    if isinstance(device_map, torch.device):
        # Handle torch.device objects
        dev_type = device_map.type
        index = device_map.index
        if dev_type == "cpu":
            return ["cpu"]
        if index is None:
            index = 0
        return [f"{dev_type}:{index}"]

    if isinstance(device_map, int):
        # Integer input → use primary available device type
        device_type = device_types[0]
        return [f"{device_type}:{device_map}"] if device_type != "cpu" else ["cpu"]

    if isinstance(device_map, str):
        # Remove whitespace
        device_map = device_map.strip()
        if device_map.lower() == "cpu":
            return ["cpu"]

        # Split by commas
        parts = [x.strip() for x in device_map.split(",") if x.strip()]
        parsed = []
        for p in parts:
            if p.isdigit():
                # Numeric → assign to first available device type
                device_type = device_types[0]
                parsed.append(f"{device_type}:{p}" if device_type != "cpu" else "cpu")
            else:
                parsed.append(p)
        return parsed

    if isinstance(device_map, dict):
        # Extract all devices recursively from dict values
        devices = set()
        for v in device_map.values():
            devices.update(parse_all_available_device(v))
        return sorted(devices)

    raise TypeError(f"Unsupported device_map type: {type(device_map)}")


# Important Notice This dispatch does not follow dict device_map, just extract all available devices and use them
def dispatch_model_by_all_available_devices(
    model: torch.nn.Module, device_map: Union[str, int, dict, None]
) -> torch.nn.Module:
    if device_map is None:
        device_map = 0

    no_split_modules = getattr(model, "_no_split_modules", [])
    if device_map == "auto":
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=no_split_modules,
        )
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_modules)
        model = dispatch_model(model, device_map=device_map)
        return model

    devices = parse_all_available_device(device_map)

    if len(devices) == 1:
        model.to(devices[0])
        return model

    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=no_split_modules,
    )

    # Filter max_memory with devices
    #  assume only one GPU model
    new_max_memory = {}
    for device in devices:
        if ":" in device:
            device = int(device.split(":")[-1])
        elif device == "cpu":
            device = "cpu"
        else:
            raise ValueError(f"Unsupported device {device} in device_map: {device_map}")
        new_max_memory[device] = max_memory[device]

    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_modules)
    model = dispatch_model(model, device_map=device_map)
    return model


def merge_lists_unionfind(list_of_lists):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # 初始化并查集
    for lst in list_of_lists:
        for item in lst:
            if item not in parent:
                parent[item] = item
        for i in range(1, len(lst)):
            union(lst[0], lst[i])

    # 收集结果
    groups = {}
    for item in parent:
        root = find(item)
        groups.setdefault(root, []).append(item)
    return list(groups.values())


def parse_shared_layers(model: torch.nn.Module, shared_patterns: Iterable[Iterable[str]]) -> list[list[str]]:
    """
    Parse shared layer groups based on regex or substring matches.

    Args:
        model (torch.nn.Module): The model whose modules will be analyzed.
        shared_patterns (Iterable[Iterable[str]]):
            Each inner iterable defines one shared group. Each element can be:
              - a string: checked by full-name or substring match
              - a regex pattern: checked by re.fullmatch or re.search

    Returns:
        list[list[str]]: A list of matched shared layer groups.
    """
    if not shared_patterns:
        return []
    # Retrieve all high-level block names (for example, transformer blocks)
    for n, m in model.named_modules():
        m.tmp_name = n  # attach global name

    block_names = get_block_names(model, quant_vision=True)
    block_names = [item for sublist in block_names for item in sublist]

    # Collect all supported layer names from the model
    supported_layer_names = [name for name, module in model.named_modules() if type(module) in SUPPORTED_LAYER_TYPES]

    # Separate groups into those already fully matched and those requiring pattern matching
    direct_match_groups = []
    fuzzy_match_groups = []
    for group in shared_patterns:
        match_status = {name: (name in supported_layer_names) for name in group}
        if all(match_status.values()):
            direct_match_groups.append(list(match_status.keys()))
        else:
            fuzzy_match_groups.append(match_status)

    matched_groups = list(direct_match_groups)

    # Search each block for modules matching remaining patterns
    for block_name in block_names:
        block_module = get_module(model, block_name)
        block_layer_local_names = [
            name for name, module in block_module.named_modules() if type(module) in SUPPORTED_LAYER_TYPES
        ]
        block_layer_names = []
        for name in block_layer_local_names:
            module = get_module(block_module, name)
            block_layer_names.append(module.tmp_name)

        for group in fuzzy_match_groups:
            matched_layers = set()
            for pattern, is_direct in group.items():
                if is_direct:
                    matched_layers.add(pattern)
                    continue

                for layer_name in block_layer_names:
                    # Try regex match first
                    try:
                        if re.fullmatch(pattern, layer_name) or re.search(pattern, layer_name):
                            matched_layers.add(layer_name)
                            continue
                    except re.error:
                        pass  # Not a valid regex, fallback to substring matching

                    # Substring or partial match
                    if pattern in layer_name:
                        matched_layers.add(layer_name)

            if matched_layers:
                matched_groups.append(sorted(matched_layers))
    matched_groups = merge_lists_unionfind(matched_groups)
    return matched_groups
