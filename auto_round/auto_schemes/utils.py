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
from dataclasses import asdict, fields
from typing import Iterable, Union

import torch

from auto_round.low_cpu_mem import get_module
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme
from auto_round.utils import check_to_quantized, get_layer_features


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
        if super_weight_bits is not None:  # reset gguf 16 bits to 32 bits, TODO gguf q4_0, q4_1 may have bug
            return 32 * n_param, 32
        return weight_bits * n_param, 16.0

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
