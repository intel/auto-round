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

"""
MLX Format Exporter for AutoRound

Exports quantized models to MLX-compatible format that can be loaded
directly by mlx-lm for inference on Apple Silicon.

MLX QuantizedLinear dequantization (affine mode):
    w_float = scale * w_int + bias

where w_int is packed into uint32, each holding (32 // bits) elements,
and scale/bias (called "biases" in MLX) have shape [out_features, num_groups].

Supported schemes: W2A16, W3A16, W4A16, W8A16
"""

import copy
import json
import os
from typing import Callable, Union

import torch
import torch.nn as nn
import transformers

from auto_round.export.utils import unsupported_meta_device
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    set_module,
    get_packing_device,
)

SUPPORTED_LAYER_TYPES = [nn.Linear, nn.Conv2d, transformers.pytorch_utils.Conv1D]


def _pack_weight_mlx(intweight, bits):
    """Pack integer weights into uint32 in MLX format.

    MLX packs elements as a contiguous bit stream across uint32 boundaries.
    For bits that evenly divide 32 (2, 4, 8), each uint32 holds 32//bits elements.
    For other bit widths (e.g. 3, 5, 6, 7), every 32 elements are packed into
    `bits` uint32s (32 * bits = bits * 32 bits total).

    Args:
        intweight: shape [out_features, in_features], values in [0, 2^bits - 1]
        bits: quantization bits (2, 3, 4, 5, 6, 7, 8)

    Returns:
        packed: uint32 tensor, shape [out_features, in_features * bits / 32]
    """
    out_features, in_features = intweight.shape

    if 32 % bits == 0:
        # Simple case: bits evenly divides 32 (2, 4, 8)
        elems_per_int = 32 // bits
        assert in_features % elems_per_int == 0, \
            f"in_features ({in_features}) must be divisible by {elems_per_int} for {bits}-bit packing"

        intweight = intweight.to(torch.int32)
        intweight = intweight.reshape(out_features, -1, elems_per_int)
        shifts = torch.arange(elems_per_int, device=intweight.device, dtype=torch.int32) * bits
        packed = (intweight << shifts).sum(dim=-1).to(torch.int32)
        return packed.view(torch.uint32)
    else:
        # Cross-word packing: 32 elements → `bits` uint32s
        # MLX packs as a contiguous bit stream across uint32 boundaries
        assert in_features % 32 == 0, \
            f"in_features ({in_features}) must be divisible by 32 for {bits}-bit packing"

        intweight = intweight.to(torch.int64)
        num_groups = in_features // 32
        # Reshape to [out_features, num_groups, 32]
        elems = intweight.reshape(out_features, num_groups, 32)

        # For each element i in [0..31], it contributes `bits` bits starting at bit position i*bits
        # in a 32*bits bit stream packed into `bits` uint32s.
        # We process each bit b of each element i:
        #   absolute_bit = i * bits + b
        #   word_idx = absolute_bit // 32
        #   bit_pos  = absolute_bit % 32
        packed = torch.zeros(out_features, num_groups, bits, dtype=torch.int64, device=intweight.device)

        for b in range(bits):
            # Extract bit b from all elements: shape [out, num_groups, 32]
            bit_vals = (elems >> b) & 1
            for i in range(32):
                abs_bit = i * bits + b
                word_idx = abs_bit // 32
                bit_pos = abs_bit % 32
                packed[:, :, word_idx] |= bit_vals[:, :, i] << bit_pos

        packed = packed.to(torch.int32).reshape(out_features, num_groups * bits)
        return packed.view(torch.uint32)


class _MLXPackedLayer(nn.Module):
    """Holds MLX-packed quantized tensors for serialization.

    Tensor names match MLX convention: weight, scales, biases, bias.
    """

    def __init__(self, weight, scales, biases, bias):
        super().__init__()
        self.register_buffer("weight", weight)    # uint32
        self.register_buffer("scales", scales)    # float16
        self.register_buffer("biases", biases)    # float16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None


def pack_layer(name, model, device=None, **kwargs):
    """Pack a single layer into MLX quantized format.

    Reads layer attributes set by auto-round quantization:
        layer.weight  - float weight [out_features, in_features]
        layer.scale   - [out_features, num_groups]
        layer.zp      - [out_features, num_groups] or scalar
        layer.bits, layer.group_size, layer.sym

    Replaces the layer with _MLXPackedLayer containing:
        weight  - uint32 packed ints [out_features, in_features * bits / 32]
        scales  - float16 [out_features, num_groups]
        biases  - float16 [out_features, num_groups]
        bias    - original linear bias (if any)
    """
    layer = get_module(model, name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if type(layer) not in SUPPORTED_LAYER_TYPES:
        return

    if not check_to_quantized(layer):
        return

    device = get_packing_device(device)

    bits = int(layer.bits)
    group_size = int(layer.group_size)
    scale = layer.scale  # [out_features, num_groups]
    zp = layer.zp        # [out_features, num_groups] or scalar

    # Get weight in [out_features, in_features] layout
    W = layer.weight.data.to(device).clone().float()
    if type(layer) == nn.Conv2d:
        W = W.flatten(1)
    if type(layer) == transformers.pytorch_utils.Conv1D:
        W = W.t()

    out_features, in_features = W.shape
    if group_size == -1:
        group_size = in_features

    maxq = 2 ** bits - 1

    # Quantize: w_int = round(W / scale + zp), clamped to [0, maxq]
    scale_dev = scale.to(device).float()
    repeat_scales = scale_dev.repeat_interleave(group_size, dim=1)[:, :in_features]

    if isinstance(zp, torch.Tensor):
        zp_dev = zp.to(device).float()
        repeat_zp = zp_dev.repeat_interleave(group_size, dim=1)[:, :in_features]
    else:
        zp_dev = zp
        repeat_zp = zp

    intweight = torch.round(W / repeat_scales + repeat_zp).clamp(0, maxq).to(torch.int32)

    # Pack weights into uint32
    packed_weight = _pack_weight_mlx(intweight, bits)

    # MLX dequant: w_float = mlx_scale * w_int + mlx_bias
    # auto-round:  w_float = scale * (w_int - zp) = scale * w_int - scale * zp
    # So: mlx_scale = scale, mlx_bias = -scale * zp
    mlx_scales = scale_dev.contiguous().to(torch.float16)

    if isinstance(zp_dev, torch.Tensor):
        mlx_biases = (-scale_dev * zp_dev).contiguous().to(torch.float16)
    else:
        mlx_biases = (-scale_dev * zp_dev).contiguous().to(torch.float16)

    # Preserve original bias
    orig_bias = None
    if layer.bias is not None:
        orig_bias = layer.bias.clone().to(torch.float16).cpu()

    new_layer = _MLXPackedLayer(
        packed_weight.cpu(),
        mlx_scales.cpu(),
        mlx_biases.cpu(),
        orig_bias,
    )
    set_module(model, name, new_layer)
    logger.debug(f"Packed layer {name} for MLX format (bits={bits}, group_size={group_size})")


def save_quantized_as_mlx(
    output_dir: str,
    model: nn.Module = None,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace: bool = True,
    device: Union[str, torch.device] = "cpu",
    serialization_dict: dict = None,
    **kwargs,
) -> nn.Module:
    """Save quantized model in MLX-compatible format.

    The output can be loaded by mlx-lm::

        from mlx_lm import load, generate
        model, tokenizer = load("output_dir")
        response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
    """
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    if not unsupported_meta_device(model):
        model = model.to("cpu")

    os.makedirs(output_dir, exist_ok=True)

    bits = serialization_dict.get("bits", 4) if serialization_dict else 4
    group_size = serialization_dict.get("group_size", 128) if serialization_dict else 128
    if group_size == -1:
        group_size = 128

    # Pack all quantized layers (skip if already packed by immediate_pack)
    if layer_config:
        for layer_name in layer_config:
            pack_layer(layer_name, model, device=device)

    # Save model weights (uint32 packed weights are saved directly by safetensors)
    if not unsupported_meta_device(model):
        model.save_pretrained(output_dir, safe_serialization=True)

    # Write config.json with MLX-compatible quantization info
    config_path = os.path.join(output_dir, "config.json")
    if hasattr(model, "config") and not os.path.exists(config_path):
        model.config.save_pretrained(output_dir)

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        quant_cfg = {"group_size": group_size, "bits": bits}
        config["quantization"] = quant_cfg
        config["quantization_config"] = quant_cfg
        # Flatten rope_parameters for mlx-lm (expects rope_theta at top level)
        rope_params = config.pop("rope_parameters", None)
        if rope_params and isinstance(rope_params, dict):
            if "rope_theta" in rope_params and "rope_theta" not in config:
                config["rope_theta"] = rope_params["rope_theta"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Save tokenizer
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    # Copy processor if available
    processor = kwargs.get("processor", None)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)

    # Try to copy Python files from model cache
    try:
        copy_python_files_from_model_cache(model, output_dir)
    except Exception as e:
        logger.warning(f"Failed to copy Python files from model cache: {e}")

    logger.info(f"Model saved to {output_dir} in MLX format (bits={bits}, group_size={group_size})")
    logger.info(f"Load with: from mlx_lm import load, generate; "
                f"model, tokenizer = load('{output_dir}')")

    return model

