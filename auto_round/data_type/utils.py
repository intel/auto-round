# Copyright (c) 2024 Intel Corporation
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

import math
from functools import lru_cache
from typing import List

from auto_round.compressors.utils import is_nv_fp
import torch
from torch.nn import Linear, Module

from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE
from auto_round.utils import logger, check_to_quantized


def reshape_pad_tensor_by_group_size(data: torch.Tensor, group_size: int, val: float = 0.0):
    """Reshapes and pads the tensor to ensure that it can be quantized in groups of `group_size`.

    This function adjusts the
    input tensor's shape so that its last dimension is a multiple
    of the specified `group_size`. If padding is required, it adds padding to the tensor
    to achieve this. If the tensor's last dimension is already divisible by `group_size`,
    no padding is applied.

    Args:
        data (torch.Tensor): The input tensor to be reshaped and padded.
        group_size (int): The size of the groups that the tensor should be reshaped into.

    Returns:
        torch.Tensor: The reshaped and padded tensor, if necessary.
        tuple: The original shape of the input tensor.
        int: The padding length applied to the tensor. Returns 0 if no padding is applied.
    """
    orig_shape = data.shape
    pad_len = 0
    if group_size == 0:
        data = data.reshape(1, -1)
        return data, orig_shape, pad_len
    if len(data.shape) > 2:
        data = data.reshape(-1, orig_shape[-1])
    if group_size == -1 or data.shape[1] < group_size:
        return data, orig_shape, pad_len
    elif data.shape[1] % group_size == 0:
        data = data.reshape(-1, group_size)
        return data, orig_shape, pad_len
    else:
        pad_len = (data.shape[1] + group_size - 1) // group_size * group_size - data.shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len), value=val)
        data_new = data_new.reshape(-1, group_size)
        return data_new, orig_shape, pad_len


def revert_tensor_by_pad(data: torch.Tensor, orig_shape: tuple, pad_len: int):
    """Reverts the tensor to its original shape by removing padding.

    This function removes the padding added during reshaping and returns the tensor to
    its original shape.

    Args:
        data (torch.Tensor): The reshaped and possibly padded tensor.
        orig_shape (tuple): The original shape of the tensor before reshaping.
        pad_len (int): The length of the padding to be removed.

    Returns:
        torch.Tensor: The tensor restored to its original shape.
    """
    if pad_len == 0:
        return data.reshape(orig_shape)
    else:
        if len(orig_shape) > 2:
            tmp_shape = torch.prod(torch.tensor(orig_shape[:-1])).item()
        else:
            tmp_shape = orig_shape[0]
        data_new = data.reshape(tmp_shape, -1)
        data_new = data_new[:, :-pad_len]
        data_new = data_new.reshape(orig_shape)
        return data_new


def get_quant_func(dtype: str, bits: int, sym: bool, disable_opt_rtn=False) -> tuple[callable, str]:
    """Retrieve the quantization function based on data type, bit width, and symmetry.

    This function returns the appropriate quantization function from the QUANT_FUNC_WITH_DTYPE
    dictionary based on the provided data type (`dtype`), bit width (`bits`), and whether
    the quantization is symmetric (`sym`). If the function does not exist, raise ValueError.

    Args:
        dtype (str): The data type for the quantization (e.g., 'int', 'mxfp4').
        bits (int): The bit width for the quantization (e.g., 2,4,8).
        sym (bool): A flag indicating whether the quantization is symmetric (True) or asymmetric (False).
        disable_opt_rtn(bool): whether to disable optimized rtn.

    Returns:
        function: The quantization function corresponding to the specified parameters.
        str
    """

    def pad_sym(data_type):
        if sym:
            data_sym = data_type + "_sym"
        else:
            data_sym = data_type + "_asym"
        return data_sym

    def pad_bits(data_type):
        return data_type + str(bits)

    if not disable_opt_rtn:
        rtn_data_type = "rtn_" + dtype
        data_types = [rtn_data_type, pad_bits(rtn_data_type), pad_sym(rtn_data_type), pad_sym(pad_bits(rtn_data_type))]
        for data_type in data_types:
            from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

            if data_type in QUANT_FUNC_WITH_DTYPE:
                return QUANT_FUNC_WITH_DTYPE[data_type], data_type

    data_types = [dtype, pad_bits(dtype), pad_sym(dtype), pad_sym(pad_bits(dtype))]
    for data_type in data_types:
        from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

        if data_type in QUANT_FUNC_WITH_DTYPE:
            return QUANT_FUNC_WITH_DTYPE[data_type], data_type


def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    """Straight-Through Estimator for floor.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.floor() - x).detach() + x


def ceil_ste(x: torch.Tensor):
    """Straight-Through Estimator for ceil.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.ceil() - x).detach() + x


@torch._dynamo.disable()
def float8_e4m3fn_ste(x: torch.Tensor):
    """Straight-Through Estimator (STE) for float8.

    Applies a quantization and dequantization step with float8 precision while maintaining
    gradient flow using a straight-through estimator.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Quantized and dequantized tensor using float8 format.
    """
    fp8 = (x.to(torch.float8_e4m3fn).to(x.dtype) - x).detach() + x

    return fp8


def float8_e5m2_ste(x: torch.Tensor):
    """Straight-Through Estimator (STE) for float8.

    Applies a quantization and dequantization step with float8 precision while maintaining
    gradient flow using a straight-through estimator.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Quantized and dequantized tensor using float8 format.
    """
    fp8 = (x.to(torch.float8_e5m2).to(x.dtype) - x).detach() + x

    return fp8


def float8_e4m3fn_hpu_ste(x: torch.Tensor):
    """Straight-Through Estimator (STE) for float8.

    Applies a quantization and dequantization step with float8 precision while maintaining
    gradient flow using a straight-through estimator.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Quantized and dequantized tensor using float8 format.
    """
    fp8 = ((torch.ops.hpu.cast_to_fp8_v2(x, 1.0, False, False, torch.float8_e4m3fn)[0]).to(x.dtype) - x).detach() + x

    return fp8


def float8_e4m3fnuz_hpu_ste(x: torch.Tensor):
    """Straight-Through Estimator (STE) for float8.

    Applies a quantization and dequantization step with float8 precision while maintaining
    gradient flow using a straight-through estimator.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Quantized and dequantized tensor using float8 format.
    """
    fp8 = ((torch.ops.hpu.cast_to_fp8_v2(x, 1.0, False, False, torch.float8_e4m3fn)[0]).to(x.dtype) - x).detach() + x
    return fp8


@lru_cache(None)
def get_gaudi_fp8_ste_func():
    from auto_round.utils import is_hpex_available

    if is_hpex_available():
        fn = float8_e4m3fn_hpu_ste
        logger.warning_once("Using HPU STE for FP8")
    else:
        fn = float8_e4m3fn_ste
        logger.warning_once("Using CUDA/CPU STE for FP8")
    return fn


# please refer from https://github.com/vllm-project/llm-compressor/blob/
# 29f4d5644b48e9c8ebb7e36d5be9f7c92747ceb7/src/llmcompressor/modifiers/utils/helpers.py#L11
def update_fused_layer_global_scales(
    submodule: Module,
    base_name: str = "weight",
):
    """
    Update global scales for fused layers under NVFP4 quantization.

    For attention layers:
      - q/k/v projections share a single global scale.

    For MLP layers:
      - gate_proj and up_proj share a single global scale.

    This behavior is currently required by vLLM and may become optional
    in the future.
    """
    global_scale_name = f"{base_name}_global_scale"

    def _collect_scales(mods: List[Module]) -> List[torch.Tensor]:
        """Collect valid global_scale tensors from modules."""
        scales = []
        for m in mods:
            if hasattr(m, global_scale_name):
                scale = getattr(m, global_scale_name)
                if isinstance(scale, torch.Tensor):
                    # Normalize shape early
                    scales.append(scale.reshape(1))
        return scales

    def _is_attention_module(module: Module):
        return "attention" in module.__class__.__name__.lower() and (
            hasattr(module, "k_proj") or hasattr(module, "v_proj") or hasattr(module, "qkv_proj")
        )

    def _is_mlp_module(module: Module):
        return "mlp" in module.__class__.__name__.lower() and (
            hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        )

    def _update_global_scales(modules: List[Module]):
        """Update global scales for a list of modules."""
        scales = _collect_scales(modules)
        if not scales:
            return

        # Move all scales to the same device before stacking
        target_device = scales[0].device
        scales_on_device = [s.to(target_device) for s in scales]
        global_scale = torch.min(torch.stack(scales_on_device), dim=0).values

        for proj in modules:
            if hasattr(proj, global_scale_name):
                # Move global_scale to the same device as the projection's current scale
                proj_scale = getattr(proj, global_scale_name)
                setattr(proj, global_scale_name, global_scale.clone().to(proj_scale.device))

    # ---------------- Attention ----------------
    if _is_attention_module(submodule):
        # Already fused
        if hasattr(submodule, "qkv_proj"):
            return
        _update_global_scales([submodule.q_proj, submodule.k_proj, submodule.v_proj])
        return

    # ---------------- MLP ----------------
    if _is_mlp_module(submodule):
        _update_global_scales([submodule.gate_proj, submodule.up_proj])


def update_block_global_scale_if_needed(block, data_type, group_size):
    if not is_nv_fp(data_type):
        return
    
    from auto_round.data_type.nvfp import calculate_gparam
    # Calculate block wise weight global scale
    for _, m in block.named_modules():
        if check_to_quantized(m) and not hasattr(m, "weight_global_scale"):
            weight_global_scale = calculate_gparam(m.weight, group_size)
            setattr(m, "weight_global_scale", weight_global_scale)
    
    # Update fused layer global scales
    for module in block.modules():
        update_fused_layer_global_scales(module)