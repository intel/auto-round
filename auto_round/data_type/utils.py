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

"""Shared tensor utilities and straight-through estimators for quantization.

This module provides helpers used across all data-type quantization modules:

- :func:`reshape_pad_tensor_by_group_size` / :func:`revert_tensor_by_pad` for
  group-wise tensor reshaping.
- Straight-through estimators (STE) for rounding, floor, ceil, and float8.
- :func:`get_quant_func` to look up a registered quantization function.
- :func:`update_fused_layer_global_scales` / :func:`update_block_global_scale_if_needed`
  for NVFP4 fused-layer global-scale synchronization.
"""

import math
from functools import lru_cache
from math import ceil
from typing import List, Union

import torch
from torch.nn import Linear, Module

from auto_round.compressors.utils import is_nv_fp
from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE
from auto_round.utils import check_to_quantized, logger


def reshape_pad_tensor_by_group_size(data: torch.Tensor, group_size: Union[int, list], val: float = 0.0):
    """Reshapes and pads the tensor to ensure that it can be quantized in groups of `group_size`.

    This function adjusts the
    input tensor's shape so that its last dimension is a multiple
    of the specified `group_size`. If padding is required, it adds padding to the tensor
    to achieve this. If the tensor's last dimension is already divisible by `group_size`,
    no padding is applied.

    Args:
        data (torch.Tensor): The input tensor to be reshaped and padded.
        group_size (int or tuple): The size of the groups that the tensor should be reshaped into.

    Returns:
        torch.Tensor: The reshaped and padded tensor, if necessary.
        tuple: The original shape of the input tensor.
        int: The padding length applied to the tensor. Returns 0 if no padding is applied.
    """
    orig_shape = data.shape
    pad_len = 0
    if isinstance(group_size, tuple):
        assert len(group_size) == 2, f"Only support 2D group_size, but get {len(group_size)}"
        M, N = group_size
        pad_len_m = ceil(orig_shape[0] / M) * M - orig_shape[0]
        pad_len_n = ceil(orig_shape[1] / N) * N - orig_shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len_n, 0, pad_len_m))
        data_new = data_new.view(data_new.shape[0] // M, M, data_new.shape[1] // N, N).permute(0, 2, 1, 3)
        return data_new, orig_shape, (pad_len_m, pad_len_n)
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
        pad_len = ceil(data.shape[1] / group_size) * group_size - data.shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len), value=val)
        data_new = data_new.reshape(-1, group_size)
        return data_new, orig_shape, pad_len


def revert_tensor_by_pad(data: torch.Tensor, orig_shape: tuple, pad_len: Union[int, list]):
    """Reverts the tensor to its original shape by removing padding.

    This function removes the padding added during reshaping and returns the tensor to
    its original shape.

    Args:
        data (torch.Tensor): The reshaped and possibly padded tensor.
        orig_shape (tuple): The original shape of the tensor before reshaping.
        pad_len (int or tuple): The length of the padding to be removed.

    Returns:
        torch.Tensor: The tensor restored to its original shape.
    """
    if isinstance(pad_len, tuple):
        assert len(pad_len) == 2, f"Only support 2D group_size, but get {len(pad_len)}"
        data = data.permute(0, 2, 1, 3).reshape(orig_shape[0] + pad_len[0], orig_shape[1] + pad_len[1])
        return data[: data.shape[0] - pad_len[0], : data.shape[1] - pad_len[1]]
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


def get_quant_func(dtype: str, bits: int, sym: bool, disable_opt_rtn=False, group_size=None) -> tuple[callable, str]:
    """Retrieve the quantization function based on data type, bit width, and symmetry.

    This function returns the appropriate quantization function from the QUANT_FUNC_WITH_DTYPE
    dictionary based on the provided data type (`dtype`), bit width (`bits`), and whether
    the quantization is symmetric (`sym`). If the function does not exist, raise ValueError.

    Args:
        dtype (str): The data type for the quantization (e.g., 'int', 'mxfp4').
        bits (int): The bit width for the quantization (e.g., 2,4,8).
        sym (bool): A flag indicating whether the quantization is symmetric (True) or asymmetric (False).
        disable_opt_rtn(bool): whether to disable optimized rtn.
        group_size (tuple): The block size for weight quantization (e.g., (128, 128)).

    Returns:
        function: The quantization function corresponding to the specified parameters.
        str
    """

    def pad_sym(data_type):
        """Append the symmetry suffix (``_sym`` or ``_asym``) to *data_type*.

        Args:
            data_type (str): Base data-type string.

        Returns:
            str: Data-type string with the appropriate symmetry suffix.
        """
        if sym:
            data_sym = data_type + "_sym"
        else:
            data_sym = data_type + "_asym"
        return data_sym

    def pad_bits(data_type):
        """Append the bit-width as a suffix to *data_type*.

        Args:
            data_type (str): Base data-type string.

        Returns:
            str: Data-type string with the bit-width suffix appended.
        """
        return data_type + str(bits)

    if not disable_opt_rtn:
        rtn_data_type = "rtn_" + dtype
        data_types = [rtn_data_type, pad_bits(rtn_data_type), pad_sym(rtn_data_type), pad_sym(pad_bits(rtn_data_type))]
        for data_type in data_types:
            from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

            if data_type in QUANT_FUNC_WITH_DTYPE:
                return QUANT_FUNC_WITH_DTYPE[data_type], data_type

    if group_size is not None and isinstance(group_size, tuple):
        block_data_type = "block_" + dtype
        data_types = [
            block_data_type,
            pad_bits(block_data_type),
            pad_sym(block_data_type),
            pad_sym(pad_bits(block_data_type)),
        ]

        from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

        for data_type in data_types:
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
    """Return the appropriate FP8 straight-through estimator for the current device.

    Checks whether the HPEx library is available (indicating an Intel Gaudi
    accelerator) and returns the HPU-specific STE; otherwise returns the
    standard CUDA/CPU STE.  The result is cached so the check happens only
    once per process.

    Returns:
        Callable: Either :func:`float8_e4m3fn_hpu_ste` (HPU) or
            :func:`float8_e4m3fn_ste` (CUDA/CPU).
    """
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
    """Update global scales for fused layers under NVFP4 quantization.

    For attention layers q/k/v projections share a single global scale equal
    to the element-wise minimum of their individual scales.  For MLP layers
    ``gate_proj`` and ``up_proj`` are treated similarly.  This behaviour is
    currently required by vLLM and may become optional in the future.

    Args:
        submodule (torch.nn.Module): The layer module to inspect and update.
            May be an attention block or an MLP block.
        base_name (str, optional): Prefix used to find the per-module global
            scale attribute (``f"{base_name}_global_scale"``).  Defaults to
            ``"weight"``.
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
        """Return True if *module* looks like an attention block with k/v projections.

        Args:
            module (torch.nn.Module): Module to inspect.

        Returns:
            bool: True when the class name contains ``"attention"`` and the
                module has ``k_proj``, ``v_proj``, or ``qkv_proj`` attributes.
        """
        return "attention" in module.__class__.__name__.lower() and (
            hasattr(module, "k_proj") or hasattr(module, "v_proj") or hasattr(module, "qkv_proj")
        )

    def _is_mlp_module(module: Module):
        """Return True if *module* looks like an MLP block with gate and up projections.

        Args:
            module (torch.nn.Module): Module to inspect.

        Returns:
            bool: True when the class name contains ``"mlp"`` and the module
                has both ``gate_proj`` and ``up_proj`` attributes.
        """
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
    """Compute and synchronize NVFP4 global scales for all quantizable layers in a block.

    If *data_type* is not an NVFP format this function returns immediately.
    Otherwise it:

    1. Calculates a per-layer ``weight_global_scale`` for every quantizable
       submodule that does not already have one.
    2. Calls :func:`update_fused_layer_global_scales` on every module so that
       fused attention (q/k/v) and MLP (gate/up) projections share the same
       global scale, as required by vLLM.

    Args:
        block (torch.nn.Module): The transformer block whose submodules are
            updated.
        data_type (str): Quantization data-type string (e.g. ``"nv_fp4"``).
            Non-NVFP types are silently skipped.
        group_size (int): Group size passed to :func:`calculate_gparam` for
            computing the global scale.
    """
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
