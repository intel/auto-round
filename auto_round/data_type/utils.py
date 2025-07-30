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

from functools import lru_cache

import torch

from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE
from auto_round.utils import logger


def reshape_pad_tensor_by_group_size(data: torch.Tensor, group_size: int):
    """Reshapes and pads the tensor to ensure that it can be quantized in groups of `group_size`.

    This function adjusts t
    he input tensor's shape so that its last dimension is a multiple
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
        data_new = torch.nn.functional.pad(data, (0, pad_len))
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


def get_quant_func(dtype, bits, sym):
    """Retrieve the quantization function based on data type, bit width, and symmetry.

    This function returns the appropriate quantization function from the QUANT_FUNC_WITH_DTYPE
    dictionary based on the provided data type (`dtype`), bit width (`bits`), and whether
    the quantization is symmetric (`sym`). If the function does not exist, raise ValueError.

    Args:
        dtype (str): The data type for the quantization (e.g., 'int', 'mxfp4').
        bits (int): The bit width for the quantization (e.g., 2,4,8).
        sym (bool): A flag indicating whether the quantization is symmetric (True) or asymmetric (False).

    Returns:
        function: The quantization function corresponding to the specified parameters.
    """
    key = dtype
    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    if sym:
        key = dtype + "_sym"
    else:
        key = dtype + "_asym"

    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    ##need to add bits
    if sym:
        key = dtype + str(bits) + "_sym"
    else:
        key = dtype + str(bits) + "_asym"

    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    if sym:
        key = dtype + "_sym"
    else:
        key = dtype + "_asym"

    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    if sym:
        key = dtype + str(bits)
    else:
        key = dtype + str(bits)

    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    raise ValueError(f"{dtype} is not supported")


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


@lru_cache(None)
def get_gaudi_fp8_ste_func():
    from auto_round.utils import is_hpu_supported

    if is_hpu_supported():
        fn = float8_e4m3fn_hpu_ste
        logger.warning_once("Using HPU STE for FP8")
    else:
        fn = float8_e4m3fn_ste
        logger.warning_once("Using CUDA/CPU STE for FP8")
    return fn
