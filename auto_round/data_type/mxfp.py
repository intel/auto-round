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

import torch

from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE, register_dtype
from auto_round.data_type.utils import (
    ceil_ste,
    floor_ste,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    round_ste,
)

MXFP_FORMAT_CACHE = {
    # data type: ebits, mbits, emax, max_norm, min_norm
    "mx_int8": (0, 8, 0, 1.984375, 0),
    "mx_int4": (0, 4, 0, 1.75, 0),
    "mx_int2": (0, 2, 0, 1.0, 0),
    "mx_fp8e5m2": (5, 4, 15, 57344.0, 6.103515625e-05),
    "mx_fp8": (4, 5, 8, 448.0, 0.015625),
    "mx_fp8e4m3": (4, 5, 8, 448.0, 0.015625),
    "mx_fp6e3m2": (3, 4, 4, 28.0, 0.25),
    "mx_fp6": (2, 5, 2, 7.5, 1.0),
    "mx_fp6e2m3": (2, 5, 2, 7.5, 1.0),
    "mx_fp4": (2, 3, 2, 6.0, 1.0),
    "mx_fp4e2m1": (2, 3, 2, 6.0, 1.0),
    "mx_float16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_fp16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_bfloat16": (8, 9, 127, 3.3895313892515355e38, 1.1754943508222875e-38),
    "mx_bf16": (8, 9, 127, 3.3895313892515355e38, 1.1754943508222875e-38),
}

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def quant_element(tensor, ebits, mbits, max_norm, mantissa_rounding="even"):
    if ebits != 0:
        private_exp = floor_ste(torch.log2(torch.abs(tensor) + (tensor == 0).type(tensor.dtype)))
        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2.0 ** float(ebits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of mbits are in the integer portion of the number
    tensor = (
        tensor * (2.0 ** float(mbits - 2))
        if private_exp is None
        else tensor / (2.0 ** private_exp.float()) * (2.0 ** float(mbits - 2))
    )
    if mantissa_rounding == "even":
        abs_tensor = torch.abs(tensor)
        mask_tensor = ((abs_tensor - 0.5) % 2 == torch.zeros_like(abs_tensor)).type(tensor.dtype)
        tensor = torch.sign(tensor) * (floor_ste(abs_tensor + 0.5) - mask_tensor)
    elif mantissa_rounding == "nearest":
        tensor = torch.sign(tensor) * round_ste(torch.abs(tensor))
    elif mantissa_rounding == "floor":
        tensor = torch.sign(tensor) * floor_ste(torch.abs(tensor))
    elif mantissa_rounding == "stochastic":
        tensor = torch.sign(tensor) * floor_ste(torch.abs(tensor) + torch.rand_like(tensor, requires_grad=False))
    else:
        raise ValueError("mantissa_rounding only supports even, nearest or floor.")

    # Undo scaling
    tensor = (
        tensor / (2.0 ** float(mbits - 2))
        if private_exp is None
        else tensor / (2.0 ** float(mbits - 2)) * (2.0 ** private_exp.float())
    )

    tensor = torch.clamp(tensor, min=-max_norm, max=max_norm)
    return tensor


def quant_mx(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    max_scale=1.0,
    mantissa_rounding="even",
    data_type="mx_fp",
    tensor_max=None,
    **kwargs
):
    """Quantize the given tensor using the specified parameters.

    This function performs quantization on the `tensor` tensor according to the
    given bit width (`bits`), data type (`data_type`), and additional parameters.
    The quantization process involves scaling the tensor values and adjusting
    the exponent and mantissa to fit within the specified format.

    Args:
        tensor (torch.Tensor): The tensor containing the tensors to be quantized.
        bits (int): The bit width to be used for quantization.
        group_size (int): The group size of sharing scale and exponent.
        data_type (str): The data type for quantization (e.g., 'mx_fp4').
        v (float): A value used for adjusting the tensors.
        max_scale (float or torch.Tensor): The maximum scale to be applied to the tensors.
        mantissa_rounding (str): rounding method for mantissa,currently support even,nearest,floor

    Returns:
        tuple: A tuple containing the quantized tensors, shared exponent, and None (reserved for future use).

    Raises:
        KeyError: If `data_type` is not found in `MXFP_FORMAT_CACHE`.
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    data_type = data_type if data_type in MXFP_FORMAT_CACHE else "mx_fp" + str(bits)
    ebits, mbits, emax, max_norm, min_norm = MXFP_FORMAT_CACHE[data_type]
    orig_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    # max_val, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)

    if tensor_max is None:
        max_val, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)
    elif isinstance(tensor_max, torch.Tensor):
        max_val = tensor_max.to(tensor.device)
        if max_val.dim() == 1:
            max_val = max_val.unsqueeze(-1)  
    else:
        max_val = torch.tensor(tensor_max, device=tensor.device)
        if max_val.dim() == 0:
            max_val = max_val.view(1, 1) 

    if isinstance(max_scale, torch.Tensor):
        ms = max_scale.to(tensor.device)
        if ms.dim() == 1:
            ms = ms.unsqueeze(-1)
        max_val = max_val * ms  
    else:
        max_val = max_val * max_scale

    # shared_exp = torch.log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype))
    shared_exp = torch.where(max_val == 0, torch.ones_like(max_val), torch.log2(max_val))
    shared_exp = floor_ste(shared_exp)
    scale_emax = 2.0 ** float(8 - 1) - 1
    shared_exp = (shared_exp - emax).clamp(min=-scale_emax, max=scale_emax)

    scale = torch.pow(2.0, shared_exp.float())
    tensor = tensor / scale + v
    tensor = torch.clamp(tensor, min=-max_norm, max=max_norm)
    tensor = quant_element(tensor, ebits, mbits, max_norm, mantissa_rounding)

    tensor = tensor * scale
    tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)
    return tensor.to(orig_dtype), shared_exp.to(orig_dtype), None


def quant_mx_rceil(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    max_scale=1.0,
    mantissa_rounding="even",
    data_type="mx_fp",
    tensor_max=None,
    **kwargs
):
    """Quantize the given tensor using the specified parameters.

    This function performs quantization on the `tensor` tensor according to the
    given bit width (`bits`), data type (`data_type`), and additional parameters.
    The quantization process involves scaling the tensor values and adjusting
    the exponent and mantissa to fit within the specified format.

    Args:
        tensor (torch.Tensor): The tensor containing the tensors to be quantized.
        bits (int): The bit width to be used for quantization.
        group_size (int): The group size of sharing scale and exponent.
        data_type (str): The data type for quantization (e.g., 'mx_fp4').
        v (float): A value used for adjusting the tensors.
        max_scale (float or torch.Tensor): The maximum scale to be applied to the tensors.
        mantissa_rounding (str): rounding method for mantissa,currently support even,nearest,floor

    Returns:
        tuple: A tuple containing the quantized tensors, shared exponent, and None (reserved for future use).

    Raises:
        KeyError: If `data_type` is not found in `MXFP_FORMAT_CACHE`.
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    data_type = data_type if data_type in MXFP_FORMAT_CACHE else "mx_fp" + str(bits)
    ebits, mbits, emax, max_norm, min_norm = MXFP_FORMAT_CACHE[data_type]
    orig_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    # max_val, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)

    if tensor_max is None:
        max_val, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)
    elif isinstance(tensor_max, torch.Tensor):
        max_val = tensor_max.to(tensor.device)
        if max_val.dim() == 1:
            max_val = max_val.unsqueeze(-1)
    else:
        max_val = torch.tensor(tensor_max, device=tensor.device)
        if max_val.dim() == 0:
            max_val = max_val.view(1, 1)

    if isinstance(max_scale, torch.Tensor):
        ms = max_scale.to(tensor.device)
        if ms.dim() == 1:
            ms = ms.unsqueeze(-1) 
        max_val = max_val * ms
    else:
        max_val = max_val * max_scale

    # shared_exp = torch.log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype))
    shared_exp = torch.where(max_val == 0, torch.ones_like(max_val), ceil_ste(torch.log2(max_val / max_norm)))
    scale_emax = 2.0 ** float(8 - 1) - 1
    shared_exp = shared_exp.clamp(min=-scale_emax, max=scale_emax)

    scale = torch.pow(2.0, shared_exp.float())
    tensor = tensor / scale + v
    tensor = torch.clamp(tensor, min=-max_norm, max=max_norm)
    tensor = quant_element(tensor, ebits, mbits, max_norm, mantissa_rounding)

    tensor = tensor * scale
    tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)
    return tensor.to(orig_dtype), shared_exp.to(orig_dtype), None


for key in MXFP_FORMAT_CACHE.keys():
    QUANT_FUNC_WITH_DTYPE[key] = quant_mx
    QUANT_FUNC_WITH_DTYPE[key + "_rceil"] = quant_mx_rceil
QUANT_FUNC_WITH_DTYPE["mx_fp_rceil"] = quant_mx_rceil

if __name__ == "__main__":
    data = torch.tensor([0.0, 0.25, 0.4, 0.75, 1.25, 1.4, 1.75, 2.5, 2.9, 3.5, 5.0, 5.1])
    data1 = quant_element(data, 2, 3, 6.0)
    gt = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0])
    assert torch.sum(torch.abs(data1 - gt)) < 1e-6

    data_neg = data * -1
    data2 = quant_element(data_neg, 2, 3, 6.0)
    assert torch.sum(torch.abs(data2 - gt * -1)) < 1e-6
