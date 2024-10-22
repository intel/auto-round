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
from .utils import floor_ste, round_ste
from auto_round.data_type.register import register_dtype, QUANT_FUNC_WITH_DTYPE

MXFP_FORMAT_CACHE = {
    # data type: ebits, mbits, emax, max_norm, min_norm
    "mx_int8": (0, 8, 0, 1.984375, 0),
    "mx_int4": (0, 4, 0, 1.75, 0),
    "mx_int2": (0, 2, 0, 1.0, 0),
    "mx_fp8e5m2": (5, 4, 15, 57344.0, 6.103515625e-05),
    "mx_fp8": (4, 5, 8, 448.0, 0.015625),
    "mx_fp8e4m3": (4, 5, 8, 448.0, 0.015625),
    "mx_fp6e3m2": (3, 4, 4, 28.0, 0.25),
    "mx_fp6e2m3": (2, 5, 2, 7.5, 1.0),
    "mx_fp4": (2, 3, 2, 6.0, 1.0),
    "mx_fp4e2m1": (2, 3, 2, 6.0, 1.0),
    "mx_float16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_fp16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_bfloat16": (8, 9, 127, 3.3895313892515355e+38, 1.1754943508222875e-38),
    "mx_bf16": (8, 9, 127, 3.3895313892515355e+38, 1.1754943508222875e-38),
}

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def quant_mx(tensor, bits, data_type, v, max_scale, mantissa_rounding="even", **kwargs):
    """Quantize the given tensor using the specified parameters.

    This function performs quantization on the `tensor` tensor according to the
    given bit width (`bits`), data type (`data_type`), and additional parameters.
    The quantization process involves scaling the tensor values and adjusting
    the exponent and mantissa to fit within the specified format.

    Args:
        tensor (torch.Tensor): The tensor containing the tensors to be quantized.
        bits (int): The bit width to be used for quantization.
        data_type (str): The data type for quantization (e.g., 'mx_fp4').
        v (float): A value used for adjusting the tensors.
        max_scale (float or torch.Tensor): The maximum scale to be applied to the tensors.
        mantissa_rounding (str): rounding method for mantissa,currently support even,nearest,floor

    Returns:
        tuple: A tuple containing the quantized tensors, shared exponent, and None (reserved for future use).

    Raises:
        KeyError: If `data_type` is not found in `MXFP_FORMAT_CACHE`.
    """
    ebits, mbits, emax, max_norm, min_norm = MXFP_FORMAT_CACHE[data_type]
    orig_dtype = tensor.dtype
    shared_exp, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)
    if isinstance(max_scale, torch.Tensor):
        shared_exp *= (max_scale.unsqueeze(dim=-1))
    else:
        shared_exp *= max_scale
    scale_emax = 2 ** (8 - 1) - 1
    shared_exp = torch.log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype))
    shared_exp[shared_exp == torch.inf] = scale_emax + emax
    shared_exp[shared_exp == -torch.inf] = -scale_emax + emax
    shared_exp = (shared_exp - emax)
    shared_exp = floor_ste(shared_exp)
    shared_exp[shared_exp > scale_emax] = scale_emax  ##changed Nan
    shared_exp[shared_exp < -scale_emax] = -scale_emax
    if (shared_exp.dtype == torch.float16 and (torch.any(shared_exp > 15) or torch.any(shared_exp < -24))) or (
            shared_exp.dtype == torch.bfloat16 and torch.any((shared_exp < -126))):
        tensor = tensor.to(torch.float32)
        shared_exp = shared_exp.to(torch.float32)
    tensor = tensor / (2 ** shared_exp)
    is_mx_fp4 = data_type == "mx_fp4" or ("mx_fp" in data_type and bits == 4)
    multiply = 2 if is_mx_fp4 else 1  ## 2 is a tricky setting
    tensor = tensor + v * multiply
    if ebits != 0:
        private_exp = floor_ste(torch.log2(torch.abs(tensor) + (tensor == 0).type(tensor.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2 ** (ebits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of mbits are in the integer portion of the number
    tensor = tensor * (2 ** (mbits - 2)) if private_exp is None else tensor / (2 ** private_exp) * (2 ** (mbits - 2))

    if mantissa_rounding == "even":
        abs_tensor = torch.abs(tensor)
        mask_tensor = ((abs_tensor - 0.5) % 2 == torch.zeros_like(abs_tensor)).type(tensor.dtype)
        tensor = torch.sign(tensor) * (floor_ste(abs_tensor + 0.5) - mask_tensor)
    elif mantissa_rounding == "nearest":
        tensor = round_ste(tensor)
    elif mantissa_rounding == "floor":
        tensor = floor_ste(tensor)
    else:
        raise ValueError("mantissa_rounding only supports even, nearest or floor.")
    max_mantissa = 2 ** (mbits - 1) - 1
    tensor = torch.clamp(tensor, -max_mantissa, max_mantissa)

    # Undo scaling
    tensor = tensor / (2 ** (mbits - 2)) if private_exp is None else tensor / (2 ** (mbits - 2)) * (2 ** private_exp)

    tensor = torch.clamp(tensor, min=-max_norm, max=max_norm)
    tensor = tensor * (2 ** shared_exp)
    return tensor.to(orig_dtype), shared_exp.to(orig_dtype), None


for key in MXFP_FORMAT_CACHE.keys():
    QUANT_FUNC_WITH_DTYPE[key] = quant_mx
