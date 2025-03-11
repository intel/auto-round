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
from auto_round.data_type.utils import get_gaudi_fp8_ste_func, float8_e4m3fn_ste
from auto_round.data_type.register import register_dtype


@register_dtype("fp8_dynamic_per_token_sym")
def fp8_dynamic_per_token_sym(tensor, max_scale=1.0, **kwargs):
    """Dynamic per-token symmetric quantization using float8.

    This function dynamically calculates a per-token scaling factor for each group of tokens
    and applies symmetric quantization using float8 format.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    orig_shape = tensor.shape
    info = torch.finfo(torch.float8_e4m3fn)
    orig_dtype = tensor.dtype

    tensor = tensor.reshape(-1, orig_shape[-1])
    max_tensor = torch.max(torch.abs(tensor), dim=-1)[
                     0] * max_scale

    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = (tensor / scale)
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = qdq_res.to(orig_dtype).reshape(orig_shape)
    return qdq_res, scale, None


@register_dtype("fp8_sym")
def quant_fp8_sym(tensor, max_scale=1.0, tensor_max=None, **kwargs):
    """Symmetric quantization using float8 format.

    Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    orig_shape = tensor.shape
    info = torch.finfo(torch.float8_e4m3fn)
    orig_dtype = tensor.dtype

    if tensor_max is None:  ##dynamic per-token
        tensor = tensor.reshape(-1, orig_shape[-1])
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[
                         0] * max_scale
    elif isinstance(tensor_max,torch.Tensor):
        max_tensor = tensor_max.clone().detach().to(tensor.device) * max_scale
    else:
        max_tensor = torch.tensor(tensor_max).to(tensor.device) * max_scale
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = (tensor / scale)
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = qdq_res.to(orig_dtype).reshape(orig_shape)
    return qdq_res, scale, None


@register_dtype("fp8_gaudi3_sym")
def quant_fp8_sym_gaudi3(tensor, max_scale=1.0, tensor_max=None, **kwargs):
    """Symmetric quantization using float8 format.

    Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    orig_shape = tensor.shape
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    orig_dtype = tensor.dtype

    if tensor_max is None:  ##dynamic per-te
        tensor = tensor.reshape(-1, orig_shape[-1])
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[
                         0] * max_scale
    elif isinstance(tensor_max, torch.Tensor):
        max_tensor = tensor_max.clone().detach().to(tensor.device) * max_scale
    else:
        max_tensor = torch.tensor(tensor_max).to(tensor.device) * max_scale
    scale = max_tensor.to(torch.float32) / fp8_max
    min_scaling_factor = float(1.0 / (fp8_max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = (tensor / scale)
    fp8_res = torch.clip(fp8_res, -fp8_max, fp8_max)
    float8_e4m3fn_ste_gaudi = get_gaudi_fp8_ste_func()
    fp8_res = float8_e4m3fn_ste_gaudi(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = qdq_res.to(orig_dtype).reshape(orig_shape)
    return qdq_res, scale, None
