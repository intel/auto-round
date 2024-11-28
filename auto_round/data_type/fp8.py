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

from auto_round.data_type.register import register_dtype


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

    if tensor_max is None:  ##dynamic per-te
        tensor = tensor.reshape(-1, orig_shape[-1])
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[
                         0] * max_scale
    else:
        max_tensor = torch.tensor(tensor_max).to(tensor.device) * max_scale
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16: ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = (tensor / scale)
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = qdq_res.to(orig_dtype).reshape(orig_shape)
    return qdq_res, scale, None


@register_dtype("fp8_to_int_sym")
def progressive_quant_fp8_int4(tensor, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, q_scale_thresh=1e-5,
                               weight_fp8_max_scale=1.0, **kwargs):
    """Two-stage quantization: quantize tensor to fp8 by per tensor, then quantize fp8 to w4g128

    This method first quantizes the input tensor into float8 format and then performs
    a secondary quantization to int4 with grouping.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        bits (int, optional): Bit precision for secondary quantization. Defaults to 4.
        group_size (int, optional): Group size for int4 quantization. Defaults to -1 (no grouping).
        v (float, optional): Optional parameter for variance tuning. Defaults to 0.
        min_scale (float, optional): Minimum scaling factor for int4 quantization. Defaults to 1.0.
        max_scale (float, optional): Maximum scaling factor for int4 quantization. Defaults to 1.0.
        q_scale_thresh (float, optional): Threshold for scaling. Defaults to 1e-5.
        weight_fp8_max_scale (float, optional): Maximum scaling factor for float8 quantization. Defaults to 1.0.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Combined scaling factor (torch.Tensor).
            - Placeholder for zp (None).
    """

    info = torch.finfo(torch.float8_e4m3fn)
    tensor_max = torch.max(torch.abs(tensor)).to(torch.float32) * weight_fp8_max_scale  ## better train a ratio
    scale = tensor_max.to(torch.float32) / info.max
    min_scaling_factor = 1.0 / (info.max * 512.0)  ##copy from vllm
    scale_bf16_to_fp8 = torch.clip(scale, min=min_scaling_factor)
    fp8_res = tensor / scale_bf16_to_fp8
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)

    ##convert to bf16
    fp8_res_using_16bit = fp8_res.to(tensor.dtype)
    ##convert to int4
    from auto_round.data_type.int import quant_tensor_sym
    qdq_int4_tensor, scale_fp8_to_int4, zp_fp8_to_int4 = quant_tensor_sym(fp8_res_using_16bit, bits=bits,
                                                                          group_size=group_size, v=v,
                                                                          min_scale=min_scale,
                                                                          max_scale=max_scale,
                                                                          scale_dtype=torch.bfloat16,
                                                                          q_scale_thresh=q_scale_thresh)
    qdq_tensor = qdq_int4_tensor * scale_bf16_to_fp8

    return qdq_tensor, scale_fp8_to_int4 * scale_bf16_to_fp8, None,
