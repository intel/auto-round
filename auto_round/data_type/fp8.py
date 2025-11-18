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
from auto_round.data_type.utils import (
    float8_e4m3fn_ste,
    float8_e5m2_ste,
    get_gaudi_fp8_ste_func,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
)

from functools import lru_cache

@lru_cache(maxsize=None)
def is_gaudi2():
    try:
        import habana_frameworks.torch.utils.experimental as htexp

        is_hpu_gaudi2 = htexp._get_device_type(
            ) == htexp.synDeviceType.synDeviceGaudi2
        return is_hpu_gaudi2
    except ImportError:
        return False

# TODO: @yi polish impl
if is_gaudi2():
    @register_dtype(("fp8_sym", "fp8"))
    def quant_fp8_sym(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, **kwargs):
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
        # !!! USE float8_e4m3fnuz for Gaudi2
        info = torch.finfo(torch.float8_e4m3fnuz)
        orig_dtype = tensor.dtype
        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
        if isinstance(max_scale, torch.Tensor):
            max_scale = max_scale.to(tensor.device)
        if isinstance(v, torch.Tensor):
            v = v.to(tensor.device)
        if tensor_max is None:  ##dynamic per-token
            max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
        elif isinstance(tensor_max, torch.Tensor):
            max_tensor = tensor_max.to(tensor.device) * max_scale
        else:
            max_tensor = torch.tensor(tensor_max).to(tensor.device) * max_scale
        scale = max_tensor.to(torch.float32) / info.max
        min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
        scale = torch.clip(scale, min=min_scaling_factor)
        if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
            tensor = tensor.to(torch.bfloat16)
        scale = scale.unsqueeze(dim=-1)
        fp8_res = tensor / scale + v
        fp8_res = torch.clip(fp8_res, info.min, info.max)
        # ste_fn = float8_e4m3fn_ste
        
        # fp8_res2 = ste_fn(fp8_res)
        from auto_round.data_type.utils import float8_e4m3fnuz_hpu_ste as ste_fn
        # float8_e4m3fn_ste_gaudi = get_gaudi_fp8_ste_func()
        # float8_e4m3fn_ste_gaudi = float8_e4m3fnuz_hpu_ste

        fp8_res2 = ste_fn(fp8_res)
        qdq_res = fp8_res2 * scale
        qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
        qdq_res = qdq_res.to(orig_dtype)
        return qdq_res, scale, None
else:
    @register_dtype(("fp8_sym", "fp8", "fp8_e4m3"))
    def quant_fp8_sym(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, **kwargs):
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
        info = torch.finfo(torch.float8_e4m3fn)
        orig_dtype = tensor.dtype
        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
        if isinstance(max_scale, torch.Tensor):
            max_scale = max_scale.to(tensor.device)
        if isinstance(v, torch.Tensor):
            v = v.to(tensor.device)
        if tensor_max is None:  ##dynamic per-token
            max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
        elif isinstance(tensor_max, torch.Tensor):
            max_tensor = tensor_max.to(tensor.device) * max_scale
        else:
            max_tensor = torch.tensor(tensor_max).to(tensor.device) * max_scale
        scale = max_tensor.to(torch.float32) / info.max
        min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
        scale = torch.clip(scale, min=min_scaling_factor)
        if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
            tensor = tensor.to(torch.bfloat16)
        scale = scale.unsqueeze(dim=-1)
        fp8_res = tensor / scale + v
        fp8_res = torch.clip(fp8_res, info.min, info.max)
        fp8_res = float8_e4m3fn_ste(fp8_res)
        qdq_res = fp8_res * scale
        qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
        qdq_res = qdq_res.to(orig_dtype)
        return qdq_res, scale, None


@register_dtype("fp8_e5m2")
def quant_fp8_e5m2(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, **kwargs):
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
    info = torch.finfo(torch.float8_e5m2)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor_max is None:  ##dynamic per-token
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
    elif isinstance(tensor_max, torch.Tensor):
        max_tensor = tensor_max.to(tensor.device) * max_scale
    else:
        max_tensor = torch.tensor(tensor_max).to(tensor.device) * max_scale
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = tensor / scale + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e5m2_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res, scale, None


@register_dtype("fp8_unit_scale")
def quant_fp8_unit_scale(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, **kwargs):
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
    info = torch.finfo(torch.float8_e4m3fn)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = torch.ones((1), device=tensor.device)
    fp8_res = tensor / scale + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res, scale, None


@register_dtype("fp8_e5m2_unit_scale")
def quant_fp8_e5m2_unit_scale(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, **kwargs):
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
    info = torch.finfo(torch.float8_e5m2)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = torch.ones((1), device=tensor.device)
    fp8_res = tensor / scale + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e5m2_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
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
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
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
    fp8_res = tensor / scale
    fp8_res = torch.clip(fp8_res, -fp8_max, fp8_max)
    float8_e4m3fn_ste_gaudi = get_gaudi_fp8_ste_func()
    fp8_res = float8_e4m3fn_ste_gaudi(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = qdq_res.to(orig_dtype).reshape(orig_shape)
    return qdq_res, scale, None
