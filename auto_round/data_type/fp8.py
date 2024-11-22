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

from auto_round.data_type import get_quant_func


def float8_e4m3fn_ste(x: torch.Tensor):
    """Straight-Through Estimator for float8.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    fp8 = x.to(torch.float8_e4m3fn).to(x.dtype)

    return fp8


def quant_fp8_dynamic_per_token(tensor, bits, data_type, v, min_scale, max_scale, **kwargs):
    ##this is mainly for activation, dynamic now, need to support static later
    # info = torch.finfo(torch.float8_e4m3fn)
    # max_tensor = torch.max(torch.abs(tensor))  ## better train a ratio
    #
    # scale = max_tensor.to(torch.float32) / info.max
    # min_scaling_factor = 1.0 / (info.max * 512.0)  ##copy from vllm
    # scale = torch.clip(scale, min=min_scaling_factor)
    # fp8_res = (tensor / scale)
    # fp8_res = torch.clip(fp8_res, info.min, info.max)
    # fp8_res = float8_e4m3fn_ste(fp8_res)
    # qdq_res = (fp8_res.to(tensor.dtype) * scale).to(tensor.dtype)
    # return qdq_res, scale, None
    orig_shape = tensor.shape
    tensor = tensor.reshape(-1, orig_shape[-1])
    orig_dtype= tensor.dtype
    info = torch.finfo(torch.float8_e4m3fn)
    max_tensor = torch.max(torch.abs(tensor),dim=-1)[0]  ## better train a ratio
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ##easy NAN Value
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = (tensor / scale)  ## if tensor is
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = qdq_res.to(orig_dtype).reshape(orig_shape)
    return qdq_res, scale, None


def progressive_quant_fp8_int4(tensor, bits, group_size, data_type, v, min_scale, max_scale, **kwargs):
    """
    quantize tensor to fp8 by per tensor, then quantize fp8 to w4g128
    """
    info = torch.finfo(torch.float8_e4m3fn)
    max_tensor = torch.max(torch.abs(tensor))  ## better train a ratio
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = 1.0 / (info.max * 512.0)  ##copy from vllm
    scale_bf16_to_fp8 = torch.clip(scale, min=min_scaling_factor)
    ##fp8_res = (tensor / scale_bf16_to_fp8).to(torch.float8_e4m3fn)  ##fp8 does not support many ops
    fp8_res = tensor / scale_bf16_to_fp8
    fp8_res = float8_e4m3fn_ste(fp8_res)

    ##convert to bf16
    fp8_res_using_16bit = fp8_res.to(tensor.dtype)
    ##convert to int4
    from auto_round.quantizer import quant_tensor
    quant_func, _ = get_quant_func("int", 4, True)
    qdq_int4_tensor, scale_fp8_to_int4, zp_fp8_to_int4 = quant_tensor(quant_func, fp8_res_using_16bit, bits=bits,
                                                                      group_size=group_size, v=v, min_scale=min_scale,
                                                                      max_scale=max_scale, scale_dtype=torch.bfloat16,
                                                                      **kwargs)
    qdq_tensor = qdq_int4_tensor * scale_bf16_to_fp8

    return qdq_tensor, scale_fp8_to_int4 * scale_bf16_to_fp8, None,
