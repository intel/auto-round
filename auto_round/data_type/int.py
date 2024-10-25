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
from .utils import round_ste
from auto_round.data_type.register import register_dtype


@register_dtype("int_sym")
def quant_tensor_sym(weight, bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16, weight_min=None,
                     weight_max=None, q_scale_thresh=0.0, **kwargs):
    """Quantize and de-quantize weight asymmetrically. full range, credict goes to llamacpp community

    Args:
        weight: Tensor containing the weight to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** (bits - 1))
    if weight_min is None or weight_max is None:
        wmin_tmp = torch.clamp(weight.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(weight.max(-1)[0], min=0)
    else:
        wmin_tmp = weight_min
        wmax_tmp = weight_max

    wmin_abs = -(wmin_tmp * min_scale)   # pylint: disable=E1130
    wmax_abs = wmax_tmp * max_scale

    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)

    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, 2 ** bits - 1)
    qdq_result = (scale * (q - zp)).to(weight.dtype)
    return qdq_result, scale, zp


@register_dtype("int_asym")
def quant_tensor_asym(weight, bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                      weight_min=None, weight_max=None, q_scale_thresh=0.0, **kwargs):
    """Quantize and de-quantize weight asymmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** bits - 1)
    if weight_min is None or weight_max is None:
        wmin_tmp = torch.clamp(weight.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(weight.max(-1)[0], min=0)
    else:
        wmin_tmp = weight_min
        wmax_tmp = weight_max
    if isinstance(min_scale, torch.Tensor):
        wmin = wmin_tmp * min_scale
        wmax = wmax_tmp * max_scale
    else:
        wmin = wmin_tmp
        wmax = wmax_tmp
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    zp = round_ste(-wmin / scale)  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(weight.dtype)
    return qdq_result, scale, zp


@register_dtype("int_sym_gptq")
def quant_tensor_sym_gptq(weight, bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16, weight_min=None,
                          weight_max=None, q_scale_thresh=0.0, **kwargs):
    """Quantize and de-quantize weight asymmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** bits - 1)
    if weight_min is None or weight_max is None:
        wmin_tmp = torch.clamp(weight.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(weight.max(-1)[0], min=0)
    else:
        wmin_tmp = weight_min
        wmax_tmp = weight_max
    if isinstance(min_scale, torch.Tensor):
        wmin = wmin_tmp * min_scale
        wmax = wmax_tmp * max_scale
    else:
        wmin = wmin_tmp
        wmax = wmax_tmp

    wmax_new = torch.max(wmin.abs(), wmax)
    tmp = wmin < 0
    wmin_new = wmin.clone()  ##must clone, otherwise inplace backward will occur
    if torch.any(tmp):
        wmin_new[tmp] = -wmax_new[tmp]

    scale = ((wmax_new - wmin_new) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    scale = scale.unsqueeze(dim=-1)
    zp = torch.full_like(scale, (maxq + 1) / 2)

    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(weight.dtype)
    return qdq_result, scale, zp


def quant_tensor_asym_wo_round(weight, bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                               weight_min=None, weight_max=None, q_scale_thresh=0.0, **kwargs):
    """Quantize and de-quantize weight asymmetrically without rounding, this is mainly for tuning bias, norm.

    Args:
        weight: Tensor containing the weight to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantize weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** bits - 1)
    if weight_min is None or weight_max is None:
        wmin_tmp = torch.clamp(weight.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(weight.max(-1)[0], min=0)
    else:
        wmin_tmp = weight_min
        wmax_tmp = weight_max
    if isinstance(min_scale, torch.Tensor):
        wmin = wmin_tmp * min_scale
        wmax = wmax_tmp * max_scale
    else:
        wmin = wmin_tmp
        wmax = wmax_tmp

    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    zp = -wmin / scale  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = weight / scale + v
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(weight.dtype)
    return qdq_result, scale, zp
