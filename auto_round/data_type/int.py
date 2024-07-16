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


@register_dtype("int_asym")
def quant_tensor_asym(weight, num_bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                      weight_min=None, weight_max=None, q_scale_thresh=0.0,**kwargs):
    """Quantizes and dequantizes weight asymmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    if isinstance(min_scale, torch.Tensor):
        if weight_min is None or weight_max is None:
            wmin_tmp = torch.clamp(weight.min(1)[0], max=0)
            wmax_tmp = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin_tmp = weight_min
            wmax_tmp = weight_max
        wmin_tmp = wmin_tmp * min_scale
        wmax_tmp = wmax_tmp * max_scale
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        if weight_min is None or weight_max is None:
            wmin = torch.clamp(weight.min(1)[0], max=0)
            wmax = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin = weight_min
            wmax = weight_max

    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    zp = round_ste(-wmin / scale)  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp

@register_dtype("int_sym")
def quant_tensor_sym(weight, num_bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16, weight_min=None,
                     weight_max=None, q_scale_thresh=0.0,**kargs):
    """Quantizes and dequantizes weight symmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight
        weight_min (Tensor, optional): Minimum weight value for quantization. Defaults to None.
        weight_max (Tensor, optional): Maximum weight value for quantization. Defaults to None.

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    if isinstance(min_scale, torch.Tensor):
        if weight_min is None or weight_max is None:
            wmin_tmp = torch.clamp(weight.min(1)[0], max=0)
            wmax_tmp = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin_tmp = weight_min
            wmax_tmp = weight_max
        wmin_tmp = wmin_tmp * min_scale
        wmax_tmp = wmax_tmp * max_scale
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        if weight_min is None or weight_max is None:
            wmin = torch.clamp(weight.min(1)[0], max=0)
            wmax = torch.clamp(weight.max(1)[0], min=0)
        else:
            wmin = weight_min
            wmax = weight_max

    wmax_new = torch.max(wmin.abs(), wmax)
    tmp = wmin < 0
    wmin_new = wmin.clone()  ##must clone, otherwise inplace backward will occur
    if torch.any(tmp):
        wmin_new[tmp] = -wmax_new[tmp]

    tmp = (wmin_new == 0) & (wmax_new == 0)
    wmin_new[tmp] = -1
    wmax_new[tmp] = +1
    scale = ((wmax_new - wmin_new) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    scale = scale.unsqueeze(dim=-1)
    zp = torch.full_like(scale, (maxq + 1) / 2)

    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp
