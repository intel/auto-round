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
import logging
from .utils import round_ste, reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round_diff.data_type.register import register_dtype, register_dtype_static
logger = logging.getLogger("autoround")

def lp_loss(pred, tgt, p=2.0):
    """
    loss function measured in L_p Norm
    """
    return (pred-tgt).abs().pow(p).mean(1)

def quantize(tensor: torch.Tensor, bits: int, sym: bool, tensor_min: torch.Tensor, tensor_max: torch.Tensor, q_scale_thresh: float=1e-5):
    if sym:
        # maxq = 2 ** bits - 1
        # if tensor_min is None or tensor_max is None:
        #     wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        #     wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
        # else:
        #     wmin_tmp = tensor_min
        #     wmax_tmp = tensor_max
        # if isinstance(min_scale, torch.Tensor):
        #     wmin = wmin_tmp * min_scale
        #     wmax = wmax_tmp * max_scale
        # else:
        #     wmin = wmin_tmp
        #     wmax = wmax_tmp
        # scale = ((wmax - wmin) / maxq).to(scale_dtype)
        # scale = torch.clamp(scale, min=q_scale_thresh)
        # zp = round_ste(-wmin / scale)  # pylint: disable=E1130
        # scale = scale.unsqueeze(dim=-1)
        # zp = zp.unsqueeze(dim=-1)
        pass
    else:
        maxq = 2**bits - 1
        scale = ((tensor_max - tensor_min) / maxq)
        scale = torch.clamp(scale, min=q_scale_thresh)
        zp = round_ste(-tensor_min / scale)  # pylint: disable=E1130
        scale = scale.unsqueeze(dim=-1)
        zp = zp.unsqueeze(dim=-1)
        int_w = round_ste(tensor / scale)
        q = torch.clamp(int_w + zp, 0, maxq)
        qdq_result = (scale * (q - zp)).to(tensor.dtype)
    return qdq_result

def search_quant_params(tensor: torch.Tensor, sym: bool, bits: int, scale_method: str, leaf_param: bool, always_zero: bool, q_scale_thresh: float=1e-5):
    scale, zero_point = None, None

    if len(tensor.shape)==4:
        tensor = tensor.reshape(tensor.shape[0], -1)
    
    if leaf_param:
        pass
        # self.x_min = x.data.min()
        # self.x_max = x.data.max()

    if 'max' in scale_method:
        tensor_min = torch.clamp(tensor.min(-1)[0], max=0)
        tensor_max = torch.clamp(tensor.max(-1)[0], min=0)
        if 'scale' in scale_method:
            tensor_min = tensor_min * (bits + 2) / 8
            tensor_max = tensor_max * (bits + 2) / 8

        if sym:
            # # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
            # x_absmax = max(abs(x_min), x_max)
            # delta = x_absmax / self.n_levels
            pass
        else:
            scale = (tensor_max - tensor_min) / (2**bits - 1)
        
        scale = torch.clamp(scale, min=q_scale_thresh)
        zero_point = round_ste(-tensor_min / scale) if not (sym or always_zero) else torch.full_like(scale, 2**(bits - 1))

    elif scale_method == 'mse':
        tensor_min = tensor.min(-1)[0]
        tensor_max = tensor.max(-1)[0]
        best_score = torch.full_like(tensor_max, 1e+10)
        scale, zero_point = torch.zeros_like(tensor_max), torch.zeros_like(tensor_max)
        for i in range(80):
            new_min = tensor_min * (1.0 - (i * 0.01))
            new_max = tensor_max * (1.0 - (i * 0.01))
            tensor_q = quantize(tensor, bits, sym, new_min, new_max)
            # L_p norm minimization as described in LAPQ
            # https://arxiv.org/abs/1911.07190
            score = lp_loss(tensor, tensor_q, p=2.4)
            mask = score < best_score
            best_score[mask] = score[mask]
            scale[mask] = (new_max[mask] - new_min[mask]) / (2 ** bits - 1) \
            if not always_zero else new_max[mask] / (2 ** bits - 1)
            zero_point[mask] = round_ste(-new_min[mask] / scale[mask]) if not always_zero else torch.full_like(scale[mask], 0.0)
    else:
        raise NotImplementedError

    return scale, zero_point



@register_dtype("int_sym")
def quant_tensor_sym(tensor, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                     tensor_min=None,
                     tensor_max=None, q_scale_thresh=1e-5, **kwargs):
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max

    wmin_abs = -(wmin_tmp * min_scale)  # pylint: disable=E1130
    wmax_abs = wmax_tmp * max_scale
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, 2 ** bits - 1)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp

@register_dtype_static("int_sym")
def quant_tensor_sym_static(tensor, bits=4, group_size=-1, quant_granularity='channel_wise', scale_dtype=torch.float16,
                     scale_method='max', tensor_min=None, tensor_max=None, q_scale_thresh=1e-5, **kwargs):
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """

    if quant_granularity == 'group_wise':
        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    
    maxq = 2 ** (bits - 1)
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max

    wmin_abs = -(wmin_tmp * min_scale)  # pylint: disable=E1130
    wmax_abs = wmax_tmp * max_scale
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale)
    q = torch.clamp(int_w + zp, 0, 2 ** bits - 1)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)

    if quant_granularity == 'group_wise':
        qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp

## the values should be positive
def double_quant_tensor(tensor, bits, q_scale_thresh):
    maxq = 2 ** bits - 1
    wmax = torch.clamp(tensor.max(-1)[0], min=0)
    scale = torch.clamp(wmax / maxq, q_scale_thresh)
    scale = scale.view(-1, 1)
    qdq_tensor = torch.clamp(round_ste(tensor / scale), max=maxq) * scale
    return qdq_tensor, scale


@register_dtype("int_asym_dq")
def quant_tensor_asym_dq(tensor, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                         tensor_min=None, tensor_max=None, q_scale_thresh=1e-5, super_group_size=8, super_bits=6,
                         **kwargs):
    """Quantize and de-quantize tensor asymmetrically.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)

    maxq = 2 ** bits - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
    if isinstance(min_scale, torch.Tensor):
        wmin = wmin_tmp * min_scale
        wmax = wmax_tmp * max_scale
    else:
        wmin = wmin_tmp
        wmax = wmax_tmp
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    scale = scale.view(-1, super_group_size)
    wmin_m = -wmin  # pylint: disable=E1130
    wmin_m = wmin_m.view(-1, super_group_size)

    ##conduct double quant
    scale, d_scale = double_quant_tensor(scale, super_bits, q_scale_thresh)
    wmin_m, d_wmin_m = double_quant_tensor(wmin_m, super_bits, q_scale_thresh)

    scale = scale.view(-1, 1)
    scale = torch.clamp(scale, q_scale_thresh)
    wmin_m = wmin_m.view(-1, 1)

    int_w = round_ste((tensor + wmin_m) / scale + v)
    q = torch.clamp(int_w, 0, maxq)
    qdq_result = (scale * q - wmin_m).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    # zp = round_ste(wmin_m / scale)  # remove this later
    return qdq_result, {"scale": scale, "d_scale": d_scale}, {"wmin_m": wmin_m, "d_wmin_m": d_wmin_m}


@register_dtype_static("int_asym")
def quant_tensor_asym_static(tensor, bits=4, inited=False, quant_granularity='channel_wise', group_size=-1, scale_method='max',
                      scale_=None, zp_=None, leaf_param=False, always_zero=False, running_stat=None, q_scale_thresh=1e-5, **kwargs):
    """Quantize and de-quantize tensor asymmetrically.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        inited: scale and zero_point is inited or not
        quant_granularity: granularity of quantization (channel_wise or group_wise)
        scale_method: method for searching initial scale and zero_point
        running_stat: using momentum update for activation quantization or not
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    if quant_granularity == 'group_wise':
        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    
    maxq = maxq = 2 ** bits - 1
    if not inited:
        # init scale and zero point
        scale, zp = search_quant_params(tensor, bits=bits, scale_method=scale_method, leaf_param=leaf_param, always_zero=always_zero, sym=False)
        if leaf_param:
            scale = torch.nn.Parameter(scale)
        inited = True
    else:
        scale = scale_
        zp = zp_

    if len(tensor.shape)==2:
        scale = scale.view(-1, 1)
        zp = zp.view(-1, 1)
    elif len(tensor.shape)==4:
        scale = scale.view(-1, 1, 1, 1)
        zp = zp.view(-1, 1, 1, 1)
    int_w = round_ste(tensor / scale)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)

    if quant_granularity == 'group_wise':
        qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    
    # res = {"qdq_res": qdq_result, "scale": scale, "zp": zp, "inited": inited}
    return qdq_result, scale, zp, inited

@register_dtype("int_asym")
def quant_tensor_asym(tensor, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                      tensor_min=None, tensor_max=None, q_scale_thresh=1e-5, **kwargs):
    """Quantize and de-quantize tensor asymmetrically.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** bits - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
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
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp

@register_dtype("int_sym_gptq")
def quant_tensor_sym_gptq(tensor, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                          tensor_min=None,
                          tensor_max=None, q_scale_thresh=1e-5, **kwargs):
    """Quantize and de-quantize tensor asymmetrically.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** bits - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
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

    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp


def quant_tensor_asym_wo_round(tensor, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0,
                               scale_dtype=torch.float16,
                               tensor_min=None, tensor_max=None, q_scale_thresh=1e-5, **kwargs):
    """Quantize and de-quantize tensor asymmetrically without rounding, this is mainly for tuning bias, norm.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantize tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** bits - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
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
    int_w = tensor / scale + v
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp
