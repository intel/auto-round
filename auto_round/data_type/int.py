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
from .utils import round_ste, reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.data_type.register import register_dtype
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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

## the values should be positive
def double_quant_tensor(tensor, bits, q_scale_thresh):
    maxq = 2 ** bits - 1
    wmax = torch.clamp(tensor.max(-1)[0], min=0)
    scale = torch.clamp(wmax / maxq, q_scale_thresh)
    scale = scale.view(-1, 1)
    qdq_tensor = torch.clamp(round_ste(tensor / scale), max=maxq) * scale
    return qdq_tensor, scale

@register_dtype("int_asym_dq")
def quant_tensor_asym_dq(tensor, cur_iter, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16,
                         tensor_min=None, tensor_max=None, q_scale_thresh=1e-5, super_group_size=8, super_bits=6, pre_scale=None, pre_wmin_m=None,
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
    
    if cur_iter%20==0 or pre_scale==None:
        scale,wmin_m = quant_tensor_k_quant_torch(tensor, num_bits=bits, group_size=group_size)
        scale = scale.squeeze(-1)
    else:
        scale = pre_scale
        wmin_m = pre_wmin_m
    
    scale = torch.clamp(scale, min=q_scale_thresh)
    wmin_m = torch.clamp(wmin_m, min=q_scale_thresh)
    scale = scale.view(-1, super_group_size)
    wmin_m = wmin_m.view(-1, super_group_size)
    #conduct double quant
    p_scale = scale
    p_wmin_m = wmin_m
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
    return qdq_result, {"scale": scale, "d_scale": d_scale, "pre_scale":p_scale}, {"wmin_m": wmin_m, "d_wmin_m": d_wmin_m, "pre_wmin_m":p_wmin_m}

def quant_tensor_k_quant_torch(data, num_bits=4, group_size=32):
    data = data.to(torch.float32)
    data = data.reshape((-1, group_size))  # nb = data.shape[0], (nb, group_size)
 
    maxq = 2 ** num_bits - 1
    minq = 0
    sum_x2 = torch.sum(data ** 2, dim=1, keepdim=True)  # (nb, 1)
    av_x = torch.sqrt(sum_x2 / group_size)  # (nb, 1)
    weights = av_x + torch.abs(data)  # (nb, group_size)
 
    rmin = torch.min(data, dim=1, keepdim=True)[0]  # (nb, 1)
    rmax = torch.max(data, dim=1, keepdim=True)[0]  # (nb, 1)
 
    sum_w = torch.sum(weights, dim=1, keepdim=True)  # (nb, 1)
    sum_x = torch.sum(weights * data, dim=1, keepdim=True)  # (nb, group_size)
 
    iscale = torch.ones_like(rmax, dtype=data.dtype)  # (nb, 1)
    mask = rmin != rmax
    iscale[mask] = (maxq - minq) / (rmax[mask] - rmin[mask])
    scale = 1 / iscale
    quant_data = torch.clamp(torch.round(iscale * (data - rmin)), minq, maxq)  # (nb, group_size)
    diff = scale * quant_data + rmin - data  # (nb, group_size)
 
    best_mad = torch.sum(weights * diff ** 2, dim=1, keepdim=True)  # (nb, 1)
 
    nstep = 20
    rdelta = 0.1
    rrmin = -1
 
    for is_ in range(nstep):
        iscale_new = torch.ones_like(rmax, dtype=data.dtype)  # (nb, 1)
        factor = rrmin + rdelta * is_ + maxq - minq
        iscale_new[mask] = factor / (rmax[mask] - rmin[mask])
 
        quant_data_new = torch.clamp(torch.round(iscale_new * (data - rmin)), minq, maxq)  # (nb, group_size)
        mul_weights_quant_data_new = weights * quant_data_new
 
        sum_l = torch.sum(mul_weights_quant_data_new, dim=1, keepdim=True)  # (nb, 1)
        sum_l2 = torch.sum(mul_weights_quant_data_new * quant_data_new, dim=1, keepdim=True)  # (nb, 1)
        sum_xl = torch.sum(mul_weights_quant_data_new * data, dim=1, keepdim=True)  # (nb, 1)
 
        D = sum_w * sum_l2 - sum_l ** 2  # (nb, 1)
 
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D  # (nb, 1)
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D  # (nb, 1)
 
        diff = this_scale * quant_data_new + this_min - data  # (nb, group_size)
        mad = torch.sum(weights * diff ** 2, dim=1, keepdim=True)  # (nb, 1)
 
        idx_to_replace = torch.where((mad < best_mad) & (D > 0))[0]
 
        quant_data[idx_to_replace] = quant_data_new[idx_to_replace]
        best_mad[idx_to_replace] = mad[idx_to_replace]
        scale[idx_to_replace] = this_scale[idx_to_replace]
        rmin[idx_to_replace] = this_min[idx_to_replace]
 
    scale = scale.to(torch.float32)
    rmin = rmin.to(torch.float32)
    return scale, -rmin

def quant_tensor_k_quant_cuda(data, num_bits=4, group_size=32):
    """Quantize tensor per group based on k quant.
    Ref: https://github.com/intel/neural-compressor/pull/2169/files
    Args:
        data : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 4.
    Returns:
        output: quantized weight
        scale: scale
        zero_point: zero point
    """
    try:
        import cupy as cp
        import torch

        if torch.cuda.is_available():
            data = data.to(torch.float64)
            data = cp.asarray(data)
            data = data.reshape((-1, group_size)).astype(cp.float32)  # nb = data.shape[0], (nb, group_size)
            maxq = 2**num_bits - 1
            minq = 0
            sum_x2 = cp.sum(data**2, axis=1, keepdims=True)  # (nb, 1)
            av_x = cp.sqrt(sum_x2 / group_size)  # (nb, 1)
            weights = cp.add(av_x, cp.abs(data))  # (nb, group_size)
            rmin = cp.min(data, axis=1, keepdims=True)  # (nb, 1)
            rmax = cp.max(data, axis=1, keepdims=True)  # (nb, 1)
            sum_w = cp.sum(weights, axis=1, keepdims=True)  # (nb, 1)
            sum_x = cp.sum(weights * data, axis=1, keepdims=True)  # (nb, group_size)
            iscale = cp.ones(rmax.shape, dtype=data.dtype)  # (nb, 1)
            mask = rmin != rmax
            iscale[mask] = (maxq - minq) / (rmax[mask] - rmin[mask])
            scale = 1 / iscale
            quant_data = cp.clip(cp.round(iscale * (data - rmin)), minq, maxq)  # (nb, group_size)
            diff = scale * quant_data + rmin - data  # (nb, group_size)
            best_mad = cp.sum(weights * diff**2, axis=1, keepdims=True)  # (nb, 1)
            nstep = 20
            rdelta = 0.1
            rrmin = 0 #origin:-1 
            for is_ in range(nstep):
                iscale_new = cp.ones(rmax.shape, dtype=data.dtype)  # (nb, 1)
                factor = cp.array([rrmin + rdelta * is_ + maxq - minq]).astype(data.dtype)[0]
                mask = rmin != rmax
                iscale_new[mask] = factor / (rmax[mask] - rmin[mask])
                quant_data_new = cp.clip(cp.round(iscale_new * (data - rmin)), minq, maxq)  # (nb, group_size)
                mul_weights_quant_data_new = weights * quant_data_new
                sum_l = cp.sum(mul_weights_quant_data_new, axis=1, keepdims=True)  # (nb, 1)
                sum_l2 = cp.sum(mul_weights_quant_data_new * quant_data_new, axis=1, keepdims=True)  # (nb, 1)
                sum_xl = cp.sum(mul_weights_quant_data_new * data, axis=1, keepdims=True)  # (nb, 1)
                D = cp.subtract(sum_w * sum_l2, sum_l**2)  # (nb, 1)

                this_scale = (sum_w * sum_xl - sum_x * sum_l) / D  # (nb, 1)
                this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D  # (nb, 1)

                diff = this_scale * quant_data_new + this_min - data  # (nb, group_size)
                mad = cp.sum(weights * diff**2, axis=1, keepdims=True)  # (nb, 1)

                mad_1 = cp.array(mad)
                best_mad_1 = cp.array(best_mad)
                idx_to_replace = cp.where((mad_1 < best_mad_1) & (D > 0))[0]
                # idx_to_replace = cp.where(mad_1 < best_mad_1)[0]
                quant_data[idx_to_replace, :] = quant_data_new[idx_to_replace, :]
                best_mad[idx_to_replace] = mad[idx_to_replace]
                scale[idx_to_replace] = this_scale[idx_to_replace]
                rmin[idx_to_replace] = this_min[idx_to_replace]

            scale = scale.astype(cp.float64)
            rmin = rmin.astype(cp.float64)
            return scale.get(),-rmin.get()
        else:
            logger.warning(
                "Try to use k-quant quantization on CUDA. However, CUDA is not available."
                "Fall back to k-quant quantization on CPU."
            )
            return quant_tensor_k_quant_cpu(data, num_bits, group_size)
    except ImportError:
        logger.info(
            "Now we are using k-quant quantization on cpu, which is time consuming."
            "Please consider install cupy to speed up on CUDA. See https://cupy.dev/"
            "Please also install torch to check CUDA availability."
        )
        return quant_tensor_k_quant_cpu(data, num_bits, group_size)

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
