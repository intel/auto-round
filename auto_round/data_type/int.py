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
from typing import Union

import torch

from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger
from auto_round.utils import get_reciprocal


def search_scales(data: torch.Tensor, bits: int, qw: Union[None, torch.Tensor, float] = None) -> torch.Tensor:
    nmax = pow(2, bits - 1)
    imax = torch.abs(data).argmax(dim=-1, keepdim=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)
    iscales = -nmax * get_reciprocal(group_max)
    scales = get_reciprocal(iscales)
    L = torch.round(1.0 * iscales * data).clip(-nmax, nmax - 1)
    if qw is None:
        qw = 1.0
    best_loss = torch.sum(((scales * L - data).to(torch.float32)) ** 2 * qw, dim=-1)
    for _is in range(-18 * 5, 18 * 5 + 1):
        if _is == 0:
            continue
        iscales = -(nmax - 0.01 * _is) * get_reciprocal(group_max)
        tmp_L = torch.round(iscales * data).clip(-nmax, nmax - 1)
        tmp_scales = get_reciprocal(iscales)
        loss = torch.sum(((tmp_scales * tmp_L - data).to(torch.float32)) ** 2 * qw, dim=-1)
        replace_id = loss < best_loss
        scales[replace_id] = tmp_scales[replace_id]
        best_loss[replace_id] = loss[replace_id]
    return scales


@register_dtype("rtn_int_sym")
def quant_tensor_rtn_sym(tensor, bits=4, group_size=-1, v=0, q_scale_thresh=1e-5, imatrix=None, **kwargs):
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
    from auto_round.data_type.gguf import _imatrix_handle_zero

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
    if imatrix is None:
        imatrix = 1.0
    else:
        imatrix = imatrix.reshape(1, -1)

        imatrix = imatrix.expand(tensor.numel() // imatrix.numel(), -1)
        imatrix = imatrix.reshape(tensor.shape)

        imatrix = _imatrix_handle_zero(imatrix, tensor, bits)

    scale = search_scales(tensor, bits, qw=imatrix)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w, -maxq, maxq - 1)
    qdq_result = (scale * q).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, maxq


@register_dtype("int_sym")
def quant_tensor_sym(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
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
    scale = scale.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w, -maxq, maxq - 1)
    qdq_result = (scale * q).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, maxq


@register_dtype("int_asym")
def quant_tensor_asym(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
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
    maxq = 2**bits - 1
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
def quant_tensor_sym_gptq(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
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
    maxq = 2**bits - 1
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


def quant_tensor_asym_wo_round(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
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
    maxq = 2**bits - 1
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
