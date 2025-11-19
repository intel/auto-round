# Copyright (c) 2025 Intel Corporation
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
from typing import Any, Callable, Union

import torch

from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES
from auto_round.export.export_to_gguf.packing import make_q3_quants, make_qx_quants, make_qx_quants_chunk
from auto_round.logger import logger
from auto_round.utils import get_reciprocal
from auto_round.utils.device import clear_memory


@register_dtype("int_sym_dq")
def quant_tensor_sym_dq(
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
    super_group_size=16,
    super_bits=6,
    **kwargs,
):
    """Quantize and de-quantize tensor symmetrically. full range, credict goes to llamacpp community

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
    scale = scale.view(-1, super_group_size)
    # conduct double quant
    scale, d_scale = double_quant_tensor_sym(scale, super_bits)

    scale = scale.view(-1, 1)
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    int_w = round_ste(tensor * get_reciprocal(scale) + v)
    q = torch.clamp(int_w + zp, 0, 2**bits - 1)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, {"scale": scale, "d_scale": d_scale}, zp


@register_dtype("int_asym_float_zp")
def quant_tensor_asym_float_zp(
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
    **kwargs,
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
    zp = -wmin / scale  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp


## the values should be positive
def double_quant_tensor(tensor, bits):
    tensor = tensor.to(torch.float32)  # Ensure tensor is in float32 for precision
    maxq = 2**bits - 1
    wmax = torch.clamp(tensor.max(-1)[0], min=0)
    scale = wmax / maxq
    scale = scale.view(-1, 1)
    # inverse_scale = torch.where(scale == 0, 0, 1 / scale)
    inverse_scale = (maxq * get_reciprocal(wmax)).clamp(min=0).view(-1, 1)
    qdq_tensor = torch.clamp(round_ste(tensor * inverse_scale), max=maxq) * scale
    return qdq_tensor, scale


def double_quant_tensor_sym(tensor, bits):
    tensor = tensor.to(torch.float32)  # Ensure tensor is in float32 for precision
    maxq = 2 ** (bits - 1)
    imax = abs(tensor).argmax(axis=-1, keepdims=True)
    wmax = torch.take_along_dim(tensor, imax, dim=-1)
    scale = wmax / -maxq
    inverse_scale = get_reciprocal(scale)
    qdq_tensor = torch.clip((round_ste(tensor * inverse_scale)), -maxq, maxq - 1) * scale
    return qdq_tensor, scale


def double_quant_tensor_sym_rtn(tensor, bits):
    """
    Inplace-optimized symmetric double quantization.
    - Uses float32 inplace where possible
    - Minimizes temporary tensor allocations
    """
    # Ensure tensor is float32 inplace (if tensor already float32, no copy)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()  # .float() creates a copy if needed

    maxq = 2 ** (bits - 1)

    # Compute absolute max along last dim
    # abs_() is inplace
    tensor_abs = tensor.abs()  # cannot inplace abs on original if we need original sign
    imax = tensor_abs.argmax(dim=-1, keepdim=True)
    wmax = torch.take_along_dim(tensor, imax, dim=-1)

    # Compute scale inplace
    scale = wmax / -maxq
    inverse_scale = get_reciprocal(scale)

    # Inplace quantization
    tensor = tensor.mul_(inverse_scale)  # tensor * inverse_scale inplace
    tensor = tensor.round_()  # round inplace
    tensor.clamp_(-maxq, maxq - 1)  # clamp inplace
    tensor.mul_(scale)  # multiply scale inplace

    return tensor, scale


def make_qp_quants(nmax, data, quant_weights):
    data = data.to(torch.float32)
    quant_weights = quant_weights.to(torch.float32)
    group_max = torch.max(data, dim=-1, keepdim=True)[0]
    scale = group_max / nmax
    iscale = get_reciprocal(scale)

    L = torch.round(iscale * data)
    diffs = data - scale * L
    best_mse = torch.sum(quant_weights * diffs * diffs, dim=-1)

    for _is in range(-4, 5):
        if _is == 0:
            continue
        scale_is = group_max / (0.1 * _is + nmax)
        iscale_is = get_reciprocal(scale_is)

        tmp_L = torch.round(iscale_is * data).clip(max=nmax)
        diffs = data - scale_is * tmp_L
        mse = torch.sum(quant_weights * diffs * diffs, dim=-1)

        replace_idx = mse < best_mse
        best_mse[replace_idx] = mse[replace_idx]
        iscale[replace_idx] = iscale_is[replace_idx]

    L = torch.round(iscale * data).clip(max=nmax)
    sumlx = torch.sum(quant_weights * data * L, dim=-1)
    suml2 = torch.sum(quant_weights * L * L, dim=-1)
    #
    # for _ in range(5):
    #     n_changed = 0
    #     for i in range(data.shape[-1]):
    #         slx = sumlx - quant_weights[:, i] * data[:, i] * L[:, i]
    #         sl2 = suml2 - quant_weights[:, i] * L[:, i] * L[:, i]
    #         replace_idx = (slx > 0) & (sl2 > 0)
    #         new_L = torch.round(data[:, i] * sl2 / slx).clip(max=nmax)
    #         replace_idx &= new_L != L[:, i]
    #         slx[replace_idx] += quant_weights[:, i][replace_idx] * data[:, i][replace_idx] * new_L[replace_idx]
    #         sl2[replace_idx] += quant_weights[:, i][replace_idx] * new_L[replace_idx] * new_L[replace_idx]
    #
    #         replace_idx &= slx * slx * suml2 > sumlx * sumlx * sl2
    #         L[:, i][replace_idx] = new_L[replace_idx]
    #         sumlx[replace_idx] = slx[replace_idx]
    #         suml2[replace_idx] = sl2[replace_idx]
    #         n_changed = replace_idx.sum()
    #     if n_changed == 0:
    #         break

    return sumlx / suml2, L


@register_dtype("int_asym_dq")
def quant_tensor_asym_dq(
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
    super_group_size=8,
    super_bits=6,
    **kwargs,
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
    scale = scale.view(-1, super_group_size)
    wmin = -wmin  # pylint: disable=E1130
    wmin = wmin.view(-1, super_group_size)

    ##conduct double quant
    scale, d_scale = double_quant_tensor(scale, super_bits)
    wmin, d_wmin = double_quant_tensor(wmin, super_bits)

    scale = scale.view(-1, 1)
    scale = torch.clamp(scale, q_scale_thresh)
    d_scale = torch.clamp(d_scale, q_scale_thresh)
    d_wmin = torch.clamp(d_wmin, q_scale_thresh)
    wmin = wmin.view(-1, 1)

    int_w = round_ste((tensor + wmin) / scale + v)
    q = torch.clamp(int_w, 0, maxq)
    qdq_result = (scale * q - wmin).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)

    return qdq_result, {"scale": scale, "d_scale": d_scale}, {"wmin": wmin, "d_wmin": d_wmin}


def _imatrix_handle_zero(imatrix: Union[torch.Tensor, float], weight: torch.Tensor, bits: int):
    if not isinstance(imatrix, torch.Tensor):
        return imatrix

    group_size = 16 if bits == 2 else 32
    imatrix = imatrix.reshape(-1, imatrix.shape[-1])
    if torch.min(imatrix) == 0:
        logger.warning_once(
            "please use more data via setting `nsamples` to improve accuracy as calibration activations contain 0"
        )

        zero_cnt = torch.sum(imatrix == 0, dim=-1)
        replace_index = zero_cnt > group_size // 2
        if torch.sum(replace_index) > 0:
            ## fallback to no imatrix
            if bits == 2:
                tmp_quant_weights = torch.abs(weight)
            elif bits == 4 or bits == 5:
                sigma2 = torch.sum(torch.pow(weight, 2), dim=-1, keepdim=True) / 32  ## Note 32 is different from QK_K
                av_x = torch.sqrt(sigma2)
                tmp_quant_weights = torch.abs(weight) + av_x
            tmp_quant_weights = tmp_quant_weights.to(imatrix.dtype)
            imatrix[replace_index, :] = tmp_quant_weights[replace_index, :]
        mean_replace_index = (zero_cnt > 0) & (zero_cnt <= group_size // 2)
        if torch.sum(mean_replace_index) > 0:
            ## use mean values to fill zero values
            tmp_quant_weights = torch.sum(imatrix, dim=-1) / (imatrix.shape[1] - zero_cnt)
            tmp_quant_weights = tmp_quant_weights.view(-1, 1).expand(-1, imatrix.shape[1])
            replace_idx = imatrix == 0
            imatrix[replace_idx] = tmp_quant_weights[replace_idx]
    return imatrix.reshape(weight.shape)


@torch.no_grad()
def search_gguf_scale_min_asym(tensor, bits=4, scale_dtype=torch.float16, imatrix=None, split_num=1):
    super_bits = 4 if bits == 2 else 6
    super_group_size = 16 if bits == 2 else 8

    if bits not in [2, 4, 5]:
        raise ValueError(f"bits={bits} not supported by rtn_int_asym_dq")
    quant_weights = None
    if imatrix is None or (imatrix is not None and torch.sum(imatrix) == 0):
        search_kwargs = {
            2: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": True},
            4: {"rmin": -1, "rdelta": 0.1, "nstep": 20, "use_mad": False},
            5: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": False},
        }
        if bits == 2:
            quant_weights = torch.abs(tensor)
        elif bits == 4 or bits == 5:
            sigma2 = torch.sum(torch.pow(tensor, 2), dim=-1, keepdim=True) / 32  # Note 32 is different from QK_K
            av_x = torch.sqrt(sigma2)
            quant_weights = torch.abs(tensor) + av_x
        params = search_kwargs[bits]
        scale, wmin = iterative_wls_quant_search(
            tensor,
            bits=bits,
            rrmin=params["rmin"],
            rdelta=params["rdelta"],
            nstep=params["nstep"],
            use_mad=params["use_mad"],
            weights=quant_weights,
            split_num=split_num,
        )
        scale = scale.to(scale_dtype)
        scale = torch.where(torch.abs(scale) < 1e-30, torch.zeros_like(scale), scale)
        scale = scale.reshape(-1, super_group_size)
        wmin = wmin.reshape(-1, super_group_size)
        scale, d_scale = double_quant_tensor(scale, super_bits)
        wmin = torch.where(torch.abs(wmin) < 1e-30, torch.zeros_like(wmin), wmin)
        wmin, d_wmin = double_quant_tensor(wmin, super_bits)
        wmin = wmin.view(-1, 1)
        scale = scale.view(-1, 1)
    else:
        imatrix = imatrix.to(tensor.device)
        search_kwargs = {
            2: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
            4: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
            5: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
        }

        weights = imatrix.reshape(1, -1)

        weights = weights.expand(tensor.numel() // weights.numel(), -1)
        quant_weights = weights.reshape(tensor.shape)

        quant_weights = _imatrix_handle_zero(quant_weights, tensor, bits)

        # sigma2 = torch.sum(torch.pow(tensor, 2), dim=-1, keepdim=True) / QK_K
        # if imatrix is None:
        #     av_x = torch.sqrt(sigma2)
        #     quant_weights = torch.abs(av_x + tensor * tensor)
        # else:
        #     imatrix = imatrix.reshape(1, -1).expand(tensor.numel() // imatrix.numel(), -1).reshape(tensor.shape)
        #     quant_weights = imatrix * torch.sqrt(sigma2 + tensor * tensor)

        params = search_kwargs[bits]

        scale, wmin_0 = iterative_wls_quant_search(
            tensor,
            bits=bits,
            rrmin=params["rmin"],
            rdelta=params["rdelta"],
            nstep=params["nstep"],
            use_mad=params["use_mad"],
            weights=quant_weights,
        )
        scale = scale.to(scale_dtype)
        scale = torch.where(torch.abs(scale) < 1e-30, torch.zeros_like(scale), scale)
        nmax = 2**super_bits - 1
        scale = scale.reshape(-1, super_group_size)
        wmin = wmin_0.reshape(-1, super_group_size)
        sum_quant_weights = quant_weights.sum(-1, keepdim=True).reshape(-1, super_group_size)

        d_scale, q_scale = make_qp_quants(nmax, scale, sum_quant_weights)
        d_wmin, q_wmin = make_qp_quants(nmax, wmin, sum_quant_weights)

        d_scale = d_scale.unsqueeze(-1)
        d_wmin = d_wmin.unsqueeze(-1)
        scale = (d_scale * q_scale).view(-1, 1)
        wmin = (d_wmin * q_wmin).view(-1, 1)
    if split_num > 1:
        clear_memory(device_list=[tensor.device])
    return scale, wmin, d_scale, d_wmin


@register_dtype("rtn_int_asym_dq")
def quant_tensor_gguf_asym_dq(
    tensor: torch.Tensor,
    bits: int = 4,
    scale_dtype=torch.float16,
    imatrix=None,
    scale=None,
    wmin=None,
    d_scale=None,
    d_wmin=None,
    split_num=None,
    **kwargs,
):
    """Quantizes and dequantizes a tensor using asymmetric integer quantization for formats like Q2_K, Q4_K, and Q5_K.
    Only fit for iters 0

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        bits (int): Number of bits for quantization.
        v (float): Perturbation added before rounding.
        scale_dtype (torch.dtype): Data type for quantized scale.
        imatrix (torch.Tensor, optional): Importance matrix for weighted quantization.

    Returns:
        Tuple: (Quantized-dequantized tensor, scale dictionary, zero-point dictionary)
    """
    orig_dtype = tensor.dtype
    maxq = 2**bits - 1
    group_size = 16 if bits == 2 else 32
    if split_num is None:
        split_num = 1
        for dim in tensor.shape:
            if dim > 100_000:
                split_num = 16
                break

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)

    tensor = tensor.to(torch.float32)
    if scale is None:
        scale, wmin, d_scale, d_wmin = search_gguf_scale_min_asym(
            tensor, bits, scale_dtype, imatrix, split_num=split_num
        )

    inverse_scale = get_reciprocal(scale)
    tensor = tensor.add_(wmin)
    tensor = (tensor.mul_(inverse_scale)).round_().clamp_(0, maxq)
    tensor = tensor.mul_(scale)
    tensor = tensor.sub_(wmin).to(orig_dtype)
    tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)
    return tensor, {"scale": scale, "d_scale": d_scale}, {"wmin": wmin, "d_wmin": d_wmin}


# TODO consolidate iterative_wls_quant_search_chunk and non-chunk
def iterative_wls_quant_search_chunk(
    data, bits=4, rrmin=-1.0, rdelta=0.1, nstep=20, use_mad=False, weights=None, split_num=8
):
    dtype = torch.float32
    data = data.to(dtype)
    maxq = 2**bits - 1
    minq = 0
    weights = 1.0 if weights is None else weights.to(dtype)

    results_scale = []
    results_rmin = []

    chunk_size = (data.shape[0] + split_num - 1) // split_num

    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        chunk = data[start:end]
        chunk_weights = weights if isinstance(weights, float) else weights[start:end]

        # Pre-allocate reusable buffers to avoid new allocations
        tmp = torch.empty_like(chunk)
        quant_data = torch.empty_like(chunk)
        diff = torch.empty_like(chunk)

        rmin = torch.min(chunk, dim=1, keepdim=True)[0]
        rmax = torch.max(chunk, dim=1, keepdim=True)[0]
        sum_w = torch.sum(chunk_weights, dim=1, keepdim=True)
        sum_x = torch.sum(chunk_weights * chunk, dim=1, keepdim=True)

        scale = (rmax - rmin) / (maxq - minq)
        iscale = get_reciprocal(scale)

        # tmp = (chunk - rmin) * iscale
        tmp.copy_(chunk).sub_(rmin).mul_(iscale)

        # quant_data = round(tmp).clamp_()
        torch.round(tmp, out=quant_data)
        quant_data.clamp_(minq, maxq)

        # diff = scale * quant_data + rmin - chunk
        diff.copy_(quant_data).mul_(scale).add_(rmin).sub_(chunk)

        if use_mad:
            best_mad = (chunk_weights * diff.abs_()).sum(dim=1, keepdim=True)
        else:
            diff.pow_(2)
            best_mad = (chunk_weights * diff).sum(dim=1, keepdim=True)

        for is_ in range(nstep):
            factor = rrmin + rdelta * is_ + maxq - minq

            scale_new = (rmax - rmin) / factor
            iscale_new = get_reciprocal(scale_new)

            # tmp = (chunk - rmin) * iscale_new
            tmp.copy_(chunk).sub_(rmin).mul_(iscale_new)

            torch.round(tmp, out=quant_data)
            quant_data.clamp_(minq, maxq)

            # tmp = chunk_weights * quant_data
            tmp.copy_(quant_data).mul_(chunk_weights)

            sum_l = tmp.sum(dim=-1, keepdim=True)
            sum_l2 = (tmp * quant_data).sum(dim=-1, keepdim=True)
            sum_xl = (tmp * chunk).sum(dim=-1, keepdim=True)

            D = sum_w * sum_l2 - sum_l * sum_l

            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D

            mask = this_min > 0
            if mask.any():
                this_min[mask] = 0
                this_scale[mask] = (sum_xl / sum_l2)[mask]

            reverse_this_scale = get_reciprocal(this_scale)

            # tmp = (chunk - this_min) * reverse_this_scale
            tmp.copy_(chunk).sub_(this_min).mul_(reverse_this_scale)

            torch.round(tmp, out=quant_data)
            quant_data.clamp_(minq, maxq)

            # diff = this_scale * quant_data + this_min - chunk
            diff.copy_(quant_data).mul_(this_scale).add_(this_min).sub_(chunk)

            if use_mad:
                mad = (chunk_weights * diff.abs_()).sum(dim=-1, keepdim=True)
            else:
                diff.pow_(2)
                mad = (chunk_weights * diff).sum(dim=-1, keepdim=True)

            idx_to_replace = torch.where((mad < best_mad) & (D > 0))[0]

            best_mad[idx_to_replace] = mad[idx_to_replace]
            scale[idx_to_replace] = this_scale[idx_to_replace]
            rmin[idx_to_replace] = this_min[idx_to_replace]

        results_scale.append(scale.to(torch.float32))
        results_rmin.append(-rmin.to(torch.float32))

        # YOUR ORIGINAL LOGIC — kept unchanged
        if split_num > 1:
            clear_memory(device_list=[data.device])

    return torch.cat(results_scale, dim=0), torch.cat(results_rmin, dim=0)


def iterative_wls_quant_search(
    data, bits=4, rrmin=-1.0, rdelta=0.1, nstep=20, use_mad=False, weights=None, split_num=1
):
    """Adapted from Llamacpp. Performs iterative weighted least squares quantization search.

    Args:
        data (torch.Tensor): Input tensor to quantize.
        bits (int): Number of quantization bits.
        rrmin (float): Initial range scaling factor.
        rdelta (float): Step size for range scaling.
        nstep (int): Number of search steps.
        use_mad (bool): Whether to use mean absolute deviation instead of squared error.
        weights (torch.Tensor): Weight matrix for each element.

    Returns:
        Tuple: (Optimal scale tensor, optimal minimum value tensor)
    """

    # TODO this one should change to try catch later

    return iterative_wls_quant_search_chunk(
        data=data,
        bits=bits,
        rrmin=rrmin,
        rdelta=rdelta,
        nstep=nstep,
        use_mad=use_mad,
        weights=weights,
        split_num=split_num,
    )


@torch.no_grad()
def search_gguf_scale_min_sym(tensor, bits, imatrix, scale_dtype, split_num):
    if imatrix is None or (imatrix is not None and torch.sum(imatrix) == 0):
        if bits == 3:
            scale, int_w = make_q3_quants(tensor, bits=bits, do_rmse=True)  # TODO split num
            ##scale, int_w = make_qx_quants(tensor, bits=bits, rmse_type=1, qw=None)
        elif bits == 6:
            scale, int_w = make_qx_quants_chunk(tensor, bits=bits, rmse_type=1, qw=None, split_num=split_num)
    else:
        imatrix = imatrix.to(tensor.device)
        weights = imatrix.reshape(1, -1)
        weights = weights.expand(tensor.numel() // weights.numel(), -1)
        quant_weights = weights.reshape(tensor.shape)

        quant_weights = _imatrix_handle_zero(quant_weights, tensor, bits)

        scale, int_w = make_qx_quants_chunk(tensor, bits=bits, rmse_type=1, qw=quant_weights, split_num=split_num)
    if split_num > 1:
        clear_memory(device_list=[tensor.device])
    return scale


@register_dtype("rtn_int_sym_dq")
def quant_tensor_gguf_sym_dq(
    tensor,
    bits=3,
    imatrix=None,
    scale=None,
    d_scale=None,
    scale_dtype=torch.float16,
    split_num=None,
    **kwargs,
):
    """Quantize and de-quantize tensor asymmetrically. For Q3_K, Q6_K.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
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

    from auto_round.export.export_to_gguf.config import K_SCALE_SIZE, QK_K

    if bits not in [3, 6]:
        raise KeyError(f"bits={bits} is not supported by gguf_int_sym_dq, please check.")

    maxq = 2 ** (bits - 1)
    group_size = 16
    if split_num is None:
        split_num = 1
        for dim in tensor.shape:
            if dim > 100_000:
                split_num = 16
                break

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    orig_dtype = tensor.dtype
    super_bits = 6 if bits == 3 else 8
    super_group_size = 16
    ggml_type = f"q{bits}_k"
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]
    tensor = tensor.to(torch.float32)
    n_blocks = tensor.nelement() // block_size
    # (nb, 16, 16)
    tensor = tensor.reshape(n_blocks, super_group_size, QK_K // super_group_size)
    if scale is None and d_scale is None:
        scale = search_gguf_scale_min_sym(tensor, bits, imatrix, scale_dtype, split_num=split_num)

    scale = scale.to(scale_dtype)
    scale = torch.where(torch.abs(scale) < 1e-30, torch.zeros_like(scale), scale)
    # conduct double quant
    scale, d_scale = double_quant_tensor_sym_rtn(scale, super_bits)

    scale = scale.unsqueeze(-1)
    # zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    inverse_scale = get_reciprocal(scale)
    # int_w = round_ste(tensor * inverse_scale).clip(-maxq, maxq - 1) + maxq
    # qdq_result = (scale * (int_w - zp)).to(orig_dtype)
    tensor = tensor.mul_(inverse_scale).round_().clamp_(-maxq, maxq - 1)
    tensor = tensor.mul_(scale).to(orig_dtype)
    tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)

    return tensor, {"scale": scale, "d_scale": d_scale}, maxq
