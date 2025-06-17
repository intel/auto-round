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

import torch
from auto_round.data_type.utils import round_ste, reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.data_type.register import register_dtype


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
        **kwargs):
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
    int_w = torch.where(scale != 0, round_ste(tensor / scale + v), 0)
    q = torch.clamp(int_w + zp, 0, 2 ** bits - 1)
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
    maxq = 2 ** bits - 1
    wmax = torch.clamp(tensor.max(-1)[0], min=0)
    scale = wmax / maxq
    scale = scale.view(-1, 1)
    inverse_scale = torch.where(scale == 0, 0, 1 / scale)
    qdq_tensor = torch.clamp(round_ste(tensor * inverse_scale), max=maxq) * scale
    return qdq_tensor, scale


def double_quant_tensor_sym(tensor, bits):
    tensor = tensor.to(torch.float32)  # Ensure tensor is in float32 for precision
    maxq = 2 ** (bits - 1)
    imax = abs(tensor).argmax(axis=-1, keepdims=True)
    wmax = torch.take_along_dim(tensor, imax, dim=-1)
    scale = wmax / -maxq
    inverse_scale = torch.where(scale == 0, 0, 1 / scale)
    qdq_tensor = (tensor * inverse_scale).clip(-maxq, maxq - 1) * scale
    return qdq_tensor, scale


def make_qp_quants(nmax, data, quant_weights):
    data = data.to(torch.float32)
    quant_weights = quant_weights.to(torch.float32)
    group_max = torch.max(data, dim=-1, keepdim=True)[0]
    scale = group_max / nmax
    iscale = torch.where(scale == 0, 0, 1 / scale)

    L = torch.round(iscale * data)
    diffs = data - scale * L
    best_mse = torch.sum(quant_weights * diffs * diffs)

    for _is in range(-4, 5):
        if _is == 0:
            continue
        scale_is = group_max / (0.1 * _is + nmax)
        iscale_is = torch.where(scale == 0, 0, 1 / scale_is)

        tmp_L = torch.round(iscale_is * data).clip(max=nmax)
        diffs = data - scale_is * tmp_L
        mse = torch.sum(quant_weights * diffs * diffs)

        replace_idx = mse < best_mse
        best_mse[replace_idx] = mse[replace_idx]
        iscale[replace_idx] = iscale_is[replace_idx]

    L = torch.round(iscale * data).clip(max=nmax)
    sumlx = torch.sum(quant_weights * data * L, dim=-1)
    suml2 = torch.sum(quant_weights * L * L, dim=-1)

    for _ in range(5):
        n_changed = 0
        for i in range(data.shape[-1]):
            slx = sumlx - quant_weights[:, i] * data[:, i] * L[:, i]
            sl2 = suml2 - quant_weights[:, i] * L[:, i] * L[:, i]
            replace_idx = (slx > 0) & (sl2 > 0)
            new_L = torch.round(data[:, i] * sl2 / slx).clip(max=nmax)
            replace_idx &= new_L != L[:, i]
            slx[replace_idx] += quant_weights[:, i][replace_idx] * data[:, i][replace_idx] * new_L[replace_idx]
            sl2[replace_idx] += quant_weights[:, i][replace_idx] * new_L[replace_idx] * new_L[replace_idx]

            replace_idx &= slx * slx * suml2 > sumlx * sumlx * sl2
            L[:, i][replace_idx] = new_L[replace_idx]
            sumlx[replace_idx] = slx[replace_idx]
            suml2[replace_idx] = sl2[replace_idx]
            n_changed = replace_idx.sum()
        if n_changed == 0:
            break

    return sumlx / suml2, L


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
    scale, d_scale = double_quant_tensor(scale, super_bits)
    wmin_m, d_wmin_m = double_quant_tensor(wmin_m, super_bits)

    scale = scale.view(-1, 1)
    scale = torch.clamp(scale, q_scale_thresh)
    d_scale = torch.clamp(d_scale, q_scale_thresh)
    d_wmin_m = torch.clamp(d_wmin_m, q_scale_thresh)
    wmin_m = wmin_m.view(-1, 1)

    int_w = round_ste((tensor + wmin_m) / scale + v)
    q = torch.clamp(int_w, 0, maxq)
    qdq_result = (scale * q - wmin_m).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)

    # zp = round_ste(wmin_m / scale)  # remove this later
    return qdq_result, {"scale": scale, "d_scale": d_scale}, {"wmin_m": wmin_m, "d_wmin_m": d_wmin_m}


@register_dtype("rtn_int_asym_dq")
def quant_tensor_gguf_asym_dq(
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
        imatrix=None,
        **kwargs):
    """Quantizes and dequantizes a tensor using asymmetric integer quantization for formats like Q2_K, Q4_K, and Q5_K.
    Only fit for iters 0

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        bits (int): Number of bits for quantization.
        group_size (int): Group size for per-group quantization.
        v (float): Perturbation added before rounding.
        min_scale (float): Minimum allowed scale value.
        max_scale (float): Maximum allowed scale value.
        scale_dtype (torch.dtype): Data type for quantized scale.
        tensor_min (torch.Tensor, optional): Minimum values for the tensor groups.
        tensor_max (torch.Tensor, optional): Maximum values for the tensor groups.
        q_scale_thresh (float): Threshold to clamp the quantized scale.
        super_group_size (int): Number of groups to bundle for secondary quantization.
        super_bits (int): Number of bits used in secondary quantization.
        imatrix (torch.Tensor, optional): Importance matrix for weighted quantization.

    Returns:
        Tuple: (Quantized-dequantized tensor, scale dictionary, zero-point dictionary)
    """
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if bits not in [2, 4, 5]:
        raise ValueError(f"bits={bits} not supported by gguf_int_asym_dq")
    maxq = 2 ** bits - 1
    QK_K = 256
    quant_weights = None
    if imatrix is None or (imatrix is not None and torch.sum(imatrix)==0):
        search_kwargs = {
            2: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": True},
            4: {"rmin": -1, "rdelta": 0.1, "nstep": 20, "use_mad": False},
            5: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": False}
        }
        if bits == 2:
            quant_weights = torch.abs(tensor)
        elif bits == 4 or bits == 5:
            sigma2 = torch.sum(tensor ** 2, dim=-1, keepdim=True) / 32  ##Note 32 is different from QK_K
            av_x = torch.sqrt(sigma2)
            quant_weights = torch.sqrt(av_x + tensor * tensor)
        params = search_kwargs[bits]
        scale, wmin_m = iterative_wls_quant_search(
            tensor, bits=bits, rrmin=params["rmin"], rdelta=params["rdelta"], nstep=params["nstep"],
            use_mad=params["use_mad"], weights=quant_weights
        )
        scale= scale.reshape(-1, super_group_size)
        wmin = wmin_m.reshape(-1, super_group_size)
        scale, d_scale = double_quant_tensor(scale, super_bits)
        wmin_m, d_wmin_m = double_quant_tensor(wmin, super_bits)
        wmin_m = wmin_m.view(-1, 1)
        scale = scale.view(-1, 1)
    else:
        imatrix = imatrix.to(tensor.device)
        search_kwargs = {
            2: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
            4: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
            5: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
        }
        if bits == 2:
            sigma2 = torch.sum(tensor ** 2, dim=-1, keepdim=True) / QK_K
            imatrix = imatrix.reshape(1, -1).expand(tensor.numel() // imatrix.numel(), -1).reshape(tensor.shape)
            quant_weights = imatrix * torch.sqrt(sigma2 + tensor * tensor)
        elif bits == 4 or bits == 5:
            sigma2 = 2 * torch.sum(tensor ** 2, dim=-1, keepdim=True) / QK_K
            av_x = torch.sqrt(sigma2)
            imatrix = imatrix.reshape(1, -1).expand(tensor.numel() // imatrix.numel(), -1).reshape(tensor.shape)
            quant_weights = imatrix * torch.sqrt(av_x + tensor * tensor)
        weights = imatrix.reshape(1, -1)
        weights = weights.expand(tensor.numel() // weights.numel(), -1)
        quant_weights = weights.reshape(tensor.shape)

        params = search_kwargs[bits]

        scale, wmin_m_0 = iterative_wls_quant_search(
            tensor, bits=bits, rrmin=params["rmin"], rdelta=params["rdelta"], nstep=params["nstep"],
            use_mad=params["use_mad"], weights=quant_weights
        )

        nmax = 2 ** super_bits - 1
        scale = scale.reshape(-1, super_group_size)
        wmin_m = wmin_m_0.reshape(-1, super_group_size)
        sum_quant_weights = quant_weights.sum(-1, keepdim=True).reshape(-1, super_group_size)

        d_scale, q_scale = make_qp_quants(nmax, scale, sum_quant_weights)
        d_wmin_m, q_wmin_m = make_qp_quants(nmax, wmin_m, sum_quant_weights)
        d_scale = d_scale.unsqueeze(-1)
        d_wmin_m = d_wmin_m.unsqueeze(-1)
        scale = (d_scale * q_scale).view(-1, 1)
        wmin_m = (d_wmin_m * q_wmin_m).view(-1, 1)
    inverse_scale = torch.where(scale == 0, 0, 1 / scale)

    int_w = torch.clamp(round_ste((tensor + wmin_m) * inverse_scale + v), 0, maxq)
    qdq_result = (scale * int_w - wmin_m).to(orig_dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, {"scale": scale, "d_scale": d_scale}, {"wmin_m": wmin_m, "d_wmin_m": d_wmin_m}


def iterative_wls_quant_search(data, bits=4, rrmin=-1.0, rdelta=0.1, nstep=20, use_mad=False, weights=None):
    """Performs iterative weighted least squares quantization search.

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
    dtype = torch.bfloat16
    data = data.to(dtype)
    maxq = 2 ** bits - 1
    minq = 0
    weights = 1.0 if weights is None else weights.to(dtype)

    rmin = torch.min(data, dim=1, keepdim=True)[0]
    rmax = torch.max(data, dim=1, keepdim=True)[0]

    sum_w = torch.sum(weights, dim=1, keepdim=True)
    sum_x = torch.sum(weights * data, dim=1, keepdim=True)

    scale = 1 / ((maxq - minq) / (rmax - rmin + 1e-8))
    quant_data = torch.clamp(torch.round((maxq - minq) / (rmax - rmin + 1e-8) * (data - rmin)), minq, maxq)
    diff = scale * quant_data + rmin - data
    best_mad = torch.sum(weights * (torch.abs(diff) if use_mad else diff ** 2), dim=1, keepdim=True)

    for is_ in range(nstep):
        factor = rrmin + rdelta * is_ + maxq - minq
        iscale_new = factor / (rmax - rmin + 1e-8)
        quant_data_new = torch.clamp(torch.round(iscale_new * (data - rmin)), minq, maxq)

        mul_weights_quant_data = weights * quant_data_new
        sum_l = torch.sum(mul_weights_quant_data, dim=1, keepdim=True)
        sum_l2 = torch.sum(mul_weights_quant_data * quant_data_new, dim=1, keepdim=True)
        sum_xl = torch.sum(mul_weights_quant_data * data, dim=1, keepdim=True)

        D = sum_w * sum_l2 - sum_l ** 2
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D

        quant_data = torch.clamp(torch.round((1 / this_scale) * (data - this_min)), minq, maxq)
        diff = this_scale * quant_data + this_min - data
        mad = torch.sum(weights * (torch.abs(diff) if use_mad else diff ** 2), dim=1, keepdim=True)

        idx_to_replace = torch.where((mad < best_mad) & (D > 0))[0]
        best_mad[idx_to_replace] = mad[idx_to_replace]
        scale[idx_to_replace] = this_scale[idx_to_replace]
        rmin[idx_to_replace] = this_min[idx_to_replace]

    return scale.to(torch.float32), -rmin.to(torch.float32)


@register_dtype("rtn_int_sym_dq")
def quant_tensor_gguf_sym_dq(
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
        imatrix=None,
        **kwargs):
    """Quantize and de-quantize tensor asymmetrically. For Q3_K, Q6_K.

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
    from auto_round.export.export_to_gguf.config import QK_K, K_SCALE_SIZE, GGML_QUANT_SIZES
    from auto_round.export.export_to_gguf.packing import make_q3_quants, make_qx_quants

    if bits not in [3, 6]:
        raise KeyError(f"bits={bits} is not supported by gguf_int_sym_dq, please check.")

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
    ggml_type = f"q{bits}_k"
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]
    orig_dtype = tensor.dtype

    tensor = tensor.to(torch.float32)
    n_blocks = tensor.nelement() // block_size
    # (nb, 16, 16)
    tensor = tensor.reshape(n_blocks, super_group_size, QK_K // super_group_size)
    if imatrix is None or (imatrix is not None and torch.sum(imatrix)==0):
        if bits == 3:
            scale, int_w = make_q3_quants(tensor, bits=bits, do_rmse=True)
        elif bits == 6:
            scale, int_w = make_qx_quants(tensor, bits=bits, rmse_type=1, qw=None)
    else:
        imatrix = imatrix.to(tensor.device)
        if bits == 3:
            sigma2 = 2 * torch.sum(tensor ** 2, dim=-1, keepdim=True) / QK_K
            imatrix = imatrix.reshape(1, -1).expand(tensor.numel() // imatrix.numel(), -1).reshape(tensor.shape)
            quant_weights = imatrix * torch.sqrt(sigma2 + tensor * tensor)
            scale, int_w = make_qx_quants(tensor, bits=bits, rmse_type=1, qw=quant_weights)
        elif bits == 6:
            weights = imatrix.reshape(1, -1)
            weights = weights.expand(tensor.numel() // weights.numel(), -1)
            quant_weights = weights.reshape(tensor.shape)
            scale, int_w = make_qx_quants(tensor, bits=bits, rmse_type=1, qw=quant_weights)

    # conduct double quant
    scale, d_scale = double_quant_tensor_sym(scale, super_bits)

    replace_ids = scale != 0
    scale = scale.unsqueeze(-1)
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    int_w[replace_ids] = (torch.round(
        tensor[replace_ids] / scale[replace_ids]).clip(-maxq, maxq - 1) + maxq).to(torch.uint8)
    qdq_result = (scale * (int_w - zp)).to(orig_dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, {"scale": scale, "d_scale": d_scale}, zp
