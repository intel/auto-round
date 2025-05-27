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

def double_quant_tensor_sym(tensor, bits, q_scale_thresh):
    maxq = 2 ** (bits - 1)
    imax = abs(tensor).argmax(axis=-1, keepdims=True)
    wmax = torch.take_along_dim(tensor, imax, axis=-1)
    scale = wmax / -maxq
    qdq_tensor = torch.where(scale != 0, round_ste(tensor / scale), 0).clip(-maxq, maxq -1) * scale
    return qdq_tensor, scale

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
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    scale = scale.view(-1, super_group_size)

    #conduct double quant
    scale, d_scale = double_quant_tensor_sym(scale, super_bits, q_scale_thresh)

    scale = scale.view(-1, 1)
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, 2 ** bits - 1)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, {"scale": scale, "d_scale": d_scale}, zp

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


@register_dtype("gguf_int_asym_dq")
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
        **kwargs):
    """Quantize and de-quantize tensor asymmetrically. For Q2_K, Q4_K, Q5_K.

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
    from auto_round.export.export_to_gguf.utils import QK_K, K_SCALE_SIZE, GGML_QUANT_SIZES
    from auto_round.export.export_to_gguf.quant_gpu import make_qkx2_quants


    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)

    search_kwargs = {
        2: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": True},
        4: {"rmin": -1, "rdelta": 0.1, "nstep": 20, "use_mad": False},
        5: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": False}
    }
    if bits not in search_kwargs:
        raise KeyError(f"bits={bits} is not supported by gguf_int_asym_dq, please check.")

    maxq = 2 ** bits - 1
    ggml_type = f"q{bits}_k"
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]
    orig_dtype = tensor.dtype

    tensor = tensor.to(torch.float32)
    n_blocks = tensor.nelement() // block_size
    # q2_k (nb, 16, 16) q4_k/q5_k: (nb, 8, 32)
    tensor = tensor.reshape(n_blocks, super_group_size, QK_K // super_group_size)

    scale, int_w, wmin_m = make_qkx2_quants(tensor, bits=bits, **search_kwargs[bits])

    ##conduct double quant
    scale, d_scale = double_quant_tensor(scale, super_bits, q_scale_thresh)
    wmin_m, d_wmin_m = double_quant_tensor(wmin_m, super_bits, q_scale_thresh)

    replace_ids = scale != 0
    scale = scale.unsqueeze(-1)
    wmin_m = wmin_m.unsqueeze(-1)
    int_w[replace_ids] = torch.round(
        (tensor[replace_ids] + wmin_m[replace_ids]) / scale[replace_ids]).clip(0, maxq).to(
            torch.uint8)
    qdq_result = (scale*int_w - wmin_m).to(orig_dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    # zp = round_ste(wmin_m / scale)  # remove this later
    scale = scale.view(-1, 1)
    wmin_m = wmin_m.view(-1, 1)
    return qdq_result, {"scale": scale, "d_scale": d_scale}, {"wmin_m": wmin_m, "d_wmin_m": d_wmin_m}


@register_dtype("gguf_int_sym_dq")
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
    from auto_round.export.export_to_gguf.utils import QK_K, K_SCALE_SIZE, GGML_QUANT_SIZES
    from auto_round.export.export_to_gguf.quant_gpu import make_q3_quants, make_qx_quant

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

    if bits == 3:
        scale, int_w = make_q3_quants(tensor, bits=3, do_rmse=True)
    else:
        scale, int_w = make_qx_quant(tensor, bits=6, rmse_type=1, qw=None)
    
    #conduct double quant
    scale, d_scale = double_quant_tensor_sym(scale, super_bits, q_scale_thresh)

    replace_ids = scale != 0
    scale = scale.unsqueeze(-1)
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    int_w[replace_ids] = (torch.round(
        tensor[replace_ids] / scale[replace_ids]).clip(-maxq, maxq - 1) + maxq).to(torch.uint8)
    qdq_result = (scale * (int_w - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, {"scale": scale, "d_scale": d_scale}, zp

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
