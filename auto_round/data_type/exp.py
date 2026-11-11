# Copyright (c) 2026 Intel Corporation
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

from auto_round.data_type.gguf import _imatrix_handle_zero
from auto_round.data_type.int import search_scales
from auto_round.data_type.nvfp import FLOAT8_UE5M3_MAX, cast_to_ue5m3_ste
from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import (
    float8_e4m3fn_ste,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    round_ste,
)


def _cast_scale(scale: torch.Tensor, *, signed: bool, q_scale_thresh: float) -> torch.Tensor:
    if signed:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = float8_e4m3fn_ste(scale.clamp(min=-fp8_max, max=fp8_max))
        return torch.where(scale < 0, scale.clamp(max=-q_scale_thresh), scale.clamp(min=q_scale_thresh))
    scale = cast_to_ue5m3_ste(scale.clamp(min=0, max=FLOAT8_UE5M3_MAX))
    return scale.clamp(min=q_scale_thresh)


def _reshape_imatrix(imatrix, tensor: torch.Tensor, group_size: int, bits: int):
    if not isinstance(imatrix, torch.Tensor):
        return None
    if imatrix.shape == tensor.shape:
        grouped_imatrix = imatrix.to(tensor.device)
    else:
        imatrix = imatrix.reshape(1, -1)
        grouped_imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        grouped_imatrix = grouped_imatrix.expand(tensor.numel() // grouped_imatrix.numel(), -1).reshape(tensor.shape)
    return _imatrix_handle_zero(grouped_imatrix, tensor, bits, group_size)


@register_dtype("exp_nvsint4")
def quant_tensor_exp_nvsint4(
    tensor,
    bits=4,
    group_size=16,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    init_scale=None,
    **kwargs,
):
    """Signed INT4 QDQ with signed E4M3 per-group scales."""
    if group_size != 16:
        raise ValueError(f"exp_nvsint4 only supports group_size=16, got {group_size}")
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0 ** (bits - 1))

    if init_scale is not None:
        scale = init_scale * (max_scale.unsqueeze(-1) if isinstance(max_scale, torch.Tensor) else max_scale)
    else:
        if tensor_min is None or tensor_max is None:
            wmin = torch.clamp(tensor.min(-1)[0], max=0)
            wmax = torch.clamp(tensor.max(-1)[0], min=0)
        else:
            wmin = tensor_min.to(tensor.device) if isinstance(tensor_min, torch.Tensor) else tensor_min
            wmax = tensor_max.to(tensor.device) if isinstance(tensor_max, torch.Tensor) else tensor_max
        wmin_abs = -(wmin * min_scale)
        wmax_abs = wmax * max_scale
        max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.maximum(wmax_abs, wmin_abs)
        scale = max_v / maxq

    scale = _cast_scale(scale, signed=True, q_scale_thresh=q_scale_thresh)
    scale = scale.unsqueeze(-1) if scale.ndim == tensor.ndim - 1 else scale
    int_w = round_ste(tensor / scale + v)
    quantized = int_w.clamp(-maxq, maxq - 1)
    qdq = (scale * quantized).to(tensor.dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scale, maxq


@register_dtype("rtn_exp_nvsint4")
def rtn_exp_nvsint4(tensor, **kwargs):
    return quant_tensor_exp_nvsint4(tensor, **kwargs)


@register_dtype("opt_rtn_exp_nvsint4")
def opt_rtn_exp_nvsint4(tensor, bits=4, group_size=16, v=0, imatrix=None, q_scale_thresh=1e-5, **kwargs):
    if group_size != 16:
        raise ValueError(f"exp_nvsint4 only supports group_size=16, got {group_size}")
    grouped, _, _ = reshape_pad_tensor_by_group_size(tensor, group_size)
    qw = _reshape_imatrix(imatrix, grouped, group_size, bits)
    init_scale = search_scales(grouped, bits, qw=qw)
    init_scale = torch.where(
        init_scale < 0,
        init_scale.clamp(max=-q_scale_thresh),
        init_scale.clamp(min=q_scale_thresh),
    )
    return quant_tensor_exp_nvsint4(
        tensor,
        bits=bits,
        group_size=group_size,
        v=v,
        init_scale=init_scale,
        q_scale_thresh=q_scale_thresh,
    )


@register_dtype("exp_nvint4")
def quant_tensor_exp_nvint4(
    tensor,
    bits=4,
    group_size=16,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    init_scale=None,
    **kwargs,
):
    """Symmetric signed INT4 QDQ with unsigned E5M3 per-group scales."""
    if group_size != 16:
        raise ValueError(f"exp_nvint4 only supports group_size=16, got {group_size}")
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0 ** (bits - 1))

    if init_scale is not None:
        if isinstance(max_scale, torch.Tensor):
            max_scale = max_scale.to(device=init_scale.device, dtype=init_scale.dtype)
            max_scale = max_scale.unsqueeze(-1)
        scale = init_scale * max_scale
    else:
        if tensor_min is None or tensor_max is None:
            wmin = torch.clamp(tensor.min(-1)[0], max=0)
            wmax = torch.clamp(tensor.max(-1)[0], min=0)
        else:
            wmin = tensor_min.to(tensor.device) if isinstance(tensor_min, torch.Tensor) else tensor_min
            wmax = tensor_max.to(tensor.device) if isinstance(tensor_max, torch.Tensor) else tensor_max
        max_v = torch.maximum(-(wmin * min_scale), wmax * max_scale)
        scale = max_v / maxq

    scale = _cast_scale(scale, signed=False, q_scale_thresh=q_scale_thresh)
    scale = scale.unsqueeze(-1) if scale.ndim == tensor.ndim - 1 else scale
    quantized = round_ste(tensor / scale + v).clamp(-maxq, maxq - 1)
    qdq = (scale * quantized).to(tensor.dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scale, maxq


@register_dtype("exp_nvint4_zp")
def quant_tensor_exp_nvint4_zp(
    tensor,
    bits=4,
    group_size=16,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    init_scale=None,
    init_zp=None,
    **kwargs,
):
    """Asymmetric INT4 QDQ with unsigned E5M3 scales and per-group zero points."""
    if group_size != 16:
        raise ValueError(f"exp_nvint4_zp only supports group_size=16, got {group_size}")
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0**bits) - 1

    if init_scale is not None:
        scale = init_scale * (max_scale.unsqueeze(-1) if isinstance(max_scale, torch.Tensor) else max_scale)
        scale = _cast_scale(scale, signed=False, q_scale_thresh=q_scale_thresh)
        zp = init_zp
        quantized = (round_ste(tensor / scale + v) + zp).clamp(0, maxq)
    else:
        if tensor_min is None or tensor_max is None:
            wmin = torch.clamp(tensor.min(-1)[0], max=0)
            wmax = torch.clamp(tensor.max(-1)[0], min=0)
        else:
            wmin = tensor_min.to(tensor.device) if isinstance(tensor_min, torch.Tensor) else tensor_min
            wmax = tensor_max.to(tensor.device) if isinstance(tensor_max, torch.Tensor) else tensor_max
        wmin = wmin * min_scale
        wmax = wmax * max_scale
        scale = _cast_scale((wmax - wmin) / maxq, signed=False, q_scale_thresh=q_scale_thresh)
        zp = round_ste(-wmin / scale)
        scale = scale.unsqueeze(-1)
        zp = zp.unsqueeze(-1)
        quantized = (round_ste(tensor / scale + v) + zp).clamp(0, maxq)

    qdq = (scale * (quantized - zp)).to(tensor.dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scale, zp


@register_dtype("rtn_exp_nvint4")
def rtn_exp_nvint4(tensor, **kwargs):
    return quant_tensor_exp_nvint4(tensor, **kwargs)


@register_dtype("opt_rtn_exp_nvint4")
def opt_rtn_exp_nvint4(tensor, bits=4, group_size=16, v=0, imatrix=None, q_scale_thresh=1e-5, **kwargs):
    if group_size != 16:
        raise ValueError(f"exp_nvint4 only supports group_size=16, got {group_size}")
    kwargs.pop("init_scale", None)
    grouped, _, _ = reshape_pad_tensor_by_group_size(tensor, group_size)
    qw = _reshape_imatrix(imatrix, grouped, group_size, bits)
    init_scale = search_scales(grouped, bits, qw=qw).abs().clamp(min=q_scale_thresh)
    return quant_tensor_exp_nvint4(
        tensor,
        bits=bits,
        group_size=group_size,
        v=v,
        init_scale=init_scale,
        q_scale_thresh=q_scale_thresh,
        **kwargs,
    )


def search_exp_nvint4_zp_scale_zero(weight_reshape, bits, imatrix=None, q_scale_thresh=1e-5):
    qw = _reshape_imatrix(imatrix, weight_reshape, 16, bits)
    maxq = 2**bits - 1
    weight = weight_reshape.to(torch.float32)
    wmin = weight.min(dim=-1, keepdim=True).values.clamp(max=0)
    wmax = weight.max(dim=-1, keepdim=True).values.clamp(min=0)
    scale = _cast_scale((wmax - wmin) / maxq, signed=False, q_scale_thresh=q_scale_thresh)

    zp_grid = torch.arange(maxq + 1, device=weight_reshape.device, dtype=torch.float32)
    rounded = torch.round(weight / scale.to(torch.float32))
    quantized = (rounded.unsqueeze(1) + zp_grid.reshape(1, -1, 1)).clamp(0, maxq)
    dequantized = scale.to(torch.float32).unsqueeze(1) * (quantized - zp_grid.reshape(1, -1, 1))
    losses = (dequantized - weight.unsqueeze(1)).square()
    if qw is not None:
        losses = losses * qw.to(torch.float32).unsqueeze(1)
    best_zp = zp_grid[losses.sum(dim=-1).argmin(dim=-1)].unsqueeze(-1)
    return scale, best_zp.to(weight_reshape.dtype)


@register_dtype("rtn_exp_nvint4_zp")
def rtn_exp_nvint4_zp(tensor, **kwargs):
    return quant_tensor_exp_nvint4_zp(tensor, **kwargs)


@register_dtype("opt_rtn_exp_nvint4_zp")
def opt_rtn_exp_nvint4_zp(tensor, bits=4, group_size=16, v=0, imatrix=None, q_scale_thresh=1e-5, **kwargs):
    if group_size != 16:
        raise ValueError(f"exp_nvint4_zp only supports group_size=16, got {group_size}")
    grouped, _, _ = reshape_pad_tensor_by_group_size(tensor, group_size)
    init_scale, init_zp = search_exp_nvint4_zp_scale_zero(grouped, bits, imatrix, q_scale_thresh)
    return quant_tensor_exp_nvint4_zp(
        tensor,
        bits=bits,
        group_size=group_size,
        v=v,
        init_scale=init_scale,
        init_zp=init_zp,
        q_scale_thresh=q_scale_thresh,
    )
