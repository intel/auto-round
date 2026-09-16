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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

import torch

from auto_round import envs
from auto_round.data_type.base import register_dtype, register_quantizer
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.utils import get_reciprocal


def search_scales(data: torch.Tensor, bits: int, qw: Union[None, torch.Tensor, float] = None) -> torch.Tensor:
    # Maximum absolute value for symmetric quantization
    nmax = int(2.0 ** (bits - 1))

    # Find per-group max along the last dimension
    imax = torch.abs(data).argmax(dim=-1, keepdim=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)

    # Compute initial inverse scales
    iscales = -nmax * get_reciprocal(group_max)
    scales = get_reciprocal(iscales)  # scale = 1 / iscales

    # Initial quantized values (in-place round and clamp)
    L = torch.empty_like(data)
    torch.round(iscales * data, out=L)
    L.clamp_(-nmax, nmax - 1)

    # Set default weight if None
    if qw is None:
        qw = 1.0
    # Compute initial best loss
    best_loss = ((scales * L - data).to(torch.float32)) ** 2
    if isinstance(qw, torch.Tensor):
        best_loss.mul_(qw)  # inplace multiply by weight
    best_loss = torch.sum(best_loss, dim=-1)
    if bits == 2:
        search_min = 18 * 5
        step = 0.01
    else:
        grid = 200
        search_ratio = envs.AR_SEARCH_SCALE_RATIO or 0.75  # default 0.5 -> nmax/2
        search_min = nmax * search_ratio
        step = search_min / grid * 2  # 0.08
        search_min = int(search_min / step)
    # Iterative search over small adjustments
    for _is in range(-search_min, search_min + 1):
        if _is == 0:
            continue

        # Update iscales in-place
        iscales_tmp = -(nmax - step * _is) * get_reciprocal(group_max)

        # Compute temporary quantized values (in-place round + clamp)
        tmp_L = torch.empty_like(data)
        torch.round(iscales_tmp * data, out=tmp_L)
        tmp_L.clamp_(-nmax, nmax - 1)

        # Compute temporary scales
        tmp_scales = get_reciprocal(iscales_tmp)

        # Compute temporary loss
        loss = ((tmp_scales * tmp_L - data).to(torch.float32)) ** 2
        if isinstance(qw, torch.Tensor):
            loss.mul_(qw)
        loss = torch.sum(loss, dim=-1)

        # Replace scales where loss improves (in-place)
        replace_id = loss < best_loss
        if replace_id.any():
            scales[replace_id] = tmp_scales[replace_id]
            best_loss[replace_id] = loss[replace_id]

    return scales


@register_dtype("opt_rtn_int_sym")
def quant_tensor_opt_rtn_sym(tensor, bits=4, group_size=-1, v=0, q_scale_thresh=1e-5, imatrix=None, **kwargs):
    """Quantize and de-quantize tensor asymmetrically. full range, credit goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    from auto_round.data_type.gguf import _imatrix_handle_zero

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0 ** (bits - 1))
    if imatrix is None:
        imatrix = 1.0
    else:
        imatrix = imatrix.reshape(1, -1)
        imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        imatrix = imatrix.expand(tensor.numel() // imatrix.numel(), -1)
        imatrix = imatrix.reshape(tensor.shape)

        imatrix = _imatrix_handle_zero(imatrix, tensor, bits, group_size)

    scale = search_scales(tensor, bits, qw=imatrix)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    int_w = tensor.div(scale).round_().clamp_(-maxq, maxq - 1)
    qdq_result = (int_w.mul_(scale)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, maxq


@register_dtype("rtn_int_sym")
def quant_tensor_rtn_sym(
    tensor,
    bits=4,
    group_size=-1,
    q_scale_thresh=1e-5,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    **kwargs,
):
    """Quantize and de-quantize tensor asymmetrically. full range, credit goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0 ** (bits - 1))

    wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
    wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    wmin_abs = -(wmin_tmp * min_scale)  # pylint: disable=E1130
    wmax_abs = wmax_tmp * max_scale
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    scale = scale.unsqueeze(dim=-1)
    int_w = tensor.div(scale).round_().clamp_(-maxq, maxq - 1)
    qdq_result = (int_w.mul_(scale)).to(tensor.dtype)
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
    init_scale=None,
    **kwargs,
):
    """Quantize and de-quantize tensor asymmetrically. full range, credit goes to llamacpp community

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
    maxq = int(2.0 ** (bits - 1))

    if init_scale is not None:
        # ``max_scale`` is a per-group tuning coefficient (Tensor) during
        # SignRound optimization, but may be a plain scalar (e.g. 1.0) when the
        # init_scale is reused for a one-shot QDQ such as AWQ's smooth/clip grid
        # search.
        if isinstance(max_scale, torch.Tensor):
            scale = init_scale * max_scale.unsqueeze(dim=-1)
        else:
            scale = init_scale * max_scale
        scale = scale.to(scale_dtype)
        scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
        int_w = round_ste(tensor / scale + v)
        q = torch.clamp(int_w, -maxq, maxq - 1)
        qdq_result = (scale * q).to(tensor.dtype)
        qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
        return qdq_result, scale, maxq

    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
        if isinstance(wmin_tmp, torch.Tensor):
            wmin_tmp = wmin_tmp.to(tensor.device)
            wmax_tmp = wmax_tmp.to(tensor.device)

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
    maxq = int(2.0**bits) - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
        if isinstance(wmin_tmp, torch.Tensor):
            wmin_tmp = wmin_tmp.to(tensor.device)
            wmax_tmp = wmax_tmp.to(tensor.device)
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
    maxq = int(2.0**bits) - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
        if isinstance(wmin_tmp, torch.Tensor):
            wmin_tmp = wmin_tmp.to(tensor.device)
            wmax_tmp = wmax_tmp.to(tensor.device)
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
    **kwargs,
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
    maxq = int(2.0**bits) - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
        if isinstance(wmin_tmp, torch.Tensor):
            wmin_tmp = wmin_tmp.to(tensor.device)
            wmax_tmp = wmax_tmp.to(tensor.device)
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


@dataclass(frozen=True)
class _IntState:
    """Per-layer integer ranges, optional optimized scale, and trainable values."""

    tunables: Mapping[str, torch.Tensor]
    tensor_min: torch.Tensor
    tensor_max: torch.Tensor
    optimized_init: torch.Tensor | None


class _IntWeightQuantizer:
    """Own integer weight QDQ and choose tuned, RTN, or optimized RTN internally."""

    def __init__(self, spec, family="plain"):
        self.spec = spec
        self.family = family

    @classmethod
    def from_spec(cls, spec, canonical=None):
        """Create the integer weight quantizer for a resolved layer."""
        return cls(spec)

    @staticmethod
    def create_activation(spec):
        """Create the matching integer activation quantizer."""
        return _IntActivationQuantizer(spec)

    def create_state(self, weight, *, imatrix=None, mode, tune_rounding, tune_minmax):
        if mode == "optimized_rtn" and self.spec.sym:
            self.family = "optimized"
        elif mode == "rtn" and self.spec.sym:
            self.family = "rtn"
        else:
            self.family = "plain"
        grouped, _, _ = reshape_pad_tensor_by_group_size(weight, self.spec.group_size)
        tensor_min = torch.clamp(grouped.amin(dim=-1), max=0)
        tensor_max = torch.clamp(grouped.amax(dim=-1), min=0)
        if self.spec.clip_max is not None:
            clip_max = self.spec.clip_max.reshape(-1).to(weight.device, tensor_max.dtype)
            clip_min = (
                -clip_max
                if self.spec.clip_min is None
                else self.spec.clip_min.reshape(-1).to(weight.device, tensor_min.dtype)
            )
            if clip_min.numel() == tensor_min.numel() and clip_max.numel() == tensor_max.numel():
                tensor_min = torch.maximum(tensor_min, clip_min.reshape_as(tensor_min))
                tensor_max = torch.minimum(tensor_max, clip_max.reshape_as(tensor_max))

        tunables = {}
        if self.family == "plain" and tune_rounding:
            tunables["value"] = torch.nn.Parameter(torch.zeros_like(grouped, dtype=torch.float32))
        if self.family != "optimized" and tune_minmax:
            shape = tensor_min.shape
            tunables["min_scale"] = torch.nn.Parameter(torch.ones(shape, device=weight.device, dtype=torch.float32))
            tunables["max_scale"] = torch.nn.Parameter(torch.ones(shape, device=weight.device, dtype=torch.float32))

        optimized_init = None
        if self.family == "optimized" and self.spec.sym:
            search_weight = weight
            if self.spec.clip_min is not None or self.spec.clip_max is not None:
                search_weight = torch.clamp(weight, min=self.spec.clip_min, max=self.spec.clip_max)
            _, optimized_init, _ = quant_tensor_opt_rtn_sym(
                search_weight.clone(),
                bits=self.spec.bits,
                group_size=self.spec.group_size,
                q_scale_thresh=self.spec.q_scale_thresh,
                imatrix=imatrix,
            )
        return _IntState(tunables, tensor_min, tensor_max, optimized_init)

    def qdq(self, weight, state, *, tunables, materialize=False):
        value = tunables.get("value", 0)
        min_scale = tunables.get("min_scale", 1.0)
        max_scale = tunables.get("max_scale", 1.0)
        if isinstance(min_scale, torch.Tensor):
            min_scale.data.clamp_(0.0, 1.0)
        if isinstance(max_scale, torch.Tensor):
            max_scale.data.clamp_(0.0, 1.0)

        kwargs = {
            "bits": self.spec.bits,
            "group_size": self.spec.group_size,
            "q_scale_thresh": self.spec.q_scale_thresh,
        }
        if self.family == "optimized" and self.spec.sym:
            quantized, scale, zero_point = quant_tensor_sym(
                weight,
                init_scale=state.optimized_init,
                scale_dtype=self.spec.scale_dtype,
                **kwargs,
            )
        elif self.family == "rtn" and self.spec.sym:
            quantized, scale, zero_point = quant_tensor_rtn_sym(
                weight,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_dtype=self.spec.scale_dtype,
                **kwargs,
            )
        else:
            primitive = quant_tensor_sym if self.spec.sym else quant_tensor_asym
            quantized, scale, zero_point = primitive(
                weight,
                v=value,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_dtype=self.spec.scale_dtype,
                tensor_min=state.tensor_min,
                tensor_max=state.tensor_max,
                **kwargs,
            )
        from auto_round.data_type.base import WeightQuantizationResult

        return WeightQuantizationResult(quantized, scale if materialize else None, zero_point if materialize else None)

    @staticmethod
    def apply_result(module, result):
        if result.scale is None:
            raise ValueError("INT weight result was not materialized")
        module.weight.data.copy_(result.weight)
        rows = result.logical_rows or result.weight.shape[0]
        module.scale = result.scale.reshape(rows, -1).cpu()
        module.zp = (
            result.zero_point.reshape(rows, -1).cpu()
            if isinstance(result.zero_point, torch.Tensor)
            else result.zero_point
        )


class _IntActivationQuantizer:
    """Quantize integer activations, optionally using a calibrated maximum."""

    def __init__(self, spec):
        self.spec = spec
        self.requires_calibration = not spec.dynamic

    def observe(self, activation, current):
        grouped, _, _ = reshape_pad_tensor_by_group_size(activation, self.spec.group_size)
        maximum = grouped.abs().amax(dim=-1)
        return maximum if current is None else torch.maximum(maximum.to(current), current)

    def qdq_with_scale(self, activation, *, observed_max=None, min_scale=1.0, max_scale=1.0):
        if self.requires_calibration and observed_max is None:
            raise ValueError(f"{self.spec.data_type} activation requires observed_max")
        primitive = quant_tensor_sym if self.spec.sym else quant_tensor_asym
        kwargs = {}
        if observed_max is not None:
            kwargs.update(tensor_min=-observed_max, tensor_max=observed_max)
        return primitive(
            activation,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=self.spec.scale_dtype,
            q_scale_thresh=self.spec.q_scale_thresh,
            **kwargs,
        )

    def qdq(self, activation, *, observed_max=None, min_scale=1.0, max_scale=1.0):
        quantized, _, _ = self.qdq_with_scale(
            activation, observed_max=observed_max, min_scale=min_scale, max_scale=max_scale
        )
        return quantized


register_quantizer(
    "int_sym",
    aliases=(
        "int",
        "int4",
        "int4_sym",
        "int8",
        "int8_sym",
        "rtn_int",
        "rtn_int4",
        "rtn_int_sym",
        "rtn_int4_sym",
        "opt_rtn_int",
        "opt_rtn_int4",
        "opt_rtn_int_sym",
        "opt_rtn_int4_sym",
    ),
)(_IntWeightQuantizer)
register_quantizer(
    "int_asym",
    aliases=("int4_asym", "rtn_int_asym", "rtn_int4_asym", "opt_rtn_int_asym", "opt_rtn_int4_asym"),
)(_IntWeightQuantizer)
