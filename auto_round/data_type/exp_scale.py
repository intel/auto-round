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

from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste


def _cast_fp4(value: torch.Tensor) -> torch.Tensor:
    magnitude = value.abs()
    step_half = round_ste(2.0 * magnitude) / 2.0
    step_one = round_ste(magnitude)
    step_two = 2.0 * round_ste(magnitude / 2.0)
    rounded = torch.where(magnitude < 2.0, step_half, torch.where(magnitude < 4.0, step_one, step_two))
    return rounded.clamp(max=6.0) * value.sign()


def _scale_dtype_scale(scale: torch.Tensor, dtype: torch.dtype, threshold: float = 1e-5) -> torch.Tensor:
    scale = scale.to(dtype)
    return torch.where(scale < 0, scale.clamp(max=-threshold), scale.clamp(min=threshold))


def _as_group_value(value, grouped: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=grouped.device, dtype=torch.float32)
        if value.numel() > 1:
            value = value.reshape(-1, 1)
        return value
    return torch.tensor(value, device=grouped.device, dtype=torch.float32)


def _quant_exp_fp4(
    tensor: torch.Tensor,
    scale_dtype: torch.dtype,
    bits: int = 4,
    group_size: int = 16,
    v: float = 0,
    max_scale=1.0,
    init_scale=None,
    **kwargs,
):
    del bits, kwargs
    if group_size != 16:
        raise ValueError(f"EXP FP4 scale experiments only support group_size=16, got {group_size}")
    orig_dtype = tensor.dtype
    grouped, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    scale = grouped.to(torch.float32).abs().amax(dim=-1, keepdim=True) / 6.0
    scale = scale * _as_group_value(max_scale, grouped)
    if init_scale is not None:
        scale = scale * _as_group_value(init_scale, grouped)
    scale = _scale_dtype_scale(scale, scale_dtype)
    scale_fp32 = scale.to(torch.float32)
    normalized = grouped.to(torch.float32) / scale_fp32 + v
    qdq = _cast_fp4(normalized.clamp(-6.0, 6.0)) * scale_fp32
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq.to(orig_dtype), scale, None


def _quant_exp_sint4(
    tensor: torch.Tensor,
    scale_dtype: torch.dtype,
    bits: int = 4,
    group_size: int = 16,
    v: float = 0,
    min_scale=1.0,
    max_scale=1.0,
    tensor_min=None,
    tensor_max=None,
    init_scale=None,
    **kwargs,
):
    del kwargs
    if group_size != 16:
        raise ValueError(f"EXP SINT4 scale experiments only support group_size=16, got {group_size}")
    orig_dtype = tensor.dtype
    grouped, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    grouped_fp32 = grouped.to(torch.float32)
    if tensor_min is None or tensor_max is None:
        weight_min = grouped_fp32.amin(dim=-1).clamp(max=0)
        weight_max = grouped_fp32.amax(dim=-1).clamp(min=0)
    else:
        weight_min = _as_group_value(tensor_min, grouped).squeeze(-1)
        weight_max = _as_group_value(tensor_max, grouped).squeeze(-1)
    if init_scale is None:
        min_scale = _as_group_value(min_scale, grouped)
        max_scale = _as_group_value(max_scale, grouped)
        min_abs = -(weight_min * min_scale.squeeze(-1))
        max_abs = weight_max * max_scale.squeeze(-1)
        signed_max = (2 * (max_abs < min_abs).to(torch.float32) - 1) * torch.maximum(max_abs, min_abs)
        scale = signed_max.unsqueeze(-1) / (2**bits // 2)
    else:
        scale = _as_group_value(init_scale, grouped) * _as_group_value(max_scale, grouped)
    scale = _scale_dtype_scale(scale, scale_dtype)
    scale_fp32 = scale.to(torch.float32)
    quantized = round_ste(grouped_fp32 / scale_fp32 + v).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    qdq = (scale_fp32 * quantized).to(orig_dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scale, 2 ** (bits - 1)


def _quant_exp_int4_zp(
    tensor: torch.Tensor,
    scale_dtype: torch.dtype,
    bits: int = 4,
    group_size: int = 16,
    v: float = 0,
    min_scale=1.0,
    max_scale=1.0,
    tensor_min=None,
    tensor_max=None,
    init_scale=None,
    init_zp=None,
    **kwargs,
):
    del kwargs
    if group_size != 16:
        raise ValueError(f"EXP INT4 scale experiments only support group_size=16, got {group_size}")
    orig_dtype = tensor.dtype
    grouped, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    grouped_fp32 = grouped.to(torch.float32)
    if tensor_min is None or tensor_max is None:
        weight_min = grouped_fp32.amin(dim=-1, keepdim=True).clamp(max=0)
        weight_max = grouped_fp32.amax(dim=-1, keepdim=True).clamp(min=0)
    else:
        weight_min = _as_group_value(tensor_min, grouped)
        weight_max = _as_group_value(tensor_max, grouped)
    maxq = 2**bits - 1
    if init_scale is None:
        scale = (weight_max * _as_group_value(max_scale, grouped) - weight_min * _as_group_value(min_scale, grouped)) / maxq
    else:
        scale = _as_group_value(init_scale, grouped) * _as_group_value(max_scale, grouped)
    scale = _scale_dtype_scale(scale, scale_dtype)
    scale_fp32 = scale.to(torch.float32)
    if init_zp is None:
        zero_point = round_ste(-weight_min / scale_fp32)
    else:
        zero_point = _as_group_value(init_zp, grouped)
    quantized = (round_ste(grouped_fp32 / scale_fp32 + v) + zero_point).clamp(0, maxq)
    qdq = (scale_fp32 * (quantized - zero_point)).to(orig_dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scale, zero_point


def _quant_exp_int4(
    tensor: torch.Tensor,
    scale_dtype: torch.dtype,
    bits: int = 4,
    group_size: int = 16,
    v: float = 0,
    min_scale=1.0,
    max_scale=1.0,
    tensor_min=None,
    tensor_max=None,
    init_scale=None,
    **kwargs,
):
    del kwargs
    if group_size != 16:
        raise ValueError(f"EXP INT4 scale experiments only support group_size=16, got {group_size}")
    orig_dtype = tensor.dtype
    grouped, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    grouped_fp32 = grouped.to(torch.float32)
    if tensor_min is None or tensor_max is None:
        weight_min = grouped_fp32.amin(dim=-1).clamp(max=0)
        weight_max = grouped_fp32.amax(dim=-1).clamp(min=0)
    else:
        weight_min = _as_group_value(tensor_min, grouped).squeeze(-1)
        weight_max = _as_group_value(tensor_max, grouped).squeeze(-1)
    maxq = 2 ** (bits - 1) - 1
    if init_scale is None:
        min_abs = -(weight_min * _as_group_value(min_scale, grouped).squeeze(-1))
        max_abs = weight_max * _as_group_value(max_scale, grouped).squeeze(-1)
        scale = torch.maximum(min_abs, max_abs).unsqueeze(-1) / maxq
    else:
        scale = _as_group_value(init_scale, grouped) * _as_group_value(max_scale, grouped)
    scale = _scale_dtype_scale(scale, scale_dtype)
    scale_fp32 = scale.to(torch.float32)
    quantized = round_ste(grouped_fp32 / scale_fp32 + v).clamp(-maxq, maxq)
    qdq = (scale_fp32 * quantized).to(orig_dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq, scale, None


def _make_quantizer(kind: str, scale_dtype: torch.dtype):
    implementation = {
        "fp4": _quant_exp_fp4,
        "int4": _quant_exp_int4,
        "int4_zp": _quant_exp_int4_zp,
        "sint4": _quant_exp_sint4,
    }[kind]

    def quantizer(tensor, **kwargs):
        kwargs.pop("scale_dtype", None)
        return implementation(tensor, scale_dtype=scale_dtype, **kwargs)

    return quantizer


for _kind in ("fp4", "int4", "int4_zp", "sint4"):
    for _suffix, _scale_dtype in (("bf16s", torch.bfloat16), ("fp16s", torch.float16)):
        _data_type = f"exp_{_kind}_{_suffix}"
        _quantizer = _make_quantizer(_kind, _scale_dtype)
        _quantizer.__name__ = f"quant_tensor_{_data_type}"
        register_dtype(_data_type)(_quantizer)
        register_dtype(f"rtn_{_data_type}")(_quantizer)