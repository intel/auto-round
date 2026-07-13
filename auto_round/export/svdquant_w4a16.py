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

"""AutoRound-owned Nunchaku W4A16 AdaNorm tensor codec."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedW4A16:
    """Physical W4A16 tensors and the metadata needed to invert their layout."""

    qweight: torch.Tensor
    wscales: torch.Tensor
    wzeros: torch.Tensor
    bias: torch.Tensor
    dtype: torch.dtype
    splits: int
    group_size: int


def _channel_major(tensor: torch.Tensor, splits: int) -> torch.Tensor:
    rows = tensor.shape[0]
    return tensor.reshape(splits, rows // splits, *tensor.shape[1:]).transpose(0, 1).reshape_as(tensor)


def _validate_inputs(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    splits: int,
    group_size: int,
) -> tuple[int, int, int, torch.Tensor]:
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.dtype
        not in (
            torch.bfloat16,
            torch.float16,
        )
    ):
        raise ValueError("weight must be a BF16 or FP16 tensor with shape [O, K]")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must contain only finite values")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size != 64:
        raise ValueError("group_size must be 64")
    if isinstance(splits, bool) or not isinstance(splits, int) or splits not in (3, 6):
        raise ValueError("splits must be 3 or 6")
    out_features, in_features = weight.shape
    if out_features == 0 or in_features == 0:
        raise ValueError("weight dimensions must be non-empty")
    if in_features % group_size:
        raise ValueError("K must be divisible by group_size 64")
    if out_features % splits:
        raise ValueError("O must be divisible by splits")
    if out_features % 4:
        raise ValueError("O must be divisible by 4")
    num_groups = in_features // group_size
    if num_groups % 16:
        raise ValueError("runtime G=K/64 must be divisible by 16")
    if not isinstance(scale, torch.Tensor) or not scale.is_floating_point():
        raise ValueError("scale must be a floating-point tensor")
    if scale.device != weight.device:
        raise ValueError("scale must be on the same device as weight")
    if tuple(scale.shape) not in ((out_features, num_groups), (out_features, 1, num_groups, 1)):
        raise ValueError(f"scale shape must be [O, G] or [O, 1, G, 1], got {tuple(scale.shape)}")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("scale must contain only positive finite values")
    if bias is not None:
        if (
            not isinstance(bias, torch.Tensor)
            or not bias.is_floating_point()
            or tuple(bias.shape) != (out_features,)
            or bias.device != weight.device
            or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("bias must be a finite floating-point [O] tensor on the weight device")
    return out_features, in_features, num_groups, scale.reshape(out_features, num_groups)


def pack_adanorm_w4a16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    splits: int = 3,
    group_size: int = 64,
) -> PackedW4A16:
    """Pack pre-scaled signed INT4 AdaNorm weights for Nunchaku W4A16."""

    out_features, in_features, num_groups, logical_scale = _validate_inputs(weight, scale, bias, splits, group_size)
    signed = torch.round(
        weight.float().reshape(out_features, num_groups, group_size) / logical_scale.float().unsqueeze(-1)
    )
    if not bool(((signed >= -7) & (signed <= 7)).all()):
        raise ValueError("quantized weight must be in [-7, 7]")
    codes = _channel_major(signed.reshape(out_features, in_features).to(torch.int32) + 7, splits)

    packed16 = codes.view(-1, 4, 8)
    packed16 = packed16[:, 0] | (packed16[:, 1] << 4) | (packed16[:, 2] << 8) | (packed16[:, 3] << 12)
    packed16 = (
        packed16.view(out_features // 4, 4, in_features // 64, 16)
        .permute(0, 2, 1, 3)
        .reshape(out_features // 4, in_features)
        .to(torch.int16)
    )
    qweight = packed16.view(torch.int32)

    channel_scale = _channel_major(logical_scale, splits).to(weight.dtype)
    if bias is None:
        channel_bias = torch.zeros(out_features, dtype=weight.dtype, device=weight.device)
    else:
        channel_bias = _channel_major(bias.reshape(out_features), splits).to(weight.dtype)
    channel_bias = channel_bias.reshape(out_features // splits, splits)
    channel_bias[:, sorted({1, splits - 2})] += 1
    channel_bias = channel_bias.reshape(out_features)
    return PackedW4A16(
        qweight=qweight,
        wscales=channel_scale.t().contiguous(),
        wzeros=(-7 * channel_scale).t().contiguous(),
        bias=channel_bias,
        dtype=weight.dtype,
        splits=splits,
        group_size=group_size,
    )


def unpack_adanorm_w4a16(packed: PackedW4A16) -> torch.Tensor:
    """Recover channel-major signed INT4 values from a packed payload."""

    out_features = packed.wscales.shape[1]
    in_features = packed.qweight.shape[1] * 2
    packed16 = packed.qweight.contiguous().view(torch.int16).reshape(out_features // 4, in_features)
    packed16 = packed16.view(out_features // 4, in_features // 64, 4, 16).permute(0, 2, 1, 3).contiguous().view(-1, 8)
    shifts = torch.arange(0, 16, 4, dtype=torch.int16, device=packed.qweight.device)
    codes = ((packed16.unsqueeze(1) >> shifts.view(1, 4, 1)) & 0xF).reshape(out_features, in_features)
    return (codes - 7).to(torch.int8)


def dequantize_adanorm_w4a16(packed: PackedW4A16) -> torch.Tensor:
    """Decode the packed weights into their QDQ values in runtime channel order."""

    signed = unpack_adanorm_w4a16(packed).to(torch.float32)
    out_features, in_features = signed.shape
    scales = packed.wscales.t().reshape(out_features, in_features // packed.group_size, 1).float()
    return (signed.reshape(out_features, -1, packed.group_size) * scales).reshape_as(signed).to(packed.dtype)


def quantize_adanorm_w4a16_rtn(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    splits: int = 3,
    group_size: int = 64,
) -> PackedW4A16:
    """Compute symmetric per-row-group RTN scales and pack the result.

    An all-zero group uses scale one. Its signed values and QDQ values remain
    exactly zero, while the emitted scale continues to satisfy the runtime's
    positive-scale contract.
    """

    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.dtype
        not in (
            torch.bfloat16,
            torch.float16,
        )
    ):
        raise ValueError("weight must be a BF16 or FP16 tensor with shape [O, K]")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size != 64:
        raise ValueError("group_size must be 64")
    out_features, in_features = weight.shape
    if out_features == 0 or in_features == 0:
        raise ValueError("weight dimensions must be non-empty")
    if in_features % group_size:
        raise ValueError("K must be divisible by group_size 64")
    grouped = weight.float().reshape(out_features, in_features // group_size, group_size)
    absmax = grouped.abs().amax(dim=-1)
    scales = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 7).to(weight.dtype)
    return pack_adanorm_w4a16(weight, scales, bias=bias, splits=splits, group_size=group_size)
