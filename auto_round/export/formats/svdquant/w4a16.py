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

import sys
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
    logical_shape: tuple[int, int]
    splits: int
    group_size: int


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise ValueError("Nunchaku W4A16 packing requires a little-endian host")


def _pack_code_rows(codes: torch.Tensor) -> torch.Tensor:
    rows, in_features = codes.shape
    packed16 = codes.view(-1, 4, 8)
    packed16 = packed16[:, 0] | (packed16[:, 1] << 4) | (packed16[:, 2] << 8) | (packed16[:, 3] << 12)
    packed16 = (
        packed16.view(rows // 4, 4, in_features // 64, 16)
        .permute(0, 2, 1, 3)
        .reshape(rows // 4, in_features)
        .to(torch.int16)
    )
    return packed16.view(torch.int32)


def _effective_chunk_rows(chunk_rows: int | None, out_features: int) -> int:
    if chunk_rows is None:
        return out_features
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0 or chunk_rows % 4:
        raise ValueError("chunk_rows must be None or a positive multiple of 4")
    return min(chunk_rows, out_features)


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
    if scale.dtype != weight.dtype or scale.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("scale dtype must exactly match weight dtype and be BF16 or FP16")
    if scale.device != weight.device:
        raise ValueError("scale must be on the same device as weight")
    if tuple(scale.shape) not in ((out_features, num_groups), (out_features, 1, num_groups, 1)):
        raise ValueError(f"scale shape must be [O, G] or [O, 1, G, 1], got {tuple(scale.shape)}")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("scale must contain only positive finite values")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
            raise ValueError("bias must be a floating-point tensor")
        if bias.dtype != weight.dtype or bias.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("bias dtype must exactly match weight dtype and be BF16 or FP16")
        if tuple(bias.shape) != (out_features,) or bias.device != weight.device or not bool(torch.isfinite(bias).all()):
            raise ValueError("bias must be a finite floating-point [O] tensor on the weight device")
    return out_features, in_features, num_groups, scale.reshape(out_features, num_groups)


def pack_adanorm_w4a16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    splits: int = 3,
    group_size: int = 64,
    chunk_rows: int | None = 256,
) -> PackedW4A16:
    """Pack pre-scaled signed INT4 AdaNorm weights for Nunchaku W4A16."""

    _require_little_endian()
    out_features, in_features, num_groups, logical_scale = _validate_inputs(weight, scale, bias, splits, group_size)
    rows_per_chunk = _effective_chunk_rows(chunk_rows, out_features)
    channels_per_field = out_features // splits
    qweight = torch.empty((out_features // 4, in_features // 2), dtype=torch.int32, device=weight.device)
    channel_scale = torch.empty((out_features, num_groups), dtype=weight.dtype, device=weight.device)
    channel_bias = torch.zeros(out_features, dtype=weight.dtype, device=weight.device)
    for start in range(0, out_features, rows_per_chunk):
        end = min(start + rows_per_chunk, out_features)
        output_rows = torch.arange(start, end, device=weight.device)
        source_rows = output_rows.remainder(splits) * channels_per_field + torch.div(
            output_rows, splits, rounding_mode="floor"
        )
        weight_rows = weight.index_select(0, source_rows)
        scale_rows = logical_scale.index_select(0, source_rows)
        normalized = weight_rows.float().reshape(end - start, num_groups, group_size)
        normalized.div_(scale_rows.float().unsqueeze(-1)).round_()
        if not bool(((normalized >= -7) & (normalized <= 7)).all()):
            raise ValueError("quantized weight must be in [-7, 7]")
        codes = normalized.reshape(end - start, in_features).to(torch.int32).add_(7)
        qweight[start // 4 : end // 4] = _pack_code_rows(codes)
        channel_scale[start:end] = scale_rows
        if bias is not None:
            channel_bias[start:end] = bias.index_select(0, source_rows)
        del codes, normalized, scale_rows, weight_rows
    channel_bias = channel_bias.reshape(out_features // splits, splits)
    identity_fields = sorted({1, splits - 2})
    identity_before = channel_bias[:, identity_fields].clone()
    channel_bias[:, identity_fields] += 1
    if bool((channel_bias[:, identity_fields] == identity_before).any()):
        raise ValueError(f"AdaNorm bias identity offset +1 must change the stored {weight.dtype} value")
    channel_bias = channel_bias.reshape(out_features)
    wscales = channel_scale.t().contiguous()
    wzeros = (-7 * channel_scale).t().contiguous()
    for name, tensor in (("wscales", wscales), ("wzeros", wzeros), ("bias", channel_bias)):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must remain finite in {weight.dtype} after packing arithmetic")
    return PackedW4A16(
        qweight=qweight,
        wscales=wscales,
        wzeros=wzeros,
        bias=channel_bias,
        dtype=weight.dtype,
        logical_shape=(out_features, in_features),
        splits=splits,
        group_size=group_size,
    )


def _validate_packed_w4a16(packed: PackedW4A16) -> tuple[int, int, int]:
    _require_little_endian()
    if not isinstance(packed, PackedW4A16):
        raise ValueError("packed must be a PackedW4A16 payload")
    if packed.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("payload dtype must be torch.bfloat16 or torch.float16")
    if (
        not isinstance(packed.logical_shape, tuple)
        or len(packed.logical_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in packed.logical_shape)
    ):
        raise ValueError("logical_shape must be a positive (O, K) integer tuple")
    out_features, in_features = packed.logical_shape
    if isinstance(packed.splits, bool) or not isinstance(packed.splits, int) or packed.splits not in (3, 6):
        raise ValueError("splits must be 3 or 6")
    if isinstance(packed.group_size, bool) or not isinstance(packed.group_size, int) or packed.group_size != 64:
        raise ValueError("group_size must be 64")
    if out_features % packed.splits or out_features % 4 or in_features % packed.group_size:
        raise ValueError("logical_shape is incompatible with splits, output packing, or group_size")
    num_groups = in_features // packed.group_size
    if num_groups % 16:
        raise ValueError("logical_shape must produce a runtime group count divisible by 16")
    if not isinstance(packed.qweight, torch.Tensor) or packed.qweight.dtype != torch.int32:
        raise ValueError("qweight dtype must be torch.int32")
    if tuple(packed.qweight.shape) != (out_features // 4, in_features // 2):
        raise ValueError("qweight shape is inconsistent with logical_shape")
    expected_shapes = {
        "wscales": (num_groups, out_features),
        "wzeros": (num_groups, out_features),
        "bias": (out_features,),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = getattr(packed, name)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != packed.dtype:
            raise ValueError(f"{name} dtype must exactly match payload dtype")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}")
        if tensor.device != packed.qweight.device:
            raise ValueError("all packed tensors must be on the same device")
    for name in ("wscales", "wzeros", "bias"):
        if not bool(torch.isfinite(getattr(packed, name)).all()):
            raise ValueError(f"{name} must contain only finite values")
    if not bool((packed.wscales > 0).all()):
        raise ValueError("wscales must contain only positive values")
    if not torch.equal(packed.wzeros, packed.wscales * -7):
        raise ValueError("wzeros must equal -7 * wscales in payload dtype")
    return out_features, in_features, num_groups


def unpack_adanorm_w4a16(packed: PackedW4A16) -> torch.Tensor:
    """Recover channel-major signed INT4 values from a packed payload."""

    out_features, in_features, _ = _validate_packed_w4a16(packed)
    packed16 = packed.qweight.contiguous().view(torch.int16).reshape(out_features // 4, in_features)
    packed16 = packed16.view(out_features // 4, in_features // 64, 4, 16).permute(0, 2, 1, 3).contiguous().view(-1, 8)
    shifts = torch.arange(0, 16, 4, dtype=torch.int16, device=packed.qweight.device)
    codes = ((packed16.unsqueeze(1) >> shifts.view(1, 4, 1)) & 0xF).reshape(out_features, in_features)
    if not bool((codes <= 14).all()):
        raise ValueError("qweight contains codes outside [0, 14]")
    return (codes - 7).to(torch.int8)


def dequantize_adanorm_w4a16(packed: PackedW4A16) -> torch.Tensor:
    """Decode the packed weights into their QDQ values in runtime channel order."""

    signed = unpack_adanorm_w4a16(packed).to(torch.float32)
    out_features, in_features = signed.shape
    scales = packed.wscales.t().reshape(out_features, in_features // packed.group_size, 1).float()
    return (signed.reshape(out_features, -1, packed.group_size) * scales).reshape_as(signed).to(packed.dtype)


def _rtn_scale_bounds(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.tensor(0, dtype=dtype, device=device)
    smallest = torch.nextafter(zero, torch.tensor(1, dtype=dtype, device=device))
    largest = torch.tensor(torch.finfo(dtype).max / 7, dtype=dtype, device=device)
    while not bool(torch.isfinite(largest * -7)):
        largest = torch.nextafter(largest, zero)
    return smallest, largest


def quantize_adanorm_w4a16_rtn(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    splits: int = 3,
    group_size: int = 64,
    chunk_rows: int | None = 256,
) -> PackedW4A16:
    """Compute symmetric per-row-group RTN scales and pack the result.

    An all-zero group uses scale one. Its signed values and QDQ values remain
    exactly zero, while the emitted scale continues to satisfy the runtime's
    positive-scale contract.
    """

    _require_little_endian()
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
    rows_per_chunk = _effective_chunk_rows(chunk_rows, out_features)
    num_groups = in_features // group_size
    scales = torch.empty((out_features, num_groups), dtype=weight.dtype, device=weight.device)
    smallest_scale, largest_scale = _rtn_scale_bounds(weight.dtype, weight.device)
    for start in range(0, out_features, rows_per_chunk):
        end = min(start + rows_per_chunk, out_features)
        grouped = weight[start:end].float().reshape(end - start, num_groups, group_size)
        absmax = grouped.abs().amax(dim=-1)
        chunk_scales = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 7).to(weight.dtype)
        chunk_scales = torch.where((absmax > 0) & (chunk_scales <= 0), smallest_scale, chunk_scales)
        scales[start:end] = torch.minimum(chunk_scales, largest_scale)
    try:
        return pack_adanorm_w4a16(
            weight,
            scales,
            bias=bias,
            splits=splits,
            group_size=group_size,
            chunk_rows=chunk_rows,
        )
    except ValueError as exc:
        if "quantized weight must be in [-7, 7]" in str(exc):
            raise ValueError("RTN scale cannot represent the weight in signed range [-7, 7]") from exc
        raise
