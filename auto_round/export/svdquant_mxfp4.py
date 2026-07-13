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

"""Logical MXFP4 E2M1 and UE8M0 reference codecs.

This module describes logical values only. It intentionally does not implement
the tile reordering required by Nunchaku kernels.

E2M1 scales use an explicit rank contract: scalars are global, scales with one
fewer dimension than values are group-aligned along a new trailing singleton,
and scales with the same rank use ordinary PyTorch broadcasting.
"""

from __future__ import annotations

import torch

from auto_round.data_type.mxfp import quant_element


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_SUPPORTED_DECODE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _broadcast_scales(scales: torch.Tensor, shape: torch.Size, *, device: torch.device) -> torch.Tensor:
    """Apply the codec scale layout contract.

    A scalar scale is global. A scale tensor with one fewer dimension than the
    values is group-aligned by appending a trailing singleton. A scale tensor
    with equal rank uses ordinary PyTorch broadcasting. Other ranks are invalid.
    """

    if not isinstance(scales, torch.Tensor):
        raise ValueError("scales must be a torch.Tensor")
    if not scales.is_floating_point():
        raise ValueError("scales must have a floating-point dtype")
    if scales.device != device:
        raise ValueError(f"scales must be on device {device}, got {scales.device}")
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("scales must contain only positive finite values")
    tensor_ndim = len(shape)
    if scales.ndim == 0:
        aligned_scales = scales
        layout = "global"
    elif scales.ndim == tensor_ndim - 1:
        aligned_scales = scales.unsqueeze(-1)
        layout = "group-aligned"
    elif scales.ndim == tensor_ndim:
        aligned_scales = scales
        layout = "ordinary-broadcast"
    else:
        raise ValueError(
            f"scales rank {scales.ndim} is invalid for tensor rank {tensor_ndim}; expected a scalar, "
            f"rank {tensor_ndim - 1} for group alignment, or rank {tensor_ndim} for ordinary broadcasting"
        )
    try:
        return torch.broadcast_to(aligned_scales, shape)
    except RuntimeError as exc:
        raise ValueError(
            f"scales shape {tuple(scales.shape)} with {layout} layout cannot broadcast to tensor shape {tuple(shape)}"
        ) from exc


def _validate_codes(codes: torch.Tensor, *, maximum: int) -> None:
    if not isinstance(codes, torch.Tensor):
        raise ValueError("codes must be a torch.Tensor")
    if codes.dtype == torch.bool or codes.is_floating_point() or codes.is_complex():
        raise ValueError("codes must have an integer dtype")
    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) > maximum):
        raise ValueError(f"codes must be in the range 0..{maximum}")


def encode_e2m1(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Encode raw values as logical E2M1 codes after dividing by ``scales``."""

    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise ValueError("values must be a floating-point torch.Tensor")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must contain only finite values")
    expanded_scales = _broadcast_scales(scales, values.shape, device=values.device)
    normalized = values.to(torch.float32) / expanded_scales.to(torch.float32)
    invalid = torch.isnan(normalized)
    if bool(invalid.any()):
        fallback = (values.to(torch.float64) / expanded_scales.to(torch.float64)).clamp(min=-6.0, max=6.0)
        normalized = torch.where(invalid, fallback.to(torch.float32), normalized)
    normalized = normalized.clamp(min=-6.0, max=6.0)
    quantized = quant_element(normalized, ebits=2, mbits=3, max_norm=6.0)

    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=quantized.dtype, device=quantized.device)
    magnitude_codes = torch.searchsorted(magnitudes, torch.abs(quantized)).to(torch.uint8)
    sign_codes = torch.signbit(normalized).to(torch.uint8) << 3
    return magnitude_codes | sign_codes


def decode_e2m1(codes: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode logical E2M1 codes and multiply by ``scales``."""

    _validate_codes(codes, maximum=15)
    if dtype not in _SUPPORTED_DECODE_DTYPES:
        raise ValueError("dtype must be one of torch.float16, torch.bfloat16, torch.float32, or torch.float64")
    expanded_scales = _broadcast_scales(scales, codes.shape, device=codes.device)
    codebook = torch.tensor(
        (*_E2M1_MAGNITUDES, *(-value for value in _E2M1_MAGNITUDES)),
        dtype=dtype,
        device=codes.device,
    )
    return codebook[codes.to(torch.long)] * expanded_scales.to(dtype)


def encode_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    """Encode scales using an unsigned exponent with bias 127.

    Non-finite and nonpositive scales use code 127, which decodes to 1.0.
    """

    if not isinstance(scales, torch.Tensor) or not scales.is_floating_point():
        raise ValueError("scales must be a floating-point torch.Tensor")
    valid = torch.isfinite(scales) & (scales > 0)
    safe_scales = torch.where(valid, scales, torch.ones_like(scales))
    exponents = torch.ceil(torch.log2(safe_scales)).clamp(min=-127, max=127)
    codes = (exponents + 127).to(torch.uint8)
    return torch.where(valid, codes, torch.full_like(codes, 127))


def decode_ue8m0(codes: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 codes to float32 powers of two.

    Code 255 remains decodable for Nunchaku runtime parity, but
    :func:`encode_ue8m0` clamps valid exponents to 127 and never emits it.
    """

    _validate_codes(codes, maximum=255)
    exponents = codes.to(torch.int16) - 127
    return torch.exp2(exponents.to(torch.float32))


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical codes along the last dimension, low nibble first."""

    _validate_codes(codes, maximum=15)
    if codes.ndim == 0:
        raise ValueError("codes must have at least one dimension")
    codes = codes.to(torch.uint8)
    if codes.shape[-1] % 2:
        padding = torch.zeros((*codes.shape[:-1], 1), dtype=torch.uint8, device=codes.device)
        codes = torch.cat((codes, padding), dim=-1)
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor, logical_count: int | None = None) -> torch.Tensor:
    """Unpack low-first nibbles along the last dimension."""

    _validate_codes(packed, maximum=255)
    if packed.ndim == 0:
        raise ValueError("packed must have at least one dimension")
    capacity = packed.shape[-1] * 2
    if logical_count is not None:
        if isinstance(logical_count, bool) or not isinstance(logical_count, int):
            raise ValueError("logical_count must be an integer or None")
        if logical_count < 0 or logical_count > capacity:
            raise ValueError(f"logical_count must be between 0 and {capacity}, got {logical_count}")
    packed = packed.to(torch.uint8)
    unpacked = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(*packed.shape[:-1], capacity)
    if logical_count is not None:
        unpacked = unpacked[..., :logical_count]
    return unpacked.contiguous()
