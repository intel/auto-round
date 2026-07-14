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

"""Logical MXFP4 codecs and Nunchaku-compatible residual packing.

E2M1 scales use an explicit rank contract: scalars are global, scales with one
fewer dimension than values are group-aligned along a new trailing singleton,
and scales with the same rank use ordinary PyTorch broadcasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from auto_round.data_type.mxfp import quant_element, quant_mx

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_SUPPORTED_DECODE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_SUPPORTED_PACKED_DTYPES = (torch.float16, torch.bfloat16)


def _validate_lowrank_weight(weight: torch.Tensor, down: bool) -> None:
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("weight must be a 2D torch.Tensor")
    if weight.dtype not in _SUPPORTED_PACKED_DTYPES:
        raise ValueError("weight dtype must be torch.float16 or torch.bfloat16")
    if not isinstance(down, bool):
        raise ValueError("down must be a bool")
    if 0 in weight.shape:
        raise ValueError("weight dimensions must be non-empty")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must contain only finite values")


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pack a logical low-rank matrix with 128-feature and 16-rank alignment."""

    _validate_lowrank_weight(weight, down)
    rows = NunchakuMXFP4Packer._ceil_to(weight.shape[0], 16 if down else 128)
    columns = NunchakuMXFP4Packer._ceil_to(weight.shape[1], 128 if down else 16)
    padded = torch.zeros((rows, columns), dtype=weight.dtype, device=weight.device)
    padded[: weight.shape[0], : weight.shape[1]] = weight
    pack_n = pack_k = 16
    if down:
        rank, channels = padded.shape
        rank_packs, channel_packs = rank // pack_n, channels // pack_k
        packed = padded.view(rank_packs, pack_n, channel_packs, pack_k).permute(2, 0, 1, 3)
    else:
        channels, rank = padded.shape
        channel_packs, rank_packs = channels // pack_n, rank // pack_k
        packed = padded.view(channel_packs, pack_n, rank_packs, pack_k).permute(0, 2, 1, 3)
    packed = packed.reshape(channel_packs, rank_packs, 2, 8, 1, 2, 4, 2)
    return packed.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous().view(channels, rank)


def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Invert :func:`pack_lowrank_weight`, retaining its padded logical shape."""

    _validate_lowrank_weight(weight, down)
    channels, rank = weight.shape
    if channels % 128 or rank % 16:
        raise ValueError("packed weight feature and rank dimensions must be divisible by 128 and 16 respectively")
    if down:
        rank_packs, channel_packs = rank // 16, channels // 16
    else:
        channel_packs, rank_packs = channels // 16, rank // 16
    unpacked = weight.view(channel_packs, rank_packs, 8, 4, 2, 2, 1, 2)
    unpacked = unpacked.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous().view(channel_packs, rank_packs, 16, 16)
    if down:
        return unpacked.permute(1, 2, 0, 3).contiguous().view(rank, channels)
    return unpacked.permute(0, 2, 1, 3).contiguous().view(channels, rank)


@dataclass(frozen=True)
class PackedMXFP4:
    """Physical MXFP4 residual tensors and their logical/padded dimensions."""

    qweight: torch.Tensor
    wscales: torch.Tensor
    logical_shape: tuple[int, int]
    padded_shape: tuple[int, int]


class NunchakuMXFP4Packer:
    """Pack MXFP4 residuals into the Nunchaku 4-bit MMA memory layout.

    The reshape/permutation constants follow the Nunchaku weight and MXFP4
    micro-scale packers at reference source commit 0abaaf0.
    """

    comp_n = 16
    comp_k = mem_k = 64
    num_n_lanes = 8
    num_k_lanes = 4
    n_pack_size = 2
    k_pack_size = 2
    reg_n = 1
    reg_k = 8
    num_k_unrolls = 2

    def __init__(self, warp_n: int = 128) -> None:
        if warp_n != 128:
            raise ValueError("warp_n must be 128")
        self.warp_n = warp_n
        self.mem_n = warp_n

    @staticmethod
    def _ceil_to(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor * divisor

    def _pack_weight_codes(self, codes: torch.Tensor) -> torch.Tensor:
        n, k = codes.shape
        weight = codes.to(torch.int32).reshape(
            n // self.mem_n,
            self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n),
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k // self.mem_k,
            1,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )
        weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=codes.device)
        packed = ((weight & 0xF) << shifts).sum(dim=-1, dtype=torch.int32)
        return packed.view(torch.int8).view(n, k // 2)

    def _pack_scale_codes(self, scales: torch.Tensor) -> torch.Tensor:
        n, num_groups = scales.shape
        scale = scales.view(n // self.warp_n, 1, 4, 4, 8, num_groups // 2, 2)
        return scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous().view(num_groups, n)

    def _unpack_weight_codes(self, qweight: torch.Tensor) -> torch.Tensor:
        n, packed_k = qweight.shape
        k = packed_k * 2
        packed = (
            qweight.contiguous()
            .view(torch.int32)
            .reshape(
                n // self.mem_n,
                k // self.mem_k,
                1,
                self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n),
                self.num_n_lanes,
                self.num_k_lanes,
                self.n_pack_size,
                self.k_pack_size,
                self.reg_n,
            )
        )
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
        weight = (packed.unsqueeze(-1) >> shifts) & 0xF
        return weight.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous().view(n, k).to(torch.uint8)

    def _unpack_scale_codes(self, wscales: torch.Tensor) -> torch.Tensor:
        num_groups, n = wscales.shape
        scale = wscales.reshape(n // self.warp_n, num_groups // 2, 1, 8, 4, 4, 2)
        return scale.permute(0, 2, 5, 4, 3, 1, 6).contiguous().view(n, num_groups)

    def pack_residual(self, weight: torch.Tensor, group_size: int = 32) -> PackedMXFP4:
        """Quantize and physically pack a logical ``[N, K]`` residual."""

        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError("weight must be a 2D floating-point torch.Tensor")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("weight must contain only finite values")
        if group_size != 32:
            raise ValueError("group_size must be 32")

        n, k = weight.shape
        if n == 0 or k == 0:
            raise ValueError("weight dimensions must be non-empty")
        n_padded = self._ceil_to(n, self.mem_n)
        k_group_padded = self._ceil_to(k, group_size)
        k_padded = self._ceil_to(k_group_padded, self.mem_k * self.num_k_unrolls)

        qdq, shared_exponent, _ = quant_mx(weight, bits=4, group_size=group_size, data_type="mx_fp4e2m1")
        num_logical_groups = k_group_padded // group_size
        scales = torch.exp2(shared_exponent.reshape(n, num_logical_groups).to(torch.float32))

        grouped_qdq = torch.zeros((n, k_group_padded), dtype=qdq.dtype, device=qdq.device)
        grouped_qdq[:, :k] = qdq
        logical_codes = encode_e2m1(grouped_qdq.reshape(n, num_logical_groups, group_size), scales)

        padded_codes = torch.zeros((n_padded, k_padded), dtype=torch.uint8, device=weight.device)
        padded_codes[:n, :k_group_padded] = logical_codes.reshape(n, k_group_padded)
        padded_scales = torch.full((n_padded, k_padded // group_size), 127, dtype=torch.uint8, device=weight.device)
        padded_scales[:n, :num_logical_groups] = encode_ue8m0(scales)

        return PackedMXFP4(
            qweight=self._pack_weight_codes(padded_codes),
            wscales=self._pack_scale_codes(padded_scales),
            logical_shape=(n, k),
            padded_shape=(n_padded, k_padded),
        )

    def unpack_residual(
        self,
        qweight: torch.Tensor,
        wscales: torch.Tensor,
        logical_shape: tuple[int, int],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Inverse the physical layout and dequantize the logical residual."""

        if not isinstance(qweight, torch.Tensor) or qweight.dtype != torch.int8:
            raise ValueError("qweight must be a torch.int8 tensor")
        if qweight.ndim != 2 or qweight.shape[0] == 0 or qweight.shape[0] % self.mem_n:
            raise ValueError("qweight shape must be [Npad, Kpad/2] with Npad divisible by 128")
        if qweight.shape[1] == 0 or (qweight.shape[1] * 2) % (self.mem_k * self.num_k_unrolls):
            raise ValueError("qweight shape must have Kpad divisible by 128")
        n_padded, packed_k = qweight.shape
        k_padded = packed_k * 2
        expected_scale_shape = (k_padded // 32, n_padded)
        if not isinstance(wscales, torch.Tensor) or wscales.dtype != torch.uint8:
            raise ValueError("wscales must be a torch.uint8 tensor")
        if tuple(wscales.shape) != expected_scale_shape:
            raise ValueError(f"wscales shape must be {expected_scale_shape}")
        if wscales.device != qweight.device:
            raise ValueError("qweight and wscales must be on the same device")
        if (
            not isinstance(logical_shape, tuple)
            or len(logical_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in logical_shape)
            or any(value <= 0 for value in logical_shape)
            or logical_shape[0] > n_padded
            or logical_shape[1] > k_padded
        ):
            raise ValueError("logical_shape must be a positive (N, K) tuple within the padded shape")
        if dtype not in _SUPPORTED_DECODE_DTYPES:
            raise ValueError("dtype must be one of torch.float16, torch.bfloat16, torch.float32, or torch.float64")
        weight_codes = self._unpack_weight_codes(qweight)
        scale_codes = self._unpack_scale_codes(wscales)
        scales = decode_ue8m0(scale_codes)
        dequantized = decode_e2m1(weight_codes.reshape(n_padded, k_padded // 32, 32), scales, dtype=dtype).reshape(
            n_padded, k_padded
        )
        n, k = logical_shape
        return dequantized[:n, :k].contiguous()


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
