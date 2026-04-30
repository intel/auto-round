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

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from math import prod, sqrt
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class TurboQuantConfig:
    bits: int = 4
    seed: int = 42
    codebook_samples: int = 65536
    codebook_iters: int = 24
    eps: float = 1e-6


@dataclass(frozen=True)
class QJLResidualConfig:
    """1-bit QJL residual correction for unbiased inner products (paper §4).

    Stores sign(residual @ S^T) as ±1 int8 + residual norm.
    Reconstruction: sqrt(π/2) / head_dim * r_norm * (signs @ S).
    """

    enabled: bool = False
    seed: int = 1729


@dataclass
class TurboQuantState:
    head_dim: int
    bits: int
    seed: int
    rotation: torch.Tensor  # (head_dim, head_dim)
    inverse_rotation: torch.Tensor  # (head_dim, head_dim)
    codebook: torch.Tensor
    boundaries: torch.Tensor
    qjl_matrix: Optional[torch.Tensor] = None  # (head_dim, head_dim) random projection


@dataclass
class TurboQuantPackedTensor:
    packed_codes: torch.Tensor
    norms: torch.Tensor  # per-vector L2 norms
    original_shape: tuple[int, ...]
    bits: int
    qjl_packed_signs: Optional[torch.Tensor] = None  # uint8, bit-packed (1 bit/sign)
    qjl_norms: Optional[torch.Tensor] = None  # float16, residual norms, shape = original_shape[:-1]

    def memory_bytes(self) -> int:
        size = self.packed_codes.numel() * self.packed_codes.element_size()
        size += self.norms.numel() * self.norms.element_size()
        if self.qjl_packed_signs is not None:
            size += self.qjl_packed_signs.numel() * self.qjl_packed_signs.element_size()
        if self.qjl_norms is not None:
            size += self.qjl_norms.numel() * self.qjl_norms.element_size()
        return size


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@lru_cache(maxsize=None)
def _lloyd_max_codebook(bits: int, samples: int, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bits < 1 or bits > 4:
        raise ValueError(f"TurboQuant only supports 1-4 bits in the codebook solver, but got {bits}.")

    levels = 1 << bits
    generator = _make_generator(1000 + bits)
    data = torch.randn(samples, generator=generator, dtype=torch.float32)
    centroids = torch.linspace(-2.5, 2.5, levels, dtype=torch.float32)

    for _ in range(iters):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        bucket_ids = torch.bucketize(data, boundaries)
        new_centroids = []
        for idx in range(levels):
            mask = bucket_ids == idx
            if mask.any():
                new_centroids.append(data[mask].mean())
            else:
                new_centroids.append(centroids[idx])
        centroids = torch.stack(new_centroids)

    if levels % 2 == 0:
        half = levels // 2
        positive = 0.5 * (centroids[half:] - centroids[:half].flip(0))
        positive = positive.abs()
        centroids = torch.cat((-positive.flip(0), positive))

    centroids = torch.sort(centroids).values
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    return centroids, boundaries


@lru_cache(maxsize=None)
def _rotation_matrix(head_dim: int, seed: int) -> torch.Tensor:
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, but got {head_dim}.")

    generator = _make_generator(seed)
    gaussian = torch.randn((head_dim, head_dim), generator=generator, dtype=torch.float32)
    q_mat, r_mat = torch.linalg.qr(gaussian)
    diag = torch.sign(torch.diag(r_mat))
    diag[diag == 0] = 1
    q_mat = q_mat * diag.unsqueeze(0)
    return q_mat.contiguous()


@lru_cache(maxsize=None)
def _qjl_random_matrix(head_dim: int, seed: int) -> torch.Tensor:
    """Generate a random Gaussian matrix S for QJL 1-bit projection.

    S is (head_dim, head_dim). The QJL transform stores sign(residual @ S^T).
    """
    generator = _make_generator(seed)
    return torch.randn((head_dim, head_dim), generator=generator, dtype=torch.float32)


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], but got {bits}.")

    codes_np = codes.reshape(-1).to(torch.uint8).cpu().numpy().astype(np.uint64)
    n = len(codes_np)
    total_bits = n * bits
    n_bytes = (total_bits + 7) // 8
    result = np.zeros(n_bytes, dtype=np.uint64)

    positions = np.arange(n, dtype=np.uint64) * bits
    for b in range(bits):
        bit_pos = positions + b
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        bit_val = (codes_np >> b) & 1
        np.add.at(result, byte_idx, bit_val << bit_off)

    return torch.tensor(result.astype(np.uint8), dtype=torch.uint8, device=codes.device)


def _unpack_codes(packed_codes: torch.Tensor, num_values: int, bits: int, device: torch.device) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], but got {bits}.")

    packed_np = packed_codes.cpu().numpy().astype(np.uint64)
    positions = np.arange(num_values, dtype=np.uint64) * bits
    result = np.zeros(num_values, dtype=np.int64)

    for b in range(bits):
        bit_pos = positions + b
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        bit_val = (packed_np[byte_idx] >> bit_off) & 1
        result |= bit_val.astype(np.int64) << b

    return torch.tensor(result, dtype=torch.long, device=device)


def build_turboquant_state(
    head_dim: int,
    bits: int,
    seed: int,
    device: torch.device,
    qjl_config: Optional[QJLResidualConfig] = None,
) -> TurboQuantState:
    if bits not in (2, 3, 4):
        raise ValueError(f"TurboQuant only supports 2/3/4-bit KV cache quantization, but got {bits}.")

    rotation = _rotation_matrix(head_dim, seed).to(device=device)
    codebook, boundaries = _lloyd_max_codebook(bits, 65536, 24)

    qjl_matrix = None
    if qjl_config is not None and qjl_config.enabled:
        qjl_matrix = _qjl_random_matrix(head_dim, qjl_config.seed).to(device=device)

    return TurboQuantState(
        head_dim=head_dim,
        bits=bits,
        seed=seed,
        rotation=rotation,
        inverse_rotation=rotation.transpose(0, 1).contiguous(),
        codebook=codebook.to(device=device),
        boundaries=boundaries.to(device=device),
        qjl_matrix=qjl_matrix,
    )


def turboquant_pack(
    tensor: torch.Tensor,
    state: TurboQuantState,
    eps: float = 1e-6,
    residual_config: Optional[QJLResidualConfig] = None,
) -> TurboQuantPackedTensor:
    if tensor.numel() == 0:
        return TurboQuantPackedTensor(
            packed_codes=torch.empty(0, dtype=torch.uint8, device=tensor.device),
            norms=torch.empty(0, dtype=tensor.dtype, device=tensor.device),
            original_shape=tuple(tensor.shape),
            bits=state.bits,
        )

    use_triton_bitpack = _HAS_TRITON and tensor.is_cuda

    # Encode: normalize → rotate → scale → bucketize  (cuBLAS matmul is faster than Triton loop)
    work_tensor = tensor.to(torch.float32)
    norms = torch.linalg.vector_norm(work_tensor, dim=-1, keepdim=True).clamp_min(eps)
    normalized = work_tensor / norms
    rotated = torch.matmul(normalized, state.rotation)
    scale = sqrt(state.head_dim)
    scaled = rotated * scale
    bucket_ids = torch.bucketize(scaled.reshape(-1), state.boundaries)

    # Bit-pack: Triton on CUDA, numpy on CPU
    if use_triton_bitpack:
        packed_codes = _triton_pack(bucket_ids, state.bits)
    else:
        packed_codes = _pack_codes(bucket_ids, state.bits)

    qjl_packed_signs = None
    qjl_norms = None
    if residual_config is not None and residual_config.enabled and state.qjl_matrix is not None:
        quantized = state.codebook[bucket_ids].view(tensor.shape) / scale
        reconstructed = torch.matmul(quantized, state.inverse_rotation)
        residual = normalized - reconstructed
        r_norm = torch.linalg.vector_norm(residual, dim=-1)
        projected = torch.matmul(residual, state.qjl_matrix.T)
        sign_bits = (projected >= 0).to(torch.uint8)  # 0/1
        # Bit-pack signs: head_dim bits → head_dim/8 bytes per vector
        if use_triton_bitpack:
            qjl_packed_signs = _triton_pack(sign_bits.reshape(-1), 1)
        else:
            qjl_packed_signs = _pack_codes(sign_bits.reshape(-1), 1)
        qjl_norms = r_norm.to(torch.float16)

    return TurboQuantPackedTensor(
        packed_codes=packed_codes,
        norms=norms.to(tensor.dtype),
        original_shape=tuple(tensor.shape),
        bits=state.bits,
        qjl_packed_signs=qjl_packed_signs,
        qjl_norms=qjl_norms,
    )


def turboquant_unpack(
    packed: TurboQuantPackedTensor,
    state: TurboQuantState,
    dtype: torch.dtype = torch.float32,
    residual_config: Optional[QJLResidualConfig] = None,
) -> torch.Tensor:
    if len(packed.original_shape) == 0 or prod(packed.original_shape) == 0:
        return torch.empty(packed.original_shape, dtype=dtype, device=state.rotation.device)

    num_values = prod(packed.original_shape)
    use_triton_bitpack = _HAS_TRITON and state.rotation.is_cuda

    # Bit-unpack: Triton on CUDA, numpy on CPU
    if use_triton_bitpack:
        bucket_ids = _triton_unpack(packed.packed_codes, num_values, packed.bits)
    else:
        bucket_ids = _unpack_codes(packed.packed_codes, num_values, packed.bits, state.rotation.device)

    # Decode: codebook gather → inverse-rotate → scale  (cuBLAS matmul always)
    scale = sqrt(state.head_dim)
    quantized = state.codebook[bucket_ids].view(packed.original_shape) / scale
    reconstructed = torch.matmul(quantized, state.inverse_rotation)

    if packed.qjl_packed_signs is not None and packed.qjl_norms is not None:
        if residual_config is None or not residual_config.enabled or state.qjl_matrix is None:
            raise ValueError("QJL signs are present, but residual_config is missing/disabled or qjl_matrix is None.")
        # Unpack 1-bit signs → ±1 float
        num_sign_values = prod(packed.original_shape)
        if use_triton_bitpack:
            sign_bits = _triton_unpack(packed.qjl_packed_signs, num_sign_values, 1)
        else:
            sign_bits = _unpack_codes(packed.qjl_packed_signs, num_sign_values, 1, state.rotation.device)
        signs_f = (sign_bits.to(torch.float32) * 2 - 1).view(packed.original_shape)
        d = state.head_dim
        qjl_scale = sqrt(math.pi / 2.0) / d
        r_norms = packed.qjl_norms.to(torch.float32).unsqueeze(-1)
        qjl_correction = qjl_scale * r_norms * torch.matmul(signs_f, state.qjl_matrix)
        reconstructed = reconstructed + qjl_correction

    return (reconstructed * packed.norms.to(torch.float32)).to(dtype)


def turboquant_qdq(
    tensor: torch.Tensor,
    state: TurboQuantState,
    eps: float = 1e-6,
    residual_config: Optional[QJLResidualConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.numel() == 0:
        return tensor, torch.zeros(1, device=tensor.device, dtype=tensor.dtype)

    packed = turboquant_pack(tensor, state, eps=eps, residual_config=residual_config)
    reconstructed = turboquant_unpack(packed, state, dtype=tensor.dtype, residual_config=residual_config)
    avg_norm = packed.norms.mean().to(tensor.dtype).reshape(1)
    return reconstructed, avg_norm


# ---------------------------------------------------------------------------
# Triton-accelerated encode/decode (optional, CUDA only)
# ---------------------------------------------------------------------------

_HAS_TRITON = False
try:
    from auto_round_extension.triton.turboquant import triton_pack_codes as _triton_pack
    from auto_round_extension.triton.turboquant import triton_unpack_codes as _triton_unpack
    from auto_round_extension.triton.turboquant import turboquant_decode as _triton_decode
    from auto_round_extension.triton.turboquant import turboquant_encode as _triton_encode

    _HAS_TRITON = True
except ImportError:
    pass


def has_triton_turboquant() -> bool:
    """Check if Triton TurboQuant kernels are available."""
    return _HAS_TRITON and torch.cuda.is_available()


def turboquant_qdq_triton(
    tensor: torch.Tensor,
    state: TurboQuantState,
) -> torch.Tensor:
    """Fast quantize→dequantize using cuBLAS matmul + Triton bitpack (no QJL).

    Input: (..., head_dim) on CUDA.
    Output: same shape, same dtype.
    """
    if not has_triton_turboquant():
        raise RuntimeError("Triton TurboQuant kernels not available.")

    packed = turboquant_pack(tensor, state)
    return turboquant_unpack(packed, state, dtype=tensor.dtype)
