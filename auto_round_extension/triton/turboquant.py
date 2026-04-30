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

"""Triton kernels for TurboQuant KV cache encode/decode.

Encode: normalize → rotate → scale → scalar quantize → store indices + norm
Decode: load indices → codebook lookup → scale⁻¹ → unrotate → scale by norm

Each kernel processes one (token, head) pair as a single Triton program.
The rotation matrix multiply is done column-by-column inside the kernel
to keep the full head vector in SRAM.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Encode kernel
# ---------------------------------------------------------------------------


@triton.jit
def _turboquant_encode_kernel(
    # Input: [num_tokens, num_kv_heads, head_size]
    x_ptr,
    # Rotation matrix PiT: [head_size, head_size], row-major
    pit_ptr,
    # Boundaries: [num_centroids - 1]
    boundaries_ptr,
    # Output indices: [num_tokens, num_kv_heads, head_size] as uint8
    indices_ptr,
    # Output norms: [num_tokens, num_kv_heads] as float32
    norms_ptr,
    # Shapes
    head_size: tl.constexpr,
    num_boundaries: tl.constexpr,
    # Scale factor: sqrt(head_size)
    scale: tl.constexpr,
    # Strides
    x_stride_token: tl.int64,
    x_stride_head: tl.int64,
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    # Padded dim (power of 2)
    BLOCK_D: tl.constexpr,
):
    """Encode one (token, head): normalize → rotate → quantize."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Load input vector
    x_base = token_idx * x_stride_token + head_idx * x_stride_head
    x_vec = tl.load(x_ptr + x_base + dim_offs, mask=mask, other=0.0).to(tl.float32)

    # L2 norm
    norm_sq = tl.sum(x_vec * x_vec, axis=0)
    norm = tl.sqrt(norm_sq + 1e-12)
    x_normed = x_vec / norm

    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head

    # For each output dim j: compute rotated[j] = dot(x_normed, PiT[:, j])
    for j in range(head_size):
        pit_col = tl.load(pit_ptr + dim_offs * head_size + j, mask=mask, other=0.0)
        y_j = tl.sum(x_normed * pit_col, axis=0) * scale

        # Scalar quantize: count how many boundaries y_j exceeds
        idx = tl.zeros([], dtype=tl.int32)
        for b in range(num_boundaries):
            bnd = tl.load(boundaries_ptr + b)
            idx = idx + (y_j > bnd).to(tl.int32)

        tl.store(indices_ptr + idx_base + j, idx.to(tl.uint8))

    # Store norm
    tl.store(norms_ptr + token_idx * norm_stride_token + head_idx, norm)


# ---------------------------------------------------------------------------
# Decode kernel
# ---------------------------------------------------------------------------


@triton.jit
def _turboquant_decode_kernel(
    # Input indices: [num_tokens, num_kv_heads, head_size] as uint8
    indices_ptr,
    # Input norms: [num_tokens, num_kv_heads] as float32
    norms_ptr,
    # Rotation matrix Pi: [head_size, head_size], row-major
    pi_ptr,
    # Codebook: [num_centroids]
    codebook_ptr,
    # Output: [num_tokens, num_kv_heads, head_size]
    out_ptr,
    # Shapes
    head_size: tl.constexpr,
    # Scale factor: sqrt(head_size)
    scale: tl.constexpr,
    # Strides
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    # Padded dim
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Decode one (token, head): codebook lookup → unrotate → scale."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Load indices and codebook lookup
    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head
    indices = tl.load(indices_ptr + idx_base + dim_offs, mask=mask, other=0).to(tl.int32)
    # Codebook gather, then divide by scale (undo the sqrt(d) scaling)
    reconstructed = tl.load(codebook_ptr + indices) / scale
    reconstructed = tl.where(mask, reconstructed, 0.0)

    # Load norm
    norm = tl.load(norms_ptr + token_idx * norm_stride_token + head_idx).to(tl.float32)

    # Unrotate: out[j] = sum_i(reconstructed[i] * Pi[i, j]) * norm
    out_base = token_idx * out_stride_token + head_idx * out_stride_head

    for j in range(head_size):
        pi_col = tl.load(pi_ptr + dim_offs * head_size + j, mask=mask, other=0.0)
        val = tl.sum(reconstructed * pi_col, axis=0) * norm

        if OUTPUT_BF16:
            tl.store(out_ptr + out_base + j, val.to(tl.bfloat16))
        else:
            tl.store(out_ptr + out_base + j, val.to(tl.float16))


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def turboquant_encode(
    x: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    pit: torch.Tensor,  # [head_size, head_size] rotation^T
    codebook: torch.Tensor,  # [num_centroids]
    boundaries: torch.Tensor,  # [num_centroids - 1]
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode K or V vectors using TurboQuant.

    Returns:
        indices: [num_tokens, num_kv_heads, head_size] uint8
        norms: [num_tokens, num_kv_heads] float32
    """
    num_tokens, num_kv_heads, head_size = x.shape
    if head_dim is None:
        head_dim = head_size
    num_boundaries = boundaries.shape[0]
    BLOCK_D = _next_power_of_2(head_size)
    scale = math.sqrt(head_dim)

    indices = torch.empty(
        (num_tokens, num_kv_heads, head_size),
        dtype=torch.uint8,
        device=x.device,
    )
    norms = torch.empty(
        (num_tokens, num_kv_heads),
        dtype=torch.float32,
        device=x.device,
    )

    grid = (num_tokens, num_kv_heads)
    _turboquant_encode_kernel[grid](
        x_ptr=x,
        pit_ptr=pit,
        boundaries_ptr=boundaries,
        indices_ptr=indices,
        norms_ptr=norms,
        head_size=head_size,
        num_boundaries=num_boundaries,
        scale=scale,
        x_stride_token=x.stride(0),
        x_stride_head=x.stride(1),
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=1,
    )

    return indices, norms


def turboquant_decode(
    indices: torch.Tensor,  # [num_tokens, num_kv_heads, head_size] uint8
    norms: torch.Tensor,  # [num_tokens, num_kv_heads] float32
    pi: torch.Tensor,  # [head_size, head_size] rotation matrix
    codebook: torch.Tensor,  # [num_centroids]
    head_dim: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode TurboQuant indices back to K or V vectors.

    Returns:
        out: [num_tokens, num_kv_heads, head_size] in output_dtype
    """
    num_tokens, num_kv_heads, head_size = indices.shape
    if head_dim is None:
        head_dim = head_size
    BLOCK_D = _next_power_of_2(head_size)
    scale = math.sqrt(head_dim)

    out = torch.empty(
        (num_tokens, num_kv_heads, head_size),
        dtype=output_dtype,
        device=indices.device,
    )

    grid = (num_tokens, num_kv_heads)
    _turboquant_decode_kernel[grid](
        indices_ptr=indices,
        norms_ptr=norms,
        pi_ptr=pi,
        codebook_ptr=codebook,
        out_ptr=out,
        head_size=head_size,
        scale=scale,
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        out_stride_token=out.stride(0),
        out_stride_head=out.stride(1),
        BLOCK_D=BLOCK_D,
        OUTPUT_BF16=(output_dtype == torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )

    return out


# ---------------------------------------------------------------------------
# Bit-pack / unpack kernels
# ---------------------------------------------------------------------------


@triton.jit
def _bitpack_kernel(
    # Input: [N] uint8 codes (each in [0, 2^bits - 1])
    codes_ptr,
    # Output: [n_bytes] uint8 packed
    packed_ptr,
    N,
    n_bytes,
    bits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack N code values into bit-packed bytes. Each program handles BLOCK output bytes."""
    pid = tl.program_id(0)
    byte_offs = pid * BLOCK + tl.arange(0, BLOCK)
    byte_mask = byte_offs < n_bytes

    # Each output byte covers bit range [byte_offs*8, byte_offs*8 + 8).
    # Build the byte bit-by-bit: for each of the 8 bit positions, find
    # which code and which bit within that code contributes.
    bit_start = byte_offs * 8  # vector[BLOCK]
    result = tl.zeros([BLOCK], dtype=tl.int32)

    for bit_in_byte in range(8):
        global_bit = bit_start + bit_in_byte
        code_idx = global_bit // bits
        bit_in_code = global_bit - code_idx * bits  # = global_bit % bits

        valid = (code_idx < N) & byte_mask
        code_val = tl.load(codes_ptr + code_idx, mask=valid, other=0).to(tl.int32)
        bit_val = (code_val >> bit_in_code) & 1
        result = result | (bit_val << bit_in_byte)

    tl.store(packed_ptr + byte_offs, result.to(tl.uint8), mask=byte_mask)


@triton.jit
def _bitunpack_kernel(
    # Input: [n_bytes] uint8 packed
    packed_ptr,
    # Output: [N] int64 codes
    codes_ptr,
    N,
    n_bytes,
    bits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Unpack N code values from bit-packed bytes. Each program handles BLOCK codes."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Each code spans [offs*bits, offs*bits + bits) in the bit stream,
    # crossing at most 2 bytes.  Load both and extract.
    bit_pos = offs * bits
    byte_lo = bit_pos >> 3  # first byte index
    bit_off = bit_pos & 7  # bit offset within that byte

    lo = tl.load(packed_ptr + byte_lo, mask=mask, other=0).to(tl.int32)
    hi_mask = mask & ((byte_lo + 1) < n_bytes)
    hi = tl.load(packed_ptr + byte_lo + 1, mask=hi_mask, other=0).to(tl.int32)
    combined = lo | (hi << 8)  # 16 bits is always enough for bits <= 8

    code_mask = (1 << bits) - 1
    codes = (combined >> bit_off) & code_mask

    tl.store(codes_ptr + offs, codes.to(tl.int64), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers for bit-pack / unpack
# ---------------------------------------------------------------------------


def triton_pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack a flat uint8/long tensor of codes on GPU.

    Args:
        codes: 1-D tensor of quantization indices on CUDA.
        bits: bits per code (2, 3, or 4).

    Returns:
        Packed uint8 tensor on the same device.
    """
    codes_flat = codes.reshape(-1).to(torch.uint8).contiguous()
    N = codes_flat.shape[0]
    n_bytes = (N * bits + 7) // 8
    packed = torch.empty(n_bytes, dtype=torch.uint8, device=codes.device)

    BLOCK = 1024
    grid = ((n_bytes + BLOCK - 1) // BLOCK,)
    _bitpack_kernel[grid](
        codes_ptr=codes_flat,
        packed_ptr=packed,
        N=N,
        n_bytes=n_bytes,
        bits=bits,
        BLOCK=BLOCK,
    )
    return packed


def triton_unpack_codes(packed: torch.Tensor, num_values: int, bits: int) -> torch.Tensor:
    """Unpack bit-packed bytes into a flat int64 tensor of codes on GPU.

    Args:
        packed: 1-D uint8 tensor of packed bytes on CUDA.
        num_values: number of code values to extract.
        bits: bits per code (2, 3, or 4).

    Returns:
        1-D int64 tensor of codes on the same device.
    """
    codes = torch.empty(num_values, dtype=torch.int64, device=packed.device)

    BLOCK = 1024
    grid = ((num_values + BLOCK - 1) // BLOCK,)
    _bitunpack_kernel[grid](
        packed_ptr=packed.contiguous(),
        codes_ptr=codes,
        N=num_values,
        n_bytes=packed.numel(),
        bits=bits,
        BLOCK=BLOCK,
    )
    return codes
