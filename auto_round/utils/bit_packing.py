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

"""Generic little-endian bit-stream packing for arbitrary weight bit-widths (1-8 bits).

Rationale
---------
Both the classic GPTQ ``qweight`` layout and the AWQ ``qweight`` layout store
sub-byte integers inside ``int32`` words.  The historical AutoRound
implementations special-cased ``bits in (2, 4, 8)`` (where ``32 % bits == 0``,
so every value fits neatly inside a single word) and ``bits == 3`` (a bespoke
hand-rolled routine).  That is why 5/6/7-bit checkpoints could not be exported.

In reality *all* of those layouts are the exact same thing: a contiguous
little-endian bit-stream laid over blocks of **32 consecutive values**, which
always yields exactly ``bits`` ``int32`` words per block::

    value i (0 <= i < 32) occupies bits [i*bits, (i+1)*bits)
    word   = (i * bits) // 32
    offset = (i * bits) %  32
    # when offset + bits > 32 the value straddles ``word`` and ``word + 1``

For ``bits in (2, 4, 8)`` no value ever straddles a word boundary, so this
degenerates to the familiar ``32 // bits`` values-per-word layout.  For
``bits == 3`` it reproduces the legacy AutoRound/AutoGPTQ 3-bit routine
bit-for-bit (10 values, a straddling value, 10 values, ... per 3 words).  And
for ``bits in (5, 6, 7)`` it "just works" -- no new concept is introduced.

This is also precisely the layout consumed by the `humming
<https://github.com/inclusionAI/humming>`_ GEMM kernels
(``humming/include/humming/kernel/pack_weight.cuh::common_pack_weight``),
which support any weight type below 8 bits.  Sharing this single definition is
what lets ``auto_round:auto_gptq`` / ``auto_round:auto_awq`` checkpoints carry
5/6/7-bit weights and still be served by a fast kernel.

Only the block size (32 values) is required to divide the packed dimension;
there is no ``32 % bits == 0`` constraint.
"""

from typing import Optional

import torch

__all__ = [
    "AWQ_PACK_ORDER",
    "AWQ_REVERSE_ORDER",
    "SUPPORTED_PACKING_BITS",
    "awq_reverse_reorder",
    "awq_reorder",
    "pack_bitstream",
    "unpack_bitstream",
    "packed_dim_size",
    "requires_generic_bit_packing",
]

# Weight bit-widths that ``pack_bitstream`` / ``unpack_bitstream`` handle.
SUPPORTED_PACKING_BITS = (1, 2, 3, 4, 5, 6, 7, 8)

# Number of source values covered by one packing block. ``PACK_BLOCK * bits``
# is always a multiple of 32, so a block maps onto exactly ``bits`` int32 words.
PACK_BLOCK = 32

# AWQ shuffles values inside groups of 8 before writing them into the
# bit-stream. ``AWQ_PACK_ORDER`` maps *logical* position -> *stored* slot and
# ``AWQ_REVERSE_ORDER`` is its inverse (stored slot -> logical position).
# For 4-bit these reproduce AutoAWQ's classic ``[0, 4, 1, 5, 2, 6, 3, 7]``
# shift table exactly.
AWQ_INTERLEAVE = 8
AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)

_UINT32_MASK = 0xFFFFFFFF
_INT32_SIGN = 1 << 31
_UINT32_SPAN = 1 << 32


def requires_generic_bit_packing(bits: int) -> bool:
    """Whether ``bits`` needs the generic bit-stream path.

    ``2/4/8`` (word-aligned) and ``3`` (legacy hand-rolled) keep their original
    fast paths so that previously exported checkpoints stay byte-identical.
    """
    return int(bits) in (5, 6, 7)


def packed_dim_size(num_values: int, bits: int) -> int:
    """Number of ``int32`` words needed to store ``num_values`` values."""
    if num_values % PACK_BLOCK != 0:
        raise ValueError(
            f"cannot pack {num_values} values at {bits} bits: the packed dimension must be a multiple of "
            f"{PACK_BLOCK}. Pad the layer or leave it unquantized."
        )
    return num_values * bits // 32


def _shift_tables(bits: int, device: torch.device):
    """Per-value (word index, bit offset) tables for one 32-value block."""
    idx = torch.arange(PACK_BLOCK, device=device, dtype=torch.int64) * bits
    word = torch.div(idx, 32, rounding_mode="floor")
    offset = idx % 32
    return word, offset


def _check_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_PACKING_BITS:
        raise ValueError(f"unsupported bits={bits} for bit-stream packing, expected one of {SUPPORTED_PACKING_BITS}")
    return bits


def pack_bitstream(values: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Pack integer ``values`` into ``int32`` along ``dim`` using the bit-stream layout.

    Args:
        values: Integer tensor holding unsigned values in ``[0, 2**bits)``.
        bits: Weight bit-width, see :data:`SUPPORTED_PACKING_BITS`.
        dim: Dimension to pack. Its size must be a multiple of 32.

    Returns:
        ``int32`` tensor with ``dim`` shrunk to ``size * bits // 32``.
    """
    bits = _check_bits(bits)
    values = values.movedim(dim, -1)
    num_values = values.shape[-1]
    num_words = packed_dim_size(num_values, bits)

    lead_shape = values.shape[:-1]
    # (..., n_blocks, 32) — one block produces exactly ``bits`` words.
    values = values.reshape(*lead_shape, num_values // PACK_BLOCK, PACK_BLOCK).to(torch.int64)
    values = values & ((1 << bits) - 1)

    word, offset = _shift_tables(bits, values.device)
    packed = torch.zeros(*values.shape[:-1], bits, dtype=torch.int64, device=values.device)

    # Low part: the (possibly partial) chunk that lands in ``word``.
    # Bit ranges never overlap, so accumulating with ``index_add_`` is
    # equivalent to OR-ing but stays fully vectorized.
    packed.index_add_(-1, word, values << offset)

    straddles = (offset + bits) > 32
    if bool(straddles.any()):
        # High part: the remainder spills into the next word's low bits.
        high = torch.where(straddles, values >> (32 - offset), torch.zeros_like(values))
        packed.index_add_(-1, torch.clamp(word + 1, max=bits - 1), high)

    packed = packed & _UINT32_MASK
    # Reinterpret as signed int32 (torch has no uint32 storage).
    packed = torch.where(packed >= _INT32_SIGN, packed - _UINT32_SPAN, packed).to(torch.int32)
    packed = packed.reshape(*lead_shape, num_words)
    return packed.movedim(-1, dim)


def unpack_bitstream(packed: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`pack_bitstream`.

    Returns an ``int32`` tensor of unsigned values in ``[0, 2**bits)`` with
    ``dim`` expanded back to ``num_words * 32 // bits``.
    """
    bits = _check_bits(bits)
    packed = packed.movedim(dim, -1)
    num_words = packed.shape[-1]
    if num_words % bits != 0:
        raise ValueError(f"packed dimension {num_words} is not a multiple of bits={bits}")

    lead_shape = packed.shape[:-1]
    packed = packed.reshape(*lead_shape, num_words // bits, bits).to(torch.int64) & _UINT32_MASK

    word, offset = _shift_tables(bits, packed.device)
    one = torch.ones((), dtype=torch.int64, device=packed.device)

    low_bits = torch.clamp(torch.full_like(offset, 32) - offset, max=bits)
    low = (packed.index_select(-1, word) >> offset) & (torch.bitwise_left_shift(one, low_bits) - 1)

    high_bits = torch.full_like(low_bits, bits) - low_bits
    if bool(torch.any(high_bits > 0)):
        next_word = packed.index_select(-1, torch.clamp(word + 1, max=bits - 1))
        high = torch.bitwise_left_shift(next_word & (torch.bitwise_left_shift(one, high_bits) - 1), low_bits)
        low = low | torch.where(high_bits > 0, high, torch.zeros_like(high))

    values = low.reshape(*lead_shape, num_words * 32 // bits).to(torch.int32)
    return values.movedim(-1, dim)


def _apply_awq_order(tensor: torch.Tensor, order, dim: int) -> torch.Tensor:
    tensor = tensor.movedim(dim, -1)
    size = tensor.shape[-1]
    if size % AWQ_INTERLEAVE != 0:
        raise ValueError(
            f"AWQ interleaving requires the packed dimension ({size}) to be a multiple of {AWQ_INTERLEAVE}"
        )
    index = torch.as_tensor(order, dtype=torch.long, device=tensor.device)
    tensor = tensor.reshape(*tensor.shape[:-1], size // AWQ_INTERLEAVE, AWQ_INTERLEAVE)
    tensor = tensor.index_select(-1, index)
    tensor = tensor.reshape(*tensor.shape[:-2], size)
    return tensor.movedim(-1, dim)


def awq_reorder(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shuffle logical values into AWQ storage order (apply before packing)."""
    return _apply_awq_order(values, AWQ_PACK_ORDER, dim)


def awq_reverse_reorder(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Undo :func:`awq_reorder` (apply after unpacking)."""
    return _apply_awq_order(values, AWQ_REVERSE_ORDER, dim)


def pack_awq_bitstream(values: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """AWQ-flavoured packing: interleave groups of 8, then bit-stream pack."""
    return pack_bitstream(awq_reorder(values, dim=dim), bits, dim=dim)


def unpack_awq_bitstream(packed: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`pack_awq_bitstream`."""
    return awq_reverse_reorder(unpack_bitstream(packed, bits, dim=dim), dim=dim)


def pack_scalar_zero(value: int, bits: int, num_words: int, shape, device=None) -> torch.Tensor:
    """Build a ``qzeros`` tensor where every group shares the same zero point.

    ``value`` is broadcast to ``PACK_BLOCK`` entries and packed once, then the
    resulting ``bits``-word pattern is tiled across ``num_words`` columns.
    """
    bits = _check_bits(bits)
    block = torch.full((PACK_BLOCK,), int(value), dtype=torch.int64, device=device)
    pattern = pack_bitstream(block, bits)  # (bits,)
    repeats = (num_words + bits - 1) // bits
    row = pattern.repeat(repeats)[:num_words]
    return row.reshape(*(1,) * (len(shape) - 1), num_words).expand(*shape).contiguous()


def infer_packed_bits(num_values: int, num_words: int) -> Optional[int]:
    """Best-effort recovery of ``bits`` from packed/unpacked dimension sizes."""
    if num_values <= 0 or num_words <= 0:
        return None
    bits = num_words * 32 // num_values
    return bits if bits in SUPPORTED_PACKING_BITS and num_values * bits == num_words * 32 else None
