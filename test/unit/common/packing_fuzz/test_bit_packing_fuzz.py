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
"""Fuzzing of the generic little-endian int32 bit-stream used by GPTQ/AWQ ``qweight``/``qzeros``.

The oracle is a pure-Python big-integer rendering of the layout documented in
``auto_round/utils/bit_packing.py``: value ``i`` of each 32-value block occupies bits
``[i*bits, (i+1)*bits)`` of the block's ``bits`` little-endian words.
"""

from test.fuzz_helpers import FUZZ_DEVICE

import pytest
import torch
from hypothesis import given, note
from hypothesis import strategies as st

from auto_round.utils.bit_packing import (
    AWQ_PACK_ORDER,
    AWQ_REVERSE_ORDER,
    SUPPORTED_PACKING_BITS,
    awq_reorder,
    awq_reverse_reorder,
    infer_packed_bits,
    pack_awq_bitstream,
    pack_bitstream,
    pack_scalar_zero,
    packed_dim_size,
    unpack_awq_bitstream,
    unpack_bitstream,
)

INT_DTYPES = (torch.int32, torch.int64, torch.int16, torch.uint8)
# AutoAWQ's classic 4-bit packing: word slot ``i`` holds logical value ``order_map[i]`` of each 8.
AUTOAWQ_ORDER_MAP = (0, 2, 4, 6, 1, 3, 5, 7)


def _to_int32(word: int) -> int:
    return word - (1 << 32) if word >= 1 << 31 else word


def _reference_block_words(block, bits):
    acc = 0
    for i, value in enumerate(block):
        acc |= (int(value) & ((1 << bits) - 1)) << (i * bits)
    return [_to_int32((acc >> (32 * w)) & 0xFFFFFFFF) for w in range(bits)]


def _reference_pack(values: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    moved = values.detach().cpu().to(torch.int64).movedim(dim, -1)
    rows = moved.reshape(-1, moved.shape[-1]).tolist()
    packed = [
        [w for b in range(0, len(row), 32) for w in _reference_block_words(row[b : b + 32], bits)] for row in rows
    ]
    out = torch.tensor(packed, dtype=torch.int32).reshape(*moved.shape[:-1], moved.shape[-1] * bits // 32)
    return out.movedim(-1, dim)


@st.composite
def bitstream_layouts(draw, block=32, max_blocks=4):
    """Shape with one packable dimension (a multiple of ``block``) at a random position."""
    lead = draw(st.lists(st.integers(1, 3), max_size=2))
    pos = draw(st.integers(0, len(lead)))
    shape = (*lead[:pos], block * draw(st.integers(1, max_blocks)), *lead[pos:])
    dim = draw(st.sampled_from([pos, pos - len(shape)]))
    return shape, dim


def _random_ints(shape, dtype, seed, high=None):
    g = torch.Generator().manual_seed(seed)
    info = torch.iinfo(dtype)
    # ``torch.randint`` cannot span the full int64 range; 2**40 already exercises bits above the word.
    low, high = (0, high) if high is not None else (max(info.min, -(2**40)), min(info.max, 2**40))
    return torch.randint(low, high, shape, generator=g, dtype=torch.int64).to(dtype).to(FUZZ_DEVICE)


@given(
    bits=st.sampled_from(SUPPORTED_PACKING_BITS),
    layout=bitstream_layouts(),
    dtype=st.sampled_from(INT_DTYPES),
    in_range=st.booleans(),
    seed=st.integers(0, 2**16),
)
def test_pack_bitstream_matches_reference_and_roundtrips(bits, layout, dtype, in_range, seed):
    shape, dim = layout
    values = _random_ints(shape, dtype, seed, high=1 << bits if in_range else None)

    packed = pack_bitstream(values, bits, dim=dim)

    expected_shape = list(shape)
    expected_shape[dim] = shape[dim] * bits // 32
    assert packed.dtype == torch.int32 and list(packed.shape) == expected_shape
    assert packed.device == values.device
    # Out-of-range inputs are masked to their low ``bits`` and never leak into neighbouring values.
    assert torch.equal(packed.cpu(), _reference_pack(values, bits, dim))
    unpacked = unpack_bitstream(packed, bits, dim=dim)
    assert unpacked.dtype == torch.int32 and unpacked.shape == values.shape
    assert torch.equal(unpacked, values.to(torch.int64) & ((1 << bits) - 1))


@given(
    bits=st.sampled_from(SUPPORTED_PACKING_BITS),
    layout=bitstream_layouts(block=1, max_blocks=24),
    seed=st.integers(0, 2**16),
)
def test_bitstream_is_a_bijection_on_int32_words(bits, layout, seed):
    """Every int32 bit pattern is a valid packed word: unpack then pack must reproduce it exactly."""
    shape, dim = layout
    shape = list(shape)
    shape[dim] *= bits
    words = _random_ints(shape, torch.int32, seed)

    values = unpack_bitstream(words, bits, dim=dim)

    assert int(values.min()) >= 0 and int(values.max()) < 1 << bits
    assert values.shape[dim] == words.shape[dim] * 32 // bits
    assert torch.equal(pack_bitstream(values, bits, dim=dim), words)


@given(layout=bitstream_layouts(block=8, max_blocks=12), seed=st.integers(0, 2**16))
def test_awq_reorder_and_reverse_are_inverse_permutations(layout, seed):
    shape, dim = layout
    values = _random_ints(shape, torch.int32, seed)

    shuffled = awq_reorder(values, dim=dim)

    assert sorted(AWQ_PACK_ORDER) == list(range(8)) and sorted(AWQ_REVERSE_ORDER) == list(range(8))
    assert torch.equal(awq_reverse_reorder(shuffled, dim=dim), values)
    assert torch.equal(awq_reorder(awq_reverse_reorder(values, dim=dim), dim=dim), values)
    assert torch.equal(shuffled.sort(dim=dim).values, values.sort(dim=dim).values)


@given(
    bits=st.sampled_from(SUPPORTED_PACKING_BITS),
    layout=bitstream_layouts(),
    seed=st.integers(0, 2**16),
)
def test_awq_bitstream_roundtrips_and_matches_autoawq_4bit_layout(bits, layout, seed):
    shape, dim = layout
    values = _random_ints(shape, torch.int32, seed, high=1 << bits)

    packed = pack_awq_bitstream(values, bits, dim=dim)

    assert torch.equal(unpack_awq_bitstream(packed, bits, dim=dim), values)
    if bits == 4:
        groups = values.movedim(dim, -1).cpu().to(torch.int64)
        groups = groups.reshape(*groups.shape[:-1], -1, 8)
        words = sum(groups[..., AUTOAWQ_ORDER_MAP[i]] << (4 * i) for i in range(8))
        words = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)
        assert torch.equal(packed.cpu(), words.movedim(-1, dim))


@given(
    bits=st.sampled_from(SUPPORTED_PACKING_BITS),
    data=st.data(),
    num_words=st.integers(1, 40),
    rows=st.integers(1, 4),
)
def test_pack_scalar_zero_tiles_the_packed_constant_block(bits, data, num_words, rows):
    value = data.draw(st.integers(0, (1 << bits) - 1), label="value")

    qzeros = pack_scalar_zero(value, bits, num_words, (rows, num_words), device=FUZZ_DEVICE)

    pattern = _reference_block_words([value] * 32, bits)
    expected = torch.tensor([pattern[i % bits] for i in range(num_words)], dtype=torch.int32).expand(rows, -1)
    assert qzeros.dtype == torch.int32 and qzeros.is_contiguous()
    assert torch.equal(qzeros.cpu(), expected)
    if num_words % bits == 0:
        assert bool((unpack_bitstream(qzeros, bits, dim=1) == value).all())


@given(num_values=st.integers(1, 4096), bits=st.sampled_from(SUPPORTED_PACKING_BITS))
def test_packed_dim_size_requires_whole_blocks_and_is_inverted_by_infer(num_values, bits):
    if num_values % 32:
        with pytest.raises(ValueError):
            packed_dim_size(num_values, bits)
        with pytest.raises(ValueError):
            pack_bitstream(torch.zeros(num_values, dtype=torch.int32), bits)
        return
    num_words = packed_dim_size(num_values, bits)
    assert num_words * 32 == num_values * bits
    assert infer_packed_bits(num_values, num_words) == bits


@given(num_values=st.integers(-2, 4096), num_words=st.integers(-2, 1024))
def test_infer_packed_bits_returns_the_unique_consistent_width(num_values, num_words):
    inferred = infer_packed_bits(num_values, num_words)

    candidates = [b for b in SUPPORTED_PACKING_BITS if num_values > 0 and num_values * b == num_words * 32]
    note(f"inferred={inferred} candidates={candidates}")
    assert inferred == (candidates[0] if candidates else None)


@given(
    bits=st.integers(-4, 40).filter(lambda b: b not in SUPPORTED_PACKING_BITS),
    words_per_bits=st.integers(1, 3),
)
def test_unsupported_widths_and_ragged_words_are_rejected(bits, words_per_bits):
    with pytest.raises(ValueError):
        pack_bitstream(torch.zeros(32, dtype=torch.int32), bits)
    with pytest.raises(ValueError):
        unpack_bitstream(torch.zeros(8, dtype=torch.int32), bits)
    with pytest.raises(ValueError):
        pack_scalar_zero(0, bits, 4, (1, 4))
    for valid_bits in SUPPORTED_PACKING_BITS[1:]:
        with pytest.raises(ValueError):
            unpack_bitstream(torch.zeros(valid_bits * words_per_bits + 1, dtype=torch.int32), valid_bits)
