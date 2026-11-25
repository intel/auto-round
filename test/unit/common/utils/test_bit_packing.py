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
"""Tests for ``auto_round/utils/bit_packing.py`` (generic int32 bit-stream packing)."""

import pytest
import torch

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
    requires_generic_bit_packing,
    unpack_awq_bitstream,
    unpack_bitstream,
)


def _rand_values(bits, *shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2**bits, shape, generator=g, dtype=torch.int32)


def _legacy_word_aligned_pack(values, bits):
    """Classic GPTQ layout: ``32 // bits`` values per int32 word, value i at bit ``i * bits``."""
    per_word = 32 // bits
    v = values.to(torch.int64).reshape(*values.shape[:-1], -1, per_word)
    shifts = torch.arange(per_word, dtype=torch.int64) * bits
    words = (v << shifts).sum(-1) & 0xFFFFFFFF
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


class TestBitstreamRoundTrip:
    @pytest.mark.parametrize("bits", SUPPORTED_PACKING_BITS)
    def test_roundtrip_last_dim(self, bits):
        values = _rand_values(bits, 3, 64)
        packed = pack_bitstream(values, bits)
        assert packed.dtype == torch.int32
        assert packed.shape == (3, 64 * bits // 32)
        torch.testing.assert_close(unpack_bitstream(packed, bits), values)

    @pytest.mark.parametrize("bits", [3, 5, 7])
    def test_roundtrip_non_last_dim(self, bits):
        values = _rand_values(bits, 64, 5)
        packed = pack_bitstream(values, bits, dim=0)
        assert packed.shape == (64 * bits // 32, 5)
        torch.testing.assert_close(unpack_bitstream(packed, bits, dim=0), values)

    def test_out_of_range_values_are_masked(self):
        values = torch.full((32,), 0xFF, dtype=torch.int32)
        torch.testing.assert_close(unpack_bitstream(pack_bitstream(values, 4), 4), torch.full_like(values, 0xF))

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_word_aligned_bits_match_legacy_layout(self, bits):
        values = _rand_values(bits, 2, 64, seed=1)
        torch.testing.assert_close(pack_bitstream(values, bits), _legacy_word_aligned_pack(values, bits))

    def test_sign_bit_produces_negative_int32(self):
        values = torch.zeros(32, dtype=torch.int32)
        values[7] = 0xF  # occupies the top nibble of word 0 at 4 bits
        packed = pack_bitstream(values, 4)
        assert packed[0].item() < 0
        torch.testing.assert_close(unpack_bitstream(packed, 4), values)


class TestValidation:
    def test_unsupported_bits(self):
        with pytest.raises(ValueError, match="unsupported bits"):
            pack_bitstream(torch.zeros(32, dtype=torch.int32), 9)
        with pytest.raises(ValueError, match="unsupported bits"):
            unpack_bitstream(torch.zeros(9, dtype=torch.int32), 9)

    def test_pack_dim_not_multiple_of_block(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            pack_bitstream(torch.zeros(48, dtype=torch.int32), 4)

    def test_unpack_dim_not_multiple_of_bits(self):
        with pytest.raises(ValueError, match="not a multiple of bits"):
            unpack_bitstream(torch.zeros(4, dtype=torch.int32), 3)

    def test_awq_dim_not_multiple_of_8(self):
        with pytest.raises(ValueError, match="multiple of 8"):
            awq_reorder(torch.zeros(12, dtype=torch.int32))

    @pytest.mark.parametrize("bits, expected", [(2, False), (3, False), (4, False), (5, True), (6, True), (7, True)])
    def test_requires_generic_bit_packing(self, bits, expected):
        assert requires_generic_bit_packing(bits) is expected

    def test_packed_dim_size(self):
        assert packed_dim_size(128, 4) == 16
        assert packed_dim_size(64, 3) == 6
        with pytest.raises(ValueError):
            packed_dim_size(100, 4)

    @pytest.mark.parametrize(
        "num_values, num_words, expected",
        [(128, 16, 4), (64, 6, 3), (64, 14, 7), (0, 4, None), (64, 0, None), (64, 7, None), (32, 288, None)],
    )
    def test_infer_packed_bits(self, num_values, num_words, expected):
        assert infer_packed_bits(num_values, num_words) == expected


class TestAwqOrdering:
    def test_orders_are_inverse_permutations(self):
        for i in range(8):
            assert AWQ_REVERSE_ORDER[AWQ_PACK_ORDER[i]] == i

    def test_reorder_roundtrip(self):
        values = torch.arange(32, dtype=torch.int32).reshape(2, 16)
        reordered = awq_reorder(values)
        assert reordered[0, :8].tolist() == list(AWQ_PACK_ORDER)
        torch.testing.assert_close(awq_reverse_reorder(reordered), values)

    def test_reorder_along_dim0(self):
        values = torch.arange(16, dtype=torch.int32).reshape(16, 1)
        torch.testing.assert_close(awq_reverse_reorder(awq_reorder(values, dim=0), dim=0), values)

    def test_awq_4bit_matches_autoawq_layout(self):
        values = _rand_values(4, 2, 32, seed=2)
        packed = pack_awq_bitstream(values, 4)
        expected = torch.zeros(2, 4, dtype=torch.int64)
        for col in range(4):
            for i, src in enumerate(AWQ_PACK_ORDER):
                expected[:, col] |= values[:, col * 8 + src].to(torch.int64) << (i * 4)
        expected = torch.where(expected >= 2**31, expected - 2**32, expected).to(torch.int32)
        torch.testing.assert_close(packed, expected)

    @pytest.mark.parametrize("bits", [3, 4, 6])
    def test_awq_roundtrip(self, bits):
        values = _rand_values(bits, 4, 64, seed=3)
        torch.testing.assert_close(unpack_awq_bitstream(pack_awq_bitstream(values, bits), bits), values)


class TestPackScalarZero:
    @pytest.mark.parametrize("bits, value", [(4, 8), (3, 4), (5, 16), (8, 128)])
    def test_every_group_unpacks_to_value(self, bits, value):
        num_words = 64 * bits // 32
        qzeros = pack_scalar_zero(value, bits, num_words, shape=(3, num_words))
        assert qzeros.shape == (3, num_words) and qzeros.is_contiguous()
        unpacked = unpack_bitstream(qzeros, bits)
        assert torch.all(unpacked == value)

    def test_partial_pattern_is_truncated(self):
        # num_words that is not a multiple of bits truncates the tiled pattern.
        qzeros = pack_scalar_zero(1, 3, 4, shape=(4,))
        full = pack_scalar_zero(1, 3, 6, shape=(6,))
        torch.testing.assert_close(qzeros, full[:4])
