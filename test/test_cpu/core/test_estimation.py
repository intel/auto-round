import math

import pytest

from auto_round.estimation import (
    _count_parameters,
    _format_bytes,
    _format_time,
    estimate_output_size,
    estimate_time,
    estimate_vram,
)


class FakeConfig:
    """Minimal config stub for testing parameter count estimation."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestCountParameters:
    def test_basic_config(self):
        config = FakeConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            tie_word_embeddings=True,
        )
        count = _count_parameters(config)
        assert count is not None
        # LLaMA-7B has ~6.7B params, our estimate should be in the right ballpark
        assert 6e9 < count < 8e9

    def test_missing_hidden_size(self):
        config = FakeConfig(num_hidden_layers=32)
        assert _count_parameters(config) is None

    def test_missing_num_layers(self):
        config = FakeConfig(hidden_size=4096)
        assert _count_parameters(config) is None

    def test_untied_embeddings(self):
        tied = FakeConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            tie_word_embeddings=True,
        )
        untied = FakeConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            tie_word_embeddings=False,
        )
        tied_count = _count_parameters(tied)
        untied_count = _count_parameters(untied)
        # Untied should have extra vocab_size * hidden_size params
        assert untied_count > tied_count
        assert untied_count - tied_count == 32000 * 4096

    def test_gqa_config(self):
        """Test grouped-query attention (fewer KV heads)."""
        config = FakeConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            vocab_size=32000,
            tie_word_embeddings=True,
        )
        count = _count_parameters(config)
        assert count is not None
        # GQA model should have fewer params than full MHA
        full_mha = FakeConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            tie_word_embeddings=True,
        )
        assert count < _count_parameters(full_mha)


class TestEstimateVram:
    def test_basic(self):
        vram = estimate_vram(
            param_count=7_000_000_000,
            model_dtype_bytes=2,
            batch_size=8,
            seqlen=2048,
            hidden_size=4096,
        )
        assert vram > 0
        # 7B params * 2 bytes = 14GB model weights, total should be more
        assert vram > 14e9

    def test_larger_batch_more_vram(self):
        small = estimate_vram(7e9, 2, batch_size=1, seqlen=2048, hidden_size=4096)
        large = estimate_vram(7e9, 2, batch_size=16, seqlen=2048, hidden_size=4096)
        assert large > small


class TestEstimateOutputSize:
    def test_4bit(self):
        size = estimate_output_size(param_count=7_000_000_000, target_bits=4, group_size=128)
        # 7B * 4 bits / 8 = ~3.5GB, plus overhead
        assert 3e9 < size < 5e9

    def test_2bit_smaller(self):
        size_4 = estimate_output_size(7e9, target_bits=4, group_size=128)
        size_2 = estimate_output_size(7e9, target_bits=2, group_size=128)
        assert size_2 < size_4

    def test_no_group(self):
        size = estimate_output_size(7e9, target_bits=4, group_size=0)
        # Without grouping, no scale/zp overhead
        expected = int(math.ceil(7e9 * 4 / 8))
        assert size == expected


class TestEstimateTime:
    def test_basic(self):
        time_s = estimate_time(num_layers=32, iters=200, nsamples=128, batch_size=8)
        assert time_s > 0

    def test_more_iters_more_time(self):
        t1 = estimate_time(32, iters=100, nsamples=128, batch_size=8)
        t2 = estimate_time(32, iters=200, nsamples=128, batch_size=8)
        assert t2 > t1


class TestFormatHelpers:
    def test_format_bytes(self):
        assert "GB" in _format_bytes(14e9)
        assert "MB" in _format_bytes(500e6)
        assert "TB" in _format_bytes(2e12)
        assert "KB" in _format_bytes(500e3)

    def test_format_time(self):
        assert "seconds" in _format_time(30)
        assert "minutes" in _format_time(300)
        assert "hours" in _format_time(7200)
