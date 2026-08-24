"""Unit tests for SignRoundQuantizer helpers."""

from auto_round.algorithms.quantization.sign_round.quantizer import _extend_total_iters_if_needed


def test_extend_total_iters_when_best_iter_is_in_last_20_percent():
    total_iters, extended = _extend_total_iters_if_needed(200, 160, False, True)

    assert total_iters == 300
    assert extended is True


def test_extend_total_iters_skips_when_best_mse_is_disabled():
    total_iters, extended = _extend_total_iters_if_needed(200, 199, False, False)

    assert total_iters == 200
    assert extended is False
