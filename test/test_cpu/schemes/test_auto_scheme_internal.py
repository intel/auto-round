import os

from auto_round.auto_scheme.delta_loss import _apply_head_trick
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme


def test_apply_head_trick_keeps_lowest_loss_candidate_when_budget_allows():
    schemes = AutoScheme(avg_bits=3, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M")).options
    total_scores = {
        "lm_head": [
            [0, 1244659712, 41.0, ["lm_head"]],
            [1, 2489319424, 11.9, ["lm_head"]],
        ],
        "model.layers.0.mlp.down_proj": [
            [0, 100, 10.0, ["model.layers.0.mlp.down_proj"]],
            [1, 200, 5.0, ["model.layers.0.mlp.down_proj"]],
        ],
    }

    _apply_head_trick(
        head_name="lm_head",
        schemes=schemes,
        sorted_indices=[1, 0],
        target_bits=3,
        target_params_cnt=4891670016,
        total_scores=total_scores,
    )

    assert total_scores["lm_head"] == [[1, 2489319424, 11.9, ["lm_head"]]]


def test_apply_head_trick_relaxes_lowest_loss_candidate_when_budget_is_tight():
    schemes = AutoScheme(avg_bits=3, options=("GGUF:Q2_K_S", "GGUF:Q4_K_M")).options
    total_scores = {
        "lm_head": [
            [0, 1244659712, 41.0, ["lm_head"]],
            [1, 2489319424, 11.9, ["lm_head"]],
        ],
        "model.layers.0.mlp.down_proj": [
            [0, 100, 10.0, ["model.layers.0.mlp.down_proj"]],
            [1, 200, 5.0, ["model.layers.0.mlp.down_proj"]],
        ],
    }

    _apply_head_trick(
        head_name="lm_head",
        schemes=schemes,
        sorted_indices=[1, 0],
        target_bits=3,
        target_params_cnt=1244659812,
        total_scores=total_scores,
    )

    assert total_scores["lm_head"] == [
        [0, 1244659712, 41.0, ["lm_head"]],
        [1, 2489319424, 11.9, ["lm_head"]],
    ]


def test_env_ar_auto_scheme_nsamples_overrides_default(monkeypatch):
    """AR_AUTO_SCHEME_NSAMPLES env var should override the built-in nsamples heuristic."""
    import auto_round.envs as envs

    monkeypatch.setenv("AR_AUTO_SCHEME_NSAMPLES", "7")
    assert envs.AR_AUTO_SCHEME_NSAMPLES == 7


def test_env_ar_auto_scheme_batch_size_overrides_default(monkeypatch):
    """AR_AUTO_SCHEME_BATCH_SIZE env var should override the built-in batch_size default."""
    import auto_round.envs as envs

    monkeypatch.setenv("AR_AUTO_SCHEME_BATCH_SIZE", "4")
    assert envs.AR_AUTO_SCHEME_BATCH_SIZE == 4


def test_env_ar_auto_scheme_batch_size_zero_raises(monkeypatch):
    """Zero value for AR_AUTO_SCHEME_BATCH_SIZE should raise ValueError."""
    import pytest

    import auto_round.envs as envs

    monkeypatch.setenv("AR_AUTO_SCHEME_BATCH_SIZE", "0")
    with pytest.raises(ValueError):
        _ = envs.AR_AUTO_SCHEME_BATCH_SIZE
