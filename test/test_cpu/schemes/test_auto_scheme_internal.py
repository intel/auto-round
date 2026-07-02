import os

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.auto_scheme.utils import _build_layer_config_header_rows, _short_summary_name


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


def test_build_layer_config_header_rows_merges_adjacent_prefixes():
    """Adjacent columns with the same prefix should be merged into one compact header cell."""
    columns = ["mlp.down_proj", "mlp.gate_proj", "self_attn.q_proj", "self_attn.v_proj"]
    assert _build_layer_config_header_rows(columns) == [
        ["block", "mlp", "", "self_attn", ""],
        ["", "down_proj", "gate_proj", "q_proj", "v_proj"],
    ]


def test_short_summary_name_keeps_one_field_before_numeric_suffix():
    """Numeric block suffixes should be shortened to keep the preceding field."""
    assert _short_summary_name("model.layers.0") == "layers.0"
