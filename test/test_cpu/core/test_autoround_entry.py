import pytest

from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.autoround import _normalize_alg_configs


def _types(configs):
    return [type(config).__name__ for config in configs]


def test_default_alg_configs_is_signround():
    configs = _normalize_alg_configs(None)

    assert _types(configs) == ["SignRoundConfig"]


@pytest.mark.parametrize(
    "value",
    [
        "signround",
        SignRoundConfig(iters=3),
        [SignRoundConfig(iters=3)],
        ("signround",),
    ],
)
def test_alg_configs_accepts_alias_config_and_sequence(value):
    configs = _normalize_alg_configs(value)

    assert len(configs) == 1
    assert isinstance(configs[0], SignRoundConfig)


def test_awq_only_appends_rtn():
    configs = _normalize_alg_configs(AWQConfig())

    assert isinstance(configs[0], AWQConfig)
    assert isinstance(configs[1], RTNConfig)


def test_awq_and_signround_does_not_append_rtn():
    configs = _normalize_alg_configs([AWQConfig(), SignRoundConfig(iters=3)])

    assert _types(configs) == ["AWQConfig", "SignRoundConfig"]


def test_direct_signround_kwargs_warn_and_apply(monkeypatch):
    warnings = []
    monkeypatch.setattr("auto_round.autoround.logger.warning", lambda message, *args: warnings.append(message % args))

    configs = _normalize_alg_configs(None, direct_kwargs={"iters": 7, "lr": 0.2})

    assert isinstance(configs[0], SignRoundConfig)
    assert configs[0].iters == 7
    assert configs[0].lr == 0.2
    assert any("alg_configs" in message for message in warnings)


def test_awq_kwargs_without_awq_error_and_are_ignored(monkeypatch):
    errors = []
    monkeypatch.setattr("auto_round.autoround.logger.error", lambda message, *args: errors.append(message % args))

    configs = _normalize_alg_configs(None, direct_kwargs={"n_grid": 7, "apply_clip": True})

    assert isinstance(configs[0], SignRoundConfig)
    assert not hasattr(configs[0], "n_grid")
    assert any("AWQ" in message for message in errors)
