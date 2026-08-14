import inspect

import pytest

from auto_round import autoround
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.hadamard.config import RotationConfig
from auto_round.autoround import _normalize_alg_configs, _split_entry_kwargs


def _types(configs):
    return [type(config).__name__ for config in configs]


def test_split_entry_kwargs_partitions_owned_fields():
    processor = object()

    grouped = _split_entry_kwargs(
        {"dataset": ["sample"], "scale_dtype": "fp32", "processor": processor, "model_free": False}
    )

    assert grouped["base"]["dataset"] == ["sample"]
    assert grouped["compressor"]["scale_dtype"] == "fp32"
    assert grouped["mllm"]["processor"] is processor
    assert grouped["route"]["model_free"] is False


def test_split_entry_kwargs_ignores_unknown_fields(monkeypatch):
    warnings = []
    monkeypatch.setattr(autoround.logger, "warning_once", lambda message, *args: warnings.append(message % args))

    grouped = _split_entry_kwargs({"unknown_option": 1}, context="test entry")

    assert all(not values for values in grouped.values())
    assert "unknown_option" in warnings[0]


def test_cli_explicit_auto_round_with_zero_iters_selects_rtn():
    """The CLI's own algorithm-handling path (not _normalize_alg_configs) also
    selects RTN when iters=0 -- a separate code path from the alg_configs
    normalization below, so it's kept as a distinct case rather than folded in.
    """
    from argparse import Namespace

    from auto_round.cli.algorithms import AlgorithmHandler

    args = Namespace(algorithm="auto_round", iters=0)
    configs = AlgorithmHandler.build_configs(args, {})

    assert type(configs[0]).__name__ == "RTNConfig"


def test_default_alg_configs_is_signround():
    configs = _normalize_alg_configs(None)

    assert _types(configs) == ["SignRoundConfig"]


def test_public_entry_keeps_legacy_algorithm_parameter():
    from auto_round import AutoRound

    assert "algorithm" in inspect.signature(AutoRound.__new__).parameters


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


def test_zero_iters_selects_rtn_and_forwards_disable_opt_rtn():
    configs = _normalize_alg_configs(None, direct_kwargs={"iters": 0, "disable_opt_rtn": True})

    assert len(configs) == 1
    assert isinstance(configs[0], RTNConfig)
    assert configs[0].disable_opt_rtn is True


@pytest.mark.parametrize("algorithm", ["signround", "auto_round", "autoround"])
def test_legacy_signround_alias_with_zero_iters_selects_rtn(algorithm):
    legacy_configs = _normalize_alg_configs(None, direct_kwargs={"algorithm": algorithm, "iters": 0})
    unified_configs = _normalize_alg_configs(algorithm, direct_kwargs={"iters": 0})

    assert len(legacy_configs) == len(unified_configs) == 1
    assert isinstance(legacy_configs[0], RTNConfig)
    assert isinstance(unified_configs[0], RTNConfig)


def test_explicit_zero_iter_signround_config_selects_rtn():
    configs = _normalize_alg_configs(SignRoundConfig(iters=0))

    assert len(configs) == 1
    assert isinstance(configs[0], RTNConfig)


def test_combined_awq_zero_iters_keeps_awq_and_selects_rtn():
    configs = _normalize_alg_configs(None, direct_kwargs={"algorithm": "awq,signround", "iters": 0})

    assert _types(configs) == ["AWQConfig", "OptimizedRTNConfig"]


@pytest.mark.parametrize(
    ("algorithm", "expected_types"),
    [
        ("rtn", ["OptimizedRTNConfig"]),
        ("awq", ["AWQConfig", "OptimizedRTNConfig"]),
        ("signround", ["SignRoundConfig"]),
        ("awq,signround", ["AWQConfig", "SignRoundConfig"]),
    ],
)
def test_legacy_algorithm_matches_alg_configs_aliases(algorithm, expected_types):
    legacy_configs = _normalize_alg_configs(None, direct_kwargs={"algorithm": algorithm})
    unified_configs = _normalize_alg_configs(algorithm)

    assert _types(legacy_configs) == expected_types
    assert _types(unified_configs) == expected_types


def test_algorithm_and_alg_configs_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot be used together"):
        _normalize_alg_configs("signround", direct_kwargs={"algorithm": "rtn"})


def test_string_and_string_sequence_use_same_alias_deduplication():
    comma_configs = _normalize_alg_configs("awq,awq,signround")
    sequence_configs = _normalize_alg_configs(["awq", "awq", "signround"])

    assert _types(comma_configs) == ["AWQConfig", "SignRoundConfig"]
    assert _types(sequence_configs) == _types(comma_configs)


def test_legacy_combined_algorithm_forwards_each_algorithm_option():
    configs = _normalize_alg_configs(
        None,
        direct_kwargs={"algorithm": "awq,signround", "n_grid": 7, "iters": 3, "enable_alg_ext": False},
    )

    assert _types(configs) == ["AWQConfig", "SignRoundConfig"]
    assert configs[0].n_grid == 7
    assert configs[1].iters == 3
    assert configs[1].enable_alg_ext is False


@pytest.mark.parametrize(
    "configs",
    [
        lambda: _normalize_alg_configs(None, direct_kwargs={"algorithm": "awq,signround", "iters": 3}),
        lambda: _normalize_alg_configs([AWQConfig(), SignRoundConfig(iters=3)]),
    ],
)
def test_legacy_and_unified_composition_select_same_v1_quantizer(configs):
    from auto_round.algorithms.composer import AlgorithmComposer

    composer = AlgorithmComposer(configs())

    assert [type(pre).__name__ for pre in composer.preprocessors] == ["AWQTransform"]
    assert type(composer.block_quantizer).__name__ == "SignRoundQuantizer"


def test_direct_enable_opt_rtn_forces_optimized_rtn():
    configs = _normalize_alg_configs(None, direct_kwargs={"iters": 0, "enable_opt_rtn": True})

    assert len(configs) == 1
    assert isinstance(configs[0], RTNConfig)
    assert configs[0].disable_opt_rtn is False
    assert configs[0].orig_disable_opt_rtn is False


def test_rtn_switches_only_when_iters_is_zero():
    configs = _normalize_alg_configs(None, direct_kwargs={"disable_opt_rtn": True})

    assert isinstance(configs[0], SignRoundConfig)


def test_direct_enable_alg_ext_normalizes_signround_variant():
    configs = _normalize_alg_configs(None, direct_kwargs={"iters": 1, "enable_alg_ext": True})

    assert len(configs) == 1
    assert isinstance(configs[0], SignRoundV2Config)


def test_direct_rotation_config_is_added_to_pipeline():
    configs = _normalize_alg_configs(None, direct_kwargs={"rotation_config": "default"})

    assert any(isinstance(config, RotationConfig) for config in configs)


def test_direct_enable_lfq_is_forwarded():
    configs = _normalize_alg_configs(None, direct_kwargs={"enable_lfq": True})

    assert isinstance(configs[0], SignRoundConfig)
    assert configs[0].enable_lfq is True


def test_rotation_config_accepts_spinquant_shorthand():
    configs = _normalize_alg_configs(None, direct_kwargs={"rotation_config": "quarot"})

    assert any(type(config).__name__ == "SpinQuantConfig" for config in configs)


def test_rotation_backend_must_be_nested_in_rotation_config():
    with pytest.raises(ValueError, match="rotation_config"):
        _normalize_alg_configs(None, direct_kwargs={"backend": "transform"})


def test_awq_kwargs_without_awq_error_and_are_ignored(monkeypatch):
    errors = []
    monkeypatch.setattr("auto_round.autoround.logger.error", lambda message, *args: errors.append(message % args))

    configs = _normalize_alg_configs(None, direct_kwargs={"n_grid": 7, "apply_clip": True})

    assert isinstance(configs[0], SignRoundConfig)
    assert not hasattr(configs[0], "n_grid")
    assert any("AWQ" in message for message in errors)
