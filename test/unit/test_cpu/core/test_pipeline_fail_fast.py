"""Fast unit tests for algorithm registry and bundle construction."""

from types import SimpleNamespace

import pytest

from auto_round import AutoRound as NewAutoRound
from auto_round import AWQConfig, OptimizedRTNConfig, RotationConfig, RTNConfig, SignRoundConfig, SpinQuantConfig
from auto_round.algorithms.composer import AlgorithmComposer, _can_compile_block_forward, _has_nvfp4_layer
from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    resolve_shared_config_values,
    split_quantization_configs,
)
from auto_round.algorithms.quantization import registry as _r
from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer
from auto_round.autoround import _select_rtn_compressor_base_cls
from auto_round.compressors.base import collect_user_scheme_overrides
from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


class PartialSharedConfig(RTNConfig):
    def __init__(self, *, weight_clip_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_clip_ratio = weight_clip_ratio


class NoWeightClipConfig(RTNConfig):
    pass


class CompileCompatibleRotation:
    def can_compile_block_forward(self):
        return True


class CompileCompatibleQuantizer:
    def can_compile_block_forward(self):
        return True


def test_split_awq_plus_rtn():
    pre, block = split_quantization_configs([AWQConfig(), RTNConfig()])
    assert len(pre) == 1 and type(pre[0]).__name__ == "AWQConfig"
    assert len(block) == 1 and type(block[0]).__name__ == "RTNConfig"


def test_pipeline_preprocessor_only_auto_appends_rtn():
    pipeline = AlgorithmComposer([AWQConfig()])
    assert type(pipeline.preprocessors[0]).__name__ == "AWQTransform"
    assert isinstance(pipeline.block_quantizer, RTNQuantizer)


def test_pipeline_duplicate_preprocessor_rejected():
    with pytest.raises(ValueError, match="Duplicate preprocessor"):
        AlgorithmComposer([AWQConfig(), AWQConfig()])


def test_pipeline_multiple_block_quantizers_rejected():
    with pytest.raises(ValueError, match="exactly one block-quantization config"):
        AlgorithmComposer([RTNConfig(), SignRoundConfig()])


def test_hadamard_disables_only_block_forward_compile():
    quantizer = CompileCompatibleQuantizer()
    hadamard = RotationConfig()

    assert not _can_compile_block_forward(quantizer, [hadamard], user_enabled=True)
    assert _can_compile_block_forward(quantizer, [CompileCompatibleRotation()], user_enabled=True)
    assert not _can_compile_block_forward(quantizer, [], user_enabled=False)


def test_detect_nvfp4_from_layer_config_scheme_override():
    orchestrator = SimpleNamespace(
        data_type="mx_fp",
        layer_config={"mlp.experts": {"scheme": "NVFP4"}},
    )
    assert _has_nvfp4_layer(orchestrator)


def test_detect_nvfp4_from_layer_config_data_type_override():
    orchestrator = SimpleNamespace(
        data_type="mx_fp",
        layer_config={"mlp.experts": {"data_type": "nv_fp"}},
    )
    assert _has_nvfp4_layer(orchestrator)


def test_needs_calibration_data_when_layer_config_overrides_to_nvfp4():
    class _DummyCompressor:
        _needs_calibration_data = Compressor._needs_calibration_data
        _layer_config_needs_act_calibration = Compressor._layer_config_needs_act_calibration

    compressor = _DummyCompressor()
    compressor._alg_configs = []
    compressor.scheme = "MXFP8"
    compressor.layer_config = {"mlp.experts": {"scheme": "NVFP4"}}
    compressor.static_kv_dtype = None
    compressor.static_attention_dtype = None

    assert compressor._needs_calibration_data() is True


def test_registry_builtin_aliases_and_unknown():
    assert isinstance(_r.resolve_alg_config("RTN"), RTNConfig)
    assert isinstance(_r.resolve_alg_config("awq"), AWQConfig)
    assert isinstance(_r.resolve_alg_config("autoround"), SignRoundConfig)
    with pytest.raises(ValueError, match="Unknown algorithm alias"):
        _r.resolve_alg_config("definitely_not_registered_abc123")


def test_registry_resolves_variant_configs_to_registered_members():
    assert get_algorithm_class(OptimizedRTNConfig()) is not None
    assert get_algorithm_class(SignRoundConfig(enable_adam=True)).__name__ == "AdamRoundQuantizer"


def test_top_level_config_exports():
    from auto_round import AWQConfig as TopAWQConfig
    from auto_round import OptimizedRTNConfig as TopOptimizedRTNConfig
    from auto_round import RotationConfig as TopRotationConfig
    from auto_round import RTNConfig as TopRTNConfig
    from auto_round import SignRoundConfig as TopSignRoundConfig
    from auto_round import SpinQuantConfig as TopSpinQuantConfig

    assert TopAWQConfig is AWQConfig
    assert TopOptimizedRTNConfig is OptimizedRTNConfig
    assert TopRTNConfig is RTNConfig
    assert TopSignRoundConfig is SignRoundConfig
    assert TopRotationConfig is RotationConfig
    assert TopSpinQuantConfig is SpinQuantConfig


def test_new_entry_defaults_to_autoround_config(monkeypatch):
    captured = {}

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(Compressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    NewAutoRound("dummy-model", scheme="W4A16", iters=1, seqlen=8, nsamples=1)

    assert isinstance(captured["config"], list)
    assert isinstance(captured["config"][0], SignRoundConfig)


def test_new_entry_zero_iters_defaults_to_rtn_config(monkeypatch):
    captured = {}

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(Compressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    NewAutoRound("dummy-model", scheme="W4A16", iters=0, disable_opt_rtn=True, disable_model_free=True)

    assert isinstance(captured["config"], list)
    assert isinstance(captured["config"][0], RTNConfig)
    assert captured["config"][0].disable_opt_rtn is True


def test_new_entry_accepts_rotation_config_in_algorithm_list(monkeypatch):
    captured = {}

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(Compressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    NewAutoRound(
        "dummy-model",
        scheme="MXFP4",
        alg_configs=[SignRoundConfig(iters=1), RotationConfig()],
    )

    assert any(isinstance(config, RotationConfig) for config in captured["config"])


def test_entry_rejects_configs_without_quantization_members():
    with pytest.raises(TypeError, match="alg_configs entries must be algorithm aliases"):
        NewAutoRound("dummy-model", scheme="W4A16", alg_configs=[RotationConfig()])


def test_shared_config_values_inherit_across_matching_attrs_only():
    awq = PartialSharedConfig(weight_clip_ratio=0.9)
    smoothquant_like = NoWeightClipConfig()
    signround = PartialSharedConfig(weight_clip_ratio=None)

    resolve_shared_config_values([awq, smoothquant_like, signround])

    assert signround.weight_clip_ratio == 0.9
    assert not hasattr(smoothquant_like, "weight_clip_ratio")


def test_shared_config_values_reject_conflicts():
    with pytest.raises(ValueError, match="Conflicting shared config field 'weight_clip_ratio'"):
        resolve_shared_config_values(
            [PartialSharedConfig(weight_clip_ratio=0.8), PartialSharedConfig(weight_clip_ratio=0.9)]
        )


# def test_shared_config_sync_from_source_skips_missing_attrs():
#     source = PartialSharedConfig(weight_clip_ratio=0.75)
#     target = PartialSharedConfig()
#     no_clip_target = NoWeightClipConfig()
#
#     sync_shared_config_from(source, [target, no_clip_target, RotationConfig()])
#
#     assert target.weight_clip_ratio == 0.75
#     assert not hasattr(no_clip_target, "weight_clip_ratio")


def test_user_scheme_overrides_merge_across_all_configs():
    awq = AWQConfig(bits=8)
    rtn = RTNConfig()
    assert collect_user_scheme_overrides([awq, rtn])["bits"] == 8

    resolve_shared_config_values([awq, rtn])

    assert rtn.bits == 8


def test_user_scheme_overrides_reject_explicit_conflicts():
    with pytest.raises(ValueError, match="Conflicting shared scheme field 'bits'"):
        collect_user_scheme_overrides([AWQConfig(bits=8), RTNConfig(bits=4)])
    with pytest.raises(ValueError, match="Conflicting shared scheme field 'bits'"):
        resolve_shared_config_values([AWQConfig(bits=8), RTNConfig(bits=4)])


# ===========================================================================
#  Scheme-dependent config heuristics must see resolved values, not just
#  whatever (often None) bits/lr the config was constructed with directly.
# ===========================================================================


@pytest.mark.parametrize(
    "scheme, expect_disable_opt_rtn",
    [
        ("W8A16", True),
        # "INT8" (bits=8, act_bits=8, data_type=int) is W8A8-equivalent but was
        # previously missed because routing only matched the literal strings
        # "W8A16"/"W8A8", not schemes reaching the same resolved values.
        ("INT8", True),
        ("W4A16", False),
        ({"bits": 8, "act_bits": 8, "data_type": "int", "sym": True}, True),
    ],
)
def test_rtn_routing_disable_opt_rtn_from_resolved_scheme(scheme, expect_disable_opt_rtn):
    config = RTNConfig()
    _select_rtn_compressor_base_cls(config, scheme, "auto_round", {})
    assert config.disable_opt_rtn is expect_disable_opt_rtn


def test_rtn_routing_respects_explicit_enable_opt_rtn():
    """An explicit user choice must not be clobbered by the W8A16/W8A8 heuristic."""
    config = RTNConfig(enable_opt_rtn=True)
    _select_rtn_compressor_base_cls(config, "W8A16", "auto_round", {})
    assert config.disable_opt_rtn is False


@pytest.mark.parametrize("bits, expected_lr", [(3, 2.0 / 1000), (4, 1.0 / 1000)])
def test_sign_round_finalize_scheme_lr_heuristic(bits, expected_lr):
    """The low-bit lr bump must apply once `bits` is resolved via the scheme,
    even though it was unset (None) at construction time (e.g. `scheme=` alone,
    no explicit `bits=`)."""
    config = SignRoundConfig(iters=1000)
    config.scheme = QuantizationScheme(bits=bits, act_bits=16, data_type="int")
    config.finalize_scheme()
    assert config.lr == expected_lr


def test_sign_round_finalize_scheme_respects_explicit_lr():
    config = SignRoundConfig(iters=1000, lr=0.01, minmax_lr=0.05)
    config.scheme = QuantizationScheme(bits=2, act_bits=16, data_type="int")
    config.finalize_scheme()
    assert config.lr == 0.01
    assert config.minmax_lr == 0.05
