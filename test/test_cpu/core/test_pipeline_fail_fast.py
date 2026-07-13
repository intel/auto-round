"""Fast unit tests for algorithm registry and pipeline construction."""

import pytest

from auto_round import AWQConfig, OptimizedRTNConfig, RotationConfig, RTNConfig, SignRoundConfig, SpinQuantConfig
from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    resolve_shared_config_values,
    split_quantization_configs,
    sync_shared_config_from,
)
from auto_round.algorithms.pipeline import QuantizationPipeline
from auto_round.algorithms.quantization import registry as _r
from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer
from auto_round.compressors.base import collect_user_scheme_overrides
from auto_round.compressors.data_driven import DataDrivenCompressor
from auto_round.compressors.entry import AutoRound as NewAutoRound
from auto_round.compressors.entry import _select_rtn_compressor_base_cls
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


class PartialSharedConfig(RTNConfig):
    def __init__(self, *, weight_clip_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_clip_ratio = weight_clip_ratio


class NoWeightClipConfig(RTNConfig):
    pass


def test_split_awq_plus_rtn():
    pre, block = split_quantization_configs([AWQConfig(), RTNConfig()])
    assert len(pre) == 1 and type(pre[0]).__name__ == "AWQConfig"
    assert len(block) == 1 and type(block[0]).__name__ == "RTNConfig"


def test_pipeline_preprocessor_only_auto_appends_rtn():
    pipeline = QuantizationPipeline.from_configs([AWQConfig()])
    assert type(pipeline.preprocessors[0]).__name__ == "AWQTransform"
    assert isinstance(pipeline.block_quantizer, RTNQuantizer)


def test_pipeline_duplicate_preprocessor_rejected():
    with pytest.raises(ValueError, match="Duplicate preprocessor"):
        QuantizationPipeline.from_configs([AWQConfig(), AWQConfig()])


def test_pipeline_multiple_block_quantizers_rejected():
    with pytest.raises(ValueError, match="exactly one block-quantization config"):
        QuantizationPipeline.from_configs([RTNConfig(), SignRoundConfig()])


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

    monkeypatch.setattr(DataDrivenCompressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    NewAutoRound("dummy-model", "W4A16", iters=1, seqlen=8, nsamples=1)

    assert isinstance(captured["config"], SignRoundConfig)


def test_entry_rejects_configs_without_quantization_members():
    with pytest.raises(ValueError, match="At least one quantization algorithm config"):
        NewAutoRound("dummy-model", "W4A16", [RotationConfig()])


def test_compat_entry_preserves_spinquant_dict_config(monkeypatch):
    captured = {}
    rotation_config = {
        "algorithm": "spinquant",
        "r1": True,
        "r2": True,
        "r3": False,
        "r4": False,
        "rotation_size": 128,
        "trainable_rotation": False,
        "trainable_smooth": False,
    }

    def _fake_init(self, config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(DataDrivenCompressor, "__init__", _fake_init)
    monkeypatch.setattr("auto_round.utils.is_mllm_model", lambda *args, **kwargs: False)
    monkeypatch.setattr("auto_round.utils.is_diffusion_model", lambda *args, **kwargs: False)
    monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *args, **kwargs: "llm")

    from auto_round.autoround import AutoRound as CompatAutoRound

    CompatAutoRound(
        "dummy-model",
        scheme="W4A16",
        iters=1,
        seqlen=8,
        nsamples=1,
        rotation_config=rotation_config,
    )

    configs = captured["config"] if isinstance(captured["config"], list) else [captured["config"]]
    spinquant_cfg = next(cfg for cfg in configs if isinstance(cfg, SpinQuantConfig))
    assert spinquant_cfg.rotation_size == rotation_config["rotation_size"]
    assert spinquant_cfg.r1 is rotation_config["r1"]
    assert spinquant_cfg.r2 is rotation_config["r2"]
    assert spinquant_cfg.r3 is rotation_config["r3"]
    assert spinquant_cfg.r4 is rotation_config["r4"]
    assert spinquant_cfg.trainable_rotation is rotation_config["trainable_rotation"]
    assert spinquant_cfg.trainable_smooth is rotation_config["trainable_smooth"]


def test_entry_warns_and_drops_unsupported_kwargs(monkeypatch, tiny_opt_model_path):
    calls = []

    def _record_warning(message, *args):
        calls.append(message % args)

    monkeypatch.setattr(logger, "warning_once", _record_warning)

    NewAutoRound(
        tiny_opt_model_path,
        "W4A16",
        RTNConfig(disable_opt_rtn=True),
        nsamples=1,
        seqlen=8,
        low_cpu_mem_usage=False,
        nonsense_kwarg=123,
    )

    assert any("unsupported kwargs nonsense_kwarg" in msg for msg in calls)


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


def test_shared_config_sync_from_source_skips_missing_attrs():
    source = PartialSharedConfig(weight_clip_ratio=0.75)
    target = PartialSharedConfig()
    no_clip_target = NoWeightClipConfig()

    sync_shared_config_from(source, [target, no_clip_target, RotationConfig()])

    assert target.weight_clip_ratio == 0.75
    assert not hasattr(no_clip_target, "weight_clip_ratio")


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
