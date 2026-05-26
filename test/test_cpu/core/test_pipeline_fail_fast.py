"""Fast unit tests for algorithm registry and pipeline construction."""

import pytest

from auto_round.algorithms.quantization import registry as _r
from auto_round.algorithms.quantization.awq.config import AWQConfig
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.quantization.pipeline import (
    QuantizationPipeline,
    resolve_shared_config_values,
    split_quantization_configs,
    sync_shared_config_from,
)
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.transforms.rotation.config import RotationConfig
from auto_round.compressors.entry import AutoRound as NewAutoRound


class PartialSharedConfig(QuantizationConfig):
    _alg_cls = "RTNQuantizer"

    def __init__(self, *, weight_clip_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_clip_ratio = weight_clip_ratio


class NoWeightClipConfig(QuantizationConfig):
    _alg_cls = "RTNQuantizer"


def test_split_awq_plus_rtn():
    pre, block = split_quantization_configs([AWQConfig(), RTNConfig()])
    assert len(pre) == 1 and type(pre[0]).__name__ == "AWQConfig"
    assert len(block) == 1 and type(block[0]).__name__ == "RTNConfig"


def test_pipeline_preprocessor_only_auto_appends_rtn():
    pipeline = QuantizationPipeline.from_configs([AWQConfig()])
    assert type(pipeline.preprocessors[0]).__name__ == "AWQQuantizer"
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


def test_entry_rejects_configs_without_quantization_members():
    with pytest.raises(ValueError, match="At least one quantization algorithm config"):
        NewAutoRound(alg_configs=[RotationConfig()], model="dummy-model")


def test_shared_config_values_inherit_across_matching_attrs_only():
    awq = PartialSharedConfig(bits=4, weight_clip_ratio=0.9)
    smoothquant_like = NoWeightClipConfig()
    signround = PartialSharedConfig(weight_clip_ratio=None)

    resolve_shared_config_values([awq, smoothquant_like, signround])

    assert signround.bits == 4
    assert smoothquant_like.bits == 4
    assert signround.weight_clip_ratio == 0.9
    assert not hasattr(smoothquant_like, "weight_clip_ratio")


def test_shared_config_values_reject_conflicts():
    with pytest.raises(ValueError, match="Conflicting shared config field 'weight_clip_ratio'"):
        resolve_shared_config_values(
            [PartialSharedConfig(weight_clip_ratio=0.8), PartialSharedConfig(weight_clip_ratio=0.9)]
        )


def test_shared_config_sync_from_source_skips_missing_attrs():
    source = PartialSharedConfig(bits=4, weight_clip_ratio=0.75)
    target = PartialSharedConfig()
    no_clip_target = NoWeightClipConfig()

    sync_shared_config_from(source, [target, no_clip_target, RotationConfig()])

    assert target.bits == 4
    assert target.weight_clip_ratio == 0.75
    assert no_clip_target.bits == 4
    assert not hasattr(no_clip_target, "weight_clip_ratio")
