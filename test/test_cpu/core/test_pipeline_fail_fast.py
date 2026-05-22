"""Fast unit tests for algorithm registry and pipeline construction."""

import pytest

from auto_round.algorithms.quantization import registry as _r
from auto_round.algorithms.quantization.awq.config import AWQConfig
from auto_round.algorithms.quantization.pipeline import QuantizationPipeline, split_quantization_configs
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig


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
