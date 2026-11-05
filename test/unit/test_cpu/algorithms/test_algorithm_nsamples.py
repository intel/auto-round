# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_round.algorithms.composer import AlgorithmComposer
from auto_round.algorithms.config_resolver import resolve_shared_config_values, sync_shared_config_from
from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.qat.config import QATConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.cli.algorithms import AlgorithmHandler


@pytest.mark.parametrize("config_cls", [QuantizationConfig, SignRoundConfig, QADConfig, QATConfig])
def test_inherited_nsamples(config_cls):
    assert config_cls().nsamples is None
    assert config_cls(nsamples=16).nsamples == 16


@pytest.mark.parametrize("config_cls", [QuantizationConfig, SignRoundConfig, QADConfig, QATConfig])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "16"])
def test_invalid_nsamples(config_cls, value):
    with pytest.raises(ValueError, match="nsamples.*positive integer"):
        config_cls(nsamples=value)


@pytest.mark.parametrize("config_cls", [QADConfig, QATConfig])
@pytest.mark.parametrize("value", [None, 16])
def test_qad_nsamples_keyword(config_cls, value):
    config = config_cls(qad_nsamples=value)
    assert config.nsamples == value
    assert "qad_nsamples" not in vars(config)


def test_qad_signature_exposes_only_stage_specific_sample_parameter():
    parameters = inspect.signature(QADConfig).parameters
    assert "qad_nsamples" in parameters
    assert "nsamples" not in parameters


@pytest.mark.parametrize("config_cls", [QADConfig, QATConfig])
@pytest.mark.parametrize("target", [None, 16])
def test_qad_nsamples_overrides_inherited_sample_count(config_cls, target):
    config = config_cls(nsamples=8, qad_nsamples=target)
    assert config.nsamples == (8 if target is None else target)


@pytest.mark.parametrize("config_cls", [QADConfig, QATConfig])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "16"])
def test_invalid_qad_nsamples(config_cls, value):
    with pytest.raises(ValueError, match="nsamples.*positive integer"):
        config_cls(qad_nsamples=value)


@pytest.mark.parametrize("algorithm", ["qad", "qat", "sign_round,qad"])
@pytest.mark.parametrize("target", [None, 4, 32])
def test_qad_cli_maps_to_algorithm_nsamples(algorithm, target):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", type=int, default=128)
    AlgorithmHandler.add_groups(parser)
    argv = ["--nsamples", "8"]
    if target is not None:
        argv.extend(["--qad_nsamples", str(target)])
    args = parser.parse_args(argv)
    args.alg_configs = algorithm
    configs = AlgorithmHandler.build_configs(args, {})
    config = configs[-1]
    assert isinstance(config, QADConfig)
    assert config.nsamples == target
    assert args.nsamples == 8
    if len(configs) == 2:
        assert configs[0].nsamples is None
    registry = type(config).get_registered_args()
    parameter = next(p for p in registry.parameters if "--qad_nsamples" in p.option_strings)
    assert parameter.field == "qad_nsamples"
    cached = list(range(args.nsamples))
    collected = list(range(target or args.nsamples))
    collect = Mock(return_value=collected)
    tokens = BaseQuantizer(config).get_calibration_token_ids(cached, SimpleNamespace(collect_token_ids=collect))
    assert len(tokens) == (target or args.nsamples)
    if target is None or target <= args.nsamples:
        collect.assert_not_called()
        if target is None:
            assert tokens is cached
    else:
        collect.assert_called_once_with(target)


def test_nsamples_is_not_shared_or_synced():
    first, second = QADConfig(qad_nsamples=8), QADConfig(qad_nsamples=16)
    resolve_shared_config_values([first, second])
    assert (first.nsamples, second.nsamples) == (8, 16)
    unset = QADConfig()
    resolve_shared_config_values([first, unset])
    sync_shared_config_from(first, [unset, second])
    assert unset.nsamples is None
    assert second.nsamples == 16


def test_warmup_keeps_common_sampling():
    composer = AlgorithmComposer([SignRoundConfig(), QADConfig(qad_nsamples=16)])
    assert composer.quantizers[0].config.nsamples is None
    assert composer.active_quantizer.config.nsamples == 16


def test_unsupported_block_sampling_fails_explicitly():
    with pytest.raises(NotImplementedError, match="only supported for whole-model tuning"):
        AlgorithmComposer([SignRoundConfig(nsamples=16)])


def _resolve(target, cached, calibration):
    # This helper only needs the calibrator, not a loaded model or GPU state.
    quantizer = BaseQuantizer(QuantizationConfig(nsamples=target))
    return quantizer.get_calibration_token_ids(cached, calibration)


@pytest.mark.parametrize("target", [None, 3])
def test_whole_model_reuses_cache(target):
    cached = [1, 2, 3]
    collect = Mock()
    assert _resolve(target, cached, SimpleNamespace(collect_token_ids=collect)) is cached
    collect.assert_not_called()


def test_whole_model_slices_cache():
    cached = [1, 2, 3]
    collect = Mock()
    assert _resolve(2, cached, SimpleNamespace(collect_token_ids=collect)) == [1, 2]
    assert cached == [1, 2, 3]
    collect.assert_not_called()


@pytest.mark.parametrize("cached", [None, [1]])
def test_whole_model_collects_tokens(cached):
    tokens = [1, 2, 3]
    collect = Mock(return_value=tokens)
    assert _resolve(3, cached, SimpleNamespace(collect_token_ids=collect)) is tokens
    collect.assert_called_once_with(3)


@pytest.mark.parametrize("calibration", [SimpleNamespace(), SimpleNamespace(collect_token_ids=lambda n: [])])
def test_whole_model_collection_falls_back(calibration):
    cached = [1]
    assert _resolve(3, cached, calibration) is cached
