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

from unittest.mock import patch

import pytest
import torch

from auto_round import AutoRound, ParallelConfig
from auto_round.algorithms import registry
from auto_round.algorithms.base import BaseAlgorithm
from auto_round.schemes import QuantizationScheme


def test_legacy_implementation_registration_still_resolves_plugins(monkeypatch):
    registry._ensure_pipeline_members_registered()
    monkeypatch.setattr(registry, "_CONFIG_IMPL_REGISTRY", registry._CONFIG_IMPL_REGISTRY.copy())

    class PluginConfig:
        pass

    @registry.register_pipeline_member(PluginConfig)
    class PluginAlgorithm(BaseAlgorithm):
        pass

    assert registry.register_pipeline_member is registry.register_algorithm_impl
    assert registry.resolve_pipeline_member(PluginConfig()) is PluginAlgorithm


@pytest.mark.parametrize("kwargs", [{}, {"act_bits": 16}, {"act_group_size": -1}])
def test_validate_allows_unset_activation_fields(kwargs):
    scheme = QuantizationScheme(**kwargs)
    before = scheme.to_dict()
    scheme.validate()
    assert scheme.to_dict() == before


def test_validate_allows_empty_scheme_before_resolution():
    QuantizationScheme.empty().validate()


def test_double_quant_defaults_use_current_gguf_export_package():
    scheme = QuantizationScheme(bits=4, group_size=32, data_type="int_asym_dq")
    scheme.validate()
    assert scheme.super_bits == 6
    assert scheme.super_group_size == 8


@pytest.mark.parametrize("kwargs", [{"act_bits": 0}, {"act_group_size": -2}, {"act_group_size": (16, 16)}])
def test_validate_still_rejects_invalid_activation_fields(kwargs):
    with pytest.raises(ValueError):
        QuantizationScheme(**kwargs).validate()


@pytest.mark.parametrize("scheme", ["MXFP4", "NVFP4"])
def test_awq_tp_scheme_rejected_before_model_loading(scheme):
    from auto_round.compressors.base import BaseOrchestrator

    with (
        patch.object(ParallelConfig, "resolve_device_map", return_value="cpu"),
        patch.object(BaseOrchestrator, "__init__", side_effect=AssertionError("must fail before model setup")),
        pytest.raises(NotImplementedError, match="AWQ TP currently supports INT"),
    ):
        AutoRound(torch.nn.Linear(32, 32), scheme=scheme, alg_configs="awq", parallel_config={"tp_size": 2})
