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
"""Tests for ``AR_NVFP_ACT_AMAX_ALPHA``, which scales the collected NVFP activation amax."""

from types import SimpleNamespace

import pytest
import torch

from auto_round import envs
from auto_round.algorithms.composer import AlgorithmComposer
from auto_round.algorithms.quantization.utils import register_act_max_hooks as register_alternate_act_max_hooks
from auto_round.data_type.utils import register_act_max_hooks
from auto_round.experimental.kv_cache import QuantizedKVParameterCache

ENV_NAME = "AR_NVFP_ACT_AMAX_ALPHA"


@pytest.fixture(autouse=True)
def _clear_alpha_env(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)


def test_default_alpha_is_one():
    assert envs.AR_NVFP_ACT_AMAX_ALPHA == 1.0


@pytest.mark.parametrize("raw", ["auto", "", "0", "-1", "nan", "inf", "abc"])
def test_invalid_alpha_raises(monkeypatch, raw):
    monkeypatch.setenv(ENV_NAME, raw)
    with pytest.raises(ValueError, match=ENV_NAME):
        _ = envs.AR_NVFP_ACT_AMAX_ALPHA


def _collect_act_max(act_data_type):
    layer = torch.nn.Linear(16, 4)
    layer.bits, layer.act_bits, layer.act_dynamic = 4, 4, False
    layer.data_type = layer.act_data_type = act_data_type
    layer.act_group_size = 16
    handles = register_act_max_hooks(layer)
    layer(torch.tensor([[1.0] * 15 + [-3.0]]))
    for handle in handles:
        handle.remove()
    return layer.act_max


@pytest.mark.parametrize("act_data_type", ["nv_fp4_with_static_gs", "nvfp4_v2_with_static_gs"])
def test_nvfp_act_max_is_scaled(monkeypatch, act_data_type):
    monkeypatch.setenv(ENV_NAME, "2")
    assert _collect_act_max(act_data_type).item() == pytest.approx(6.0)


def test_repeated_collection_does_not_compound(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "2")
    layer = torch.nn.Linear(16, 4)
    layer.bits, layer.act_bits, layer.act_dynamic = 4, 4, False
    layer.data_type = layer.act_data_type = "nv_fp4_with_static_gs"
    layer.act_group_size = 16
    handles = register_act_max_hooks(layer)
    for _ in range(3):
        layer(torch.tensor([[1.0] * 15 + [-3.0]]))
    for handle in handles:
        handle.remove()
    assert layer.act_max.item() == pytest.approx(6.0)


def _nvfp_identity_layer():
    layer = torch.nn.Identity()
    layer.bits, layer.act_bits, layer.act_dynamic = 4, 4, False
    layer.data_type = layer.act_data_type = "nv_fp4_with_static_gs"
    layer.act_group_size = 16
    return layer


def test_half_precision_act_max_scaling_stays_finite(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "2")
    layer = _nvfp_identity_layer()
    handles = register_act_max_hooks(layer)
    inputs = torch.full((1, 16), 60000.0, dtype=torch.float16)
    layer(inputs)
    layer(inputs)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(layer.act_max, torch.tensor([120000.0]))


def test_outside_block_act_max_scales_half_precision_inputs(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "2")
    layer = SimpleNamespace(
        act_group_size=16,
        group_size=16,
        act_data_type="nv_fp4_with_static_gs",
        data_type="nv_fp4_with_static_gs",
    )
    composer = object.__new__(AlgorithmComposer)
    inputs = [torch.full((1, 16), 60000.0, dtype=torch.float16)]
    composer._attach_act_max_for_outside_layer(layer, inputs, None)
    torch.testing.assert_close(layer.act_max, torch.tensor([120000.0]))


def test_alternate_act_max_hook_scales_half_precision_inputs(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "2")
    layer = _nvfp_identity_layer()
    quantizer = SimpleNamespace(act_group_size=16, layer_config={})
    handles = register_alternate_act_max_hooks(quantizer, layer)
    layer(torch.full((1, 16), 60000.0, dtype=torch.float16))
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(layer.act_max, torch.tensor([120000.0]))


def test_non_nvfp_act_max_is_not_scaled(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "2")
    assert _collect_act_max("mx_fp").max().item() == pytest.approx(3.0)


def test_kv_amax_is_not_scaled(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "2")
    QuantizedKVParameterCache._instance = None
    QuantizedKVParameterCache._initialized = False
    cache = QuantizedKVParameterCache(dtype="nvfp4")
    cache.update(torch.full((1, 1, 1, 16), 3.0), torch.full((1, 1, 1, 16), 5.0), layer_idx=0)
    assert cache.k_amax[0].item() == pytest.approx(3.0)
    assert cache.v_amax[0].item() == pytest.approx(5.0)
    cache.reset()
