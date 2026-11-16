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

from unittest.mock import Mock

import pytest

from auto_round import AdamRoundConfig, AutoRound
from auto_round.algorithms.config_resolver import _prepare_entry_kwargs
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.qat.config import QATConfig
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.schemes import QuantizationScheme


@pytest.mark.parametrize(
    "config_cls",
    [
        QuantizationConfig,
        SignRoundConfig,
        SignRoundV2Config,
        AdamRoundConfig,
        QADConfig,
        QATConfig,
        RTNConfig,
        OptimizedRTNConfig,
        AWQConfig,
    ],
)
@pytest.mark.parametrize("field", [*QuantizationScheme.get_attributes(), "scheme"])
def test_algorithm_configs_reject_scheme_arguments(config_cls, field):
    with pytest.raises(TypeError, match="AutoRound"):
        config_cls(**{field: None})


@pytest.mark.parametrize("config_cls", [SignRoundConfig, RTNConfig, AWQConfig])
def test_algorithm_configs_do_not_own_scheme_state(config_cls):
    config = config_cls()
    assert not hasattr(config, "scheme")
    for field in QuantizationScheme.get_attributes():
        assert not hasattr(config, field)


def test_entry_keeps_scheme_overrides_separate_from_algorithm_configs():
    config = SignRoundConfig(iters=2)
    configs, runtime = _prepare_entry_kwargs(config, {"bits": 4, "group_size": 16, "sym": True, "lr": 0.01})
    assert runtime["scheme_overrides"] == {"bits": 4, "group_size": 16, "sym": True}
    assert configs[0].lr == 0.01
    assert not hasattr(configs[0], "scheme")
    assert not hasattr(configs[0], "group_size")


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
    )


def test_entry_overrides_reach_real_quantization_without_polluting_configs():
    config = RTNConfig(disable_opt_rtn=True)
    compressor = AutoRound(
        _tiny_model(),
        tokenizer=Mock(),
        alg_configs=config,
        scheme="W4A16",
        group_size=16,
        sym=True,
        device_map="cpu",
        low_cpu_mem_usage=False,
    )
    model, _ = compressor.quantize()
    assert model is compressor.model
    assert compressor.scheme_context.group_size == 16
    assert compressor.scheme_context.sym is True
    assert model.model.layers[0].self_attn.q_proj.group_size == 16
    assert model.model.layers[0].self_attn.q_proj.bits == 4
    assert hasattr(model.model.layers[0].self_attn.q_proj, "scale")
    assert not hasattr(config, "scheme")
    assert not hasattr(config, "group_size")


def test_reusing_algorithm_config_does_not_share_schemes_between_runs():
    config = RTNConfig(disable_opt_rtn=True)
    runs = []
    for bits, group_size in [(4, 16), (8, 32)]:
        compressor = AutoRound(
            _tiny_model(),
            tokenizer=Mock(),
            alg_configs=config,
            bits=bits,
            group_size=group_size,
            device_map="cpu",
            low_cpu_mem_usage=False,
        )
        compressor.resolve_scheme()
        runs.append(compressor)
    assert [(run.scheme_context.bits, run.scheme_context.group_size) for run in runs] == [(4, 16), (8, 32)]
    assert not hasattr(config, "scheme")


def test_rotation_defaults_use_entry_group_size_override():
    from auto_round.algorithms.transforms.hadamard.config import RotationConfig

    rotation = RotationConfig(block_size=None)
    compressor = AutoRound(
        _tiny_model(),
        tokenizer=Mock(),
        alg_configs=[rotation, RTNConfig(disable_opt_rtn=True)],
        group_size=16,
        device_map="cpu",
        low_cpu_mem_usage=False,
    )
    assert compressor.rotation_configs[0].block_size == 16


@pytest.mark.parametrize("algorithm", ["signround", "awq"])
def test_data_driven_algorithms_use_bound_scheme(algorithm):
    import torch

    config = SignRoundConfig(iters=2) if algorithm == "signround" else AWQConfig(n_grid=1)
    data = torch.utils.data.DataLoader([torch.randint(0, 32, (8,)) for _ in range(2)], batch_size=1)
    compressor = AutoRound(
        _tiny_model(),
        tokenizer=Mock(model_max_length=2048, pad_token_id=0, eos_token_id=1),
        alg_configs=config,
        group_size=16,
        sym=True,
        dataset=data,
        nsamples=2,
        seqlen=8,
        batch_size=1,
        device_map="cpu",
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    model, _ = compressor.quantize()
    assert model.model.layers[0].self_attn.q_proj.group_size == 16
    assert model.model.layers[0].self_attn.q_proj.bits == 4
    assert hasattr(model.model.layers[0].self_attn.q_proj, "scale")
    assert not hasattr(config, "scheme")
    assert not hasattr(config, "bits")


@pytest.mark.parametrize("rotation,hadamard_type", [("default", "hadamard"), ("random_hadamard", "random_hadamard")])
@pytest.mark.parametrize("as_algorithm_config", [False, True])
def test_rotation_metadata_survives_entry_scheme_serialization(rotation, hadamard_type, as_algorithm_config):
    import json

    from auto_round.algorithms.transforms.hadamard.config import RotationConfig
    from auto_round.experimental.qmodules.mx import MXFP4QuantLinear

    config = RTNConfig(disable_opt_rtn=True)
    kwargs = {"rotation_config": rotation}
    if as_algorithm_config:
        config = [config, RotationConfig(hadamard_type=hadamard_type)]
        kwargs = {}
    compressor = AutoRound(
        _tiny_model(),
        tokenizer=Mock(),
        alg_configs=config,
        scheme="MXFP4",
        device_map="cpu",
        low_cpu_mem_usage=False,
        **kwargs,
    )
    compressor.resolve_scheme()
    restored = QuantizationScheme.from_dict(json.loads(json.dumps(compressor.scheme_context.to_dict())))
    assert restored.rotation_config["hadamard_type"] == hadamard_type
    assert restored.rotation_config["block_size"] == restored.group_size
    layer = MXFP4QuantLinear(32, 32, config=restored)
    assert hasattr(layer, "hadamard_matrix") == (hadamard_type == "random_hadamard")
