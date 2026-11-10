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

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound, RTNConfig, SignRoundConfig
from auto_round.utils import get_block_names


def test_post_init_builds_authoritative_compression_plan(tiny_opt_model_path):
    compressor = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        alg_configs=RTNConfig(),
        format="auto_round",
        nsamples=1,
        seqlen=8,
        dataset=["local calibration sample"],
        low_cpu_mem_usage=False,
    )

    compressor.post_init()

    assert compressor._format_resolution.scheme.preset_name == compressor.compression_plan.scheme.preset_name
    assert compressor._format_resolution.formats == compressor.compression_plan.formats
    assert compressor.compression_plan.scheme.value == compressor.scheme_context
    assert compressor.compression_plan.formats == tuple(compressor.formats)
    assert dict(compressor.compression_plan.layer_config) == compressor.layer_config
    assert dict(compressor.compression_plan.regex_config) == compressor.regex_config
    assert compressor.compression_plan.has_qlayer_outside_block == compressor.has_qlayer_outside_block


def test_string_format_plan_keeps_selected_quant_blocks(tiny_opt_model_path):
    compressor = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        alg_configs=RTNConfig(),
        format="auto_round",
        to_quant_block_names=r"model\.decoder\.layers\.0$",
        nsamples=1,
        seqlen=8,
        dataset=["local calibration sample"],
        low_cpu_mem_usage=False,
    )

    compressor.post_init()

    expected = (("model.decoder.layers.0",),)
    assert compressor._format_resolution.quant_block_list == expected
    assert compressor.compression_plan.quant_block_list == expected
    assert compressor.quant_block_list == [["model.decoder.layers.0"]]


def test_selected_blocks_control_cache_targets_with_meta_nontext_tower(tiny_opt_model_path, monkeypatch):
    model = AutoModelForCausalLM.from_pretrained(tiny_opt_model_path, device_map="cpu")
    model.sound_encoder = torch.nn.Module()
    model.sound_encoder.encoder = torch.nn.Module()
    model.sound_encoder.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8, device="meta") for _ in range(2)])
    monkeypatch.setattr("auto_round.context.model.unsupported_meta_device", lambda _: False)

    compressor = AutoRound(
        model,
        AutoTokenizer.from_pretrained(tiny_opt_model_path),
        scheme="W4A16",
        format="auto_round",
        to_quant_block_names=r"model\.decoder\.layers\.0$",
        device_map="cpu",
        iters=1,
        nsamples=1,
        seqlen=8,
        dataset=["local calibration sample"],
        low_cpu_mem_usage=False,
        enable_torch_compile=False,
    )
    compressor.post_init()

    assert get_block_names(model) == [
        ["model.decoder.layers.0", "model.decoder.layers.1"],
        ["sound_encoder.encoder.layers.0", "sound_encoder.encoder.layers.1"],
    ]
    assert model.sound_encoder.encoder.layers[0].weight.device.type == "meta"
    assert compressor.compression_plan.quant_block_list == (("model.decoder.layers.0",),)

    class CacheReached(Exception):
        pass

    def capture_cache_targets(block_names, nsamples, layer_names=None, last_cache_name=None):
        assert block_names == ["model.decoder.layers.0"]
        assert nsamples == 1
        assert last_cache_name is None
        raise CacheReached

    compressor.cache_data = capture_cache_targets
    with pytest.raises(CacheReached):
        compressor._quantize_data_driven()


def test_legacy_state_views_cannot_mutate_authoritative_plan(tiny_opt_model_path):
    compressor = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        alg_configs=RTNConfig(),
        format="auto_round",
        nsamples=1,
        seqlen=8,
        dataset=["local calibration sample"],
        low_cpu_mem_usage=False,
    )
    compressor.post_init()

    compressor.scheme_context.bits = 2
    compressor.formats.clear()
    compressor.layer_config.clear()
    compressor.regex_config.clear()

    assert compressor.scheme_context.bits == compressor.compression_plan.scheme.value.bits == 4
    assert compressor.formats == list(compressor.compression_plan.formats)
    assert compressor.layer_config == {
        name: dict(config) for name, config in compressor.compression_plan.layer_config.items()
    }
    assert compressor.regex_config == {
        name: dict(config) for name, config in compressor.compression_plan.regex_config.items()
    }


def test_direct_and_algorithm_config_entries_build_equivalent_plans(tiny_opt_model_path):
    common = {
        "model": tiny_opt_model_path,
        "scheme": "W4A16",
        "iters": 0,
        "nsamples": 1,
        "seqlen": 8,
        "dataset": ["local calibration sample"],
        "low_cpu_mem_usage": False,
    }
    direct = AutoRound(bits=4, group_size=128, **common)
    configured = AutoRound(alg_configs=SignRoundConfig(bits=4, group_size=128, iters=0), **common)

    direct.post_init()
    configured.post_init()

    assert direct.compression_plan.scheme.value == configured.compression_plan.scheme.value
    assert direct.compression_plan.layer_config == configured.compression_plan.layer_config
