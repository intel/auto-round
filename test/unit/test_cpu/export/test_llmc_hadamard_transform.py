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

import json
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.transform import TransformConfig
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from auto_round.algorithms.quantization.rtn.batched_search import _staged_weight
from auto_round.algorithms.transforms.hadamard.apply import apply_rotation_transform
from auto_round.algorithms.transforms.hadamard.config import RotationConfig
from auto_round.algorithms.transforms.teq.protocol import apply_teq_equivalent_weight_transform
from auto_round.export.formats.autoround.qlinear_fp import QuantLinear
from auto_round.export.formats.compressed_tensors import integer as llmc_export
from auto_round.export.formats.compressed_tensors.float import save_checkpoint as save_quantized_as_fp
from auto_round.export.formats.compressed_tensors.integer import save_checkpoint as save_quantized_as_llmcompressor
from auto_round.export.formats.compressed_tensors.transform import (
    _export_hadamard_transform_config,
)
from auto_round.wrapper import WrapperLinear


def _tiny_llama():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return LlamaForCausalLM(config)


def test_no_rotation_leaves_transform_config_absent():
    model = _tiny_llama()

    assert _export_hadamard_transform_config(model, "mx_fp") is None
    assert not hasattr(model, "transform_config")


def test_hadamard_transform_config_and_shared_weights(tmp_path):
    model = _tiny_llama()
    model._rotation_config = RotationConfig(backend="transform", block_size=32)

    config_dict = _export_hadamard_transform_config(model, "mx_fp")
    config = TransformConfig.model_validate(config_dict)
    scheme = config.config_groups["v"]

    assert scheme.type == "hadamard"
    assert scheme.head_dim == 32
    assert scheme.precision == torch.float64
    assert scheme.randomize is False
    assert [(args.location, args.inverse) for args in scheme.apply] == [
        ("input", False),
        ("weight_input", True),
    ]
    for args in scheme.apply:
        assert args.targets == ["Linear"]
        assert args.ignore == ["lm_head"]

    attention = model.model.layers[0].self_attn
    assert attention.q_proj.v_input.weight.data_ptr() == attention.k_proj.v_input.weight.data_ptr()
    assert not hasattr(model.lm_head, "v_input")

    model.save_pretrained(tmp_path)
    with safe_open(tmp_path / "model.safetensors", framework="pt") as f:
        keys = [key for key in f.keys() if key.endswith(".v_input.weight")]
        assert keys == ["model.layers.0.self_attn.q_proj.v_input.weight"]
        matrix = f.get_tensor(keys[0])

    assert matrix.dtype == torch.float64
    assert matrix.shape == (32, 32)
    assert matrix[0, 0].item() == 1.0


def test_llmcompressor_export_includes_transform_config(monkeypatch, tmp_path):
    model = _tiny_llama()
    model._rotation_config = RotationConfig(backend="transform", block_size=32)
    monkeypatch.setattr(llmc_export, "pack_layer", lambda name, model, device: None)

    class FakeQuantizationConfig:
        @classmethod
        def from_pretrained(cls, model, format=None):
            return SimpleNamespace(
                to_dict=lambda: {
                    "config_groups": {},
                    "format": "pack-quantized",
                }
            )

    monkeypatch.setattr(
        "compressed_tensors.quantization.QuantizationConfig",
        FakeQuantizationConfig,
    )
    save_quantized_as_llmcompressor(str(tmp_path), model=model)

    with open(tmp_path / "config.json") as f:
        nested_config = json.load(f)["quantization_config"]
    with open(tmp_path / "quantization_config.json") as f:
        direct_config = json.load(f)

    for quantization_config in (nested_config, direct_config):
        config = TransformConfig.model_validate(quantization_config["transform_config"])
        assert config.config_groups["v"].head_dim == 32


def test_random_hadamard_is_rejected():
    model = _tiny_llama()
    model._rotation_config = RotationConfig(
        backend="transform",
        hadamard_type="random_hadamard",
        block_size=32,
    )

    with pytest.raises(NotImplementedError, match="hadamard_type='hadamard'"):
        _export_hadamard_transform_config(model, "mx_fp")


def test_inplace_backend_is_rejected():
    model = _tiny_llama()
    model._rotation_config = RotationConfig(backend="inplace", block_size=32)

    with pytest.raises(NotImplementedError, match="per-Linear transform backend"):
        _export_hadamard_transform_config(model, "mx_fp")


def test_fp_export_immediate_saving_includes_transform_and_weights(monkeypatch, tmp_path):
    model = _tiny_llama()
    model._rotation_config = RotationConfig(backend="transform", block_size=32)
    monkeypatch.setattr(
        "auto_round.export.formats.compressed_tensors.float.is_immediate_saving_mode",
        lambda model, serialization_dict: True,
    )
    save_file(
        {"model.embed_tokens.weight": torch.zeros(1, 1)},
        tmp_path / "model.safetensors",
    )

    save_quantized_as_fp(
        str(tmp_path),
        model=model,
        layer_config={},
        serialization_dict={
            "bits": 4,
            "data_type": "mx_fp",
            "act_bits": 4,
            "act_data_type": "mx_fp",
            "regex_config": {},
        },
    )

    config = TransformConfig.model_validate(model.config.quantization_config["transform_config"])
    assert config.config_groups["v"].head_dim == 32

    with open(tmp_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    matrix_keys = [key for key in index["weight_map"] if key.endswith(".v_input.weight")]
    assert matrix_keys == ["model.layers.0.self_attn.q_proj.v_input.weight"]
    matrix_key = matrix_keys[0]
    assert index["weight_map"][matrix_key] == "transform.safetensors"
    assert index["weight_map"]["model.embed_tokens.weight"] == "model.safetensors"

    with safe_open(tmp_path / "transform.safetensors", framework="pt") as f:
        matrix = f.get_tensor(matrix_key)
    assert matrix.dtype == torch.float64
    assert matrix.shape == (32, 32)
    assert index["metadata"]["total_size"] > (tmp_path / "model.safetensors").stat().st_size


def test_packed_quantlinear_receives_online_transform():
    model = _tiny_llama()
    model._rotation_config = RotationConfig(backend="transform", block_size=32)
    attention = model.model.layers[0].self_attn
    original = attention.q_proj
    packed = QuantLinear(
        bits=4,
        group_size=32,
        infeatures=original.in_features,
        outfeatures=original.out_features,
        bias=False,
        data_type="mx_fp",
        act_bits=4,
        act_data_type="mx_fp",
    )
    attention.q_proj = packed

    _export_hadamard_transform_config(model, "mx_fp")

    assert hasattr(packed, "v_input")
    assert packed.v_input.weight.shape == (32, 32)


def test_batched_rtn_staging_applies_weight_rotation():
    layer = torch.nn.Linear(64, 4, bias=False)
    layer.bits = 4
    layer.group_size = 32
    layer.sym = True
    layer.data_type = "mx_fp"
    layer.act_bits = 16
    layer.act_data_type = "mx_fp"
    layer.act_group_size = 32
    layer.act_sym = True
    layer.scale_dtype = torch.uint8
    model = torch.nn.Sequential(layer)
    original = layer.weight.detach().clone()
    apply_rotation_transform(model, "hadamard", use_tqdm=False)
    wrapper = WrapperLinear(layer, enable_round_tuning=False, disable_opt_rtn=True, device="cpu")

    staged = _staged_weight(wrapper)
    expected = apply_teq_equivalent_weight_transform(layer, original)

    assert torch.allclose(staged, expected)
    assert not torch.allclose(staged, original)
