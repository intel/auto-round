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

"""Exercise MLX's unified save interface with a local tiny model."""

import json
from types import SimpleNamespace

import pytest
from transformers import OPTConfig, OPTForCausalLM

from auto_round.compressors.config_resolution import resolve_scheme_value
from auto_round.export.formats.mlx.format import MLXFormat


def test_mlx_save_uses_format_packing_and_writes_metadata(tmp_path):
    model = OPTForCausalLM(
        OPTConfig(
            vocab_size=128,
            hidden_size=64,
            ffn_dim=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
    )
    fmt = MLXFormat("mlx", resolve_scheme_value("W4A16", {}).value, SimpleNamespace(mllm=False))
    packed = []

    def pack(layer_name, model, device=None, **kwargs):
        packed.append(layer_name)

    fmt.pack_layer = pack
    saved = fmt.save_quantized_as_format(
        str(tmp_path),
        model=model,
        layer_config={"model.decoder.layers.0.fc1": {}},
        serialization_dict={"bits": 4, "group_size": 64},
    )
    assert saved is model
    assert packed == ["model.decoder.layers.0.fc1"]
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["quantization"]["bits"] == 4
    assert config["quantization"]["group_size"] == 64
    assert list(tmp_path.glob("*.safetensors"))


@pytest.mark.parametrize("format_name", ["mlx", "fake"])
def test_format_copy_does_not_move_input_model(tmp_path, monkeypatch, format_name):
    import torch

    model = OPTForCausalLM(
        OPTConfig(
            vocab_size=128,
            hidden_size=64,
            ffn_dim=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
        )
    )
    from auto_round.export import resolve_formats

    (fmt,) = resolve_formats(resolve_scheme_value("W4A16", {}), format=format_name).formats
    moved_original = []
    original_to = torch.nn.Module.to

    def record_to(module, *args, **kwargs):
        if module is model:
            moved_original.append(True)
        return original_to(module, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", record_to)
    saved = fmt.save_quantized_as_format(str(tmp_path), model=model, inplace=False)
    assert saved is not model
    assert not moved_original
