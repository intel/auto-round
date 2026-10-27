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
import torch

from auto_round.export.formats.autoround.format import AutoRoundFormat
from auto_round.export.formats.compressed_tensors.format import LLMCompressorFormat


@pytest.mark.parametrize(
    "format_class,backend,module_name",
    [
        (AutoRoundFormat, "auto_round:fp8_static", "auto_round.export.formats.autoround.fp8"),
        (AutoRoundFormat, "auto_round:mx_fp", "auto_round.export.formats.autoround.float"),
        (LLMCompressorFormat, "llm_compressor:fp8_static", "auto_round.export.formats.compressed_tensors.static_fp"),
        (LLMCompressorFormat, "llm_compressor:mx_fp", "auto_round.export.formats.compressed_tensors.float"),
        (LLMCompressorFormat, "llm_compressor:int8_w8a8", "auto_round.export.formats.compressed_tensors.integer"),
    ],
)
def test_pack_and_save_use_selected_implementation(monkeypatch, format_class, backend, module_name):
    import importlib

    module = importlib.import_module(module_name)
    pack = Mock()
    save = Mock(return_value="saved")
    monkeypatch.setattr(module, "pack_layer", pack)
    monkeypatch.setattr(module, "save_checkpoint", save)
    fmt = object.__new__(format_class)
    fmt.output_format = backend
    fmt.backend = None
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    fmt.pack_layer("0", model, device="cpu")
    assert pack.call_count == 1
    assert fmt.save_quantized_as_format(None, model=model, serialization_dict={}) == "saved"
    selected_pack = save.call_args.kwargs["_pack_layer"]
    selected_pack("0", model, device="cpu")
    assert pack.call_count == 2


def test_autoround_fp8_metadata_uses_serialization(monkeypatch):
    from auto_round.export.formats.autoround import fp8

    save = Mock()
    monkeypatch.setattr(fp8, "save_checkpoint", save)
    fmt = object.__new__(AutoRoundFormat)
    fmt.output_format = "auto_round:fp8"
    fmt.backend = None
    fmt.save_quantized_as_format(None, serialization_dict={"data_type": "fp", "bits": 8, "act_bits": 8})
    assert save.call_args.kwargs["backend"] == "auto_round:fp8_static"


@pytest.mark.parametrize("output", [False, True])
def test_fp8_payload_and_checkpoint_preserve_copy(monkeypatch, tmp_path, output):
    from dataclasses import asdict
    from types import SimpleNamespace

    from auto_round.export.formats.autoround import fp8
    from auto_round.schemes import PRESET_SCHEMES

    scheme = asdict(PRESET_SCHEMES["FPW8A16"])
    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False), torch.nn.Linear(2, 2, bias=False))
    model.dtype = torch.float32
    model.config = SimpleNamespace()
    layer = model[0]
    for name, value in scheme.items():
        setattr(layer, name, value)
    layer.group_size = -1
    layer.scale = torch.tensor([[0.5], [2.0]])
    layer.zp = None
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, -1.0, -2.0], [4.0, 8.0, -4.0, -8.0]]))
    original_weight = layer.weight.detach().clone()
    model[1].bits = model[1].act_bits = 16
    full_precision_weight = model[1].weight.detach().clone()
    captured = {}

    def capture_save(packed, directory, **kwargs):
        captured["state"] = packed.state_dict()
        captured["config"] = packed.config.quantization_config

    monkeypatch.setattr(fp8, "save_model", capture_save)
    serialization = {**scheme, "to_quant_block_names": None, "regex_config": None}
    fmt = object.__new__(AutoRoundFormat)
    fmt.output_format = "auto_round:fp8"
    fmt.backend = None
    result = fmt.save_quantized_as_format(
        str(tmp_path) if output else None,
        model=model,
        layer_config={
            "0": {**scheme, "group_size": -1, "in_blocks": True},
            "1": {**scheme, "bits": 16, "act_bits": 16, "in_blocks": True},
        },
        serialization_dict=serialization,
        inplace=False,
    )
    assert result is not model
    assert isinstance(model[0], torch.nn.Linear)
    torch.testing.assert_close(model[0].weight, original_weight)
    assert isinstance(result[0], fp8.FP8QLinear)
    assert isinstance(result[1], torch.nn.Linear)
    torch.testing.assert_close(result[1].weight, full_precision_weight)
    assert result.config.quantization_config["extra_config"]["1"]["bits"] == 16
    expected = original_weight / torch.tensor([[0.5], [2.0]])
    torch.testing.assert_close(result[0].weight.float(), expected)
    assert result.config.quantization_config["quant_method"] == "auto-round"
    assert result.config.quantization_config["fmt"] == "e4m3"
    if output:
        assert captured["state"]["0.weight"].dtype == torch.float8_e4m3fn
        assert captured["config"] == result.config.quantization_config
    else:
        assert not captured


def test_autoround_e5m3_pairs_ct_packing_with_native_metadata(monkeypatch):
    from auto_round.export.formats.autoround import float as native_float
    from auto_round.export.formats.compressed_tensors import float as compressed_float

    pack = Mock()
    save = Mock()
    monkeypatch.setattr(compressed_float, "pack_layer", pack)
    monkeypatch.setattr(native_float, "save_checkpoint", save)
    fmt = object.__new__(AutoRoundFormat)
    fmt.output_format = "auto_round:nvfp4_v2"
    fmt.backend = None
    fmt.pack_layer("linear", None)
    fmt.save_quantized_as_format(None, serialization_dict={})
    assert save.call_args.kwargs["_pack_layer"] is pack
    assert save.call_args.kwargs["backend"] == "auto_round:llm_compressor_nvfp4_e5m3"
