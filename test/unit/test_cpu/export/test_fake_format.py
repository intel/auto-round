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
import os
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, OPTConfig, OPTForCausalLM

from auto_round.experimental.qmodules.fake import FakeActQuantLinear
from auto_round.formats import FakeFormat
from auto_round.inference.backend import get_layer_backend
from auto_round.inference.convert_model import convert_hf_model
from auto_round.schemes import PRESET_SCHEMES
from auto_round.wrapper import WrapperWALayer


def _assert_has_act_hook(layer):
    assert isinstance(layer, FakeActQuantLinear)
    assert hasattr(layer, "qdq_input")
    assert callable(layer.qdq_input)


class _WrappedLinear(WrapperWALayer):
    def __init__(self, linear):
        torch.nn.Module.__init__(self)
        self.orig_layer = linear
        self.register_buffer("act_max_scale", torch.ones(1))

    def forward(self, inputs):
        return self.orig_layer(inputs)


class _SaveableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        linear = torch.nn.Linear(4, 3)
        linear.register_parameter("act_max_scale", torch.nn.Parameter(torch.ones(1)))
        self.linear = _WrappedLinear(linear)
        self.config = SimpleNamespace()

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        with open(os.path.join(output_dir, "config.json"), "w") as config_file:
            json.dump({"quantization_config": self.config.quantization_config}, config_file)


def test_fake_format_unwraps_quantized_layers_before_save(tmp_path):
    model = _SaveableModel()
    expected_weight = model.linear.orig_layer.weight.detach().clone()
    output_dir = str(tmp_path / "fake_model")

    saved_model = FakeFormat("fake", PRESET_SCHEMES["NVFP4_E5M3"], SimpleNamespace(mllm=False)).save_quantized(
        output_dir=output_dir,
        model=model,
        inplace=False,
        serialization_dict={
            "bits": 4,
            "group_size": 16,
            "sym": True,
            "data_type": "nvfp4_v2",
            "act_bits": 4,
            "act_group_size": 16,
            "act_sym": True,
            "act_data_type": "nvfp4_v2",
            "to_quant_block_names": ["block"],
            "supported_types": [torch.nn.Linear],
        },
    )

    state_dict = torch.load(os.path.join(output_dir, "pytorch_model.bin"), weights_only=True)
    assert set(state_dict) == {"linear.weight", "linear.bias"}
    assert torch.equal(state_dict["linear.weight"], expected_weight)
    assert isinstance(saved_model.linear, FakeActQuantLinear)
    assert not hasattr(saved_model.linear, "orig_layer")
    activation = torch.randn(2, 4)
    assert not torch.equal(saved_model.linear.qdq_input(activation), activation)
    with open(os.path.join(output_dir, "config.json")) as config_file:
        quantization_config = json.load(config_file)["quantization_config"]
    assert "supported_types" not in quantization_config
    assert quantization_config["packing_format"] == "auto_round:fake"
    assert quantization_config["quant_method"] == "auto-round"
    assert quantization_config["act_bits"] == 4
    assert quantization_config["block_name_to_quantize"] == ["block"]

    loaded_model = _TinyLoadModel(SimpleNamespace(**quantization_config))
    loaded_model, used_backends = convert_hf_model(loaded_model, target_device="cpu")
    assert used_backends == ["auto_round:fake"]
    _assert_has_act_hook(loaded_model.block.linear)
    roundtrip_activation = torch.randn(2, 3, 16)
    assert not torch.equal(loaded_model.block.linear.qdq_input(roundtrip_activation), roundtrip_activation)


class _TinyLoadModel(torch.nn.Module):
    def __init__(self, quantization_config):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.linear = torch.nn.Linear(16, 4)
        self.lm_head = torch.nn.Linear(16, 4)
        self.config = SimpleNamespace(quantization_config=quantization_config)


def test_fake_config_replaces_linear_and_qdq_activation_on_load():
    quantization_config = SimpleNamespace(
        bits=4,
        group_size=16,
        sym=True,
        data_type="nvfp4_v2",
        act_bits=4,
        act_group_size=16,
        act_sym=True,
        act_data_type="nvfp4_v2",
        act_dynamic=True,
        quant_method="auto-round",
        packing_format="auto_round:fake",
        block_name_to_quantize="block",
        backend="auto",
        extra_config={},
        modules_to_not_convert=[],
    )
    model = _TinyLoadModel(quantization_config)
    original_weight = model.block.linear.weight.detach().clone()
    activation = torch.randn(2, 3, 16)

    model, used_backends = convert_hf_model(model, target_device="cpu")

    assert used_backends == ["auto_round:fake"]
    _assert_has_act_hook(model.block.linear)
    assert torch.equal(model.block.linear.weight, original_weight)
    qdq_activation = model.block.linear.qdq_input(activation)
    assert not torch.equal(qdq_activation, activation)
    expected = torch.nn.functional.linear(qdq_activation, model.block.linear.weight, model.block.linear.bias)
    assert torch.equal(model.block.linear(activation), expected)


def test_fake_config_keeps_modules_to_not_convert_in_full_precision():
    quantization_config = SimpleNamespace(
        bits=4,
        group_size=16,
        sym=True,
        data_type="nvfp4_v2",
        act_bits=4,
        act_group_size=16,
        act_sym=True,
        act_data_type="nvfp4_v2",
        act_dynamic=True,
        quant_method="auto-round",
        packing_format="auto_round:fake",
        block_name_to_quantize="",
        backend="auto",
        extra_config={},
        modules_to_not_convert=["lm_head"],
    )
    model = _TinyLoadModel(quantization_config)
    original_lm_head = model.lm_head

    model, used_backends = convert_hf_model(model, target_device="cpu")

    assert used_backends == ["auto_round:fake"]
    assert isinstance(model.block.linear, FakeActQuantLinear)
    assert model.lm_head is original_lm_head


def test_transformers_load_replaces_fake_linear(tmp_path):
    config = OPTConfig(
        vocab_size=32,
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
        word_embed_proj_dim=16,
    )
    config.quantization_config = {
        "bits": 4,
        "group_size": 16,
        "sym": True,
        "data_type": "nvfp4_v2",
        "act_bits": 4,
        "act_group_size": 16,
        "act_sym": True,
        "act_data_type": "nvfp4_v2",
        "act_dynamic": True,
        "quant_method": "auto-round",
        "packing_format": "auto_round:fake",
        "block_name_to_quantize": "model.decoder.layers",
    }
    model_dir = str(tmp_path / "fake_opt")
    OPTForCausalLM(config).save_pretrained(model_dir)

    loaded_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu")

    q_proj = loaded_model.model.decoder.layers[0].self_attn.q_proj
    _assert_has_act_hook(q_proj)
    activation = torch.randn(1, 2, 16)
    assert not torch.equal(q_proj.qdq_input(activation), activation)


def test_fake_format_keeps_woq_packing_format(tmp_path):
    model = _SaveableModel()
    output_dir = str(tmp_path / "woq_model")

    FakeFormat("fake", PRESET_SCHEMES["INT4"], SimpleNamespace(mllm=False)).save_quantized(
        output_dir=output_dir,
        model=model,
        inplace=False,
        serialization_dict={
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "data_type": "int",
            "quant_method": "auto-round",
            "packing_format": "auto_round:auto_gptq",
            "to_quant_block_names": ["block"],
            "supported_types": [torch.nn.Linear],
        },
    )

    with open(os.path.join(output_dir, "config.json")) as config_file:
        quantization_config = json.load(config_file)["quantization_config"]

    assert quantization_config["packing_format"] == "auto_round:auto_gptq"
    assert "act_bits" not in quantization_config


def test_fake_backend_accepts_mxfp_roundtrip_config():
    layer_backend = get_layer_backend(
        "cpu",
        "auto",
        "auto_round:fake",
        {
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "data_type": "mx_fp",
            "act_bits": 4,
            "act_group_size": 32,
            "act_sym": True,
            "act_data_type": "mx_fp",
            "act_dynamic": True,
        },
        32,
        32,
    )

    assert layer_backend == "auto_round:fake"


def test_fake_backend_accepts_mxfp8_roundtrip_config():
    layer_backend = get_layer_backend(
        "cpu",
        "auto",
        "auto_round:fake",
        {
            "bits": 8,
            "group_size": 32,
            "sym": True,
            "data_type": "mx_fp",
            "act_bits": 8,
            "act_group_size": 32,
            "act_sym": True,
            "act_data_type": "mx_fp",
            "act_dynamic": True,
        },
        32,
        32,
    )

    assert layer_backend == "auto_round:fake"


def test_fake_backend_accepts_nvfp_roundtrip_config():
    layer_backend = get_layer_backend(
        "cpu",
        "auto",
        "auto_round:fake",
        {
            "bits": 4,
            "group_size": 16,
            "sym": True,
            "data_type": "nv_fp",
            "act_bits": 4,
            "act_group_size": 16,
            "act_sym": True,
            "act_data_type": "nv_fp",
            "act_dynamic": True,
        },
        16,
        16,
    )

    assert layer_backend == "auto_round:fake"
