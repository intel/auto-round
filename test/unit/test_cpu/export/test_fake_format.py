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

import auto_round.experimental.qmodules.fake as fake_qmodule
from auto_round.experimental.qmodules.fake import FakeActQuantLinear
from auto_round.export.formats.backends.fake import (
    _normalize_state_dict_keys,
    _rewrite_saved_weights_without_orig_layer,
)
from auto_round.formats import FakeFormat
from auto_round.inference.backend import get_layer_backend
from auto_round.inference.convert_model import convert_hf_model
from auto_round.schemes import PRESET_SCHEMES
from auto_round.wrapper import WrapperWALayer


def _assert_has_act_hook(layer):
    assert isinstance(layer, FakeActQuantLinear)
    assert hasattr(layer, "qdq_input")
    assert callable(layer.qdq_input)


def test_fake_nvfp4_qdq_uses_saved_input_global_scale(monkeypatch):
    captured_kwargs = {}

    def quant_func(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs["tensor"], None, None

    monkeypatch.setattr(fake_qmodule, "get_quant_func", lambda **kwargs: (quant_func, None))
    layer = FakeActQuantLinear(16, 4, PRESET_SCHEMES["NVFP4"], dtype=torch.float32)
    layer.input_global_scale.fill_(1.5)

    layer.qdq_input(torch.randn(2, 16))

    assert captured_kwargs["global_scale"] is layer.input_global_scale


def test_fake_evaluation_wrapper_uses_saved_input_global_scale():
    captured_kwargs = {}
    linear = torch.nn.Linear(16, 4)
    linear.act_max_scale = torch.ones(1)
    linear.act_min_scale = torch.ones(1)
    linear.input_global_scale = torch.tensor([1.5], dtype=torch.float32)

    class ActivationQuantizer:
        def qdq(self, activation, **kwargs):
            captured_kwargs.update(kwargs)
            return activation

    wrapper = WrapperWALayer(linear, enable_torch_compile=False, activation_quantizer=ActivationQuantizer())

    wrapper(torch.randn(2, 16))

    assert captured_kwargs["global_scale"] is linear.input_global_scale


def test_fake_evaluation_wrapper_does_not_pass_global_scale_to_other_quantizers():
    linear = torch.nn.Linear(16, 4)
    linear.act_max_scale = torch.ones(1)
    linear.act_min_scale = torch.ones(1)

    class ActivationQuantizer:
        def qdq(self, activation, *, observed_max=None, min_scale=1.0, max_scale=1.0):
            return activation

    wrapper = WrapperWALayer(linear, enable_torch_compile=False, activation_quantizer=ActivationQuantizer())

    wrapper(torch.randn(2, 16))


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
        config = {}
        if hasattr(self.config, "quantization_config"):
            config["quantization_config"] = self.config.quantization_config
        with open(os.path.join(output_dir, "config.json"), "w") as config_file:
            json.dump(config, config_file)


def test_fake_format_unwraps_quantized_layers_before_save(tmp_path):
    model = _SaveableModel()
    model.linear.orig_layer.act_data_type = "nv_fp4_with_static_gs"
    model.linear.orig_layer.input_global_scale = torch.tensor([1.5], dtype=torch.float32)
    expected_weight = model.linear.orig_layer.weight.detach().clone()
    output_dir = str(tmp_path / "fake_model")

    saved_model = FakeFormat("fake", PRESET_SCHEMES["NVFP4"], SimpleNamespace(mllm=False)).save_quantized(
        output_dir=output_dir,
        model=model,
        inplace=False,
        serialization_dict={
            "bits": 4,
            "group_size": 16,
            "sym": True,
            "data_type": "nv_fp",
            "act_bits": 4,
            "act_group_size": 16,
            "act_sym": True,
            "act_data_type": "nv_fp4_with_static_gs",
            "to_quant_block_names": ["block"],
            "supported_types": [torch.nn.Linear],
        },
    )

    state_dict = torch.load(os.path.join(output_dir, "pytorch_model.bin"), weights_only=True)
    assert set(state_dict) == {"linear.weight", "linear.bias", "linear.act_max_scale", "linear.input_global_scale"}
    assert torch.equal(state_dict["linear.weight"], expected_weight)
    assert torch.equal(state_dict["linear.input_global_scale"], torch.tensor([1.5], dtype=torch.float32))
    # Save-time should keep in-memory wrappers unchanged; replacement happens on load.
    assert hasattr(saved_model.linear, "orig_layer")
    with open(os.path.join(output_dir, "config.json")) as config_file:
        quantization_config = json.load(config_file)["quantization_config"]
    assert "supported_types" not in quantization_config
    assert quantization_config["packing_format"] == "auto_round:fake"
    assert quantization_config["quant_method"] == "auto-round"
    assert quantization_config["act_bits"] == 4
    assert quantization_config["block_name_to_quantize"] == ["block"]

    loaded_model = _TinyLoadModel(SimpleNamespace(**quantization_config))
    loaded_model, used_backends = convert_hf_model(loaded_model, target_device="cpu")
    loaded_model.block.linear.load_state_dict(
        {"input_global_scale": state_dict["linear.input_global_scale"]}, strict=False
    )
    assert used_backends == ["auto_round:fake"]
    _assert_has_act_hook(loaded_model.block.linear)
    assert torch.equal(loaded_model.block.linear.input_global_scale, torch.tensor([1.5], dtype=torch.float32))
    roundtrip_activation = torch.randn(2, 3, 16)
    assert not torch.equal(loaded_model.block.linear.qdq_input(roundtrip_activation), roundtrip_activation)


def test_fake_format_normalizes_sharded_safetensors_index(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    shard_name = "model-00001-of-00001.safetensors"
    original_key = "model.layers.0.orig_layer.weight"
    normalized_key = "model.layers.0.weight"
    save_file({original_key: torch.ones(2, 2)}, tmp_path / shard_name)
    with open(tmp_path / "model.safetensors.index.json", "w") as index_file:
        json.dump({"metadata": {}, "weight_map": {original_key: shard_name}}, index_file)

    _rewrite_saved_weights_without_orig_layer(str(tmp_path))

    with safe_open(tmp_path / shard_name, framework="pt", device="cpu") as shard:
        assert list(shard.keys()) == [normalized_key]
    with open(tmp_path / "model.safetensors.index.json") as index_file:
        assert json.load(index_file)["weight_map"] == {normalized_key: shard_name}


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


def test_fake_format_omits_woq_quantization_config(tmp_path):
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
        config = json.load(config_file)

    assert "quantization_config" not in config


def test_fake_format_still_saves_when_env_disabled(tmp_path):
    model = _SaveableModel()
    output_dir = str(tmp_path / "save_model_even_when_env_disabled")

    returned = FakeFormat("fake", PRESET_SCHEMES["NVFP4_E5M3"], SimpleNamespace(mllm=False)).save_quantized(
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
    assert returned is not None
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "config.json"))


def test_fake_format_meta_device_applies_post_save_source_fixes(tmp_path, monkeypatch):
    model = _SaveableModel()
    output_dir = str(tmp_path / "meta_device_model")
    fixup_calls = []

    monkeypatch.setattr("auto_round.export.formats.backends.fake.unsupported_meta_device", lambda model: True)
    monkeypatch.setattr("auto_round.export.utils.save_config_artifact", lambda model, save_dir: None)
    monkeypatch.setattr(
        "auto_round.export.utils.apply_post_save_source_fixes",
        lambda saved_model, save_dir: fixup_calls.append((saved_model, save_dir)),
    )

    FakeFormat("fake", PRESET_SCHEMES["INT4"], SimpleNamespace(mllm=False)).save_quantized(
        output_dir=output_dir,
        model=model,
    )

    assert fixup_calls == [(model, output_dir)]


class _SaveableSafetensorsModel(torch.nn.Module):
    """Minimal model whose ``save_pretrained`` writes real safetensors, so the
    fp32-restore / MTP-copy post-processing in ``FakeFormat.save_quantized`` has
    something to act on."""

    def __init__(self, source_dir: str, norm_weight: torch.Tensor):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.config = SimpleNamespace(_name_or_path=source_dir)
        self.name_or_path = source_dir
        self._norm_weight = norm_weight

    def save_pretrained(self, output_dir):
        from safetensors.torch import save_file

        os.makedirs(output_dir, exist_ok=True)
        # Simulate transformers downcasting the FP32 tensor to BF16 on save.
        save_file(
            {
                "model.language_model.layers.0.linear_attn.norm.weight": self._norm_weight.to(torch.bfloat16),
                "model.embed_tokens.weight": torch.randn(4, 4),
            },
            os.path.join(output_dir, "model.safetensors"),
        )
        with open(os.path.join(output_dir, "config.json"), "w") as config_file:
            json.dump({}, config_file)


def test_fake_format_restores_fp32_and_copies_mtp_tensors(tmp_path, monkeypatch):
    """Qwen/Qwen3.5-0.8B-style checkpoints keep ``linear_attn.norm.weight`` in FP32
    and MTP tensors that transformers does not load. The fake export path must
    restore/copy both by default, not just the ``auto_round`` save path."""
    from safetensors.torch import save_file

    import auto_round.envs as envs

    source_dir = str(tmp_path / "source")
    os.makedirs(source_dir)
    norm_weight = torch.tensor([1.0001, -2.0002, 3.0003, 4.0004], dtype=torch.float32)
    mtp_weight = torch.randn(8, 4)
    save_file(
        {
            "model.language_model.layers.0.linear_attn.norm.weight": norm_weight,
            "model.language_model.mtp.0.fc.weight": mtp_weight,
        },
        os.path.join(source_dir, "model.safetensors"),
    )
    monkeypatch.setattr(envs, "AR_DISABLE_COPY_MTP_WEIGHTS", False)

    model = _SaveableSafetensorsModel(source_dir, norm_weight)
    output_dir = str(tmp_path / "fake_qwen35")

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

    from safetensors import safe_open

    with safe_open(os.path.join(output_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        restored = f.get_tensor("model.language_model.layers.0.linear_attn.norm.weight")
    assert restored.dtype == torch.float32
    assert torch.equal(restored, norm_weight)

    with safe_open(os.path.join(output_dir, "model_extra_tensors.safetensors"), framework="pt", device="cpu") as f:
        assert torch.equal(f.get_tensor("model.language_model.mtp.0.fc.weight"), mtp_weight)


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


def test_fake_backend_fallback_for_woq_w4a8_dynamic():
    """WOQ packing with unsupported act scheme should gracefully fallback to fake backend."""
    layer_backend = get_layer_backend(
        "cpu",
        "auto",
        "auto_round:auto_gptq",
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "data_type": "int",
            "act_bits": 8,
            "act_group_size": 32,
            "act_sym": True,
            "act_data_type": "int",
            "act_dynamic": True,
        },
        3072,
        768,
    )

    assert layer_backend == "auto_round:fake"


def test_fake_format_keeps_in_memory_wrapper_structure_on_save(tmp_path):
    model = _SaveableModel()
    output_dir = str(tmp_path / "fake_keep_wrapper_model")

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
        },
    )

    assert hasattr(saved_model.linear, "orig_layer")
    assert not isinstance(saved_model.linear, FakeActQuantLinear)


def test_normalize_state_dict_keys_strips_orig_layer_segment():
    state = {
        "model.layers.0.fc1.orig_layer.weight": torch.ones(1),
        "model.layers.0.fc1.orig_layer.bias": torch.zeros(1),
        "model.layers.0.fc2.weight": torch.randn(1),
    }

    normalized, changed = _normalize_state_dict_keys(state)

    assert changed is True
    assert "model.layers.0.fc1.weight" in normalized
    assert "model.layers.0.fc1.bias" in normalized
    assert "model.layers.0.fc1.orig_layer.weight" not in normalized
    assert "model.layers.0.fc2.weight" in normalized


def test_rewrite_saved_weights_without_orig_layer_for_safetensors(tmp_path):
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

    ckpt_dir = tmp_path / "fake_ckpt"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "model.safetensors"

    safe_save_file(
        {
            "model.layers.0.fc1.orig_layer.weight": torch.randn(2, 2),
            "model.layers.0.fc1.orig_layer.bias": torch.randn(2),
            "model.layers.0.fc2.weight": torch.randn(2, 2),
        },
        str(ckpt_path),
    )

    _rewrite_saved_weights_without_orig_layer(str(ckpt_dir))

    rewritten = safe_load_file(str(ckpt_path))
    assert "model.layers.0.fc1.orig_layer.weight" not in rewritten
    assert "model.layers.0.fc1.weight" in rewritten
    assert "model.layers.0.fc1.bias" in rewritten
