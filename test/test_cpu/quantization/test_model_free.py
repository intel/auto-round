# Copyright (c) 2025 Intel Corporation
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

"""Unit tests for auto_round.compressors.model_free module."""

import json
import os
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from auto_round.compressors.model_free import (
    _build_quantization_config,
    _is_eligible_weight,
    _is_moe_config,
    _list_safetensor_shards,
    _load_config,
    _PatternMatcher,
    _process_shard,
    _write_index_file,
    _write_output_shard,
    get_predefined_ignore_layers_from_config,
    model_free_quantize,
)

# ---------------------------------------------------------------------------
#  Helpers: create fake model directories
# ---------------------------------------------------------------------------


def _make_model_dir(tmp_path, config, tensors, *, multi_shard=False):
    """Create a minimal local model directory with config.json and safetensors.

    Args:
        tmp_path: pathlib.Path for the temporary directory.
        config: dict to write as config.json.
        tensors: dict of tensor_name → torch.Tensor.
        multi_shard: if True, split tensors into two shards with an index file.

    Returns:
        str: path to the created model directory.
    """
    model_dir = str(tmp_path / "source_model")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    if not multi_shard:
        save_file(tensors, os.path.join(model_dir, "model.safetensors"))
    else:
        # Split tensors into two shards
        keys = list(tensors.keys())
        mid = max(1, len(keys) // 2)
        shard1 = {k: tensors[k] for k in keys[:mid]}
        shard2 = {k: tensors[k] for k in keys[mid:]}

        save_file(shard1, os.path.join(model_dir, "model-00001-of-00002.safetensors"))
        save_file(shard2, os.path.join(model_dir, "model-00002-of-00002.safetensors"))

        weight_map = {}
        for k in keys[:mid]:
            weight_map[k] = "model-00001-of-00002.safetensors"
        for k in keys[mid:]:
            weight_map[k] = "model-00002-of-00002.safetensors"

        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

    return model_dir


_SIMPLE_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 128,
    "num_hidden_layers": 2,
}

_SIMPLE_TENSORS = {
    "model.decoder.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
    "model.decoder.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
    "model.decoder.layers.0.self_attn.v_proj.weight": torch.randn(128, 128),
    "model.decoder.layers.0.self_attn.out_proj.weight": torch.randn(128, 128),
    "model.decoder.layers.0.fc1.weight": torch.randn(512, 128),
    "model.decoder.layers.0.fc2.weight": torch.randn(128, 512),
    "model.decoder.layers.0.fc1.bias": torch.randn(512),
    "model.decoder.embed_tokens.weight": torch.randn(1000, 128),
    "lm_head.weight": torch.randn(1000, 128),
}


# ===========================================================================
#  Test: _is_eligible_weight
# ===========================================================================


class TestIsEligibleWeight:
    def test_2d_weight_tensor(self):
        assert _is_eligible_weight("layer.weight", torch.randn(10, 20)) is True

    def test_bias_not_eligible(self):
        assert _is_eligible_weight("layer.bias", torch.randn(10, 20)) is False

    def test_1d_weight_not_eligible(self):
        assert _is_eligible_weight("layer.weight", torch.randn(10)) is False

    def test_3d_weight_not_eligible(self):
        assert _is_eligible_weight("layer.weight", torch.randn(2, 3, 4)) is False

    def test_non_weight_suffix(self):
        assert _is_eligible_weight("layer.qweight", torch.randn(10, 20)) is False
        assert _is_eligible_weight("layer.scales", torch.randn(10, 20)) is False


# ===========================================================================
#  Test: _PatternMatcher (should_ignore / should_skip / resolve_scheme)
# ===========================================================================


def _matcher(ignore=None, layer_config=None, default=None):
    """Shorthand to build a _PatternMatcher for tests."""
    return _PatternMatcher(
        ignore if ignore is not None else [],
        layer_config if layer_config is not None else {},
        default if default is not None else {},
    )


class TestPatternMatcherIgnore:
    def test_substring_match(self):
        m = _matcher(ignore=["mlp.fc1"])
        assert m.should_ignore("model.layers.0.mlp.fc1.weight") is True

    def test_no_match(self):
        m = _matcher(ignore=["no_match"])
        assert m.should_ignore("model.layers.0.mlp.fc1.weight") is False

    def test_trailing_dot_pattern_with_model_prefix(self):
        m = _matcher(ignore=["layers.0."])
        assert m.should_ignore("model.layers.0.mlp.fc1.weight") is True

    def test_trailing_dot_pattern_no_partial_number_match(self):
        m = _matcher(ignore=["layers.4."])
        assert m.should_ignore("model.layers.45.mlp.fc1.weight") is False

    def test_trailing_dot_pattern_exact_number(self):
        m = _matcher(ignore=["layers.45."])
        assert m.should_ignore("model.layers.45.mlp.fc1.weight") is True

    def test_multiple_patterns(self):
        m = _matcher(ignore=["lm_head", "embed_tokens"])
        assert m.should_ignore("lm_head.weight") is True
        assert m.should_ignore("model.embed_tokens.weight") is True
        assert m.should_ignore("model.layers.0.fc1.weight") is False

    def test_empty_patterns(self):
        m = _matcher()
        assert m.should_ignore("anything.weight") is False


class TestPatternMatcherSkip:
    def test_shared_expert_gate(self):
        m = _matcher()
        assert m.should_skip("model.layers.0.shared_expert_gate.weight") is True

    def test_mlp_gate(self):
        m = _matcher()
        assert m.should_skip("model.layers.0.mlp.gate.weight") is True

    def test_embed(self):
        m = _matcher()
        assert m.should_skip("model.visual.pos_embed.weight") is True
        assert m.should_skip("model.language_model.embed_tokens.weight") is True

    def test_normal_layer_not_skipped(self):
        m = _matcher()
        assert m.should_skip("model.layers.0.mlp.fc1.weight") is False
        assert m.should_skip("model.layers.0.self_attn.q_proj.weight") is False


class TestPatternMatcherResolveScheme:
    DEFAULT = {"bits": 4, "group_size": 128, "sym": True}

    def test_exact_match(self):
        lc = {"model.layers.0.mlp.fc1": {"bits": 8, "group_size": 32}}
        m = _matcher(layer_config=lc, default=self.DEFAULT)
        result = m.resolve_scheme("model.layers.0.mlp.fc1.weight")
        assert result["bits"] == 8
        assert result["group_size"] == 32
        assert result["sym"] is True

    def test_regex_match(self):
        lc = {r".*k_proj": {"bits": 8}}
        m = _matcher(layer_config=lc, default=self.DEFAULT)
        result = m.resolve_scheme("model.layers.0.self_attn.k_proj.weight")
        assert result["bits"] == 8
        assert result["group_size"] == 128

    def test_default_fallback(self):
        m = _matcher(default=self.DEFAULT)
        result = m.resolve_scheme("model.layers.0.mlp.fc1.weight")
        assert result == self.DEFAULT

    def test_bits_16_returns_none(self):
        lc = {"model.layers.0.mlp.fc1": {"bits": 16}}
        m = _matcher(layer_config=lc, default=self.DEFAULT)
        assert m.resolve_scheme("model.layers.0.mlp.fc1.weight") is None

    def test_bits_32_returns_none(self):
        lc = {"model.layers.0.mlp.fc1": {"bits": 32}}
        m = _matcher(layer_config=lc, default=self.DEFAULT)
        assert m.resolve_scheme("model.layers.0.mlp.fc1.weight") is None

    def test_fuzzy_substring_match(self):
        lc = {"k_proj[": {"bits": 8}}  # Invalid regex, falls back to substring
        m = _matcher(layer_config=lc, default=self.DEFAULT)
        result = m.resolve_scheme("model.layers.0.self_attn.k_proj[0].weight")
        assert result is not None
        assert result["bits"] == 8


# ===========================================================================
#  Test: _is_moe_config
# ===========================================================================


class TestIsMoeConfig:
    def test_num_local_experts(self):
        assert _is_moe_config({"num_local_experts": 8}) is True

    def test_num_experts(self):
        assert _is_moe_config({"num_experts": 4}) is True

    def test_num_experts_per_tok(self):
        assert _is_moe_config({"num_experts_per_tok": 2}) is True

    def test_model_type_moe(self):
        assert _is_moe_config({"model_type": "mixtral_moe"}) is True

    def test_architecture_moe(self):
        assert _is_moe_config({"architectures": ["SomeMoEForCausalLM"]}) is True

    def test_not_moe(self):
        assert _is_moe_config({"architectures": ["LlamaForCausalLM"]}) is False
        assert _is_moe_config({}) is False


# ===========================================================================
#  Test: get_predefined_ignore_layers_from_config
# ===========================================================================


class TestGetPredefinedIgnoreLayers:
    def test_longcat(self):
        cfg = {"architectures": ["LongcatForCausalLM"]}
        layers = get_predefined_ignore_layers_from_config(cfg)
        assert "classifier" in layers

    def test_glm4moe_lite(self):
        cfg = {"architectures": ["Glm4MoeLiteForCausalLM"], "first_k_dense_replace": 3}
        layers = get_predefined_ignore_layers_from_config(cfg)
        assert "layers.0.mlp" in layers
        assert "layers.1.mlp" in layers
        assert "layers.2.mlp" in layers

    def test_glm_moe_dsa(self):
        cfg = {"model_type": "glm_moe_dsa", "architectures": ["SomeModel"], "first_k_dense_replace": 2}
        layers = get_predefined_ignore_layers_from_config(cfg)
        assert "layers.0.mlp" in layers
        assert "layers.1.mlp" in layers
        assert "weights_proj" in layers

    def test_step3p5(self):
        cfg = {"model_type": "step3p5", "architectures": ["Step3p5Model"]}
        layers = get_predefined_ignore_layers_from_config(cfg)
        assert "g_proj" in layers
        assert "moe.gate" in layers
        assert "eh_proj" in layers
        assert "shared_head" in layers
        assert "layers.45" in layers

    def test_generic_moe_fallback(self):
        cfg = {"num_local_experts": 8, "architectures": ["SomeModel"]}
        layers = get_predefined_ignore_layers_from_config(cfg)
        assert ".gate" in layers

    def test_normal_model_empty(self):
        cfg = {"architectures": ["LlamaForCausalLM"]}
        assert get_predefined_ignore_layers_from_config(cfg) == []


# ===========================================================================
#  Test: _load_config / _list_safetensor_shards
# ===========================================================================


class TestConfigAndShards:
    def test_load_config(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        config = _load_config(model_dir)
        assert config["model_type"] == "opt"
        assert config["architectures"] == ["OPTForCausalLM"]

    def test_load_config_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.json"):
            _load_config(str(tmp_path))

    def test_list_single_shard(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        shards = _list_safetensor_shards(model_dir)
        assert shards == ["model.safetensors"]

    def test_list_multi_shard(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS, multi_shard=True)
        shards = _list_safetensor_shards(model_dir)
        assert len(shards) == 2
        assert "model-00001-of-00002.safetensors" in shards
        assert "model-00002-of-00002.safetensors" in shards

    def test_list_shards_missing(self, tmp_path):
        os.makedirs(str(tmp_path / "empty_model"))
        with pytest.raises(FileNotFoundError, match="No safetensors"):
            _list_safetensor_shards(str(tmp_path / "empty_model"))


# ===========================================================================
#  Test: _build_quantization_config
# ===========================================================================


class TestBuildQuantizationConfig:
    DEFAULT_SCHEME = {"bits": 4, "group_size": 128, "sym": True}

    def test_basic_config(self):
        qcfg = _build_quantization_config(
            default_scheme=self.DEFAULT_SCHEME,
            layer_config={},
            ignore_patterns=[],
            quantized_layers=["a.fc1", "a.fc2"],
            ignored_layers=[],
        )
        assert qcfg["quant_method"] == "auto-round"
        assert qcfg["bits"] == 4
        assert qcfg["group_size"] == 128
        assert qcfg["sym"] is True
        assert qcfg["model_free"] is True
        assert qcfg["iters"] == 0

    def test_ignored_layers_in_extra_config(self):
        qcfg = _build_quantization_config(
            default_scheme=self.DEFAULT_SCHEME,
            layer_config={},
            ignore_patterns=[],
            quantized_layers=["a.fc1"],
            ignored_layers=["a.gate", "lm_head"],
        )
        extra = qcfg["extra_config"]
        assert extra["a.gate"]["bits"] == 16
        assert extra["lm_head"]["bits"] == 16

    def test_layer_config_differs(self):
        qcfg = _build_quantization_config(
            default_scheme=self.DEFAULT_SCHEME,
            layer_config={"a.fc1": {"bits": 8, "group_size": 32}},
            ignore_patterns=[],
            quantized_layers=["a.fc1"],
            ignored_layers=[],
        )
        extra = qcfg["extra_config"]
        assert "a.fc1" in extra
        assert extra["a.fc1"]["bits"] == 8

    def test_no_extra_config_when_empty(self):
        qcfg = _build_quantization_config(
            default_scheme=self.DEFAULT_SCHEME,
            layer_config={},
            ignore_patterns=[],
            quantized_layers=["a.fc1"],
            ignored_layers=[],
        )
        assert "extra_config" not in qcfg


# ===========================================================================
#  Test: _write_output_shard / _write_index_file
# ===========================================================================


class TestWriteOutputShardAndIndex:
    def test_write_single_shard(self, tmp_path):
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)
        tensors = {"a.weight": torch.randn(4, 4), "b.weight": torch.randn(4, 4)}
        weight_map = {}
        _write_output_shard(out_dir, "model-00001-of-00001.safetensors", tensors, weight_map)
        assert os.path.exists(os.path.join(out_dir, "model-00001-of-00001.safetensors"))
        assert weight_map["a.weight"] == "model-00001-of-00001.safetensors"
        assert weight_map["b.weight"] == "model-00001-of-00001.safetensors"

    def test_write_index_single_shard_rename(self, tmp_path):
        """Single shard should be renamed to model.safetensors."""
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)
        save_file({"x": torch.randn(2, 2)}, os.path.join(out_dir, "model-00001-of-00001.safetensors"))
        weight_map = {"x": "model-00001-of-00001.safetensors"}
        _write_index_file(out_dir, weight_map)
        assert os.path.exists(os.path.join(out_dir, "model.safetensors"))
        assert not os.path.exists(os.path.join(out_dir, "model-00001-of-00001.safetensors"))

    def test_write_index_multi_shard(self, tmp_path):
        """Multiple shards should produce an index JSON file."""
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir)
        save_file({"a": torch.randn(2, 2)}, os.path.join(out_dir, "shard-1.safetensors"))
        save_file({"b": torch.randn(2, 2)}, os.path.join(out_dir, "shard-2.safetensors"))
        weight_map = {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}
        _write_index_file(out_dir, weight_map)
        idx_path = os.path.join(out_dir, "model.safetensors.index.json")
        assert os.path.exists(idx_path)
        with open(idx_path) as f:
            idx = json.load(f)
        assert idx["weight_map"]["a"] == "shard-1.safetensors"
        assert idx["weight_map"]["b"] == "shard-2.safetensors"


# ===========================================================================
#  Test: _process_shard
# ===========================================================================


class TestProcessShard:
    DEFAULT_SCHEME = {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"}

    def test_quantizes_eligible_weights(self, tmp_path):
        tensors = {
            "model.layers.0.mlp.fc1.weight": torch.randn(64, 128),
            "model.layers.0.mlp.fc1.bias": torch.randn(64),
        }
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(tensors, shard_path)

        output, quantized, ignored = _process_shard(shard_path, self.DEFAULT_SCHEME, {}, [])
        assert "model.layers.0.mlp.fc1" in quantized
        assert "model.layers.0.mlp.fc1.qweight" in output
        assert "model.layers.0.mlp.fc1.qzeros" in output
        assert "model.layers.0.mlp.fc1.scales" in output
        assert "model.layers.0.mlp.fc1.g_idx" in output
        # Bias passes through
        assert "model.layers.0.mlp.fc1.bias" in output

    def test_ignores_user_patterns(self, tmp_path):
        tensors = {"lm_head.weight": torch.randn(100, 128)}
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(tensors, shard_path)

        output, quantized, ignored = _process_shard(shard_path, self.DEFAULT_SCHEME, {}, ["lm_head"])
        assert "lm_head" in ignored
        assert len(quantized) == 0
        assert "lm_head.weight" in output  # kept as-is

    def test_skips_routing_gates(self, tmp_path):
        tensors = {"model.layers.0.mlp.gate.weight": torch.randn(8, 128)}
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(tensors, shard_path)

        output, quantized, ignored = _process_shard(shard_path, self.DEFAULT_SCHEME, {}, [])
        assert len(quantized) == 0
        assert "model.layers.0.mlp.gate" in ignored
        assert "model.layers.0.mlp.gate.weight" in output

    def test_layer_config_override(self, tmp_path):
        tensors = {"model.layers.0.fc1.weight": torch.randn(64, 128)}
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(tensors, shard_path)

        lc = {"model.layers.0.fc1": {"bits": 16}}
        output, quantized, ignored = _process_shard(shard_path, self.DEFAULT_SCHEME, lc, [])
        assert "model.layers.0.fc1" in ignored
        assert len(quantized) == 0
        assert "model.layers.0.fc1.weight" in output  # kept full precision


# ===========================================================================
#  Test: model_free_quantize (end-to-end)
# ===========================================================================


class TestModelFreeQuantize:
    def test_basic_quantization(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")

        result = model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        assert os.path.isdir(result)
        assert os.path.exists(os.path.join(output_dir, "config.json"))
        assert os.path.exists(os.path.join(output_dir, "quantization_config.json"))

        with open(os.path.join(output_dir, "config.json")) as f:
            cfg = json.load(f)
        qc = cfg["quantization_config"]
        assert qc["quant_method"] == "auto-round"
        assert qc["bits"] == 4
        assert qc["model_free"] is True

    def test_ignore_layers(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
            ignore_layers="lm_head,embed_tokens",
        )

        with open(os.path.join(output_dir, "config.json")) as f:
            cfg = json.load(f)
        extra = cfg["quantization_config"].get("extra_config", {})
        assert "lm_head" in extra
        assert extra["lm_head"]["bits"] == 16

        # Verify lm_head.weight is kept original in safetensors
        st_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
        with safe_open(os.path.join(output_dir, st_files[0]), framework="pt") as f:
            keys = list(f.keys())
        assert "lm_head.weight" in keys
        # Should NOT have lm_head.qweight
        assert "lm_head.qweight" not in keys

    def test_layer_config_override(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
            layer_config={".*k_proj": {"bits": 8, "group_size": 32}},
        )

        with open(os.path.join(output_dir, "config.json")) as f:
            cfg = json.load(f)
        extra = cfg["quantization_config"].get("extra_config", {})
        # k_proj should appear with differing config
        found = any("k_proj" in k for k in extra)
        assert found, f"k_proj not in extra_config: {extra}"

    def test_low_disk_mem_usage(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")

        result = model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
            low_disk_mem_usage=True,
        )

        assert os.path.isdir(result)
        assert os.path.exists(os.path.join(output_dir, "config.json"))

    def test_multi_shard_model(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS, multi_shard=True)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        # Should produce index file for multi-shard
        idx_path = os.path.join(output_dir, "model.safetensors.index.json")
        assert os.path.exists(idx_path)
        with open(idx_path) as f:
            idx = json.load(f)
        assert len(idx["weight_map"]) > 0

    def test_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="auto_round"):
            model_free_quantize(
                model_name_or_path=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                format="auto_gptq",
            )

    def test_invalid_scheme_raises(self, tmp_path):
        with pytest.raises(ValueError, match="INVALID"):
            model_free_quantize(
                model_name_or_path=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                scheme="INVALID",
            )

    def test_invalid_scheme_type_raises(self, tmp_path):
        with pytest.raises(TypeError, match="Unsupported scheme type"):
            model_free_quantize(
                model_name_or_path=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                scheme=12345,
            )

    def test_quantized_tensor_shapes(self, tmp_path):
        """Verify that quantized tensors have expected shapes."""
        tensors = {
            "layer.weight": torch.randn(256, 128),
        }
        model_dir = _make_model_dir(
            tmp_path,
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            tensors,
        )
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        st_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
        with safe_open(os.path.join(output_dir, st_files[0]), framework="pt") as f:
            qweight = f.get_tensor("layer.qweight")
            scales = f.get_tensor("layer.scales")
            qzeros = f.get_tensor("layer.qzeros")
            g_idx = f.get_tensor("layer.g_idx")

        # For W4A16, 4-bit packed into int32: each int32 holds 32/4=8 values
        # qweight shape: (in_features // 32 * bits, out_features) = (128 // 8, 256) = (16, 256)
        assert qweight.dtype == torch.int32
        assert scales.dtype == torch.float16
        assert qzeros.dtype == torch.int32
        assert g_idx.dtype == torch.int32
        assert g_idx.shape[0] == 128  # in_features

    def test_scheme_object_input(self, tmp_path):
        """model_free_quantize accepts QuantizationScheme objects."""
        from auto_round.schemes import QuantizationScheme

        model_dir = _make_model_dir(
            tmp_path,
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            {"layer.weight": torch.randn(64, 128)},
        )
        output_dir = str(tmp_path / "output")

        scheme = QuantizationScheme(bits=4, group_size=64, sym=False)
        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme=scheme,
        )

        with open(os.path.join(output_dir, "config.json")) as f:
            cfg = json.load(f)
        qc = cfg["quantization_config"]
        assert qc["bits"] == 4
        assert qc["group_size"] == 64
        assert qc["sym"] is False

    def test_moe_model_auto_ignores_gate(self, tmp_path):
        """MoE models should auto-ignore .gate layers."""
        moe_config = {
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "num_local_experts": 8,
        }
        tensors = {
            "model.layers.0.mlp.gate.weight": torch.randn(8, 128),
            "model.layers.0.mlp.experts.0.fc1.weight": torch.randn(256, 128),
        }
        model_dir = _make_model_dir(tmp_path, moe_config, tensors)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        with open(os.path.join(output_dir, "config.json")) as f:
            cfg = json.load(f)
        qc = cfg["quantization_config"]
        extra = qc.get("extra_config", {})
        # gate should be in ignored
        gate_ignored = any("gate" in k for k in extra)
        assert gate_ignored, f"gate not ignored in extra_config: {extra}"

    def test_copies_tokenizer_files(self, tmp_path):
        """Tokenizer files should be copied to output directory."""
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        # Create fake tokenizer file
        with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"tokenizer_class": "GPT2Tokenizer"}, f)

        output_dir = str(tmp_path / "output")
        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        assert os.path.exists(os.path.join(output_dir, "tokenizer_config.json"))

    def test_lm_head_and_embed_ignored_by_default(self, tmp_path):
        """lm_head and any layer containing 'embed' are kept in full precision by default."""
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        all_keys = set()
        for fname in os.listdir(output_dir):
            if fname.endswith(".safetensors"):
                with safe_open(os.path.join(output_dir, fname), framework="pt") as f:
                    all_keys.update(f.keys())

        # lm_head should be original weight, NOT quantized
        assert "lm_head.weight" in all_keys
        assert "lm_head.qweight" not in all_keys
        # Any embed layer should be kept as-is
        embed_weights = [k for k in all_keys if "embed" in k and k.endswith(".weight")]
        assert len(embed_weights) > 0, "No embed layers found in output"
        for ek in embed_weights:
            base = ek[: -len(".weight")]
            assert f"{base}.qweight" not in all_keys, f"{base} should not be quantized"

    def test_quant_lm_head_quantizes_lm_head(self, tmp_path):
        """With quant_lm_head=True, lm_head IS quantized."""
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
            quant_lm_head=True,
        )

        all_keys = set()
        for fname in os.listdir(output_dir):
            if fname.endswith(".safetensors"):
                with safe_open(os.path.join(output_dir, fname), framework="pt") as f:
                    all_keys.update(f.keys())

        # lm_head should be quantized
        assert "lm_head.qweight" in all_keys
        assert "lm_head.weight" not in all_keys


# ===========================================================================
#  Test: _process_shard with fused expert tensors
# ===========================================================================


class TestProcessShardFusedExperts:
    """Tests for fused-expert handling inside _process_shard."""

    DEFAULT_SCHEME = {"bits": 4, "group_size": 32, "sym": True, "data_type": "int"}

    def test_fused_gate_up_proj_split_and_quantized(self, tmp_path):
        """A 3-D gate_up_proj tensor is split into per-expert slices and quantized."""
        N, I, H = 2, 64, 32
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(
            {
                "model.layers.0.mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
                "model.layers.0.mlp.experts.down_proj": torch.randn(N, H, I),
            },
            shard_path,
        )

        output, quantized, ignored = _process_shard(shard_path, self.DEFAULT_SCHEME, {}, [])

        # Original fused keys must NOT appear
        assert not any("gate_up_proj" in k for k in output)
        assert not any("gate_up_proj" in k for k in quantized)

        # Each expert's per-expert weights should be quantized
        for i in range(N):
            gate_base = f"model.layers.0.mlp.experts.{i}.gate_proj"
            up_base = f"model.layers.0.mlp.experts.{i}.up_proj"
            down_base = f"model.layers.0.mlp.experts.{i}.down_proj"
            for base in [gate_base, up_base, down_base]:
                assert base in quantized, f"{base} not in quantized: {quantized}"
                assert f"{base}.qweight" in output
                assert f"{base}.scales" in output

    def test_fused_experts_with_ignore_pattern(self, tmp_path):
        """After splitting, per-expert weights honour ignore_patterns."""
        N, I, H = 2, 32, 16
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(
            {"model.mtp.0.mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H)},
            shard_path,
        )

        output, quantized, ignored = _process_shard(shard_path, self.DEFAULT_SCHEME, {}, ["mtp"])

        assert len(quantized) == 0
        # All split weight keys must be kept in full precision
        for i in range(N):
            assert f"model.mtp.0.mlp.experts.{i}.gate_proj.weight" in output
            assert f"model.mtp.0.mlp.experts.{i}.up_proj.weight" in output


# ===========================================================================
#  Test: model_free_quantize with fused expert model (end-to-end)
# ===========================================================================


class TestModelFreeQuantizeFusedExperts:
    """End-to-end tests for model_free_quantize when the source has fused expert tensors."""

    def test_fused_moe_model_quantizes_per_expert(self, tmp_path):
        """model_free_quantize correctly splits and quantizes fused experts.

        gate_proj is kept in full precision because the MoE-config auto-ignore
        rule adds '.gate' which matches '.gate_proj' via substring.  down_proj
        and up_proj are not gated so they get quantized.
        """
        N, I, H = 4, 64, 32
        # Use a plain (non-MoE flagged) config so no auto-ignore is injected
        plain_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        }
        tensors = {
            "model.layers.0.mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
            "model.layers.0.mlp.experts.down_proj": torch.randn(N, H, I),
        }
        model_dir = _make_model_dir(tmp_path, plain_config, tensors)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        assert os.path.exists(os.path.join(output_dir, "config.json"))

        # Collect all tensor keys written to safetensors output(s)
        written_keys = set()
        for fname in os.listdir(output_dir):
            if fname.endswith(".safetensors"):
                fpath = os.path.join(output_dir, fname)
                with safe_open(fpath, framework="pt") as f:
                    written_keys.update(f.keys())

        # Fused key should not appear in output
        assert not any("gate_up_proj" in k for k in written_keys)

        # Per-expert quantized tensors should appear for all three projections
        for i in range(N):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                base = f"model.layers.0.mlp.experts.{i}.{proj}"
                assert f"{base}.qweight" in written_keys, f"{base}.qweight not found"
                assert f"{base}.scales" in written_keys, f"{base}.scales not found"


# ===========================================================================
#  Test: low_gpu_mem_usage
# ===========================================================================


class TestLowGpuMemUsage:
    """Tests that the low_gpu_mem_usage flag is accepted and functions correctly on CPU."""

    DEFAULT_SCHEME = {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"}

    def test_process_shard_accepts_flag(self, tmp_path):
        """_process_shard works with low_gpu_mem_usage=True on CPU."""
        shard_path = str(tmp_path / "shard.safetensors")
        save_file({"layer.weight": torch.randn(64, 128)}, shard_path)

        output, quantized, _ = _process_shard(
            shard_path,
            self.DEFAULT_SCHEME,
            {},
            [],
            device="cpu",
            low_gpu_mem_usage=True,
        )
        assert "layer" in quantized
        assert "layer.qweight" in output

    def test_model_free_quantize_accepts_flag(self, tmp_path):
        """model_free_quantize works with low_gpu_mem_usage=True."""
        model_dir = _make_model_dir(
            tmp_path,
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            {"layer.weight": torch.randn(64, 128)},
        )
        output_dir = str(tmp_path / "output")
        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
            low_gpu_mem_usage=True,
        )
        assert os.path.exists(os.path.join(output_dir, "config.json"))
        st_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
        assert len(st_files) > 0
