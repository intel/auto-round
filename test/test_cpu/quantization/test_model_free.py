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

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from auto_round import AutoRound
from auto_round.compressors.model_free import (
    _dequant_fp8_tensors,
    _ModelFreeCompressorCore,
    _PatternMatcher,
    _process_shard,
    get_predefined_ignore_layers_from_config,
    is_model_free_supported_scheme,
)
from auto_round.schemes import QuantizationScheme

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_LLAMA_CFG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
_DEFAULT_SCHEME = {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"}

_SIMPLE_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 128,
    "num_hidden_layers": 2,
}

_SIMPLE_TENSORS = {
    "model.decoder.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
    "model.decoder.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
    "model.decoder.layers.0.fc1.weight": torch.randn(512, 128),
    "model.decoder.layers.0.fc2.weight": torch.randn(128, 512),
    "model.decoder.layers.0.fc1.bias": torch.randn(512),
    "model.decoder.embed_tokens.weight": torch.randn(1000, 128),
    "lm_head.weight": torch.randn(1000, 128),
}


def _make_model_dir(tmp_path, config, tensors, *, multi_shard=False):
    """Create a minimal local model directory with config.json and safetensors."""
    model_dir = str(tmp_path / "source_model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    if not multi_shard:
        save_file(tensors, os.path.join(model_dir, "model.safetensors"))
    else:
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
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)

    return model_dir


def _matcher(ignore=None, layer_config=None, default=None):
    return _PatternMatcher(ignore or [], layer_config or {}, default or {})


def _read_output_keys(output_dir):
    keys = set()
    for f in os.listdir(output_dir):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(output_dir, f), framework="pt") as sf:
                keys.update(sf.keys())
    return keys


def _read_qconfig(output_dir):
    with open(os.path.join(output_dir, "config.json")) as f:
        return json.load(f).get("quantization_config", {})


# ===========================================================================
#  _PatternMatcher
# ===========================================================================


class TestPatternMatcher:
    def test_ignore_substring(self):
        m = _matcher(ignore=["mlp"])
        assert m.should_ignore("model.layers.0.mlp.fc1.weight") is True
        assert m.should_ignore("model.layers.0.self_attn.q_proj.weight") is False

    def test_ignore_trailing_dot(self):
        m = _matcher(ignore=["layers.4."])
        assert m.should_ignore("model.layers.4.mlp.fc1.weight") is True
        assert m.should_ignore("model.layers.45.mlp.fc1.weight") is False

    def test_skip_predefined(self):
        m = _matcher()
        assert m.should_skip("model.layers.0.shared_expert_gate.weight") is True
        assert m.should_skip("model.layers.0.mlp.gate.weight") is True
        assert m.should_skip("model.embed_tokens.weight") is True
        assert m.should_skip("model.layers.0.mlp.fc1.weight") is False

    def test_resolve_scheme_exact_regex_and_default(self):
        default = {"bits": 4, "group_size": 128, "sym": True}
        lc = {
            "model.layers.0.mlp.fc1": {"bits": 8, "group_size": 32},
            r".*k_proj": {"bits": 8},
        }
        m = _matcher(layer_config=lc, default=default)
        assert m.resolve_scheme("model.layers.0.mlp.fc1.weight")["bits"] == 8
        assert m.resolve_scheme("model.layers.0.self_attn.k_proj.weight")["bits"] == 8
        assert m.resolve_scheme("model.layers.0.mlp.fc2.weight") == default

    def test_resolve_bits16_returns_none(self):
        m = _matcher(layer_config={"model.layers.0.fc1": {"bits": 16}}, default={"bits": 4, "group_size": 128})
        assert m.resolve_scheme("model.layers.0.fc1.weight") is None

    def test_resolve_substring_pattern(self):
        default = {"bits": 4, "group_size": 128, "sym": True}
        m = _matcher(layer_config={".ffn.experts.": {"bits": 2, "group_size": 64}}, default=default)
        r = m.resolve_scheme("model.layers.0.ffn.experts.3.gate_proj.weight")
        assert r["bits"] == 2 and r["group_size"] == 64
        assert m.resolve_scheme("model.layers.0.self_attn.q_proj.weight") == default


# ===========================================================================
#  _parse_layer_config — scheme key resolution
# ===========================================================================


class TestParseLayerConfig:
    @staticmethod
    def _make_core(layer_config_input):
        core = _ModelFreeCompressorCore(
            model_name_or_path="dummy",
            output_dir="dummy_out",
            scheme="W4A16",
        )
        core.layer_config_input = layer_config_input
        core._parse_scheme()
        core._parse_layer_config()
        return core

    def test_scheme_key_resolves(self):
        core = self._make_core({".ffn.experts.": {"scheme": "W2A16"}})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2 and "scheme" not in cfg

        m = _matcher(layer_config=core.layer_config, default=core.default_scheme)
        assert m.resolve_scheme("model.layers.0.ffn.experts.3.gate_proj.weight")["bits"] == 2

    def test_scheme_key_with_overrides(self):
        core = self._make_core({".ffn.experts.": {"scheme": "W2A16", "group_size": 32}})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2 and cfg["group_size"] == 32

    def test_string_value(self):
        core = self._make_core({".ffn.experts.": "W2A16"})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2

    def test_quantization_scheme_value(self):
        core = self._make_core({".ffn.experts.": QuantizationScheme(bits=2, group_size=64)})
        cfg = next(v for k, v in core.layer_config.items() if "ffn.experts" in k)
        assert cfg["bits"] == 2 and cfg["group_size"] == 64


# ===========================================================================
#  _process_shard
# ===========================================================================


class TestProcessShard:
    def test_quantizes_eligible_weights(self, tmp_path):
        shard_path = str(tmp_path / "shard.safetensors")
        save_file({"layer.fc1.weight": torch.randn(64, 128), "layer.fc1.bias": torch.randn(64)}, shard_path)
        output, quantized, _ = _process_shard(shard_path, _DEFAULT_SCHEME, {}, [])
        assert "layer.fc1" in quantized
        assert "layer.fc1.qweight" in output and "layer.fc1.bias" in output

    def test_ignores_and_skips(self, tmp_path):
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(
            {"lm_head.weight": torch.randn(100, 128), "model.layers.0.mlp.gate.weight": torch.randn(8, 128)},
            shard_path,
        )
        _, quantized, ignored = _process_shard(shard_path, _DEFAULT_SCHEME, {}, ["lm_head"])
        assert len(quantized) == 0
        assert "lm_head" in ignored and "model.layers.0.mlp.gate" in ignored

    def test_fused_expert_split(self, tmp_path):
        N, I, H = 2, 64, 32
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(
            {
                "model.layers.0.mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
                "model.layers.0.mlp.experts.down_proj": torch.randn(N, H, I),
            },
            shard_path,
        )
        output, quantized, _ = _process_shard(shard_path, _DEFAULT_SCHEME, {}, [])
        for i in range(N):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                base = f"model.layers.0.mlp.experts.{i}.{proj}"
                assert base in quantized and f"{base}.qweight" in output


# ===========================================================================
#  FP8 source model
# ===========================================================================


class TestFP8Source:
    def test_dequant_fp8(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        raw = {"layer.weight": w, "layer.weight_scale_inv": torch.tensor(0.5), "layer.bias": torch.randn(64)}
        result = _dequant_fp8_tensors(raw, block_size=None)
        assert result["layer.weight"].dtype == torch.bfloat16 and "layer.weight_scale_inv" not in result

    def test_no_fp8_noop(self):
        raw = {"layer.weight": torch.randn(64, 128)}
        assert _dequant_fp8_tensors(raw, block_size=None) is raw

    def test_process_shard_fp8(self, tmp_path):
        shard_path = str(tmp_path / "shard.safetensors")
        w = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        save_file({"layer.weight": w, "layer.weight_scale_inv": torch.tensor(1.0)}, shard_path)
        output, quantized, _ = _process_shard(shard_path, _DEFAULT_SCHEME, {}, [], device="cpu", fp8_block_size=None)
        assert "layer" in quantized and "layer.qweight" in output


# ===========================================================================
#  End-to-end ModelFreeQuantize
# ===========================================================================


class TestModelFreeQuantize:
    def test_basic(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="W4A16", model_free=True).quantize_and_save(output_dir)
        qc = _read_qconfig(output_dir)
        assert qc["quant_method"] == "auto-round" and qc["bits"] == 4 and qc["model_free"] is True
        keys = _read_output_keys(output_dir)
        assert "lm_head.weight" in keys and "lm_head.qweight" not in keys

    def test_ignore_layers(self, tmp_path):
        tensors = {
            "model.layers.0.mlp.fc1.weight": torch.randn(512, 128),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="W4A16", model_free=True, ignore_layers="mlp").quantize_and_save(output_dir)
        keys = _read_output_keys(output_dir)
        assert "model.layers.0.mlp.fc1.weight" in keys and "model.layers.0.mlp.fc1.qweight" not in keys
        assert "model.layers.0.self_attn.q_proj.qweight" in keys

    def test_multi_shard(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS, multi_shard=True)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="W4A16", model_free=True).quantize_and_save(output_dir)
        assert os.path.exists(os.path.join(output_dir, "model.safetensors.index.json"))

    def test_quant_lm_head(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="W4A16", model_free=True, quant_lm_head=True).quantize_and_save(output_dir)
        assert "lm_head.qweight" in _read_output_keys(output_dir)

    def test_asym(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, {"layer.weight": torch.randn(64, 128)})
        output_dir = str(tmp_path / "output")
        AutoRound(
            model=model_dir, scheme=QuantizationScheme(bits=4, group_size=64, sym=False), model_free=True
        ).quantize_and_save(output_dir)
        qc = _read_qconfig(output_dir)
        assert qc["sym"] is False and qc["group_size"] == 64


# ===========================================================================
#  Scheme validation
# ===========================================================================


_SUPPORTED = ["W2A16", "W2A16G32", "W2A16G64", "W4A16", "W4A16_MIXED", "W8A16"]
_UNSUPPORTED = ["W3A16", "FPW8A16", "BF16", "MXFP4", "MXFP8", "MXINT4", "NVFP4", "FP8_BLOCK", "FP8_STATIC", "INT8_W8A8"]


class TestSchemeValidation:
    @pytest.mark.parametrize("name", _SUPPORTED)
    def test_supported(self, tmp_path, name):
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, {"model.layers.0.mlp.fc1.weight": torch.randn(64, 128)})
        out = str(tmp_path / f"out_{name}")
        AutoRound(model=model_dir, scheme=name, model_free=True).quantize_and_save(out)
        assert "model.layers.0.mlp.fc1.qweight" in _read_output_keys(out)

    @pytest.mark.parametrize("name", _UNSUPPORTED)
    def test_unsupported_raises(self, tmp_path, name):
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, {"layer.weight": torch.randn(64, 128)})
        with pytest.raises(ValueError):
            AutoRound(model=model_dir, model_free=True, scheme=name).quantize_and_save(str(tmp_path / "out"))

    def test_is_model_free_supported_scheme(self):
        for name in _SUPPORTED:
            assert is_model_free_supported_scheme(name) is True
        for name in _UNSUPPORTED:
            assert is_model_free_supported_scheme(name) is False
        assert is_model_free_supported_scheme("DOES_NOT_EXIST") is False


# ===========================================================================
#  CLI auto-routing
# ===========================================================================


class TestCliAutoRouting:
    def test_auto_routes(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, {"layer.weight": torch.randn(64, 128)})
        out_dir = str(tmp_path / "out")
        from auto_round.__main__ import BasicArgumentParser, tune

        args = BasicArgumentParser().parse_args(
            [
                "--model",
                model_dir,
                "--scheme",
                "W4A16",
                "--iters",
                "0",
                "--disable_opt_rtn",
                "--format",
                "auto_round",
                "--output_dir",
                out_dir,
            ]
        )
        tune(args)
        assert _read_qconfig(out_dir).get("model_free") is True

    def test_disable_model_free_flag(self):
        from auto_round.__main__ import BasicArgumentParser

        args = BasicArgumentParser().parse_args(
            [
                "--model",
                "dummy",
                "--scheme",
                "W4A16",
                "--iters",
                "0",
                "--disable_opt_rtn",
                "--disable_model_free",
            ]
        )
        auto_route = (
            not args.model_free
            and not args.disable_model_free
            and args.iters == 0
            and args.disable_opt_rtn is True
            and is_model_free_supported_scheme(args.scheme)
        )
        assert auto_route is False


# ===========================================================================
#  Predefined ignore layers
# ===========================================================================


class TestPredefinedIgnoreLayers:
    def test_normal_model_empty(self):
        assert get_predefined_ignore_layers_from_config({"architectures": ["LlamaForCausalLM"]}) == []

    def test_step3p5_ignore_layers(self):
        cfg = {"model_type": "step3p5"}
        assert get_predefined_ignore_layers_from_config(cfg) == [
            "g_proj",
            "moe.gate",
            "eh_proj",
            "shared_head",
            "layers.45",
        ]
