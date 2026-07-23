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
    _build_mxfp_quantization_config,
    _build_quantization_config,
    _convert_auto_scheme_layer_config,
    _dequant_fp8_tensors,
    _dequant_mxfp_tensors,
    _expand_e8m0_block_scale,
    _handle_mxfp_source_tensors,
    _looks_like_auto_scheme,
    _ModelFreeCompressorCore,
    _PatternMatcher,
    _preprocess_model_type_source_tensors,
    _process_shard,
    _process_single_shard_task,
    _quantize_weight_mxfp,
    _validate_auto_scheme_options,
    get_predefined_ignore_layers_from_config,
    is_model_free_supported_scheme,
)
from auto_round.schemes import QuantizationScheme

from ...envs import require_compressed_tensors

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
        assert m.should_skip("model.layers.0.mlp.gate_proj.weight") is False
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
    def _make_core(layer_config_input, scheme="W4A16"):
        core = _ModelFreeCompressorCore(
            model_name_or_path="dummy",
            output_dir="dummy_out",
            scheme=scheme,
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

    def test_w4a16_mixed_recipe_in_model_free(self):
        core = self._make_core({}, scheme="W4A16_MIXED")
        assert core.default_scheme["bits"] == 8
        assert core.layer_config[".experts."]["bits"] == 4
        assert core.layer_config[".moe."]["bits"] == 4
        assert core.layer_config[".shared_expert."]["bits"] == 8


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

    def test_moe_stacked_weights_are_split_and_quantized(self, tmp_path):
        shard_path = str(tmp_path / "shard.safetensors")
        n_experts, hidden, intermediate = 3, 64, 32
        save_file(
            {
                "model.layers.3.moe.down_proj.weight": torch.randn(n_experts, hidden, intermediate),
                "model.layers.3.moe.gate_proj.weight": torch.randn(n_experts, intermediate, hidden),
                "model.layers.3.moe.up_proj.weight": torch.randn(n_experts, intermediate, hidden),
                "model.layers.3.moe.gate.weight": torch.randn(n_experts, hidden),
                "model.layers.3.moe.router_bias": torch.randn(n_experts),
            },
            shard_path,
        )

        output, quantized, ignored = _process_shard(shard_path, _DEFAULT_SCHEME, {}, [])

        for i in range(n_experts):
            for proj in ["down_proj", "gate_proj", "up_proj"]:
                base = f"model.layers.3.moe.experts.{i}.{proj}"
                assert base in quantized
                assert f"{base}.qweight" in output

        # Router gate stays in full precision by predefined skip rules.
        assert "model.layers.3.moe.gate" in ignored
        assert "model.layers.3.moe.gate.weight" in output
        # router_bias is not a 2D linear weight and remains unchanged.
        assert "model.layers.3.moe.router_bias" in output

    def test_3d_weight_in_ignored_layers(self, tmp_path):
        """A non-eligible 3D .weight tensor must appear in ignored_layers."""
        shard_path = str(tmp_path / "shard.safetensors")
        save_file({"model.layers.0.mlp.branch.weight": torch.randn(4, 8, 16)}, shard_path)

        output, quantized, ignored = _process_shard(shard_path, _DEFAULT_SCHEME, {}, [])

        assert "model.layers.0.mlp.branch.weight" in output
        assert quantized == []
        assert "model.layers.0.mlp.branch" in ignored


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

    def test_ignored_layer_preserves_original_fp8(self, tmp_path):
        """Ignored layers keep their original quantized tensors (no dequant)."""
        shard_path = str(tmp_path / "shard.safetensors")
        w_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.5)
        save_file(
            {"lm_head.weight": w_fp8, "lm_head.weight_scale_inv": scale, "layer.weight": torch.randn(64, 128)},
            shard_path,
        )
        output, quantized, ignored = _process_shard(
            shard_path, _DEFAULT_SCHEME, {}, ["lm_head"], device="cpu", fp8_block_size=None
        )
        # lm_head should be ignored and kept in original FP8 format
        assert "lm_head" in ignored
        assert output["lm_head.weight"].dtype == torch.float8_e4m3fn
        assert "lm_head.weight_scale_inv" in output
        # non-ignored layer should be quantized normally
        assert "layer" in quantized


# ===========================================================================
#  Quantization config builder
# ===========================================================================


class TestBuildQuantizationConfig:
    def test_extra_config_filters_embed_conv_only(self):
        default = {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"}
        ignored = [
            "model.embed_tokens",
            "model.conv1",
            "model.layers.0.shared_expert_gate",
            "model.layers.0.mlp.gate",
        ]

        cfg = _build_quantization_config(
            default_scheme=default,
            layer_config={},
            ignore_patterns=[],
            quantized_layers=[],
            ignored_layers=ignored,
        )

        extra = cfg.get("extra_config", {})
        # Non-Linear ops are filtered out.
        assert "model.embed_tokens" not in extra
        assert "model.conv1" not in extra
        # Other ignored layers should still be recorded.
        assert extra["model.layers.0.shared_expert_gate"] == {"bits": 16, "data_type": "float"}
        assert extra["model.layers.0.mlp.gate"] == {"bits": 16, "data_type": "float"}


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

    def test_streaming_uses_dedicated_source_shard_cache(self, tmp_path, monkeypatch):
        """Streaming source shards must not reuse same-named output shard files."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        shard_name = "model-00001-of-00001.safetensors"
        stale_output_shard = os.path.join(output_dir, shard_name)

        # Simulate stale quantized output shard from a previous run.
        save_file({"lm_head.weight": torch.randn(8, 8)}, stale_output_shard)

        called_local_dirs = []

        def _fake_download_single_shard(model_name_or_path, shard_filename, local_dir):
            called_local_dirs.append(local_dir)
            os.makedirs(local_dir, exist_ok=True)
            source_path = os.path.join(local_dir, shard_filename)
            # Source shard contains quantizable linear weight.
            save_file({"layer.weight": torch.randn(8, 8)}, source_path)
            return source_path

        monkeypatch.setattr(
            "auto_round.compressors.model_free._download_single_shard",
            _fake_download_single_shard,
        )

        result = _process_single_shard_task(
            shard_idx=0,
            shard_name=shard_name,
            model_name_or_path="org/dummy-model",
            work_dir=output_dir,
            source_dir="",
            is_streaming=True,
            device="cpu",
            default_scheme=_DEFAULT_SCHEME,
            layer_config={},
            ignore_patterns=["lm_head"],
            fp8_block_size=None,
            model_type=None,
            quant_output_dir=output_dir,
            total_shards=1,
        )

        _, _, _, out_shard_name, _, quantized, _ = result
        assert out_shard_name == shard_name
        assert "layer" in quantized
        assert called_local_dirs
        assert called_local_dirs[0] != output_dir
        assert called_local_dirs[0].startswith(os.path.join(output_dir, ".cache"))

        with safe_open(stale_output_shard, framework="pt") as sf:
            out_keys = set(sf.keys())
        assert "layer.qweight" in out_keys


# ===========================================================================
#  MXFP4 / MXFP8 model-free quantization
# ===========================================================================


class TestModelFreeMXFP:
    """End-to-end tests for MXFP4/MXFP8 model-free quantization."""

    def test_quantize_weight_mxfp4_shapes(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        out = _quantize_weight_mxfp(w, "layer", bits=4, group_size=32, data_type="mx_fp")
        assert out["layer.weight_packed"].shape == (64, 64)  # in_features / 2
        assert out["layer.weight_packed"].dtype == torch.uint8
        assert out["layer.weight_scale"].shape == (64, 4)  # in_features / group_size
        assert out["layer.weight_scale"].dtype == torch.uint8

    def test_quantize_weight_mxfp8_shapes(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        out = _quantize_weight_mxfp(w, "layer", bits=8, group_size=32, data_type="mx_fp")
        assert out["layer.weight"].shape == (64, 128)
        assert out["layer.weight"].dtype == torch.float8_e4m3fn
        assert out["layer.weight_scale"].shape == (64, 4)
        assert out["layer.weight_scale"].dtype == torch.uint8

    @require_compressed_tensors
    @pytest.mark.parametrize("scheme,fmt", [("MXFP4", "mxfp4-pack-quantized"), ("MXFP8", "mxfp8-quantized")])
    def test_e2e_mxfp(self, tmp_path, scheme, fmt):
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.fc1.weight": torch.randn(512, 128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        _ModelFreeCompressorCore(model_name_or_path=model_dir, output_dir=output_dir, scheme=scheme).run()
        qc = _read_qconfig(output_dir)
        assert qc["format"] == fmt
        assert qc["quant_method"] == "compressed-tensors"
        assert "lm_head" in qc["ignore"]
        keys = _read_output_keys(output_dir)
        # MXFP4 produces weight_packed, MXFP8 produces weight
        if scheme == "MXFP4":
            assert "model.layers.0.fc1.weight_packed" in keys
        else:
            assert "model.layers.0.fc1.weight" in keys
        assert "model.layers.0.fc1.weight_scale" in keys
        # lm_head stays full precision
        assert "lm_head.weight" in keys
        assert "lm_head.weight_packed" not in keys

    @require_compressed_tensors
    def test_mxfp4_via_autoround_api(self, tmp_path):
        tensors = {"model.layers.0.fc1.weight": torch.randn(128, 128)}
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="MXFP4", model_free=True).quantize_and_save(
            output_dir, format="llm_compressor"
        )
        qc = _read_qconfig(output_dir)
        assert qc["format"] == "mxfp4-pack-quantized"

    @require_compressed_tensors
    def test_process_shard_mxfp(self, tmp_path):
        shard_path = str(tmp_path / "shard.safetensors")
        save_file({"layer.fc1.weight": torch.randn(64, 128)}, shard_path)
        scheme = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        output, quantized, _ = _process_shard(shard_path, scheme, {}, [])
        assert "layer.fc1" in quantized
        assert "layer.fc1.weight_packed" in output
        assert "layer.fc1.weight_scale" in output

    @require_compressed_tensors
    def test_build_mxfp_mixed_config_uniform(self):
        """Single-scheme path: no layer_config overrides → uniform format."""
        default = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        quantized = ["model.layers.0.fc1", "model.layers.0.fc2"]
        ignored = ["lm_head"]
        cfg = _build_mxfp_quantization_config(default, quantized, ignored, layer_config={})
        assert cfg["format"] == "mxfp4-pack-quantized"
        assert "lm_head" in cfg["ignore"]
        assert len(cfg["config_groups"]) == 1

    @require_compressed_tensors
    def test_build_mxfp_mixed_config_two_groups(self):
        """Mixed MXFP4+MXFP8: override layers get explicit targets; default gets ["Linear"]."""
        default = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        layer_config = {
            "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
            "model.layers.0.self_attn.k_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
        }
        quantized = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.mlp.fc1",
            "model.layers.0.mlp.fc2",
        ]
        ignored = ["lm_head"]
        cfg = _build_mxfp_quantization_config(default, quantized, ignored, layer_config=layer_config)

        assert cfg["format"] == "mixed-precision"
        assert len(cfg["config_groups"]) == 2

        # MXFP8 group (override, should come first) — explicit targets
        mxfp8_group = next(g for g in cfg["config_groups"].values() if g["format"] == "mxfp8-quantized")
        assert set(mxfp8_group["targets"]) == {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        }

        # MXFP4 group (default) — catch-all
        mxfp4_group = next(g for g in cfg["config_groups"].values() if g["format"] == "mxfp4-pack-quantized")
        assert mxfp4_group["targets"] == ["Linear"]

    @require_compressed_tensors
    def test_build_mxfp_mixed_config_adds_routedexperts_for_expert_group(self):
        """Expert layers in a non-Linear group should get RoutedExperts prepended."""
        default = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        layer_config = {
            "model.layers.0.mlp.experts.0.down_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
            "model.layers.0.mlp.experts.1.down_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
        }
        quantized = [
            "model.layers.0.mlp.experts.0.down_proj",
            "model.layers.0.mlp.experts.1.down_proj",
            "model.layers.0.mlp.gate_proj",
        ]
        cfg = _build_mxfp_quantization_config(default, quantized, ignored_layers=[], layer_config=layer_config)

        assert cfg["format"] == "mixed-precision"
        mxfp8_group = next(g for g in cfg["config_groups"].values() if g["format"] == "mxfp8-quantized")
        # RoutedExperts must be first
        assert mxfp8_group["targets"][0] == "RoutedExperts"
        assert "model.layers.0.mlp.experts.0.down_proj" in mxfp8_group["targets"]
        assert "model.layers.0.mlp.experts.1.down_proj" in mxfp8_group["targets"]
        # default MXFP4 group must NOT have RoutedExperts
        mxfp4_group = next(g for g in cfg["config_groups"].values() if g["format"] == "mxfp4-pack-quantized")
        assert "RoutedExperts" not in mxfp4_group["targets"]

    @require_compressed_tensors
    def test_build_mxfp_mixed_config_no_routedexperts_without_expert_layers(self):
        """Non-expert explicit group must not get RoutedExperts."""
        default = {"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"}
        layer_config = {
            "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
        }
        quantized = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.fc1",
        ]
        cfg = _build_mxfp_quantization_config(default, quantized, ignored_layers=[], layer_config=layer_config)

        mxfp8_group = next(g for g in cfg["config_groups"].values() if g["format"] == "mxfp8-quantized")
        assert "RoutedExperts" not in mxfp8_group["targets"]

    @require_compressed_tensors
    def test_e2e_mxfp_mixed(self, tmp_path):
        """End-to-end: default MXFP4 with some layers overridden to MXFP8."""
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
            "model.layers.0.mlp.fc1.weight": torch.randn(512, 128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        layer_config = {
            "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
            "model.layers.0.self_attn.k_proj": {"bits": 8, "group_size": 32, "data_type": "mx_fp"},
        }
        _ModelFreeCompressorCore(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="MXFP4",
            layer_config=layer_config,
        ).run()

        qc = _read_qconfig(output_dir)
        assert qc["format"] == "mixed-precision"
        assert qc["quant_method"] == "compressed-tensors"
        assert len(qc["config_groups"]) == 2
        assert "lm_head" in qc["ignore"]

        keys = _read_output_keys(output_dir)
        # MXFP8-overridden layers → .weight (float8_e4m3fn) + .weight_scale
        assert "model.layers.0.self_attn.q_proj.weight" in keys
        assert "model.layers.0.self_attn.q_proj.weight_scale" in keys
        # MXFP4-default layers → .weight_packed + .weight_scale
        assert "model.layers.0.mlp.fc1.weight_packed" in keys
        assert "model.layers.0.mlp.fc1.weight_scale" in keys
        # lm_head stays full precision
        assert "lm_head.weight" in keys
        assert "lm_head.weight_packed" not in keys


# ===========================================================================
#  deepseek_v4 MXFP-quantized source models
# ===========================================================================

_DEEPSEEK_V4_CFG = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}


def _make_deepseek_v4_mxfp8(out_f, in_f, block_h, block_w):
    """Build deepseek_v4-style MXFP8 source tensors.

    Returns ``(weight_fp8, scale_e8m0_coarse)``:

    * ``weight_fp8``         — ``float8_e4m3fn``, shape ``[out_f, in_f]``.
    * ``scale_e8m0_coarse``  — ``uint8`` E8M0, *coarse* 2D shape
      ``[out_f // block_h, in_f // block_w]`` (all exponents = bias 127, i.e.
      scale 1.0, to keep the round-trip deterministic).
    """
    weight_fp8 = torch.randn(out_f, in_f, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale = torch.full((out_f // block_h, in_f // block_w), 127, dtype=torch.uint8)
    return weight_fp8, scale


class TestExpandE8M0BlockScale:
    """The coarse-block → per-group E8M0 scale expansion helper."""

    def test_expand_repeat_interleave(self):
        # Coarse [2, 2] block scale for a [64, 128] weight, group_size=32.
        scale = torch.tensor([[100, 101], [102, 103]], dtype=torch.uint8)
        out = _expand_e8m0_block_scale(scale, out_features=64, in_features=128, group_size=32)
        assert out.shape == (64, 4)  # 128 // 32 == 4 groups
        assert out.dtype == torch.uint8
        # Row 0 (first 32 rows) maps to coarse row 0; cols 0..1 → coarse col 0, 2..3 → col 1.
        assert out[0].tolist() == [100, 100, 101, 101]
        assert out[63].tolist() == [102, 102, 103, 103]

    def test_expand_noop_when_already_fine(self):
        scale = torch.full((64, 4), 127, dtype=torch.uint8)
        out = _expand_e8m0_block_scale(scale, out_features=64, in_features=128, group_size=32)
        assert out.shape == (64, 4) and torch.equal(out, scale)

    def test_expand_invalid_shape_raises(self):
        scale = torch.full((3, 4), 127, dtype=torch.uint8)
        with pytest.raises(ValueError):
            _expand_e8m0_block_scale(scale, out_features=64, in_features=128, group_size=32)


class TestDeepseekV4MXFP8Source:
    """deepseek_v4 source models stored as float8 weights + coarse E8M0 scales."""

    def test_resolve_model_type(self):
        core = _ModelFreeCompressorCore(model_name_or_path="x", output_dir="o", scheme="MXFP8")
        core.config = _DEEPSEEK_V4_CFG
        core._resolve_model_type()
        assert core.model_type == "deepseek_v4"

    def test_resolve_model_type_negative(self):
        core = _ModelFreeCompressorCore(model_name_or_path="x", output_dir="o", scheme="MXFP8")
        core.config = _LLAMA_CFG
        core._resolve_model_type()
        assert core.model_type == "llama"


# ===========================================================================
#  llm-compressor MXFP source models (generic, e.g. Qwen3-MXFP4-MXFP8)
# ===========================================================================

_LLMCOMPRESSOR_MXFP_CFG_FP8 = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "quantization_config": {"quant_method": "compressed-tensors", "format": "mxfp8-quantized"},
}
_LLMCOMPRESSOR_MIXED_CFG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "quantization_config": {"quant_method": "compressed-tensors", "format": "mixed-precision"},
}


class TestLLMCompressorMXFPSource:
    """llm-compressor MXFP8/MXFP4 source models (generic non-deepseek_v4 path)."""

    def test_resolve_model_type_qwen3(self):
        core = _ModelFreeCompressorCore(model_name_or_path="x", output_dir="o", scheme="MXFP8")
        core.config = _LLMCOMPRESSOR_MXFP_CFG_FP8
        core._resolve_model_type()
        assert core.model_type == "qwen3"

    def test_resolve_model_type_mixed(self):
        core = _ModelFreeCompressorCore(model_name_or_path="x", output_dir="o", scheme="MXFP8")
        core.config = _LLMCOMPRESSOR_MIXED_CFG
        core._resolve_model_type()
        assert core.model_type == "qwen3"

    def test_resolve_model_type_negative_not_compressed_tensors(self):
        core = _ModelFreeCompressorCore(model_name_or_path="x", output_dir="o", scheme="MXFP8")
        core.config = _LLAMA_CFG
        core._resolve_model_type()
        assert core.model_type == "llama"

    def test_passthrough_mxfp8_same_target(self):
        """MXFP8 source + MXFP8 target → passthrough (bytes preserved)."""
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {
            "layer.weight": weight_fp8.clone(),
            "layer.weight_scale": weight_scale.clone(),
        }
        matcher = _matcher(default={"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)

        assert layers == ["layer"]
        assert passthrough["layer.weight"].dtype == torch.float8_e4m3fn
        assert torch.equal(passthrough["layer.weight"].view(torch.uint8), weight_fp8.view(torch.uint8))
        assert passthrough["layer.weight_scale"].dtype == torch.uint8
        assert "layer.weight" not in raw_out
        assert "layer.weight_scale" not in raw_out

    def test_passthrough_mxfp4_same_target(self):
        """MXFP4 source + MXFP4 target → passthrough."""
        weight_packed = torch.randint(0, 255, (64, 64), dtype=torch.uint8)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {
            "layer.weight_packed": weight_packed.clone(),
            "layer.weight_scale": weight_scale.clone(),
        }
        matcher = _matcher(default={"bits": 4, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)

        assert layers == ["layer"]
        assert torch.equal(passthrough["layer.weight_packed"], weight_packed)
        assert passthrough["layer.weight_scale"].dtype == torch.uint8
        assert "layer.weight_packed" not in raw_out

    def test_mxfp8_dequant_when_int_target(self):
        """MXFP8 source + int target → dequantized to bf16 .weight."""
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)  # scale 1.0
        raw = {
            "layer.weight": weight_fp8.clone(),
            "layer.weight_scale": weight_scale.clone(),
        }
        matcher = _matcher(default={"bits": 4, "group_size": 128, "sym": True, "data_type": "int"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)

        assert layers == [] and passthrough == {}
        assert raw_out["layer.weight"].dtype == torch.bfloat16
        assert torch.allclose(raw_out["layer.weight"], weight_fp8.to(torch.bfloat16))
        assert "layer.weight_scale" not in raw_out

    def test_mixed_passthrough_and_dequant(self):
        """Mixed: MXFP8 layer passthrough + MXFP4 layer dequanted (target MXFP8)."""
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_fp8 = torch.full((64, 4), 127, dtype=torch.uint8)
        weight_packed = torch.randint(0, 255, (64, 64), dtype=torch.uint8)
        scale_packed = torch.full((64, 4), 127, dtype=torch.uint8)
        raw = {
            "attn.weight": weight_fp8.clone(),
            "attn.weight_scale": scale_fp8.clone(),
            "mlp.weight_packed": weight_packed.clone(),
            "mlp.weight_scale": scale_packed.clone(),
        }
        matcher = _matcher(default={"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)

        assert "attn" in layers  # fp8 layer passthrough
        assert passthrough["attn.weight"].dtype == torch.float8_e4m3fn
        # mlp fp4 layer dequantized because target is MXFP8 (bits=8 != 4)
        assert "mlp" not in layers
        assert raw_out["mlp.weight"].dtype == torch.bfloat16
        assert raw_out["mlp.weight"].shape == (64, 128)

    def test_noop_without_mxfp_tensors(self):
        """No MXFP tensors → input returned unchanged."""
        raw = {"layer.weight": torch.randn(64, 128, dtype=torch.bfloat16)}
        matcher = _matcher(default={"bits": 8, "group_size": 32, "data_type": "mx_fp"})
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw, matcher)
        assert raw_out is raw and passthrough == {} and layers == []

    @require_compressed_tensors
    def test_e2e_mxfp8_passthrough(self, tmp_path):
        """End-to-end: MXFP8 source + MXFP8 target → passthrough, weight bytes unchanged."""
        weight_fp8 = torch.randn(128, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.full((128, 4), 127, dtype=torch.uint8)
        tensors = {
            "model.layers.0.mlp.fc1.weight": weight_fp8,
            "model.layers.0.mlp.fc1.weight_scale": weight_scale,
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLMCOMPRESSOR_MXFP_CFG_FP8, tensors)
        output_dir = str(tmp_path / "output")
        _ModelFreeCompressorCore(model_name_or_path=model_dir, output_dir=output_dir, scheme="MXFP8").run()

        qc = _read_qconfig(output_dir)
        assert qc["format"] == "mxfp8-quantized"
        assert qc["quant_method"] == "compressed-tensors"
        assert "lm_head" in qc["ignore"]

        wp = ws = None
        for f in os.listdir(output_dir):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(output_dir, f), framework="pt") as sf:
                    if "model.layers.0.mlp.fc1.weight" in sf.keys():
                        wp = sf.get_tensor("model.layers.0.mlp.fc1.weight")
                        ws = sf.get_tensor("model.layers.0.mlp.fc1.weight_scale")
        assert wp.dtype == torch.float8_e4m3fn
        assert torch.equal(wp.view(torch.uint8), weight_fp8.view(torch.uint8))
        assert ws.dtype == torch.uint8

    def test_convert_passthrough_when_target_mxfp8(self):
        """Target MXFP8 → converted tensors emitted directly (weight bytes preserved)."""
        weight_fp8, scale = _make_deepseek_v4_mxfp8(64, 128, block_h=32, block_w=64)
        raw = {"layer.weight": weight_fp8.clone(), "layer.scale": scale.clone()}
        matcher = _matcher(default={"bits": 8, "group_size": 32, "sym": True, "data_type": "mx_fp"})
        raw_out, state = _preprocess_model_type_source_tensors(raw, model_type="deepseek_v4")
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw_out, matcher, source_state=state)

        assert layers == ["layer"]
        # weight kept as float8 under .weight; scale expanded to per-group uint8.
        assert passthrough["layer.weight"].dtype == torch.float8_e4m3fn
        assert torch.equal(passthrough["layer.weight"].view(torch.uint8), weight_fp8.view(torch.uint8))
        assert passthrough["layer.weight_scale"].dtype == torch.uint8
        assert passthrough["layer.weight_scale"].shape == (64, 4)
        # source tensors consumed, nothing left in raw.
        assert "layer.weight" not in raw_out
        assert "layer.scale" not in raw_out

    def test_convert_dequant_when_target_int(self):
        """Non-MXFP8 target → tensors dequantized to bfloat16 .weight."""
        weight_fp8, scale = _make_deepseek_v4_mxfp8(64, 128, block_h=32, block_w=64)
        raw = {"layer.weight": weight_fp8.clone(), "layer.scale": scale.clone()}
        matcher = _matcher(default={"bits": 4, "group_size": 128, "sym": True, "data_type": "int"})
        raw_out, state = _preprocess_model_type_source_tensors(raw, model_type="deepseek_v4")
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw_out, matcher, source_state=state)

        assert layers == [] and passthrough == {}
        assert raw_out["layer.weight"].dtype == torch.bfloat16
        assert raw_out["layer.weight"].shape == (64, 128)
        # scale 1.0 → dequant weight equals fp8 cast to bf16.
        assert torch.allclose(raw_out["layer.weight"], weight_fp8.to(torch.bfloat16))
        assert "layer.scale" not in raw_out and "layer.weight_scale" not in raw_out

    def test_convert_noop_without_quantized(self):
        """No float8/packed weights → input returned unchanged."""
        raw = {"layer.weight": torch.randn(64, 128, dtype=torch.bfloat16)}
        matcher = _matcher(default={"bits": 8, "group_size": 32, "data_type": "mx_fp"})
        raw_out, state = _preprocess_model_type_source_tensors(raw, model_type="deepseek_v4")
        assert state == {}
        raw_out, passthrough, layers = _handle_mxfp_source_tensors(raw_out, matcher, source_state=state)
        assert raw_out is raw and passthrough == {} and layers == []

    def test_dequant_mxfp_tensors_mxfp8(self):
        """Generic MXFP dequant: float8 .weight + .weight_scale → bf16."""
        weight_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)  # scale 1.0
        raw = {"layer.weight": weight_fp8.clone(), "layer.weight_scale": weight_scale.clone()}
        out = _dequant_mxfp_tensors(raw)
        assert out["layer.weight"].dtype == torch.bfloat16
        assert torch.allclose(out["layer.weight"], weight_fp8.to(torch.bfloat16))
        assert "layer.weight_scale" not in out

    @require_compressed_tensors
    def test_e2e_passthrough_mxfp8_target(self, tmp_path):
        """deepseek_v4 source + MXFP8 target → passthrough preserves weight bytes."""
        weight_fp8, scale = _make_deepseek_v4_mxfp8(128, 128, block_h=32, block_w=64)
        tensors = {
            "model.layers.0.mlp.fc1.weight": weight_fp8,
            "model.layers.0.mlp.fc1.scale": scale,
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _DEEPSEEK_V4_CFG, tensors)
        output_dir = str(tmp_path / "output")
        _ModelFreeCompressorCore(model_name_or_path=model_dir, output_dir=output_dir, scheme="MXFP8").run()

        qc = _read_qconfig(output_dir)
        assert qc["format"] == "mxfp8-quantized"
        assert qc["quant_method"] == "compressed-tensors"
        assert "lm_head" in qc["ignore"]

        # Read back the converted tensors and verify the weight bytes are unchanged.
        wp = ws = None
        for f in os.listdir(output_dir):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(output_dir, f), framework="pt") as sf:
                    if "model.layers.0.mlp.fc1.weight" in sf.keys():
                        wp = sf.get_tensor("model.layers.0.mlp.fc1.weight")
                        ws = sf.get_tensor("model.layers.0.mlp.fc1.weight_scale")
        assert wp.dtype == torch.float8_e4m3fn
        assert torch.equal(wp.view(torch.uint8), weight_fp8.view(torch.uint8))
        assert ws.dtype == torch.uint8 and ws.shape == (128, 4)

    def test_e2e_dequant_int_target(self, tmp_path):
        """deepseek_v4 source + W4A16 target → dequant then RTN requantize (qweight)."""
        weight_fp8, scale = _make_deepseek_v4_mxfp8(128, 128, block_h=32, block_w=64)
        tensors = {
            "model.layers.0.mlp.fc1.weight": weight_fp8,
            "model.layers.0.mlp.fc1.scale": scale,
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _DEEPSEEK_V4_CFG, tensors)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="W4A16", model_free=True).quantize_and_save(output_dir)

        qc = _read_qconfig(output_dir)
        assert qc["quant_method"] == "auto-round" and qc["bits"] == 4
        keys = _read_output_keys(output_dir)
        assert "model.layers.0.mlp.fc1.qweight" in keys
        # the raw source tensors must not leak into the output
        assert "model.layers.0.mlp.fc1.scale" not in keys
        assert "model.layers.0.mlp.fc1.weight" not in keys


# ===========================================================================
#  Scheme validation
# ===========================================================================


_SUPPORTED = ["W2A16", "W2A16G32", "W2A16G64", "W4A16", "W4A16_MIXED", "W8A16", "MXFP4", "MXFP8"]
_UNSUPPORTED = [
    "W3A16",
    "FPW8A16",
    "BF16",
    "MXINT4",
    "NVFP4",
    "FP8_BLOCK",
    "FP8_STATIC",
    "INT8_W8A8",
    "MXFP4_RCEIL",
    "MXFP8_RCEIL",
]


class TestSchemeValidation:
    @pytest.mark.parametrize("name", _SUPPORTED)
    def test_supported(self, tmp_path, name):
        """Each supported preset must resolve and quantize without error.

        This exercises the same scheme-resolution code (``_parse_scheme`` /
        ``_parse_layer_config`` / ``_build_ignore_patterns``) used by the real
        pipeline, then quantizes a shard directly via ``_process_shard`` —
        skipping the multiprocessing shard pipeline (already covered by the
        full end-to-end tests in ``TestModelFreeQuantize`` / ``TestModelFreeMXFP``)
        to keep this parametrized check fast.
        """
        if name.startswith("MXFP"):
            pytest.importorskip("compressed_tensors", reason="test requires compressed-tensors")

        core = _ModelFreeCompressorCore(model_name_or_path="unused", output_dir=str(tmp_path), scheme=name)
        core._parse_scheme()
        core._parse_layer_config()
        core._build_ignore_patterns()

        shard_path = str(tmp_path / f"shard_{name}.safetensors")
        save_file({"model.layers.0.mlp.fc1.weight": torch.randn(64, 128)}, shard_path)
        output, quantized, _ignored = _process_shard(
            shard_path,
            default_scheme=core.default_scheme,
            layer_config=core.layer_config,
            ignore_patterns=core.ignore_patterns,
        )
        assert "model.layers.0.mlp.fc1" in quantized
        if name.startswith("MXFP"):
            assert "model.layers.0.mlp.fc1.weight_scale" in output
        else:
            assert "model.layers.0.mlp.fc1.qweight" in output

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
        from auto_round.cli.main import tune
        from auto_round.cli.parser import build_quantize_parser

        args = build_quantize_parser().parse_args(
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
        from auto_round.cli.parser import build_quantize_parser

        args = build_quantize_parser().parse_args(
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


# ===========================================================================
#  Copy metadata & subfolder handling
# ===========================================================================


def _make_diffusion_model_dir(tmp_path, transformer_config, transformer_tensors):
    """Create a minimal diffusion model directory layout.

    Layout::

        root/
            model_index.json
            transformer/
                config.json
                model.safetensors
            vae/
                config.json
            scheduler/
                scheduler_config.json
            tokenizer/
                tokenizer.json
                nested/
                    vocab.txt
    """
    root_dir = str(tmp_path / "diffusion_model")
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, "model_index.json"), "w") as f:
        json.dump({"_class_name": "FluxPipeline"}, f)

    # transformer component with weights
    transformer_dir = os.path.join(root_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(transformer_config, f)
    save_file(transformer_tensors, os.path.join(transformer_dir, "model.safetensors"))

    # vae component
    vae_dir = os.path.join(root_dir, "vae")
    os.makedirs(vae_dir, exist_ok=True)
    with open(os.path.join(vae_dir, "config.json"), "w") as f:
        json.dump({"_class_name": "AutoencoderKL"}, f)

    # scheduler component
    sched_dir = os.path.join(root_dir, "scheduler")
    os.makedirs(sched_dir, exist_ok=True)
    with open(os.path.join(sched_dir, "scheduler_config.json"), "w") as f:
        json.dump({"_class_name": "FlowMatchEulerDiscreteScheduler"}, f)

    # tokenizer component with nested subdir
    tok_dir = os.path.join(root_dir, "tokenizer")
    os.makedirs(os.path.join(tok_dir, "nested"), exist_ok=True)
    with open(os.path.join(tok_dir, "tokenizer.json"), "w") as f:
        json.dump({"type": "BPE"}, f)
    with open(os.path.join(tok_dir, "nested", "vocab.txt"), "w") as f:
        f.write("hello\nworld\n")

    return root_dir


_TRANSFORMER_CONFIG = {
    "architectures": ["FluxTransformer2DModel"],
    "model_type": "flux",
    "hidden_size": 128,
    "num_hidden_layers": 1,
}

_TRANSFORMER_TENSORS = {
    "transformer_blocks.0.attn.to_q.weight": torch.randn(128, 128),
    "transformer_blocks.0.attn.to_k.weight": torch.randn(128, 128),
    "transformer_blocks.0.ff.net.0.proj.weight": torch.randn(512, 128),
    "transformer_blocks.0.ff.net.2.weight": torch.randn(128, 512),
}


class TestCopyMetadataSubfolders:
    """Tests for _copy_metadata_files including subdirectory handling."""

    def test_non_diffusion_copies_subfolders(self, tmp_path):
        """Non-diffusion model: subdirectories should be copied to output."""
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        # Add a subdirectory with files
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        os.makedirs(os.path.join(tokenizer_dir, "nested"), exist_ok=True)
        with open(os.path.join(tokenizer_dir, "tokenizer.json"), "w") as f:
            json.dump({"type": "BPE"}, f)
        with open(os.path.join(tokenizer_dir, "nested", "vocab.txt"), "w") as f:
            f.write("hello\nworld\n")

        output_dir = str(tmp_path / "output")
        core = _ModelFreeCompressorCore(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )
        core.run()

        assert os.path.isdir(os.path.join(output_dir, "tokenizer"))
        assert os.path.isfile(os.path.join(output_dir, "tokenizer", "tokenizer.json"))
        assert os.path.isdir(os.path.join(output_dir, "tokenizer", "nested"))
        assert os.path.isfile(os.path.join(output_dir, "tokenizer", "nested", "vocab.txt"))

    def test_diffusion_copies_subfolders(self, tmp_path):
        """Diffusion model: non-transformer subdirectories should be copied."""
        root_dir = _make_diffusion_model_dir(tmp_path, _TRANSFORMER_CONFIG, _TRANSFORMER_TENSORS)
        output_dir = str(tmp_path / "output")

        core = _ModelFreeCompressorCore(
            model_name_or_path=root_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )
        core.run()

        # Non-transformer subdirs must be present
        assert os.path.isdir(os.path.join(output_dir, "vae"))
        assert os.path.isfile(os.path.join(output_dir, "vae", "config.json"))
        assert os.path.isdir(os.path.join(output_dir, "scheduler"))
        assert os.path.isfile(os.path.join(output_dir, "scheduler", "scheduler_config.json"))
        assert os.path.isdir(os.path.join(output_dir, "tokenizer"))
        assert os.path.isfile(os.path.join(output_dir, "tokenizer", "tokenizer.json"))
        assert os.path.isfile(os.path.join(output_dir, "tokenizer", "nested", "vocab.txt"))
        # Root-level file
        assert os.path.isfile(os.path.join(output_dir, "model_index.json"))
        # Quantized transformer must also exist
        assert os.path.isdir(os.path.join(output_dir, "transformer"))
        assert os.path.isfile(os.path.join(output_dir, "transformer", "config.json"))

    def test_diffusion_does_not_overwrite_quantized_transformer(self, tmp_path):
        """Copying subfolders must not overwrite the quantized transformer."""
        root_dir = _make_diffusion_model_dir(tmp_path, _TRANSFORMER_CONFIG, _TRANSFORMER_TENSORS)
        output_dir = str(tmp_path / "output")

        core = _ModelFreeCompressorCore(
            model_name_or_path=root_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )
        core.run()

        # The output transformer/ should contain quantized weights, not
        # the original model.safetensors from the source.
        transformer_out = os.path.join(output_dir, "transformer")
        out_files = os.listdir(transformer_out)
        # The original single shard would have been renamed to model.safetensors
        # by _write_index_file; confirm it has quantized tensor names.
        keys = set()
        for f in out_files:
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(transformer_out, f), framework="pt") as sf:
                    keys.update(sf.keys())
        assert any(
            k.endswith(".qweight") for k in keys
        ), f"Quantized transformer should contain .qweight tensors, got: {keys}"


# ===========================================================================
#  Shard parallelism — _resolve_shard_parallelism + end-to-end with non-divisible counts
# ===========================================================================


class TestResolveShardParallelism:
    """Tests for the automatic and env-controlled shard parallelism policy."""

    @staticmethod
    def _core_with_n_shards(n: int) -> _ModelFreeCompressorCore:
        core = _ModelFreeCompressorCore.__new__(_ModelFreeCompressorCore)
        core.shard_names = [f"shard_{i:02d}.safetensors" for i in range(n)]
        core.shard_parallelism = 1
        return core

    def test_auto_policy_formula(self, monkeypatch):
        monkeypatch.delenv("AR_MODEL_FREE_SHARD_PARALLELISM", raising=False)
        cases = [
            (1, 1),  # 1 // 4 = 0 -> min 1
            (3, 1),  # 3 // 4 = 0 -> min 1
            (4, 1),  # 4 // 4 = 1
            (8, 2),  # 8 // 4 = 2
            (12, 3),  # 12 // 4 = 3
            (40, 10),  # 40 // 4 = 10 (at cap)
            (80, 10),  # 80 // 4 = 20 -> capped at 10
        ]
        for n, expected in cases:
            core = self._core_with_n_shards(n)
            p, src = core._resolve_shard_parallelism()
            assert p == expected, f"n={n}: expected {expected}, got {p}"
            assert "auto" in src

    def test_env_override_respected(self, monkeypatch):
        monkeypatch.setenv("AR_MODEL_FREE_SHARD_PARALLELISM", "7")
        core = self._core_with_n_shards(25)
        p, src = core._resolve_shard_parallelism()
        assert p == 7
        assert "env=7" in src

    def test_env_capped_at_shard_count(self, monkeypatch):
        monkeypatch.setenv("AR_MODEL_FREE_SHARD_PARALLELISM", "100")
        core = self._core_with_n_shards(3)
        p, _ = core._resolve_shard_parallelism()
        assert p == 3

    def test_env_below_1_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("AR_MODEL_FREE_SHARD_PARALLELISM", "0")
        core = self._core_with_n_shards(25)
        p, src = core._resolve_shard_parallelism()
        assert p == 25 // 4
        assert "invalid" in src

    def test_env_invalid_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("AR_MODEL_FREE_SHARD_PARALLELISM", "notanumber")
        core = self._core_with_n_shards(25)
        p, src = core._resolve_shard_parallelism()
        assert p == 25 // 4  # auto formula: shard_count // 4
        assert "invalid" in src

    def test_nondivisible_shard_count_all_shards_processed(self, tmp_path, monkeypatch):
        """Parallelism that does not evenly divide the shard count must still
        process every shard and produce correct output.

        7 shards with parallelism=3 → 7 % 3 == 1 (non-divisible).
        """
        monkeypatch.setenv("AR_MODEL_FREE_SHARD_PARALLELISM", "3")

        # Build 7 shards of simple linear weights
        layer_names = [f"model.layers.{i}.fc.weight" for i in range(7)]
        model_dir = str(tmp_path / "source")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(_SIMPLE_CONFIG, f)

        weight_map = {}
        for shard_idx, layer_name in enumerate(layer_names):
            shard_filename = f"model-{shard_idx + 1:05d}-of-{len(layer_names):05d}.safetensors"
            save_file({layer_name: torch.randn(128, 128)}, os.path.join(model_dir, shard_filename))
            weight_map[layer_name] = shard_filename
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)

        output_dir = str(tmp_path / "output")
        core = _ModelFreeCompressorCore(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
            quant_lm_head=True,
        )
        core.run()

        # Every layer must appear as quantized in the output
        out_keys = _read_output_keys(output_dir)
        for layer_name in layer_names:
            base = layer_name.replace(".weight", "")
            assert f"{base}.qweight" in out_keys, (
                f"Layer '{base}' missing from output after non-divisible shard processing. "
                f"Output keys: {sorted(out_keys)[:20]}"
            )

        # The index must reference exactly 7 shards
        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        unique_shards = set(index["weight_map"].values())
        assert len(unique_shards) == 7, f"Expected 7 output shards, got {len(unique_shards)}: {unique_shards}"


# ===========================================================================
#  Model-free + AutoScheme (two-phase delta-loss selection + packing)
# ===========================================================================


class TestModelFreeAutoScheme:
    """Model-free support for ``AutoScheme`` mixed-bit selection."""

    def test_looks_like_auto_scheme(self):
        from auto_round import AutoScheme

        assert _looks_like_auto_scheme(AutoScheme(avg_bits=3, options=("W2A16", "W4A16")))
        assert not _looks_like_auto_scheme("W4A16")
        assert not _looks_like_auto_scheme(QuantizationScheme(bits=4))

    def test_validate_options_int_family(self):
        from auto_round import AutoScheme

        assert _validate_auto_scheme_options(AutoScheme(avg_bits=3, options=("W2A16", "W4A16", "W8A16"))) == "int"

    def test_validate_options_mxfp_family(self):
        from auto_round import AutoScheme

        assert _validate_auto_scheme_options(AutoScheme(avg_bits=6, options=("MXFP4", "MXFP8"))) == "mx_fp"

    def test_validate_options_mixed_family_raises(self):
        from auto_round import AutoScheme

        with pytest.raises(ValueError, match="mix INT and MXFP"):
            _validate_auto_scheme_options(AutoScheme(avg_bits=4, options=("W4A16", "MXFP4")))

    @pytest.mark.parametrize(
        "options",
        [
            ("W3A16", "W4A16"),
            ("GGUF:Q4_K_M", "W8A16"),
            ("NVFP4", "W4A16"),
            ("MXFP4_RCEIL", "MXFP4"),
        ],
    )
    def test_validate_options_unsupported_raises(self, options):
        from auto_round import AutoScheme

        with pytest.raises(ValueError, match="unsupported option"):
            _validate_auto_scheme_options(AutoScheme(avg_bits=4, options=options))

    def test_convert_layer_config(self):
        generated = {
            "model.layers.0.q_proj": {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
            "model.layers.0.k_proj": {"bits": 2, "group_size": 128, "sym": True, "data_type": "int"},
            "model.layers.0.v_proj": {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
            "model.embed_tokens": {"bits": 16, "group_size": 128, "sym": True, "data_type": "int"},
        }
        base_scheme, per_layer, fp16_layers = _convert_auto_scheme_layer_config(generated)
        # Most common quantized scheme (4-bit) becomes the base.
        assert base_scheme.bits == 4 and base_scheme.group_size == 128
        assert per_layer["model.layers.0.k_proj"]["bits"] == 2
        assert "model.embed_tokens" not in per_layer
        assert fp16_layers == ["model.embed_tokens"]

    def test_e2e_int_auto_scheme(self, tmp_path, tiny_opt_model_path):
        from auto_round import AutoScheme

        output_dir = str(tmp_path / "output")
        scheme = AutoScheme(avg_bits=3.0, options=("W2A16", "W4A16", "W8A16"), nsamples=1)
        ar = AutoRound(model=tiny_opt_model_path, scheme=scheme, iters=0, model_free=True, nsamples=1)
        ar.quantize_and_save(output_dir, format="auto_round")

        qc = _read_qconfig(output_dir)
        assert qc["quant_method"] == "auto-round"
        assert qc["model_free"] is True
        # A genuine mix of bit-widths must have been selected across layers.
        extra = qc.get("extra_config", {})
        selected_bits = {qc["bits"]} | {v["bits"] for v in extra.values() if v.get("bits", 16) < 16}
        assert len(selected_bits) >= 2, f"expected mixed bit-widths, got {selected_bits}"

    @require_compressed_tensors
    def test_e2e_mxfp_auto_scheme(self, tmp_path, tiny_opt_model_path):
        from auto_round import AutoScheme

        output_dir = str(tmp_path / "output")
        scheme = AutoScheme(avg_bits=6.0, options=("MXFP4", "MXFP8"), nsamples=1)
        ar = AutoRound(model=tiny_opt_model_path, scheme=scheme, iters=0, model_free=True, nsamples=1)
        ar.quantize_and_save(output_dir, format="llm_compressor")

        qc = _read_qconfig(output_dir)
        assert qc["quant_method"] == "compressed-tensors"
        assert qc["provider"] == "auto-round"
