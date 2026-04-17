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

from auto_round.compressors.model_free import (
    _PatternMatcher,
    _process_shard,
    get_predefined_ignore_layers_from_config,
    model_free_quantize,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


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


def _matcher(ignore=None, layer_config=None, default=None):
    return _PatternMatcher(
        ignore if ignore is not None else [],
        layer_config if layer_config is not None else {},
        default if default is not None else {},
    )


# ===========================================================================
#  Test: _PatternMatcher
# ===========================================================================


class TestPatternMatcher:
    def test_ignore_substring_match(self):
        m = _matcher(ignore=["mlp"])
        assert m.should_ignore("model.layers.0.mlp.fc1.weight") is True
        assert m.should_ignore("model.layers.0.self_attn.q_proj.weight") is False

    def test_ignore_trailing_dot_no_partial_number_match(self):
        m = _matcher(ignore=["layers.4."])
        assert m.should_ignore("model.layers.4.mlp.fc1.weight") is True
        assert m.should_ignore("model.layers.45.mlp.fc1.weight") is False

    def test_skip_predefined(self):
        m = _matcher()
        assert m.should_skip("model.layers.0.shared_expert_gate.weight") is True
        assert m.should_skip("model.layers.0.mlp.gate.weight") is True
        assert m.should_skip("model.embed_tokens.weight") is True
        assert m.should_skip("model.layers.0.mlp.fc1.weight") is False

    def test_resolve_scheme_exact_and_regex(self):
        default = {"bits": 4, "group_size": 128, "sym": True}
        lc = {
            "model.layers.0.mlp.fc1": {"bits": 8, "group_size": 32},
            r".*k_proj": {"bits": 8},
        }
        m = _matcher(layer_config=lc, default=default)

        r = m.resolve_scheme("model.layers.0.mlp.fc1.weight")
        assert r["bits"] == 8 and r["group_size"] == 32

        r = m.resolve_scheme("model.layers.0.self_attn.k_proj.weight")
        assert r["bits"] == 8 and r["group_size"] == 128

        assert m.resolve_scheme("model.layers.0.mlp.fc2.weight") == default

    def test_resolve_scheme_bits16_returns_none(self):
        m = _matcher(
            layer_config={"model.layers.0.fc1": {"bits": 16}},
            default={"bits": 4, "group_size": 128},
        )
        assert m.resolve_scheme("model.layers.0.fc1.weight") is None


# ===========================================================================
#  Test: get_predefined_ignore_layers_from_config
# ===========================================================================


class TestGetPredefinedIgnoreLayers:
    def test_normal_model_empty(self):
        cfg = {"architectures": ["LlamaForCausalLM"]}
        assert get_predefined_ignore_layers_from_config(cfg) == []


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
        assert "model.layers.0.mlp.fc1.scales" in output
        assert "model.layers.0.mlp.fc1.bias" in output

    def test_ignores_and_skips(self, tmp_path):
        tensors = {
            "lm_head.weight": torch.randn(100, 128),
            "model.layers.0.mlp.gate.weight": torch.randn(8, 128),
        }
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(tensors, shard_path)

        output, quantized, ignored = _process_shard(shard_path, self.DEFAULT_SCHEME, {}, ["lm_head"])
        assert len(quantized) == 0
        assert "lm_head" in ignored
        assert "model.layers.0.mlp.gate" in ignored

    def test_fused_expert_split_and_quantized(self, tmp_path):
        N, I, H = 2, 64, 32
        shard_path = str(tmp_path / "shard.safetensors")
        save_file(
            {
                "model.layers.0.mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
                "model.layers.0.mlp.experts.down_proj": torch.randn(N, H, I),
            },
            shard_path,
        )

        output, quantized, _ = _process_shard(shard_path, self.DEFAULT_SCHEME, {}, [])
        assert not any("gate_up_proj" in k for k in output)
        for i in range(N):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                base = f"model.layers.0.mlp.experts.{i}.{proj}"
                assert base in quantized
                assert f"{base}.qweight" in output


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

        with open(os.path.join(output_dir, "config.json")) as f:
            cfg = json.load(f)
        qc = cfg["quantization_config"]
        assert qc["quant_method"] == "auto-round"
        assert qc["bits"] == 4
        assert qc["model_free"] is True

        # lm_head and embed should NOT be quantized by default
        all_keys = set()
        for fname in os.listdir(output_dir):
            if fname.endswith(".safetensors"):
                with safe_open(os.path.join(output_dir, fname), framework="pt") as f:
                    all_keys.update(f.keys())
        assert "lm_head.weight" in all_keys
        assert "lm_head.qweight" not in all_keys

    def test_ignore_layers_broad_substring(self, tmp_path):
        """ignore_layers='mlp' keeps ALL mlp layers in full precision."""
        tensors = {
            "model.layers.0.mlp.fc1.weight": torch.randn(512, 128),
            "model.layers.0.mlp.fc2.weight": torch.randn(128, 512),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "lm_head.weight": torch.randn(1000, 128),
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
            ignore_layers="mlp",
        )

        st_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
        with safe_open(os.path.join(output_dir, st_files[0]), framework="pt") as f:
            keys = set(f.keys())

        assert "model.layers.0.mlp.fc1.weight" in keys
        assert "model.layers.0.mlp.fc1.qweight" not in keys
        assert "model.layers.0.self_attn.q_proj.qweight" in keys

    def test_multi_shard(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS, multi_shard=True)
        output_dir = str(tmp_path / "output")

        model_free_quantize(
            model_name_or_path=model_dir,
            output_dir=output_dir,
            scheme="W4A16",
        )

        assert os.path.exists(os.path.join(output_dir, "model.safetensors.index.json"))

    def test_quant_lm_head(self, tmp_path):
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
        assert "lm_head.qweight" in all_keys

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

    def test_non_woq_scheme_raises(self, tmp_path):
        """Non-WOQ schemes (act_bits < 16) should be rejected."""
        with pytest.raises(ValueError, match="weight-only quantization"):
            model_free_quantize(
                model_name_or_path=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                scheme="MXFP4",
            )

    def test_group_size_asym_quantization(self, tmp_path):
        """Asymmetric quantization via QuantizationScheme."""
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
        assert cfg["quantization_config"]["sym"] is False
        assert cfg["quantization_config"]["group_size"] == 64


# ===========================================================================
#  Test: FP8 source model support
# ===========================================================================


class TestFP8SourceModel:
    def test_dequant_fp8_tensors(self):
        from auto_round.compressors.model_free import _dequant_fp8_tensors

        w = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        s = torch.tensor(0.5, dtype=torch.float32)
        raw = {
            "layer.weight": w,
            "layer.weight_scale_inv": s,
            "layer.bias": torch.randn(64),
        }
        result = _dequant_fp8_tensors(raw, block_size=None)
        assert result["layer.weight"].dtype == torch.bfloat16
        assert "layer.weight_scale_inv" not in result

    def test_no_fp8_is_noop(self):
        from auto_round.compressors.model_free import _dequant_fp8_tensors

        raw = {"layer.weight": torch.randn(64, 128)}
        assert _dequant_fp8_tensors(raw, block_size=None) is raw

    def test_process_shard_fp8(self, tmp_path):
        shard_path = str(tmp_path / "shard.safetensors")
        w = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        s = torch.tensor(1.0, dtype=torch.float32)
        save_file({"layer.weight": w, "layer.weight_scale_inv": s}, shard_path)

        output, quantized, _ = _process_shard(
            shard_path,
            {"bits": 4, "group_size": 128, "sym": True, "data_type": "int"},
            {},
            [],
            device="cpu",
            fp8_block_size=None,
        )
        assert "layer" in quantized
        assert "layer.qweight" in output
        assert "layer.weight_scale_inv" not in output
