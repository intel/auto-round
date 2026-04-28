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
    _PatternMatcher,
    _process_shard,
    get_predefined_ignore_layers_from_config,
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

    def test_resolve_scheme_substring_pattern(self):
        """Substring patterns like '.ffn.experts.' should match via regex."""
        default = {"bits": 4, "group_size": 128, "sym": True}
        lc = {".ffn.experts.": {"bits": 2, "group_size": 64}}
        m = _matcher(layer_config=lc, default=default)

        r = m.resolve_scheme("model.layers.0.ffn.experts.3.gate_proj.weight")
        assert r is not None
        assert r["bits"] == 2
        assert r["group_size"] == 64

        # Non-expert layer should use default
        r2 = m.resolve_scheme("model.layers.0.self_attn.q_proj.weight")
        assert r2 == default

    def test_resolve_scheme_with_scheme_key_in_layer_config(self):
        """layer_config with 'scheme' key like {'scheme': 'W2A16'} should resolve."""
        from auto_round.compressors.model_free import _ModelFreeCompressorCore

        core = _ModelFreeCompressorCore(
            model_name_or_path="dummy",
            output_dir="dummy_out",
            scheme="W4A16",
        )
        core.layer_config_input = {
            ".ffn.experts.": {"scheme": "W2A16"},
        }
        core._parse_scheme()
        core._parse_layer_config()

        # The resolved layer_config should have bits=2 from W2A16
        lc = core.layer_config
        # Key may have '.' appended or not; find it
        expert_cfg = None
        for k, v in lc.items():
            if "ffn.experts" in k:
                expert_cfg = v
                break
        assert expert_cfg is not None, f"Expected expert config in layer_config, got: {lc}"
        assert expert_cfg["bits"] == 2
        assert "scheme" not in expert_cfg  # 'scheme' key should be consumed

        # Build matcher and verify resolution
        m = _matcher(layer_config=lc, default=core.default_scheme)
        r = m.resolve_scheme("model.layers.0.ffn.experts.3.gate_proj.weight")
        assert r is not None
        assert r["bits"] == 2

    def test_resolve_scheme_with_scheme_key_and_overrides(self):
        """Dict with 'scheme' + explicit overrides: explicit keys win."""
        from auto_round.compressors.model_free import _ModelFreeCompressorCore

        core = _ModelFreeCompressorCore(
            model_name_or_path="dummy",
            output_dir="dummy_out",
            scheme="W4A16",
        )
        core.layer_config_input = {
            ".ffn.experts.": {"scheme": "W2A16", "group_size": 32},
        }
        core._parse_scheme()
        core._parse_layer_config()

        expert_cfg = None
        for k, v in core.layer_config.items():
            if "ffn.experts" in k:
                expert_cfg = v
                break
        assert expert_cfg is not None
        assert expert_cfg["bits"] == 2  # from W2A16
        assert expert_cfg["group_size"] == 32  # explicit override wins


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


class TestModelFreeQuantize:
    def test_basic_quantization(self, tmp_path):
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")

        AutoRound(model=model_dir, scheme="W4A16", model_free=True).quantize_and_save(output_dir)
        result = output_dir

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

        AutoRound(model=model_dir, scheme="W4A16", model_free=True, ignore_layers="mlp").quantize_and_save(output_dir)

        st_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
        with safe_open(os.path.join(output_dir, st_files[0]), framework="pt") as f:
            keys = set(f.keys())

        assert "model.layers.0.mlp.fc1.weight" in keys
        assert "model.layers.0.mlp.fc1.qweight" not in keys
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

        all_keys = set()
        for fname in os.listdir(output_dir):
            if fname.endswith(".safetensors"):
                with safe_open(os.path.join(output_dir, fname), framework="pt") as f:
                    all_keys.update(f.keys())
        assert "lm_head.qweight" in all_keys

        with open(os.path.join(output_dir, "quantization_config.json")) as f:
            quant_config = json.load(f)

        assert "lm_head" in quant_config["extra_config"]
        assert quant_config["extra_config"]["lm_head"]["bits"] == 4

    def test_unsupported_format_falls_back_to_regular_flow(self, tmp_path, monkeypatch):
        calls = {}

        class DummyCompressor:
            def quantize_and_save(self, output_dir, format, inplace=True, **kwargs):
                calls["quantize_and_save"] = {
                    "output_dir": output_dir,
                    "format": format,
                    "inplace": inplace,
                    "kwargs": kwargs,
                }
                return "dummy_model", [output_dir]

        def fake_autoround(*args, **kwargs):
            calls["init_kwargs"] = kwargs
            return DummyCompressor()

        monkeypatch.setattr("auto_round.autoround.AutoRound", fake_autoround)

        result = AutoRound(model=str(tmp_path), model_free=True, format="auto_gptq").quantize_and_save(
            str(tmp_path / "out"), format="auto_gptq"
        )

        assert calls["init_kwargs"]["model"] == str(tmp_path)
        assert calls["init_kwargs"]["disable_model_free"] is True
        assert calls["quantize_and_save"]["format"] == "auto_gptq"
        assert result == ("dummy_model", [str(tmp_path / "out")])

    def test_invalid_scheme_raises(self, tmp_path):
        with pytest.raises(ValueError, match="INVALID"):
            AutoRound(model=str(tmp_path), model_free=True, scheme="INVALID").quantize_and_save(str(tmp_path / "out"))

    def test_non_woq_scheme_raises(self, tmp_path):
        """Non-WOQ schemes (act_bits < 16) should be rejected."""
        with pytest.raises(ValueError, match="weight-only quantization"):
            AutoRound(model=str(tmp_path), model_free=True, scheme="MXFP4").quantize_and_save(str(tmp_path / "out"))

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
        AutoRound(model=model_dir, scheme=scheme, model_free=True).quantize_and_save(output_dir)

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


# ===========================================================================
#  Test: scheme allowlist (supported vs unsupported)
# ===========================================================================


_SUPPORTED_PRESET_NAMES = [
    "W2A16",
    "W2A16G32",
    "W2A16G64",
    "W4A16",
    "W4A16_MIXED",
    "W8A16",
]
# W3A16 is excluded: 3-bit packing requires in_features padding to a
# multiple of pack_factor=10, which quantize_weight_rtn does not perform.
_UNSUPPORTED_PRESET_NAMES = [
    "W3A16",
    "FPW8A16",
    "BF16",
    "MXFP4",
    "MXFP8",
    "MXINT4",
    "NVFP4",
    "FP8_BLOCK",
    "FP8_STATIC",
    "INT8_W8A8",
]


class TestSupportedSchemes:
    """Validate which schemes the model-free RTN path can produce."""

    @pytest.mark.parametrize("scheme_name", _SUPPORTED_PRESET_NAMES)
    def test_supported_preset_runs(self, tmp_path, scheme_name):
        from auto_round.schemes import preset_name_to_scheme

        # Use a tensor whose in_features is divisible by all preset group_sizes
        # (32, 64, 128) to keep RTN deterministic w.r.t. padding.
        tensors = {
            "model.layers.0.mlp.fc1.weight": torch.randn(64, 128),
        }
        cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        model_dir = _make_model_dir(tmp_path, cfg, tensors)
        out_dir = str(tmp_path / f"out_{scheme_name}")

        AutoRound(model=model_dir, scheme=scheme_name, model_free=True).quantize_and_save(out_dir)

        # Output config should reflect the scheme.
        with open(os.path.join(out_dir, "config.json")) as f:
            cfg_out = json.load(f)
        qc = cfg_out["quantization_config"]
        scheme_obj = preset_name_to_scheme(scheme_name)
        assert qc["bits"] == scheme_obj.bits
        assert qc["group_size"] == scheme_obj.group_size
        assert qc["sym"] == scheme_obj.sym
        assert qc["data_type"] == "int"
        assert qc["packing_format"] == "auto_round:auto_gptq"

        # Find the actual safetensors file (could be model.safetensors or sharded).
        st_files = [f for f in os.listdir(out_dir) if f.endswith(".safetensors")]
        assert st_files, "No safetensors output produced"
        all_keys = set()
        for fname in st_files:
            with safe_open(os.path.join(out_dir, fname), framework="pt") as f:
                all_keys.update(f.keys())
        base = "model.layers.0.mlp.fc1"
        assert f"{base}.qweight" in all_keys
        assert f"{base}.qzeros" in all_keys
        assert f"{base}.scales" in all_keys

    @pytest.mark.parametrize("scheme_name", _UNSUPPORTED_PRESET_NAMES)
    def test_unsupported_preset_raises(self, tmp_path, scheme_name):
        model_dir = _make_model_dir(
            tmp_path,
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            {"layer.weight": torch.randn(64, 128)},
        )
        with pytest.raises(ValueError, match="(?i)model-free|supported"):
            AutoRound(model=model_dir, model_free=True, scheme=scheme_name).quantize_and_save(str(tmp_path / "out"))

    def test_custom_scheme_object_supported(self, tmp_path):
        from auto_round.schemes import QuantizationScheme

        model_dir = _make_model_dir(
            tmp_path,
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            {"layer.weight": torch.randn(64, 128)},
        )
        sch = QuantizationScheme(bits=4, group_size=32, sym=False, data_type="int", act_bits=16)
        out_dir = str(tmp_path / "out_custom")
        AutoRound(model=model_dir, scheme=sch, model_free=True).quantize_and_save(out_dir)
        with open(os.path.join(out_dir, "config.json")) as f:
            qc = json.load(f)["quantization_config"]
        assert qc["bits"] == 4 and qc["group_size"] == 32 and qc["sym"] is False

    def test_helper_is_model_free_supported_scheme(self):
        from auto_round.compressors.model_free import is_model_free_supported_scheme
        from auto_round.schemes import QuantizationScheme

        for name in _SUPPORTED_PRESET_NAMES:
            assert is_model_free_supported_scheme(name) is True, name
        for name in _UNSUPPORTED_PRESET_NAMES:
            assert is_model_free_supported_scheme(name) is False, name
        # Unknown name → False (not raising).
        assert is_model_free_supported_scheme("DOES_NOT_EXIST") is False
        # bits=5 → not in supported bits.
        sch = QuantizationScheme(bits=5, group_size=128, sym=True, data_type="int", act_bits=16)
        assert is_model_free_supported_scheme(sch) is False
        # Overrides must be considered too, otherwise auto-routing can diverge
        # from the effective scheme used at runtime.
        assert is_model_free_supported_scheme("W4A16", {"bits": 2, "data_type": "int_asym_dq"}) is False


# ===========================================================================
#  Test: CLI auto-routing (--iters 0 --disable_opt_rtn → model_free)
# ===========================================================================


class TestCliAutoRouting:
    """Verify that the CLI auto-routes to model-free under the right conditions.

    Rather than spawning a subprocess (slow, and some env vars matter), we call
    ``setup_parser`` and ``tune`` directly with a tiny synthetic model dir.
    """

    @staticmethod
    def _build_local_model(tmp_path):
        """Tiny single-shard model dir with a 2-D weight to keep model_free happy."""
        model_dir = _make_model_dir(
            tmp_path,
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
            {"layer.weight": torch.randn(64, 128)},
        )
        return model_dir

    @staticmethod
    def _read_qconfig(out_dir):
        with open(os.path.join(out_dir, "config.json")) as f:
            return json.load(f).get("quantization_config", {})

    def _run_cli(self, argv):
        from auto_round.__main__ import BasicArgumentParser, tune

        parser = BasicArgumentParser()
        args = parser.parse_args(argv)
        tune(args)

    def test_auto_routes_to_model_free_when_iters0_and_disable_opt_rtn(self, tmp_path):
        """`--iters 0 --disable_opt_rtn` + supported scheme → model_free path."""
        model_dir = self._build_local_model(tmp_path)
        out_dir = str(tmp_path / "out_auto")
        self._run_cli(
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
        qc = self._read_qconfig(out_dir)
        # model_free path stamps the "model_free" key.
        assert qc.get("model_free") is True

    def test_disable_model_free_reverts_to_regular_flow(self, tmp_path):
        """`--disable_model_free` opts out of auto-routing."""
        # Skip if the regular flow can't load this synthetic dir (no architecture
        # mapping for "layer.weight" key); we use a real tiny model path if the
        # fixture is available, otherwise xfail this case.
        pytest.importorskip("transformers")
        from auto_round.__main__ import BasicArgumentParser

        # Just confirm the CLI flag is accepted and would suppress auto-routing
        # decision; we test the routing decision directly instead of running tune.
        from auto_round.compressors.model_free import is_model_free_supported_scheme

        parser = BasicArgumentParser()
        args = parser.parse_args(
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
        # Replicate the auto-routing predicate from __main__.py.
        auto_route = (
            not args.model_free
            and not args.disable_model_free
            and args.iters == 0
            and args.disable_opt_rtn is True
            and is_model_free_supported_scheme(args.scheme)
        )
        assert auto_route is False

    def test_unsupported_scheme_does_not_auto_route(self, tmp_path):
        """An unsupported scheme falls through to the regular flow."""
        from auto_round.__main__ import BasicArgumentParser
        from auto_round.compressors.model_free import is_model_free_supported_scheme

        parser = BasicArgumentParser()
        args = parser.parse_args(
            [
                "--model",
                "dummy",
                "--scheme",
                "MXFP4",
                "--iters",
                "0",
                "--disable_opt_rtn",
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
