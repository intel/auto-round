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
from unittest.mock import Mock

import pytest


def test_quantize_and_save_restores_temporary_state_on_failure(monkeypatch):
    from auto_round.compressors.model_free import ModelFreeCompressor

    compressor = ModelFreeCompressor.__new__(ModelFreeCompressor)
    compressor.scheme_input = "W4A16"
    compressor.layer_config_input = None
    compressor.user_scheme_overrides = None
    compressor._auto_scheme_family = None
    compressor.output_dir = "original-output"
    compressor.format = "auto_round"
    compressor.quantized = False

    def fail_run():
        raise RuntimeError("save failed")

    monkeypatch.setattr(compressor, "run", fail_run)

    with pytest.raises(RuntimeError, match="save failed"):
        compressor.quantize_and_save("temporary-output", format="auto_round:auto_gptq")

    assert compressor.output_dir == "original-output"
    assert compressor.format == "auto_round"
    assert compressor.quantized is False


import torch
from safetensors import safe_open
from safetensors.torch import save_file

from auto_round import AutoRound
from auto_round.compressors.model_free import (
    _ModelFreeCompressorCore,
    _process_single_shard_task,
)
from auto_round.schemes import QuantizationScheme
from auto_round.utils.model_free_utils import (
    _build_mxfp_autoround_quantization_config,
    _build_mxfp_quantization_config,
    _build_quantization_config,
    _convert_auto_scheme_layer_config,
    _dequant_mxfp_tensors,
    _expand_e8m0_block_scale,
    _handle_mxfp_source_tensors,
    _looks_like_auto_scheme,
    _PatternMatcher,
    _process_shard,
    _quantize_weight_mxfp,
    _quantize_weight_nvfp4_e5m3,
    _validate_auto_scheme_options,
    is_model_free_supported_scheme,
)
from auto_round.utils.model_free_utils import (
    preprocess_model_type_source_tensors as _preprocess_model_type_source_tensors,
)


def test_model_free_preserves_explicit_scheme_overrides():
    from auto_round.compressors.model_free import ModelFreeCompressor

    compressor = ModelFreeCompressor("unused-model-path", scheme="W8A16", sym=False)
    compressor._parse_scheme()

    assert compressor.default_scheme["sym"] is False


def test_model_free_entry_passes_resolved_scheme_overrides(tiny_opt_model_path):
    from auto_round import AutoRound

    compressor = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        sym=False,
        iters=0,
        disable_opt_rtn=True,
        device_map="cpu",
        enable_torch_compile=False,
    )

    assert type(compressor).__name__ == "ModelFreeCompressor"
    assert compressor.scheme_input.sym is False
    assert compressor.disable_opt_rtn is True


def test_model_free_entry_preserves_enabled_opt_rtn(tiny_opt_model_path):
    compressor = AutoRound(
        tiny_opt_model_path,
        scheme="MXFP4",
        iters=0,
        disable_opt_rtn=False,
        model_free=True,
        device_map="cpu",
        enable_torch_compile=False,
    )

    assert type(compressor).__name__ == "ModelFreeCompressor"
    assert compressor.disable_opt_rtn is False


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
#  _build_ignore_patterns
# ===========================================================================


class TestBuildIgnorePatterns:
    @staticmethod
    def _make_core(layer_config=None, quant_lm_head=False, scheme="W4A16"):
        core = _ModelFreeCompressorCore(
            model_name_or_path="dummy",
            output_dir="dummy_out",
            scheme=scheme,
            quant_lm_head=quant_lm_head,
        )
        core._parse_scheme()
        core._parse_layer_config()
        if layer_config:
            # Merge any explicit layer_config entries on top of the parsed defaults.
            for k, v in layer_config.items():
                core.layer_config[k] = v
        return core

    def test_lm_head_ignored_by_default(self):
        """Without quant_lm_head or an explicit layer_config entry, lm_head is skipped."""
        core = self._make_core()
        core._build_ignore_patterns()
        assert "lm_head" in core.ignore_patterns

    def test_lm_head_not_ignored_when_quant_lm_head_true(self):
        """quant_lm_head=True removes lm_head from ignore list."""
        core = self._make_core(quant_lm_head=True)
        core._build_ignore_patterns()
        assert "lm_head" not in core.ignore_patterns

    def test_lm_head_not_ignored_when_in_layer_config(self):
        """Explicit lm_head entry in layer_config removes it from the ignore list."""
        core = self._make_core(layer_config={"lm_head": {"bits": 4}})
        core._build_ignore_patterns()
        assert "lm_head" not in core.ignore_patterns

    def test_head_not_ignored_when_in_layer_config(self):
        """DeepSeek v4 uses 'head' as the lm_head layer name; explicit entry in layer_config removes it from ignore."""
        core = self._make_core(layer_config={"head": {"bits": 4}})
        core._build_ignore_patterns()
        assert "head" not in core.ignore_patterns

    def test_head_still_ignored_when_lm_head_in_layer_config_but_not_head(self):
        """Specifying 'lm_head' in layer_config should not unblock the separate 'head' pattern."""
        core = self._make_core(layer_config={"lm_head": {"bits": 4}})
        core._build_ignore_patterns()
        # 'lm_head' itself is unblocked, but 'head' (deepseek v4) remains ignored
        assert "lm_head" not in core.ignore_patterns
        assert "head" in core.ignore_patterns

    def test_embed_out_not_ignored_when_in_layer_config(self):
        """Pythia/Dolly models use 'embed_out' as the lm_head layer name."""
        core = self._make_core(layer_config={"embed_out": {"bits": 4}})
        core._build_ignore_patterns()
        assert "embed_out" not in core.ignore_patterns

    def test_output_not_ignored_when_in_layer_config(self):
        """Some InternLM variants use 'output' as the lm_head layer name."""
        core = self._make_core(layer_config={"output": {"bits": 4}})
        core._build_ignore_patterns()
        assert "output" not in core.ignore_patterns


# ===========================================================================
#  _process_shard
# ===========================================================================


class TestProcessShard:
    @pytest.mark.parametrize("enabled, expected_calls", [(True, 1), (False, 0)])
    def test_torch_compile_setting(self, tmp_path, monkeypatch, enabled, expected_calls):
        shard_path = str(tmp_path / "shard.safetensors")
        save_file({"layer.fc1.weight": torch.randn(64, 128)}, shard_path)
        compile_mock = Mock(side_effect=lambda func, _device: func)
        monkeypatch.setattr("auto_round.utils.model_free_utils.compile_func", compile_mock)

        _process_shard(shard_path, _DEFAULT_SCHEME, {}, [], enable_torch_compile=enabled)

        assert compile_mock.call_count == expected_calls

    def test_quantizes_eligible_weights(self, tmp_path):
        shard_path = str(tmp_path / "shard.safetensors")
        save_file({"layer.fc1.weight": torch.randn(64, 128), "layer.fc1.bias": torch.randn(64)}, shard_path)
        output, quantized, _ = _process_shard(shard_path, _DEFAULT_SCHEME, {}, [])
        assert "layer.fc1" in quantized
        assert "layer.fc1.qweight" in output and "layer.fc1.bias" in output


def test_nvfp4_e5m3_model_free_fake_quantization():
    weight = torch.randn(8, 32)
    output = _quantize_weight_nvfp4_e5m3(weight, "layer.fc", group_size=16)

    assert set(output) == {"layer.fc.weight"}
    assert output["layer.fc.weight"].shape == weight.shape
    assert output["layer.fc.weight"].dtype == weight.dtype
    assert not torch.equal(output["layer.fc.weight"], weight)
    assert is_model_free_supported_scheme("NVFP4_E5M3")
    assert not is_model_free_supported_scheme("UNVFP4")
    assert not is_model_free_supported_scheme("NVFP4+")


def test_nvfp4_e5m3_model_free_end_to_end(tmp_path):
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "lm_head.weight": torch.randn(64, 32),
    }
    model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
        json.dump({"stale": True}, f)

    compressor = _ModelFreeCompressorCore(model_name_or_path=model_dir, output_dir=output_dir, scheme="NVFP4_E5M3")
    compressor.run()

    output_keys = _read_output_keys(output_dir)
    assert "model.layers.0.self_attn.q_proj.weight" not in output_keys
    assert "model.layers.0.self_attn.q_proj.weight_packed" in output_keys
    assert "model.layers.0.self_attn.q_proj.weight_scale" in output_keys
    assert "lm_head.weight" in output_keys
    assert compressor.format == "auto_round"
    quantization_config = _read_qconfig(output_dir)
    assert quantization_config["packing_format"] == "auto_round:llm_compressor_nvfp4_e5m3"
    assert quantization_config["data_type"] == "nvfp4_v2"
    assert quantization_config["act_bits"] == 4
    assert quantization_config["act_data_type"] == "nvfp4_v2"
    assert quantization_config["act_group_size"] == 16
    assert quantization_config["act_sym"] is True
    assert quantization_config["extra_config"]["lm_head"] == {
        "bits": 16,
        "data_type": "float",
        "act_bits": 16,
        "act_data_type": "float",
    }
    assert os.path.exists(os.path.join(output_dir, "quantization_config.json"))


def test_nvfp4_e5m3_model_free_llm_compressor(tmp_path):
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "lm_head.weight": torch.randn(64, 32),
    }
    model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
    output_dir = str(tmp_path / "output")

    compressor = _ModelFreeCompressorCore(
        model_name_or_path=model_dir,
        output_dir=output_dir,
        scheme="NVFP4_E5M3",
        format="llm_compressor",
    )
    compressor.run()

    output_keys = _read_output_keys(output_dir)
    prefix = "model.layers.0.self_attn.q_proj"
    assert f"{prefix}.weight_packed" in output_keys
    assert f"{prefix}.weight_scale" in output_keys
    assert f"{prefix}.weight" not in output_keys
    assert f"{prefix}.weight_global_scale" not in output_keys
    assert f"{prefix}.input_global_scale" not in output_keys
    quantization_config = _read_qconfig(output_dir)
    group = quantization_config["config_groups"]["group_0"]
    assert quantization_config["format"] == "nvfp4-e5m3-pack-quantized"
    assert quantization_config["quant_method"] == "compressed-tensors"
    assert quantization_config["provider"] == "auto-round"
    assert group["weights"]["group_size"] == 16
    assert group["input_activations"]["dynamic"] == "local"


def test_model_free_legacy_nvfp4_is_normalized_and_passthrough(tmp_path):
    prefix = "model.layers.0.mlp.down_proj"
    tensors = {
        f"{prefix}.weight": torch.randint(0, 256, (32, 64), dtype=torch.uint8),
        f"{prefix}.weight_scale": torch.randint(0, 256, (32, 4), dtype=torch.uint8),
        f"{prefix}.weight_scale_2": torch.tensor([2.0], dtype=torch.float32),
        f"{prefix}.input_scale": torch.tensor([4.0], dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
    }
    shard_path = str(tmp_path / "shard.safetensors")
    save_file(tensors, shard_path)

    layer_config = {
        prefix: {
            "bits": 4,
            "group_size": 16,
            "sym": True,
            "data_type": "nv_fp",
        }
    }
    output, quantized, ignored = _process_shard(shard_path, _DEFAULT_SCHEME, layer_config, [])

    # Legacy naming should be normalized to llm-compressor-style global-scale keys.
    assert f"{prefix}.weight_packed" in output
    assert f"{prefix}.weight_scale" in output
    assert f"{prefix}.weight_global_scale" in output
    assert f"{prefix}.input_global_scale" in output
    assert f"{prefix}.weight" not in output
    assert f"{prefix}.weight_scale_2" not in output
    assert f"{prefix}.input_scale" not in output
    assert torch.allclose(output[f"{prefix}.weight_global_scale"], torch.tensor([0.5], dtype=torch.float32))
    assert torch.allclose(output[f"{prefix}.input_global_scale"], torch.tensor([0.25], dtype=torch.float32))

    # The NVFP4 layer is treated as already quantized (passthrough) while
    # other Linear layers in the shard are still quantized by model-free RTN.
    assert prefix in quantized
    assert "model.layers.0.self_attn.q_proj" in quantized
    assert prefix not in ignored

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
#  FP8 source model — moved to test/unit/test_cpu/utils/test_model_free_utils.py
# ===========================================================================


# ===========================================================================
#  Quantization config builder
# ===========================================================================

# (TestBuildQuantizationConfig moved to test/unit/test_cpu/utils/test_model_free_utils.py)


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

    def test_layer_config_lm_head_bits_takes_effect(self, tmp_path):
        """layer_config for lm_head should quantize lm_head even without quant_lm_head=True."""
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")
        AutoRound(
            model=model_dir,
            scheme="W2A16G64",
            model_free=True,
            layer_config={"lm_head": {"bits": 4}},
        ).quantize_and_save(output_dir)
        assert "lm_head.qweight" in _read_output_keys(output_dir)

    def test_layer_config_lm_head_scheme_takes_effect(self, tmp_path):
        """layer_config with scheme override for lm_head should quantize lm_head even without quant_lm_head=True."""
        model_dir = _make_model_dir(tmp_path, _SIMPLE_CONFIG, _SIMPLE_TENSORS)
        output_dir = str(tmp_path / "output")
        AutoRound(
            model=model_dir,
            scheme="W2A16G64",
            model_free=True,
            layer_config={"lm_head": {"scheme": "W4A16"}},
        ).quantize_and_save(output_dir)
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
    """End-to-end tests for MXFP4/MXFP8 model-free quantization.

    Unit tests for _quantize_weight_mxfp and _build_mxfp_quantization_config
    have been moved to test/unit/test_cpu/utils/test_model_free_utils.py.
    """

    @require_compressed_tensors
    @pytest.mark.parametrize("scheme,fmt", [("MXFP4", "mxfp4-pack-quantized")])
    def test_e2e_mxfp(self, tmp_path, scheme, fmt):
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.fc1.weight": torch.randn(512, 128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        _ModelFreeCompressorCore(
            model_name_or_path=model_dir, output_dir=output_dir, scheme=scheme, format="llm_compressor"
        ).run()
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

    # Unit tests for _build_mxfp_quantization_config moved to
    # test/unit/test_cpu/utils/test_model_free_utils.py -> TestBuildMxfpConfig

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
            format="llm_compressor",
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


# TestExpandE8M0BlockScale moved to test/unit/test_cpu/utils/test_model_free_utils.py
# TestDeepseekV4MXFP8Source (resolve_model_type tests) moved to test/unit/test_cpu/utils/test_model_free_utils.py


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
    """llm-compressor MXFP8/MXFP4 source models (generic non-deepseek_v4 path).

    Unit tests for _handle_mxfp_source_tensors and _dequant_mxfp_tensors moved to
    test/unit/test_cpu/utils/test_model_free_utils.py -> TestHandleMXFPSourceTensors.
    resolve_model_type tests moved to test/unit/test_cpu/utils/test_model_free_utils.py.
    """

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
        _ModelFreeCompressorCore(
            model_name_or_path=model_dir, output_dir=output_dir, scheme="MXFP8", format="llm_compressor"
        ).run()

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

    # Unit tests test_convert_* and test_dequant_* moved to
    # test/unit/test_cpu/utils/test_model_free_utils.py -> TestHandleMXFPSourceTensors

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
        _ModelFreeCompressorCore(
            model_name_or_path=model_dir, output_dir=output_dir, scheme="MXFP8", format="llm_compressor"
        ).run()

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


# Keep representative presets per family to reduce redundant runtime:
# INT symmetric (2/4/8-bit), mixed override recipe, MXFP, and BF16 passthrough.
_SUPPORTED = ["W2A16G32", "W4A16", "W4A16_MIXED", "W8A16", "MXFP4", "BF16"]
_UNSUPPORTED = [
    "W3A16",
    "FPW8A16",  # unsupported FP family
    "MXINT4",
    "NVFP4",  # unsupported MX/NV family
    "FP8_BLOCK",  # unsupported FP8 route
    "INT8_W8A8",
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
        if name == "BF16":
            # BF16 default: no weights quantized; the layer is kept in full precision.
            assert len(quantized) == 0
            assert "model.layers.0.mlp.fc1.weight" in output
        else:
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
#  kimi_k25 INT4 packed source models — moved to test/unit/test_cpu/utils/test_model_free_utils.py
# ===========================================================================


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


# TestPredefinedIgnoreLayers moved to test/unit/test_cpu/utils/test_model_free_utils.py


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


class TestKimiK25Int4Source:
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
    """Model-free support for ``AutoScheme`` mixed-bit selection.

    Pure unit tests for helper functions (_looks_like_auto_scheme,
    _validate_auto_scheme_options, _convert_auto_scheme_layer_config) have been
    moved to test/unit/test_cpu/utils/test_model_free_utils.py -> TestAutoSchemeHelpers.
    """

    @pytest.mark.timeout(120)
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


# ===========================================================================
#  MXFP auto-round format (packing_format=auto_round:llm_compressor)
# ===========================================================================


class TestMXFPAutoRoundFormat:
    """Tests for MXFP4/MXFP8 model-free quantization with format='auto_round'.

    The weight tensors on disk are identical to the llm_compressor path;
    only the ``quantization_config`` metadata differs.

    Unit tests for _build_mxfp_autoround_quantization_config and
    _build_quantization_config routing have been moved to
    test/unit/test_cpu/utils/test_model_free_utils.py -> TestBuildMxfpAutoRoundConfig.
    """

    # ------------------------------------------------------------------
    # End-to-end tests via AutoRound.quantize_and_save
    # ------------------------------------------------------------------

    @require_compressed_tensors
    def test_e2e_mxfp4_config_fields(self, tmp_path):
        """MXFP4 + format='auto_round': quantization_config matches real AutoRound output."""
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.fc1.weight": torch.randn(512, 128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="MXFP4", model_free=True).quantize_and_save(output_dir, format="auto_round")
        qc = _read_qconfig(output_dir)
        assert qc["quant_method"] == "auto-round"
        assert qc["packing_format"] == "auto_round:llm_compressor"
        assert qc["bits"] == 4
        assert qc["data_type"] == "mx_fp"
        assert qc["act_bits"] == 4
        assert qc["act_data_type"] == "mx_fp"
        assert qc["enable_quanted_input"] is False
        assert qc["model_free"] is True
        # lm_head kept full-precision → extra_config entry
        extra = qc.get("extra_config", {})
        assert extra.get("lm_head") == {
            "bits": 16,
            "data_type": "float",
            "act_bits": 16,
            "act_data_type": "float",
        }

    @require_compressed_tensors
    def test_e2e_mxfp8_autoround_format(self, tmp_path):
        """MXFP8 + format='auto_round': bits=8, weight layout uses float8 .weight."""
        tensors = {
            "model.layers.0.fc1.weight": torch.randn(128, 128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="MXFP8", model_free=True).quantize_and_save(output_dir, format="auto_round")
        qc = _read_qconfig(output_dir)
        assert qc["quant_method"] == "auto-round"
        assert qc["packing_format"] == "auto_round:llm_compressor"
        assert qc["bits"] == 8
        assert qc["act_bits"] == 8
        keys = _read_output_keys(output_dir)
        assert "model.layers.0.fc1.weight" in keys
        assert "model.layers.0.fc1.weight_scale" in keys

    @require_compressed_tensors
    def test_e2e_mxfp4_quant_lm_head_extra_config(self, tmp_path):
        """When lm_head is quantized, it appears in extra_config with its full scheme."""
        tensors = {
            "model.layers.0.fc1.weight": torch.randn(128, 128),
            "lm_head.weight": torch.randn(1000, 128),
        }
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)
        output_dir = str(tmp_path / "output")
        AutoRound(model=model_dir, scheme="MXFP4", model_free=True, quant_lm_head=True).quantize_and_save(
            output_dir, format="auto_round"
        )
        qc = _read_qconfig(output_dir)
        extra = qc.get("extra_config", {})
        assert "lm_head" in extra
        assert extra["lm_head"]["bits"] == 4
        assert extra["lm_head"]["data_type"] == "mx_fp"
        keys = _read_output_keys(output_dir)
        assert "lm_head.weight_packed" in keys

    @require_compressed_tensors
    def test_e2e_mxfp4_autoround_format_same_weights_as_llmcompressor(self, tmp_path):
        """auto_round and llm_compressor format produce identical weight bytes."""
        tensors = {"model.layers.0.fc1.weight": torch.randn(64, 128)}
        model_dir = _make_model_dir(tmp_path, _LLAMA_CFG, tensors)

        out_ar = str(tmp_path / "out_ar")
        out_llm = str(tmp_path / "out_llm")
        AutoRound(model=model_dir, scheme="MXFP4", model_free=True).quantize_and_save(out_ar, format="auto_round")
        AutoRound(model=model_dir, scheme="MXFP4", model_free=True).quantize_and_save(out_llm, format="llm_compressor")

        # Weight bytes must match; only quantization_config differs.
        def _load_tensor(directory, name):
            for f in os.listdir(directory):
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(directory, f), framework="pt") as sf:
                        if name in sf.keys():
                            return sf.get_tensor(name)
            return None

        wp_ar = _load_tensor(out_ar, "model.layers.0.fc1.weight_packed")
        ws_ar = _load_tensor(out_ar, "model.layers.0.fc1.weight_scale")
        wp_llm = _load_tensor(out_llm, "model.layers.0.fc1.weight_packed")
        ws_llm = _load_tensor(out_llm, "model.layers.0.fc1.weight_scale")
        assert wp_ar is not None and wp_llm is not None
        assert torch.equal(wp_ar, wp_llm), "weight_packed must be identical"
        assert torch.equal(ws_ar, ws_llm), "weight_scale must be identical"
