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

import json
import os
import tempfile

import pytest
import torch

from auto_round.utils.missing_tensors import (
    _get_woq_config_from_dir,
    copy_missing_tensors_from_source,
    quantize_weight_rtn,
    split_fused_expert_tensors,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _save_safetensors(tensors: dict, path: str) -> None:
    from safetensors.torch import save_file

    save_file(tensors, path)


def _load_safetensors(path: str) -> dict:
    from safetensors import safe_open

    result = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)
    return result


def _write_config(target_dir: str, qcfg: dict | None = None) -> None:
    config = {}
    if qcfg is not None:
        config["quantization_config"] = qcfg
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f)


def _make_auto_round_config(bits=4, group_size=128, sym=True, block_name_to_quantize=None) -> dict:
    qcfg: dict = {
        "quantization_config": {
            "quant_method": "auto-round",
            "packing_format": "auto_round:auto_gptq",
            "bits": bits,
            "group_size": group_size,
            "sym": sym,
        }
    }
    if block_name_to_quantize is not None:
        qcfg["quantization_config"]["block_name_to_quantize"] = block_name_to_quantize
    return qcfg


# ===========================================================================
#  split_fused_expert_tensors
# ===========================================================================


class TestSplitFusedExpertTensors:

    def test_2d_and_non_expert_pass_through(self):
        """2-D tensors and 3-D non-expert tensors are returned unchanged."""
        tensors = {
            "mlp.fc1.weight": torch.randn(64, 128),
            "mlp.fc1.bias": torch.randn(64),
            "model.layers.0.something.weight": torch.randn(4, 8, 16),
        }
        result = split_fused_expert_tensors(tensors)
        assert set(result.keys()) == set(tensors.keys())
        for k in tensors:
            assert torch.equal(result[k], tensors[k])

    def test_empty_dict_returns_empty(self):
        assert split_fused_expert_tensors({}) == {}

    def test_gate_up_proj_split(self):
        """experts.gate_up_proj [N,2I,H] → N*(gate_proj + up_proj)."""
        N, I, H = 4, 64, 32
        fused = torch.randn(N, 2 * I, H)
        result = split_fused_expert_tensors({"model.layers.0.mlp.experts.gate_up_proj": fused})

        assert "model.layers.0.mlp.experts.gate_up_proj" not in result
        assert len(result) == N * 2
        for i in range(N):
            gk = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            uk = f"model.layers.0.mlp.experts.{i}.up_proj.weight"
            assert result[gk].shape == (I, H)
            assert result[uk].shape == (I, H)
            assert torch.equal(result[gk], fused[i, :I, :])
            assert torch.equal(result[uk], fused[i, I:, :])

    def test_gate_up_proj_with_weight_suffix(self):
        N, I, H = 2, 16, 8
        fused = torch.randn(N, 2 * I, H)
        result = split_fused_expert_tensors({"model.experts.gate_up_proj.weight": fused})
        assert "model.experts.gate_up_proj.weight" not in result
        for i in range(N):
            assert f"model.experts.{i}.gate_proj.weight" in result
            assert f"model.experts.{i}.up_proj.weight" in result

    def test_stacked_down_proj(self):
        """experts.down_proj [N,O,I] → N * down_proj.weight."""
        N, O, I = 3, 32, 16
        stacked = torch.randn(N, O, I)
        result = split_fused_expert_tensors({"model.layers.0.mlp.experts.down_proj": stacked})
        assert len(result) == N
        for i in range(N):
            key = f"model.layers.0.mlp.experts.{i}.down_proj.weight"
            assert result[key].shape == (O, I)
            assert torch.equal(result[key], stacked[i])

    def test_mixed_fused_and_normal(self):
        N, I, H = 2, 32, 16
        tensors = {
            "mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
            "mlp.experts.down_proj": torch.randn(N, H, I),
            "attn.q_proj.weight": torch.randn(H, H),
        }
        result = split_fused_expert_tensors(tensors)
        assert "mlp.experts.gate_up_proj" not in result
        assert "mlp.experts.down_proj" not in result
        assert "attn.q_proj.weight" in result
        for i in range(N):
            assert f"mlp.experts.{i}.gate_proj.weight" in result
            assert f"mlp.experts.{i}.down_proj.weight" in result

    def test_output_tensors_are_contiguous(self):
        N, I, H = 2, 32, 16
        result = split_fused_expert_tensors(
            {
                "mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
                "mlp.experts.down_proj": torch.randn(N, H, I),
            }
        )
        for k, v in result.items():
            assert v.is_contiguous(), f"{k} is not contiguous"


# ===========================================================================
#  quantize_weight_rtn
# ===========================================================================


class TestQuantizeWeightRtn:

    def test_output_shapes_4bit_sym(self):
        out_f, in_f, bits, gs = 32, 64, 4, 32
        qw, qz, sc = quantize_weight_rtn(torch.randn(out_f, in_f), bits=bits, group_size=gs, sym=True)
        pf = 32 // bits
        ng = in_f // gs
        assert qw.shape == (in_f // pf, out_f)
        assert qz.shape == (ng, out_f // pf)
        assert sc.shape == (ng, out_f)
        assert qw.dtype == torch.int32
        assert sc.dtype == torch.float16

    def test_output_shapes_4bit_asym(self):
        out_f, in_f, bits, gs = 32, 64, 4, 32
        qw, qz, sc = quantize_weight_rtn(torch.randn(out_f, in_f), bits=bits, group_size=gs, sym=False)
        assert qw.shape == (in_f // (32 // bits), out_f)

    def test_output_shapes_8bit(self):
        out_f, in_f, bits, gs = 16, 128, 8, 128
        qw, qz, sc = quantize_weight_rtn(torch.randn(out_f, in_f), bits=bits, group_size=gs, sym=True)
        pf = 32 // bits
        assert qw.shape == (in_f // pf, out_f)

    def test_outputs_on_cpu(self):
        qw, qz, sc = quantize_weight_rtn(torch.randn(16, 64), bits=4, group_size=32)
        assert qw.device.type == "cpu"
        assert qz.device.type == "cpu"
        assert sc.device.type == "cpu"

    def test_raises_on_1d(self):
        with pytest.raises(AssertionError):
            quantize_weight_rtn(torch.randn(64), bits=4, group_size=32)

    def test_in_features_padding(self):
        qw, _, _ = quantize_weight_rtn(torch.randn(16, 70), bits=4, group_size=32)
        assert qw.shape == (96 // (32 // 4), 16)  # padded to 96


# ===========================================================================
#  _get_woq_config_from_dir
# ===========================================================================


class TestGetWoqConfigFromDir:

    def test_returns_none_when_no_config(self, tmp_path):
        assert _get_woq_config_from_dir(str(tmp_path)) is None

    def test_returns_none_for_wrong_quant_method(self, tmp_path):
        _write_config(
            str(tmp_path),
            {
                "quant_method": "gptq",
                "packing_format": "auto_round:auto_gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
        )
        assert _get_woq_config_from_dir(str(tmp_path)) is None

    def test_returns_none_for_wrong_packing_format(self, tmp_path):
        _write_config(
            str(tmp_path),
            {"quant_method": "auto-round", "packing_format": "other", "bits": 4, "group_size": 128, "sym": True},
        )
        assert _get_woq_config_from_dir(str(tmp_path)) is None

    def test_returns_none_for_missing_bits_or_high_bits(self, tmp_path):
        _write_config(
            str(tmp_path),
            {"quant_method": "auto-round", "packing_format": "auto_round:auto_gptq", "group_size": 128, "sym": True},
        )
        assert _get_woq_config_from_dir(str(tmp_path)) is None

    def test_valid_config(self, tmp_path):
        _write_config(
            str(tmp_path),
            {
                "quant_method": "auto-round",
                "packing_format": "auto_round:auto_gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
        )
        cfg = _get_woq_config_from_dir(str(tmp_path))
        assert cfg is not None
        assert cfg["bits"] == 4
        assert "block_name_to_quantize" in cfg
        assert "extra_config" in cfg


# ===========================================================================
#  copy_missing_tensors_from_source
# ===========================================================================


class TestCopyMissingTensorsFromSource:

    # ----- Detection -----

    def test_plain_copy_of_missing_tensor(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        mtp_norm = torch.randn(64)
        _save_safetensors({"mtp.0.norm.weight": mtp_norm}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        result = _load_safetensors(os.path.join(tgt, "model_extra_tensors.safetensors"))
        assert "mtp.0.norm.weight" in result
        torch.testing.assert_close(result["mtp.0.norm.weight"], mtp_norm)

    def test_known_parent_layer_not_copied(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors(
            {"model.layers.0.self_attn.q_proj.weight": torch.randn(32, 64)}, os.path.join(src, "model.safetensors")
        )
        _save_safetensors(
            {"model.layers.0.self_attn.q_proj.qweight": torch.randint(0, 2**31, (8, 32), dtype=torch.int32)},
            os.path.join(tgt, "model.safetensors"),
        )
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        assert not os.path.exists(os.path.join(tgt, "model_extra_tensors.safetensors"))

    def test_known_block_prefix_not_copied(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors(
            {"model.layers.0.mlp.extra_gate.weight": torch.randn(32, 64)}, os.path.join(src, "model.safetensors")
        )
        _save_safetensors(
            {"model.layers.0.mlp.gate_proj.qweight": torch.randint(0, 2**31, (8, 32), dtype=torch.int32)},
            os.path.join(tgt, "model.safetensors"),
        )
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        assert not os.path.exists(os.path.join(tgt, "model_extra_tensors.safetensors"))

    def test_lm_head_never_copied(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"lm_head.weight": torch.randn(32, 64)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(32, 64)}, os.path.join(tgt, "model.safetensors"))
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        assert not os.path.exists(os.path.join(tgt, "model_extra_tensors.safetensors"))

    def test_already_present_not_duplicated(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        t = torch.randn(64)
        _save_safetensors({"mtp.0.norm.weight": t}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"mtp.0.norm.weight": t}, os.path.join(tgt, "model.safetensors"))
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        assert not os.path.exists(os.path.join(tgt, "model_extra_tensors.safetensors"))

    def test_detects_multiple_missing_tensors_from_different_blocks(self, tmp_path):
        """All source tensors from blocks absent in the saved output are copied."""
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors(
            {
                "custom_block.0.norm.weight": torch.randn(64),
                "mtp.0.norm.weight": torch.randn(64),
                "model.embed_tokens.weight": torch.randn(8, 64),
            },
            os.path.join(src, "model.safetensors"),
        )
        _save_safetensors(
            {"model.embed_tokens.weight": torch.randn(8, 64)},
            os.path.join(tgt, "model.safetensors"),
        )
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        extra_shard = os.path.join(tgt, "model_extra_tensors.safetensors")
        assert os.path.exists(extra_shard)
        result = _load_safetensors(extra_shard)
        assert "custom_block.0.norm.weight" in result
        assert "mtp.0.norm.weight" in result
        assert "model.embed_tokens.weight" not in result

    # ----- Fused expert splitting -----

    def test_fused_experts_split_and_copied(self, tmp_path):
        N, I, H = 2, 32, 16
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors(
            {
                "model.mtp.0.mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
                "model.mtp.0.mlp.experts.down_proj": torch.randn(N, H, I),
            },
            os.path.join(src, "model.safetensors"),
        )
        _save_safetensors({"model.embed_tokens.weight": torch.randn(10, H)}, os.path.join(tgt, "model.safetensors"))
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        result = _load_safetensors(os.path.join(tgt, "model_extra_tensors.safetensors"))
        assert "model.mtp.0.mlp.experts.gate_up_proj" not in result
        for i in range(N):
            assert f"model.mtp.0.mlp.experts.{i}.gate_proj.weight" in result
            assert f"model.mtp.0.mlp.experts.{i}.up_proj.weight" in result
            assert f"model.mtp.0.mlp.experts.{i}.down_proj.weight" in result

    def test_fused_experts_woq_quantized(self, tmp_path):
        N, I, H = 2, 32, 16
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors(
            {
                "model.mtp.0.mlp.experts.gate_up_proj": torch.randn(N, 2 * I, H),
                "model.mtp.0.mlp.experts.down_proj": torch.randn(N, H, I),
            },
            os.path.join(src, "model.safetensors"),
        )
        _save_safetensors(
            {"model.layers.0.mlp.fc1.qweight": torch.zeros(H // 8, H, dtype=torch.int32)},
            os.path.join(tgt, "model.safetensors"),
        )
        _write_config(
            tgt,
            {
                "quant_method": "auto-round",
                "packing_format": "auto_round:auto_gptq",
                "bits": 4,
                "group_size": H,
                "sym": True,
            },
        )
        copy_missing_tensors_from_source(src, tgt)
        result = _load_safetensors(os.path.join(tgt, "model_extra_tensors.safetensors"))
        for i in range(N):
            assert f"model.mtp.0.mlp.experts.{i}.gate_proj.qweight" in result
            assert f"model.mtp.0.mlp.experts.{i}.down_proj.qweight" in result
        assert "model.mtp.0.mlp.experts.0.gate_proj.weight" not in result

    # ----- FP8 dequantization -----

    def test_fp8_dequantized_to_bf16(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors(
            {
                "mtp.0.ffn.weight": torch.randn(64, 128).to(torch.float8_e4m3fn),
                "mtp.0.ffn.weight_scale_inv": torch.tensor(0.5),
            },
            os.path.join(src, "model.safetensors"),
        )
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        result = _load_safetensors(os.path.join(tgt, "model_extra_tensors.safetensors"))
        assert "mtp.0.ffn.weight" in result
        assert "mtp.0.ffn.weight_scale_inv" not in result
        assert result["mtp.0.ffn.weight"].dtype == torch.bfloat16

    # ----- WOQ quantization -----

    def test_woq_quantizes_missing_weights(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.ffn.weight": torch.randn(32, 128)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        with open(os.path.join(tgt, "config.json"), "w") as f:
            json.dump(_make_auto_round_config(), f)
        copy_missing_tensors_from_source(src, tgt)
        result = _load_safetensors(os.path.join(tgt, "model_extra_tensors.safetensors"))
        assert "mtp.0.ffn.weight" not in result
        assert "mtp.0.ffn.qweight" in result

    def test_block_name_to_ignore_not_quantized(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.mlp.gate.weight": torch.randn(32, 128)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        with open(os.path.join(tgt, "config.json"), "w") as f:
            json.dump(_make_auto_round_config(), f)
        copy_missing_tensors_from_source(src, tgt)
        result = _load_safetensors(os.path.join(tgt, "model_extra_tensors.safetensors"))
        assert "mtp.0.mlp.gate.weight" in result
        assert "mtp.0.mlp.gate.qweight" not in result

    # ----- Config / index file updates -----

    def test_config_updated_after_woq(self, tmp_path):
        """block_name_to_quantize is extended with the new mtp block when one already exists."""
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.ffn.weight": torch.randn(32, 128)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        with open(os.path.join(tgt, "config.json"), "w") as f:
            json.dump(_make_auto_round_config(block_name_to_quantize=["model.layers"]), f)
        copy_missing_tensors_from_source(src, tgt)
        with open(os.path.join(tgt, "config.json")) as f:
            cfg = json.load(f)
        block_names = cfg.get("quantization_config", {}).get("block_name_to_quantize", [])
        assert any("mtp" in b for b in block_names)

    def test_config_updated_without_block_name_to_quantize_after_woq(self, tmp_path):
        """When no block_name_to_quantize is set initially, it stays empty after WOQ."""
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.ffn.weight": torch.randn(32, 128)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        with open(os.path.join(tgt, "config.json"), "w") as f:
            json.dump(_make_auto_round_config(), f)
        copy_missing_tensors_from_source(src, tgt)
        with open(os.path.join(tgt, "config.json")) as f:
            cfg = json.load(f)
        block_names = cfg.get("quantization_config", {}).get("block_name_to_quantize", [])
        assert block_names == [], f"Expected no block_name_to_quantize, got: {block_names}"

    def test_config_updated_with_block_name_to_quantize_after_woq(self, tmp_path):
        """When block_name_to_quantize is set, the new mtp block is appended."""
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.ffn.weight": torch.randn(32, 128)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        with open(os.path.join(tgt, "config.json"), "w") as f:
            json.dump(_make_auto_round_config(block_name_to_quantize=["model.layers"]), f)
        copy_missing_tensors_from_source(src, tgt)
        with open(os.path.join(tgt, "config.json")) as f:
            cfg = json.load(f)
        block_names = cfg.get("quantization_config", {}).get("block_name_to_quantize", [])
        assert any("mtp" in b for b in block_names), f"Expected an 'mtp' block, got: {block_names}"

    def test_config_updated_with_extra_config_for_unquantized_2d_weight(self, tmp_path):
        """When a 2-D weight is skipped by BLOCK_NAME_TO_IGNORE, extra_config is updated
        to mark it as full-precision so the inference stack knows not to dequantize it."""
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.mlp.gate.weight": torch.randn(32, 128)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        with open(os.path.join(tgt, "config.json"), "w") as f:
            json.dump(_make_auto_round_config(), f)
        copy_missing_tensors_from_source(src, tgt)
        with open(os.path.join(tgt, "config.json")) as f:
            cfg = json.load(f)
        extra_config = cfg.get("quantization_config", {}).get("extra_config", {})
        assert "mtp.0.mlp.gate" in extra_config
        layer_cfg = extra_config["mtp.0.mlp.gate"]
        assert layer_cfg.get("bits") == 16
        assert layer_cfg.get("data_type") == "fp"

    def test_index_json_updated_for_sharded_target(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.norm.weight": torch.randn(64)}, os.path.join(src, "model.safetensors"))
        saved_shard = "model-00001-of-00001.safetensors"
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, saved_shard))
        with open(os.path.join(tgt, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": 0}, "weight_map": {"model.embed_tokens.weight": saved_shard}}, f)
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        with open(os.path.join(tgt, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        assert idx["weight_map"]["mtp.0.norm.weight"] == "model_extra_tensors.safetensors"

    def test_index_json_is_created_for_single_file_target(self, tmp_path):
        """When the target uses a single safetensors file and tensors are copied, a new
        model.safetensors.index.json is created mapping all tensors to their shards."""
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors({"mtp.0.norm.weight": torch.randn(64)}, os.path.join(src, "model.safetensors"))
        _save_safetensors({"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(tgt, "model.safetensors"))
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        index_path = os.path.join(tgt, "model.safetensors.index.json")
        assert os.path.exists(index_path), "An index.json should be created for single-file targets"
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        assert "mtp.0.norm.weight" in weight_map
        assert "model.embed_tokens.weight" in weight_map
        assert weight_map["mtp.0.norm.weight"] == "model_extra_tensors.safetensors"
        assert weight_map["model.embed_tokens.weight"] == "model.safetensors"

    def test_known_block_prefix_not_copied_gemma(self, tmp_path):
        src, tgt = str(tmp_path / "src"), str(tmp_path / "tgt")
        os.makedirs(src)
        os.makedirs(tgt)
        _save_safetensors(
            {
                "language_model.model.layers.0.mlp.gate_proj.weight": torch.randn(32, 64),
                "language_model.model.norm.weight": torch.randn(64),
                "language_model.layers.0.mlp.gate_proj.weight": torch.randn(32, 64),
                "language_model.norm.weight": torch.randn(64),
            },
            os.path.join(src, "model.safetensors"),
        )
        _save_safetensors(
            {
                "language_model.model.layers.0.mlp.gate_proj.weight": torch.randn(32, 64),
                "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(32, 64),
                "model.language_model.norm.weight": torch.randn(64),
            },
            os.path.join(tgt, "model.safetensors"),
        )
        _write_config(tgt)
        copy_missing_tensors_from_source(src, tgt)
        assert not os.path.exists(os.path.join(tgt, "model_extra_tensors.safetensors"))
