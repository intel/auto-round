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
import unittest

import torch

from auto_round.utils.missing_tensors import (
    _get_woq_config_from_dir,
    copy_missing_tensors_from_source,
    quantize_weight_rtn,
)


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


class TestQuantizeWeightRtn(unittest.TestCase):
    """Tests for the standalone quantize_weight_rtn helper."""

    def test_output_shapes_4bit_sym(self):
        """Check that output tensors have the expected shapes for 4-bit symmetric quantization."""
        out_features, in_features = 32, 64
        bits, group_size = 4, 32
        weight = torch.randn(out_features, in_features)

        pack_factor = 32 // bits  # 8
        num_groups = in_features // group_size  # 2

        qweight, qzeros, scales = quantize_weight_rtn(weight, bits=bits, group_size=group_size, sym=True)

        self.assertEqual(qweight.shape, (in_features // pack_factor, out_features))
        self.assertEqual(qzeros.shape, (num_groups, out_features // pack_factor))
        self.assertEqual(scales.shape, (num_groups, out_features))
        self.assertEqual(qweight.dtype, torch.int32)
        self.assertEqual(qzeros.dtype, torch.int32)
        self.assertEqual(scales.dtype, torch.float16)

    def test_output_shapes_4bit_asym(self):
        """Check output shapes for 4-bit asymmetric quantization."""
        out_features, in_features = 32, 64
        bits, group_size = 4, 32
        weight = torch.randn(out_features, in_features)

        pack_factor = 32 // bits
        num_groups = in_features // group_size

        qweight, qzeros, scales = quantize_weight_rtn(weight, bits=bits, group_size=group_size, sym=False)

        self.assertEqual(qweight.shape, (in_features // pack_factor, out_features))
        self.assertEqual(qzeros.shape, (num_groups, out_features // pack_factor))
        self.assertEqual(scales.shape, (num_groups, out_features))

    def test_output_shapes_8bit(self):
        """Check output shapes for 8-bit symmetric quantization."""
        out_features, in_features = 16, 128
        bits, group_size = 8, 128
        weight = torch.randn(out_features, in_features)

        pack_factor = 32 // bits  # 4
        num_groups = in_features // group_size  # 1

        qweight, qzeros, scales = quantize_weight_rtn(weight, bits=bits, group_size=group_size, sym=True)

        self.assertEqual(qweight.shape, (in_features // pack_factor, out_features))
        self.assertEqual(qzeros.shape, (num_groups, out_features // pack_factor))
        self.assertEqual(scales.shape, (num_groups, out_features))

    def test_outputs_are_on_cpu(self):
        """Verify that all returned tensors are on the CPU regardless of input device."""
        weight = torch.randn(16, 64)
        qweight, qzeros, scales = quantize_weight_rtn(weight, bits=4, group_size=32)
        self.assertEqual(qweight.device.type, "cpu")
        self.assertEqual(qzeros.device.type, "cpu")
        self.assertEqual(scales.device.type, "cpu")

    def test_raises_on_1d_weight(self):
        """quantize_weight_rtn must reject 1-D tensors."""
        with self.assertRaises(AssertionError):
            quantize_weight_rtn(torch.randn(64), bits=4, group_size=32)

    def test_in_features_padding(self):
        """in_features that is not a multiple of group_size should be padded internally."""
        # in_features=70 is not a multiple of group_size=32
        out_features, in_features = 16, 70
        bits, group_size = 4, 32
        weight = torch.randn(out_features, in_features)
        # padded in_features = 96 (next multiple of 32)
        padded_in = 96
        pack_factor = 32 // bits

        qweight, qzeros, scales = quantize_weight_rtn(weight, bits=bits, group_size=group_size)
        self.assertEqual(qweight.shape, (padded_in // pack_factor, out_features))


class TestGetWoqConfigFromDir(unittest.TestCase):
    """Tests for _get_woq_config_from_dir."""

    def test_returns_none_when_no_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_get_woq_config_from_dir(d))

    def test_returns_none_for_wrong_quant_method(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(
                d,
                {
                    "quant_method": "gptq",
                    "packing_format": "auto_round:auto_gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                },
            )
            self.assertIsNone(_get_woq_config_from_dir(d))

    def test_returns_none_for_wrong_packing_format(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(
                d,
                {
                    "quant_method": "auto-round",
                    "packing_format": "other_format",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                },
            )
            self.assertIsNone(_get_woq_config_from_dir(d))

    def test_returns_none_for_missing_bits(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(
                d,
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:auto_gptq",
                    "group_size": 128,
                    "sym": True,
                },
            )
            self.assertIsNone(_get_woq_config_from_dir(d))

    def test_returns_none_for_bits_greater_than_8(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(
                d,
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:auto_gptq",
                    "bits": 16,
                    "group_size": 128,
                    "sym": True,
                },
            )
            self.assertIsNone(_get_woq_config_from_dir(d))

    def test_valid_config_returns_required_fields(self):
        """A fully valid config returns a dict with bits, group_size, sym,
        block_name_to_quantize, and extra_config."""
        with tempfile.TemporaryDirectory() as d:
            _write_config(
                d,
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:auto_gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                },
            )
            cfg = _get_woq_config_from_dir(d)
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["bits"], 4)
            self.assertEqual(cfg["group_size"], 128)
            self.assertTrue(cfg["sym"])
            # New fields must also be present
            self.assertIn("block_name_to_quantize", cfg)
            self.assertIn("extra_config", cfg)

    def test_valid_config_with_extra_config_and_block_name(self):
        """extra_config and block_name_to_quantize from config.json are propagated."""
        with tempfile.TemporaryDirectory() as d:
            _write_config(
                d,
                {
                    "quant_method": "auto-round",
                    "packing_format": "auto_round:auto_gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                    "block_name_to_quantize": ["model.layers"],
                    "extra_config": {"mtp.0.fc": {"bits": 8}},
                },
            )
            cfg = _get_woq_config_from_dir(d)
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["block_name_to_quantize"], ["model.layers"])
            self.assertEqual(cfg["extra_config"], {"mtp.0.fc": {"bits": 8}})


class TestCopyMissingTensorsFromSource(unittest.TestCase):
    """End-to-end tests for copy_missing_tensors_from_source.

    Detection is now purely name-based (parent-layer + block-prefix comparison),
    so no model loading or mocking is required.

    A source tensor T is "missing" when ALL of:
      1. T is not already present in the saved output,
      2. T's parent layer (rsplit('.', 1)[0]) is absent from the saved parent layers,
      3. T's first numeric block-prefix (e.g. 'mtp.0') is absent from the saved
         block-prefixes — OR T has no numeric segment at all.
    Special case: 'lm_head.weight' is always excluded.
    """

    def _make_auto_round_config(self, bits=4, group_size=128, sym=True) -> dict:
        return {
            "quantization_config": {
                "quant_method": "auto-round",
                "packing_format": "auto_round:auto_gptq",
                "bits": bits,
                "group_size": group_size,
                "sym": sym,
            }
        }

    # ------------------------------------------------------------------ #
    # Detection correctness                                                #
    # ------------------------------------------------------------------ #

    def test_plain_copy_of_missing_mtp_tensor(self):
        """Tensors from blocks entirely absent in the saved output are copied verbatim."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            mtp_norm = torch.randn(64)
            _save_safetensors({"mtp.0.norm.weight": mtp_norm}, os.path.join(source_dir, "model.safetensors"))

            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(32, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertTrue(os.path.exists(extra_shard), "Extra shard should be created")

            result = _load_safetensors(extra_shard)
            self.assertIn("mtp.0.norm.weight", result)
            torch.testing.assert_close(result["mtp.0.norm.weight"], mtp_norm)

    def test_tensor_with_known_parent_layer_not_copied(self):
        """Source tensor whose parent layer already has tensors in the target is not missing.

        Scenario: source has the original '.weight'; target has the packed '.qweight'
        under the same parent layer — the layer was processed, not skipped.
        """
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors(
                {"model.layers.0.self_attn.q_proj.weight": torch.randn(32, 64)},
                os.path.join(source_dir, "model.safetensors"),
            )
            # Target has the quantised form — same parent layer, different suffix
            _save_safetensors(
                {"model.layers.0.self_attn.q_proj.qweight": torch.randint(0, 2**31, (8, 32), dtype=torch.int32)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertFalse(os.path.exists(extra_shard), "No extra shard expected: parent layer is known")

    def test_tensor_with_known_block_prefix_not_copied(self):
        """Source tensor whose numeric block-prefix exists in the saved output is not missing.

        Scenario: source has an extra gate weight inside 'model.layers.0'; the target
        already contains other tensors from 'model.layers.0' — the block was processed.
        """
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors(
                {"model.layers.0.mlp.extra_gate.weight": torch.randn(32, 64)},
                os.path.join(source_dir, "model.safetensors"),
            )
            # Target has a different sub-layer inside the same block
            _save_safetensors(
                {"model.layers.0.mlp.gate_proj.qweight": torch.randint(0, 2**31, (8, 32), dtype=torch.int32)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertFalse(os.path.exists(extra_shard), "No extra shard expected: block prefix is known")

    def test_lm_head_weight_is_never_copied(self):
        """lm_head.weight is always excluded, even when it is absent from the target."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors(
                {"lm_head.weight": torch.randn(32, 64)},
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(32, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertFalse(os.path.exists(extra_shard), "lm_head.weight must never be copied")

    def test_already_saved_tensors_are_not_duplicated(self):
        """Tensors that are already present in the target are not written again."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            mtp_tensor = torch.randn(64)
            _save_safetensors(
                {"mtp.0.shared_head.norm.weight": mtp_tensor},
                os.path.join(source_dir, "model.safetensors"),
            )
            # Target already has exactly this tensor
            _save_safetensors(
                {"mtp.0.shared_head.norm.weight": mtp_tensor},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertFalse(os.path.exists(extra_shard), "No extra shard expected: tensor already present")

    def test_detects_multiple_missing_tensors_from_different_blocks(self):
        """All source tensors from blocks absent in the saved output are copied."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors(
                {
                    "custom_block.0.norm.weight": torch.randn(64),
                    "mtp.0.norm.weight": torch.randn(64),
                    "model.embed_tokens.weight": torch.randn(8, 64),
                },
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertTrue(os.path.exists(extra_shard))

            result = _load_safetensors(extra_shard)
            self.assertIn("custom_block.0.norm.weight", result)
            self.assertIn("mtp.0.norm.weight", result)
            # Model-known tensor must NOT be in the extra shard
            self.assertNotIn("model.embed_tokens.weight", result)

    # ------------------------------------------------------------------ #
    # FP8 dequantization                                                   #
    # ------------------------------------------------------------------ #

    def test_fp8_weight_is_dequantized_to_bf16(self):
        """FP8 weights paired with weight_scale_inv are dequantized to BF16 before saving."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            fp8_weight = torch.randn(64, 128).to(torch.float8_e4m3fn)
            scale_inv = torch.tensor(0.5)

            _save_safetensors(
                {
                    "mtp.0.ffn.weight": fp8_weight,
                    "mtp.0.ffn.weight_scale_inv": scale_inv,
                },
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertTrue(os.path.exists(extra_shard))

            result = _load_safetensors(extra_shard)
            self.assertIn("mtp.0.ffn.weight", result)
            # Scale tensor must be dropped after dequantization
            self.assertNotIn("mtp.0.ffn.weight_scale_inv", result)
            self.assertEqual(result["mtp.0.ffn.weight"].dtype, torch.bfloat16)

    # ------------------------------------------------------------------ #
    # WOQ quantization                                                     #
    # ------------------------------------------------------------------ #

    def test_woq_quantization_replaces_weight_with_packed_tensors(self):
        """2-D MTP weight tensors are quantized and packed when an auto-round WOQ config is present."""
        out_features, in_features = 32, 128
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            mtp_weight = torch.randn(out_features, in_features)
            _save_safetensors(
                {"mtp.0.ffn.weight": mtp_weight},
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            with open(os.path.join(target_dir, "config.json"), "w") as f:
                json.dump(self._make_auto_round_config(bits=4, group_size=128, sym=True), f)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertTrue(os.path.exists(extra_shard))

            result = _load_safetensors(extra_shard)
            # Original weight must be replaced by the three packed tensors
            self.assertNotIn("mtp.0.ffn.weight", result)
            self.assertIn("mtp.0.ffn.qweight", result)
            self.assertIn("mtp.0.ffn.qzeros", result)
            self.assertIn("mtp.0.ffn.scales", result)

    def test_block_name_to_ignore_not_quantized_in_woq_mode(self):
        """Weights matching BLOCK_NAME_TO_IGNORE patterns are copied as-is, not quantized."""
        out_features, in_features = 32, 128
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            gate_weight = torch.randn(out_features, in_features)
            # "mlp.gate." is in BLOCK_NAME_TO_IGNORE
            _save_safetensors(
                {"mtp.0.mlp.gate.weight": gate_weight},
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            with open(os.path.join(target_dir, "config.json"), "w") as f:
                json.dump(self._make_auto_round_config(bits=4, group_size=128, sym=True), f)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertTrue(os.path.exists(extra_shard))

            result = _load_safetensors(extra_shard)
            # Weight must be present as-is — not packed
            self.assertIn("mtp.0.mlp.gate.weight", result)
            self.assertNotIn("mtp.0.mlp.gate.qweight", result)

    def test_config_updated_with_block_name_to_quantize_after_woq(self):
        """After WOQ, config.json is updated so the new block appears in block_name_to_quantize."""
        out_features, in_features = 32, 128
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors(
                {"mtp.0.ffn.weight": torch.randn(out_features, in_features)},
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            with open(os.path.join(target_dir, "config.json"), "w") as f:
                json.dump(self._make_auto_round_config(bits=4, group_size=128, sym=True), f)

            copy_missing_tensors_from_source(source_dir, target_dir)

            with open(os.path.join(target_dir, "config.json")) as f:
                updated_cfg = json.load(f)

            qcfg = updated_cfg.get("quantization_config", {})
            block_names = qcfg.get("block_name_to_quantize", [])
            self.assertTrue(
                any("mtp" in b for b in block_names),
                f"Expected an 'mtp' block in block_name_to_quantize, got: {block_names}",
            )

    def test_config_updated_with_extra_config_for_unquantized_2d_weight(self):
        """When a 2-D weight is skipped by BLOCK_NAME_TO_IGNORE, extra_config is updated
        to mark it as full-precision so the inference stack knows not to dequantize it."""
        out_features, in_features = 32, 128
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            # "mlp.gate." is in BLOCK_NAME_TO_IGNORE → not quantized → should go to extra_config
            _save_safetensors(
                {"mtp.0.mlp.gate.weight": torch.randn(out_features, in_features)},
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            with open(os.path.join(target_dir, "config.json"), "w") as f:
                json.dump(self._make_auto_round_config(bits=4, group_size=128, sym=True), f)

            copy_missing_tensors_from_source(source_dir, target_dir)

            with open(os.path.join(target_dir, "config.json")) as f:
                updated_cfg = json.load(f)

            extra_config = updated_cfg.get("quantization_config", {}).get("extra_config", {})
            self.assertIn("mtp.0.mlp.gate", extra_config)
            layer_cfg = extra_config["mtp.0.mlp.gate"]
            self.assertEqual(layer_cfg.get("bits"), 16)
            self.assertEqual(layer_cfg.get("data_type"), "fp")

    # ------------------------------------------------------------------ #
    # Index file handling                                                  #
    # ------------------------------------------------------------------ #

    def test_index_json_is_updated_for_sharded_target(self):
        """When target uses a sharded index, copied tensor names are added to weight_map."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            mtp_norm = torch.randn(64)
            _save_safetensors({"mtp.0.norm.weight": mtp_norm}, os.path.join(source_dir, "model.safetensors"))

            saved_shard = "model-00001-of-00001.safetensors"
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, saved_shard),
            )
            index = {"metadata": {"total_size": 0}, "weight_map": {"model.embed_tokens.weight": saved_shard}}
            with open(os.path.join(target_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            with open(os.path.join(target_dir, "model.safetensors.index.json")) as f:
                updated_index = json.load(f)

            self.assertIn("mtp.0.norm.weight", updated_index["weight_map"])
            self.assertEqual(updated_index["weight_map"]["mtp.0.norm.weight"], "model_extra_tensors.safetensors")

    def test_index_json_is_created_for_single_file_target(self):
        """When the target uses a single safetensors file and tensors are copied, a new
        model.safetensors.index.json is created mapping all tensors to their shards."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            mtp_norm = torch.randn(64)
            _save_safetensors({"mtp.0.norm.weight": mtp_norm}, os.path.join(source_dir, "model.safetensors"))

            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            index_path = os.path.join(target_dir, "model.safetensors.index.json")
            self.assertTrue(os.path.exists(index_path), "An index.json should be created for single-file targets")

            with open(index_path) as f:
                index = json.load(f)

            weight_map = index["weight_map"]
            self.assertIn("mtp.0.norm.weight", weight_map)
            self.assertIn("model.embed_tokens.weight", weight_map)
            self.assertEqual(weight_map["mtp.0.norm.weight"], "model_extra_tensors.safetensors")
            self.assertEqual(weight_map["model.embed_tokens.weight"], "model.safetensors")


if __name__ == "__main__":
    unittest.main()
