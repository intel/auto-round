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

    def test_valid_config_returns_dict(self):
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


class TestCopyMissingTensorsFromSource(unittest.TestCase):
    """End-to-end tests for copy_missing_tensors_from_source."""

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

    def test_plain_copy_of_missing_mtp_tensor(self):
        """Non-weight tensors tagged with 'mtp' prefix are copied verbatim."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            # Source has an MTP norm tensor (1-D, not a weight)
            mtp_norm = torch.randn(64)
            _save_safetensors({"mtp.0.norm.weight": mtp_norm}, os.path.join(source_dir, "model.safetensors"))

            # Target has a different tensor (already saved)
            saved_tensor = torch.randn(32, 64)
            _save_safetensors(
                {"model.embed_tokens.weight": saved_tensor}, os.path.join(target_dir, "model.safetensors")
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertTrue(os.path.exists(extra_shard), "Extra shard should be created")

            result = _load_safetensors(extra_shard)
            self.assertIn("mtp.0.norm.weight", result)
            torch.testing.assert_close(result["mtp.0.norm.weight"], mtp_norm)

    def test_non_mtp_tensors_are_not_copied(self):
        """Tensors without the 'mtp' prefix should not be copied."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors(
                {
                    "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 64),
                    "model.embed_tokens.weight": torch.randn(8, 64),
                },
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)}, os.path.join(target_dir, "model.safetensors")
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertFalse(os.path.exists(extra_shard), "No extra shard expected when all tensors match")

    def test_already_saved_tensors_are_not_duplicated(self):
        """Tensors already present in target should not appear in the extra shard."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            mtp_tensor = torch.randn(64)
            _save_safetensors(
                {"mtp.0.shared_head.norm.weight": mtp_tensor, "model.norm.weight": torch.randn(64)},
                os.path.join(source_dir, "model.safetensors"),
            )
            # Target already has the mtp tensor
            _save_safetensors(
                {"mtp.0.shared_head.norm.weight": mtp_tensor},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir)

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertFalse(os.path.exists(extra_shard))

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
            self.assertNotIn("mtp.0.ffn.weight_scale_inv", result)
            self.assertEqual(result["mtp.0.ffn.weight"].dtype, torch.bfloat16)

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
            # Original weight tensor must be replaced by packed tensors
            self.assertNotIn("mtp.0.ffn.weight", result)
            self.assertIn("mtp.0.ffn.qweight", result)
            self.assertIn("mtp.0.ffn.qzeros", result)
            self.assertIn("mtp.0.ffn.scales", result)

    def test_index_json_is_updated_for_sharded_target(self):
        """When target uses a sharded index, copied tensor names are added to weight_map."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            mtp_norm = torch.randn(64)
            _save_safetensors({"mtp.0.norm.weight": mtp_norm}, os.path.join(source_dir, "model.safetensors"))

            # Create a sharded target with an index file
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

    def test_empty_prefix_list_skips_copy(self):
        """Passing an empty prefix list should result in no copying."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors({"mtp.0.norm.weight": torch.randn(64)}, os.path.join(source_dir, "model.safetensors"))
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir, missing_param_prefix=[])

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertFalse(os.path.exists(extra_shard))

    def test_custom_prefix_matches_correctly(self):
        """A custom prefix should match only tensors whose components start with that prefix."""
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            _save_safetensors(
                {
                    "custom_block.0.norm.weight": torch.randn(64),
                    "mtp.0.norm.weight": torch.randn(64),
                },
                os.path.join(source_dir, "model.safetensors"),
            )
            _save_safetensors(
                {"model.embed_tokens.weight": torch.randn(8, 64)},
                os.path.join(target_dir, "model.safetensors"),
            )
            _write_config(target_dir)

            copy_missing_tensors_from_source(source_dir, target_dir, missing_param_prefix=["custom_block"])

            extra_shard = os.path.join(target_dir, "model_extra_tensors.safetensors")
            self.assertTrue(os.path.exists(extra_shard))

            result = _load_safetensors(extra_shard)
            self.assertIn("custom_block.0.norm.weight", result)
            self.assertNotIn("mtp.0.norm.weight", result)


if __name__ == "__main__":
    unittest.main()
