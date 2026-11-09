# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Packed export must not reverse a stale source FP8 loading conversion."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.core_model_loading import WeightConverter
from transformers.integrations.finegrained_fp8 import Fp8Dequantize

from auto_round.export.common.serialization import save_model
from auto_round.modeling.deepseek_v4 import checkpoint_name, checkpoint_state_dict


class PackedModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, parameter=False):
        super().__init__(PretrainedConfig())
        self.anchor = torch.nn.Parameter(torch.ones(1))
        packed = torch.arange(256, dtype=torch.uint8).repeat(128, 1)
        if parameter:
            self.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
        else:
            self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", torch.ones(128, 16, dtype=torch.float8_e4m3fn))
        self._weight_conversions = [
            WeightConverter(
                ["weight", "weight_scale_inv"],
                "weight",
                [Fp8Dequantize(SimpleNamespace(quantization_config=SimpleNamespace(weight_block_size=(128, 128))))],
            )
        ]


class TestPackedExport(unittest.TestCase):
    def test_native_v4_names_preserve_all_quantization_suffixes(self):
        pairs = {
            "model.layers.0.mlp.experts.7.gate_proj": "layers.0.ffn.experts.7.w1",
            "model.layers.0.mlp.experts.7.up_proj": "layers.0.ffn.experts.7.w3",
            "model.layers.0.mlp.experts.7.down_proj": "layers.0.ffn.experts.7.w2",
            "model.layers.0.mlp.shared_experts.down_proj": "layers.0.ffn.shared_experts.w2",
            "model.layers.0.self_attn.o_a_proj": "layers.0.attn.wo_a",
            "model.layers.0.self_attn.kv_norm": "layers.0.attn.kv_norm",
            "model.layers.0.self_attn.compressor.kv_norm": "layers.0.attn.compressor.norm",
            "model.layers.0.self_attn.compressor.indexer.kv_proj": "layers.0.attn.indexer.compressor.wkv",
            "model.layers.0.self_attn.compressor.indexer.scorer.weights_proj": "layers.0.attn.indexer.weights_proj",
            "model.layers.0.self_attn.compressor.indexer.q_b_proj": "layers.0.attn.indexer.wq_b",
            "model.embed_tokens": "embed",
            "lm_head": "head",
        }
        for source, target in pairs.items():
            for suffix in ("weight", "weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
                self.assertEqual(checkpoint_name(source + "." + suffix), target + "." + suffix)
        for source, target in {
            "model.layers.0.attn_hc.base": "layers.0.hc_attn_base",
            "model.layers.0.ffn_hc.scale": "layers.0.hc_ffn_scale",
            "model.hc_head.hc_fn": "hc_head_fn",
            "model.layers.0.mlp.gate.e_score_correction_bias": "layers.0.ffn.gate.bias",
            "model.layers.0.self_attn.sinks": "layers.0.attn.attn_sink",
        }.items():
            self.assertEqual(checkpoint_name(source), target)
            self.assertEqual(checkpoint_name(target), target)

    def test_native_v4_full_and_manual_save_preserve_bytes_and_metadata(self):
        for fallback in (False, True):
            model = PackedModel()
            model.config.model_type = "deepseek_v4"
            model.model = torch.nn.Module()
            model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            model.model.layers[0].mlp = torch.nn.Module()
            model.model.layers[0].mlp.experts = torch.nn.ModuleList([torch.nn.Module()])
            model.model.layers[0].mlp.experts[0].gate_proj = torch.nn.Module()
            proj = model.model.layers[0].mlp.experts[0].gate_proj
            proj.register_buffer("weight_packed", model.weight_packed.clone())
            proj.register_buffer("weight_scale", model.weight_scale.clone())
            module_name = "model.layers.0.mlp.experts.0.gate_proj"
            original_config = {
                "ignore": [module_name],
                "extra_config": {module_name: {"bits": 16}},
                "config_groups": {"group_0": {"targets": [module_name]}},
            }
            model.config.quantization_config = original_config
            conversions = model._weight_conversions
            with tempfile.TemporaryDirectory() as directory:
                with (
                    patch("auto_round.export.common.serialization.apply_post_save_source_fixes"),
                    patch("auto_round.export.common.serialization._save_source_config", return_value=False),
                ):
                    if fallback:
                        with patch.object(model, "save_pretrained", side_effect=TypeError("config failure")):
                            save_model(model, directory)
                    else:
                        save_model(model, directory)
                saved = load_file(directory + "/model.safetensors")
                target = "layers.0.ffn.experts.0.w1"
                self.assertNotIn(module_name + ".weight_packed", saved)
                torch.testing.assert_close(saved[target + ".weight_packed"], proj.weight_packed)
                torch.testing.assert_close(
                    saved[target + ".weight_scale"].view(torch.uint8), proj.weight_scale.view(torch.uint8)
                )
                with open(directory + "/quantization_config.json") as f:
                    qconfig = json.load(f)
                with open(directory + "/config.json") as f:
                    self.assertEqual(json.load(f)["quantization_config"], qconfig)
                self.assertEqual(qconfig["ignore"], [target])
                self.assertEqual(qconfig["config_groups"]["group_0"]["targets"], [target])
                self.assertIn(target, qconfig["extra_config"])
            self.assertIs(model._weight_conversions, conversions)
            self.assertEqual(model.config.quantization_config, original_config)

    def test_native_v4_fused_experts_and_collision(self):
        tensor = torch.arange(48).reshape(2, 6, 4)
        saved = checkpoint_state_dict({"model.layers.0.mlp.experts.gate_up_proj": tensor})
        self.assertEqual(len(saved), 4)
        torch.testing.assert_close(saved["layers.0.ffn.experts.1.w1.weight"], tensor[1, :3])
        torch.testing.assert_close(saved["layers.0.ffn.experts.1.w3.weight"], tensor[1, 3:])
        with self.assertRaisesRegex(ValueError, "collision"):
            checkpoint_state_dict({"model.embed_tokens.weight": tensor, "embed.weight": tensor})

    def test_native_v4_shard_writer_and_config_reverter(self):
        from auto_round.compressors.shard_writer import ShardWriter
        from auto_round.export.formats.compressed_tensors.utils import get_checkpoint_name_reverter

        model = PackedModel()
        model.config.model_type = "deepseek_v4"
        ShardWriter.reset()
        try:
            writer = ShardWriter(model, bits=4)
            name = "model.layers.0.mlp.experts.0.gate_proj.weight_packed"
            writer._add_tensor(name, model.weight_packed)
            self.assertIn("layers.0.ffn.experts.0.w1.weight_packed", writer.current_shard_tensors)
            tensor = torch.ones(2, 4, 8)
            writer._add_tensor("model.layers.0.mlp.experts.gate_up_proj", tensor)
            self.assertIn("layers.0.ffn.experts.1.w3.weight", writer.current_shard_tensors)
            self.assertEqual(get_checkpoint_name_reverter(model)(name), "layers.0.ffn.experts.0.w1.weight_packed")
            with tempfile.TemporaryDirectory() as directory:
                with (
                    patch.object(ShardWriter, "output_dir", new_callable=PropertyMock, return_value=directory),
                    patch.object(writer, "_offload_to_meta"),
                ):
                    writer._flush_shard()
                shard = load_file(directory + "/" + writer.shard_meta[0]["tmp_file"])
                torch.testing.assert_close(shard["layers.0.ffn.experts.0.w1.weight_packed"], model.weight_packed)
                self.assertFalse(any(key.startswith("model.") for key in shard))
        finally:
            ShardWriter.reset()

    def test_stale_fp8_converter_does_not_modify_packed_bytes(self):
        for parameter in (False, True):
            with self.subTest(parameter=parameter):
                model = PackedModel(parameter=parameter)
                conversions = model._weight_conversions
                with tempfile.TemporaryDirectory() as directory:
                    with (
                        patch("auto_round.export.common.serialization.apply_post_save_source_fixes"),
                        patch("auto_round.export.common.serialization._save_source_config"),
                    ):
                        save_model(model, directory, dtype=torch.bfloat16)
                    saved = load_file(directory + "/model.safetensors")
                self.assertEqual(saved["weight_packed"].dtype, torch.uint8)
                torch.testing.assert_close(saved["weight_packed"], model.weight_packed)
                torch.testing.assert_close(
                    saved["weight_scale"].view(torch.uint8), model.weight_scale.view(torch.uint8)
                )
                self.assertNotIn("weight_packed_scale", saved)
                self.assertIs(model._weight_conversions, conversions)

    def test_legacy_save_pretrained_receives_no_new_keyword(self):
        model = PackedModel()
        calls = []

        def legacy_save(directory, max_shard_size, safe_serialization):
            calls.append(directory)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(model, "save_pretrained", legacy_save),
            patch("auto_round.export.common.serialization.apply_post_save_source_fixes"),
            patch("auto_round.export.common.serialization._save_source_config"),
        ):
            save_model(model, directory)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
