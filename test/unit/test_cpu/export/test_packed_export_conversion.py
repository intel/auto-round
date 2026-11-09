# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Packed export must not reverse a stale source FP8 loading conversion."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.core_model_loading import WeightConverter
from transformers.integrations.finegrained_fp8 import Fp8Dequantize

from auto_round.export.common.serialization import save_model


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
