# Copyright (c) 2026 Intel Corporation
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

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from auto_round.compressors.utils import set_layer_config

# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    """A tiny two-linear block, used to build multi-layer test models."""

    def __init__(self, size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)


class _SimpleModel(nn.Module):
    """Model with 12 numbered blocks + lm_head.

    Named modules that matter for these tests (all ``nn.Linear`` leaves):
      layers.0.fc1, layers.0.fc2,
      layers.1.fc1, layers.1.fc2,
      ...
      layers.11.fc1, layers.11.fc2,
      lm_head
    """

    def __init__(self, num_layers: int = 12, size: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([_Block(size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(size, size * 2, bias=False)

    def forward(self, x):  # pragma: no cover
        for block in self.layers:
            x = block.fc1(x) + block.fc2(x)
        return self.lm_head(x)


# Shared helpers
_SUPPORTED_TYPES = (nn.Linear,)
_INNER_SUPPORTED_TYPES = ()
_DEFAULT_SCHEME = "W4A16"
_SCALE_DTYPE = torch.float16


def _call_set_layer_config(model, layer_config=None, ignore_layers="", quant_lm_head=False):
    """Thin wrapper so tests don't repeat all positional args."""
    return set_layer_config(
        model=model,
        layer_config=layer_config or {},
        default_scheme=_DEFAULT_SCHEME,
        default_scale_dtype=_SCALE_DTYPE,
        supported_types=_SUPPORTED_TYPES,
        inner_supported_types=_INNER_SUPPORTED_TYPES,
        ignore_layers=ignore_layers,
        quant_lm_head=quant_lm_head,
    )


@pytest.fixture
def model_12layers():
    return _SimpleModel(num_layers=12)


@pytest.fixture
def model():
    return _SimpleModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIgnoreLayersDigitSuffix:
    """A digit-ending name should match only its own subtree, not higher-numbered ones."""

    def test_digit_ending_name_does_not_bleed_into_higher_indices(self, model_12layers):
        """'layers.1' should ignore layers.1.* but NOT layers.10.* or layers.11.*"""
        layer_config, _, _ = _call_set_layer_config(model_12layers, ignore_layers="layers.1")

        # layers.1 subtree must be fp16
        assert "layers.1.fc1" in layer_config
        assert layer_config["layers.1.fc1"]["bits"] == 16
        assert "layers.1.fc2" in layer_config
        assert layer_config["layers.1.fc2"]["bits"] == 16

        # layers.10 and layers.11 must NOT be force-fp16 by this setting
        # (they are absent from ignore-forced entries; their bits should be 4 from default)
        assert layer_config.get("layers.10.fc1", {}).get("bits", 4) != 16
        assert layer_config.get("layers.11.fc1", {}).get("bits", 4) != 16

    def test_block_level_name_ignores_entire_block(self, model_12layers):
        """'layers.0' (digit-ending) should force-fp16 both fc1 and fc2 under layers.0."""
        layer_config, _, _ = _call_set_layer_config(model_12layers, ignore_layers="layers.0")
        assert layer_config["layers.0.fc1"]["bits"] == 16
        assert layer_config["layers.0.fc2"]["bits"] == 16


class TestUnmatchedLayerConfigWarns:
    """An unrecognised key in layer_config must trigger a warning, not ValueError."""

    def test_unknown_name_emits_warning_not_valueerror(self, model):
        """A layer name that matches nothing should warn and not raise."""
        from auto_round.logger import logger

        bad_layer_config = {"nonexistent_layer": {"bits": 4}}
        with patch.object(logger, "warning") as mock_warn:
            layer_config, _, _ = _call_set_layer_config(model, layer_config=bad_layer_config)

        # The warning should mention the bad name
        warning_messages = " ".join(str(call) for call in mock_warn.call_args_list)
        assert "nonexistent_layer" in warning_messages
