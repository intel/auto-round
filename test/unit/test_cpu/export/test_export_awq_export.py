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
"""Tests for ``auto_round/export/export_to_awq/export.py``."""

import torch
import torch.nn as nn

from auto_round.export.export_to_awq.export import _is_supported_layer


class TestIsSupportedLayer:
    def test_linear_is_supported(self):
        layer = nn.Linear(8, 8)
        assert _is_supported_layer(layer) is True

    def test_conv1d_via_classname(self):
        """INNER_SUPPORTED_LAYER_TYPES is matched by classname; FP8Linear is
        the only CPU-reachable classname-based entry. We simulate a fake class
        with a known classname."""

        class _FakeInner(nn.Module):
            pass

        # Default classname is "FakeInner" — not in the supported list
        assert _is_supported_layer(_FakeInner()) is False

    def test_arbitrary_module_not_supported(self):
        """A Conv2d is not in SUPPORTED_LAYER_TYPES."""
        conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        assert _is_supported_layer(conv) is False

    def test_conv1d_is_supported(self):
        """A transformers Conv1D is in SUPPORTED_LAYER_TYPES."""
        try:
            from transformers.pytorch_utils import Conv1D
        except ImportError:
            pytest.skip("transformers Conv1D not available")
        c1d = Conv1D(nf=8, nx=8)
        assert _is_supported_layer(c1d) is True
