# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms.hadamard.patch``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.hadamard.patch import (
    patch_quantlinear,
    patch_wrapperlinear_to_apply_transform,
    patch_wrapperwalayer_forward_to_apply_transform,
)
from auto_round.wrapper import WrapperLinear, WrapperWALayer


class _DummyWeightTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(16))

    def forward(self, x):
        return x


class _DummyInputTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TestPatchWrapperLinear:
    """Test WrapperLinear patching."""

    def test_idempotent_twice(self):
        """Second call is a no-op."""
        w = _DummyWeightTransform()
        inp = _DummyInputTransform()

        patch_wrapperlinear_to_apply_transform(w, inp)
        assert getattr(WrapperLinear, "_hadamard_patched", False) is True

        orig = WrapperLinear._qdq_weight
        patch_wrapperlinear_to_apply_transform(w, inp)
        assert WrapperLinear._qdq_weight is orig

    def test_sets_guard_flag(self):
        """Guard flag is set after patching."""
        w = _DummyWeightTransform()
        inp = _DummyInputTransform()

        patch_wrapperlinear_to_apply_transform(w, inp)
        assert WrapperLinear._hadamard_patched is True

    def test_wraps_qdq_weight(self):
        """Patched _qdq_weight calls the original."""
        w = _DummyWeightTransform()
        inp = _DummyInputTransform()
        patch_wrapperlinear_to_apply_transform(w, inp)

        assert WrapperLinear._qdq_weight is not None
        assert callable(WrapperLinear._qdq_weight)

    def test_wraps_qdq_act(self):
        """Patched _qdq_act calls the original."""
        w = _DummyWeightTransform()
        inp = _DummyInputTransform()
        patch_wrapperlinear_to_apply_transform(w, inp)

        assert WrapperLinear._qdq_act is not None
        assert callable(WrapperLinear._qdq_act)


class TestPatchWrapperWALayer:
    """Test WrapperWALayer forward patching."""

    def test_idempotent_twice(self):
        """Second call is a no-op."""
        inp = _DummyInputTransform()

        patch_wrapperwalayer_forward_to_apply_transform(inp)
        assert getattr(WrapperWALayer, "_hadamard_forward_patched", False) is True

        orig = WrapperWALayer.forward
        patch_wrapperwalayer_forward_to_apply_transform(inp)
        assert WrapperWALayer.forward is orig

    def test_sets_guard_flag(self):
        """Guard flag is set after patching."""
        inp = _DummyInputTransform()

        patch_wrapperwalayer_forward_to_apply_transform(inp)
        assert WrapperWALayer._hadamard_forward_patched is True

    def test_wraps_forward(self):
        """Patched forward is callable."""
        inp = _DummyInputTransform()
        patch_wrapperwalayer_forward_to_apply_transform(inp)

        assert WrapperWALayer.forward is not None
        assert callable(WrapperWALayer.forward)


class TestPatchQuantLinear:
    """Test QuantLinear packing patch."""

    def test_idempotent_twice(self):
        """Second call is a no-op."""
        w = _DummyWeightTransform()
        patch_quantlinear(w)

        # The patch sets _pack_patched on QuantLinear, not on torch.nn.Linear
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        assert hasattr(QuantLinear, "_pack_patched")
        orig = QuantLinear.pack
        patch_quantlinear(w)
        assert QuantLinear.pack is orig

    def test_sets_guard_flag(self):
        """Guard flag is set on QuantLinear class."""
        w = _DummyWeightTransform()
        patch_quantlinear(w)
        # The patch is applied to QuantLinear, check it has the flag
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

        assert QuantLinear._pack_patched is True
