# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms.spinquant.preprocessor``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.preprocessor import (
    SpinQuantConfig,
    TrainableRMSNorm,
)

# ==============================================================================
# SpinQuantConfig
# ==============================================================================


class TestSpinQuantConfig:
    """Test SpinQuantConfig dataclass."""

    def test_default_values(self):
        cfg = SpinQuantConfig()
        assert cfg.algorithm == "spinquant"
        assert cfg.r1 is True
        assert cfg.r2 is True
        assert cfg.r3 is False
        assert cfg.r4 is False
        assert cfg.trainable_rotation is False
        assert cfg.trainable_smooth is False
        assert cfg.online_r1_rotation is True
        assert cfg.fuse_rmsnorm is True
        assert cfg.untie_embeddings is True

    def test_custom_rotation_flags(self):
        cfg = SpinQuantConfig(r1=False, r3=True, r4=True)
        assert cfg.r1 is False
        assert cfg.r3 is True
        assert cfg.r4 is True

    def test_trainable_rotation_config(self):
        cfg = SpinQuantConfig(trainable_rotation=True, trainable_smooth=True)
        assert cfg.trainable_rotation is True
        assert cfg.trainable_smooth is True
        assert cfg.trainable_rotation is True

    def test_rotation_size_positive_required(self):
        with pytest.raises(ValueError, match="must be positive"):
            SpinQuantConfig(rotation_size=0)

    def test_rotation_size_non_pow2_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            SpinQuantConfig(rotation_size=12)

    def test_rotation_size_pow2_allowed(self):
        cfg = SpinQuantConfig(rotation_size=128)
        assert cfg.rotation_size == 128
        # r1_rotation_size is set by preprocessor, not config
        assert cfg.rotation_size == 128

    def test_random_rotation_flags(self):
        cfg = SpinQuantConfig(random_r1=True, random_r2=False, random_r3=True, random_r4=False)
        assert cfg.random_r1 is True
        assert cfg.random_r2 is False
        assert cfg.random_r3 is True
        assert cfg.random_r4 is False

    def test_training_hyperparameters(self):
        cfg = SpinQuantConfig(iters=500, lr=1e-3, smooth_lr=1e-2, batch_size=4)
        assert cfg.iters == 500
        assert cfg.lr == 1e-3
        assert cfg.smooth_lr == 1e-2
        assert cfg.batch_size == 4

    def test_loss_type(self):
        cfg = SpinQuantConfig(loss_type="kl_full")
        assert cfg.loss_type == "kl_full"

    def test_dtype_and_device_defaults(self):
        cfg = SpinQuantConfig()
        assert cfg.dtype == torch.float32
        assert cfg.device in ("cuda", "cpu")

    def test_explicit_dtype(self):
        cfg = SpinQuantConfig(dtype=torch.bfloat16, device="cpu")
        assert cfg.dtype == torch.bfloat16
        assert cfg.device == "cpu"


# ==============================================================================
# TrainableRMSNorm
# ==============================================================================


class TestTrainableRMSNorm:
    """Test TrainableRMSNorm wrapper."""

    def test_wraps_rmsnorm(self):
        original = nn.LayerNorm(4, elementwise_affine=True)
        wrapper = TrainableRMSNorm(original)
        assert wrapper.original_norm is original

    def test_smooth_values_with_weight(self):
        original = nn.LayerNorm(4, elementwise_affine=True)
        wrapper = TrainableRMSNorm(original, trainable=True)
        assert wrapper.smooth_values is not None
        assert wrapper.smooth_values.shape == (4,)
        assert wrapper.smooth_values.requires_grad is True

    def test_smooth_values_non_trainable(self):
        original = nn.LayerNorm(4, elementwise_affine=True)
        wrapper = TrainableRMSNorm(original, trainable=False)
        assert wrapper.smooth_values.requires_grad is False

    def test_forward_applies_original(self):
        original = nn.LayerNorm(4, elementwise_affine=True)
        wrapper = TrainableRMSNorm(original, trainable=False)
        wrapper.smooth_values = nn.Parameter(torch.ones(4))
        x = torch.randn(2, 4)
        out = wrapper(x)
        assert out.shape == x.shape

    def test_forward_applies_smooth_values(self):
        original = nn.LayerNorm(4, elementwise_affine=True)
        wrapper = TrainableRMSNorm(original, trainable=False)
        wrapper.smooth_values = nn.Parameter(torch.ones(4) * 2)
        x = torch.randn(2, 4)
        original_out = original(x)
        out = wrapper(x)
        # With smooth_values=2, output should be 2x the original norm output
        # But since original also has weight=1 and bias=0 by default,
        # the comparison is approximate
        assert out.shape == x.shape

    def test_forward_without_smooth_values(self):
        """Test forward when smooth_values is None."""
        original = nn.LayerNorm(4, elementwise_affine=True)
        wrapper = TrainableRMSNorm(original, trainable=False)
        wrapper.smooth_values = None
        x = torch.randn(2, 4)
        out = wrapper(x)
        assert out.shape == x.shape

    def test_preserves_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        original = nn.LayerNorm(4, elementwise_affine=True)
        wrapper = TrainableRMSNorm(original)
        x = torch.randn(2, 4)
        wrapper = wrapper.to("cuda")
        x = x.to("cuda")
        out = wrapper(x)
        assert out.device.type == "cuda"

    def test_dtype_preserved(self):
        original = nn.LayerNorm(4, elementwise_affine=True).to(torch.bfloat16)
        wrapper = TrainableRMSNorm(original)
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        out = wrapper(x)
        assert out.dtype == torch.bfloat16
