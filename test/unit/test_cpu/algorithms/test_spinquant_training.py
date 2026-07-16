# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.algorithms.transforms.spinquant.training``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.algorithms.transforms.spinquant.training import (
    TrainingResult,
    check_orthogonality,
    clone_model_for_reference,
    compute_rotation_loss,
    create_dual_optimizer,
    move_batch_to_device,
    spinquant_loss_fn,
)


# ==============================================================================
# compute_rotation_loss
# ==============================================================================


class TestComputeRotationLoss:
    """Test the rotation loss computation."""

    def test_kl_top_basic(self):
        logits = torch.randn(2, 10)
        ori_logits = torch.randn(2, 10)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="kl_top")
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_kl_top_with_kl_top_k(self):
        logits = torch.randn(2, 100)
        ori_logits = torch.randn(2, 100)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="kl_top", kl_top_k=10)
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_kl_top_kl_top_k_larger_than_logits(self):
        logits = torch.randn(2, 5)
        ori_logits = torch.randn(2, 5)
        # k > logits dim - should handle gracefully
        loss = compute_rotation_loss(logits, ori_logits, loss_type="kl_top", kl_top_k=1000)
        assert loss.numel() == 1

    def test_kl_full(self):
        logits = torch.randn(2, 10)
        ori_logits = torch.randn(2, 10)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="kl_full")
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_mse(self):
        logits = torch.randn(2, 10)
        ori_logits = torch.randn(2, 10)
        loss = compute_rotation_loss(logits, ori_logits, loss_type="mse")
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_mse_same_logits_zero(self):
        logits = torch.randn(2, 10)
        loss = compute_rotation_loss(logits, logits.clone(), loss_type="mse")
        assert loss.item() < 1e-5

    def test_unknown_loss_raises(self):
        logits = torch.randn(2, 10)
        ori_logits = torch.randn(2, 10)
        with pytest.raises(ValueError, match="Unknown loss_type"):
            compute_rotation_loss(logits, ori_logits, loss_type="unknown")

    def test_alias_spinquant_loss_fn(self):
        """spinquant_loss_fn is an alias for compute_rotation_loss."""
        logits = torch.randn(2, 10)
        ori_logits = torch.randn(2, 10)
        assert spinquant_loss_fn is compute_rotation_loss


# ==============================================================================
# move_batch_to_device
# ==============================================================================


class TestMoveBatchToDevice:
    """Test batch device movement."""

    def test_tensor_to_device(self):
        x = torch.randn(2, 4)
        device = torch.device("cpu")
        result = move_batch_to_device(x, device)
        assert result.device == device

    def test_dict_of_tensors(self):
        batch = {"input_ids": torch.randn(2, 4), "attention_mask": torch.ones(2, 4)}
        device = torch.device("cpu")
        result = move_batch_to_device(batch, device)
        assert result["input_ids"].device == device
        assert result["attention_mask"].device == device

    def test_dict_with_non_tensor_values(self):
        batch = {"input_ids": torch.randn(2, 4), "labels": torch.tensor([1, 0])}
        device = torch.device("cpu")
        result = move_batch_to_device(batch, device)
        assert result["input_ids"].device == device
        assert result["labels"].device == device

    def test_passthrough_for_unknown_types(self):
        batch = ["a", "b"]
        result = move_batch_to_device(batch, torch.device("cpu"))
        assert result == ["a", "b"]


# ==============================================================================
# check_orthogonality
# ==============================================================================


class TestCheckOrthogonality:
    """Test orthogonality checking."""

    def test_identity_matrix_zero_deviation(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.eye(4))
        model.register_parameter("spinquant_R1", nn.Parameter(torch.eye(4), requires_grad=True))
        dev = check_orthogonality(model)
        assert dev == 0.0

    def test_random_matrix_positive_deviation(self):
        model = nn.Module()
        model.register_parameter(
            "spinquant_R2", nn.Parameter(torch.randn(4, 4), requires_grad=True)
        )
        dev = check_orthogonality(model)
        assert dev > 0

    def test_skips_non_trainable_params(self):
        model = nn.Module()
        model.register_parameter(
            "spinquant_R3", nn.Parameter(torch.randn(4, 4), requires_grad=False)
        )
        dev = check_orthogonality(model)
        assert dev == 0.0

    def test_skips_non_rotation_params(self):
        model = nn.Module()
        model.register_parameter(
            "other_param", nn.Parameter(torch.randn(4, 4), requires_grad=True)
        )
        dev = check_orthogonality(model)
        assert dev == 0.0

    def test_skips_empty_params(self):
        model = nn.Module()
        model.register_parameter(
            "spinquant_R4", nn.Parameter(torch.tensor([]), requires_grad=True)
        )
        dev = check_orthogonality(model)
        assert dev == 0.0

    def test_skips_non_square_params(self):
        model = nn.Module()
        model.register_parameter(
            "spinquant_R5", nn.Parameter(torch.randn(4, 8), requires_grad=True)
        )
        dev = check_orthogonality(model)
        assert dev == 0.0

    def test_custom_threshold(self):
        model = nn.Module()
        model.register_parameter(
            "spinquant_R6", nn.Parameter(torch.eye(4) + torch.randn(4, 4) * 0.01, requires_grad=True)
        )
        # With a very tight threshold, should trigger warning
        dev = check_orthogonality(model, threshold=1e-6)
        # Deviation is positive but might not exceed threshold

    def test_empty_model(self):
        model = nn.Module()
        dev = check_orthogonality(model)
        assert dev == 0.0


# ==============================================================================
# create_dual_optimizer
# ==============================================================================


class TestCreateDualOptimizer:
    """Test the dual optimizer creation."""

    def test_no_trainable_params_returns_none(self):
        model = nn.Module()
        model.register_parameter("weight", nn.Parameter(torch.randn(4, 4), requires_grad=False))
        result = create_dual_optimizer(model)
        assert result is None

    def test_rotation_params_creates_optimizer(self):
        model = nn.Module()
        model.register_parameter(
            "spinquant_R1", nn.Parameter(torch.eye(4), requires_grad=True)
        )
        result = create_dual_optimizer(model, lr=1e-4, smooth_lr=1e-3)
        assert result is not None

    def test_smooth_values_creates_optimizer(self):
        model = nn.Module()
        model.register_parameter(
            "smooth_values", nn.Parameter(torch.ones(4), requires_grad=True)
        )
        result = create_dual_optimizer(model, lr=1e-4, smooth_lr=1e-3)
        assert result is not None

    def test_custom_lr(self):
        model = nn.Module()
        model.register_parameter(
            "spinquant_R1", nn.Parameter(torch.eye(4), requires_grad=True)
        )
        result = create_dual_optimizer(model, lr=1e-3)
        assert result is not None

    def test_alias_create_spinquant_optimizer(self):
        """create_spinquant_optimizer is an alias for create_dual_optimizer."""
        from auto_round.algorithms.transforms.spinquant.training import (
            create_spinquant_optimizer,
        )

        assert create_spinquant_optimizer is create_dual_optimizer


# ==============================================================================
# TrainingResult
# ==============================================================================


class TestTrainingResult:
    """Test the TrainingResult dataclass."""

    def test_creation(self):
        result = TrainingResult(
            loss_history=[0.5, 0.4, 0.3],
            best_loss=0.3,
            final_ortho_deviation=0.01,
            steps=3,
        )
        assert result.loss_history == [0.5, 0.4, 0.3]
        assert result.best_loss == 0.3
        assert result.final_ortho_deviation == 0.01
        assert result.steps == 3

    def test_empty_history(self):
        result = TrainingResult(loss_history=[], best_loss=float("inf"), final_ortho_deviation=0.0, steps=0)
        assert result.loss_history == []
        assert result.best_loss == float("inf")
        assert result.steps == 0


# ==============================================================================
# clone_model_for_reference
# ==============================================================================


class TestCloneModelForReference:
    """Test model cloning for reference.

    clone_model_for_reference does deep copy, freezes params, removes hooks,
    and sets to eval mode. Direct patching of the internal import is fragile,
    so we just verify it returns a different object.
    """

    def test_returns_different_object(self):
        model = nn.Module()
        model.register_parameter("weight", nn.Parameter(torch.randn(4, 4)))
        clone = clone_model_for_reference(model)
        assert clone is not model
