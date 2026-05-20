# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
Training utilities for SpinQuant in AutoRound.

⚠️  **Experimental**: These training utilities provide hooks and helpers for
integrating SpinQuant training into AutoRound's calibration pipeline, but the
training path has NOT been validated end-to-end. For production use, prefer
QuaRot mode (fixed Hadamard, no training) via ``SpinQuantPreprocessor``.

Provides hooks, callbacks, and helper functions for integrating
SpinQuant training into AutoRound's calibration pipeline.

Key design: SpinQuant training happens BEFORE AutoRound's block-wise
quantization calibration. After rotation matrices are learned and fused,
the model is mathematically equivalent to the original, so AutoRound
can apply its standard quantization without any modifications.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.cayley_optimizer import (
    AdamAndSGDG,
)
from auto_round.algorithms.transforms.spinquant.preprocessor import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
)


class SpinQuantTrainingHook:
    """
    A hook that can be inserted into AutoRound's calibration pipeline.

    This hook runs SpinQuant preprocessing before the main quantization
    calibration, using the same dataloader that AutoRound has prepared.

    Example integration::

        >>> from auto_round.calibration import get_dataloader
        >>> dataloader = get_dataloader(tokenizer, dataset, ...)
        >>> hook = SpinQuantTrainingHook(model, config)
        >>> hook.preprocess(dataloader)
        >>> # Now proceed with AutoRound quantization
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SpinQuantConfig] = None,
        enabled: bool = True,
    ):
        self.model = model
        self.config = config or SpinQuantConfig()
        self.enabled = enabled
        self.preprocessor: Optional[SpinQuantPreprocessor] = None

    def preprocess(self, dataloader: Any) -> nn.Module:
        """Execute SpinQuant preprocessing."""
        if not self.enabled:
            return self.model

        self.preprocessor = SpinQuantPreprocessor(self.model, self.config)
        return self.preprocessor.preprocess(dataloader)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return trainable parameters for external optimizer management."""
        if self.preprocessor is None:
            return []
        return self.preprocessor.rotation_params + self.preprocessor.smooth_params


class OrthogonalTrainingCallback:
    """
    Monitor orthogonality constraint during training.

    Tracks ``R @ R.T ≈ I`` for all rotation matrices and reports
    maximum deviation after each training step.
    """

    def __init__(self, model: nn.Module, log_interval: int = 50):
        self.model = model
        self.log_interval = log_interval
        self.step = 0
        self.max_deviation_history: list[float] = []

    def on_step_end(self) -> None:
        """Call after each training step to check orthogonality."""
        self.step += 1
        if self.step % self.log_interval != 0:
            return

        max_dev = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or "spinquant_R" not in name:
                continue

            R = param.data
            if R.dim() != 2 or R.shape[0] != R.shape[1]:
                continue

            I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
            deviation = (torch.matmul(R, R.t()) - I).abs().max().item()
            max_dev = max(max_dev, deviation)

        self.max_deviation_history.append(max_dev)

        if max_dev > 1e-4:
            print(f"[SpinQuant] Step {self.step}: max orthogonality deviation = {max_dev:.2e}")

        if max_dev > 1e-2:
            print("  🔴 WARNING: Orthogonality significantly violated! Consider reducing lr.")


def create_spinquant_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    smooth_lr: float = 1e-3,
) -> Optional[torch.optim.Optimizer]:
    """
    Create an AdamAndSGDG optimizer for SpinQuant parameters in a model.

    This is useful when AutoRound's own training loop wants to manage
    the optimizer externally rather than letting SpinQuant create it internally.

    Args:
        model: The model that has been set up with SpinQuant parameters
        lr: Learning rate for rotation matrices (SGDG)
        smooth_lr: Learning rate for smooth values (Adam)

    Returns:
        AdamAndSGDG optimizer or None if no SpinQuant parameters found
    """
    rotation_params = []
    smooth_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "spinquant_rotation" in name or "spinquant_R" in name:
            rotation_params.append(param)
        elif "spinquant_smooth" in name or "smooth_values" in name:
            smooth_params.append(param)

    if len(rotation_params) == 0 and len(smooth_params) == 0:
        return None

    return AdamAndSGDG(
        adam_params=smooth_params,
        sgdg_params=rotation_params,
        learning_rate=lr,
        smooth_learning_rate=smooth_lr,
    )


def spinquant_loss_fn(
    logits: torch.Tensor,
    original_logits: torch.Tensor,
    loss_type: str = "kl_top",
    kl_top_k: int = 1000,
) -> torch.Tensor:
    """
    Compute the SpinQuant training loss.

    Supports KL divergence (top-k or full) and MSE.

    Args:
        logits: Logits from the rotated model
        original_logits: Logits from the original (unrotated) model
        loss_type: ``"kl_top"`` | ``"kl_full"`` | ``"mse"``
        kl_top_k: Number of top logits for KL divergence

    Returns:
        Loss tensor
    """
    import torch.nn.functional as F

    if loss_type == "kl_top":
        k = min(kl_top_k, logits.size(-1))
        top_ori_logits, indices = original_logits.topk(k, dim=-1, sorted=False)
        top_logits = logits.gather(-1, indices)

        loss = F.kl_div(
            F.log_softmax(top_logits.flatten(0, -2), dim=-1),
            F.softmax(top_ori_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    elif loss_type == "kl_full":
        loss = F.kl_div(
            F.log_softmax(logits.flatten(0, -2), dim=-1),
            F.softmax(original_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    elif loss_type == "mse":
        loss = F.mse_loss(logits, original_logits)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Supported: kl_top, kl_full, mse")

    return loss


class SpinQuantState:
    """
    Simple state container for tracking SpinQuant training progress.

    Can be used by AutoRound's logging infrastructure to report
    SpinQuant-specific metrics.
    """

    def __init__(self):
        self.enabled = False
        self.iteration = 0
        self.max_iterations = 0
        self.loss_history: list[float] = []
        self.rotation_names: list[str] = []
        self.orthogonality_deviation: list[float] = []

    def update(self, loss: float, ortho_dev: float = 0.0) -> None:
        self.loss_history.append(loss)
        self.orthogonality_deviation.append(ortho_dev)
        self.iteration += 1

    @property
    def avg_loss(self) -> float:
        if not self.loss_history:
            return 0.0
        return sum(self.loss_history) / len(self.loss_history)

    @property
    def final_ortho_dev(self) -> float:
        return self.orthogonality_deviation[-1] if self.orthogonality_deviation else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "iterations": self.iteration,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "avg_loss": self.avg_loss,
            "rotation_params": self.rotation_names,
            "orthogonality_deviation": self.orthogonality_deviation[-1] if self.orthogonality_deviation else None,
        }
