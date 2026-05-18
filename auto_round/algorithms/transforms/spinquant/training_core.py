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
"""Shared training primitives for SpinQuant rotation learning.

This module provides the **single implementation** of the rotation training
loop used by both:

* :meth:`SpinQuantPreprocessor._train_rotations` — minimal embedded loop
* :class:`RotationTrainer` — full-featured trainer with callbacks

By centralising the training step and loss computation here, both entry
points stay in sync and avoid code duplication.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.algorithms.transforms.spinquant.cayley_optimizer import AdamAndSGDG

logger = logging.getLogger("autoround.spinquant")


# ---------------------------------------------------------------------------
# Shared loss computation
# ---------------------------------------------------------------------------


def compute_rotation_loss(
    logits: torch.Tensor,
    ori_logits: torch.Tensor,
    loss_type: str = "kl_top",
    kl_top_k: int = 1000,
) -> torch.Tensor:
    """Compute loss between rotated-model logits and original-model logits.

    This is the single implementation shared by both the preprocessor's
    embedded loop and the standalone :class:`RotationTrainer`.

    Args:
        logits: Rotated model output logits.
        ori_logits: Original (un-rotated) model output logits (detached).
        loss_type: ``"kl_top"`` | ``"kl_full"`` | ``"mse"``.
        kl_top_k: Number of top logits to use for ``kl_top``.

    Returns:
        Scalar loss tensor.
    """
    if loss_type == "kl_top":
        k = min(kl_top_k, logits.size(-1))
        top_ori, indices = ori_logits.topk(k, dim=-1, sorted=False)
        top_logits = logits.gather(-1, indices)
        return F.kl_div(
            F.log_softmax(top_logits.flatten(0, -2), dim=-1),
            F.softmax(top_ori.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if loss_type == "kl_full":
        return F.kl_div(
            F.log_softmax(logits.flatten(0, -2), dim=-1),
            F.softmax(ori_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if loss_type == "mse":
        return F.mse_loss(logits, ori_logits)
    raise ValueError(f"Unknown loss_type={loss_type!r}")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move a batch (tensor or dict of tensors) to *device*."""
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    return batch


def check_orthogonality(model: nn.Module, threshold: float = 1e-3) -> float:
    """Check orthogonality of all ``spinquant_R*`` parameters.

    Returns:
        Maximum deviation from identity (``max ||R @ R^T - I||_inf``).
    """
    max_dev = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or "spinquant_R" not in name:
            continue
        R = param.data
        if R.dim() != 2 or R.shape[0] != R.shape[1] or R.numel() == 0:
            continue
        I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
        dev = (torch.matmul(R, R.t()) - I).abs().max().item()
        max_dev = max(max_dev, dev)
        if dev > 1e-4:
            logger.warning(f"  {name} orthogonality deviation={dev:.2e}")
    if max_dev > 0:
        logger.info(f"[SpinQuant] Max orthogonality deviation={max_dev:.2e}")
        if max_dev > threshold:
            logger.warning("  Orthogonality constraint significantly violated!")
    return max_dev


def clone_model_for_reference(model: nn.Module) -> nn.Module:
    """Create a frozen deep-copy of *model* for KL-divergence reference.

    Hooks registered by SpinQuant are removed from the clone so it runs
    as the un-rotated baseline.
    """
    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        remove_spinquant_hooks_from_model,
    )

    original = copy.deepcopy(model)
    original.eval()
    for p in original.parameters():
        p.requires_grad = False
    remove_spinquant_hooks_from_model(original)
    return original


def create_dual_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    smooth_lr: float = 1e-3,
) -> Optional[AdamAndSGDG]:
    """Create the Adam (smooth) + SGDG (rotation) dual optimiser.

    Returns ``None`` if no trainable parameters are found.
    """
    rot_params = [p for n, p in model.named_parameters() if p.requires_grad and "spinquant_R" in n]
    smooth_params = [p for n, p in model.named_parameters() if p.requires_grad and "smooth_values" in n]

    if not rot_params and not smooth_params:
        logger.info("[SpinQuant] No trainable parameters — nothing to train.")
        return None

    n_rot = len(rot_params)
    n_smooth = len(smooth_params)
    total_params = sum(p.numel() for p in rot_params) + sum(p.numel() for p in smooth_params)
    logger.info(
        f"[SpinQuant] Training: {n_rot} rotation params + {n_smooth} smooth params "
        f"= {total_params:,} trainable parameters"
    )

    return AdamAndSGDG(
        adam_params=smooth_params,
        sgdg_params=rot_params,
        learning_rate=lr,
        smooth_learning_rate=smooth_lr,
    )


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Result of :func:`run_training_loop`."""

    loss_history: list[float]
    best_loss: float
    final_ortho_deviation: float
    steps: int


def run_training_loop(
    model: nn.Module,
    original_model: nn.Module,
    optimizer: AdamAndSGDG,
    dataloader: Any,
    *,
    max_iters: int = 200,
    loss_type: str = "kl_top",
    kl_top_k: int = 1000,
    compute_loss_fn: Optional[Callable] = None,
    on_step_end: Optional[Callable[[int, float, float], None]] = None,
    log_interval: int = 50,
) -> TrainingResult:
    """Run the SpinQuant rotation training loop.

    This is the **single** training loop implementation shared by
    :class:`SpinQuantPreprocessor` (embedded) and :class:`RotationTrainer`
    (standalone).

    Args:
        model: The model being trained (rotation params + optional smooth params).
        original_model: Frozen reference model for KL divergence.
        optimizer: :class:`AdamAndSGDG` dual optimiser.
        dataloader: Iterable of batches.
        max_iters: Stop after this many steps.
        loss_type: ``"kl_top"`` | ``"kl_full"`` | ``"mse"``.
        kl_top_k: Top-k for ``kl_top`` loss.
        compute_loss_fn: Optional override for loss computation. Signature:
            ``(logits, ori_logits) -> Tensor``.  If ``None``, uses
            :func:`compute_rotation_loss`.
        on_step_end: Optional callback ``(step, loss, avg_loss) -> None``
            called after each gradient step.
        log_interval: Default logging interval (used when no ``on_step_end``
            is provided).

    Returns:
        :class:`TrainingResult` with loss history and orthogonality stats.
    """
    device = next(model.parameters()).device
    model.train()

    loss_fn = compute_loss_fn or (
        lambda logits, ori_logits: compute_rotation_loss(logits, ori_logits, loss_type, kl_top_k)
    )

    loss_history: list[float] = []
    best_loss = float("inf")
    step = 0

    for batch in dataloader:
        if step >= max_iters:
            break

        batch = move_batch_to_device(batch, device)

        # Forward: rotated model
        out_rot = model(**batch)
        logits = out_rot.logits if hasattr(out_rot, "logits") else out_rot

        # Forward: original model (no grad)
        with torch.no_grad():
            out_ori = original_model(**batch)
            ori_logits = out_ori.logits if hasattr(out_ori, "logits") else out_ori

        # Loss + backward
        loss = loss_fn(logits, ori_logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        loss_history.append(loss_val)
        best_loss = min(best_loss, loss_val)

        # Logging / callbacks
        if on_step_end is not None:
            avg = sum(loss_history[-50:]) / len(loss_history[-50:])
            on_step_end(step, loss_val, avg)
        elif step % log_interval == 0:
            avg = sum(loss_history[-50:]) / len(loss_history[-50:])
            logger.info(f"[SpinQuant] Step {step}/{max_iters}, loss={loss_val:.6f} (avg={avg:.6f})")

        step += 1

    # Final orthogonality check
    ortho_dev = check_orthogonality(model)

    model.eval()

    return TrainingResult(
        loss_history=loss_history,
        best_loss=best_loss,
        final_ortho_deviation=ortho_dev,
        steps=step,
    )
