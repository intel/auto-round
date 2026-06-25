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
"""SpinQuant / QuaRot training infrastructure.

⚠️  **Experimental**: The SpinQuant training path has basic infrastructure
(Cayley SGD optimizer, KL-divergence loss, callbacks, checkpointing) but
has NOT been validated end-to-end on real models. For production use, prefer
QuaRot mode (fixed Hadamard, no training) via ``SpinQuantPreprocessor``.

This module consolidates all training-related functionality:

* **Core primitives** — loss computation, training loop, optimizer creation
* **Pipeline hooks** — ``SpinQuantTrainingHook`` for AutoRound integration
* **Standalone trainer** — ``RotationTrainer`` (HF Trainer style)
* **Callbacks** — orthogonality monitoring, loss logging
* **State tracking** — ``SpinQuantState`` for metrics

Usage (standalone trainer)::

    from auto_round.algorithms.transforms.spinquant.training import (
        RotationTrainer, RotationTrainerConfig
    )

    trainer = RotationTrainer(model, config=RotationTrainerConfig(iters=200))
    trainer.train(dataloader)
    model = trainer.fuse()

Usage (pipeline hook)::

    hook = SpinQuantTrainingHook(model, config)
    hook.preprocess(dataloader)
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.algorithms.transforms.spinquant.cayley_optimizer import AdamAndSGDG

logger = logging.getLogger("autoround.spinquant")


# ===========================================================================
# Core primitives: loss, utilities, training loop
# ===========================================================================


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


# Alias for backward compatibility (training.py used to export this)
spinquant_loss_fn = compute_rotation_loss


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


# Alias for backward compatibility
create_spinquant_optimizer = create_dual_optimizer


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


# ===========================================================================
# Pipeline hooks: SpinQuantTrainingHook for AutoRound integration
# ===========================================================================


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
        config: Any = None,
        enabled: bool = True,
    ) -> None:
        from auto_round.algorithms.transforms.spinquant.preprocessor import (
            SpinQuantConfig,
            SpinQuantPreprocessor,
        )

        self.model = model
        self.config = config or SpinQuantConfig()
        self.enabled = enabled
        self.preprocessor: Optional[SpinQuantPreprocessor] = None

    def preprocess(self, dataloader: Any) -> nn.Module:
        """Execute SpinQuant preprocessing."""
        if not self.enabled:
            return self.model

        from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantPreprocessor

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

    def __init__(self, model: nn.Module, log_interval: int = 50) -> None:
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
            logger.info(f"[SpinQuant] Step {self.step}: max orthogonality deviation = {max_dev:.2e}")

        if max_dev > 1e-2:
            logger.warning("[SpinQuant] Orthogonality significantly violated! Consider reducing lr.")


class SpinQuantState:
    """
    Simple state container for tracking SpinQuant training progress.

    Can be used by AutoRound's logging infrastructure to report
    SpinQuant-specific metrics.
    """

    def __init__(self) -> None:
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


# ===========================================================================
# Standalone trainer: RotationTrainer (HF Trainer style)
# ===========================================================================


@dataclass
class RotationTrainerConfig:
    """Training hyperparameters for ``RotationTrainer``.

    ⚠️  **Experimental**: The SpinQuant training loop has basic infrastructure
    (Cayley SGD, KL loss, callbacks, checkpointing) but has NOT been validated
    end-to-end on real models. For production use, prefer QuaRot mode via
    ``SpinQuantPreprocessor`` with ``trainable_rotation=False``.
    """

    # ----------  Rotation knobs  ----------
    r1: bool = True
    r2: bool = True
    r3: bool = True
    r4: bool = True

    trainable_rotation: bool = True  # False = fixed Hadamard (QuaRot)
    trainable_smooth: bool = True  # joint SmoothQuant training
    online_r1_rotation: bool = False

    # ----------  Optimiser  ----------
    lr: float = 1e-4  # SGDG lr  (rotation matrices)
    smooth_lr: float = 1e-3  # Adam lr  (smooth values)
    iters: int = 200
    batch_size: int = 1

    # ----------  Loss  ----------
    loss_type: str = "kl_top"  # kl_top | kl_full | mse
    kl_top_k: int = 1000

    # ----------  Pipeline  ----------
    fuse_rmsnorm: bool = True
    untie_embeddings: bool = True

    # ----------  Misc  ----------
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None
    log_interval: int = 50  # print every N steps
    eval_interval: int = 0  # 0 = never
    save_interval: int = 0  # 0 = never
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class RotationTrainerCallback:
    """
    Minimal callback interface (inspired by HF ``TrainerCallback``).

    Subclass and override any of the following hooks:

    * ``on_train_begin``
    * ``on_train_end``
    * ``on_step_begin``
    * ``on_step_end``
    * ``on_evaluate``
    """

    def on_train_begin(self, args: RotationTrainerConfig, state: dict, control: dict):
        pass

    def on_train_end(self, args: RotationTrainerConfig, state: dict, control: dict):
        pass

    def on_step_begin(self, args: RotationTrainerConfig, state: dict, control: dict):
        pass

    def on_step_end(self, args: RotationTrainerConfig, state: dict, control: dict):
        pass

    def on_evaluate(self, args: RotationTrainerConfig, state: dict, metrics: dict):
        pass


class OrthogonalityMonitor(RotationTrainerCallback):
    """
    Callback that monitors ``R @ R.T ≈ I`` during training.

    Prints a warning when the deviation exceeds a threshold.
    """

    def __init__(self, threshold: float = 1e-3, log_interval: int = 50) -> None:
        self.threshold = threshold
        self.log_interval = log_interval

    def on_step_end(self, args, state, control):
        step = state.get("step", 0)
        if step % self.log_interval != 0:
            return

        model = state.get("model")
        if model is None:
            return

        max_dev = 0.0
        for name, param in model.named_parameters():
            if not param.requires_grad or "spinquant_R" not in name:
                continue
            R = param.data
            if R.dim() != 2 or R.shape[0] != R.shape[1]:
                continue
            I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
            dev = (torch.matmul(R, R.t()) - I).abs().max().item()
            max_dev = max(max_dev, dev)

        state["ortho_deviation"] = max_dev
        if max_dev > self.threshold:
            print(f"  [OrthogonalityMonitor] step={step}  dev={max_dev:.2e}  > {self.threshold}")


class LossLogger(RotationTrainerCallback):
    """Callback that prints loss every ``log_interval`` steps."""

    def __init__(self, log_interval: int = 50) -> None:
        self.log_interval = log_interval

    def on_step_end(self, args, state, control):
        step = state.get("step", 0)
        if step % self.log_interval == 0:
            loss = state.get("loss", 0.0)
            avg_loss = state.get("avg_loss", 0.0)
            print(f"  [LossLogger] step={step}/{args.iters}  loss={loss:.6f}  avg={avg_loss:.6f}")


class RotationTrainer:
    """
    Trainer for SpinQuant / QuaRot orthogonal rotation matrices.

    ⚠️  **Experimental**: This trainer has basic infrastructure but the
    SpinQuant training path has NOT been validated end-to-end on real models.
    For production QuaRot usage (fixed Hadamard, no training), use
    ``SpinQuantPreprocessor`` directly instead.

    Similar to Quark's ``SpinQuantTrainer`` but decoupled from
    ``transformers.Trainer``, making it usable inside AutoRound without
    extra heavy dependencies.

    Lifecycle::

        trainer = RotationTrainer(model, config, callbacks=[...])
        trainer.train(dataloader)        # training loop
        model = trainer.fuse()           # fuse into weights
        # ... now feed model to AutoRound ...

    Arguments:
        model: The transformer model (modified **in place**).
        config: ``RotationTrainerConfig`` instance.
        callbacks: List of ``RotationTrainerCallback`` subclasses.
        compute_loss_fn: Optional custom loss callable. Signature::

            compute_loss_fn(logits, ori_logits, config) -> Tensor
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[RotationTrainerConfig] = None,
        callbacks: Optional[list[RotationTrainerCallback]] = None,
        compute_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, RotationTrainerConfig], torch.Tensor]] = None,
    ) -> None:
        from auto_round.algorithms.transforms.spinquant.preprocessor import (
            SpinQuantConfig,
            SpinQuantPreprocessor,
            get_model_arch_info,
        )

        self.model = model
        self.config = config or RotationTrainerConfig()
        self.callbacks = callbacks or [LossLogger(), OrthogonalityMonitor()]
        self.compute_loss_fn = compute_loss_fn or self._default_compute_loss

        # Internal state (shared with callbacks)
        self.state: dict[str, Any] = {
            "step": 0,
            "global_step": 0,
            "loss": 0.0,
            "avg_loss": 0.0,
            "best_loss": float("inf"),
            "ortho_deviation": 0.0,
            "model": model,
        }

        # Architecture info
        info = get_model_arch_info(model)
        self.hidden_size = info.get("hidden_size", 0)
        self.head_dim = info.get("head_dim", 0)
        self.num_q_heads = info.get("num_q_heads", 0)
        self.num_kv_heads = info.get("num_kv_heads", 0)
        self.intermediate_size = info.get("intermediate_size", 0)

        # Training components (created lazily)
        self.optimizer = None
        self._original_model: Optional[nn.Module] = None
        self._hook_handles: list[Any] = []
        self._rotated_modules: set[nn.Module] = set()
        self._loss_buffer: list[float] = []

        # SpinQuant preprocessor (reused for init / fusion)
        self._preprocessor = SpinQuantPreprocessor(model, self._to_sq_config(self.config))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, dataloader: Any) -> dict[str, Any]:
        """
        Run the full SpinQuant training loop.

        Returns a dict of training metrics::

            {
                "loss_history": list[float],
                "best_loss": float,
                "final_ortho_deviation": float,
                "steps": int,
                "elapsed_sec": float,
            }
        """
        self._setup_training(dataloader)
        self._trigger_event("on_train_begin")

        t0 = time.time()
        for batch in dataloader:
            if self.state["step"] >= self.config.iters:
                break

            self._trigger_event("on_step_begin")
            loss = self._training_step(batch)
            self._loss_buffer.append(loss)
            self.state["loss"] = loss
            self.state["avg_loss"] = sum(self._loss_buffer[-50:]) / len(self._loss_buffer[-50:])
            self.state["best_loss"] = min(self.state["best_loss"], loss)
            self.state["step"] += 1
            self.state["global_step"] += 1

            # Mid-training evaluation
            if self.config.eval_interval > 0 and self.state["step"] % self.config.eval_interval == 0:
                metrics = self.evaluate(dataloader)
                self._trigger_event("on_evaluate", metrics=metrics)

            # Checkpointing
            if self.config.save_interval > 0 and self.state["step"] % self.config.save_interval == 0:
                self.save_checkpoint()

            self._trigger_event("on_step_end")

        elapsed = time.time() - t0
        self._trigger_event("on_train_end")
        self._teardown_training()

        return {
            "loss_history": self._loss_buffer,
            "best_loss": self.state["best_loss"],
            "final_ortho_deviation": self.state.get("ortho_deviation", 0.0),
            "steps": self.state["step"],
            "elapsed_sec": elapsed,
        }

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        """
        Evaluate current rotated model against the original on a subset.

        Returns a dict with ``loss``, ``max_diff``, etc.
        """
        if self._original_model is None:
            return {}

        self.model.eval()
        device = next(self.model.parameters()).device
        losses = []
        max_diffs = []

        for i, batch in enumerate(dataloader):
            if i >= 10:  # evaluate on first 10 batches only
                break
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                out_rot = self.model(**batch)
                logits_rot = out_rot.logits if hasattr(out_rot, "logits") else out_rot
                out_ori = self._original_model(**batch)
                logits_ori = out_ori.logits if hasattr(out_ori, "logits") else out_ori

                loss = compute_rotation_loss(
                    logits_rot,
                    logits_ori,
                    loss_type=self.config.loss_type,
                    kl_top_k=self.config.kl_top_k,
                )
                losses.append(loss.item())
                max_diffs.append((logits_rot - logits_ori).abs().max().item())

        self.model.train()
        if not losses:
            return {"eval_loss": 0.0, "eval_max_diff": 0.0}
        return {
            "eval_loss": sum(losses) / len(losses),
            "eval_max_diff": sum(max_diffs) / len(max_diffs),
        }

    def fuse(self) -> nn.Module:
        """
        Fuse offline rotations into model weights and clean up.

        Returns the modified ``model`` (same object, in-place).
        """
        self._preprocessor._fuse_offline_rotations()
        self._preprocessor._cleanup()
        return self.model

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save rotation + smooth params to disk."""
        if path is None:
            path = f"{self.config.checkpoint_dir or '.'}/spinquant_ckpt_step{self.state['step']}.pt"
        ckpt = {
            "step": self.state["step"],
            "config": asdict(self.config),
            "rotation_params": {
                n: p.data.cpu() for n, p in self.model.named_parameters() if p.requires_grad and "spinquant_R" in n
            },
            "smooth_params": {
                n: p.data.cpu() for n, p in self.model.named_parameters() if p.requires_grad and "smooth_values" in n
            },
        }
        torch.save(ckpt, path)
        logger.info(f"[RotationTrainer] Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Restore rotation + smooth params from disk."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        for n, p in self.model.named_parameters():
            if n in ckpt["rotation_params"]:
                p.data.copy_(ckpt["rotation_params"][n].to(p.device))
            if n in ckpt["smooth_params"]:
                p.data.copy_(ckpt["smooth_params"][n].to(p.device))
        self.state["step"] = ckpt["step"]
        logger.info(f"[RotationTrainer] Checkpoint loaded: {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_training(self, dataloader: Any) -> None:
        """Initialise rotations, optimiser, and original-model clone."""
        from auto_round.algorithms.transforms.spinquant.inplace.apply import (
            register_spinquant_hooks,
        )
        from auto_round.algorithms.transforms.spinquant.rotation_utils import (
            fuse_rmsnorm_in_model,
            untie_word_embeddings_if_needed,
        )

        cfg = self.config

        # 1. Untie + RMSNorm fusion
        if cfg.untie_embeddings:
            untie_word_embeddings_if_needed(self.model)
        if cfg.fuse_rmsnorm:
            fuse_rmsnorm_in_model(self.model)

        # 2. Replace norms with trainable wrappers (SmoothQuant)
        if cfg.trainable_smooth:
            self._preprocessor._replace_norms_with_trainable()

        # 3. Initialise rotation matrices
        self._preprocessor._init_rotation_matrices()

        # 4. Register online hooks (R3 / R4)
        if cfg.r3 or cfg.r4:
            self._hook_handles = register_spinquant_hooks(
                self.model,
                self._to_sq_config(cfg),
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
            )

        # 5. Create dual optimiser
        self.optimizer = create_dual_optimizer(
            self.model,
            lr=cfg.lr,
            smooth_lr=cfg.smooth_lr,
        )
        if self.optimizer is None:
            raise ValueError(
                "SpinQuant training requires at least one trainable parameter group, "
                "but create_dual_optimizer() returned None. Ensure that training is "
                "configured with trainable rotation and/or smooth parameters enabled."
            )

        # 6. Clone original model for KL reference
        self._original_model = clone_model_for_reference(self.model)

        # 7. Use model's existing device (don't move)
        self.model.train()

    def _training_step(self, batch: Any) -> float:
        """Single training step. Returns loss scalar."""
        device = next(self.model.parameters()).device
        batch = move_batch_to_device(batch, device)

        # Forward: rotated model
        out_rot = self.model(**batch)
        logits_rot = out_rot.logits if hasattr(out_rot, "logits") else out_rot

        # Forward: original model (no grad)
        with torch.no_grad():
            assert self._original_model is not None
            out_ori = self._original_model(**batch)
            logits_ori = out_ori.logits if hasattr(out_ori, "logits") else out_ori

        # Loss & backward
        if self.compute_loss_fn is not self._default_compute_loss:
            loss = self.compute_loss_fn(logits_rot, logits_ori, self.config)
        else:
            loss = compute_rotation_loss(
                logits_rot,
                logits_ori,
                loss_type=self.config.loss_type,
                kl_top_k=self.config.kl_top_k,
            )
        loss.backward()
        self.optimizer.step()  # type: ignore[union-attr]
        self.optimizer.zero_grad()  # type: ignore[union-attr]

        return loss.item()

    def _teardown_training(self) -> None:
        """Free original model and set eval mode."""
        if self._original_model is not None:
            del self._original_model
            self._original_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model.eval()

    def _trigger_event(self, event_name: str, **kwargs) -> None:
        """Fire callback hooks."""
        for cb in self.callbacks:
            method = getattr(cb, event_name, None)
            if callable(method):
                method(self.config, self.state, {"should_training_stop": False, **kwargs})

    @staticmethod
    def _default_compute_loss(
        logits: torch.Tensor,
        ori_logits: torch.Tensor,
        config: RotationTrainerConfig,
    ) -> torch.Tensor:
        """Default loss — delegates to :func:`compute_rotation_loss`."""
        return compute_rotation_loss(logits, ori_logits, config.loss_type, config.kl_top_k)

    @staticmethod
    def _to_sq_config(cfg: RotationTrainerConfig):
        """Convert trainer config to SpinQuantConfig for internal reuse."""
        from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

        return SpinQuantConfig(
            r1=cfg.r1,
            r2=cfg.r2,
            r3=cfg.r3,
            r4=cfg.r4,
            trainable_rotation=cfg.trainable_rotation,
            trainable_smooth=cfg.trainable_smooth,
            online_r1_rotation=cfg.online_r1_rotation,
            iters=cfg.iters,
            lr=cfg.lr,
            smooth_lr=cfg.smooth_lr,
            batch_size=cfg.batch_size,
            loss_type=cfg.loss_type,
            kl_top_k=cfg.kl_top_k,
            fuse_rmsnorm=cfg.fuse_rmsnorm,
            untie_embeddings=cfg.untie_embeddings,
            dtype=cfg.dtype,
            device=cfg.device,
        )
