# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
SpinQuant / QuaRot Trainer for AutoRound.

Provides a HuggingFace-Trainer-style interface for SpinQuant training,
while remaining dependency-light (does not require ``transformers.Trainer``).

The trainer manages the full training lifecycle:

    1. Setup – init rotations, smooth values, optimiser, original-model clone
    2. Train – customisable training loop with logging / evaluation
    3. Fuse – offline rotation fusion into weights
    4. Export – return the transformed model for AutoRound quantisation

Key differences from Quark's ``SpinQuantTrainer``:
    * No hard dependency on ``transformers.Trainer`` (works standalone).
    * Plug-and-play with AutoRound's dataloader format.
    * Supports mid-training evaluation (e.g. perplexity) via callbacks.
    * Can be used both as an independent trainer and as a component inside
      ``AutoRound.quantize()``.

Usage::

    from auto_round.algorithms.transforms.spinquant import (
        RotationTrainer, RotationTrainerConfig
    )

    trainer = RotationTrainer(
        model,
        config=RotationTrainerConfig(iters=200, lr=1e-4),
    )
    trainer.train(dataloader)
    model_rotated = trainer.fuse()

    # Proceed to AutoRound quantisation
    autoround = AutoRound(model_rotated, tokenizer, bits=4, ...)
    autoround.quantize()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    register_spinquant_hooks,
)
from auto_round.algorithms.transforms.spinquant.preprocessor import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
    get_model_arch_info,
)
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    fuse_rmsnorm_in_model,
    untie_word_embeddings_if_needed,
)


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

    def __init__(self, threshold: float = 1e-3, log_interval: int = 50):
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
            print(f"  [OrthogonalityMonitor] step={step}  dev={max_dev:.2e}  ⚠️ > {self.threshold}")


class LossLogger(RotationTrainerCallback):
    """Callback that prints loss every ``log_interval`` steps."""

    def __init__(self, log_interval: int = 50):
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
    ):
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
        from auto_round.algorithms.transforms.spinquant.training_core import (
            compute_rotation_loss,
            move_batch_to_device,
        )

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
            "config": self.config,
            "rotation_params": {
                n: p.data.cpu() for n, p in self.model.named_parameters() if p.requires_grad and "spinquant_R" in n
            },
            "smooth_params": {
                n: p.data.cpu() for n, p in self.model.named_parameters() if p.requires_grad and "smooth_values" in n
            },
        }
        torch.save(ckpt, path)
        print(f"[RotationTrainer] Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Restore rotation + smooth params from disk."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for n, p in self.model.named_parameters():
            if n in ckpt["rotation_params"]:
                p.data.copy_(ckpt["rotation_params"][n].to(p.device))
            if n in ckpt["smooth_params"]:
                p.data.copy_(ckpt["smooth_params"][n].to(p.device))
        self.state["step"] = ckpt["step"]
        print(f"[RotationTrainer] Checkpoint loaded: {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_training(self, dataloader: Any) -> None:
        """Initialise rotations, optimiser, and original-model clone."""
        from auto_round.algorithms.transforms.spinquant.training_core import (
            clone_model_for_reference,
            create_dual_optimizer,
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

        # 5. Create dual optimiser (shared helper)
        self.optimizer = create_dual_optimizer(
            self.model,
            lr=cfg.lr,
            smooth_lr=cfg.smooth_lr,
        )

        # 6. Clone original model for KL reference (shared helper)
        self._original_model = clone_model_for_reference(self.model)

        # 7. Use model's existing device (don't move)
        self.model.train()

    def _training_step(self, batch: Any) -> float:
        """Single training step. Returns loss scalar."""
        from auto_round.algorithms.transforms.spinquant.training_core import (
            compute_rotation_loss,
            move_batch_to_device,
        )

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

        # Loss & backward — use custom loss_fn if set, else shared core
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
        """Default loss — delegates to :func:`training_core.compute_rotation_loss`."""
        from auto_round.algorithms.transforms.spinquant.training_core import compute_rotation_loss

        return compute_rotation_loss(logits, ori_logits, config.loss_type, config.kl_top_k)

    @staticmethod
    def _to_sq_config(cfg: RotationTrainerConfig) -> SpinQuantConfig:
        """Convert trainer config to SpinQuantConfig for internal reuse."""
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
