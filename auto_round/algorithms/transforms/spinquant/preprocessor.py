# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
SpinQuant preprocessor for Intel AutoRound.

Main class that orchestrates the SpinQuant / QuaRot rotation pipeline:

    1. Fuse RMSNorm parameters into linear layers
    2. Replace RMSNorm with TrainableRMSNorm (if ``trainable_smooth``)
    3. Initialise rotation matrices (R1, R2, R3, R4)
    4. (If trainable) train rotations & smooth values via KL divergence
    5. Fuse offline rotations (R1, R2) into model weights
    6. Register online hooks (R3, R4) and clean up

The module delegates in-place hook registration to
``spinquant.inplace.apply`` so that the online-rotation logic follows the
same pattern as AutoRound's ``rotation.inplace`` package.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.base import BaseRotationConfig

# ---------------------------------------------------------------------------
# Delegate online-hook registration to the inplace sub-package.
# ---------------------------------------------------------------------------
from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    register_spinquant_hooks,
    remove_spinquant_hooks,
)
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    apply_hadamard_to_linear,
    create_block_diag_from_head_matrix,
    deterministic_hadamard_matrix,
    fuse_rmsnorm_in_model,
    get_hadamard_K,
    get_model_arch_info,
    is_pow2,
    matmul_hadU,
    random_hadamard_matrix,
    rotate_in_channels_,
    rotate_out_channels_,
    untie_word_embeddings_if_needed,
)

logger = logging.getLogger("autoround.spinquant")


# NOTE: Online R1 uses forward_pre_hook on target modules. Hooks are
# compatible with auto-round's WrapperLinear (which runs orig_layer hooks)
# and WrapperWALayer (which steals and runs them at inference).


@dataclass
class SpinQuantConfig(BaseRotationConfig):
    """Configuration for SpinQuant / QuaRot preprocessing.

    Inherits from :class:`BaseRotationConfig` so that instances can be
    collected by :class:`BaseCompressor` alongside Hadamard configs and
    dispatched through the unified ``apply_rotation()`` / ``BaseRotation``
    registry.

    Feature Status:
        ✅ QuaRot mode (``trainable_rotation=False``): Fully supported.
           Fixed Hadamard rotation (R1–R4), no training needed, no calibration data.
        ⚠️  SpinQuant mode (``trainable_rotation=True``): Experimental.
           Training loop exists but not fully validated on real models.
        ✅  Model save/load: fully implemented with rebuilt hooks.
    """

    #: Registry key — used by ``BaseRotation.from_config()`` to dispatch.
    algorithm: str = "spinquant"

    # Rotation dimensions
    r1: bool = True  # R1: hidden_size rotation (offline fused)
    r2: bool = True  # R2: head_dim rotation (offline fused)
    r3: bool = False  # R3: Q/K online rotation
    r4: bool = False  # R4: MLP activation online rotation

    # Rotation size override (None = use full dimension from model config)
    # When set, R1 uses rotation_size instead of hidden_size,
    # and R4 uses rotation_size instead of intermediate_size.
    # R2 always uses head_dim, R3 does not support custom size.
    # This follows the same convention as Quark's rotation_size.
    rotation_size: Optional[int] = None

    # Rotation matrix type for R1–R4
    # - False (default): deterministic Hadamard (same matrix every time, no need to persist)
    # - True: random Hadamard = H × D where D = diag(±1) random (must persist matrix)
    # Only relevant when trainable_rotation=False (QuaRot mode).
    # When trainable_rotation=True (SpinQuant mode), R1/R2 init from identity regardless.
    # For online rotations (R1/R3/R4), random uses explicit x @ R instead of
    # butterfly algorithm — slightly slower but functionally equivalent.
    random_r1: bool = False
    random_r2: bool = False
    random_r3: bool = False
    random_r4: bool = False

    # Training control
    # ⚠️ trainable_rotation=True (SpinQuant mode) is experimental — training
    #    loop exists but not validated end-to-end. Use trainable_rotation=False
    #    (QuaRot mode) for production use.
    trainable_rotation: bool = True  # Learn R via Cayley SGD (False = QuaRot fixed Hadamard)
    trainable_smooth: bool = True  # Learn smooth_values via Adam (joint SmoothQuant)
    online_r1_rotation: bool = True  # Online R1: rotate target weights + hook (Quark default)

    # Training hyperparameters
    iters: int = 200  # Training iterations
    lr: float = 1e-4  # SGDG learning rate (rotation matrices)
    smooth_lr: float = 1e-3  # Adam learning rate (smooth values)
    batch_size: int = 1

    # Loss
    loss_type: str = "kl_top"  # "kl_top" | "kl_full" | "mse"
    kl_top_k: int = 1000

    # Pipeline steps
    fuse_rmsnorm: bool = True
    untie_embeddings: bool = True

    # Numerics
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.rotation_size is not None:
            if self.rotation_size <= 0:
                raise ValueError(f"rotation_size must be positive, got {self.rotation_size}")
            if not is_pow2(self.rotation_size):
                raise ValueError(
                    f"rotation_size must be a power of 2, got {self.rotation_size}. "
                    f"Valid values: 16, 32, 64, 128, 256, 512, 1024, ..."
                )


class TrainableRMSNorm(nn.Module):
    """
    RMSNorm wrapper with trainable ``smooth_values`` for joint
    SpinQuant + SmoothQuant.

    Original RMSNorm::

        output = x / RMS(x) * gamma

    Trainable version::

        output = x / RMS(x) * gamma * smooth_values

    The ``smooth_values`` (diagonal scaling ``D``) are learned jointly
    with rotation matrices to minimise quantisation error.
    """

    def __init__(self, original_norm: nn.Module, trainable: bool = True):
        super().__init__()
        self.original_norm = original_norm
        self.trainable = trainable

        if hasattr(original_norm, "weight"):
            shape = original_norm.weight.shape
            self.smooth_values = nn.Parameter(
                torch.ones(shape, device=original_norm.weight.device, dtype=original_norm.weight.dtype),
                requires_grad=trainable,
            )
        else:
            self.smooth_values = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.original_norm(hidden_states)
        if self.smooth_values is not None:
            out = out * self.smooth_values
        return out


class SpinQuantPreprocessor:
    """
    SpinQuant preprocessor following AutoRound's transform conventions.

    After preprocessing the model is mathematically equivalent to the
    original but with weight distributions better suited for quantisation.
    """

    def __init__(self, model: nn.Module, config: Optional[SpinQuantConfig] = None):
        self.model = model
        self.config = config or SpinQuantConfig()

        # Architecture metadata
        info = get_model_arch_info(model)
        self.hidden_size = info.get("hidden_size", 0)
        self.head_dim = info.get("head_dim", 0)
        self.num_q_heads = info.get("num_q_heads", 0)
        self.num_kv_heads = info.get("num_kv_heads", 0)
        self.intermediate_size = info.get("intermediate_size", 0)

        # Resolve effective rotation sizes based on config.rotation_size
        # R1: rotation_size or hidden_size
        self.r1_rotation_size = self.config.rotation_size or self.hidden_size
        # R2: always head_dim (not affected by rotation_size)
        self.r2_rotation_size = self.head_dim
        # R3: always head_dim (not affected by rotation_size, same as Quark)
        self.r3_rotation_size = self.head_dim
        # R4: rotation_size or intermediate_size
        self.r4_rotation_size = self.config.rotation_size or self.intermediate_size

        # Training state
        self.rotation_params: list[nn.Parameter] = []
        self.smooth_params: list[nn.Parameter] = []

        # Fusion deduplication
        self._rotated_modules: set[nn.Module] = set()

        # Hook handles (managed by inplace sub-package)
        self._hook_handles: list[Any] = []
        # R1 online hook handles (managed by _apply_online_r1)
        self._r1_hook_handles: list[Any] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def preprocess(self, dataloader: Optional[Any] = None) -> nn.Module:
        logger.info("[SpinQuant] Starting preprocessing...")
        logger.info(
            f"[SpinQuant] Model architecture info: hidden_size={self.hidden_size}, "
            f"head_dim={self.head_dim}, num_q_heads={self.num_q_heads}, "
            f"num_kv_heads={self.num_kv_heads}, intermediate_size={self.intermediate_size}"
        )
        if self.config.rotation_size is not None:
            logger.info(
                f"[SpinQuant] Custom rotation_size={self.config.rotation_size} → "
                f"R1 size={self.r1_rotation_size}, R4 size={self.r4_rotation_size} "
                f"(R2/R3 always use head_dim={self.head_dim})"
            )
        logger.info(
            f"[SpinQuant] Rotation config: R1={self.config.r1}, R2={self.config.r2}, "
            f"R3={self.config.r3}, R4={self.config.r4}, "
            f"online_r1={self.config.online_r1_rotation}, "
            f"trainable_rotation={self.config.trainable_rotation}, "
            f"trainable_smooth={self.config.trainable_smooth}"
        )

        # Validate dimensions for enabled rotations
        self._validate_dimensions()

        # Step 1: untie embeddings (only needed for offline R1 — it rotates embed_tokens)
        if self.config.untie_embeddings and not self.config.online_r1_rotation:
            if untie_word_embeddings_if_needed(self.model):
                logger.info("[SpinQuant] Untied input/output embeddings")
        elif self.config.online_r1_rotation:
            logger.info("[SpinQuant] Online R1: skipping untie embeddings (embed_tokens unchanged)")

        # Step 2: fuse RMSNorm gamma into linear weights
        # Online R1 does NOT fuse RMSNorm (matching Quark's behavior):
        # the rotation is local per-module, so gamma doesn't need to commute.
        if self.config.fuse_rmsnorm and not self.config.online_r1_rotation:
            logger.info("[SpinQuant] Fusing RMSNorm parameters into linear weights...")
            fuse_rmsnorm_in_model(self.model)
        elif self.config.online_r1_rotation:
            logger.info("[SpinQuant] Online R1: skipping RMSNorm fusion (not needed)")

        # Step 3: replace RMSNorm with TrainableRMSNorm (SmoothQuant)
        if self.config.trainable_smooth:
            logger.info("[SpinQuant] Adding trainable smooth values...")
            self._replace_norms_with_trainable()

        # Step 4: initialise rotation matrices
        logger.info("[SpinQuant] Initialising rotation matrices...")
        self._init_rotation_matrices()

        # Step 5: train if requested
        if self.config.trainable_rotation or self.config.trainable_smooth:
            if dataloader is None:
                raise ValueError("dataloader required when trainable=True")
            logger.info(f"[SpinQuant] Training for {self.config.iters} iterations...")
            self._train_rotations(dataloader)

        # Step 6: apply R1 rotation
        if self.config.r1 and self.config.online_r1_rotation:
            logger.info("[SpinQuant] Applying online R1 rotation (weight + wrapper)...")
            self._apply_online_r1()
            # Still fuse R2 and R4 offline
            self._fuse_r2_rotation()
            self._fuse_r4_rotation()
        else:
            logger.info("[SpinQuant] Fusing offline rotations into weights...")
            self._fuse_offline_rotations()

        # Step 7: register online hooks (R3 / R4)
        if self.config.r3 or self.config.r4:
            logger.info("[SpinQuant] Registering online rotation hooks...")
            self._hook_handles = register_spinquant_hooks(
                self.model,
                self.config,
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
                r4_rotation_size=self.r4_rotation_size,
            )

        # Step 8: cleanup training artefacts
        self._cleanup()

        # Store config on model for downstream serialization (export pipeline)
        self.model._rotation_config = self.config
        self.model._spinquant_config = self.config  # legacy alias

        # Print per-layer transformation summary table
        self._print_transformation_summary()

        logger.info("[SpinQuant] Preprocessing complete!")
        return self.model

    def _validate_dimensions(self) -> None:
        """Validate dimension requirements and disable rotations that can't work."""
        # R1: check r1_rotation_size divides hidden_size and is power of 2
        if self.config.r1 and self.r1_rotation_size > 0:
            if not is_pow2(self.r1_rotation_size):
                logger.warning(
                    f"[SpinQuant] R1 rotation_size must be a power of 2, "
                    f"but got {self.r1_rotation_size}. Disabling R1."
                )
                self.config.r1 = False
            elif self.hidden_size % self.r1_rotation_size != 0:
                logger.warning(
                    f"[SpinQuant] R1 rotation_size={self.r1_rotation_size} must divide "
                    f"hidden_size={self.hidden_size}. Disabling R1."
                )
                self.config.r1 = False

        if self.config.r2 and self.head_dim > 0 and not is_pow2(self.head_dim):
            logger.warning(
                f"[SpinQuant] R2 requires head_dim to be a power of 2, "
                f"but got head_dim={self.head_dim}. Disabling R2."
            )
            self.config.r2 = False

        if self.config.r3 and self.head_dim > 0 and not is_pow2(self.head_dim):
            logger.warning(
                f"[SpinQuant] R3 requires head_dim to be a power of 2, "
                f"but got head_dim={self.head_dim}. Disabling R3."
            )
            self.config.r3 = False

        if self.config.r3 and self.config.rotation_size is not None:
            logger.warning(
                f"[SpinQuant] R3 does not support custom rotation_size "
                f"(always uses head_dim={self.head_dim}). Ignoring rotation_size for R3."
            )

        if self.config.r4 and self.r4_rotation_size > 0:
            # Validate using get_hadamard_K — same decomposition used by matmul_hadU
            inter = self.r4_rotation_size
            try:
                _, K = get_hadamard_K(inter)
            except ValueError:
                logger.warning(
                    f"[SpinQuant] R4 cannot find Hadamard decomposition for " f"r4_rotation_size={inter}. Disabling R4."
                )
                self.config.r4 = False
                K = 0

            if self.config.r4 and self.intermediate_size % self.r4_rotation_size != 0:
                logger.warning(
                    f"[SpinQuant] R4 rotation_size={self.r4_rotation_size} must divide "
                    f"intermediate_size={self.intermediate_size}. Disabling R4."
                )
                self.config.r4 = False
            elif self.config.r4:
                logger.info(
                    f"[SpinQuant] R4 Hadamard: K={K}, "
                    f"r4_rotation_size={inter}, intermediate_size={self.intermediate_size}"
                )

    # ------------------------------------------------------------------
    # Step 3: Trainable RMSNorm
    # ------------------------------------------------------------------
    def _replace_norms_with_trainable(self) -> None:
        """Replace RMSNorm modules with TrainableRMSNorm wrappers."""
        layers = list(self._get_layers())
        if not layers:
            return

        for layer in layers:
            if hasattr(layer, "input_layernorm"):
                layer.input_layernorm = TrainableRMSNorm(layer.input_layernorm, trainable=self.config.trainable_smooth)
                if self.config.trainable_smooth:
                    self.smooth_params.append(layer.input_layernorm.smooth_values)

            if hasattr(layer, "post_attention_layernorm"):
                layer.post_attention_layernorm = TrainableRMSNorm(
                    layer.post_attention_layernorm, trainable=self.config.trainable_smooth
                )
                if self.config.trainable_smooth:
                    self.smooth_params.append(layer.post_attention_layernorm.smooth_values)

        # Final norm
        final_norm = None
        for path in ("model.norm", "norm"):
            parts = path.split(".")
            obj = self.model
            for p in parts:
                if not hasattr(obj, p):
                    break
                obj = getattr(obj, p)
            else:
                final_norm = obj
                break

        if final_norm is not None:
            wrapped = TrainableRMSNorm(final_norm, trainable=self.config.trainable_smooth)
            # Replace on model
            for path in ("model.norm", "norm"):
                parts = path.split(".")
                if len(parts) == 1:
                    if hasattr(self.model, parts[0]):
                        setattr(self.model, parts[0], wrapped)
                        break
                elif len(parts) == 2:
                    parent = getattr(self.model, parts[0], None)
                    if parent is not None and hasattr(parent, parts[1]):
                        setattr(parent, parts[1], wrapped)
                        break
            if self.config.trainable_smooth:
                self.smooth_params.append(wrapped.smooth_values)

    # ------------------------------------------------------------------
    # Step 4: Initialise rotation matrices
    # ------------------------------------------------------------------
    def _init_rotation_matrices(self) -> None:
        # Use the model's actual device, not config.device, to avoid mismatch
        model_device = next(self.model.parameters()).device
        dtype = self.config.dtype

        # R1: r1_rotation_size x r1_rotation_size
        if self.config.r1:
            r1_size = self.r1_rotation_size
            if self.config.online_r1_rotation and not self.config.trainable_rotation and not self.config.random_r1:
                # Online deterministic R1: Hadamard computed on-the-fly via
                # butterfly algorithm — no need to store a matrix buffer.
                logger.info(
                    f"[SpinQuant] R1: Online mode — deterministic Hadamard will be computed "
                    f"on-the-fly via butterfly algorithm (rotation_size={r1_size})"
                )
            elif self.config.trainable_rotation and not self.config.online_r1_rotation:
                R1 = nn.Parameter(torch.eye(r1_size, device=model_device, dtype=dtype))
                logger.info(f"[SpinQuant] R1: Trainable rotation matrix [{r1_size}×{r1_size}] (identity init)")
                self._register_rotation("spinquant_R1", R1)
            else:
                if self.config.random_r1:
                    R1 = nn.Parameter(
                        random_hadamard_matrix(r1_size, dtype=dtype, device=model_device),
                        requires_grad=False,
                    )
                    matrix_type = "Random Hadamard"
                else:
                    R1 = nn.Parameter(
                        deterministic_hadamard_matrix(r1_size, dtype=dtype, device=model_device),
                        requires_grad=False,
                    )
                    matrix_type = "Deterministic Hadamard"
                mode = "online" if self.config.online_r1_rotation else "offline fuse"
                if r1_size < self.hidden_size:
                    logger.info(
                        f"[SpinQuant] R1: {matrix_type} [{r1_size}×{r1_size}] block rotation "
                        f"({self.hidden_size // r1_size} blocks, fixed, {mode})"
                    )
                else:
                    logger.info(f"[SpinQuant] R1: {matrix_type} [{r1_size}×{r1_size}] (fixed, {mode})")
                self._register_rotation("spinquant_R1", R1)

        # R2_head: head_dim x head_dim
        if self.config.r2 and self.head_dim > 0:
            if self.config.trainable_rotation:
                R2_head = nn.Parameter(torch.eye(self.head_dim, device=model_device, dtype=dtype))
                logger.info(
                    f"[SpinQuant] R2: Trainable per-head rotation [{self.head_dim}×{self.head_dim}] (identity init)"
                )
            else:
                if self.config.random_r2:
                    R2_head = nn.Parameter(
                        random_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device),
                        requires_grad=False,
                    )
                    matrix_type = "Random Hadamard"
                else:
                    R2_head = nn.Parameter(
                        deterministic_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device),
                        requires_grad=False,
                    )
                    matrix_type = "Deterministic Hadamard"
                logger.info(
                    f"[SpinQuant] R2: {matrix_type} [{self.head_dim}×{self.head_dim}] per head (fixed, offline fuse)"
                )
            self._register_rotation("spinquant_R2_head", R2_head)

        # R3_head: head_dim × head_dim (online, after RoPE on Q/K)
        if self.config.r3 and self.head_dim > 0:
            if self.config.random_r3:
                R3 = random_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device)
                matrix_type = "Random Hadamard"
                mode_str = "online, x @ R"
            else:
                R3 = deterministic_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device)
                matrix_type = "Deterministic Hadamard"
                mode_str = "online, butterfly"
            self.model.register_buffer("spinquant_R3_head", R3)
            logger.info(f"[SpinQuant] R3: {matrix_type} " f"[{self.head_dim}×{self.head_dim}] after RoPE ({mode_str})")

        # R4: r4_rotation_size (online hook on down_proj + offline fuse)
        if self.config.r4 and self.r4_rotation_size > 0:
            r4_size = self.r4_rotation_size
            if self.config.random_r4:
                R4 = random_hadamard_matrix(r4_size, dtype=dtype, device=model_device)
                self.model.register_buffer("spinquant_R4_matrix", R4)
                matrix_type = "Random Hadamard"
                mode_str = "online x @ R + offline fuse"
            else:
                matrix_type = "Deterministic Hadamard"
                mode_str = "online butterfly + offline fuse"
            # Always store Hadamard decomposition (needed for deterministic path
            # and for backward compat with serialize.py buffer detection)
            had_K, K = get_hadamard_K(r4_size)
            had_K = had_K.to(dtype=dtype, device=model_device)
            self.model.register_buffer("spinquant_R4_had_K", had_K)
            self.model.register_buffer("spinquant_R4_K", torch.tensor(K, device=model_device))
            if r4_size < self.intermediate_size:
                logger.info(
                    f"[SpinQuant] R4: {matrix_type} K={K}, "
                    f"rotation_size={r4_size}, "
                    f"{self.intermediate_size // r4_size} rotation blocks on down_proj ({mode_str})"
                )
            else:
                logger.info(f"[SpinQuant] R4: {matrix_type} K={K}, " f"rotation_size={r4_size} ({mode_str})")

    def _register_rotation(self, name: str, param: nn.Parameter) -> None:
        self.model.register_parameter(name, param)
        if param.requires_grad:
            self.rotation_params.append(param)

    # ------------------------------------------------------------------
    # Step 5: Training
    # ------------------------------------------------------------------
    def _train_rotations(self, dataloader: Any) -> None:
        """Train rotation matrices using the shared training loop.

        Delegates to :func:`training_core.run_training_loop` which is the
        single implementation shared with :class:`RotationTrainer`.

        For advanced training features (callbacks, evaluation, checkpointing,
        custom loss), use ``RotationTrainer`` from ``trainer.py`` directly.
        """
        from auto_round.algorithms.transforms.spinquant.training_core import (
            clone_model_for_reference,
            create_dual_optimizer,
            run_training_loop,
        )

        optimizer = create_dual_optimizer(
            self.model,
            lr=self.config.lr,
            smooth_lr=self.config.smooth_lr,
        )
        if optimizer is None:
            return

        logger.info("[SpinQuant] Cloning original model for KL reference...")
        original_model = clone_model_for_reference(self.model)

        result = run_training_loop(
            model=self.model,
            original_model=original_model,
            optimizer=optimizer,
            dataloader=dataloader,
            max_iters=self.config.iters,
            loss_type=self.config.loss_type,
            kl_top_k=self.config.kl_top_k,
            log_interval=50,
        )

        logger.info(
            f"[SpinQuant] Training complete: {result.steps} steps, "
            f"best_loss={result.best_loss:.6f}, ortho_dev={result.final_ortho_deviation:.2e}"
        )

        del original_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_embed_tokens(self) -> Optional[nn.Module]:
        """Get embedding module, supporting both model.embed_tokens and model.model.embed_tokens."""
        for attr_path in ("embed_tokens", "model.embed_tokens"):
            parts = attr_path.split(".")
            obj = self.model
            for p in parts:
                if not hasattr(obj, p):
                    break
                obj = getattr(obj, p)
            else:
                return obj
        return None

    def _get_layers(self):
        """Yield transformer layers, supporting multiple nesting patterns."""
        for attr_path in ("layers", "model.layers", "transformer.h", "model.decoder.layers"):
            parts = attr_path.split(".")
            obj = self.model
            for p in parts:
                if not hasattr(obj, p):
                    break
                obj = getattr(obj, p)
            else:
                if hasattr(obj, "__iter__"):
                    for layer in obj:
                        yield layer
                    return
        # Fallback: search recursively
        for name, module in self.model.named_modules():
            if name.endswith(".layers") or name == "layers":
                if hasattr(module, "__iter__"):
                    for layer in module:
                        yield layer
                    return

    def _get_lm_head(self) -> Optional[nn.Module]:
        """Get LM head module."""
        return getattr(self.model, "lm_head", None)

    # ------------------------------------------------------------------
    # Step 6a: Online R1 rotation (matching Quark's default behavior)
    # ------------------------------------------------------------------
    def _apply_online_r1(self) -> None:
        """Apply online R1 rotation: rotate target module weights and register
        ``forward_pre_hook`` s so the matching activation rotation is applied at
        runtime.

        Supports two modes:
        - **Deterministic Hadamard** (default): uses butterfly algorithm
          (``matmul_hadU``) for both weight rotation and activation hooks —
          no matrix needs to be stored.
        - **Random Hadamard** (``random_r1=True``): uses the stored
          ``spinquant_R1`` matrix for explicit ``x @ R`` in both weight
          rotation and activation hooks.

        This matches Quark's ``apply_online_r1()`` behavior:
        - Target modules (q/k/v_proj, gate/up_proj) get their weights rotated
        - A ``forward_pre_hook`` on each target module applies the same rotation
          to activations at runtime
        - The two transforms cancel out: ``R(x) @ (W @ R).T = x @ W.T``
        - prev_modules (embed_tokens, o_proj, down_proj) are NOT modified
        - RMSNorm gamma is NOT fused
        - lm_head is NOT modified (last_layer skipped, matching Quark)

        Hooks are compatible with auto-round's quantization pipeline:
        WrapperLinear.forward runs ``orig_layer._forward_pre_hooks`` before
        the linear computation, and WrapperWALayer steals & runs them at
        inference time.

        .. warning::
            Hook-based online R1 is **not serializable** — ``save_pretrained()``
            will NOT save the activation hooks.  If you need to save and reload
            the rotated model, use offline R1 instead
            (``SpinQuantConfig(online_r1_rotation=False)``).
        """
        r1_size = self.r1_rotation_size
        use_random = self.config.random_r1

        logger.warning(
            "[SpinQuant] Online R1 uses forward_pre_hooks which are NOT saved by "
            "save_pretrained(). The saved model will lose activation rotation hooks. "
            "Use SpinQuantConfig(online_r1_rotation=False) for offline R1 if you "
            "need to save/reload the model."
        )

        model_device = next(self.model.parameters()).device

        # For random R1: use the stored full matrix
        # For deterministic R1: use butterfly algorithm
        if use_random:
            R1_full = self._get_rotation_tensor("spinquant_R1")
            if R1_full is None:
                raise RuntimeError(
                    "[SpinQuant] random_r1=True but spinquant_R1 buffer not found. "
                    "Ensure _init_rotation_matrices() was called first."
                )
            R1_full = R1_full.data.to(torch.float64)
            hadamard_K, K = None, None
        else:
            R1_full = None
            hadamard_K, K = get_hadamard_K(r1_size)
            hadamard_K = hadamard_K.to(model_device)

        n_rotated = 0
        n_hooked = 0

        for layer in self._get_layers():
            if not (hasattr(layer, "self_attn") and hasattr(layer, "mlp")):
                continue

            layer_device = next(layer.parameters()).device

            attn = layer.self_attn
            mlp = layer.mlp

            # Target modules: (parent_module, attr_name)
            target_specs = []
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                if hasattr(attn, proj_name):
                    target_specs.append((attn, proj_name))
            for proj_name in ("gate_proj", "up_proj"):
                if hasattr(mlp, proj_name):
                    target_specs.append((mlp, proj_name))

            for parent, attr_name in target_specs:
                module = getattr(parent, attr_name)
                dtype = module.weight.data.dtype
                in_features = module.weight.shape[-1]

                if use_random:
                    # Random R1: explicit matrix multiply
                    R = R1_full.to(layer_device)
                    if r1_size == in_features:
                        W = module.weight.data.to(torch.float64)
                        module.weight.data = (W @ R).to(dtype)
                    elif in_features % r1_size == 0:
                        rotate_in_channels_(module, R_in=R)
                    else:
                        raise ValueError(
                            f"Online R1: in_features={in_features} not compatible " f"with r1_rotation_size={r1_size}"
                        )
                else:
                    # Deterministic Hadamard: butterfly algorithm
                    had_K_local = hadamard_K.to(layer_device)
                    if r1_size == in_features:
                        module.weight.data = matmul_hadU(module.weight.data, hadamard_K=had_K_local, K=K).to(dtype)
                    elif in_features % r1_size == 0:
                        R_block = had_K_local.to(torch.float64)
                        if R_block.shape[0] != r1_size:
                            had_1, _ = get_hadamard_K(r1_size // K)
                            R_block = torch.kron(
                                had_K_local.to(device="cpu", dtype=torch.float64),
                                had_1.to(device="cpu", dtype=torch.float64),
                            )
                        R_block = R_block / math.sqrt(r1_size)
                        rotate_in_channels_(module, R_in=R_block)
                    else:
                        raise ValueError(
                            f"Online R1: in_features={in_features} not compatible " f"with r1_rotation_size={r1_size}"
                        )
                n_rotated += 1

                # Register forward_pre_hook for online activation rotation
                if use_random:
                    hook = self._make_online_r1_hook_matrix(R1_full.to(layer_device), r1_size, in_features)
                else:
                    hook = self._make_online_r1_hook_butterfly(r1_size, in_features, hadamard_K.to(layer_device), K)
                hook._spinquant_hook = True  # tag for selective removal
                handle = module.register_forward_pre_hook(hook)
                self._r1_hook_handles.append(handle)
                n_hooked += 1

        mode_str = "random matrix (x @ R)" if use_random else "deterministic butterfly"
        logger.info(
            f"[SpinQuant] Online R1: rotated {n_rotated} target modules, "
            f"registered {n_hooked} activation hooks "
            f"(rotation_size={r1_size}, mode={mode_str}, "
            f"lm_head/embed_tokens/o_proj/down_proj unchanged)"
        )

    @staticmethod
    def _make_online_r1_hook_butterfly(r1_size, in_features, hadamard_K, K):
        """Create a forward_pre_hook using butterfly algorithm (deterministic Hadamard)."""
        if r1_size == in_features:

            def hook(module, args):
                x = args[0]
                x = matmul_hadU(x, hadamard_K=hadamard_K.to(x.device), K=K)
                return (x,) + args[1:]

        else:
            R_block = hadamard_K.to(torch.float64)
            if R_block.shape[0] != r1_size:
                had_1, _ = get_hadamard_K(r1_size // K)
                R_block = torch.kron(
                    hadamard_K.to(device="cpu", dtype=torch.float64),
                    had_1.to(device="cpu", dtype=torch.float64),
                )
            R_block = R_block / math.sqrt(r1_size)
            R_block_f32 = R_block.float()

            def hook(module, args):
                x = args[0]
                dtype = x.dtype
                shape = x.shape
                R = R_block_f32.to(x.device, dtype=x.dtype)
                x = x.reshape(*shape[:-1], -1, r1_size)
                x = (x @ R).reshape(shape).to(dtype)
                return (x,) + args[1:]

        return hook

    @staticmethod
    def _make_online_r1_hook_matrix(R1_matrix, r1_size, in_features):
        """Create a forward_pre_hook using stored full matrix (random/trained R1)."""
        R1_f32 = R1_matrix.float()

        if r1_size == in_features:

            def hook(module, args):
                x = args[0]
                R = R1_f32.to(x.device, dtype=x.dtype)
                x = x @ R
                return (x,) + args[1:]

        else:

            def hook(module, args):
                x = args[0]
                dtype = x.dtype
                shape = x.shape
                R = R1_f32.to(x.device, dtype=x.dtype)
                x = x.reshape(*shape[:-1], -1, r1_size)
                x = (x @ R).reshape(shape).to(dtype)
                return (x,) + args[1:]

        return hook

    # ------------------------------------------------------------------
    # Step 6b: Fuse offline rotations
    # ------------------------------------------------------------------
    def _fuse_offline_rotations(self) -> None:
        self._rotated_modules.clear()

        if not self.config.r1:
            # Even without R1, we may still need R2 and R4
            self._fuse_r2_rotation()
            self._fuse_r4_rotation()
            return

        R1 = self._get_rotation_tensor("spinquant_R1")
        if R1 is None or R1.numel() == 0:
            return
        R1_inv = R1.t()
        r1_size = R1.shape[0]

        # Embed tokens (output rotation: W_embed @ R1)
        embed = self._get_embed_tokens()
        if embed is not None:
            with torch.no_grad():
                W_f64 = embed.weight.data.to(torch.float64)
                R_f64 = R1.to(embed.weight.device).to(torch.float64)
                if W_f64.shape[-1] == r1_size:
                    new_w = torch.matmul(W_f64, R_f64)
                elif W_f64.shape[-1] % r1_size == 0:
                    # Block rotation for embedding
                    w_reshaped = W_f64.reshape(*W_f64.shape[:-1], -1, r1_size)
                    new_w = (w_reshaped @ R_f64).reshape(W_f64.shape)
                else:
                    raise ValueError(f"embed_tokens dim={W_f64.shape[-1]} not divisible by R1 size={r1_size}")
                embed.weight.data = new_w.to(embed.weight.dtype)

        # Transformer layers
        n_layers = 0
        for layer in self._get_layers():
            if not (hasattr(layer, "self_attn") and hasattr(layer, "mlp")):
                continue
            attn = layer.self_attn
            mlp = layer.mlp

            # Ensure R1_inv is on the same device as layer weights
            layer_device = next(layer.parameters()).device
            R1_inv_local = R1_inv.to(layer_device)
            R1_local = R1.to(layer_device)

            # Attention: in-channel uses R1_inv (→ W @ R1), out-channel uses R1 (→ R1_inv @ W)
            if hasattr(attn, "q_proj"):
                rotate_in_channels_(attn.q_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(attn, "k_proj"):
                rotate_in_channels_(attn.k_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(attn, "v_proj"):
                rotate_in_channels_(attn.v_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(attn, "o_proj"):
                rotate_out_channels_(attn.o_proj, R_out=R1_local, rotated_modules=self._rotated_modules)

            # MLP: same convention
            if hasattr(mlp, "gate_proj"):
                rotate_in_channels_(mlp.gate_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(mlp, "up_proj"):
                rotate_in_channels_(mlp.up_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(mlp, "down_proj"):
                rotate_out_channels_(mlp.down_proj, R_out=R1_local, rotated_modules=self._rotated_modules)
            n_layers += 1

        logger.info(
            f"[SpinQuant] R1 fused into {n_layers} layers (embed_tokens, q/k/v/o_proj, gate/up/down_proj, lm_head)"
        )

        # LM head: in-channel uses R1_inv (→ W @ R1)
        lm_head = self._get_lm_head()
        if lm_head is not None:
            lm_device = lm_head.weight.device
            rotate_in_channels_(lm_head, R_in=R1_inv.to(lm_device), rotated_modules=self._rotated_modules)

        # R2 head-dim rotation (per-head Hadamard on v_proj output + o_proj input)
        self._fuse_r2_rotation()

        # R4 offline fusion (Hadamard on down_proj input side)
        self._fuse_r4_rotation()

    def _fuse_r2_rotation(self) -> None:
        """Fuse R2 per-head rotation into v_proj and o_proj.

        Uses the stored ``spinquant_R2_head`` matrix (which may be deterministic
        Hadamard, random Hadamard, or a trained orthogonal matrix) to rotate
        v_proj output channels and o_proj input channels per attention head.

        Math:
            v_rotated = v @ R2  per head  →  fuse into v_proj: W_new = R2^T @ W
            o_proj input is R2-rotated    →  fuse into o_proj: W_new = W @ R2
        """
        if not self.config.r2 or self.head_dim <= 0:
            return

        R2_head = self._get_rotation_tensor("spinquant_R2_head")
        if R2_head is None:
            return

        R2 = R2_head.data.to(torch.float64)
        R2_T = R2.t()

        n_fused = 0
        for layer in self._get_layers():
            if not hasattr(layer, "self_attn"):
                continue
            attn = layer.self_attn

            # v_proj: W_new = R2^T @ W per head on output dimension
            if hasattr(attn, "v_proj"):
                W = attn.v_proj.weight.data
                dtype = W.dtype
                W = W.to(torch.float64)
                n_heads = W.shape[0] // self.head_dim
                W_reshaped = W.reshape(n_heads, self.head_dim, W.shape[1])
                W_reshaped = torch.einsum("ij,kjl->kil", R2_T, W_reshaped)
                attn.v_proj.weight.data = W_reshaped.reshape(W.shape).to(dtype)

            # o_proj: W_new = W @ R2 per head on input dimension
            # (R2^{-1} = R2^T on the activation side ↔ W @ R2 on the weight side)
            if hasattr(attn, "o_proj"):
                W = attn.o_proj.weight.data
                dtype = W.dtype
                W = W.to(torch.float64)
                n_heads = W.shape[1] // self.head_dim
                W_reshaped = W.reshape(W.shape[0], n_heads, self.head_dim)
                W_reshaped = torch.einsum("ijk,kl->ijl", W_reshaped, R2)
                attn.o_proj.weight.data = W_reshaped.reshape(W.shape).to(dtype)

            n_fused += 1

        logger.info(f"[SpinQuant] R2 fused into {n_fused} layers (v_proj out + o_proj in, head_dim={self.head_dim})")

    def _fuse_r4_rotation(self) -> None:
        """Fuse R4 rotation into down_proj's input side.

        For deterministic Hadamard: uses :func:`matmul_hadU` (butterfly algorithm).
        For random Hadamard: uses the stored ``spinquant_R4_matrix`` buffer with
        explicit ``W @ R`` matrix multiply.

        The R4 hook applies ``x → x @ R`` on activations before down_proj.
        To cancel this in the weight: ``(x @ R) @ W^T = x @ (W @ R)^T``,
        so we fuse ``W_new = W @ R`` on the input dimension.
        """
        if not self.config.r4 or self.r4_rotation_size <= 0:
            return

        r4_size = self.r4_rotation_size
        use_random = self.config.random_r4

        # Get the rotation matrix
        if use_random:
            R4_matrix = getattr(self.model, "spinquant_R4_matrix", None)
            if R4_matrix is None:
                raise RuntimeError("[SpinQuant] random_r4=True but spinquant_R4_matrix buffer not found.")
            R4 = R4_matrix.to(torch.float64)

        n_fused = 0
        for layer in self._get_layers():
            if not hasattr(layer, "mlp"):
                continue
            mlp = layer.mlp
            if hasattr(mlp, "down_proj"):
                W = mlp.down_proj.weight.data
                dtype = W.dtype

                if use_random:
                    # Random: explicit W @ R per block
                    W = W.to(torch.float64)
                    if r4_size == W.shape[1]:
                        mlp.down_proj.weight.data = (W @ R4).to(dtype)
                    else:
                        out_feat, in_feat = W.shape
                        n_blocks = in_feat // r4_size
                        W_reshaped = W.reshape(out_feat, n_blocks, r4_size)
                        W_reshaped = torch.einsum("ijk,kl->ijl", W_reshaped, R4)
                        mlp.down_proj.weight.data = W_reshaped.reshape(out_feat, in_feat).to(dtype)
                else:
                    # Deterministic: matmul_hadU (butterfly algorithm)
                    # matmul_hadU operates on the last dimension — for weight
                    # shape [out, in], last dim = in_features which is what
                    # we want to rotate (input channels of down_proj).
                    if r4_size == W.shape[1]:
                        mlp.down_proj.weight.data = matmul_hadU(W).to(dtype)
                    else:
                        out_feat, in_feat = W.shape
                        n_blocks = in_feat // r4_size
                        W_reshaped = W.reshape(out_feat, n_blocks, r4_size)
                        W_rotated = matmul_hadU(W_reshaped)
                        mlp.down_proj.weight.data = W_rotated.reshape(out_feat, in_feat).to(dtype)
                n_fused += 1

        mode_str = "random x @ R" if use_random else "deterministic butterfly"
        logger.info(
            f"[SpinQuant] R4 offline fused into {n_fused} down_proj layers "
            f"(r4_rotation_size={r4_size}, mode={mode_str})"
        )

    # ------------------------------------------------------------------
    # Step 8: Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        self.model.eval()

        # Keep rotation matrices on the model — they are needed by
        # inject_buffers_on_layer during serialization (random / trained
        # matrices must be read back via _get_stored_rotation).
        # Only the internal tracking lists are cleared below.

        # NOTE: Do NOT remove online hooks/monkeypatches here. R3/R4 hooks
        # must persist for correct inference. Only remove training-related state.

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Clear internal tracking
        self.rotation_params.clear()
        self.smooth_params.clear()
        self._rotated_modules.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_rotation_tensor(self, name: str) -> Optional[torch.Tensor]:
        if hasattr(self.model, name):
            tensor = getattr(self.model, name)
            if isinstance(tensor, (nn.Parameter, torch.Tensor)):
                return tensor.data if isinstance(tensor, nn.Parameter) else tensor
        return None

    def _print_transformation_summary(self) -> None:
        """Print a per-layer table summarizing all applied transformations and hooks."""
        lines = []
        lines.append("")
        lines.append("=" * 100)
        lines.append("SpinQuant Transformation Summary")
        lines.append("=" * 100)

        # --- Global transforms ---
        lines.append("")
        lines.append("Global Transforms:")
        lines.append(f"  {'Component':<30} {'Transform':<50} {'Status'}")
        lines.append(f"  {'-'*30} {'-'*50} {'-'*10}")

        # Untie embeddings
        tied = getattr(self.model.config, "tie_word_embeddings", False) if hasattr(self.model, "config") else "unknown"
        if self.config.online_r1_rotation:
            lines.append(
                f"  {'embed_tokens / lm_head':<30} {'Untie word embeddings (skipped: online R1)':<50} {'- n/a'}"
            )
        else:
            lines.append(
                f"  {'embed_tokens / lm_head':<30} {'Untie word embeddings':<50} "
                f"{'✓ untied' if not tied else '✗ still tied'}"
            )

        # RMSNorm fusion
        if self.config.online_r1_rotation:
            lines.append(f"  {'All RMSNorm layers':<30} {'Fuse gamma (skipped: online R1)':<50} {'- n/a'}")
        else:
            lines.append(
                f"  {'All RMSNorm layers':<30} {'Fuse gamma into linear weights':<50} "
                f"{'✓ fused' if self.config.fuse_rmsnorm else '✗ skipped'}"
            )

        # R1
        if self.config.r1:
            r1_size = self.r1_rotation_size
            mode = "online" if self.config.online_r1_rotation else "offline"
            if r1_size < self.hidden_size:
                r1_desc = f"Block Hadamard {r1_size}×{r1_size}, {self.hidden_size // r1_size} blocks ({mode})"
            else:
                r1_desc = f"Hadamard {r1_size}×{r1_size} ({mode})"
        else:
            r1_desc = "Disabled"
        lines.append(f"  {'Residual stream (R1)':<30} {r1_desc:<50} {'✓' if self.config.r1 else '✗'}")

        # --- Per-layer table ---
        lines.append("")
        lines.append("Per-Layer Transforms:")

        # Determine R4 K
        r4_K = 0
        if self.config.r4 and self.r4_rotation_size > 0:
            K = 1
            r4_size = self.r4_rotation_size
            while K * 2 <= r4_size and r4_size % (K * 2) == 0:
                K *= 2
            if K > 1:
                r4_K = K

        # Table header
        col_layer = "Layer"
        col_r1 = "R1 (online)" if self.config.online_r1_rotation else "R1 (weight fuse)"
        col_r2 = "R2 (weight fuse)"
        col_r3 = "R3 (online hook)"
        col_r4 = "R4 (online hook)"
        header = f"  {col_layer:<35} {col_r1:<20} {col_r2:<20} {col_r3:<22} {col_r4:<22}"
        lines.append(header)
        lines.append(f"  {'-'*35} {'-'*20} {'-'*20} {'-'*22} {'-'*22}")

        online_r1 = self.config.online_r1_rotation

        # embed_tokens
        if online_r1:
            r1_embed = "-"  # online R1 doesn't touch embed
        else:
            r1_embed = f"W@R ({self.r1_rotation_size})" if self.config.r1 else "-"
        lines.append(f"  {'model.embed_tokens':<35} {r1_embed:<20} {'-':<20} {'-':<22} {'-':<22}")

        # Transformer layers
        layers = list(self._get_layers())
        n_layers = sum(1 for l in layers if hasattr(l, "self_attn") and hasattr(l, "mlp"))

        for i, layer in enumerate(layers):
            if not (hasattr(layer, "self_attn") and hasattr(layer, "mlp")):
                continue

            # Only show first 2 and last layer to keep output concise
            if n_layers > 5 and 2 <= i < n_layers - 1:
                if i == 2:
                    lines.append(f"  {'  ... (same pattern for all layers)':<35}")
                continue

            layer_name = f"layers.{i}"

            if online_r1 and self.config.r1:
                r1_attn = "q/k/v:W@H+hook"
                r1_mlp = "g/u:W@H+hook"
            elif self.config.r1:
                r1_attn = "q/k/v:R⁻¹@W o:W@R"
                r1_mlp = "g/u:R⁻¹@W d:W@R"
            else:
                r1_attn = "-"
                r1_mlp = "-"

            # R2 for this layer: affects v_proj out, o_proj in
            r2_status = f"H({self.head_dim}) v↔o" if self.config.r2 else "-"

            # R3: check if monkeypatch applied (look for wrapper in self_attn)
            r3_status = "-"
            if self.config.r3 and self.head_dim > 0 and is_pow2(self.head_dim):
                r3_status = f"H({self.head_dim}) Q,K post-RoPE"

            # R4: check if hook on down_proj
            r4_status = "-"
            if self.config.r4 and r4_K > 0:
                n_blocks = self.r4_rotation_size // r4_K
                r4_status = f"blockH(K={r4_K},b={n_blocks})"

            # Print attention row
            lines.append(f"  {layer_name + '.self_attn':<35} {r1_attn:<20} {r2_status:<20} {r3_status:<22} {'-':<22}")
            # Print MLP row
            lines.append(f"  {layer_name + '.mlp':<35} {r1_mlp:<20} {'-':<20} {'-':<22} {r4_status:<22}")

        # lm_head
        if online_r1:
            r1_lm = "-"  # online R1 doesn't touch lm_head
        else:
            r1_lm = f"R⁻¹@W ({self.r1_rotation_size})" if self.config.r1 else "-"
        lines.append(f"  {'model.lm_head':<35} {r1_lm:<20} {'-':<20} {'-':<22} {'-':<22}")

        # --- Hook summary ---
        lines.append("")
        lines.append("Registered Hooks:")

        # R1 online hooks are stored in self._r1_hook_handles
        r1_hooks = len(self._r1_hook_handles)

        # R3/R4 hooks are stored in self._hook_handles (from register_spinquant_hooks)
        r3_hooks = sum(1 for h in self._hook_handles if isinstance(h, tuple) and h[0] == "r3_monkeypatch")
        r4_hooks = sum(1 for h in self._hook_handles if not (isinstance(h, tuple) and h[0] == "r3_monkeypatch"))

        lines.append(f"  R1 online hooks (Hadamard on target module input):          {r1_hooks} modules")
        lines.append(f"  R3 monkeypatch (apply_rotary_pos_emb → QKRotationWrapper):  {r3_hooks} attention layers")
        lines.append(f"  R4 forward_pre_hook (block Hadamard on down_proj input):    {r4_hooks} MLP layers")

        # --- Summary totals ---
        lines.append("")
        lines.append("Totals:")
        lines.append(f"  Transformer layers:   {n_layers}")
        if online_r1 and self.config.r1:
            lines.append(f"  Online R1 targets:    {n_layers}×(q/k/v_proj, gate/up_proj) = {r1_hooks} modules")
            fused_parts = []
        else:
            fused_parts = ["embed_tokens"]
            if self.config.r1:
                fused_parts.append(f"{n_layers}×(q/k/v/o_proj, gate/up/down_proj), lm_head")
        if self.config.r2:
            fused_parts.append(f"{n_layers}×(v_proj↔o_proj)")
        if self.config.r4 and r4_K > 0:
            fused_parts.append(f"{n_layers}×(down_proj)")
        lines.append(f"  Offline-fused params: {', '.join(fused_parts) if fused_parts else 'none'}")
        total_hooks = r1_hooks + r3_hooks + r4_hooks
        lines.append(f"  Online hooks:         {total_hooks} total ({r1_hooks} R1 + {r3_hooks} R3 + {r4_hooks} R4)")
        lines.append(f"  Inference overhead:   R1={'O(seq×hidden×log₂H) per module' if r1_hooks > 0 else 'none'}")
        lines.append(
            f"                        R3={'O(seq×heads×d_head×log₂d_head) per layer' if r3_hooks > 0 else 'none'}"
        )
        lines.append(f"                        R4={'O(seq×inter×log₂K) per layer' if r4_hooks > 0 else 'none'}")
        lines.append("=" * 100)

        # Log as a single multi-line message
        logger.info("\n".join(lines))


def remove_spinquant_hooks_from_model(model: nn.Module) -> None:
    """Remove all SpinQuant hooks and R3 monkeypatches from a model.

    Removes:
    - Forward hooks / pre-hooks tagged with ``_spinquant_hook = True`` (R1/R4)
    - R3 monkeypatches on attention modules (tagged with ``_spinquant_r3_patched``)

    Hooks from other frameworks are left untouched.
    """
    for module in model.modules():
        # Remove tagged forward hooks (R4) and pre-hooks (R1 online)
        if hasattr(module, "_forward_hooks"):
            for hook_id in list(module._forward_hooks.keys()):
                hook = module._forward_hooks[hook_id]
                if getattr(hook, "_spinquant_hook", False):
                    del module._forward_hooks[hook_id]
        if hasattr(module, "_forward_pre_hooks"):
            for hook_id in list(module._forward_pre_hooks.keys()):
                hook = module._forward_pre_hooks[hook_id]
                if getattr(hook, "_spinquant_hook", False):
                    del module._forward_pre_hooks[hook_id]

        # Remove R3 monkeypatches (instance-level forward override)
        if getattr(module, "_spinquant_r3_patched", False):
            if "forward" in module.__dict__:
                del module.__dict__["forward"]
            delattr(module, "_spinquant_r3_patched")
