# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
SpinQuant in-place application utilities.

This module provides ``apply_spinquant_in_place`` and hook registration
that follow the same patterns used by AutoRound's
``auto_round.algorithms.transforms.rotation.inplace`` package.

R3 rotation uses the architecture-generic monkeypatch approach from QuaRot/Quark:
we replace ``apply_rotary_pos_emb`` in the attention forward's globals with a
wrapper that applies Hadamard after RoPE. This works for any HuggingFace model
(Llama, Qwen2, Qwen3, Mistral, Phi, Gemma, etc.).

R4 rotation uses a forward_pre_hook on ``down_proj`` that applies block Hadamard.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    is_pow2,
    matmul_hadU,
)

logger = logging.getLogger("autoround.spinquant")


def register_spinquant_hooks(
    model: nn.Module,
    config: Any,
    compute_device: Optional[torch.device] = None,
    head_dim: int = 0,
    intermediate_size: int = 0,
    r4_rotation_size: int = 0,
) -> list[Any]:
    """Register online rotation hooks for SpinQuant R3 (Q/K) and R4 (MLP activation).

    R3 uses the architecture-generic monkeypatch approach: replaces
    ``apply_rotary_pos_emb`` in attention forward's globals with a wrapper
    that applies rotation to Q and K after RoPE.

    R4 registers a forward_pre_hook on each ``down_proj`` that applies
    rotation to the activation before the linear layer.

    Both R3 and R4 support two modes:
    - **Deterministic** (default): butterfly algorithm via ``matmul_hadU``
    - **Random** (``random_r3/r4=True``): explicit ``x @ R`` using stored matrix

    Args:
        model: The transformer model to patch.
        config: A ``SpinQuantConfig`` instance (or any object with ``r3``,
            ``r4`` booleans and optional ``random_r3``, ``random_r4``).
        compute_device: Device for hook computation.
        head_dim: Per-head dimension for R3 rotation. If 0, tries config.head_dim.
        intermediate_size: MLP intermediate dimension for R4. If 0, tries config.intermediate_size.
        r4_rotation_size: Override R4 rotation size (for custom rotation_size).
            If 0, falls back to intermediate_size.

    Returns:
        A list of hook handles that can be used to remove the hooks later.
    """
    handles: list[Any] = []

    if compute_device is None:
        compute_device = next(model.parameters()).device

    # Resolve dimensions from explicit args or config fallback
    if head_dim <= 0:
        head_dim = getattr(config, "head_dim", 0)
    if intermediate_size <= 0:
        intermediate_size = getattr(config, "intermediate_size", 0)

    random_r3 = getattr(config, "random_r3", False)
    random_r4 = getattr(config, "random_r4", False)

    # ------------------------------------------------------------------
    # R3: Q/K rotation after RoPE (head_dim rotation)
    # Uses monkeypatch to wrap apply_rotary_pos_emb in attention forward.
    # ------------------------------------------------------------------
    if getattr(config, "r3", False) and head_dim > 0:
        # Validate head_dim is power-of-2
        if not is_pow2(head_dim):
            logger.warning(
                f"[SpinQuant] R3 requires head_dim to be a power of 2, but got head_dim={head_dim}. "
                f"Skipping R3 rotation. Model accuracy may be affected."
            )
        else:
            from auto_round.algorithms.transforms.spinquant.monkeypatch import (
                QKRotationWrapper,
                add_qk_rotation_after_rope,
            )

            # Get stored R3 matrix for random mode
            r3_matrix = getattr(model, "spinquant_R3_head", None)

            r3_count = 0
            for name, module in model.named_modules():
                if name.endswith("self_attn") and hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                    try:
                        wrapper = add_qk_rotation_after_rope(
                            module,
                            rope_function_name="apply_rotary_pos_emb",
                        )
                        if random_r3 and r3_matrix is not None:
                            wrapper.set_matrix(r3_matrix)
                        else:
                            wrapper.set_hadamard(None, head_dim)
                        module._spinquant_r3_patched = True
                        handles.append(("r3_monkeypatch", name, module, wrapper))
                        r3_count += 1
                    except ValueError as e:
                        if r3_count == 0:
                            # First layer failed - likely unsupported architecture
                            logger.warning(
                                f"[SpinQuant] R3 monkeypatch failed for '{name}': {e}. "
                                f"This model architecture may not support R3 rotation. Skipping R3."
                            )
                            break
                        else:
                            logger.warning(f"[SpinQuant] R3 monkeypatch failed for '{name}': {e}")

            if r3_count > 0:
                mode = "random x @ R" if (random_r3 and r3_matrix is not None) else "deterministic butterfly"
                logger.info(
                    f"[SpinQuant] R3: Applied rotation(head_dim={head_dim}, mode={mode}) "
                    f"after RoPE on {r3_count} attention layers"
                )

    # ------------------------------------------------------------------
    # R4: MLP activation rotation (intermediate_size rotation)
    # ------------------------------------------------------------------
    if getattr(config, "r4", False) and intermediate_size > 0:
        # Use r4_rotation_size if provided, otherwise fall back to intermediate_size
        r4_size = r4_rotation_size if r4_rotation_size > 0 else intermediate_size

        # Determine if we need block rotation (r4_size < intermediate_size)
        need_block_rotation = r4_size < intermediate_size

        if random_r4:
            # Random R4: use stored full matrix
            r4_matrix = getattr(model, "spinquant_R4_matrix", None)
            if r4_matrix is None:
                logger.warning(
                    "[SpinQuant] R4: random_r4=True but spinquant_R4_matrix buffer not found. "
                    "Falling back to deterministic Hadamard."
                )
                random_r4 = False

        if random_r4:
            R4 = r4_matrix.to(device=compute_device, dtype=torch.float32)

            def _make_r4_hook_matrix(R, rot_size, block_mode):
                def hook(module, args):
                    x = args[0]
                    R_local = R.to(x.device, dtype=x.dtype)
                    if block_mode:
                        shape = x.shape
                        x = x.reshape(*shape[:-1], -1, rot_size)
                        x = (x @ R_local).reshape(shape)
                    else:
                        x = x @ R_local
                    return (x,) + args[1:]

                return hook

            r4_count = 0
            for name, module in list(model.named_modules()):
                if "down_proj" in name and isinstance(module, nn.Linear):
                    hook = _make_r4_hook_matrix(R4, r4_size, need_block_rotation)
                    hook._spinquant_hook = True
                    handle = module.register_forward_pre_hook(hook)
                    handles.append(handle)
                    r4_count += 1

            logger.info(
                f"[SpinQuant] R4: Registered forward_pre_hook(rotation_size={r4_size}, "
                f"mode=random x @ R, block_rotation={need_block_rotation}) on {r4_count} down_proj layers"
            )
        else:
            # Deterministic: butterfly algorithm
            try:
                from auto_round.algorithms.transforms.spinquant.rotation_utils import (
                    get_hadamard_K,
                )

                had_K_mat, had_K_val = get_hadamard_K(r4_size)
            except ValueError:
                logger.warning(
                    f"[SpinQuant] R4: no Hadamard decomposition for r4_rotation_size={r4_size}. "
                    f"Skipping R4 rotation."
                )
                had_K_mat, had_K_val = None, None

            if had_K_mat is not None:
                had_K_mat = had_K_mat.to(device=compute_device, dtype=torch.float32)

                def _make_r4_hook_butterfly(had_mat, k_val, rot_size, block_mode):
                    def hook(module, args):
                        x = args[0]
                        if block_mode:
                            shape = x.shape
                            x = x.reshape(*shape[:-1], -1, rot_size)
                            x = matmul_hadU(x, hadamard_K=had_mat.to(x.device), K=k_val)
                            x = x.reshape(shape)
                        else:
                            x = matmul_hadU(x, hadamard_K=had_mat.to(x.device), K=k_val)
                        return (x,) + args[1:]

                    return hook

                r4_count = 0
                for name, module in list(model.named_modules()):
                    if "down_proj" in name and isinstance(module, nn.Linear):
                        hook = _make_r4_hook_butterfly(had_K_mat, had_K_val, r4_size, need_block_rotation)
                        hook._spinquant_hook = True
                        handle = module.register_forward_pre_hook(hook)
                        handles.append(handle)
                        r4_count += 1

                logger.info(
                    f"[SpinQuant] R4: Registered forward_pre_hook(rotation_size={r4_size}, "
                    f"K={had_K_val}, mode=deterministic butterfly, "
                    f"block_rotation={need_block_rotation}) on {r4_count} down_proj layers"
                )

    return handles


def remove_spinquant_hooks(handles: list[Any]) -> None:
    """Safely remove all SpinQuant hook handles and R3 monkeypatches."""
    for h in handles:
        try:
            if isinstance(h, tuple) and h[0] == "r3_monkeypatch":
                _, name, module, wrapper = h
                # The monkeypatch replaced the forward method via globals
                # patching; remove the instance-level override so it falls
                # back to the original class method.
                if "forward" in module.__dict__:
                    del module.__dict__["forward"]
                if hasattr(module, "_spinquant_r3_patched"):
                    delattr(module, "_spinquant_r3_patched")
            elif isinstance(h, tuple) and h[0] == "r3_patch":
                # Legacy: restore original forward
                _, name, module = h
                if hasattr(module, "_spinquant_original_forward"):
                    module.forward = module._spinquant_original_forward
                    delattr(module, "_spinquant_original_forward")
            else:
                # Standard hook handle (R4 forward_pre_hook, etc.)
                h.remove()
        except Exception:
            pass


def apply_spinquant_in_place(
    model: nn.Module,
    config: Any,
    dataloader: Optional[Any] = None,
) -> nn.Module:
    """Apply SpinQuant rotations to a model **in-place**.

    This is the SpinQuant equivalent of AutoRound's
    ``apply_in_place`` in ``rotation.inplace.apply``.

    Steps:
        1. Fuse RMSNorm gamma into linear weights.
        2. Optionally replace RMSNorm with TrainableRMSNorm.
        3. Initialise rotation matrices.
        4. (If trainable) run training loop.
        5. Fuse offline rotations into weights.
        6. Register online hooks (R3 / R4).

    Args:
        model: The model to modify.
        config: ``SpinQuantConfig`` instance.
        dataloader: Calibration data (required when training).

    Returns:
        The modified ``model`` (same object, mutated in-place).
    """
    from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantPreprocessor

    preprocessor = SpinQuantPreprocessor(model, config)
    return preprocessor.preprocess(dataloader)
