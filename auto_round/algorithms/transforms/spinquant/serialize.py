# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SpinQuant / QuaRot rotation serialization for quantized models.

This module handles saving and loading online rotation state so that
rotated + quantized models can be saved with ``save_pretrained()`` and
loaded back for inference without manual hook re-registration.

Strategy (Plan A + C):
- **R1/R4 online rotations**: Stored as buffers on QuantLinear modules.
  At inference, QuantLinear.forward() detects these buffers and applies
  the rotation before the quantized matmul.
- **R3 online rotation**: Stored in ``config.json`` only (deterministic
  Hadamard, reconstructible from head_dim). Rebuilt via monkeypatch after
  model loading.
- **R1/R2 offline (fused)**: No serialization needed — already in weights.

Buffer storage format:
- Deterministic Hadamard: ``int8`` (±1 values) + rotation_size metadata
- Random Hadamard: ``int8`` (±1 values, full matrix after random sign flip)
- Trained orthogonal: ``float32`` (full matrix, cannot be compressed)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    get_hadamard_K,
    matmul_hadU,
)

if TYPE_CHECKING:
    from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

logger = logging.getLogger("autoround.spinquant.serialize")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Buffer name prefixes (avoid collisions with other rotation implementations)
_R1_PREFIX = "spinquant_r1"
_R4_PREFIX = "spinquant_r4"

# Rotation type flags stored as scalar tensors
ROTATION_TYPE_HADAMARD = 0  # Deterministic, reconstruct from size
ROTATION_TYPE_RANDOM = 1  # Random Hadamard, stored as int8 (±1)
ROTATION_TYPE_TRAINED = 2  # Trained orthogonal, stored as float32


# --------------------------------------------------------------------------
# Save-side: inject rotation buffers into QuantLinear before save
# --------------------------------------------------------------------------


def inject_spinquant_buffers(
    model: nn.Module,
    config: "SpinQuantConfig",
) -> int:
    """Inject SpinQuant rotation buffers into QuantLinear modules for serialization.

    Must be called AFTER quantization/packing (QuantLinear modules exist) and
    BEFORE ``model.save_pretrained()``.

    Args:
        model: Quantized model with QuantLinear modules.
        config: SpinQuantConfig used during preprocessing.

    Returns:
        Number of QuantLinear modules that received rotation buffers.
    """
    from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

    n_injected = 0

    # Determine rotation parameters
    hidden_size = _get_hidden_size(model)
    head_dim = _get_head_dim(model)
    intermediate_size = _get_intermediate_size(model)

    r1_size = config.rotation_size or hidden_size
    r4_size = config.rotation_size or intermediate_size

    # Online R1 targets: q_proj, k_proj, v_proj, gate_proj, up_proj
    if config.r1 and config.online_r1_rotation:
        r1_targets = _get_online_r1_target_names(model)
        for name, module in model.named_modules():
            if not _is_quantlinear(module):
                continue
            if name in r1_targets:
                _inject_rotation_buffers(
                    module,
                    _R1_PREFIX,
                    r1_size,
                    config.random_r1,
                    is_trained=False,
                    rotation_matrix=_get_stored_rotation(model, "spinquant_R1"),
                )
                n_injected += 1

    # Online R4 targets: down_proj
    if config.r4:
        r4_targets = _get_r4_target_names(model)
        for name, module in model.named_modules():
            if not _is_quantlinear(module):
                continue
            if name in r4_targets:
                _inject_rotation_buffers(
                    module,
                    _R4_PREFIX,
                    r4_size,
                    random=config.random_r4,
                    is_trained=False,
                    rotation_matrix=_get_stored_rotation(model, "spinquant_R4_matrix") if config.random_r4 else None,
                )
                n_injected += 1

    logger.info(f"[SpinQuant Serialize] Injected rotation buffers into " f"{n_injected} QuantLinear modules")
    return n_injected


def save_spinquant_config(
    model: nn.Module,
    save_dir: str,
    config: "SpinQuantConfig",
) -> None:
    """Save SpinQuant config into the model's config.json for load-time reconstruction.

    This enables R3 monkeypatch reconstruction and general rotation metadata
    persistence. Called after model.save_pretrained().

    Args:
        model: The model (used to extract architecture info).
        save_dir: Directory where config.json was saved.
        config: SpinQuantConfig to persist.
    """
    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        logger.warning(f"[SpinQuant Serialize] config.json not found at {save_dir}, " f"cannot save spinquant_config")
        return

    with open(config_path, "r") as f:
        model_config = json.load(f)

    # Build serializable spinquant config
    spinquant_dict = _config_to_serializable(config, model)
    spinquant_dict["algorithm"] = "spinquant"

    # Store under quantization_config if it exists, else at top level.
    # Key is "spinquant_config" (not "rotation_config" — that key is
    # already used by the Hadamard rotation system with a different schema).
    if "quantization_config" in model_config:
        qcfg = model_config["quantization_config"]
        qcfg["spinquant_config"] = spinquant_dict
    else:
        model_config["spinquant_config"] = spinquant_dict

    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    logger.info(f"[SpinQuant Serialize] Saved spinquant_config to {config_path}")


# --------------------------------------------------------------------------
# Load-side: pre-register empty buffers before weight loading
# --------------------------------------------------------------------------


def preregister_spinquant_buffers(
    model: nn.Module,
    spinquant_config: dict,
) -> int:
    """Pre-register empty SpinQuant buffers on QuantLinear modules.

    Must be called AFTER ``convert_hf_model()`` replaces Linear → QuantLinear
    but BEFORE HuggingFace loads the state_dict from safetensors.
    This ensures the spinquant buffers in the checkpoint are not silently
    dropped by ``load_state_dict(strict=False)``.

    Args:
        model: Model with QuantLinear modules (post convert_hf_model).
        spinquant_config: Dict from config.json's spinquant_config.

    Returns:
        Number of modules that received pre-registered buffers.
    """
    if not isinstance(spinquant_config, dict):
        return 0

    r1_online = spinquant_config.get("online_r1_rotation", True)
    r1_enabled = spinquant_config.get("r1", True)
    r4_enabled = spinquant_config.get("r4", False)
    random_r1 = spinquant_config.get("random_r1", False)
    trained = spinquant_config.get("trainable_rotation", False)
    rotation_size_cfg = spinquant_config.get("rotation_size")

    hidden_size = spinquant_config.get("hidden_size", _get_hidden_size(model))
    intermediate_size = spinquant_config.get("intermediate_size", _get_intermediate_size(model))

    r1_size = rotation_size_cfg or hidden_size
    r4_size = rotation_size_cfg or intermediate_size

    # Determine which type of matrix storage
    if trained:
        needs_matrix = True
        matrix_dtype = torch.float32
    elif random_r1:
        needs_matrix = True
        matrix_dtype = torch.int8
    else:
        needs_matrix = False
        matrix_dtype = torch.int8

    n_registered = 0

    r1_targets = _get_online_r1_target_names(model) if (r1_enabled and r1_online) else set()
    r4_targets = _get_r4_target_names(model) if r4_enabled else set()

    for name, module in model.named_modules():
        if not _is_quantlinear(module):
            continue

        if name in r1_targets:
            _preregister_buffers_on_module(module, _R1_PREFIX, r1_size, needs_matrix, matrix_dtype)
            n_registered += 1

        if name in r4_targets:
            random_r4 = spinquant_config.get("random_r4", False)
            r4_needs_matrix = random_r4 or trained
            _preregister_buffers_on_module(
                module,
                _R4_PREFIX,
                r4_size,
                needs_matrix=r4_needs_matrix,
                matrix_dtype=torch.float32 if trained else torch.int8,
            )
            n_registered += 1

    if n_registered > 0:
        buf_types = []
        if r1_targets:
            buf_types.append(f"R1({len(r1_targets)} modules)")
        if r4_targets:
            buf_types.append(f"R4({len(r4_targets)} modules)")
        logger.info(
            f"[SpinQuant] Pre-registered empty buffers on "
            f"{n_registered} QuantLinear modules for state_dict loading "
            f"[{', '.join(buf_types)}]"
        )
    return n_registered


def _preregister_buffers_on_module(
    module: nn.Module,
    prefix: str,
    rotation_size: int,
    needs_matrix: bool,
    matrix_dtype: torch.dtype,
) -> None:
    """Register empty spinquant buffers on a module for state_dict loading."""
    # Use CPU: at pre-registration time QuantLinear params may be on meta
    # device; HF's state_dict loader will overwrite these with real values.
    device = torch.device("cpu")
    module.register_buffer(
        f"{prefix}_type",
        torch.tensor(0, dtype=torch.int32, device=device),
    )
    module.register_buffer(
        f"{prefix}_size",
        torch.tensor(0, dtype=torch.int32, device=device),
    )
    if needs_matrix:
        module.register_buffer(
            f"{prefix}_matrix",
            torch.zeros(
                rotation_size,
                rotation_size,
                dtype=matrix_dtype,
                device=device,
            ),
        )


# --------------------------------------------------------------------------
# Load-side: reconstruct online rotations from buffers and config
# --------------------------------------------------------------------------


def rebuild_spinquant_online(
    model: nn.Module,
    config: Optional["SpinQuantConfig"] = None,
) -> nn.Module:
    """Rebuild online SpinQuant rotations after loading a quantized model.

    Handles:
    - R1/R4: QuantLinear forward is patched to apply rotation from buffers.
    - R3: If config indicates R3 was enabled, re-applies monkeypatch.

    Args:
        model: Loaded quantized model (QuantLinear modules have spinquant buffers).
        config: SpinQuantConfig. If None, attempts to read from model.config.

    Returns:
        Model with online rotations restored.
    """
    if config is None:
        config = _load_config_from_model(model)
        if config is None:
            logger.warning("[SpinQuant] No spinquant_config found on model. " "Cannot rebuild online rotations.")
            return model

    # Patch QuantLinear forward for R1/R4 buffer-based rotation
    n_patched = _patch_quantlinear_forward_spinquant(model)

    # Rebuild R3 monkeypatch if enabled
    if config.r3:
        head_dim = _get_head_dim(model)
        if head_dim > 0:
            # Build a minimal config with R3=True, R4=False for hook rebuild.
            # R1/R4 are handled by QuantLinear buffer-based forward patching.
            from types import SimpleNamespace

            from auto_round.algorithms.transforms.spinquant.inplace.apply import (
                register_spinquant_hooks,
            )

            r3_config = SimpleNamespace(
                r3=True,
                r4=False,
                random_r3=getattr(config, "random_r3", False),
                random_r4=False,
                head_dim=head_dim,
                intermediate_size=0,
            )
            register_spinquant_hooks(
                model,
                config=r3_config,
                head_dim=head_dim,
                r4_rotation_size=0,
            )
            logger.info(f"[SpinQuant] Rebuilt R3 monkeypatch (head_dim={head_dim})")

    # Build descriptive summary of which rotations are active
    active = []
    if config.r1:
        r1_mode = "online" if getattr(config, "online_r1_rotation", True) else "offline"
        active.append(f"R1({r1_mode})")
    if getattr(config, "r2", False):
        active.append("R2(offline, fused into weights)")
    if config.r3:
        active.append("R3(hook)")
    if config.r4:
        active.append("R4(buffer)")
    active_str = ", ".join(active) if active else "none"

    logger.info(
        f"[SpinQuant] Rebuilt online rotations: " f"{n_patched} QuantLinear patched, " f"active rotations: {active_str}"
    )
    return model


# --------------------------------------------------------------------------
# QuantLinear forward patching
# --------------------------------------------------------------------------

_QUANTLINEAR_PATCHED = False


def _patch_quantlinear_forward_spinquant(model: nn.Module) -> int:
    """Patch QuantLinear.forward() to apply spinquant rotation from buffers.

    This is a class-level patch applied once. Each QuantLinear instance checks
    at forward time whether it has spinquant buffers.

    Returns:
        Number of QuantLinear modules with spinquant buffers found.
    """
    global _QUANTLINEAR_PATCHED

    n_with_buffers = 0
    quantlinear_classes = set()

    for _, module in model.named_modules():
        if _is_quantlinear(module) and _has_spinquant_buffers(module):
            n_with_buffers += 1
            quantlinear_classes.add(type(module))

    if n_with_buffers == 0:
        return 0

    # Patch each QuantLinear class variant found
    for cls in quantlinear_classes:
        if getattr(cls, "_spinquant_forward_patched", False):
            continue
        _monkey_patch_forward(cls)
        cls._spinquant_forward_patched = True

    return n_with_buffers


def _monkey_patch_forward(cls: type) -> None:
    """Replace cls.forward with a version that applies spinquant rotation."""
    original_forward = cls.forward

    def forward_with_spinquant(self, x):
        # Apply R1 rotation if buffer present
        if hasattr(self, f"{_R1_PREFIX}_type"):
            x = _apply_rotation_from_buffer(self, x, _R1_PREFIX)

        # Apply R4 rotation if buffer present
        if hasattr(self, f"{_R4_PREFIX}_type"):
            x = _apply_rotation_from_buffer(self, x, _R4_PREFIX)

        return original_forward(self, x)

    cls.forward = forward_with_spinquant


def _apply_rotation_from_buffer(
    module: nn.Module,
    x: torch.Tensor,
    prefix: str,
) -> torch.Tensor:
    """Apply rotation using stored buffers on a QuantLinear module.

    Args:
        module: QuantLinear with spinquant buffers.
        x: Input tensor [..., in_features].
        prefix: Buffer name prefix ("spinquant_r1" or "spinquant_r4").

    Returns:
        Rotated tensor.
    """
    rot_type = int(getattr(module, f"{prefix}_type"))
    rot_size = int(getattr(module, f"{prefix}_size"))
    in_features = x.shape[-1]

    if rot_type == ROTATION_TYPE_HADAMARD:
        # Deterministic Hadamard — reconstruct from size
        # Cache the decomposition on the module for efficiency
        cache_key = f"_cached_{prefix}_had"
        if not hasattr(module, cache_key):
            had_K, K = get_hadamard_K(rot_size)
            setattr(module, cache_key, (had_K, K))
        had_K, K = getattr(module, cache_key)
        had_K = had_K.to(x.device)

        if rot_size == in_features:
            x = matmul_hadU(x, hadamard_K=had_K, K=K)
        else:
            # Block rotation
            x = _apply_block_rotation_butterfly(x, had_K, K, rot_size)

    elif rot_type in (ROTATION_TYPE_RANDOM, ROTATION_TYPE_TRAINED):
        # Full matrix stored in buffer
        matrix = getattr(module, f"{prefix}_matrix")
        R = matrix.to(x.device, dtype=x.dtype)

        if rot_type == ROTATION_TYPE_RANDOM:
            # int8 ±1 matrix, normalize
            R = R.float() / math.sqrt(rot_size)
            R = R.to(x.dtype)

        if rot_size == in_features:
            x = x @ R
        else:
            # Block rotation
            shape = x.shape
            x = x.reshape(*shape[:-1], -1, rot_size)
            x = (x @ R).reshape(shape)

    return x


def _apply_block_rotation_butterfly(
    x: torch.Tensor,
    had_K: torch.Tensor,
    K: int,
    rot_size: int,
) -> torch.Tensor:
    """Apply block-wise Hadamard rotation using butterfly or direct matmul."""
    in_features = x.shape[-1]
    if in_features == rot_size:
        return matmul_hadU(x, hadamard_K=had_K, K=K)

    # Block-wise: reshape and apply per block
    shape = x.shape
    dtype = x.dtype

    # Build full rotation matrix for the block
    R = had_K.to(torch.float64)
    if R.shape[0] != rot_size:
        had_1, _ = get_hadamard_K(rot_size // K)
        R = torch.kron(
            had_K.to(device="cpu", dtype=torch.float64),
            had_1.to(device="cpu", dtype=torch.float64),
        )
    R = (R / math.sqrt(rot_size)).to(device=x.device, dtype=dtype)

    x = x.reshape(*shape[:-1], -1, rot_size)
    x = (x @ R).reshape(shape)
    return x


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _inject_rotation_buffers(
    module: nn.Module,
    prefix: str,
    rotation_size: int,
    random: bool,
    is_trained: bool,
    rotation_matrix: Optional[torch.Tensor] = None,
) -> None:
    """Register rotation buffers on a QuantLinear module.

    Args:
        module: Target QuantLinear.
        prefix: Buffer name prefix.
        rotation_size: Size of the rotation block.
        random: Whether this is a random Hadamard (needs full matrix storage).
        is_trained: Whether this is a trained orthogonal matrix.
        rotation_matrix: Pre-computed rotation matrix (for random/trained).
    """
    # Always use CPU for rotation metadata buffers.  After packing,
    # QuantLinear parameters may live on the ``meta`` device and
    # ``save_pretrained`` silently skips meta tensors.  CPU ensures the
    # buffers are persisted to safetensors.
    device = torch.device("cpu")

    # Determine storage type
    if is_trained:
        rot_type = ROTATION_TYPE_TRAINED
    elif random:
        rot_type = ROTATION_TYPE_RANDOM
    else:
        rot_type = ROTATION_TYPE_HADAMARD

    # Always store: type and size
    module.register_buffer(
        f"{prefix}_type",
        torch.tensor(rot_type, dtype=torch.int32, device=device),
    )
    module.register_buffer(
        f"{prefix}_size",
        torch.tensor(rotation_size, dtype=torch.int32, device=device),
    )

    # Store matrix for random/trained types
    if rot_type == ROTATION_TYPE_RANDOM:
        if rotation_matrix is None:
            logger.warning(
                f"Random rotation ({prefix}) requires rotation_matrix but got None. "
                "This usually means the matrix was deleted during cleanup. "
                "Falling back to deterministic Hadamard — the saved model may be INCORRECT."
            )
            had_K, K = get_hadamard_K(rotation_size)
            if had_K.shape[0] != rotation_size:
                had_1, _ = get_hadamard_K(rotation_size // K)
                rotation_matrix = torch.kron(had_K, had_1)
            else:
                rotation_matrix = had_K
        # Store as int8 (±1 values only)
        module.register_buffer(
            f"{prefix}_matrix",
            rotation_matrix.sign().to(torch.int8).to(device),
        )

    elif rot_type == ROTATION_TYPE_TRAINED:
        if rotation_matrix is None:
            raise ValueError("Trained rotation requires rotation_matrix to be provided")
        # Store as float32 (arbitrary values)
        module.register_buffer(
            f"{prefix}_matrix",
            rotation_matrix.to(torch.float32).to(device),
        )
    # For HADAMARD type: no matrix needed (reconstructed from size at runtime)


def _is_quantlinear(module: nn.Module) -> bool:
    """Check if a module is any variant of quantized linear layer.

    Covers:
    - QuantLinear (INT W4A16/W3A16/W8A16 from export.py / qlinear_int.py / qlinear_fp.py)
    - NVFP4QuantLinear, MXFP4QuantLinear, MXFP8QuantLinear, MXINT4QuantLinear (from qmodules)
    - WeightFP8ActFP8StaticQuantLinear (FP8 static)
    - Any class with 'QuantLinear' in its name or inheriting QModuleBase
    """
    cls_name = type(module).__name__
    # Direct name match for standard QuantLinear
    if cls_name == "QuantLinear":
        return True
    # Match FP-family quantized linears (NVFP4QuantLinear, MXFP4QuantLinear, etc.)
    if "QuantLinear" in cls_name:
        return True
    # Match QModuleBase subclasses (FP8/MXFP/NVFP inference modules)
    for base in type(module).__mro__:
        if base.__name__ == "QModuleBase":
            return True
    return False


def _has_spinquant_buffers(module: nn.Module) -> bool:
    """Check if a module has any spinquant rotation buffers."""
    return hasattr(module, f"{_R1_PREFIX}_type") or hasattr(module, f"{_R4_PREFIX}_type")


def _get_online_r1_target_names(model: nn.Module) -> set:
    """Get full qualified names of modules that should have online R1 rotation.

    These are: q_proj, k_proj, v_proj, gate_proj, up_proj
    (the modules whose input channels were rotated during preprocessing).
    """
    targets = set()
    r1_proj_names = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")

    for name, module in model.named_modules():
        # Match by suffix (handles both attn.q_proj and self_attn.q_proj patterns)
        parts = name.split(".")
        if parts and parts[-1] in r1_proj_names:
            targets.add(name)

    return targets


def _get_r4_target_names(model: nn.Module) -> set:
    """Get full qualified names of modules that should have R4 rotation.

    R4 is applied to the input of down_proj.
    """
    targets = set()
    for name, module in model.named_modules():
        parts = name.split(".")
        if parts and parts[-1] == "down_proj":
            targets.add(name)
    return targets


def _get_stored_rotation(model: nn.Module, param_name: str) -> Optional[torch.Tensor]:
    """Get a stored rotation matrix/parameter from the model.

    During preprocessing, rotation matrices are stored as model-level
    parameters (e.g., model.spinquant_R1).
    """
    if hasattr(model, param_name):
        param = getattr(model, param_name)
        if isinstance(param, (torch.Tensor, nn.Parameter)):
            return param.data
    return None


def _get_hidden_size(model: nn.Module) -> int:
    """Extract hidden_size from model config."""
    if hasattr(model, "config"):
        return getattr(model.config, "hidden_size", 0)
    return 0


def _get_head_dim(model: nn.Module) -> int:
    """Extract head_dim from model config."""
    if hasattr(model, "config"):
        cfg = model.config
        if hasattr(cfg, "head_dim"):
            return cfg.head_dim
        if hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads"):
            return cfg.hidden_size // cfg.num_attention_heads
    return 0


def _get_intermediate_size(model: nn.Module) -> int:
    """Extract intermediate_size from model config."""
    if hasattr(model, "config"):
        return getattr(model.config, "intermediate_size", 0)
    return 0


def _config_to_serializable(config: "SpinQuantConfig", model: nn.Module) -> dict:
    """Convert SpinQuantConfig to a JSON-serializable dict with model info."""
    from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

    # Get relevant fields from config
    result = {
        "r1": config.r1,
        "r2": config.r2,
        "r3": config.r3,
        "r4": config.r4,
        "online_r1_rotation": config.online_r1_rotation,
        "rotation_size": config.rotation_size,
        "random_r1": config.random_r1,
        "random_r2": config.random_r2,
        "random_r3": config.random_r3,
        "random_r4": config.random_r4,
        "trainable_rotation": config.trainable_rotation,
        # Model architecture info for reconstruction
        "head_dim": _get_head_dim(model),
        "hidden_size": _get_hidden_size(model),
        "intermediate_size": _get_intermediate_size(model),
    }
    return result


def _load_config_from_model(
    model: nn.Module,
) -> Optional["SpinQuantConfig"]:
    """Try to load SpinQuantConfig from model.config."""
    from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

    if not hasattr(model, "config"):
        return None

    cfg = model.config
    spinquant_dict = None

    # Check in quantization_config (may be a dict or QuantizationConfigMixin)
    if hasattr(cfg, "quantization_config"):
        qcfg = cfg.quantization_config
        if isinstance(qcfg, dict):
            spinquant_dict = qcfg.get("spinquant_config")
        else:
            # QuantizationConfigMixin or similar object
            spinquant_dict = getattr(qcfg, "spinquant_config", None)

    # Check at top level
    if spinquant_dict is None and hasattr(cfg, "spinquant_config"):
        spinquant_dict = cfg.spinquant_config

    if spinquant_dict is None:
        return None

    # Reconstruct SpinQuantConfig from dict
    try:
        return SpinQuantConfig(
            r1=spinquant_dict.get("r1", True),
            r2=spinquant_dict.get("r2", True),
            r3=spinquant_dict.get("r3", False),
            r4=spinquant_dict.get("r4", False),
            online_r1_rotation=spinquant_dict.get("online_r1_rotation", True),
            rotation_size=spinquant_dict.get("rotation_size"),
            random_r1=spinquant_dict.get("random_r1", False),
            random_r2=spinquant_dict.get("random_r2", False),
            random_r3=spinquant_dict.get("random_r3", False),
            random_r4=spinquant_dict.get("random_r4", False),
            trainable_rotation=spinquant_dict.get("trainable_rotation", False),
        )
    except Exception as e:
        logger.warning(f"[SpinQuant] Failed to parse spinquant_config: {e}")
        return None
