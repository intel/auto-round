# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
SpinQuant / QuaRot rotation utilities.

This module provides rotation fusion functions for SpinQuant. Where possible,
it delegates to AutoRound's ``rotation.utils.matrix`` and
``rotation.utils.math`` modules for Hadamard matrix generation and
linear-algebra helpers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hadamard matrix generation (always use our own, to avoid AutoRound
# version incompatibilities that may produce unnormalized / low-precision
# matrices).
# ---------------------------------------------------------------------------


def deterministic_hadamard_matrix(
    size: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate a normalized Sylvester Hadamard matrix (H / sqrt(N))."""
    if size <= 0 or size & (size - 1) != 0:
        raise ValueError(f"deterministic_hadamard_matrix requires power-of-2 size, got {size}")
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.size(0) < size:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(size)


def random_hadamard_matrix(
    size: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate a random Hadamard matrix: H @ diag(±1) / sqrt(N).

    Works for any size supported by :func:`get_hadamard_K` (power-of-2 or
    known non-power-of-2 sizes like 12, 20, 28, etc.).
    """
    D = torch.randint(0, 2, (size,), dtype=torch.float64, device=device) * 2 - 1
    Q = torch.diag(D)
    hadamard_K, K = get_hadamard_K(size)
    hadamard_K = hadamard_K.to(dtype=torch.float64, device=device)
    result = matmul_hadU(Q, hadamard_K=hadamard_K, K=K)
    return result.to(dtype=dtype)


def is_pow2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def get_hadamard_K(n: int) -> Tuple[torch.Tensor, int]:
    """Get the Hadamard matrix and block dimension K for a given input size.

    For power-of-2 sizes, K=1 (full Walsh-Hadamard via butterfly).
    For non-power-of-2 sizes, K is the largest known Hadamard size that
    divides n such that n/K is a power of 2.  Known sizes are sourced from
    :mod:`known_hadamard` (ported from Quark/AMD), covering sizes like 12,
    20, 28, 36, 40, 52, 60, 108, 140, 156, 172.

    Examples:
        - n=3072: 3072 = 12 × 256  →  K=12, returns 12×12 Hadamard
        - n=1024: power-of-2       →  K=1,  returns 1024×1024 Hadamard

    Returns:
        (hadamard_K, K): The K×K Hadamard matrix and the value K.
    """
    from auto_round.algorithms.transforms.spinquant.known_hadamard import (
        KNOWN_HADAMARD_MATRICES,
    )

    if is_pow2(n):
        # Sylvester construction (unnormalized) — replaces scipy.linalg.hadamard
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return H, 1
    else:
        # Search known Hadamard matrices: find K such that n % K == 0 and n/K is pow2
        for size in KNOWN_HADAMARD_MATRICES:
            if n % size == 0 and is_pow2(n // size):
                had_K = KNOWN_HADAMARD_MATRICES[size]()
                return had_K, size

        raise ValueError(
            f"Cannot find suitable Hadamard decomposition for n={n}. "
            f"Known non-pow2 sizes: {sorted(KNOWN_HADAMARD_MATRICES.keys())}. "
            f"n must be a power of 2, or n = K × 2^m for a known K."
        )


def matmul_hadU(X: torch.Tensor, hadamard_K: Optional[torch.Tensor] = None, K: Optional[int] = None) -> torch.Tensor:
    """Apply normalized Hadamard transform to the last dimension of X.

    Uses the efficient butterfly algorithm for power-of-2 dimensions,
    combined with an explicit K×K Hadamard matrix for any residual block.

    This is equivalent to X @ H where H is the normalized Hadamard matrix,
    but computed more efficiently via recursive butterfly operations.

    Based on QuaRot/Quark's implementation.

    Args:
        X: Input tensor with shape [..., n].
        hadamard_K: Optional pre-computed K×K Hadamard matrix (unnormalized).
            If None, computed automatically.
        K: Block size for the explicit Hadamard step. If None, computed automatically.

    Returns:
        Transformed tensor of same shape as X.
    """
    n = X.shape[-1]

    if hadamard_K is None or K is None:
        hadamard_K, K = get_hadamard_K(n)
        hadamard_K = hadamard_K.to(dtype=X.dtype, device=X.device)

    inp = X.clone().reshape(-1, n, 1)
    output = inp.clone()

    # Butterfly decomposition (Walsh-Hadamard for the n/K part)
    while inp.shape[1] > K:
        inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
        output = output.view(inp.shape)
        output[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
        output[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
        output = output.view(inp.shape[0], inp.shape[1], -1)
        inp, output = (output, inp)
    del output

    # Apply the K×K Hadamard block (if K > 1)
    if K > 1:
        had = hadamard_K.to(inp.device).to(inp.dtype)
        inp = had.view(1, K, K) @ inp

    # Normalize: butterfly + K-block gives unnormalized result, divide by √n
    result = inp.view(X.shape) / math.sqrt(n)

    return result


# ---------------------------------------------------------------------------
# Delegate to AutoRound's rotation utilities where available.
# If the AutoRound modules are not present (e.g. during standalone tests),
# the fallback implementations below are used.
# ---------------------------------------------------------------------------
try:
    from auto_round.algorithms.transforms.hadamard.utils.matrix import apply_transform_weight
except ImportError:
    # Fallback for standalone usage.
    def apply_transform_weight(
        transform_weight: torch.Tensor,
        value: torch.Tensor,
        location: str,
        module_type: type[nn.Module],
    ) -> torch.Tensor:
        if location == "input":
            return value @ transform_weight.T
        if location == "output":
            return transform_weight.T @ value
        raise NotImplementedError(f"apply_transform_weight: unsupported location={location!r}")


__all__ = [
    "rotate_in_channels_",
    "rotate_out_channels_",
    "fuse_rmsnorm_in_model",
    "untie_word_embeddings_if_needed",
    "deterministic_hadamard_matrix",
    "random_hadamard_matrix",
    "is_pow2",
    "get_hadamard_K",
    "matmul_hadU",
    "create_block_diag_from_head_matrix",
    "apply_hadamard_to_linear",
    "get_model_arch_info",
    "InputRotationWrapperHadamard",
]


# ---------------------------------------------------------------------------
# InputRotationWrapperHadamard — nn.Module wrapper for online rotation.
#
# Replaces a Linear layer with a wrapper that applies Hadamard rotation to
# the input activation before the linear forward pass.  Since the wrapper is
# a proper nn.Module, it is serialised by ``save_pretrained`` / ``state_dict``
# and automatically reconstructed on ``load_state_dict``.
#
# The wrapper takes ownership of the original Linear's weight and bias
# parameters (they are NOT stored as a submodule).  This prevents auto-round's
# quantization pipeline from discovering and double-wrapping the inner Linear.
#
# Matches Quark's ``InputRotationWrapperHadamard`` semantics.
# ---------------------------------------------------------------------------


class InputRotationWrapperHadamard(nn.Module):
    """nn.Module wrapper that applies Hadamard rotation to input before Linear.

    When online R1 rotation is used, target modules (q/k/v/gate/up_proj) have
    their weights pre-rotated as ``W' = W @ R``.  This wrapper applies the
    matching rotation ``x' = x @ R`` to the activation at runtime so that the
    full-precision result is preserved::

        y = (x @ R) @ (W @ R).T = x @ R @ R.T @ W.T = x @ W.T

    The wrapper stores the Hadamard matrix as a buffer (``hadamard_K``), so it
    is saved alongside the model weights and automatically restored on load.

    Weight and bias are owned directly by this module (not via a submodule),
    so ``state_dict()`` produces compatible key names (``q_proj.weight``, not
    ``q_proj.original_module.weight``).  This also ensures that auto-round's
    quantization pipeline treats this module as a leaf and doesn't try to wrap
    the inner Linear separately.

    Args:
        original_module: The ``nn.Linear`` layer (with already-rotated weights).
            Its weight and bias are *moved* into this wrapper.
        rotation_size: Size of the block rotation (= in_features for full rotation).
        hadamard_K: Pre-computed Hadamard matrix from ``get_hadamard_K()``.
            If None, computed automatically from ``rotation_size``.
        K: Hadamard block dimension (1 for power-of-2 sizes).
    """

    def __init__(
        self,
        original_module: nn.Linear,
        rotation_size: int,
        hadamard_K: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not isinstance(original_module, nn.Linear):
            raise ValueError(
                f"InputRotationWrapperHadamard only supports nn.Linear, " f"got {type(original_module).__name__}"
            )

        self._rotation_size = rotation_size
        self._in_features = original_module.in_features
        self._out_features = original_module.out_features

        # Take ownership of weight and bias from the original module.
        # The original module is NOT stored as a submodule — this prevents
        # auto-round from discovering it via named_modules() and wrapping it.
        self.weight = original_module.weight  # nn.Parameter, auto-registered
        if original_module.bias is not None:
            self.bias = original_module.bias  # nn.Parameter
        else:
            self.register_parameter("bias", None)

        if self._in_features == rotation_size:
            self._use_butterfly = True
        elif self._in_features % rotation_size == 0:
            self._use_butterfly = False
        else:
            raise ValueError(f"rotation_size={rotation_size} not compatible with " f"in_features={self._in_features}")

        # Compute / store Hadamard matrix
        if hadamard_K is None or K is None:
            hadamard_K, K = get_hadamard_K(rotation_size)

        self._K = K
        device = self.weight.device

        # Store hadamard_K as buffer (serialised with model)
        self.register_buffer(
            "hadamard_K",
            hadamard_K.to(device=device, dtype=torch.float32),
        )

        # For block rotation, pre-build the full rotation matrix
        if not self._use_butterfly:
            rot_mat = hadamard_K.to(torch.float64)
            if rot_mat.shape[0] != rotation_size:
                had_1, _ = get_hadamard_K(rotation_size // K)
                rot_mat = torch.kron(
                    hadamard_K.to(device="cpu", dtype=torch.float64),
                    had_1.to(device="cpu", dtype=torch.float64),
                )
            self.register_buffer(
                "rotation_matrix",
                rot_mat.to(device=device, dtype=torch.float32),
            )

    # --- Properties for compatibility with nn.Linear consumers ---

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    # --- Forward: rotate input, then F.linear ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_butterfly:
            x = matmul_hadU(
                x,
                hadamard_K=self.hadamard_K.to(x.device),
                K=self._K,
            )
        else:
            dtype = x.dtype
            shape = x.shape
            R = self.rotation_matrix.to(x.device, dtype=x.dtype)
            x = x.reshape(*shape[:-1], -1, self._rotation_size)
            x = (x @ R).reshape(shape).to(dtype)
        return nn.functional.linear(x, self.weight, self.bias)  # pylint: disable=not-callable

    def __repr__(self) -> str:
        return (
            f"InputRotationWrapperHadamard(\n"
            f"  in_features={self._in_features}, out_features={self._out_features},\n"
            f"  rotation_size={self._rotation_size}, "
            f"butterfly={self._use_butterfly}, K={self._K}\n"
            f")"
        )


def rotate_in_channels_(
    layer: nn.Linear,
    rotation_matrix: Optional[torch.Tensor] = None,
    R_in: Optional[torch.Tensor] = None,
    rotated_modules: Optional[set] = None,
) -> None:
    """Fuse an input-side rotation into a linear layer's weight.

    Mathematically::

        W_new = W @ R.T

    When ``R`` is smaller than the last dimension of ``W`` (block rotation),
    the rotation is applied as a block-diagonal transform: the last dimension
    is reshaped into ``(-1, rotation_size)``, ``@ R.T`` is applied, and the
    result is flattened back. This follows the same convention as Quark's
    ``rotate_with_size``.

    Args:
        layer: The ``nn.Linear`` layer whose weight will be rotated.
        rotation_matrix: Full rotation matrix ``R`` (uses ``R.T`` internally).
        R_in: Alias for ``rotation_matrix`` provided for API symmetry with
            ``rotate_out_channels_``.
        rotated_modules: Optional ``set`` used to deduplicate rotations
            when a module is shared across layers (e.g. MoE).
    """
    if rotated_modules is not None:
        if layer in rotated_modules:
            return
        rotated_modules.add(layer)

    W = layer.weight.data
    R = R_in if R_in is not None else rotation_matrix
    if R is None:
        return

    rot_size = R.shape[0]
    W_f64 = W.to(torch.float64)
    R_f64 = R.to(torch.float64)

    if W.shape[-1] == rot_size:
        # Full rotation: W_new = W @ R.T
        layer.weight.data = (W_f64 @ R_f64.T).to(W.dtype)
    elif W.shape[-1] % rot_size == 0:
        # Block rotation: reshape → rotate → flatten
        w_reshaped = W_f64.reshape(*W_f64.shape[:-1], -1, rot_size)
        w_rotated = w_reshaped @ R_f64.T
        layer.weight.data = w_rotated.reshape(W.shape).to(W.dtype)
    else:
        raise ValueError(f"rotate_in_channels_: rotation_size={rot_size} does not divide " f"weight dim={W.shape[-1]}")

    # input-side rotation does NOT affect bias:
    # y = (W @ R.T) @ (R @ x) + b = W @ x + b
    # bias unchanged.


def rotate_out_channels_(
    layer: nn.Linear,
    rotation_matrix: Optional[torch.Tensor] = None,
    R_out: Optional[torch.Tensor] = None,
    rotated_modules: Optional[set] = None,
) -> None:
    """Fuse an output-side rotation into a linear layer's weight.

    Mathematically::

        W_new = R.T @ W

    When ``R`` is smaller than the output dimension (block rotation), the
    first dimension (output channels / rows) is reshaped into
    ``(-1, rotation_size)``, transposed to apply ``@ R.T``, and reshaped back.
    This follows the same convention as Quark's ``rotate_with_size``.

    Args:
        layer: The ``nn.Linear`` layer whose weight will be rotated.
        rotation_matrix: Full rotation matrix ``R`` (uses ``R.T`` internally).
        R_out: Alias for ``rotation_matrix``.
        rotated_modules: Optional ``set`` for deduplication.
    """
    if rotated_modules is not None:
        if layer in rotated_modules:
            return
        rotated_modules.add(layer)

    W = layer.weight.data
    R = R_out if R_out is not None else rotation_matrix
    if R is None:
        return

    rot_size = R.shape[0]
    W_f64 = W.to(torch.float64)
    R_f64 = R.to(torch.float64)

    # output rotation: we need to rotate rows (output dim) of W
    # For block rotation, use W.T so rotation acts on the last dim,
    # then transpose back (same as Quark's approach)
    out_dim = W.shape[0]
    if out_dim == rot_size:
        # Full rotation: W_new = R.T @ W
        layer.weight.data = (R_f64.T @ W_f64).to(W.dtype)
    elif out_dim % rot_size == 0:
        # Block rotation on output dim: transpose, block rotate, transpose back
        WT = W_f64.T  # [in_features, out_features]
        wt_reshaped = WT.reshape(*WT.shape[:-1], -1, rot_size)
        wt_rotated = wt_reshaped @ R_f64
        layer.weight.data = wt_rotated.reshape(WT.shape).T.contiguous().to(W.dtype)
    else:
        raise ValueError(f"rotate_out_channels_: rotation_size={rot_size} does not divide " f"output dim={out_dim}")

    # Rotate bias if present
    if layer.bias is not None:
        bias_f64 = layer.bias.data.to(torch.float64)
        if bias_f64.shape[0] == rot_size:
            layer.bias.data = (R_f64.T @ bias_f64).to(layer.bias.dtype)
        elif bias_f64.shape[0] % rot_size == 0:
            b_reshaped = bias_f64.reshape(-1, rot_size)
            b_rotated = (b_reshaped @ R_f64).reshape(-1)
            layer.bias.data = b_rotated.to(layer.bias.dtype)
        else:
            raise ValueError(
                f"rotate_out_channels_: rotation_size={rot_size} does not divide " f"bias dim={bias_f64.shape[0]}"
            )


def fuse_rmsnorm_in_model(model: nn.Module) -> None:
    """
    Fuse RMSNorm / LayerNorm ``gamma`` into the subsequent linear layers.

    After fusion the norm layers become pure normalisation (``gamma == 1``).
    Uses ``float64`` intermediate computation to avoid precision loss.
    """
    # Attempt to use AutoRound's model_config layer discovery if available.
    try:
        from auto_round.algorithms.transforms.hadamard.inplace.model_config import get_scaling_layers

        layer_paths = get_scaling_layers(model.config.model_type if hasattr(model, "config") else "")
        if layer_paths:
            # Model-config-driven fusion (supports GPT-2, OPT, etc.)
            _fuse_rmsnorm_with_layer_paths(model, layer_paths)
            return
    except ImportError:
        pass

    # Fallback: hard-coded Llama-like traversal.
    _fuse_rmsnorm_llama_like(model)


def _fuse_rmsnorm_llama_like(model: nn.Module) -> None:
    """Traverse layers, supporting both model.layers and model.model.layers patterns."""
    # Try different layer container paths
    layers = None
    for path in ("model.layers", "layers", "transformer.h"):
        parts = path.split(".")
        obj = model
        for p in parts:
            if not hasattr(obj, p):
                break
            obj = getattr(obj, p)
        else:
            layers = obj
            break

    if layers is None:
        # Try recursive search
        for name, module in model.named_modules():
            if name.endswith(".layers") or name == "layers":
                if hasattr(module, "__iter__"):
                    layers = module
                    break

    if layers is None:
        return

    for layer in layers:
        # 1. input_layernorm -> q / k / v
        if hasattr(layer, "input_layernorm") and hasattr(layer.input_layernorm, "weight"):
            gamma = layer.input_layernorm.weight.data.to(torch.float64)
            if hasattr(layer, "self_attn"):
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(layer.self_attn, proj_name):
                        proj = getattr(layer.self_attn, proj_name)
                        w = proj.weight.data.to(torch.float64)
                        proj.weight.data = (w * gamma.view(1, -1)).to(proj.weight.dtype)
            layer.input_layernorm.weight.data.fill_(1.0)

        # 2. post_attention_layernorm -> gate / up
        if hasattr(layer, "post_attention_layernorm") and hasattr(layer.post_attention_layernorm, "weight"):
            gamma = layer.post_attention_layernorm.weight.data.to(torch.float64)
            if hasattr(layer, "mlp"):
                for proj_name in ("gate_proj", "up_proj"):
                    if hasattr(layer.mlp, proj_name):
                        proj = getattr(layer.mlp, proj_name)
                        w = proj.weight.data.to(torch.float64)
                        proj.weight.data = (w * gamma.view(1, -1)).to(proj.weight.dtype)
            layer.post_attention_layernorm.weight.data.fill_(1.0)

    # 3. final norm -> lm_head
    final_norm = None
    for path in ("model.norm", "norm"):
        parts = path.split(".")
        obj = model
        for p in parts:
            if not hasattr(obj, p):
                break
            obj = getattr(obj, p)
        else:
            final_norm = obj
            break

    lm_head = getattr(model, "lm_head", None)
    if final_norm is not None and lm_head is not None and hasattr(final_norm, "weight"):
        gamma = final_norm.weight.data.to(torch.float64)
        w = lm_head.weight.data.to(torch.float64)
        lm_head.weight.data = (w * gamma.view(1, -1)).to(lm_head.weight.dtype)
        final_norm.weight.data.fill_(1.0)


def _fuse_rmsnorm_with_layer_paths(model: nn.Module, layer_paths: list[str]) -> None:
    """Model-config-driven RMSNorm fusion (supports non-Llama architectures)."""
    raise NotImplementedError(
        "Model-config-driven RMSNorm fusion is not yet implemented. "
        "Use fuse_rmsnorm_into_linear() for Llama-family models, or set "
        "r1=False to skip R1 rotation for unsupported architectures."
    )


def untie_word_embeddings_if_needed(model: nn.Module) -> bool:
    """Untie input and output embeddings if they share storage.

    Also sets ``config.tie_word_embeddings = False`` so that downstream
    frameworks (e.g. lm_eval's HFLM, HuggingFace Trainer) do not re-tie
    the weights after we've separated them.
    """
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and hasattr(model, "lm_head"):
        embed = model.model.embed_tokens.weight
        head = model.lm_head.weight
        if embed.data_ptr() == head.data_ptr():
            model.lm_head.weight = nn.Parameter(head.clone())
            # Prevent frameworks from re-tying the now-separate weights
            if hasattr(model, "config"):
                model.config.tie_word_embeddings = False
            return True
    return False


def create_block_diag_from_head_matrix(R_head: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Build a block-diagonal matrix from a per-head rotation matrix."""
    return torch.block_diag(*[R_head for _ in range(num_heads)])


def apply_hadamard_to_linear(
    module: nn.Linear,
    had_dim: int = -1,
    output: bool = False,
) -> None:
    """Apply deterministic Hadamard transform to a linear layer's weight.

    This is a CPU-compatible equivalent of Quark's ``apply_exact_had_to_linear``.
    It applies a normalized Hadamard transform per chunk of size ``had_dim``.

    Args:
        module: The linear layer to modify in-place.
        had_dim: Hadamard block dimension. If -1, uses the full dimension
            (in_features for input side, out_features for output side).
        output: If True, apply to output channels (rows). If False, apply
            to input channels (columns).
    """
    # Support both nn.Linear and InputRotationWrapperHadamard
    assert isinstance(
        module, (nn.Linear, InputRotationWrapperHadamard)
    ), f"module must be nn.Linear or InputRotationWrapperHadamard, got {type(module).__name__}"
    in_features = module.in_features
    out_features = module.out_features

    W = module.weight.data
    dtype = W.dtype
    device = W.device
    W = W.to(torch.float64)

    if had_dim == -1:
        # Full Hadamard on the relevant dimension
        dim = out_features if output else in_features
        H = deterministic_hadamard_matrix(dim, dtype=torch.float64, device=device)
        if output:
            # W_new = H @ W  (rotate output channels)
            W = H @ W
        else:
            # W_new = W @ H  (rotate input channels)
            W = W @ H
    else:
        # Per-chunk Hadamard (block-diagonal application)
        H = deterministic_hadamard_matrix(had_dim, dtype=torch.float64, device=device)
        if output:
            # Apply H per had_dim chunk on the output (row) dimension
            # W shape: [out_features, in_features]
            # Reshape to [out_features // had_dim, had_dim, in_features]
            assert out_features % had_dim == 0, f"out_features={out_features} not divisible by had_dim={had_dim}"
            n_chunks = out_features // had_dim
            W_reshaped = W.reshape(n_chunks, had_dim, in_features)
            # Apply H to each chunk: H @ chunk (along dim 1)
            W_reshaped = torch.einsum("ij,kjl->kil", H, W_reshaped)
            W = W_reshaped.reshape(out_features, in_features)
        else:
            # Apply H per had_dim chunk on the input (column) dimension
            # W shape: [out_features, in_features]
            # Reshape to [out_features, in_features // had_dim, had_dim]
            assert in_features % had_dim == 0, f"in_features={in_features} not divisible by had_dim={had_dim}"
            n_chunks = in_features // had_dim
            W_reshaped = W.reshape(out_features, n_chunks, had_dim)
            # Apply H to each chunk: chunk @ H.T (along last dim)
            W_reshaped = torch.einsum("ijk,lk->ijl", W_reshaped, H)
            W = W_reshaped.reshape(out_features, in_features)

    module.weight.data = W.to(dtype)

    # Handle bias for output-side rotation
    if output and module.bias is not None:
        b = module.bias.data.to(torch.float64)
        if had_dim == -1:
            b = H @ b
        else:
            b_reshaped = b.view(n_chunks, had_dim)
            b_reshaped = torch.einsum("ij,kj->ki", H, b_reshaped)
            b = b_reshaped.view(-1)
        module.bias.data = b.to(module.bias.dtype)


def get_model_arch_info(model: nn.Module) -> dict:
    """
    Extract architecture metadata from a model.

    Returns a dict with keys: ``hidden_size``, ``head_dim``, ``num_q_heads``,
    ``num_kv_heads``, ``intermediate_size``, ``model_type``.
    """
    info: dict = {"model_type": "unknown"}
    if hasattr(model, "config"):
        cfg = model.config
        info["model_type"] = getattr(cfg, "model_type", "unknown")
        info["hidden_size"] = getattr(cfg, "hidden_size", 0)
        info["num_q_heads"] = getattr(cfg, "num_attention_heads", 0)
        info["num_kv_heads"] = getattr(cfg, "num_key_value_heads", info["num_q_heads"])
        info["intermediate_size"] = getattr(cfg, "intermediate_size", 0)
        info["head_dim"] = getattr(cfg, "head_dim", info["hidden_size"] // max(info["num_q_heads"], 1))
        if all(v for v in [info["hidden_size"], info["head_dim"], info["intermediate_size"]]):
            return info

    # Fallback: inspect model layers directly when .config is absent or incomplete
    # Try common attribute paths for embeddings / decoder layers
    embed = None
    for path in ("embed_tokens", "model.embed_tokens", "transformer.wte", "decoder.embed_tokens"):
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        else:
            embed = obj
            break

    if embed is not None:
        info["hidden_size"] = getattr(embed, "embedding_dim", getattr(embed, "weight", torch.empty(0)).shape[-1])

    # Inspect first decoder layer for attention / MLP dimensions
    first_layer = None
    for path in ("layers", "model.layers", "transformer.h", "model.decoder.layers"):
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        else:
            if hasattr(obj, "__iter__"):
                for layer in obj:
                    first_layer = layer
                    break
            break

    if first_layer is not None:
        # Attention
        attn = getattr(first_layer, "self_attn", None)
        if attn is None:
            attn = getattr(first_layer, "attn", None)
        if attn is not None:
            q_proj = getattr(attn, "q_proj", None)
            if q_proj is not None:
                out_dim = q_proj.weight.shape[0]  # hidden_size
                in_dim = q_proj.weight.shape[1]  # hidden_size
                info.setdefault("hidden_size", out_dim)
                # Infer head count from output projection if available
                o_proj = getattr(attn, "o_proj", None)
                if o_proj is not None:
                    info.setdefault("hidden_size", o_proj.weight.shape[1])

            # Try to infer head_dim from rotary_emb or k_proj
            k_proj = getattr(attn, "k_proj", None)
            if k_proj is not None:
                kv_dim = k_proj.weight.shape[0]
                # num_kv_heads = hidden_size // head_dim (approximate)
                # We'll set head_dim = hidden_size // num_q_heads if we can find it

        # MLP intermediate size
        mlp = getattr(first_layer, "mlp", None)
        if mlp is None:
            mlp = getattr(first_layer, "ffn", None)
        if mlp is not None:
            gate = getattr(mlp, "gate_proj", None)
            if gate is not None:
                info.setdefault("intermediate_size", gate.weight.shape[0])
            up = getattr(mlp, "up_proj", None)
            if up is not None:
                info.setdefault("intermediate_size", up.weight.shape[0])

    # Final defaults
    info.setdefault("hidden_size", 0)
    info.setdefault("intermediate_size", 0)
    info.setdefault("num_q_heads", 0)
    info.setdefault("num_kv_heads", info["num_q_heads"])
    info.setdefault("head_dim", info["hidden_size"] // max(info["num_q_heads"], 1))

    return info


def get_attention_layers(model: nn.Module):
    """Yield attention modules using model_config if available, else fall back."""
    try:
        from auto_round.algorithms.transforms.hadamard.inplace.model_config import get_attention_layers as _get

        return _get(model)
    except ImportError:
        pass

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                yield layer.self_attn
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for block in model.transformer.h:
            if hasattr(block, "attn"):
                yield block.attn
