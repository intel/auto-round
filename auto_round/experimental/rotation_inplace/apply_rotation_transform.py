# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Hadamard inplace rotation — public API and rotation primitives.

Supports LLaMA-2, LLaMA-3, Qwen-3 (and any model with the same layout).
The entry point is :func:`apply_hadamard_rotation`.
"""

import gc
import typing
from typing import Dict, Union

import torch
import tqdm

from auto_round.experimental.rotation_inplace.model_config import (
    MAPPING_REGISTRY,
    RotationMapping,
    _resolve,
    infer_mapping_from_model,
)
from auto_round.experimental.rotation_inplace.special_model_handler import apply_special_overrides
from auto_round.experimental.rotation_inplace.utils import (
    CrossHeadOnlineHadamardHook,
    FullOnlineHadamardHook,
    GroupOnlineHadamardHook,
    _get_custom_had,
    _normalize_rotation_matrix,
    _resolve_compute_device,
    _rotate_embedding_grouped,
    _rotate_linear_grouped,
    apply_cross_head_had_to_linear,
    apply_exact_had_to_linear,
    deterministic_hadamard_matrix,
    get_hadK,
    get_or_create_random_hadamard,
)

# ---------------------------------------------------------------------------
# Low-level primitives (model-agnostic via RotationMapping)
# ---------------------------------------------------------------------------


def _resolve_head_dim(mapping, config, hidden_size, num_heads):
    """Resolve the per-head attention dimension.

    Resolution order:
      1. ``mapping.attn_head_dim`` (explicit override on the RotationMapping).
      2. ``config.head_dim`` if present (Qwen-3 and other models declare an
         explicit ``head_dim`` that does not necessarily equal
         ``hidden_size // num_heads``; e.g. Qwen3-32B has hidden=5120,
         heads=64, head_dim=128 → o_proj.in_features = 8192, not 5120).
      3. ``hidden_size // num_heads`` as a last-resort default.
    """
    if mapping.attn_head_dim:
        return mapping.attn_head_dim
    cfg_head_dim = getattr(config, "head_dim", None)
    if isinstance(cfg_head_dim, int) and cfg_head_dim > 0:
        return cfg_head_dim
    return hidden_size // num_heads


def _fuse_ln_linear(
    layernorm: torch.nn.Module,
    linear_layers: typing.Iterable[torch.nn.Linear],
) -> None:
    """Fuse the linear operations in LayerNorm into adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        dev = linear.weight.device

        W_ = linear.weight.data.double()
        ln_weight = layernorm.weight.double().to(dev)
        linear.weight.data = (W_ * ln_weight).to(linear_dtype)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64, device=dev))
            ln_bias = layernorm.bias.double().to(dev)
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, ln_bias)
            linear.bias.data = linear.bias.data.to(linear_dtype)


def _reset_ln_params(layernorm: torch.nn.Module) -> None:
    """Reset LayerNorm to identity: weight=1, bias=0."""
    layernorm.weight.data.fill_(1.0)
    if hasattr(layernorm, "bias") and layernorm.bias is not None:
        layernorm.bias.data.fill_(0.0)


def _rotate_weight_chunked(
    weight: torch.Tensor,
    Q: torch.Tensor,
    side: str,
    compute_device,
    chunk: int = 4096,
) -> torch.Tensor:
    """Compute the rotated weight without ever materialising the full fp64 copy.

    * ``side == 'input'``  → returns ``W @ Q``  (chunked over rows of ``W``).
    * ``side == 'output'`` → returns ``Q^T @ W`` (chunked over columns of ``W``).

    The output is pre-allocated in the **original** dtype on the **original**
    device of ``weight``. At any moment only a single chunk lives in fp64 on
    ``compute_device``, so peak transient memory is roughly
    ``chunk * other_dim * 8`` bytes instead of ``W.numel() * 8``.

    Embedding/lm_head on Qwen3-14B (151936 × 5120) drops from ~12 GB to a few
    hundred MB transient.
    """
    dtype = weight.dtype
    dev = weight.device
    out = torch.empty_like(weight)
    Q_ = Q.to(device=compute_device, dtype=torch.float64)
    try:
        if side == "input":
            # (R, C) @ (C, C) → (R, C); chunk over R.
            R = weight.shape[0]
            for i in range(0, R, chunk):
                j = min(i + chunk, R)
                blk = weight.data[i:j].to(device=compute_device, dtype=torch.float64)
                rotated = (blk @ Q_).to(device=dev, dtype=dtype)
                out[i:j].copy_(rotated)
                del blk, rotated
        elif side == "output":
            # Q^T @ (R, C) → (R, C); chunk over C so each block is (R, chunk).
            C = weight.shape[1]
            Q_T = Q_.T.contiguous()
            for i in range(0, C, chunk):
                j = min(i + chunk, C)
                blk = weight.data[:, i:j].to(device=compute_device, dtype=torch.float64)
                rotated = (Q_T @ blk).to(device=dev, dtype=dtype)
                out[:, i:j].copy_(rotated)
                del blk, rotated
            del Q_T
        else:
            raise ValueError(f"side must be 'input' or 'output', got {side!r}")
    finally:
        del Q_
    return out


def _rotate_linear_by_Q(module: torch.nn.Linear, Q: torch.Tensor, side: str, compute_device=None) -> None:
    """Apply rotation *Q* to a Linear layer's weight (and bias if present).

    Memory-efficient: never materialises the full fp64 weight at once.

    Args:
        side: ``'input'``  →  W = W @ Q   (rotate input side)
              ``'output'`` →  W = Q^T @ W  (rotate output side)
        compute_device: Device to run computation on. If None, auto-detects GPU.
    """
    cdev = _resolve_compute_device(compute_device)
    module.weight.data = _rotate_weight_chunked(module.weight.data, Q, side, cdev)
    if side == "output" and module.bias is not None:
        dtype = module.bias.data.dtype
        dev = module.bias.data.device
        # Bias is a 1-D vector → small; safe to do in one shot.
        b = module.bias.data.to(device=cdev, dtype=torch.float64)
        Q_ = Q.to(device=cdev, dtype=torch.float64)
        new_b = torch.matmul(Q_.T, b).to(device=dev, dtype=dtype)
        del b, Q_
        module.bias.data = new_b


def _untie_word_embeddings(model, mapping: RotationMapping) -> None:
    """Break tied weights between lm_head and embedding if they share the same tensor."""
    embedding = _resolve(model, mapping.embedding)
    lm_head = _resolve(model, mapping.lm_head)

    if lm_head.weight.data_ptr() != embedding.weight.data_ptr():
        return

    lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


def _uses_layernorm_with_mean(model, mapping: RotationMapping) -> bool:
    """Check whether the model uses standard LayerNorm (which subtracts mean)."""
    layers = _resolve(model, mapping.layers_attr)
    first_ln = _resolve(layers[0], mapping.attn_input_ln)
    return isinstance(first_ln, torch.nn.LayerNorm)


def _bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """Subtract column-wise mean from a Linear layer's weight (and mean from bias)."""
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ - W_.mean(dim=-2, keepdim=True)).to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = (b_ - b_.mean()).to(linear_dtype)


def _subtract_embedding_mean(model, mapping: RotationMapping) -> None:
    """Subtract per-row mean from the embedding weight matrix."""
    W = _resolve(model, mapping.embedding)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(dtype=dtype)

    if mapping.positional_embedding is not None:
        P = _resolve(model, mapping.positional_embedding)
        p_dtype = P.weight.data.dtype
        P_ = P.weight.data.to(dtype=torch.float64)
        P.weight.data = (P_ - P_.mean(dim=-1, keepdim=True)).to(dtype=p_dtype)


class _RMSNorm(torch.nn.Module):
    """RMS Normalization (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _replace_layernorms_with_rmsnorm(model) -> None:
    """Replace all ``nn.LayerNorm`` modules with ``_RMSNorm``."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = _resolve(model, parts[0])
            attr = parts[1]
        else:
            parent = model
            attr = parts[0]
        rms = _RMSNorm(module.normalized_shape[0], eps=module.eps)
        rms = rms.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr, rms)


# ---------------------------------------------------------------------------
# High-level steps driven by RotationMapping
# ---------------------------------------------------------------------------


def _fuse_layer_norms(model, mapping: RotationMapping) -> None:
    """Fuse all LayerNorm parameters into adjacent Linear layers."""
    layers = _resolve(model, mapping.layers_attr)

    for layer in layers:
        mlp_ln = _resolve(layer, mapping.mlp_input_ln)
        mlp_linears = [_resolve(layer, p) for p in mapping.mlp_in]
        _fuse_ln_linear(mlp_ln, mlp_linears)
        _reset_ln_params(mlp_ln)

        attn_ln = _resolve(layer, mapping.attn_input_ln)
        attn_linears = [
            _resolve(layer, mapping.attn_q),
            _resolve(layer, mapping.attn_k),
            _resolve(layer, mapping.attn_v),
        ]
        _fuse_ln_linear(attn_ln, attn_linears)
        _reset_ln_params(attn_ln)

    pre_head_ln = _resolve(model, mapping.pre_head_ln)
    lm_head = _resolve(model, mapping.lm_head)
    _fuse_ln_linear(pre_head_ln, [lm_head])
    _reset_ln_params(pre_head_ln)


# ---------------------------------------------------------------------------
# Unified weight rotation (full or grouped)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _rotate_weights(
    model,
    mapping: RotationMapping,
    use_fast_had: bool = True,
    group_size: int = None,
    compute_device: torch.device = None,
    had_dict: dict = None,
    preset: str = None,
    fuse_online_to_weight: bool = True,
) -> None:
    """Apply Hadamard rotation to all weights.

    Args:
        group_size: ``None`` → full Hadamard rotation.
                    ``int``  → block-diagonal rotation with this block size.
        compute_device: Device to run Hadamard computation on (e.g. ``"cuda:0"``).
            Weights are moved there temporarily and moved back afterwards.
            If ``None``, auto-detects GPU availability.
        allow_online_rotation: If ``True`` (default), apply extra input-side
            Hadamard rotations on ``down_proj`` and the OV pair (``v_proj``
            output + ``o_proj`` input) that require compensating online hooks
            at inference time.  If ``False``, skip those extra rotations so
            that **no** online hooks are needed.
        had_dict: Normalized ``dict[int, Tensor]`` of custom Hadamard matrices
            (keyed by dimension).  Only used in grouped mode.
        preset: Rotation preset name (``"quarot_hadamard"``, ``"hadamard"``,
            ``"random_hadamard"``, or ``None``).

            * ``"quarot_hadamard"``: fusable (residual-stream) rotations use
              ``fast_hadamard_transform`` / random Hadamard; non-fusable
              (online-paired) rotations and their weight-side counterparts use
              deterministic ``get_hadK``/``matmul_hadU`` so that the online
              hook at inference produces the exact same transform.
            * ``"hadamard"``: all rotations use deterministic ``get_hadK`` /
              ``matmul_hadU``.  Full-mode Q is a deterministic Hadamard matrix.
            * ``"random_hadamard"``: all rotations use random Hadamard matrices
              from the global cache (``get_or_create_random_hadamard``).
              Same dimension → same matrix everywhere.
            * ``None``: same behaviour as ``"hadamard"`` (built-in butterfly).
    """
    compute_device = _resolve_compute_device(compute_device)
    config = model.config
    hidden_size = getattr(config, mapping.hidden_size_attr)
    intermediate_size = getattr(config, mapping.intermediate_size_attr)
    num_heads = getattr(config, mapping.num_heads_attr)
    head_dim = _resolve_head_dim(mapping, config, hidden_size, num_heads)

    is_grouped = group_size is not None and group_size > 0
    desc = f"Rotating (group_size={group_size})" if is_grouped else "Rotating"

    # ----- Resolve per-operation Hadamard sources -----
    fused_fast = use_fast_had
    online_fast = False
    if preset == "random_hadamard":
        fused_fast = False

    # -- Matrix resolution --
    had_matrix, _found = _get_custom_had(had_dict, group_size) if is_grouped else (None, False)

    online_had_matrix = had_matrix
    if preset == "random_hadamard" and had_matrix is None:
        had_matrix = get_or_create_random_hadamard(group_size if is_grouped else hidden_size, compute_device)
        online_had_matrix = had_matrix
    if preset == "quarot_hadamard" and is_grouped:
        online_had_matrix = None  # force deterministic for online-paired

    # -- Helper: look up cached random matrix for online-paired ops --
    def _online_had(dim):
        """Return cached random matrix for *dim* under random_hadamard, else None."""
        if preset == "random_hadamard":
            return get_or_create_random_hadamard(dim, compute_device)
        return None

    if is_grouped:
        assert hidden_size % group_size == 0, f"group_size={group_size} must divide hidden_size={hidden_size}"
        assert (
            intermediate_size % group_size == 0
        ), f"group_size={group_size} must divide intermediate_size={intermediate_size}"

    # --- Full mode: build Hadamard matrix Q ---
    Q = None
    if not is_grouped:
        if preset == "hadamard":
            Q = deterministic_hadamard_matrix(hidden_size, compute_device)
        else:
            # "random_hadamard", "quarot_hadamard", None — same shape → same matrix
            Q = get_or_create_random_hadamard(hidden_size, compute_device)

    # ---- Top-level: embedding / lm_head ----
    # When fuse_online_to_weight=False, skip embedding and lm_head rotation:
    # each layer is self-contained (weight rotation + online hook cancel out).
    if fuse_online_to_weight:
        embedding = _resolve(model, mapping.embedding)
        if is_grouped:
            _rotate_embedding_grouped(
                embedding, group_size, use_fast_had=fused_fast, compute_device=compute_device, had_matrix=had_matrix
            )
        else:
            # Chunked: avoids a full fp64 copy of the (vocab, hidden) embedding,
            # which on Qwen3-14B is ~6 GB on its own.
            embedding.weight.data = _rotate_weight_chunked(
                embedding.weight.data, Q, side="input", compute_device=compute_device
            )

        if mapping.positional_embedding is not None:
            pos_emb = _resolve(model, mapping.positional_embedding)
            if is_grouped:
                _rotate_embedding_grouped(
                    pos_emb, group_size, use_fast_had=fused_fast, compute_device=compute_device, had_matrix=had_matrix
                )
            else:
                pos_emb.weight.data = _rotate_weight_chunked(
                    pos_emb.weight.data, Q, side="input", compute_device=compute_device
                )

        # ---- Top-level: lm_head ----
        lm_head = _resolve(model, mapping.lm_head)
        if is_grouped:
            _rotate_linear_grouped(
                lm_head,
                group_size,
                side="input",
                use_fast_had=fused_fast,
                compute_device=compute_device,
                had_matrix=had_matrix,
            )
        else:
            _rotate_linear_by_Q(lm_head, Q, side="input", compute_device=compute_device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Per-layer rotation ----
    layers = _resolve(model, mapping.layers_attr)
    for layer in tqdm.tqdm(layers, unit="layer", desc=desc):
        if fuse_online_to_weight:
            # ---- fuse mode: QuaRot-style residual stream rotation ----
            # Q/K/V: only residual Q on input (no online Had stacking, no hook).
            #   When Q == online Had (e.g. preset="hadamard"), Q @ Q = I cancels
            #   the rotation entirely, destroying quantization benefit.
            # gate/up: only residual Q on input (no online Had stacking, no hook).
            # down_proj: residual Q^T on output + online Had on input (+ hook).
            # v_proj/o_proj: per-head/cross-head Had below (+ hook on o_proj).
            for attr in (mapping.attn_q, mapping.attn_k, mapping.attn_v):
                mod = _resolve(layer, attr)
                if is_grouped:
                    _rotate_linear_grouped(
                        mod,
                        group_size,
                        side="input",
                        use_fast_had=fused_fast,
                        compute_device=compute_device,
                        had_matrix=had_matrix,
                    )
                else:
                    _rotate_linear_by_Q(mod, Q, side="input", compute_device=compute_device)

            # o_proj: residual stream output rotation
            if is_grouped:
                _rotate_linear_grouped(
                    _resolve(layer, mapping.attn_o),
                    group_size,
                    side="output",
                    use_fast_had=fused_fast,
                    compute_device=compute_device,
                    had_matrix=had_matrix,
                )
            else:
                _rotate_linear_by_Q(_resolve(layer, mapping.attn_o), Q, side="output", compute_device=compute_device)

            # gate/up: only residual Q on input
            for attr in mapping.mlp_in:
                mod = _resolve(layer, attr)
                if is_grouped:
                    _rotate_linear_grouped(
                        mod,
                        group_size,
                        side="input",
                        use_fast_had=fused_fast,
                        compute_device=compute_device,
                        had_matrix=had_matrix,
                    )
                else:
                    _rotate_linear_by_Q(mod, Q, side="input", compute_device=compute_device)

            # down_proj: residual output + online input Had
            down_proj = _resolve(layer, mapping.mlp_out)
            if is_grouped:
                _rotate_linear_grouped(
                    down_proj,
                    group_size,
                    side="output",
                    use_fast_had=fused_fast,
                    compute_device=compute_device,
                    had_matrix=had_matrix,
                )
                _rotate_linear_grouped(
                    down_proj,
                    group_size,
                    side="input",
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=online_had_matrix,
                )
            else:
                _rotate_linear_by_Q(down_proj, Q, side="output", compute_device=compute_device)
                apply_exact_had_to_linear(
                    down_proj,
                    had_dim=-1,
                    output=False,
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=_online_had(intermediate_size),
                )

            # OV projection: v_proj per-head output + o_proj decomposed input
            #
            # The online hook on o_proj applies  (H_cross ⊗ I_head)⁻¹  at
            # runtime, so the weight-side rotation must equal exactly
            # (H_cross ⊗ I_head)(I_heads ⊗ H_head) = H_cross ⊗ H_head.
            #
            # IMPORTANT: we must NOT use a single full-dimension Hadamard
            # (``had_dim=-1``) on o_proj, because the butterfly construction
            # ``matmul_hadU(hidden_size)`` does NOT satisfy the Kronecker
            # decomposition ``H_hidden = H_num_heads ⊗ H_head_dim`` when
            # ``num_heads`` is not a power of 2 (e.g. Qwen3-14B, num_heads=40).
            # Instead we always apply per-head + cross-head separately.
            v_proj = _resolve(layer, mapping.attn_v)
            o_proj = _resolve(layer, mapping.attn_o)
            if is_grouped:
                pass
            else:
                online_head_had = _online_had(head_dim)
                apply_exact_had_to_linear(
                    v_proj,
                    had_dim=head_dim,
                    output=True,
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=online_head_had,
                )
                apply_exact_had_to_linear(
                    o_proj,
                    had_dim=head_dim,
                    output=False,
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=online_head_had,
                )
                apply_cross_head_had_to_linear(
                    o_proj,
                    num_heads,
                    head_dim,
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=_online_had(num_heads),
                )

        else:
            # ---- unfused mode: no residual rotation, only input-side Had ----
            # Each layer gets Had fused on input side + compensating hook → equivalent.
            # No embedding/lm_head rotation. No self-cancelling pair.
            # v_proj treated same as Q/K (input Had only, no per-head/cross-head).

            # Q/K/V: input-side Had on hidden_size
            for attr in (mapping.attn_q, mapping.attn_k, mapping.attn_v):
                mod = _resolve(layer, attr)
                if is_grouped:
                    _rotate_linear_grouped(
                        mod,
                        group_size,
                        side="input",
                        use_fast_had=online_fast,
                        compute_device=compute_device,
                        had_matrix=online_had_matrix,
                    )
                else:
                    apply_exact_had_to_linear(
                        mod,
                        had_dim=-1,
                        output=False,
                        use_fast_had=online_fast,
                        compute_device=compute_device,
                        had_matrix=_online_had(hidden_size),
                    )

            # o_proj: input-side Had on hidden_size (full Had, not cross-head)
            o_proj = _resolve(layer, mapping.attn_o)
            if is_grouped:
                _rotate_linear_grouped(
                    o_proj,
                    group_size,
                    side="input",
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=online_had_matrix,
                )
            else:
                apply_exact_had_to_linear(
                    o_proj,
                    had_dim=-1,
                    output=False,
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=_online_had(hidden_size),
                )

            # gate/up: input-side Had on hidden_size
            for attr in mapping.mlp_in:
                mod = _resolve(layer, attr)
                if is_grouped:
                    _rotate_linear_grouped(
                        mod,
                        group_size,
                        side="input",
                        use_fast_had=online_fast,
                        compute_device=compute_device,
                        had_matrix=online_had_matrix,
                    )
                else:
                    apply_exact_had_to_linear(
                        mod,
                        had_dim=-1,
                        output=False,
                        use_fast_had=online_fast,
                        compute_device=compute_device,
                        had_matrix=_online_had(hidden_size),
                    )

            # down_proj: input-side Had on intermediate_size
            down_proj = _resolve(layer, mapping.mlp_out)
            if is_grouped:
                _rotate_linear_grouped(
                    down_proj,
                    group_size,
                    side="input",
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=online_had_matrix,
                )
            else:
                apply_exact_had_to_linear(
                    down_proj,
                    had_dim=-1,
                    output=False,
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=_online_had(intermediate_size),
                )

        # Per-layer cleanup: drop fp64 temporaries and CUDA caching allocator
        # blocks so peak memory stays at ~1 layer's worth instead of accumulating
        # across all 32+ decoder layers (was the main cause of 33 GB RAM on 8B).
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Unified online hook registration
# ---------------------------------------------------------------------------


def _register_online_hooks(
    model,
    mapping: RotationMapping,
    fp32_had: bool = False,
    use_fast_had: bool = True,
    group_size: int = None,
    had_dict: dict = None,
    preset: str = None,
    fuse_online_to_weight: bool = True,
):
    """Register online Hadamard pre-forward hooks on ``down_proj`` and ``o_proj``.

    Online hooks must use the **same** Hadamard matrix that was applied to the
    weight-side counterpart during ``_rotate_weights``.  For ``quarot_hadamard``
    this is always the deterministic ``get_hadK``/``matmul_hadU`` path
    (``use_fast_had=False``).  For ``"random_hadamard"`` it is the random matrix that
    was generated once and stored in ``had_dict``.

    Args:
        group_size: ``None`` → full Hadamard hooks (original QuaRot).
                    ``int``  → per-group Hadamard hooks.
        had_dict: Normalized ``dict[int, Tensor]`` of custom Hadamard matrices.
        preset: Rotation preset name.
    Returns:
        list of hook handles.
    """
    config = model.config
    num_heads = getattr(config, mapping.num_heads_attr)
    hidden_size = getattr(config, mapping.hidden_size_attr)
    intermediate_size = getattr(config, mapping.intermediate_size_attr)
    head_dim = _resolve_head_dim(mapping, config, hidden_size, num_heads)

    is_grouped = group_size is not None and group_size > 0

    # Online hooks always use deterministic (fixed) Hadamard — never fast_had
    # for quarot_hadamard; for "random_hadamard" they use the same random matrix
    # that was cached in had_dict by _rotate_weights.
    online_fast = False

    # -- Matrix resolution (must match the *online-paired* matrix used by
    # _rotate_weights for down_proj input / OV pair). Variable name kept in
    # sync with _rotate_weights to make any future drift obvious.
    online_had_matrix, _ = _get_custom_had(had_dict, group_size) if is_grouped else (None, False)
    if preset == "random_hadamard" and online_had_matrix is None:
        online_had_matrix = get_or_create_random_hadamard(group_size if is_grouped else hidden_size)
    if preset == "quarot_hadamard" and is_grouped:
        online_had_matrix = None

    # -- Helper: look up cached random matrix for online-paired hooks --
    def _online_had(dim):
        if preset == "random_hadamard":
            return get_or_create_random_hadamard(dim)
        return None

    mlp_out_suffix = mapping.mlp_out.split(".")[-1]
    attn_o_suffix = mapping.attn_o.split(".")[-1]

    # Suffixes for Q/K/V and gate/up (for online input Had hooks)
    attn_qkv_suffixes = set(attr.split(".")[-1] for attr in (mapping.attn_q, mapping.attn_k, mapping.attn_v))
    mlp_in_suffixes = set(attr.split(".")[-1] for attr in mapping.mlp_in)

    # --- Build hook factories ---
    def _make_down_proj_hook():
        if is_grouped:
            return GroupOnlineHadamardHook(
                group_size=group_size, fp32_had=fp32_had, use_fast_had=online_fast, had_matrix=online_had_matrix
            )
        online_mat = _online_had(intermediate_size)
        if online_mat is not None:
            return FullOnlineHadamardHook(
                had_K=None, K=None, fp32_had=fp32_had, use_fast_had=online_fast, had_matrix=online_mat
            )
        had_K, K = get_hadK(intermediate_size)
        return FullOnlineHadamardHook(had_K=had_K, K=K, fp32_had=fp32_had, use_fast_had=online_fast)

    def _make_hidden_had_hook():
        """Full Had hook on hidden_size (for Q/K/V and gate/up input)."""
        if is_grouped:
            return GroupOnlineHadamardHook(
                group_size=group_size, fp32_had=fp32_had, use_fast_had=online_fast, had_matrix=online_had_matrix
            )
        online_mat = _online_had(hidden_size)
        if online_mat is not None:
            return FullOnlineHadamardHook(
                had_K=None, K=None, fp32_had=fp32_had, use_fast_had=online_fast, had_matrix=online_mat
            )
        had_K, K = get_hadK(hidden_size)
        return FullOnlineHadamardHook(had_K=had_K, K=K, fp32_had=fp32_had, use_fast_had=online_fast)

    def _make_o_proj_hook():
        online_mat = _online_had(num_heads)
        if online_mat is not None:
            return CrossHeadOnlineHadamardHook(
                had_K=None,
                K=None,
                head_dim=head_dim,
                fp32_had=fp32_had,
                use_fast_had=online_fast,
                had_matrix=online_mat,
            )
        had_K, K = get_hadK(num_heads)
        return CrossHeadOnlineHadamardHook(
            had_K=had_K,
            K=K,
            head_dim=head_dim,
            fp32_had=fp32_had,
            use_fast_had=online_fast,
        )

    # --- Register ---
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.split(".")[-1]

        if name.endswith(mlp_out_suffix):
            # down_proj: full Had on intermediate_size input
            h = module.register_forward_pre_hook(_make_down_proj_hook())
            handles.append(h)
        elif name.endswith(attn_o_suffix):
            if fuse_online_to_weight and not is_grouped:
                # o_proj: cross-head Had on input (fused mode, full only)
                h = module.register_forward_pre_hook(_make_o_proj_hook())
                handles.append(h)
            elif not fuse_online_to_weight:
                # o_proj: full Had on hidden_size input (unfused mode, matches weight rotation)
                h = module.register_forward_pre_hook(_make_hidden_had_hook())
                handles.append(h)
        elif suffix in attn_qkv_suffixes:
            if not fuse_online_to_weight:
                # Q/K/V: full Had on hidden_size input (unfused mode only).
                # In fused mode Q/K/V only have residual Q on weight (no online Had),
                # and activations come pre-rotated from residual stream → no hook needed.
                h = module.register_forward_pre_hook(_make_hidden_had_hook())
                handles.append(h)
        elif suffix in mlp_in_suffixes:
            if not fuse_online_to_weight:
                # gate/up: full Had on hidden_size input (unfused mode only).
                # Same reasoning as Q/K/V above.
                h = module.register_forward_pre_hook(_make_hidden_had_hook())
                handles.append(h)

    return handles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_rotation_transform(
    model,
    group_size: int = None,
    allow_online_rotation: bool = True,
    rotation_matrix: Union[str, torch.Tensor, Dict[int, torch.Tensor], None] = None,
    compute_device: torch.device | str = None,
    fp32_had: bool = False,
    fuse_online_to_weight: bool = None,
):
    """Fuse layer norms, rotate weights, and register online Hadamard hooks.

    This is the single entry point for applying Hadamard inplace rotation.
    The model architecture is auto-detected via ``model.config.model_type``.

    Args:
        model: A HuggingFace CausalLM model (LLaMA-2/3, Qwen-3, etc.).
        fp32_had: Whether to compute the online Hadamard transform in fp32.
        group_size: If ``None`` (default), use full-dimension Hadamard rotation.
        compute_device: Device to run Hadamard computation on.
        allow_online_rotation: If ``True`` (default), apply online Hadamard
            rotations on ``down_proj`` input and the OV pair.
        rotation_matrix: Rotation matrix selection (``"hadamard"``,
            ``"random_hadamard"``, ``"quarot_hadamard"``, Tensor, dict, or None).
        fuse_online_to_weight: If ``True`` (default), fuse online Hadamard
            rotation into weights (down_proj input, v_proj output, o_proj input)
            and register compensating online hooks.  If ``False``, skip
            embedding/lm_head rotation; each linear layer is self-contained
            with input-side Had on weight + compensating online hook on
            activation.  No v_proj cross-head or inner-head rotation.

    Returns:
        list of hook handles."""
    if fuse_online_to_weight is None:
        if model.config.model_type in MAPPING_REGISTRY or model.__class__.__name__ in MAPPING_REGISTRY:
            fuse_online_to_weight = True
        else:
            fuse_online_to_weight = False

    # ---- Model-specific overrides ----
    # Some models require a specific rotation configuration to preserve
    # accuracy or to run correctly. The mapping lives in
    # ``special_model_handler.SPECIAL_MODEL_REGISTRY`` so that adding a new
    # special-cased model is a one-liner there instead of a code change here.
    _override_kwargs = {
        "rotation_matrix": rotation_matrix,
        "fuse_online_to_weight": fuse_online_to_weight,
        "group_size": group_size,
        "allow_online_rotation": allow_online_rotation,
    }
    apply_special_overrides(model, _override_kwargs)
    rotation_matrix = _override_kwargs["rotation_matrix"]
    fuse_online_to_weight = _override_kwargs["fuse_online_to_weight"]
    group_size = _override_kwargs["group_size"]
    allow_online_rotation = _override_kwargs["allow_online_rotation"]

    had_dict, use_fast_had, preset = _normalize_rotation_matrix(rotation_matrix, group_size)
    compute_device = _resolve_compute_device(compute_device)

    if use_fast_had:
        from auto_round.utils import logger

        try:
            import fast_hadamard_transform  # noqa: F401

            if group_size is None:
                logger.warning(
                    "fast_hadamard_transform uses a different Hadamard matrix than the "
                    "default implementation. Please ensure consistency between training "
                    "and inference. This will be refined later."
                )
        except ImportError:
            logger.warning("Importing fast_hadamard_transform failed, falling back to default implementation.")
            use_fast_had = False

    mapping = infer_mapping_from_model(model)

    _untie_word_embeddings(model, mapping)

    if _uses_layernorm_with_mean(model, mapping):
        _subtract_embedding_mean(model, mapping)

    _fuse_layer_norms(model, mapping)

    if _uses_layernorm_with_mean(model, mapping):
        layers = _resolve(model, mapping.layers_attr)
        for layer in layers:
            _bake_mean_into_linear(_resolve(layer, mapping.attn_o))
            _bake_mean_into_linear(_resolve(layer, mapping.mlp_out))
        _replace_layernorms_with_rmsnorm(model)

    _rotate_weights(
        model,
        mapping,
        use_fast_had=use_fast_had,
        group_size=group_size,
        compute_device=compute_device,
        had_dict=had_dict,
        preset=preset,
        fuse_online_to_weight=fuse_online_to_weight,
    )

    handles = []
    if fuse_online_to_weight or allow_online_rotation:
        handles = _register_online_hooks(
            model,
            mapping,
            fp32_had=fp32_had,
            use_fast_had=use_fast_had,
            group_size=group_size,
            had_dict=had_dict,
            preset=preset,
            fuse_online_to_weight=fuse_online_to_weight,
        )

    return model, handles


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "/models/Qwen3-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    apply_rotation_transform(
        model, group_size=128, allow_online_rotation=True, rotation_matrix="hadamard", fuse_online_to_weight=True
    )
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
    #
    # model_name = "/models/Qwen3-8B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    # apply_rotation_transform(model, group_size=-1, allow_online_rotation=True, fuse_online_to_weight=True)
    # model.to("cuda")
    # text = "There is a girl who likes adventure,"
    # inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
    #
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    #
    # model_name = "/models/Meta-Llama-3.1-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    # apply_rotation_transform(model, fuse_online_to_weight=True, group_size=32)
    # model.to("cuda")
    # text = "There is a girl who likes adventure,"
    # inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
    #
    # model_name = "/models/Llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    # apply_hadamard_rotation(model)
    # model.to("cuda")
    # text = "There is a girl who likes adventure,"
    # inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
