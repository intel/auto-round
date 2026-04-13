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

from auto_round.experimental.hadamard_inplace.model_config import (
    RotationMapping,
    _resolve,
    infer_mapping_from_model,
)
from auto_round.experimental.hadamard_inplace.utils import (
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
    random_hadamard_matrix,
)

# ---------------------------------------------------------------------------
# Low-level primitives (model-agnostic via RotationMapping)
# ---------------------------------------------------------------------------


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


def _rotate_linear_by_Q(module: torch.nn.Linear, Q: torch.Tensor, side: str, compute_device=None) -> None:
    """Apply rotation *Q* to a Linear layer's weight (and bias if present).

    Args:
        side: ``'input'``  →  W = W @ Q   (rotate input side)
              ``'output'`` →  W = Q^T @ W  (rotate output side)
        compute_device: Device to run computation on. If None, auto-detects GPU.
    """
    dtype = module.weight.data.dtype
    dev = module.weight.data.device
    cdev = _resolve_compute_device(compute_device)
    W_ = module.weight.data.to(device=cdev, dtype=torch.float64)
    Q_ = Q.to(device=cdev)
    if side == "input":
        module.weight.data = torch.matmul(W_, Q_).to(device=dev, dtype=dtype)
    else:
        module.weight.data = torch.matmul(Q_.T, W_).to(device=dev, dtype=dtype)
        if module.bias is not None:
            b = module.bias.data.to(device=cdev, dtype=torch.float64)
            module.bias.data = torch.matmul(Q_.T, b).to(device=dev, dtype=dtype)


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
    allow_online_hadamard: bool = True,
    had_dict: dict = None,
    preset: str = None,
) -> None:
    """Apply Hadamard rotation to all weights.

    Args:
        group_size: ``None`` → full Hadamard rotation.
                    ``int``  → block-diagonal rotation with this block size.
        compute_device: Device to run Hadamard computation on (e.g. ``"cuda:0"``).
            Weights are moved there temporarily and moved back afterwards.
            If ``None``, auto-detects GPU availability.
        allow_online_hadamard: If ``True`` (default), apply extra input-side
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
    head_dim = mapping.attn_head_dim or (hidden_size // num_heads)

    is_grouped = group_size is not None and group_size > 0
    desc = f"Rotating (group_size={group_size})" if is_grouped else "Rotating"

    # ----- Resolve per-operation Hadamard sources -----
    fused_fast = use_fast_had
    online_fast = False
    if preset == "random_hadamard":
        fused_fast = False

    # -- Grouped-mode matrix resolution --
    had_matrix, _found = _get_custom_had(had_dict, group_size) if is_grouped else (None, False)

    online_had_matrix = had_matrix
    if preset == "random_hadamard" and is_grouped and had_matrix is None:
        had_matrix = get_or_create_random_hadamard(group_size, compute_device)
        online_had_matrix = had_matrix
    if preset == "quarot_hadamard" and is_grouped:
        online_had_matrix = None  # force deterministic for online-paired

    # -- Helper: look up random matrix for online-paired ops in full mode --
    def _online_had(dim):
        """Return cached random matrix for *dim* under random_hadamard, else None."""
        if preset == "random_hadamard" and not is_grouped:
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

    # ---- Top-level: embedding ----
    embedding = _resolve(model, mapping.embedding)
    if is_grouped:
        _rotate_embedding_grouped(
            embedding, group_size, use_fast_had=fused_fast, compute_device=compute_device, had_matrix=had_matrix
        )
    else:
        dtype = embedding.weight.data.dtype
        dev = embedding.weight.data.device
        cdev = compute_device
        W_ = embedding.weight.data.to(device=cdev, dtype=torch.float64)
        embedding.weight.data = torch.matmul(W_, Q.to(cdev)).to(device=dev, dtype=dtype)

    if mapping.positional_embedding is not None:
        pos_emb = _resolve(model, mapping.positional_embedding)
        if is_grouped:
            _rotate_embedding_grouped(
                pos_emb, group_size, use_fast_had=fused_fast, compute_device=compute_device, had_matrix=had_matrix
            )
        else:
            pos_dtype = pos_emb.weight.data.dtype
            pos_dev = pos_emb.weight.data.device
            cdev = compute_device
            P_ = pos_emb.weight.data.to(device=cdev, dtype=torch.float64)
            pos_emb.weight.data = torch.matmul(P_, Q.to(cdev)).to(device=pos_dev, dtype=pos_dtype)

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
        # Attention inputs: Q/K/V  (fusable — residual stream)
        for attr in (mapping.attn_q, mapping.attn_k, mapping.attn_v):
            if is_grouped:
                _rotate_linear_grouped(
                    _resolve(layer, attr),
                    group_size,
                    side="input",
                    use_fast_had=fused_fast,
                    compute_device=compute_device,
                    had_matrix=had_matrix,
                )
            else:
                _rotate_linear_by_Q(_resolve(layer, attr), Q, side="input", compute_device=compute_device)

        # Attention output: o_proj  (fusable — residual stream)
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

        # MLP inputs: gate/up  (fusable — residual stream)
        for attr in mapping.mlp_in:
            if is_grouped:
                _rotate_linear_grouped(
                    _resolve(layer, attr),
                    group_size,
                    side="input",
                    use_fast_had=fused_fast,
                    compute_device=compute_device,
                    had_matrix=had_matrix,
                )
            else:
                _rotate_linear_by_Q(_resolve(layer, attr), Q, side="input", compute_device=compute_device)

        # MLP output: down_proj
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
            if allow_online_hadamard:
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
            if allow_online_hadamard:
                apply_exact_had_to_linear(
                    down_proj,
                    had_dim=-1,
                    output=False,
                    use_fast_had=online_fast,
                    compute_device=compute_device,
                    had_matrix=_online_had(intermediate_size),
                )

        # OV projection
        v_proj = _resolve(layer, mapping.attn_v)
        o_proj = _resolve(layer, mapping.attn_o)
        if is_grouped:
            pass
        elif allow_online_hadamard:
            # Full mode: per-head Had on v_proj output + Had on o_proj input.
            # For random_hadamard we decompose o_proj into within-head + cross-head
            # because a random matrix has no Kronecker structure.
            online_head_had = _online_had(head_dim)
            apply_exact_had_to_linear(
                v_proj,
                had_dim=head_dim,
                output=True,
                use_fast_had=online_fast,
                compute_device=compute_device,
                had_matrix=online_head_had,
            )
            if preset == "random_hadamard":
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
                apply_exact_had_to_linear(
                    o_proj, had_dim=-1, output=False, use_fast_had=online_fast, compute_device=compute_device
                )
        else:
            # Full mode, no online hooks — self-cancelling per-head pairs.
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
    head_dim = mapping.attn_head_dim or (hidden_size // num_heads)

    is_grouped = group_size is not None and group_size > 0

    # Online hooks always use deterministic (fixed) Hadamard — never fast_had
    # for quarot_hadamard; for "random_hadamard" they use the same random matrix
    # that was cached in had_dict by _rotate_weights.
    online_fast = False

    # -- Grouped-mode matrix resolution (must match _rotate_weights) --
    had_matrix, _ = _get_custom_had(had_dict, group_size) if is_grouped else (None, False)
    if preset == "random_hadamard" and is_grouped and had_matrix is None:
        had_matrix = get_or_create_random_hadamard(group_size)
    if preset == "quarot_hadamard" and is_grouped:
        had_matrix = None

    # -- Helper: look up random matrix for full-mode online-paired hooks --
    def _online_had(dim):
        if preset == "random_hadamard" and not is_grouped:
            return get_or_create_random_hadamard(dim)
        return None

    mlp_out_suffix = mapping.mlp_out.split(".")[-1]
    attn_o_suffix = mapping.attn_o.split(".")[-1]

    # --- Build hook factories ---
    def _make_down_proj_hook():
        if is_grouped:
            return GroupOnlineHadamardHook(
                group_size=group_size, fp32_had=fp32_had, use_fast_had=online_fast, had_matrix=had_matrix
            )
        online_mat = _online_had(intermediate_size)
        if online_mat is not None:
            return FullOnlineHadamardHook(
                had_K=None, K=None, fp32_had=fp32_had, use_fast_had=online_fast, had_matrix=online_mat
            )
        had_K, K = get_hadK(intermediate_size)
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
        if name.endswith(mlp_out_suffix) and isinstance(module, torch.nn.Linear):
            h = module.register_forward_pre_hook(_make_down_proj_hook())
            handles.append(h)
        elif not is_grouped and name.endswith(attn_o_suffix) and isinstance(module, torch.nn.Linear):
            h = module.register_forward_pre_hook(_make_o_proj_hook())
            handles.append(h)

    return handles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_hadamard_rotation(
    model,
    fp32_had: bool = False,
    group_size: int = None,
    compute_device: torch.device = None,
    allow_online_hadamard: bool = True,
    rotation_matrix: Union[str, torch.Tensor, Dict[int, torch.Tensor], None] = None,
):
    """Fuse layer norms, rotate weights, and register online Hadamard hooks.

    This is the single entry point for applying Hadamard inplace rotation.
    The model architecture is auto-detected via ``model.config.model_type``.

    Args:
        model: A HuggingFace CausalLM model (LLaMA-2/3, Qwen-3, etc.).
        fp32_had: Whether to compute the online Hadamard transform in fp32.
        group_size: If ``None`` (default), use full-dimension Hadamard rotation
            (original QuaRot).  If set to an integer (e.g. 32 or 64), use
            block-diagonal rotation where each block is ``group_size × group_size``.
        compute_device: Device to run Hadamard computation on (e.g. ``"cuda:0"``).
            When the model lives on CPU, set this to a GPU device so that the
            heavy matrix operations run on GPU; weights are temporarily moved
            there and moved back afterwards.  If ``None`` (default), auto-detects
            GPU availability (CUDA → XPU → CPU fallback).
        allow_online_hadamard: If ``True`` (default), apply the full QuaRot-style
            rotation including extra input-side Hadamard on ``down_proj`` and
            the OV pair (``v_proj`` output + ``o_proj`` input).  These extra
            rotations require compensating online Hadamard hooks at inference.
            If ``False``, only apply the residual-stream rotation (embedding,
            lm_head, Q/K/V input, o_proj output, gate/up input, down_proj
            output).  This is still a valid equivalence transform but does
            **not** need any online hooks — a fully fused mode.
        rotation_matrix: Rotation matrix selection.  Accepted forms:

            * ``None`` (default) — use the built-in butterfly Hadamard
              (``matmul_hadU`` / ``get_hadK``); full-mode Q is a random
              Hadamard matrix, grouped/online operations are deterministic.
            * ``"hadamard"`` — same as ``None``: all rotations use the
              deterministic ``get_hadK`` / ``matmul_hadU`` path.  In full mode
              the Q matrix is a deterministic Hadamard (no random sign flips).
            * ``"random_hadamard"`` — all rotations use ``random_hadamard_matrix``.
              In grouped mode a random matrix of size ``group_size`` is
              generated once and reused for every rotation (including online
              hooks).
            * ``"quarot_hadamard"`` — fusable (residual-stream) rotations use
              ``fast_hadamard_transform`` (random in full mode); non-fusable
              (online-paired) rotations **and their weight-side counterparts**
              use deterministic ``get_hadK``/``matmul_hadU`` so that the
              online hook at inference produces the exact same transform.
            * A ``torch.Tensor`` of shape ``(n, n)`` — custom rotation matrix,
              used as-is for all grouped rotations.  Only valid when
              ``group_size`` is set (``n`` should equal ``group_size``).
            * A ``dict[int, torch.Tensor]`` — mapping from dimension to
              rotation matrix.  Only valid when ``group_size`` is set.

            Raises ``ValueError`` when a Tensor/dict is provided without a
            positive ``group_size``, or when an unknown ``str`` preset is given.

    Returns:
        list of hook handles (empty when ``allow_online_hadamard=False``)."""
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
        allow_online_hadamard=allow_online_hadamard,
        had_dict=had_dict,
        preset=preset,
    )

    handles = []
    if allow_online_hadamard:
        handles = _register_online_hooks(
            model,
            mapping,
            fp32_had=fp32_had,
            use_fast_had=use_fast_had,
            group_size=group_size,
            had_dict=had_dict,
            preset=preset,
        )

    return handles


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import auto_round
    from auto_round.experimental.transform.utils.hadamard import _fetch_hadamard_divisor

    K_list = [172, 156, 140, 108, 60, 52, 36, 28, 40, 20, 12]
    for d in K_list:
        fn = getattr(auto_round.experimental.hadamard_inplace.hadamard_matrix, f"get_had{d}")
        quarot_had = fn()
        hadK = _fetch_hadamard_divisor(d, torch.float, torch.device("cpu"))
        torch.equal(quarot_had, hadK)
    print("equal test passed")
    model_name = "/models/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    apply_hadamard_rotation(model, group_size=-1, allow_online_hadamard=False, rotation_matrix="random_hadamard")
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    model_name = "/models/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    apply_hadamard_rotation(model, group_size=-1, allow_online_hadamard=False)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "/models/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    apply_hadamard_rotation(model)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    model_name = "/models/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    apply_hadamard_rotation(model)
    model.to("cuda")
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
