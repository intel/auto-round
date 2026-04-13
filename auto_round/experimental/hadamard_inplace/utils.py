# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Online Hadamard transform hooks.

After weight rotation, down_proj and o_proj require an online Hadamard
transform on their *input activations* at inference time.  This module
provides the hooks and a helper to register them on the model.
"""

import math

import torch
import torch.nn as nn

try:
    import fast_hadamard_transform
except ImportError:
    fast_hadamard_transform = None


def _resolve_compute_device(compute_device) -> torch.device:
    """Return *compute_device* if explicitly given, otherwise auto-detect GPU.

    When ``compute_device`` is ``None`` the function checks for CUDA / XPU
    availability and returns the first accelerator it finds so that heavy
    matrix operations are offloaded to GPU even when the model weights live
    on CPU.  Falls back to ``torch.device("cpu")`` when no accelerator is
    present.
    """
    if compute_device is not None:
        return torch.device(compute_device) if not isinstance(compute_device, torch.device) else compute_device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    return torch.device("cpu")


BUILTIN_ROTATION_PRESETS = {"quarot_hadamard", "hadamard", "random_hadamard"}

# Global cache for random Hadamard matrices keyed by dimension.
# Ensures the same shape always returns the exact same random matrix within
# a process, across all calls to ``_rotate_weights`` / ``_register_online_hooks``.
_RANDOM_HADAMARD_CACHE: dict = {}


def get_or_create_random_hadamard(dim: int, device=None) -> torch.Tensor:
    """Return a random Hadamard matrix for *dim*, creating and caching it if needed.

    The matrix is cached globally in ``_RANDOM_HADAMARD_CACHE`` so that every
    caller that requests the same *dim* receives the identical matrix.
    """
    if dim in _RANDOM_HADAMARD_CACHE:
        mat = _RANDOM_HADAMARD_CACHE[dim]
        if device is not None:
            mat = mat.to(device)
        return mat
    mat = random_hadamard_matrix(dim, device or torch.device("cpu"))
    _RANDOM_HADAMARD_CACHE[dim] = mat
    return mat


def clear_random_hadamard_cache():
    """Clear the global random Hadamard matrix cache.

    Call this when you want subsequent ``random_hadamard`` preset runs to
    generate fresh random matrices (e.g. between independent experiments).
    """
    _RANDOM_HADAMARD_CACHE.clear()


def _normalize_rotation_matrix(rotation_matrix, group_size):
    """Normalize ``rotation_matrix`` into a ``(had_dict, use_fast_had, preset)`` tuple.

    Accepted inputs:
      * ``None`` → ``(None, False, None)``  — use built-in butterfly ``matmul_hadU``.
      * ``"quarot_hadamard"`` → ``(None, True, "quarot_hadamard")`` — fusable
        rotations use ``fast_hadamard_transform`` (random); non-fusable
        (online-paired) rotations use deterministic ``get_hadK``/``matmul_hadU``.
      * ``"hadamard"`` → ``(None, False, "hadamard")`` — all rotations use
        deterministic ``get_hadK``/``matmul_hadU``.
      * ``"random_hadamard"`` → ``(None, False, "random_hadamard")`` — all rotations use
        ``random_hadamard_matrix``.
      * A ``torch.Tensor`` of shape ``(n, n)`` → ``({n: tensor}, False, None)``.
      * A ``dict[int, Tensor]`` → ``(dict, False, None)`` — returned as-is.

    Returns:
        ``(had_dict, use_fast_had, preset)``

    Raises:
        ValueError: if a non-``str`` *rotation_matrix* is given but
            *group_size* is not a positive integer, or an unknown preset.
    """
    if rotation_matrix is None:
        return None, False, None

    if isinstance(rotation_matrix, str):
        if rotation_matrix not in BUILTIN_ROTATION_PRESETS:
            raise ValueError(
                f"Unknown rotation_matrix preset '{rotation_matrix}'. "
                f"Supported presets: {BUILTIN_ROTATION_PRESETS}."
            )
        if rotation_matrix == "quarot_hadamard":
            return None, True, "quarot_hadamard"
        elif rotation_matrix == "hadamard":
            return None, False, "hadamard"
        else:  # "random_hadamard"
            return None, False, "random_hadamard"

    is_grouped = group_size is not None and group_size > 0
    if not is_grouped and not isinstance(rotation_matrix, dict):
        raise ValueError(
            "rotation_matrix (Tensor/dict) can only be used with a positive group_size. "
            f"Got group_size={group_size}."
        )

    if isinstance(rotation_matrix, torch.Tensor):
        assert rotation_matrix.ndim == 2 and rotation_matrix.shape[0] == rotation_matrix.shape[1], (
            f"rotation_matrix must be square, got shape {rotation_matrix.shape}"
        )
        return {rotation_matrix.shape[0]: rotation_matrix}, False, None

    if isinstance(rotation_matrix, dict):
        for k, t in rotation_matrix.items():
            assert isinstance(t, torch.Tensor) and t.ndim == 2 and t.shape[0] == t.shape[1], (
                f"rotation_matrix[{k}] must be a square tensor, got shape {t.shape}"
            )
        return rotation_matrix, False, None

    raise TypeError(
        f"rotation_matrix must be a Tensor, dict[int, Tensor], str, or None. "
        f"Got {type(rotation_matrix)}."
    )


def _get_custom_had(had_dict, size):
    """Look up a custom Hadamard matrix for *size* from the normalized dict.

    Returns ``(had_tensor, True)`` if found, ``(None, False)`` otherwise.
    """
    if had_dict is None:
        return None, False
    if size in had_dict:
        return had_dict[size], True
    return None, False


# ---------------------------------------------------------------------------
# Hook implementations
# ---------------------------------------------------------------------------


class FullOnlineHadamardHook(nn.Module):
    """Pre-forward hook: full Hadamard on the entire last dimension (for ``down_proj``)."""

    def __init__(self, had_K, K, fp32_had=False, use_fast_had=True, had_matrix=None):
        super().__init__()
        self.custom_had = had_matrix is not None
        if had_matrix is not None:
            self.register_buffer("had_matrix", had_matrix)
            self.had_K = None
            self.K = None
        else:
            if had_K is not None:
                self.register_buffer("had_K", had_K)
            else:
                self.had_K = None
            self.K = K
        self.fp32_had = fp32_had
        self.use_fast_had = use_fast_had

    def __call__(self, module: nn.Module, args):
        x = args[0] if isinstance(args, tuple) else args
        x_dtype = x.dtype

        if self.custom_had:
            H = self.had_matrix.to(x.dtype)
            if self.fp32_had:
                H = self.had_matrix.float()
                x = (x.float() @ H.T).to(x_dtype)
            else:
                x = x @ H.T
        elif self.fp32_had:
            x = matmul_hadU_cuda(x.float(), self.had_K, self.K, use_fast_had=self.use_fast_had).to(x_dtype)
        else:
            x = matmul_hadU_cuda(x, self.had_K, self.K, use_fast_had=self.use_fast_had)

        if isinstance(args, tuple):
            return (x,) + args[1:]
        return x


class CrossHeadOnlineHadamardHook(nn.Module):
    """Pre-forward hook: **cross-head** Hadamard on the ``num_heads`` dimension
    (for ``o_proj``).

    After offline rotation:
      - ``v_proj`` absorbed a per-head (within-head) Hadamard on ``head_dim``.
      - ``o_proj`` absorbed a full Hadamard on ``hidden_size``.

    Since ``H_full = H_cross ⊗ H_within`` (Kronecker decomposition) and the
    within-head part is already cancelled by ``v_proj`` through the attention
    path (``H_within² = I``), the online hook only needs to apply the residual
    **cross-head** Hadamard (``H_cross ⊗ I``):

      * reshape ``(*, hidden_size)`` → ``(*, num_heads, head_dim)``
      * transpose → ``(*, head_dim, num_heads)``
      * Hadamard on the **num_heads** axis (last dim)
      * transpose back and reshape
    """

    def __init__(self, had_K, K, head_dim, fp32_had=False, use_fast_had=True, had_matrix=None):
        """
        Args:
            had_K: Hadamard sub-matrix from ``get_hadK(num_heads)``.
            K: Block size from ``get_hadK(num_heads)``.
            head_dim: ``hidden_size // num_attention_heads``.
            fp32_had: Compute in fp32.
            use_fast_had: If True use fast_hadamard_transform; if False use matmul_hadU.
            had_matrix: Optional custom rotation matrix of shape ``(num_heads, num_heads)``.
        """
        super().__init__()
        self.custom_had = had_matrix is not None
        if had_matrix is not None:
            self.register_buffer("had_matrix", had_matrix)
            self.had_K = None
            self.K = None
        else:
            if had_K is not None:
                self.register_buffer("had_K", had_K)
            else:
                self.had_K = None
            self.K = K
        self.had_dim = head_dim
        self.fp32_had = fp32_had
        self.use_fast_had = use_fast_had

    def __call__(self, module: nn.Module, args):
        x = args[0] if isinstance(args, tuple) else args
        x_dtype = x.dtype

        if self.fp32_had:
            x = x.float()

        init_shape = x.shape
        num_heads = init_shape[-1] // self.had_dim

        if self.custom_had:
            H = self.had_matrix.to(x.dtype)
            # reshape (*, hidden) → (*, num_heads, head_dim), transpose → (*, head_dim, num_heads)
            x = x.reshape(-1, num_heads, self.had_dim).transpose(1, 2)
            # apply H on last dim (num_heads): x @ H.T
            x = (x @ H.T).transpose(1, 2)
        elif self.use_fast_had and fast_hadamard_transform is not None and self.K == 1:
            x = fast_hadamard_transform.hadamard_transform(
                x.reshape(-1, num_heads, self.had_dim).transpose(1, 2),
                scale=1 / math.sqrt(num_heads),
            ).transpose(1, 2)
        else:
            # Fallback: use matmul_hadU (pure butterfly + had_K, no fast_hadamard_transform)
            x = x.reshape(-1, num_heads, self.had_dim).transpose(1, 2)
            x = matmul_hadU(x.contiguous())
            x = x.transpose(1, 2)

        if self.fp32_had:
            x = x.to(x_dtype)
        x = x.reshape(init_shape)

        if isinstance(args, tuple):
            return (x,) + args[1:]
        return x


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_online_had_hooks(model, mapping=None, fp32_had=False, use_fast_had=True):
    """Register online Hadamard pre-forward hooks on ``down_proj`` and ``o_proj``.

    * **down_proj** (``online_full_had``): full Hadamard on ``intermediate_size``.
      Compensates ``apply_exact_had_to_linear(down_proj, had_dim=-1, output=False)``.

    * **o_proj** (``online cross-head had``): cross-head Hadamard on ``num_heads``.
      Compensates the residual after v_proj's within-head Hadamard cancels.

    Args:
        model: A HuggingFace model whose weights have already been rotated.
        mapping: A :class:`RotationMapping` (auto-inferred if ``None``).
        fp32_had: Whether to compute the Hadamard transform in fp32.
        use_fast_had: If True use fast_hadamard_transform; if False use matmul_hadU.

    Returns:
        list of hook handles (call ``handle.remove()`` to detach).
    """
    if mapping is None:
        from auto_round.experimental.hadamard_inplace.model_config import infer_mapping_from_model

        mapping = infer_mapping_from_model(model)

    config = model.config
    num_heads = getattr(config, mapping.num_heads_attr)
    hidden_size = getattr(config, mapping.hidden_size_attr)
    intermediate_size = getattr(config, mapping.intermediate_size_attr)
    head_dim = mapping.attn_head_dim or (hidden_size // num_heads)

    # down_proj: full Hadamard on intermediate_size
    had_K_full, K_full = get_hadK(intermediate_size)

    # o_proj: cross-head Hadamard on num_heads
    had_K_head, K_head = get_hadK(num_heads)

    # Identify target module suffixes from mapping
    mlp_out_suffix = mapping.mlp_out.split(".")[-1]  # e.g. "down_proj"
    attn_o_suffix = mapping.attn_o.split(".")[-1]  # e.g. "o_proj"

    handles = []
    for name, module in model.named_modules():
        if name.endswith(mlp_out_suffix) and isinstance(module, nn.Linear):
            hook = FullOnlineHadamardHook(
                had_K=had_K_full,
                K=K_full,
                fp32_had=fp32_had,
                use_fast_had=use_fast_had,
            )
            h = module.register_forward_pre_hook(hook)
            handles.append(h)
        elif name.endswith(attn_o_suffix) and isinstance(module, nn.Linear):
            hook = CrossHeadOnlineHadamardHook(
                had_K=had_K_head,
                K=K_head,
                head_dim=head_dim,
                fp32_had=fp32_had,
                use_fast_had=use_fast_had,
            )
            h = module.register_forward_pre_hook(hook)
            handles.append(h)

    return handles


# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0


from auto_round.experimental.hadamard_inplace.hadamard_matrix import *


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py
def get_hadK(n: int, transpose=False) -> (torch.Tensor, int):
    hadK, K = None, None
    if n % 172 == 0:  # llama-2-7b up
        assert is_pow2(n // 172)
        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert is_pow2(n // 156)
        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert is_pow2(n // 140)
        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert is_pow2(n // 108)
        K = 108
        hadK = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert is_pow2(n // 60)
        K = 60
        hadK = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert is_pow2(n // 52)
        K = 52
        hadK = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        assert is_pow2(n // 36)
        K = 36
        hadK = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:
        assert is_pow2(n // 28)
        K = 28
        hadK = get_had28().T if transpose else get_had28()
    elif n % 40 == 0:
        assert is_pow2(n // 40)
        K = 40
        hadK = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        assert is_pow2(n // 20)
        K = 20
        hadK = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        assert is_pow2(n // 12)
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else:
        if is_pow2(n):
            K = 1
            return hadK, K
        else:
            from auto_round.experimental.transform.utils.hadamard import _fetch_hadamard_divisor

            hadK = _fetch_hadamard_divisor(n, torch.float, torch.device("cpu"))
            if hadK is not None:
                return hadK, 1 if is_pow2(hadK.shape[0]) else hadK.shape[0]
            assert is_pow2(n)

    return hadK, K


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        input, output = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def deterministic_hadamard_matrix(size, device):
    """Build a deterministic Hadamard matrix of the given *size*.

    Applies the butterfly ``matmul_hadU`` to an identity matrix so that the
    result is purely determined by ``get_hadK`` (no random sign flips).
    """
    Q = torch.eye(size, dtype=torch.float64)
    return matmul_hadU(Q).to(device)


def matmul_hadU_cuda(X, hadK, K, use_fast_had=True):
    n = X.shape[-1]
    if not use_fast_had or fast_hadamard_transform is None:
        return matmul_hadU(X)
    if K == 1:
        return fast_hadamard_transform.hadamard_transform(X.contiguous(), 1.0 / torch.tensor(n).sqrt())
        # if transpose:
    #     hadK = hadK.T.contiguous()
    input = X.view(*X.shape[:-1], K, n // K)
    input = fast_hadamard_transform.hadamard_transform(input.contiguous(), 1.0 / torch.tensor(n).sqrt())
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)


def matmul_hadUt_cuda(X, hadK, K, use_fast_had=True):
    return matmul_hadU_cuda(X, hadK, K, use_fast_had=use_fast_had)


def apply_exact_had_to_linear(module, had_dim=-1, output=False, use_fast_had=True, compute_device=None,
                              had_matrix=None):
    """Apply Hadamard rotation to a Linear layer's weight in-place.

    Args:
        module: ``nn.Linear`` layer.
        had_dim: Dimension of each Hadamard block (``-1`` for full dimension).
        output: If ``True`` rotate the output (row) side; otherwise input (col).
        use_fast_had: Use ``fast_hadamard_transform`` when available.
        compute_device: Device to run computation on.
        had_matrix: Optional custom rotation matrix.  When ``had_dim == -1``
            this should be a square tensor whose size equals
            ``out_features`` (output) or ``in_features`` (input).  When
            ``had_dim > 0`` the size should equal ``had_dim``.
    """
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1 and had_matrix is None:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    compute_dev = _resolve_compute_device(compute_device)
    W_ = W_.double().to(compute_dev)

    if had_matrix is not None:
        H = had_matrix.to(device=compute_dev, dtype=torch.float64)
        if had_dim == -1:
            # Full-dimension custom matrix
            if output:
                # W.T = H @ W.T  →  W = (H @ W.T).T = W @ H.T
                W_ = W_ @ H.T
            else:
                # W = H @ W  (rotate input columns: W_new[i,:] = sum H[i,j]*W[j,:])
                # Actually for input side: W_new = W @ H (each row is rotated)
                W_ = W_ @ H.T
        else:
            # Per-block custom matrix
            if output:
                W_ = W_.t()
                transposed_shape = W_.shape
                flat = W_.reshape(-1, had_dim)
                W_ = (flat @ H.T).reshape(transposed_shape).t()
            else:
                flat = W_.reshape(-1, had_dim)
                W_ = (flat @ H.T).reshape(init_shape)
    elif had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K, use_fast_had=use_fast_had).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K, use_fast_had=use_fast_had)
    else:
        # Apply Hadamard to the last had_dim chunks of the weights
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            if use_fast_had and fast_hadamard_transform is not None:
                W_ = (
                    fast_hadamard_transform.hadamard_transform(
                        W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim), scale=1 / math.sqrt(had_dim)
                    )
                    .reshape(transposed_shape)
                    .t()
                )
            else:
                W_ = matmul_hadU(W_.reshape(-1, had_dim)).reshape(transposed_shape).t()
        else:
            if use_fast_had and fast_hadamard_transform is not None:
                n = W_.shape[1]
                W_ = fast_hadamard_transform.hadamard_transform(
                    W_.reshape(-1, n // had_dim, had_dim), scale=1 / math.sqrt(had_dim)
                ).reshape(init_shape)
            else:
                W_ = matmul_hadU(W_.reshape(-1, had_dim)).reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def apply_cross_head_had_to_linear(module, num_heads, head_dim, use_fast_had=True, compute_device=None,
                                   had_matrix=None):
    """Apply a cross-head Hadamard rotation to a Linear layer's input side.

    The operation is equivalent to ``(H_cross ⊗ I_head_dim)`` applied to the
    input columns:

    * Reshape columns ``(hidden_size,)`` → ``(num_heads, head_dim)``
    * Transpose → ``(head_dim, num_heads)``
    * Hadamard on the ``num_heads`` axis
    * Transpose back and reshape

    This mirrors what :class:`CrossHeadOnlineHadamardHook` does at runtime.

    Args:
        module: ``nn.Linear`` layer whose ``in_features == num_heads * head_dim``.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension.
        use_fast_had: Use ``fast_hadamard_transform`` when available.
        compute_device: Device to run computation on.
        had_matrix: Optional custom rotation matrix of shape ``(num_heads, num_heads)``.
    """
    assert isinstance(module, torch.nn.Linear)
    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    compute_dev = _resolve_compute_device(compute_device)
    W_ = W_.double().to(compute_dev)

    out_f = W_.shape[0]
    # W shape: (out_features, hidden_size) where hidden_size = num_heads * head_dim
    # Reshape columns: (out_f, num_heads, head_dim)
    W_ = W_.reshape(out_f, num_heads, head_dim)
    # Transpose last two dims: (out_f, head_dim, num_heads)
    W_ = W_.transpose(1, 2).contiguous()

    if had_matrix is not None:
        H = had_matrix.to(device=compute_dev, dtype=torch.float64)
        # Apply H on last dim (num_heads): flat @ H.T
        flat = W_.reshape(-1, num_heads)
        W_ = (flat @ H.T).reshape(out_f, head_dim, num_heads)
    elif use_fast_had and fast_hadamard_transform is not None and is_pow2(num_heads):
        W_ = fast_hadamard_transform.hadamard_transform(
            W_, scale=1.0 / math.sqrt(num_heads)
        )
    else:
        W_ = matmul_hadU(W_.reshape(-1, num_heads)).reshape(out_f, head_dim, num_heads)

    # Transpose back: (out_f, num_heads, head_dim) → (out_f, hidden_size)
    W_ = W_.transpose(1, 2).contiguous().reshape(out_f, num_heads * head_dim)
    module.weight.data = W_.to(device=dev, dtype=dtype)


# ---------------------------------------------------------------------------
# Grouped (block-diagonal) Hadamard utilities
# ---------------------------------------------------------------------------


class GroupOnlineHadamardHook(nn.Module):
    """Pre-forward hook: block-diagonal Hadamard with fixed ``group_size`` on last dim.

    Reshapes ``(*, D)`` → ``(*, D // group_size, group_size)``, applies Hadamard
    per group, then reshapes back.  Much cheaper than a full-dimension Hadamard.
    """

    def __init__(self, group_size, fp32_had=False, use_fast_had=True, had_matrix=None):
        super().__init__()
        self.group_size = group_size
        self.fp32_had = fp32_had
        self.use_fast_had = use_fast_had
        self.custom_had = had_matrix is not None

        if had_matrix is not None:
            self.register_buffer("had_matrix", had_matrix)
            self.had_K = None
            self.K = None
        elif not is_pow2(group_size):
            had_K, K = get_hadK(group_size)
            if had_K is not None:
                self.register_buffer("had_K", had_K)
            else:
                self.had_K = None
            self.K = K
        else:
            self.had_K = None
            self.K = 1

    def __call__(self, module: nn.Module, args):
        x = args[0] if isinstance(args, tuple) else args
        x_dtype = x.dtype
        init_shape = x.shape
        gs = self.group_size

        if self.fp32_had:
            x = x.float()

        # Reshape: (*, D) → (*, D//gs, gs)
        x = x.reshape(*init_shape[:-1], init_shape[-1] // gs, gs)

        if self.custom_had:
            H = self.had_matrix.to(x.dtype)
            flat = x.reshape(-1, gs)
            x = (flat @ H.T).reshape(*init_shape[:-1], init_shape[-1] // gs, gs)
        elif self.use_fast_had and fast_hadamard_transform is not None and self.K == 1:
            x = fast_hadamard_transform.hadamard_transform(x, scale=1.0 / math.sqrt(gs))
        else:
            x = x.reshape(-1, gs)
            x = matmul_hadU(x)
            x = x.reshape(*init_shape[:-1], init_shape[-1] // gs, gs)

        x = x.reshape(init_shape)

        if self.fp32_had:
            x = x.to(x_dtype)

        if isinstance(args, tuple):
            return (x,) + args[1:]
        return x


def _apply_grouped_had_to_weight(W, group_size, side="input", use_fast_had=True, had_matrix=None):
    """Apply block-diagonal Hadamard to a weight matrix.

    Args:
        W: Weight tensor, shape (out_features, in_features).
        group_size: Block size for the Hadamard rotation.
        side: ``'input'`` rotates columns (in_features dim),
              ``'output'`` rotates rows (out_features dim).
        use_fast_had: Use fast_hadamard_transform if available.
        had_matrix: Optional custom Hadamard matrix of shape ``(gs, gs)``
            to use instead of the built-in Hadamard.

    Returns:
        Rotated weight tensor.
    """
    gs = group_size
    dtype = W.dtype
    W = W.double()

    def _had_on_last_dim(X):
        """Apply Hadamard on the last dimension (size gs) of X shaped (..., gs)."""
        if had_matrix is not None:
            H = had_matrix.to(device=X.device, dtype=X.dtype)
            # X: (..., gs) → batch matmul with H^T  →  X @ H^T
            flat = X.reshape(-1, gs)
            return (flat @ H.T).reshape(X.shape)
        if use_fast_had and fast_hadamard_transform is not None and is_pow2(gs):
            return fast_hadamard_transform.hadamard_transform(X, scale=1.0 / math.sqrt(gs))
        orig_shape = X.shape
        return matmul_hadU(X.reshape(-1, gs)).reshape(orig_shape)

    if side == "input":
        out_f, in_f = W.shape
        W = W.reshape(out_f, in_f // gs, gs)
        W = _had_on_last_dim(W)
        W = W.reshape(out_f, in_f)
    else:
        out_f, in_f = W.shape
        Wt = W.t().contiguous()
        Wt = Wt.reshape(in_f, out_f // gs, gs)
        Wt = _had_on_last_dim(Wt)
        W = Wt.reshape(in_f, out_f).t().contiguous()

    return W.to(dtype)


def _rotate_linear_grouped(module, group_size, side="input", use_fast_had=True, compute_device=None, had_matrix=None):
    """Apply block-diagonal Hadamard rotation to a Linear layer's weight.

    Args:
        module: ``nn.Linear`` layer.
        group_size: Block size.
        side: ``'input'`` or ``'output'``.
        use_fast_had: Use fast_hadamard_transform.
        compute_device: Device to run computation on. If None, auto-detects GPU.
        had_matrix: Optional custom Hadamard matrix of shape ``(gs, gs)``.
    """
    dtype = module.weight.data.dtype
    dev = module.weight.data.device
    compute_dev = _resolve_compute_device(compute_device)
    W = module.weight.data.to(device=compute_dev, dtype=torch.float64)
    W = _apply_grouped_had_to_weight(W, group_size, side=side, use_fast_had=use_fast_had, had_matrix=had_matrix)
    module.weight.data = W.to(device=dev, dtype=dtype)

    if side == "output" and module.bias is not None:
        bias = module.bias.data.to(device=compute_dev, dtype=torch.float64)
        gs = group_size
        bias = bias.reshape(-1, gs)
        if had_matrix is not None:
            H = had_matrix.to(device=compute_dev, dtype=torch.float64)
            bias = (bias @ H.T).reshape(-1)
        elif use_fast_had and fast_hadamard_transform is not None and is_pow2(gs):
            bias = fast_hadamard_transform.hadamard_transform(
                bias.unsqueeze(0), scale=1.0 / math.sqrt(gs)
            ).squeeze(0).reshape(-1)
        else:
            bias = matmul_hadU(bias).reshape(-1)
        module.bias.data = bias.to(device=dev, dtype=dtype)


def _rotate_embedding_grouped(embedding, group_size, use_fast_had=True, compute_device=None, had_matrix=None):
    """Apply block-diagonal Hadamard rotation to an Embedding layer.

    Embedding weight: (vocab, hidden_size) → rotate on hidden_size (columns).
    """
    dtype = embedding.weight.data.dtype
    dev = embedding.weight.data.device
    compute_dev = _resolve_compute_device(compute_device)
    W = embedding.weight.data.to(device=compute_dev, dtype=torch.float64)
    W = _apply_grouped_had_to_weight(W, group_size, side="input", use_fast_had=use_fast_had, had_matrix=had_matrix)
    embedding.weight.data = W.to(device=dev, dtype=dtype)


def register_online_had_hooks_grouped(model, mapping, group_size, fp32_had=False, use_fast_had=True):
    """Register per-group online Hadamard hooks on ``down_proj`` and ``o_proj``.

    In grouped mode:
      - **down_proj**: block-diagonal Hadamard on ``intermediate_size`` with ``group_size``.
      - **o_proj**: block-diagonal Hadamard on ``hidden_size`` with ``group_size``.

    Args:
        model: HuggingFace model with rotated weights.
        mapping: RotationMapping.
        group_size: Block size for block-diagonal Hadamard.
        fp32_had: Compute in fp32.
        use_fast_had: Use fast_hadamard_transform.

    Returns:
        list of hook handles.
    """
    mlp_out_suffix = mapping.mlp_out.split(".")[-1]
    attn_o_suffix = mapping.attn_o.split(".")[-1]

    handles = []
    for name, module in model.named_modules():
        if name.endswith(mlp_out_suffix) and isinstance(module, nn.Linear):
            hook = GroupOnlineHadamardHook(
                group_size=group_size, fp32_had=fp32_had, use_fast_had=use_fast_had,
            )
            h = module.register_forward_pre_hook(hook)
            handles.append(h)
        elif name.endswith(attn_o_suffix) and isinstance(module, nn.Linear):
            hook = GroupOnlineHadamardHook(
                group_size=group_size, fp32_had=fp32_had, use_fast_had=use_fast_had,
            )
            h = module.register_forward_pre_hook(hook)
            handles.append(h)

    return handles

