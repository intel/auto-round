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


# ---------------------------------------------------------------------------
# Hook implementations
# ---------------------------------------------------------------------------


class FullOnlineHadamardHook(nn.Module):
    """Pre-forward hook: full Hadamard on the entire last dimension (for ``down_proj``)."""

    def __init__(self, had_K, K, fp32_had=False, use_fast_had=True):
        super().__init__()
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

        if self.fp32_had:
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

    def __init__(self, had_K, K, head_dim, fp32_had=False, use_fast_had=True):
        """
        Args:
            had_K: Hadamard sub-matrix from ``get_hadK(num_heads)``.
            K: Block size from ``get_hadK(num_heads)``.
            head_dim: ``hidden_size // num_attention_heads``.
            fp32_had: Compute in fp32.
            use_fast_had: If True use fast_hadamard_transform; if False use matmul_hadU.
        """
        super().__init__()
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

        if self.use_fast_had:
            # Important Notice: fast_hadamard_transform does not use the same had_K,
            # it only handles the power-of-2 part (K==1). For K>1, use had_K matrix directly.
            if self.K == 1 and fast_hadamard_transform:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(-1, num_heads, self.had_dim).transpose(1, 2),
                    scale=1 / math.sqrt(num_heads),
                ).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, num_heads, self.had_dim)) / math.sqrt(num_heads)
        else:
            # Fallback: use matmul_hadU (pure butterfly, no fast_hadamard_transform)
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


def matmul_hadU_cuda(X, hadK, K, use_fast_had=True):
    n = X.shape[-1]
    if not use_fast_had:
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


def apply_exact_had_to_linear(module, had_dim=-1, output=False, use_fast_had=True):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()

    if had_dim == -1:
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
            if use_fast_had:
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
            if use_fast_had:
                n = W_.shape[1]
                W_ = fast_hadamard_transform.hadamard_transform(
                    W_.reshape(-1, n // had_dim, had_dim), scale=1 / math.sqrt(had_dim)
                ).reshape(init_shape)
            else:
                n = W_.shape[1]
                W_ = matmul_hadU(W_.reshape(-1, had_dim)).reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
