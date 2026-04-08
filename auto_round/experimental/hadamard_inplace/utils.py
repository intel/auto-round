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

from auto_round.experimental.hadamard_inplace.hadamard import get_hadK, matmul_hadU_cuda


# ---------------------------------------------------------------------------
# Hook implementations
# ---------------------------------------------------------------------------

class FullOnlineHadamardHook(nn.Module):
    """Pre-forward hook: full Hadamard on the entire last dimension (for ``down_proj``)."""

    def __init__(self, had_K, K, fp32_had=False):
        super().__init__()
        if had_K is not None:
            self.register_buffer("had_K", had_K)
        else:
            self.had_K = None
        self.K = K
        self.fp32_had = fp32_had

    def __call__(self, module: nn.Module, args):
        x = args[0] if isinstance(args, tuple) else args
        x_dtype = x.dtype

        if self.fp32_had:
            x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
        else:
            x = matmul_hadU_cuda(x, self.had_K, self.K)

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

    def __init__(self, had_K, K, head_dim, fp32_had=False):
        """
        Args:
            had_K: Hadamard sub-matrix from ``get_hadK(num_heads)``.
            K: Block size from ``get_hadK(num_heads)``.
            head_dim: ``hidden_size // num_attention_heads``.
            fp32_had: Compute in fp32.
        """
        super().__init__()
        if had_K is not None:
            self.register_buffer("had_K", had_K)
        else:
            self.had_K = None
        self.K = K
        self.had_dim = head_dim
        self.fp32_had = fp32_had

    def __call__(self, module: nn.Module, args):
        x = args[0] if isinstance(args, tuple) else args
        x_dtype = x.dtype

        if self.fp32_had:
            x = x.float()

        init_shape = x.shape
        # Important Notice: fast_hadamard_transform does not use the same had_K,
        # it only handles the power-of-2 part (K==1). For K>1, use had_K matrix directly.
        if self.K == 1 and fast_hadamard_transform:
            x = fast_hadamard_transform.hadamard_transform(
                x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                scale=1 / math.sqrt(init_shape[-1] // self.had_dim)).transpose(1, 2)
        else:
            x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(
                init_shape[-1] // self.had_dim)

        if self.fp32_had:
            x = x.to(x_dtype)
        x = x.reshape(init_shape)


        if isinstance(args, tuple):
            return (x,) + args[1:]
        return x


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_online_had_hooks(model, fp32_had=False):
    """Register online Hadamard pre-forward hooks on ``down_proj`` and ``o_proj``.

    * **down_proj** (``online_full_had``): full Hadamard on ``intermediate_size``.
      Compensates ``apply_exact_had_to_linear(down_proj, had_dim=-1, output=False)``.

    * **o_proj** (``online per-head had``): Hadamard **within each head** on
      ``head_dim``.  Compensates
      ``apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)``.

    Args:
        model: A HuggingFace model whose weights have already been rotated.
        fp32_had: Whether to compute the Hadamard transform in fp32.

    Returns:
        list of hook handles (call ``handle.remove()`` to detach).
    """
    config = model.config
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    # down_proj: full Hadamard on intermediate_size
    had_K_full, K_full = get_hadK(intermediate_size)

    # o_proj: per-head Hadamard on head_dim (within each head)
    had_K_head, K_head = get_hadK(num_heads)

    handles = []
    for name, module in model.named_modules():
        if name.endswith("down_proj") and isinstance(module, nn.Linear):
            hook = FullOnlineHadamardHook(
                had_K=had_K_full, K=K_full, fp32_had=fp32_had,
            )
            h = module.register_forward_pre_hook(hook)
            handles.append(h)
        elif name.endswith("o_proj") and isinstance(module, nn.Linear):
            hook = CrossHeadOnlineHadamardHook(
                had_K=had_K_head, K=K_head,
                head_dim=head_dim, fp32_had=fp32_had,
            )
            h = module.register_forward_pre_hook(hook)
            handles.append(h)

    return handles
