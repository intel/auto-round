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

class FullOnlineHadamardHook:
    """Pre-forward hook: full Hadamard on the entire last dimension (for ``down_proj``)."""

    def __init__(self, had_K, K, fp32_had=False):
        self.had_K = had_K
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


class PartialOnlineHadamardHook:
    """Pre-forward hook: per-head Hadamard across the ``num_heads`` dimension (for ``o_proj``).

    Matches the ``online_partial_had`` path in the original ``ActQuantWrapper``:
      * reshape ``(*, hidden_size)`` → ``(*, num_heads, head_dim)``
      * transpose → ``(*, head_dim, num_heads)``
      * Hadamard on the **num_heads** axis (last dim after transpose)
      * transpose back → ``(*, num_heads, head_dim)``
      * reshape back to original
    """

    def __init__(self, had_K, K, had_dim, fp32_had=False):
        """
        Args:
            had_K: Hadamard sub-matrix from ``get_hadK(num_heads)``, or ``None`` when K==1.
            K: Block size from ``get_hadK(num_heads)``.
            had_dim: ``head_dim = hidden_size // num_attention_heads``.
            fp32_had: Compute in fp32.
        """
        self.had_K = had_K
        self.K = K
        self.had_dim = had_dim
        self.fp32_had = fp32_had

    def __call__(self, module: nn.Module, args):
        x = args[0] if isinstance(args, tuple) else args
        x_dtype = x.dtype

        if self.fp32_had:
            x = x.float()

        init_shape = x.shape
        had_dim = self.had_dim
        num_heads = init_shape[-1] // had_dim

        if self.K == 1:
            # reshape to (-1, num_heads, head_dim), transpose to (-1, head_dim, num_heads)
            # hadamard_transform on last dim = num_heads, then transpose back
            x = fast_hadamard_transform.hadamard_transform(
                x.reshape(-1, num_heads, had_dim).transpose(1, 2),
                scale=1.0 / math.sqrt(num_heads),
            ).transpose(1, 2)
        else:
            # had_K is (K, K) where K divides num_heads
            # reshape to (-1, num_heads, head_dim), then had_K @ x operates on num_heads dim
            x = (
                self.had_K.to(x.dtype).to(x.device)
                @ x.reshape(-1, num_heads, had_dim)
            ) / math.sqrt(num_heads)

        x = x.reshape(init_shape)

        if self.fp32_had:
            x = x.to(x_dtype)

        if isinstance(args, tuple):
            return (x,) + args[1:]
        return x


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_online_had_hooks(model, fp32_had=False):
    """Register online Hadamard pre-forward hooks on ``down_proj`` and ``o_proj``.

    Exactly reproduces the ``ActQuantWrapper`` behaviour:

    * **down_proj** (``online_full_had``): full Hadamard on ``intermediate_size``.
    * **o_proj** (``online_partial_had``): Hadamard **across heads** (on the
      ``num_heads`` axis), *not* within each head.

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

    # o_proj: partial Hadamard — had_K from num_heads, had_dim = head_dim
    had_K_partial, K_partial = get_hadK(num_heads)

    handles = []
    for name, module in model.named_modules():
        if name.endswith("down_proj") and isinstance(module, nn.Linear):
            hook = FullOnlineHadamardHook(
                had_K=had_K_full, K=K_full, fp32_had=fp32_had,
            )
            h = module.register_forward_pre_hook(hook)
            handles.append(h)
        #
        # elif name.endswith("o_proj") and isinstance(module, nn.Linear):
        #     hook = PartialOnlineHadamardHook(
        #         had_K=had_K_partial, K=K_partial,
        #         had_dim=head_dim, fp32_had=fp32_had,
        #     )
        #     h = module.register_forward_pre_hook(hook)
        #     handles.append(h)

    return handles
