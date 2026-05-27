# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
SpinQuant / QuaRot monkeypatch utilities for online rotations.

This module provides the architecture-generic mechanism to inject R3 rotation
(Hadamard on Q/K **after** RoPE) by replacing the ``apply_rotary_pos_emb``
function reference in the attention forward method's globals.

This approach comes from QuaRot (https://github.com/spcl/QuaRot) and is used
by both QuaRot and AMD Quark. It avoids rewriting the full attention forward
and works for any HuggingFace model that calls ``apply_rotary_pos_emb`` in its
attention forward (Llama, Qwen2, Qwen3, Mistral, Phi, Gemma, etc.).
"""

from __future__ import annotations

import copy
import functools
import types
from typing import Any, Callable

import torch
import torch.nn as nn


def copy_func_with_new_globals(f: Callable[..., Any], globals: dict[str, Any] | None = None) -> Callable[..., Any]:
    """Create a copy of function ``f`` with modified globals dict.

    Based on https://stackoverflow.com/a/13503277/2988730
    """
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(
        f.__code__,
        globals,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def add_wrapper_after_function_call_in_method(
    module: nn.Module,
    method_name: str,
    function_name: str,
    wrapper_cls: type,
) -> Any:
    """Replace a function referenced in a method's globals with a wrapper.

    This modifies ``module.method_name`` so that any call to ``function_name``
    inside it now goes through ``wrapper_cls(original_function)``.

    For R3, we wrap ``apply_rotary_pos_emb`` with ``QKRotationWrapper`` so that
    Hadamard is applied to Q and K immediately after RoPE.

    Args:
        module: The nn.Module whose method will be patched.
        method_name: Name of the method to patch (typically "forward").
        function_name: Name of the function to wrap in the method's globals.
        wrapper_cls: A callable class that takes the original function and
            returns a replacement (e.g., ``QKRotationWrapper``).

    Returns:
        The wrapper instance (useful for later removal or inspection).
    """
    original_method = getattr(module, method_name).__func__

    # Handle decorated methods (e.g., @torch.no_grad wrapper in transformers)
    if original_method.__closure__ is not None:
        for cell in original_method.__closure__:
            try:
                if isinstance(cell.cell_contents, types.FunctionType) and cell.cell_contents.__name__ == method_name:
                    original_method = cell.cell_contents
                    break
            except ValueError:
                continue

    method_globals = dict(original_method.__globals__)

    if function_name not in method_globals:
        raise ValueError(
            f"Function '{function_name}' not found in globals of {module.__class__.__name__}.{method_name}. "
            f"This model architecture may not be supported for R3 rotation."
        )

    wrapper = wrapper_cls(method_globals[function_name])
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, types.MethodType(new_method, module))
    return wrapper


class QKRotationWrapper(nn.Module):
    """Wraps ``apply_rotary_pos_emb`` to apply R3 rotation after RoPE.

    The wrapped function is called with the same arguments as the original
    ``apply_rotary_pos_emb(q, k, cos, sin, ...)`` and applies a rotation
    transform to both Q and K outputs.

    Supports two modes:
    - **Butterfly** (default via ``set_hadamard``): deterministic Hadamard
      using ``matmul_hadU`` — fast, no matrix storage needed.
    - **Matrix** (via ``set_matrix``): explicit ``x @ R`` using a stored
      matrix — works for random/trained rotations.

    Math: For attention scores, (Q@R) @ (K@R).T = Q @ R @ R.T @ K.T = Q @ K.T
    since R is orthogonal (R @ R.T = I).
    """

    def __init__(self, original_func: Callable[..., Any]):
        super().__init__()
        self.original_func = original_func
        self._had_K: torch.Tensor | None = None
        self._K: int = 1
        self._head_dim: int = 0
        self._full_matrix: torch.Tensor | None = None

    def set_hadamard(self, had_matrix: Any, head_dim: int) -> None:
        """Use butterfly algorithm for deterministic Hadamard.

        The ``had_matrix`` parameter is accepted for API compatibility but
        ignored — the decomposition is always computed from ``head_dim``.
        """
        from auto_round.algorithms.transforms.spinquant.rotation_utils import get_hadamard_K

        self._head_dim = head_dim
        had_K, K = get_hadamard_K(head_dim)
        self._had_K = had_K
        self._K = K
        self._full_matrix = None

    def set_matrix(self, matrix: torch.Tensor) -> None:
        """Use a stored full matrix for explicit x @ R (random/trained)."""
        self._full_matrix = matrix
        self._head_dim = matrix.shape[0]
        self._had_K = None

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = self.original_func(*args, **kwargs)

        if self._full_matrix is not None:
            # Explicit matrix multiply (random/trained)
            orig_dtype = q.dtype
            R = self._full_matrix.to(device=q.device, dtype=q.dtype)
            q = (q @ R).to(orig_dtype)
            k = (k @ R).to(orig_dtype)
        elif self._had_K is not None:
            # Butterfly algorithm (deterministic Hadamard)
            from auto_round.algorithms.transforms.spinquant.rotation_utils import matmul_hadU

            orig_dtype = q.dtype
            had_K = self._had_K.to(device=q.device, dtype=torch.float32)
            q = matmul_hadU(q.float(), hadamard_K=had_K, K=self._K).to(orig_dtype)
            k = matmul_hadU(k.float(), hadamard_K=had_K, K=self._K).to(orig_dtype)

        return q, k


def add_qk_rotation_after_rope(
    attn_module: nn.Module,
    rope_function_name: str = "apply_rotary_pos_emb",
) -> QKRotationWrapper:
    """Add R3 rotation after RoPE in an attention module's forward.

    This is the architecture-generic R3 injection point. It works for any model
    whose attention forward calls ``apply_rotary_pos_emb`` (or similar) as a
    module-level function.

    Args:
        attn_module: The attention module (e.g., ``model.layers[0].self_attn``).
        rope_function_name: The name of the RoPE function called in forward.

    Returns:
        The QKRotationWrapper instance (call ``.set_hadamard()`` to activate).
    """
    wrapper = add_wrapper_after_function_call_in_method(attn_module, "forward", rope_function_name, QKRotationWrapper)
    return wrapper
