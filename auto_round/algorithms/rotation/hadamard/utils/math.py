# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hadamard matrix construction utilities.

Provides ``deterministic_hadamard_matrix`` (Sylvester construction) and
``random_hadamard_matrix`` (loaded from a precomputed safetensors file).
"""
# note that hadamard matrix multiplication reuses code from
# https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/transform/utils/hadamard.py

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors import safe_open

__all__ = ["deterministic_hadamard_matrix", "random_hadamard_matrix", "is_pow2"]

# Precomputed Hadamard matrices for non-power-of-2 sizes.
_HADAMARD_MATRICES_PATH: Path = Path(__file__).parent / "hadamards.safetensors"


def deterministic_hadamard_matrix(
    size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Construct an ``(size × size)`` Hadamard matrix via Sylvester's construction.

    ``size`` must be a power of 2.

    Adapted from https://github.com/scipy/scipy/blob/v1.15.2/scipy/linalg/_special_matrices.py

    Args:
        size:   Order of the matrix; must be a power of 2.
        dtype:  Output dtype.
        device: Output device.

    Returns:
        Hadamard tensor of shape ``(size, size)``.
    """
    if size <= 0:
        raise ValueError("Cannot construct Hadamard matrix with size <= 0")
    log2 = int(math.log2(size))
    if size != 2**log2:
        raise ValueError("Deterministic Hadamard requires size == 2^n")

    H = torch.tensor([[1]], dtype=dtype, device=device)
    for _ in range(log2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))
    return H


def random_hadamard_matrix(
    size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a randomly signed Hadamard matrix of order *size*.

    Supports non-powers-of-2 by reading a precomputed base matrix from
    ``hadamards.safetensors`` and composing it with a random ±1 diagonal.

    Adapted from https://github.com/facebookresearch/SpinQuant/blob/main/utils/hadamard_utils.py

    Args:
        size:   Dimension of the matrix.
        dtype:  Output dtype.
        device: Output device.
        gen:    Optional seeded ``torch.Generator`` for reproducibility.

    Returns:
        Randomly signed Hadamard tensor of shape ``(size, size)``.
    """
    Q = torch.randint(0, 2, (size,), generator=gen, dtype=dtype).to(device)
    Q = Q * 2 - 1
    return _matmul_hadU(torch.diag(Q))


def is_pow2(n: int) -> bool:
    """Return ``True`` iff *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_hadamard_divisor(
    n: int,
    dtype: torch.dtype,
    device: torch.device = torch.device("cpu"),
    file_path: Path = _HADAMARD_MATRICES_PATH,
) -> torch.Tensor | None:
    """Return the largest precomputed Hadamard divisor ``k`` of *n* such that
    ``n / k`` is a power of two, or ``None`` if no such entry exists."""
    open_device = torch.device("cpu") if device.type == "meta" else device
    with safe_open(str(file_path), framework="pt", device=str(open_device)) as f:
        divisors = sorted((int(key) for key in f.keys()), reverse=True)
        for divisor in divisors:
            if n % divisor == 0 and is_pow2(n // divisor):
                return f.get_tensor(str(divisor)).to(dtype=dtype, device=device)
    return None


def _matmul_hadU(X: torch.Tensor) -> torch.Tensor:
    """Multiply *X* (a diagonal matrix) by the appropriate Hadamard matrix."""
    size = X.size(0)
    dtype = X.dtype
    device = X.device

    hadK = _fetch_hadamard_divisor(size, dtype, device=device)
    if hadK is None:
        raise ValueError(f"Cannot construct random Hadamard matrix of size {size}")
    K = hadK.size(0)

    inp = X.clone().view(-1, size, 1)
    out = inp.clone()
    while inp.shape[1] > K:
        inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
        out = out.view(inp.shape)
        out[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
        out[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
        out = out.view(inp.shape[0], inp.shape[1], -1)
        inp, out = out, inp
    assert inp.shape[1] == K
    del out
    return (hadK.view(1, K, K).to(inp) @ inp).view(X.shape)
