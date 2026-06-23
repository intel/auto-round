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
"""Linear-algebra helpers for applying weight/activation rotation matrices.

Note: ``apply_transform_weight`` reuses ideas from
https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/transform/utils/matrix.py
"""

from __future__ import annotations

import torch

__all__ = ["apply_transform_weight", "multihead_matmul"]


def apply_transform_weight(
    transform_weight: torch.Tensor,
    value: torch.Tensor,
    location: str,
    module_type: type[torch.nn.Module],
) -> torch.Tensor:
    """Apply *transform_weight* to *value* according to *location*.

    The mathematical relationship for a ``torch.nn.Linear`` layer:

    .. code-block:: none

        y  = x W.T                       (standard linear)
        yh = (x V) (U.T W Vi.T).T        (rotated linear)

    where *V* is the input-side rotation and *U* the output-side rotation.

    Args:
        transform_weight: The rotation matrix to apply.
        value:            The tensor to rotate (weight or activation).
        location:         ``"input"`` or ``"weight"``.
        module_type:      ``type(module)`` – determines how the weight transform
                          is oriented.

    Returns:
        Rotated tensor with the same shape as *value*.
    """
    if location == "input":
        return multihead_matmul(value, transform_weight)

    if module_type is torch.nn.Linear:
        return multihead_matmul(value, transform_weight.T)

    raise NotImplementedError(
        f"apply_transform_weight: unsupported location={location!r} " f"with module_type={module_type}"
    )


def multihead_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Block-diagonal matrix multiplication over the last two dimensions.

    Handles the case where *A* and *B* have different sizes in their inner
    dimension by treating the smaller matrix as a repeated block-diagonal.

    For example, if ``A.shape[-1] == 2 * B.shape[-2]``, this is equivalent to::

        A @ block_diag(B, B)

    Args:
        A: Left-hand tensor.
        B: Right-hand tensor.

    Returns:
        Result of the generalised matrix multiplication.

    Raises:
        ValueError: If the inner dimensions are not evenly divisible.
    """
    a_inner = A.shape[-1]
    b_inner = B.shape[-2]

    if a_inner > b_inner:
        if a_inner % b_inner != 0:
            raise ValueError(f"multihead_matmul: A.shape[-1]={a_inner} is not divisible " f"by B.shape[-2]={b_inner}")
        num_heads = a_inner // b_inner
        A = A.unflatten(-1, (num_heads, b_inner))
        return (A @ B).flatten(-2, -1)

    if a_inner < b_inner:
        if b_inner % a_inner != 0:
            raise ValueError(f"multihead_matmul: B.shape[-2]={b_inner} is not divisible " f"by A.shape[-1]={a_inner}")
        num_heads = b_inner // a_inner
        B = B.unflatten(-2, (num_heads, a_inner))
        return (A @ B).flatten(-3, -2)

    return A @ B
