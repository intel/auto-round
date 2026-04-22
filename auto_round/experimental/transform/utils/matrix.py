# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch

__all__ = ["apply_transform_weight"]

# note that apply_transform_weight reuses some code from
# https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/transform/utils/matrix.py


def apply_transform_weight(
    transform_weight: torch.Tensor,
    value: torch.Tensor,
    location: str,
    module_type: type[torch.nn.Module],
) -> torch.Tensor:
    """
    Using the transform location, apply the transform_weight to the
    given value wrt linear weights. For more info on input and output transforms,
    see `TransformLocation`

    The following explains how weights should be applied to values according to location

    let  x          be input activation
         W          be weight,
         yh, xh, Wh be transformed output, input, weight

    note that
         y  = (x W.T)        // torch.nn.Linear

    Choose values for yh, xh, and Wh which incorporate matrix transforms

    let  V, Vi      be transform matrices on input side
         U, Ui      be transform matrices on output side

    pick xh = (x V)
         Wh = (U.T W Vi.T)
         yh = (y U)

    The following shows that `yh = (xh) (Wh).T` for the chosen values of yh, xh, and Wh

    (xh) (Wh).T = (x V) (U.T W Vi.T).T
                = (x V) (Vi W.T U)        // transpose matrix product identity
                = (x W.T) U
                = y U
                = yh

    :param transform_weight: transform weight to apply
    :param value: value to apply transform_weight to
    :param location: determines how weight should be applied
    :param model_type: result of type(module), passed in to determine application of
        weight transform
    :return: value after transform_weight has been applied
    """

    if location == "input":
        return _multihead_matmul(value, transform_weight)

    if module_type == torch.nn.Linear:
        return _multihead_matmul(value, transform_weight.T)


def _multihead_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs A @ B for last two dims of two matrices A and B that possibly
    have different shapes, as is the case in multi-headed dimension. If
    shapes are different, this is equivalent to converting the last two dims
    of the smaller matrix into a block-diagonal matrix with the same shape as
    the last two dims of the larger matrix.

    E.g. if A is half the size of B, this function will perform
    [[A  ]  @ B
     [  A]]

    If B is a third of the size of A, this function will perform
    A @ [[B    ]
         [  B  ]
         [    B]]

    This function will error out if the shapes are not evenly divisible

    :param A: left-hand tensor
    :param B: right-hand tensor
    :return: result
    """
    if A.shape[-1] > B.shape[-2]:
        head_dim = B.shape[-2]
        num_heads = A.shape[-1] // head_dim
        A = A.unflatten(-1, (num_heads, head_dim))
        return (A @ B).flatten(-2, -1)
    elif A.shape[-1] < B.shape[-2]:
        head_dim = A.shape[-1]
        num_heads = B.shape[-2] // head_dim
        B = B.unflatten(-2, (num_heads, head_dim))
        return (A @ B).flatten(-3, -2)
    else:
        return A @ B
