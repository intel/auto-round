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

"""Residual low-rank decomposition inspired by SVDQuant.

Reference: Li et al., "SVDQuant: Absorbing Outliers by Low-Rank Components for
4-Bit Diffusion Models", ICLR 2025, https://arxiv.org/abs/2411.05007.

This module is an AutoRound implementation and does not copy source code from
the reference implementation.
"""

import math
from dataclasses import dataclass

import torch

from auto_round.data_type.utils import get_quant_func

_FIXED_MXFP4_DTYPES = frozenset({"mx_fp4", "mx_fp4e2m1"})
_MXFP4_ALIASES = frozenset({"mx_fp", *_FIXED_MXFP4_DTYPES})


def _validate_scheme_values(scheme):
    values = {}
    for field in ("data_type", "bits", "group_size", "sym"):
        try:
            values[field] = getattr(scheme, field)
        except AttributeError as exc:
            raise ValueError(f"Residual quantization scheme is missing required value {field!r}.") from exc

    if not isinstance(values["data_type"], str) or not values["data_type"].strip():
        raise ValueError("Residual quantization scheme data_type must be a non-empty string.")
    if not isinstance(values["bits"], int) or isinstance(values["bits"], bool) or values["bits"] <= 0:
        raise ValueError("Residual quantization scheme bits must be a positive integer.")
    if values["data_type"] in _FIXED_MXFP4_DTYPES and values["bits"] != 4:
        raise ValueError(
            f"Residual quantization scheme data_type={values['data_type']!r} requires bits=4; "
            f"got bits={values['bits']}."
        )

    group_size = values["group_size"]
    scalar_group_size = isinstance(group_size, int) and not isinstance(group_size, bool) and group_size >= -1
    block_group_size = (
        isinstance(group_size, tuple)
        and len(group_size) == 2
        and all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in group_size)
    )
    if not scalar_group_size and not block_group_size:
        raise ValueError(
            "Residual quantization scheme group_size must be -1, 0, a positive integer, or a pair of positive integers."
        )
    if not isinstance(values["sym"], bool):
        raise ValueError("Residual quantization scheme sym must be a boolean.")
    return values


@dataclass(frozen=True)
class ResidualQuantScheme:
    """Weight quantization settings for stateless residual QDQ."""

    data_type: str | None = None
    bits: int | None = None
    group_size: int | tuple[int, int] | None = None
    sym: bool | None = None

    def __post_init__(self) -> None:
        _validate_scheme_values(self)


@dataclass(frozen=True)
class ActivationQuantScheme:
    """Activation quantization settings for stateless calibration QDQ."""

    data_type: str | None = None
    bits: int | None = None
    group_size: int | tuple[int, int] | None = None
    sym: bool | None = None

    def __post_init__(self) -> None:
        _validate_scheme_values(self)


@dataclass(frozen=True)
class ResidualDecomposition:
    """Best deployment-materialized candidate from residual outer iteration."""

    residual: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor
    selected_iteration: int
    error: float


def _rtn_qdq_tensor(tensor: torch.Tensor, scheme, *, tensor_name: str) -> torch.Tensor:
    values = _validate_scheme_values(scheme)
    requested_dtype = values["data_type"]
    if values["bits"] == 4 and requested_dtype in _MXFP4_ALIASES:
        requested_dtype = f"{requested_dtype}_rceil"
    quant_func, resolved_dtype = get_quant_func(
        dtype=requested_dtype,
        bits=values["bits"],
        sym=values["sym"],
        disable_opt_rtn=True,
        group_size=values["group_size"],
        iters=0,
    )
    logical_dtype = resolved_dtype.removeprefix("rtn_")
    resolved_base_dtype = logical_dtype.removesuffix("_rceil")
    if (
        resolved_base_dtype in _MXFP4_ALIASES
        and values["bits"] == 4
        and (
            not isinstance(values["group_size"], int)
            or isinstance(values["group_size"], bool)
            or values["group_size"] != 32
        )
    ):
        raise ValueError(
            f"Deployable MXFP4 {tensor_name} QDQ requires scalar group_size=32; "
            f"got group_size={values['group_size']!r}."
        )

    qdq, _, _ = quant_func(
        tensor=tensor,
        bits=values["bits"],
        group_size=values["group_size"],
        data_type=logical_dtype,
    )
    if qdq.shape != tensor.shape or qdq.dtype != tensor.dtype:
        raise ValueError(
            f"{tensor_name.capitalize()} RTN QDQ must preserve the input shape and dtype; "
            f"got shape={tuple(qdq.shape)}, dtype={qdq.dtype}."
        )
    if qdq.device != tensor.device:
        raise ValueError(
            f"{tensor_name.capitalize()} RTN QDQ must preserve the input device; "
            f"got input device={tensor.device}, output device={qdq.device}."
        )
    if not torch.isfinite(qdq).all():
        raise ValueError(f"{tensor_name.capitalize()} RTN QDQ produced non-finite values.")
    return qdq


@torch.inference_mode()
def rtn_qdq_residual(weight: torch.Tensor, scheme: ResidualQuantScheme) -> torch.Tensor:
    """Apply the registered RTN quantize-dequantize function to a residual."""
    return _rtn_qdq_tensor(weight, scheme, tensor_name="residual")


@torch.inference_mode()
def rtn_qdq_activation(activation: torch.Tensor, scheme: ActivationQuantScheme) -> torch.Tensor:
    """Apply deployment-compatible dynamic activation quantize-dequantize."""
    return _rtn_qdq_tensor(activation, scheme, tensor_name="activation")


@torch.inference_mode()
def truncated_svd(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a rank-limited reconstruction and its shared down/up factors."""
    if weight.ndim != 2:
        raise ValueError(f"SVDQuant expects a two-dimensional weight matrix, got shape={tuple(weight.shape)}.")
    max_rank = min(weight.shape)
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank <= max_rank:
        raise ValueError(f"SVDQuant rank must be an integer in [0, {max_rank}], got {rank!r}.")

    out_features, in_features = weight.shape
    if rank == 0:
        low_rank = torch.zeros_like(weight)
        down_weight = torch.empty((0, in_features), dtype=weight.dtype, device=weight.device)
        up_weight = torch.empty((out_features, 0), dtype=weight.dtype, device=weight.device)
        return low_rank, down_weight, up_weight

    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    down_weight = vh[:rank, :]
    up_weight = u[:, :rank] * s[:rank].reshape(1, -1)
    return up_weight @ down_weight, down_weight, up_weight


@torch.inference_mode()
def iterate_residual_decomposition(
    weight: torch.Tensor,
    *,
    rank: int,
    scheme: ResidualQuantScheme,
    iterations: int,
    early_stop: bool,
    residual_dtype: torch.dtype,
    low_rank_dtype: torch.dtype,
) -> ResidualDecomposition:
    """Select the lowest weight-MSE residual/low-rank candidate after deployment casting."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError(f"SVDQuant residual iterations must be a positive integer, got {iterations!r}.")

    quantized_residual = torch.zeros_like(weight)
    best_down = None
    best_up = None
    best_error = float("inf")
    best_iteration = None

    for iteration in range(1, iterations + 1):
        low_rank, down, up = truncated_svd(weight - quantized_residual, rank)
        if not all(torch.isfinite(tensor).all() for tensor in (low_rank, down, up)):
            break

        residual = (weight - low_rank).to(residual_dtype)
        quantized_residual = rtn_qdq_residual(residual, scheme).to(weight.dtype)
        deployed_down = down.to(low_rank_dtype)
        deployed_up = up.to(low_rank_dtype)
        deployed_low_rank = deployed_up.to(weight.dtype) @ deployed_down.to(weight.dtype)
        error = torch.sum((weight - (quantized_residual + deployed_low_rank)).square()).item()
        accepted = math.isfinite(error) and error <= best_error

        if accepted:
            best_down = deployed_down.clone()
            best_up = deployed_up.clone()
            best_error = error
            best_iteration = iteration
        elif early_stop and best_iteration is not None:
            break

    if best_down is None or best_up is None or best_iteration is None:
        raise ValueError("SVDQuant residual iteration did not produce a finite candidate.")

    deployed_low_rank = best_up.to(weight.dtype) @ best_down.to(weight.dtype)
    residual = weight - deployed_low_rank
    if not torch.isfinite(residual).all():
        raise ValueError("SVDQuant residual iteration produced a non-finite residual.")
    return ResidualDecomposition(residual, best_down, best_up, best_iteration, best_error)
