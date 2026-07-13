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

from dataclasses import dataclass

import torch

from auto_round.data_type.utils import get_quant_func

_MXFP4_ALIASES = frozenset({"mx_fp", "mx_fp4", "mx_fp4e2m1"})


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

    group_size = values["group_size"]
    scalar_group_size = isinstance(group_size, int) and not isinstance(group_size, bool) and group_size > 0
    block_group_size = (
        isinstance(group_size, tuple)
        and len(group_size) == 2
        and all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in group_size)
    )
    if not scalar_group_size and not block_group_size:
        raise ValueError(
            "Residual quantization scheme group_size must be a positive integer or a pair of positive integers."
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


@torch.inference_mode()
def rtn_qdq_residual(weight: torch.Tensor, scheme: ResidualQuantScheme) -> torch.Tensor:
    """Apply the registered RTN quantize-dequantize function to a residual."""
    values = _validate_scheme_values(scheme)
    if values["data_type"] in _MXFP4_ALIASES and values["bits"] == 4 and (
        not isinstance(values["group_size"], int)
        or isinstance(values["group_size"], bool)
        or values["group_size"] != 32
    ):
        raise ValueError(
            "Deployable MXFP4 residual QDQ requires scalar group_size=32; "
            f"got group_size={values['group_size']!r}."
        )

    quant_func, resolved_dtype = get_quant_func(
        dtype=values["data_type"],
        bits=values["bits"],
        sym=values["sym"],
        disable_opt_rtn=True,
        group_size=values["group_size"],
        iters=0,
    )
    qdq, _, _ = quant_func(
        tensor=weight,
        bits=values["bits"],
        group_size=values["group_size"],
        data_type=resolved_dtype.removeprefix("rtn_"),
    )
    if qdq.shape != weight.shape or qdq.dtype != weight.dtype:
        raise ValueError(
            "Residual RTN QDQ must preserve the input shape and dtype; "
            f"got shape={tuple(qdq.shape)}, dtype={qdq.dtype}."
        )
    if not torch.isfinite(qdq).all():
        raise ValueError("Residual RTN QDQ produced non-finite values.")
    return qdq
