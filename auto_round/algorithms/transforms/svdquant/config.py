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

from __future__ import annotations

from auto_round.algorithms.quantization.config import QuantizationConfig


class SVDQuantConfig(QuantizationConfig):
    """Configuration for the SVDQuant structural transform.

    SVDQuant does not define the final quantization datatype. It prepares each
    target Linear as a high-precision low-rank branch plus a residual Linear,
    then delegates the residual Linear to the downstream RTN/SignRound quantizer.
    """

    def __init__(
        self,
        *,
        rank: int = 32,
        smooth_alpha: float = 0.5,
        target_modules: list[str] | tuple[str, ...] | str | None = None,
        exclude_modules: list[str] | tuple[str, ...] | str | None = None,
        low_rank_dtype: str = "bf16",
        smooth_eps: float = 1e-6,
        residual_iters: int = 1,
        residual_early_stop: bool = False,
        residual_quant_method: str = "rtn",
        **kwargs,
    ):
        super().__init__(**kwargs)
        residual_quant_method = residual_quant_method.lower()
        if rank < 0:
            raise ValueError(f"`rank` must be non-negative, got {rank!r}")
        if not 0.0 <= smooth_alpha <= 1.0:
            raise ValueError(f"`smooth_alpha` must be in [0, 1], got {smooth_alpha!r}")
        if smooth_eps <= 0:
            raise ValueError(f"`smooth_eps` must be positive, got {smooth_eps!r}")
        if type(residual_iters) is not int or residual_iters < 1:
            raise ValueError(f"`residual_iters` must be a positive integer, got {residual_iters!r}")
        if residual_iters > 1 and residual_quant_method != "rtn":
            raise ValueError(
                "`residual_quant_method` must be 'rtn' when `residual_iters` is greater than 1, "
                f"got {residual_quant_method!r}"
            )

        self.rank = rank
        self.smooth_alpha = smooth_alpha
        self.target_modules = _normalize_patterns(target_modules)
        self.exclude_modules = _normalize_patterns(exclude_modules)
        self.low_rank_dtype = low_rank_dtype
        self.smooth_eps = smooth_eps
        self.residual_iters = residual_iters
        self.residual_early_stop = residual_early_stop
        self.residual_quant_method = residual_quant_method

    def __repr__(self) -> str:
        return (
            f"SVDQuantConfig(rank={self.rank}, smooth_alpha={self.smooth_alpha}, "
            f"low_rank_dtype={self.low_rank_dtype!r}, "
            f"target_modules={self.target_modules}, exclude_modules={self.exclude_modules}, "
            f"residual_iters={self.residual_iters}, residual_early_stop={self.residual_early_stop!r}, "
            f"residual_quant_method={self.residual_quant_method!r})"
        )


def _normalize_patterns(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)
