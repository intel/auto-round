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
    """Configuration for the SVDQuant structural preprocessor.

    Args:
        rank: Rank of the floating-point low-rank branch.
        smooth_enabled: Whether to search activation-aware input smoothing factors.
        smooth_num_grids: Grid resolution for Alpha/Beta smooth candidates.
        smooth_max_calibration_calls: Maximum retained calls per smooth group.
        target_modules: Module-name substrings eligible for decomposition.
        exclude_modules: Module-name substrings excluded from decomposition.
        low_rank_dtype: Deployment dtype for the low-rank factors.
        smooth_eps: Positive floor used to construct finite smooth factors.
        residual_iters: Number of alternating low-rank/residual fitting iterations.
        residual_early_stop: Stop when the reconstruction objective stops improving.
        model_adapter: Architecture adapter used for grouping and export.
        **kwargs: Shared :class:`QuantizationConfig` options.
    """

    def __init__(
        self,
        *,
        rank: int = 32,
        smooth_enabled: bool = False,
        smooth_num_grids: int = 20,
        smooth_max_calibration_calls: int = 128,
        target_modules: list[str] | tuple[str, ...] | str | None = None,
        exclude_modules: list[str] | tuple[str, ...] | str | None = None,
        low_rank_dtype: str = "bf16",
        smooth_eps: float = 1e-6,
        residual_iters: int = 1,
        residual_early_stop: bool = False,
        model_adapter: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if type(smooth_enabled) is not bool:
            raise ValueError(f"`smooth_enabled` must be a bool, got {smooth_enabled!r}")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError(f"`rank` must be a non-negative integer, got {rank!r}")
        if type(smooth_num_grids) is not int or smooth_num_grids < 2:
            raise ValueError(
                f"`smooth_num_grids` must be an integer greater than or equal to 2, got {smooth_num_grids!r}"
            )
        if type(smooth_max_calibration_calls) is not int or smooth_max_calibration_calls < 1:
            raise ValueError(
                f"`smooth_max_calibration_calls` must be a positive integer, got {smooth_max_calibration_calls!r}"
            )
        if smooth_eps <= 0:
            raise ValueError(f"`smooth_eps` must be positive, got {smooth_eps!r}")
        if not isinstance(low_rank_dtype, str) or low_rank_dtype.lower() not in {
            "bf16",
            "bfloat16",
            "fp16",
            "float16",
            "fp32",
            "float32",
        }:
            raise ValueError(
                "`low_rank_dtype` must be one of bf16, bfloat16, fp16, float16, fp32, or float32, "
                f"got {low_rank_dtype!r}"
            )
        if type(residual_iters) is not int or residual_iters < 1:
            raise ValueError(f"`residual_iters` must be a positive integer, got {residual_iters!r}")
        if type(residual_early_stop) is not bool:
            raise ValueError(f"`residual_early_stop` must be a bool, got {residual_early_stop!r}")
        self.rank = rank
        self.smooth_enabled = smooth_enabled
        self.smooth_num_grids = smooth_num_grids
        self.smooth_max_calibration_calls = smooth_max_calibration_calls
        self.target_modules = _normalize_patterns(target_modules)
        self.exclude_modules = _normalize_patterns(exclude_modules)
        self.low_rank_dtype = low_rank_dtype
        self.smooth_eps = smooth_eps
        self.residual_iters = residual_iters
        self.residual_early_stop = residual_early_stop
        self.model_adapter = model_adapter
        self.need_calib = smooth_enabled

    def __repr__(self) -> str:
        return (
            f"SVDQuantConfig(rank={self.rank}, smooth_enabled={self.smooth_enabled!r}, "
            f"smooth_num_grids={self.smooth_num_grids}, "
            f"smooth_max_calibration_calls={self.smooth_max_calibration_calls}, "
            f"low_rank_dtype={self.low_rank_dtype!r}, "
            f"target_modules={self.target_modules}, exclude_modules={self.exclude_modules}, "
            f"residual_iters={self.residual_iters}, residual_early_stop={self.residual_early_stop!r}, "
            f"model_adapter={self.model_adapter!r})"
        )


def _normalize_patterns(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)
