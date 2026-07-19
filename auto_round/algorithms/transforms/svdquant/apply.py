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

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

import auto_round.algorithms.transforms.svdquant.residual as residual_module
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.transforms.base import BaseWeightTransformer
from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
from auto_round.algorithms.transforms.svdquant.smooth import (
    absmax_channel_span,
    build_alpha_beta_candidates,
    build_smooth_scale,
    select_best_layer_candidate,
)
from auto_round.algorithms.transforms.svdquant.smooth_adapters import (
    SmoothSearchGroup,
    discover_smooth_search_groups,
)
from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.logger import logger

_SCHEME_ATTRS = (
    "bits",
    "group_size",
    "sym",
    "data_type",
    "act_bits",
    "act_group_size",
    "act_sym",
    "act_data_type",
    "act_dynamic",
    "super_bits",
    "super_group_size",
    "super_sym",
    "scale_dtype",
    "weight_global_scale",
    "tuning_device",
)


@dataclass
class CapturedEvaluation:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any


@dataclass
class SmoothGroupCalibration:
    group: SmoothSearchGroup
    projection_inputs: list[torch.Tensor] = field(default_factory=list)
    evaluation_calls: list[CapturedEvaluation] = field(default_factory=list)


def _detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    return value


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


@register_pipeline_member(SVDQuantConfig)
class SVDQuantTransform(BaseWeightTransformer):
    """Apply SVDQuant decomposition before downstream RTN/SignRound quantization."""

    def __init__(self, config: SVDQuantConfig) -> None:
        super().__init__(config)
        self._act_max: dict[int, torch.Tensor] = {}
        self._smooth_calibration: dict[str, SmoothGroupCalibration] = {}

    @contextmanager
    def block_forward_hooks(self, ctx):
        handles = []
        if not self.config.smooth_enabled:
            yield handles
            return

        self._clear_smooth_calibration()
        groups = discover_smooth_search_groups(ctx.block, self._is_target)
        self._smooth_calibration = {group.key: SmoothGroupCalibration(group) for group in groups}
        num_samples = getattr(ctx, "num_samples", 1)
        batch_size = max(int(getattr(ctx, "bs", 1)), 1)
        capture_calls = max(math.ceil(int(num_samples) / batch_size), 1)
        projection_owners: dict[int, list[SmoothGroupCalibration]] = {}
        evaluation_owners: dict[int, list[SmoothGroupCalibration]] = {}
        modules: dict[int, torch.nn.Module] = {}
        for calibration in self._smooth_calibration.values():
            projection_module = calibration.group.projection_input_module
            evaluation_module = calibration.group.evaluation_module
            projection_owners.setdefault(id(projection_module), []).append(calibration)
            evaluation_owners.setdefault(id(evaluation_module), []).append(calibration)
            modules[id(projection_module)] = projection_module
            modules[id(evaluation_module)] = evaluation_module

        def collect_calibration(module, inputs, kwargs, output):
            projection_calibrations = projection_owners.get(id(module), ())
            if projection_calibrations and inputs and torch.is_tensor(inputs[0]):
                x = inputs[0]
                for calibration in projection_calibrations:
                    if len(calibration.projection_inputs) >= capture_calls:
                        continue
                    if x.shape[-1] != calibration.group.projections[0].in_features:
                        continue
                    captured = _detach_to_cpu(x)
                    calibration.projection_inputs.append(captured)
                    amax = captured.abs().reshape(-1, captured.shape[-1]).amax(dim=0).to(torch.float32)
                    for projection in calibration.group.projections:
                        old = self._act_max.get(id(projection))
                        self._act_max[id(projection)] = amax if old is None else torch.maximum(old, amax)

            evaluation_calibrations = evaluation_owners.get(id(module), ())
            pending = [
                calibration
                for calibration in evaluation_calibrations
                if len(calibration.evaluation_calls) < capture_calls
            ]
            if pending:
                captured = CapturedEvaluation(
                    args=_detach_to_cpu(inputs),
                    kwargs=_detach_to_cpu(kwargs),
                    output=_detach_to_cpu(output),
                )
                for calibration in pending:
                    calibration.evaluation_calls.append(captured)

        for module in modules.values():
            handles.append(module.register_forward_hook(collect_calibration, with_kwargs=True))
        try:
            yield handles
        finally:
            for handle in handles:
                handle.remove()

    def _clear_smooth_calibration(self) -> None:
        for calibration in self._smooth_calibration.values():
            for projection in calibration.group.projections:
                self._act_max.pop(id(projection), None)
        self._smooth_calibration.clear()

    @torch.no_grad()
    def pre_quantize_block(self, ctx) -> None:
        if self.config.smooth_enabled:
            self._pre_quantize_smoothed_block(ctx)
            return

        replacements = []
        for name, module in ctx.block.named_modules():
            if self._is_target(name, module):
                replacements.append((name, module))

        for name, module in replacements:
            _set_child_module(ctx.block, name, self._decompose_linear(module))

    def _pre_quantize_smoothed_block(self, ctx) -> None:
        if not self._smooth_calibration:
            groups = discover_smooth_search_groups(ctx.block, self._is_target)
            self._smooth_calibration = {group.key: SmoothGroupCalibration(group) for group in groups}

        local_names = {id(module): name for name, module in ctx.block.named_modules() if name}
        selected_scales: dict[str, torch.Tensor] = {}
        try:
            for key, calibration in self._smooth_calibration.items():
                if not calibration.projection_inputs or not calibration.evaluation_calls:
                    raise ValueError(f"SVDQuant smooth calibration inputs are missing for group {key!r}.")
                selected_scales[key] = self._search_group_scale(calibration, ctx.block, local_names)

            replacements: list[tuple[str, SVDQuantLinear]] = []
            for key, calibration in self._smooth_calibration.items():
                wrappers = self._decompose_group(calibration.group, selected_scales[key])
                for projection, wrapper in zip(calibration.group.projections, wrappers):
                    local_name = local_names.get(id(projection))
                    if local_name is None:
                        raise ValueError(f"SVDQuant could not locate projection {self._module_name(projection)!r}.")
                    replacements.append((local_name, wrapper))

            for local_name, wrapper in replacements:
                _set_child_module(ctx.block, local_name, wrapper)
        finally:
            self._clear_smooth_calibration()

    def _search_group_scale(
        self,
        calibration: SmoothGroupCalibration,
        block: torch.nn.Module,
        local_names: dict[int, str],
    ) -> torch.Tensor:
        group = calibration.group
        device = group.projections[0].weight.device
        x_span = torch.stack([absmax_channel_span(inputs, -1) for inputs in calibration.projection_inputs], dim=0).amax(
            dim=0
        )
        weights = [
            projection.weight.detach().to(device=device, dtype=torch.float32) for projection in group.projections
        ]
        w_span = absmax_channel_span(torch.cat(weights, dim=0), 1).cpu()
        scored = []
        for alpha, beta in build_alpha_beta_candidates(self.config.smooth_num_grids):
            scale = build_smooth_scale(x_span, w_span, alpha, beta)
            try:
                error = self._score_group_candidate(calibration, scale, block, local_names)
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.debug("Skipping SVDQuant smooth candidate (%s, %s) for %s: %s", alpha, beta, group.key, exc)
                error = float("inf")
            scored.append((scale, error))
        return select_best_layer_candidate(scored, module_name=group.key)

    def _score_group_candidate(
        self,
        calibration: SmoothGroupCalibration,
        scale: torch.Tensor,
        block: torch.nn.Module,
        local_names: dict[int, str],
    ) -> float:
        group = calibration.group
        wrappers = self._candidate_group_wrappers(group, scale)
        replacements = []
        for projection, wrapper in zip(group.projections, wrappers):
            local_name = local_names.get(id(projection))
            if local_name is None:
                raise ValueError(f"SVDQuant could not locate projection {self._module_name(projection)!r}.")
            replacements.append((local_name, projection, wrapper))

        try:
            for local_name, _projection, wrapper in replacements:
                _set_child_module(block, local_name, wrapper)

            error = torch.zeros((), dtype=torch.float64)
            for call in calibration.evaluation_calls:
                evaluation_module = group.evaluation_module
                if len(group.projections) == 1 and evaluation_module is group.projections[0]:
                    evaluation_module = wrappers[0]
                device = group.projections[0].weight.device
                args = _move_to_device(call.args, device)
                kwargs = group.filter_evaluation_kwargs(_move_to_device(call.kwargs, device))
                actual = group.normalize_output(evaluation_module(*args, **kwargs))
                reference = tuple(tensor.to(device) for tensor in group.normalize_output(call.output))
                if len(actual) != len(reference):
                    raise ValueError("smooth-search output tensor count changed")
                for actual_tensor, reference_tensor in zip(actual, reference):
                    if actual_tensor.shape != reference_tensor.shape:
                        raise ValueError("smooth-search output tensor shape changed")
                    error += torch.sum((actual_tensor.float() - reference_tensor.float()).square()).double().cpu()
            return error.item()
        finally:
            for local_name, projection, _wrapper in replacements:
                _set_child_module(block, local_name, projection)

    def _candidate_group_wrappers(
        self,
        group: SmoothSearchGroup,
        scale: torch.Tensor,
    ) -> list[SVDQuantLinear]:
        weights = [
            projection.weight.detach().to(torch.float32) * scale.to(projection.weight.device)
            for projection in group.projections
        ]
        stacked = torch.cat(weights, dim=0)
        rank = min(self.config.rank, *stacked.shape)
        _low_rank, down_weight, stacked_up = _truncated_svd(stacked, rank)
        low_rank_dtype = self._resolve_low_rank_dtype(group.projections[0].weight.dtype)
        deployed_down = down_weight.to(low_rank_dtype)
        deployed_up = stacked_up.to(low_rank_dtype)
        deployed_low_rank = deployed_up.float() @ deployed_down.float()
        low_rank_parts = deployed_low_rank.split([projection.out_features for projection in group.projections], dim=0)
        up_parts = deployed_up.split([projection.out_features for projection in group.projections], dim=0)
        residuals = []
        for projection, weight, low_rank in zip(group.projections, weights, low_rank_parts):
            residual = (weight - low_rank).to(projection.weight.dtype)
            residuals.append(residual_module.rtn_qdq_residual(residual, self._residual_quant_scheme(projection)))
        return self._build_group_wrappers(group, residuals, deployed_down, up_parts, scale)

    def _decompose_group(self, group: SmoothSearchGroup, scale: torch.Tensor) -> list[SVDQuantLinear]:
        weights = [
            projection.weight.detach().to(torch.float32) * scale.to(projection.weight.device)
            for projection in group.projections
        ]
        stacked = torch.cat(weights, dim=0)
        rank = min(self.config.rank, *stacked.shape)
        low_rank_dtype = self._resolve_low_rank_dtype(group.projections[0].weight.dtype)
        if self.config.residual_iters == 1:
            _low_rank, down_weight, stacked_up = _truncated_svd(stacked, rank)
            deployed_down = down_weight.to(low_rank_dtype)
            deployed_up = stacked_up.to(low_rank_dtype)
            deployed_low_rank = deployed_up.float() @ deployed_down.float()
            residuals = [
                residual.to(projection.weight.dtype)
                for residual, projection in zip(
                    (stacked - deployed_low_rank).split(
                        [projection.out_features for projection in group.projections], dim=0
                    ),
                    group.projections,
                )
            ]
        else:
            residuals, deployed_down, deployed_up = self._iterate_group_residual(group, stacked, rank, low_rank_dtype)
        up_parts = deployed_up.split([projection.out_features for projection in group.projections], dim=0)
        return self._build_group_wrappers(group, residuals, deployed_down, up_parts, scale)

    def _iterate_group_residual(
        self,
        group: SmoothSearchGroup,
        stacked: torch.Tensor,
        rank: int,
        low_rank_dtype: torch.dtype,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        output_sizes = [projection.out_features for projection in group.projections]
        quantized_residual = torch.zeros_like(stacked)
        best: tuple[torch.Tensor, torch.Tensor] | None = None
        best_error = float("inf")
        last_failure = "no finite candidate was produced"
        for iteration in range(1, self.config.residual_iters + 1):
            try:
                _low_rank, down_weight, stacked_up = _truncated_svd(stacked - quantized_residual, rank)
                deployed_down = down_weight.to(low_rank_dtype)
                deployed_up = stacked_up.to(low_rank_dtype)
                deployed_low_rank = deployed_up.float() @ deployed_down.float()
                residual_parts = (stacked - deployed_low_rank).split(output_sizes, dim=0)
                qdq_parts = [
                    residual_module.rtn_qdq_residual(
                        residual.to(projection.weight.dtype), self._residual_quant_scheme(projection)
                    ).float()
                    for residual, projection in zip(residual_parts, group.projections)
                ]
                quantized_residual = torch.cat(qdq_parts, dim=0)
                error_value = torch.sum((stacked - (quantized_residual + deployed_low_rank)).square()).item()
            except (RuntimeError, ValueError) as exc:
                last_failure = f"iteration {iteration}: {exc}"
                break
            improved = math.isfinite(error_value) and error_value < best_error
            if improved:
                best = (deployed_down.clone(), deployed_up.clone())
                best_error = error_value
            elif not math.isfinite(error_value):
                last_failure = f"iteration {iteration}: reconstruction error is non-finite"
            if self.config.residual_early_stop and best is not None and not improved:
                break

        if best is None:
            raise ValueError(f"SVDQuant residual iteration failed for group {group.key!r}: {last_failure}.")
        deployed_down, deployed_up = best
        deployed_low_rank = deployed_up.float() @ deployed_down.float()
        residuals = [
            residual.to(projection.weight.dtype)
            for residual, projection in zip((stacked - deployed_low_rank).split(output_sizes, dim=0), group.projections)
        ]
        return residuals, deployed_down, deployed_up

    def _build_group_wrappers(
        self,
        group: SmoothSearchGroup,
        residuals: list[torch.Tensor],
        down_weight: torch.Tensor,
        up_parts: tuple[torch.Tensor, ...],
        scale: torch.Tensor,
    ) -> list[SVDQuantLinear]:
        wrappers = []
        low_rank_dtype = down_weight.dtype
        rank = down_weight.shape[0]
        smooth = scale.reciprocal()
        for projection, residual_weight, up_weight in zip(group.projections, residuals, up_parts):
            residual = self._new_linear_like(projection, residual_weight, projection.bias)
            lora_down = torch.nn.Linear(
                projection.in_features,
                rank,
                bias=False,
                dtype=low_rank_dtype,
                device=projection.weight.device,
            )
            lora_up = torch.nn.Linear(
                rank,
                projection.out_features,
                bias=False,
                dtype=low_rank_dtype,
                device=projection.weight.device,
            )
            lora_down.weight.copy_(down_weight.to(projection.weight.device))
            lora_up.weight.copy_(up_weight.to(projection.weight.device))
            self._mark_unquantized(lora_down)
            self._mark_unquantized(lora_up)
            self._copy_quant_attrs(projection, residual, suffix=".residual_linear")
            wrappers.append(SVDQuantLinear(residual, lora_down, lora_up, smooth.to(projection.weight.dtype)))
        return wrappers

    def _is_target(self, name: str, module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        if isinstance(module, SVDQuantLinear):
            return False
        targets = self.config.target_modules
        excludes = self.config.exclude_modules
        if targets and not any(pattern in name for pattern in targets):
            return False
        if excludes and any(pattern in name for pattern in excludes):
            return False
        return True

    def _decompose_linear(self, module: torch.nn.Linear) -> SVDQuantLinear:
        if not hasattr(module, "weight"):
            return module

        weight = module.weight.detach().to(torch.float32)
        out_features, in_features = weight.shape
        rank = min(self.config.rank, out_features, in_features)
        smooth = self._build_smooth(module, weight)
        smooth = smooth.to(device=weight.device, dtype=weight.dtype)
        weight_hat = weight / smooth.reshape(1, -1)
        low_rank_dtype = self._resolve_low_rank_dtype(module.weight.dtype)

        if self.config.residual_iters == 1:
            try:
                low_rank, down_weight, up_weight = _truncated_svd(weight_hat, rank)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"SVDQuant decomposition failed for module {self._module_name(module)!r} at iteration 1: {exc}."
                ) from exc
            residual_weight = weight_hat - low_rank
            self._raise_if_nonfinite(module, 1, residual_weight, low_rank, down_weight, up_weight)
        else:
            residual_weight, down_weight, up_weight = self._iterate_residual(module, weight_hat, rank, low_rank_dtype)

        residual = self._new_linear_like(module, residual_weight, module.bias)
        lora_down = torch.nn.Linear(in_features, rank, bias=False, dtype=low_rank_dtype, device=module.weight.device)
        lora_up = torch.nn.Linear(rank, out_features, bias=False, dtype=low_rank_dtype, device=module.weight.device)
        with torch.no_grad():
            lora_down.weight.copy_(down_weight.to(low_rank_dtype))
            lora_up.weight.copy_(up_weight.to(low_rank_dtype))

        self._mark_unquantized(lora_down)
        self._mark_unquantized(lora_up)
        self._copy_quant_attrs(module, residual, suffix=".residual_linear")
        return SVDQuantLinear(residual, lora_down, lora_up, smooth.to(module.weight.dtype))

    def _iterate_residual(
        self,
        module: torch.nn.Linear,
        weight_hat: torch.Tensor,
        rank: int,
        low_rank_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scheme = self._residual_quant_scheme(module)
        quantized_residual = torch.zeros_like(weight_hat)
        best_down = None
        best_up = None
        best_error = float("inf")
        best_iteration = None
        last_failure = None

        for iteration in range(1, self.config.residual_iters + 1):
            try:
                low_rank, down_weight, up_weight = _truncated_svd(weight_hat - quantized_residual, rank)
            except (RuntimeError, ValueError) as exc:
                last_failure = (iteration, f"SVD failed: {exc}")
                del quantized_residual
                break
            del quantized_residual

            if not all(torch.isfinite(tensor).all() for tensor in (low_rank, down_weight, up_weight)):
                last_failure = (iteration, "SVD produced non-finite values")
                del low_rank, down_weight, up_weight
                break

            residual_weight = weight_hat - low_rank
            if not torch.isfinite(residual_weight).all():
                last_failure = (iteration, "residual computation produced non-finite values")
                del residual_weight, low_rank, down_weight, up_weight
                break

            materialized_residual = residual_weight.to(module.weight.dtype)
            del residual_weight, low_rank
            try:
                qdq_residual = residual_module.rtn_qdq_residual(materialized_residual, scheme)
            except (RuntimeError, ValueError) as exc:
                last_failure = (iteration, f"RTN residual QDQ failed: {exc}")
                del materialized_residual, down_weight, up_weight
                break

            if not torch.isfinite(qdq_residual).all():
                last_failure = (iteration, "RTN residual QDQ produced non-finite values")
                del qdq_residual, materialized_residual, down_weight, up_weight
                break

            quantized_residual = qdq_residual.float()
            del qdq_residual, materialized_residual
            deployed_down = down_weight.to(low_rank_dtype)
            deployed_up = up_weight.to(low_rank_dtype)
            deployed_low_rank = deployed_up.float() @ deployed_down.float()
            error = torch.sum((weight_hat - (quantized_residual + deployed_low_rank)).square())
            error_value = error.item()
            improved = False
            if math.isfinite(error_value):
                if error_value < best_error:
                    best_down = down_weight.clone()
                    best_up = up_weight.clone()
                    best_error = error_value
                    best_iteration = iteration
                    improved = True
            else:
                last_failure = (iteration, "reconstruction error is non-finite")

            del deployed_down, deployed_up, deployed_low_rank, down_weight, up_weight, error

            if self.config.residual_early_stop and best_down is not None and not improved:
                break

        if best_down is not None and best_up is not None:
            assert best_iteration is not None
            residual_weight = weight_hat - best_up @ best_down
            self._raise_if_nonfinite(module, best_iteration, residual_weight, best_down, best_up)
            return residual_weight, best_down, best_up

        failure_iteration, reason = last_failure or (self.config.residual_iters, "no finite candidate was produced")
        raise ValueError(
            f"SVDQuant residual iteration failed for module {self._module_name(module)!r} "
            f"at iteration {failure_iteration}: {reason}."
        )

    def _residual_quant_scheme(self, module: torch.nn.Linear) -> residual_module.ResidualQuantScheme:
        required = ("data_type", "bits", "group_size", "sym")
        missing = [attr for attr in required if not hasattr(module, attr) or getattr(module, attr) is None]
        if missing:
            raise ValueError(
                f"SVDQuant residual iteration requires data_type, bits, group_size, and sym for module "
                f"{self._module_name(module)!r}; missing: {', '.join(missing)}."
            )
        try:
            return residual_module.ResidualQuantScheme(**{attr: getattr(module, attr) for attr in required})
        except ValueError as exc:
            raise ValueError(
                f"Invalid residual quantization scheme for module {self._module_name(module)!r}: {exc}"
            ) from exc

    def _raise_if_nonfinite(self, module: torch.nn.Linear, iteration: int, *tensors: torch.Tensor) -> None:
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError(
                f"SVDQuant decomposition produced non-finite values for module {self._module_name(module)!r} "
                f"at iteration {iteration}."
            )

    def _module_name(self, module: torch.nn.Linear) -> str:
        return str(getattr(module, "global_name", module.__class__.__name__))

    def _build_smooth(self, module: torch.nn.Linear, weight: torch.Tensor) -> torch.Tensor:
        self._act_max.pop(id(module), None)
        return torch.ones(weight.shape[1], dtype=torch.float32, device=weight.device)

    def _new_linear_like(self, module: torch.nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None):
        residual = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        with torch.no_grad():
            residual.weight.copy_(weight.to(module.weight.dtype))
            if bias is not None:
                residual.bias.copy_(bias.detach().to(module.weight.dtype))
        return residual

    def _copy_quant_attrs(self, src: torch.nn.Module, dst: torch.nn.Module, suffix: str) -> None:
        for attr in _SCHEME_ATTRS:
            if hasattr(src, attr):
                setattr(dst, attr, getattr(src, attr))
        if hasattr(src, "global_name"):
            dst.global_name = f"{src.global_name}{suffix}"

    def _mark_unquantized(self, module: torch.nn.Module) -> None:
        module.bits = 16
        module.act_bits = 16

    def _resolve_low_rank_dtype(self, fallback: torch.dtype) -> torch.dtype:
        dtype = str(self.config.low_rank_dtype).lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        return fallback


def _set_child_module(root: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    if not name:
        return
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = (
            parent[int(part)] if part.isdigit() and isinstance(parent, torch.nn.Sequential) else getattr(parent, part)
        )
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, torch.nn.Sequential):
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def _truncated_svd(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a rank-limited reconstruction and its down/up factors."""
    out_features, in_features = weight.shape
    if rank == 0:
        low_rank = torch.zeros_like(weight)
        down_weight = torch.empty((0, in_features), dtype=weight.dtype, device=weight.device)
        up_weight = torch.empty((out_features, 0), dtype=weight.dtype, device=weight.device)
        return low_rank, down_weight, up_weight

    # Exact SVD is intentional for one-round compatibility and stable multi-round quality.
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    down_weight = vh[:rank, :]
    up_weight = u[:, :rank] * s[:rank].reshape(1, -1)
    return up_weight @ down_weight, down_weight, up_weight
