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
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

import auto_round.algorithms.transforms.svdquant.residual as residual_module
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.transforms.base import BasePreprocessor
from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
from auto_round.algorithms.transforms.svdquant.residual import (
    ActivationQuantScheme,
    ResidualQuantScheme,
    iterate_residual_decomposition,
    truncated_svd,
)
from auto_round.algorithms.transforms.svdquant.smooth import (
    SmoothCandidate,
    absmax_channel_span,
    build_alpha_beta_candidates,
    build_smooth_scale,
    select_best_layer_candidate,
    summarize_smooth_scale,
    validate_smooth_scale_for_deployment,
)
from auto_round.algorithms.transforms.svdquant.smooth_adapters import SmoothSearchGroup, discover_svdquant_groups
from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils.model import map_nested_tensors

_SCHEME_ATTRS = set(QuantizationScheme.get_attributes())
_RUNTIME_QUANT_ATTRS = {"scale_dtype", "weight_global_scale", "tuning_device"}
_MXFP4_ALIASES = frozenset({"mx_fp", "mx_fp4", "mx_fp4e2m1"})


def _detach_to_cpu(value: Any) -> Any:
    return map_nested_tensors(value, lambda tensor: tensor.detach().to("cpu", copy=True))


def _move_to_device(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    def move(tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
        return tensor.to(device=device, dtype=target_dtype)

    return map_nested_tensors(value, move)


@dataclass
class CapturedEvaluation:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any


@dataclass
class SmoothGroupCalibration:
    group: SmoothSearchGroup
    limit: int
    projection_inputs: list[torch.Tensor] = field(default_factory=list)
    evaluation_calls: list[CapturedEvaluation] = field(default_factory=list)
    seen_calls: int = 0
    pending_slot: int | None = None
    pending_input: torch.Tensor | None = None
    random: random.Random = field(default_factory=lambda: random.Random(0))

    def begin_call(self, inputs: torch.Tensor) -> None:
        self.seen_calls += 1
        if len(self.projection_inputs) < self.limit:
            slot = len(self.projection_inputs)
        else:
            candidate = self.random.randrange(self.seen_calls)
            slot = candidate if candidate < self.limit else None
        self.pending_slot = slot
        self.pending_input = _detach_to_cpu(inputs) if slot is not None else None

    def finish_call(self, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        slot = self.pending_slot
        captured_input = self.pending_input
        self.pending_slot = None
        self.pending_input = None
        if slot is None or captured_input is None:
            return
        captured = CapturedEvaluation(_detach_to_cpu(args), _detach_to_cpu(kwargs), _detach_to_cpu(output))
        if slot == len(self.projection_inputs):
            self.projection_inputs.append(captured_input)
            self.evaluation_calls.append(captured)
        else:
            self.projection_inputs[slot] = captured_input
            self.evaluation_calls[slot] = captured


@register_pipeline_member(SVDQuantConfig)
class SVDQuantTransform(BasePreprocessor):
    """Split target linears into a quantized residual and an FP low-rank branch."""

    def __init__(self, config: SVDQuantConfig) -> None:
        super().__init__(config)
        self._configured_block_names: tuple[str, ...] = ()
        self._block_groups: dict[str, list[SmoothSearchGroup]] = {}
        self._smooth_calibration: dict[str, SmoothGroupCalibration] = {}
        self._target_modules = config.target_modules
        if self._target_modules is None and config.model_adapter == "flux":
            from auto_round.export.svdquant_adapters import FLUX_SVDQUANT_TARGET_MODULES

            self._target_modules = FLUX_SVDQUANT_TARGET_MODULES

    def bind(self, orchestrator) -> None:
        super().bind(orchestrator)
        nblocks = getattr(orchestrator, "nblocks", 1)
        if nblocks != 1:
            raise ValueError(f"SVDQuant requires nblocks=1, got nblocks={nblocks}.")
        quant_block_list = getattr(orchestrator, "quant_block_list", None) or ()
        self._configured_block_names = tuple(
            block_name for block_group in quant_block_list for block_name in block_group
        )

    def prepare_run(self, composer=None) -> None:
        self._block_groups.clear()
        if self.model is None:
            return
        self._resolve_model_adapter(self.model)
        for block_name in self._configured_block_names:
            block = self.model.get_submodule(block_name)
            self._block_groups[block_name] = discover_svdquant_groups(block, self._is_target)
        logger.info(
            "SVDQuant: resolved %d projection groups across %d blocks.",
            sum(len(groups) for groups in self._block_groups.values()),
            len(self._block_groups),
        )

    def _resolve_model_adapter(self, model: torch.nn.Module | None, block: torch.nn.Module | None = None) -> str:
        model_adapter = self.config.model_adapter or "auto"
        if model_adapter == "auto":
            model_config = getattr(model, "config", None)
            class_name = (
                model_config.get("_class_name", type(model).__name__)
                if hasattr(model_config, "get")
                else type(model).__name__
            )
            if "fluxtransformer" in str(class_name).lower():
                model_adapter = "flux"
            elif block is not None and block.__class__.__name__ in {
                "FluxTransformerBlock",
                "FluxSingleTransformerBlock",
            }:
                model_adapter = "flux"
        if model is not None:
            model._autoround_svdquant_model_adapter = model_adapter
        if model_adapter == "flux":
            from auto_round.algorithms.transforms.svdquant.smooth_adapters.flux import warn_if_unverified_flux_model

            if model is not None:
                warn_if_unverified_flux_model(model)
        self._target_modules = self.config.target_modules
        if self._target_modules is None and model_adapter == "flux":
            from auto_round.export.svdquant_adapters import FLUX_SVDQUANT_TARGET_MODULES

            self._target_modules = FLUX_SVDQUANT_TARGET_MODULES
        return model_adapter

    def register_fp_input_forward_hooks(self, block) -> list:
        if not self.config.smooth_enabled:
            return []
        self._resolve_model_adapter(self.model, block)
        self._clear_smooth_calibration()
        block_name = str(getattr(block, "global_name", ""))
        groups = self._block_groups.get(block_name)
        if groups is None:
            groups = discover_svdquant_groups(block, self._is_target)
            self._block_groups[block_name] = groups
        self._smooth_calibration = {
            group.key: SmoothGroupCalibration(group, self.config.smooth_max_calibration_calls) for group in groups
        }

        projection_owners = {}
        evaluation_owners = {}
        modules = {}
        for calibration in self._smooth_calibration.values():
            projection_module = calibration.group.projection_input_module
            evaluation_module = calibration.group.evaluation_module
            projection_owners.setdefault(id(projection_module), []).append(calibration)
            evaluation_owners.setdefault(id(evaluation_module), []).append(calibration)
            modules[id(projection_module)] = projection_module
            modules[id(evaluation_module)] = evaluation_module

        def collect_calibration(module, inputs, kwargs, output):
            for calibration in projection_owners.get(id(module), ()):
                if inputs and torch.is_tensor(inputs[0]):
                    value = inputs[0]
                    if value.shape[-1] == calibration.group.projections[0].in_features:
                        calibration.begin_call(value)
            for calibration in evaluation_owners.get(id(module), ()):
                calibration.finish_call(inputs, kwargs, output)

        return [module.register_forward_hook(collect_calibration, with_kwargs=True) for module in modules.values()]

    @torch.no_grad()
    def pre_quantize_block(self, ctx) -> None:
        if len(ctx.block_names) != 1:
            raise ValueError(f"SVDQuant requires one block at a time, got {ctx.block_names!r}.")
        block_name = ctx.block_name
        block = ctx.model.get_submodule(block_name)
        self._resolve_model_adapter(ctx.model, block)
        groups = self._block_groups.get(block_name)
        if groups is None:
            groups = discover_svdquant_groups(block, self._is_target)
            self._block_groups[block_name] = groups

        if self.config.smooth_enabled:
            self._pre_quantize_smoothed_block(block, groups)
            return

        local_names = {id(module): name for name, module in block.named_modules() if name}
        replacements = []
        for group in groups:
            wrappers = self._decompose_group(group)
            for projection, wrapper in zip(group.projections, wrappers):
                local_name = local_names.get(id(projection))
                if local_name is None:
                    raise ValueError(f"SVDQuant could not locate projection {self._module_name(projection)!r}.")
                replacements.append((local_name, wrapper))

        for local_name, wrapper in replacements:
            _set_child_module(block, local_name, wrapper)

    def post_quantize_block(self, ctx) -> None:
        self._clear_smooth_calibration()
        self._block_groups.pop(ctx.block_name, None)

    def finalize_run(self) -> None:
        self._clear_smooth_calibration()
        self._block_groups.clear()

    def _clear_smooth_calibration(self) -> None:
        self._smooth_calibration.clear()

    def _pre_quantize_smoothed_block(self, block: torch.nn.Module, groups: list[SmoothSearchGroup]) -> None:
        if not self._smooth_calibration:
            raise ValueError("SVDQuant smooth calibration inputs are missing for the current block.")
        local_names = {id(module): name for name, module in block.named_modules() if name}
        try:
            selected_scales = {}
            for group in groups:
                calibration = self._smooth_calibration.get(group.key)
                if calibration is None or not calibration.projection_inputs or not calibration.evaluation_calls:
                    raise ValueError(f"SVDQuant smooth calibration inputs are missing for group {group.key!r}.")
                selected_scales[group.key] = self._search_group_scale(calibration, block, local_names)

            replacements = []
            for group in groups:
                calibration = self._smooth_calibration[group.key]
                wrappers = self._decompose_smoothed_group(calibration, selected_scales[group.key], block, local_names)
                for projection, wrapper in zip(group.projections, wrappers):
                    local_name = local_names.get(id(projection))
                    if local_name is None:
                        raise ValueError(f"SVDQuant could not locate projection {self._module_name(projection)!r}.")
                    replacements.append((local_name, wrapper))
            for local_name, wrapper in replacements:
                _set_child_module(block, local_name, wrapper)
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
            scale = build_smooth_scale(x_span, w_span, alpha, beta, eps=self.config.smooth_eps)
            try:
                scale = validate_smooth_scale_for_deployment(
                    scale, dtype=group.projections[0].weight.dtype, module_name=group.key
                ).to(torch.float32)
                error = self._score_group_wrappers(
                    calibration, self._candidate_group_wrappers(group, scale), block, local_names
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.debug("Skipping SVDQuant smooth candidate (%s, %s) for %s: %s", alpha, beta, group.key, exc)
                error = float("inf")
            candidate = SmoothCandidate(alpha, beta, scale)
            scored.append((candidate, error))
        selected = select_best_layer_candidate(scored, module_name=group.key)
        error = next(error for candidate, error in reversed(scored) if candidate is selected)
        self._log_selected_smooth_candidate(group.key, selected, error)
        return selected.scale

    @staticmethod
    def _log_selected_smooth_candidate(module_name: str, candidate: SmoothCandidate, error: float) -> None:
        stats = summarize_smooth_scale(candidate.scale)
        logger.info(
            "SVDQuant smooth selected for %s: alpha=%.6g beta=%.6g error=%.6g "
            "scale_min=%.6g scale_max=%.6g scale_ratio=%.6g below_1e-3=%d above_20=%d",
            module_name,
            candidate.alpha,
            candidate.beta,
            error,
            stats.minimum,
            stats.maximum,
            stats.ratio,
            stats.below_min_count,
            stats.above_max_count,
        )
        if stats.below_min_count or stats.above_max_count:
            logger.warning(
                "SVDQuant smooth scale for %s contains extreme values: below_1e-3=%d above_20=%d",
                module_name,
                stats.below_min_count,
                stats.above_max_count,
            )

    def _score_group_wrappers(
        self,
        calibration: SmoothGroupCalibration,
        wrappers: list[SVDQuantLinear],
        block: torch.nn.Module,
        local_names: dict[int, str],
    ) -> float:
        group = calibration.group
        replacements = []
        for projection, wrapper in zip(group.projections, wrappers):
            local_name = local_names.get(id(projection))
            if local_name is None:
                raise ValueError(f"SVDQuant could not locate projection {self._module_name(projection)!r}.")
            replacements.append((local_name, projection, wrapper))
        try:
            for local_name, _, wrapper in replacements:
                _set_child_module(block, local_name, wrapper)
            error = torch.zeros((), dtype=torch.float64)
            for call in calibration.evaluation_calls:
                evaluation_module = group.evaluation_module
                if len(group.projections) == 1 and evaluation_module is group.projections[0]:
                    evaluation_module = wrappers[0]
                device = group.projections[0].weight.device
                dtype = group.projections[0].weight.dtype
                args = _move_to_device(call.args, device, dtype)
                kwargs = group.filter_evaluation_kwargs(_move_to_device(call.kwargs, device, dtype))
                actual = group.normalize_output(evaluation_module(*args, **kwargs))
                reference = tuple(tensor.to(device) for tensor in group.normalize_output(call.output))
                if len(actual) != len(reference):
                    raise ValueError("SVDQuant smooth output tensor count changed.")
                for actual_tensor, reference_tensor in zip(actual, reference):
                    if actual_tensor.shape != reference_tensor.shape:
                        raise ValueError("SVDQuant smooth output tensor shape changed.")
                    error += torch.sum((actual_tensor.float() - reference_tensor.float()).square()).double().cpu()
            return error.item()
        finally:
            for local_name, projection, _ in replacements:
                _set_child_module(block, local_name, projection)

    def _candidate_group_wrappers(self, group: SmoothSearchGroup, scale: torch.Tensor) -> list[SVDQuantLinear]:
        weights = [
            projection.weight.detach().to(torch.float32) * scale.to(projection.weight.device)
            for projection in group.projections
        ]
        stacked = torch.cat(weights, dim=0)
        output_sizes = [projection.out_features for projection in group.projections]
        rank = min(self.config.rank, *stacked.shape)
        _, down, up = truncated_svd(stacked, rank)
        low_rank_dtype = self._resolve_low_rank_dtype(group.projections[0].weight.dtype)
        deployed_down = down.to(low_rank_dtype)
        deployed_up = up.to(low_rank_dtype)
        low_rank_parts = (deployed_up.float() @ deployed_down.float()).split(output_sizes, dim=0)
        residuals = []
        for projection, weight, low_rank in zip(group.projections, weights, low_rank_parts):
            residual = (weight - low_rank).to(projection.weight.dtype)
            residuals.append(residual_module.rtn_qdq_residual(residual, self._residual_quant_scheme(projection)))
        return self._build_group_wrappers(
            group,
            residuals,
            deployed_down,
            deployed_up.split(output_sizes, dim=0),
            scale,
            activation_scheme=self._group_activation_quant_scheme(group),
        )

    def _decompose_smoothed_group(
        self,
        calibration: SmoothGroupCalibration,
        scale: torch.Tensor,
        block: torch.nn.Module,
        local_names: dict[int, str],
    ) -> list[SVDQuantLinear]:
        group = calibration.group
        weights = [
            projection.weight.detach().to(torch.float32) * scale.to(projection.weight.device)
            for projection in group.projections
        ]
        stacked = torch.cat(weights, dim=0)
        output_sizes = [projection.out_features for projection in group.projections]
        rank = min(self.config.rank, *stacked.shape)
        low_rank_dtype = self._resolve_low_rank_dtype(group.projections[0].weight.dtype)
        if self.config.residual_iters == 1:
            _, down, up = truncated_svd(stacked, rank)
            deployed_down = down.to(low_rank_dtype)
            deployed_up = up.to(low_rank_dtype)
            low_rank = deployed_up.float() @ deployed_down.float()
            residuals = [
                residual.to(projection.weight.dtype)
                for residual, projection in zip(stacked.sub(low_rank).split(output_sizes, dim=0), group.projections)
            ]
        else:
            residuals, deployed_down, deployed_up = self._iterate_smoothed_group_residual(
                calibration, block, local_names, stacked, rank, low_rank_dtype, scale
            )
        return self._build_group_wrappers(
            group, residuals, deployed_down, deployed_up.split(output_sizes, dim=0), scale
        )

    def _iterate_smoothed_group_residual(
        self,
        calibration: SmoothGroupCalibration,
        block: torch.nn.Module,
        local_names: dict[int, str],
        stacked: torch.Tensor,
        rank: int,
        low_rank_dtype: torch.dtype,
        scale: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        group = calibration.group
        output_sizes = [projection.out_features for projection in group.projections]
        quantized_residual = torch.zeros_like(stacked)
        best = None
        best_error = float("inf")
        activation_scheme = self._group_activation_quant_scheme(group)
        for iteration in range(1, self.config.residual_iters + 1):
            _, down, up = truncated_svd(stacked - quantized_residual, rank)
            deployed_down = down.to(low_rank_dtype)
            deployed_up = up.to(low_rank_dtype)
            low_rank = deployed_up.float() @ deployed_down.float()
            residual_parts = (stacked - low_rank).split(output_sizes, dim=0)
            qdq_residuals = [
                residual_module.rtn_qdq_residual(
                    residual.to(projection.weight.dtype), self._residual_quant_scheme(projection)
                )
                for residual, projection in zip(residual_parts, group.projections)
            ]
            quantized_residual = torch.cat([residual.float() for residual in qdq_residuals], dim=0)
            wrappers = self._build_group_wrappers(
                group,
                qdq_residuals,
                deployed_down,
                deployed_up.split(output_sizes, dim=0),
                scale,
                activation_scheme=activation_scheme,
            )
            error = self._score_group_wrappers(calibration, wrappers, block, local_names)
            accepted = math.isfinite(error) and error <= best_error
            if accepted:
                best = (deployed_down.clone(), deployed_up.clone(), iteration)
                best_error = error
            elif self.config.residual_early_stop and best is not None:
                logger.info(
                    "SVDQuant residual early stop for %s at iteration %d: output error %.6g > best %.6g",
                    group.key,
                    iteration,
                    error,
                    best_error,
                )
                break
        if best is None:
            raise ValueError(f"SVDQuant residual iteration failed for group {group.key!r}.")
        deployed_down, deployed_up, selected_iteration = best
        logger.info(
            "SVDQuant residual selected iteration %d/%d for %s: output error %.6g",
            selected_iteration,
            self.config.residual_iters,
            group.key,
            best_error,
        )
        low_rank = deployed_up.float() @ deployed_down.float()
        residuals = [
            residual.to(projection.weight.dtype)
            for residual, projection in zip((stacked - low_rank).split(output_sizes, dim=0), group.projections)
        ]
        return residuals, deployed_down, deployed_up

    def _decompose_group(self, group: SmoothSearchGroup) -> list[SVDQuantLinear]:
        weights = [projection.weight.detach().to(torch.float32) for projection in group.projections]
        stacked = torch.cat(weights, dim=0)
        output_sizes = [projection.out_features for projection in group.projections]
        rank = min(self.config.rank, *stacked.shape)
        low_rank_dtype = self._resolve_low_rank_dtype(group.projections[0].weight.dtype)

        if self.config.residual_iters == 1:
            _, down, up = truncated_svd(stacked, rank)
            deployed_down = down.to(low_rank_dtype)
            deployed_up = up.to(low_rank_dtype)
            deployed_low_rank = deployed_up.float() @ deployed_down.float()
            residuals = [
                residual.to(projection.weight.dtype)
                for residual, projection in zip(
                    (stacked - deployed_low_rank).split(output_sizes, dim=0), group.projections
                )
            ]
        else:
            scheme = self._shared_residual_scheme(group)
            result = iterate_residual_decomposition(
                stacked,
                rank=rank,
                scheme=scheme,
                iterations=self.config.residual_iters,
                early_stop=self.config.residual_early_stop,
                residual_dtype=group.projections[0].weight.dtype,
                low_rank_dtype=low_rank_dtype,
            )
            deployed_down = result.down
            deployed_up = result.up
            residuals = [
                residual.to(projection.weight.dtype)
                for residual, projection in zip(result.residual.split(output_sizes, dim=0), group.projections)
            ]
            logger.info(
                "SVDQuant residual selected iteration %d/%d for %s: weight error %.6g",
                result.selected_iteration,
                self.config.residual_iters,
                group.key,
                result.error,
            )

        up_parts = deployed_up.split(output_sizes, dim=0)
        smooth = torch.ones(stacked.shape[1], device=stacked.device, dtype=torch.float32)
        return self._build_group_wrappers(group, residuals, deployed_down, up_parts, smooth)

    def _build_group_wrappers(
        self,
        group: SmoothSearchGroup,
        residuals: list[torch.Tensor],
        down: torch.Tensor,
        up_parts: tuple[torch.Tensor, ...],
        smooth: torch.Tensor,
        activation_scheme: ActivationQuantScheme | None = None,
    ) -> list[SVDQuantLinear]:
        wrappers = []
        rank = down.shape[0]
        input_smooth = smooth.reciprocal()
        for projection, residual_weight, up in zip(group.projections, residuals, up_parts):
            residual = self._new_linear_like(projection, residual_weight, projection.bias)
            lora_down = torch.nn.Linear(
                projection.in_features,
                rank,
                bias=False,
                dtype=down.dtype,
                device=projection.weight.device,
            )
            lora_up = torch.nn.Linear(
                rank,
                projection.out_features,
                bias=False,
                dtype=up.dtype,
                device=projection.weight.device,
            )
            lora_down.weight.copy_(down)
            lora_up.weight.copy_(up)
            self._mark_unquantized(lora_down)
            self._mark_unquantized(lora_up)
            self._copy_quant_attrs(projection, residual, suffix=".residual_linear")
            activation_qdq = (
                None
                if activation_scheme is None
                else partial(residual_module.rtn_qdq_activation, scheme=activation_scheme)
            )
            wrappers.append(
                SVDQuantLinear(
                    residual,
                    lora_down,
                    lora_up,
                    input_smooth.to(projection.weight.dtype),
                    activation_qdq=activation_qdq,
                )
            )
        return wrappers

    def _shared_residual_scheme(self, group: SmoothSearchGroup) -> ResidualQuantScheme:
        schemes = tuple(self._residual_quant_scheme(projection) for projection in group.projections)
        if any(scheme != schemes[0] for scheme in schemes[1:]):
            raise ValueError(f"SVDQuant group {group.key!r} has inconsistent residual quantization schemes.")
        return schemes[0]

    def _group_activation_quant_scheme(self, group: SmoothSearchGroup) -> ActivationQuantScheme:
        schemes = tuple(self._activation_quant_scheme(projection) for projection in group.projections)
        if any(scheme != schemes[0] for scheme in schemes[1:]):
            raise ValueError(f"SVDQuant group {group.key!r} has inconsistent activation quantization schemes.")
        return schemes[0]

    def _activation_quant_scheme(self, module: torch.nn.Linear) -> ActivationQuantScheme:
        attributes = {
            "data_type": "act_data_type",
            "bits": "act_bits",
            "group_size": "act_group_size",
            "sym": "act_sym",
        }
        missing = [
            source for source in attributes.values() if not hasattr(module, source) or getattr(module, source) is None
        ]
        if missing:
            raise ValueError(
                f"SVDQuant smooth calibration requires a complete activation scheme for "
                f"{self._module_name(module)!r}; missing: {', '.join(missing)}."
            )
        return ActivationQuantScheme(**{target: getattr(module, source) for target, source in attributes.items()})

    def _residual_quant_scheme(self, module: torch.nn.Linear) -> ResidualQuantScheme:
        required = ("data_type", "bits", "group_size", "sym")
        missing = [attr for attr in required if not hasattr(module, attr) or getattr(module, attr) is None]
        if missing:
            raise ValueError(
                f"SVDQuant residual iteration requires a complete quantization scheme for "
                f"{self._module_name(module)!r}; missing: {', '.join(missing)}."
            )
        return ResidualQuantScheme(**{attr: getattr(module, attr) for attr in required})

    def _is_target(self, name: str, module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        full_name = str(getattr(module, "global_name", name))
        if self._target_modules and not any(
            pattern in name or pattern in full_name for pattern in self._target_modules
        ):
            return False
        if self.config.exclude_modules and any(
            pattern in name or pattern in full_name for pattern in self.config.exclude_modules
        ):
            return False
        return True

    @staticmethod
    def _new_linear_like(module: torch.nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None):
        residual = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        residual.weight.copy_(weight.to(module.weight.dtype))
        if bias is not None:
            residual.bias.copy_(bias.detach().to(module.weight.dtype))
        return residual

    @staticmethod
    def _mark_unquantized(module: torch.nn.Module) -> None:
        module.bits = 16
        module.act_bits = 16

    @staticmethod
    def _copy_quant_attrs(src: torch.nn.Module, dst: torch.nn.Module, suffix: str) -> None:
        for attr in _SCHEME_ATTRS | _RUNTIME_QUANT_ATTRS:
            if hasattr(src, attr):
                setattr(dst, attr, getattr(src, attr))
        if getattr(dst, "bits", None) == 4 and getattr(dst, "data_type", None) in _MXFP4_ALIASES:
            dst.data_type = f"{dst.data_type}_rceil"
        if getattr(dst, "act_bits", None) == 4 and getattr(dst, "act_data_type", None) in _MXFP4_ALIASES:
            dst.act_data_type = f"{dst.act_data_type}_rceil"
        if hasattr(src, "global_name"):
            dst.global_name = f"{src.global_name}{suffix}"

    def _resolve_low_rank_dtype(self, fallback: torch.dtype) -> torch.dtype:
        dtype = str(self.config.low_rank_dtype).lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        return fallback

    @staticmethod
    def _module_name(module: torch.nn.Module) -> str:
        return str(getattr(module, "global_name", module.__class__.__name__))


def _set_child_module(root: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = (
            parent[int(part)]
            if part.isdigit() and isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential))
            else getattr(parent, part)
        )
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)):
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)
