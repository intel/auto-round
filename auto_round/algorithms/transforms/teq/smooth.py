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

"""Smoothing strategies for TEQ's trainable equivalent-transform scale."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from auto_round.algorithms.transforms.awq.smooth import (
    compute_layer_means,
    smooth_group_from_mapping,
)
from auto_round.algorithms.transforms.smoothing.calibration import move_to_device
from auto_round.algorithms.transforms.smoothing.engine import SmoothEngine
from auto_round.algorithms.transforms.smoothing.errors import NoFiniteCandidateError
from auto_round.logger import logger
from auto_round.utils.model import set_module

if TYPE_CHECKING:
    from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping
    from auto_round.algorithms.transforms.smoothing.groups import SmoothGroup
    from auto_round.algorithms.transforms.teq.base import TEQTransform


@dataclass
class TEQSmoothCalibration:
    """Calibration state shared by AWQ initialization and TEQ training."""

    samples: list[tuple[tuple[Any, ...], dict[str, Any]]]
    reference_outputs: list[torch.Tensor]
    initial_alpha: torch.Tensor
    activation_mean: torch.Tensor | None = None


@dataclass(frozen=True)
class TEQSmoothOutcome:
    """Selected transform and diagnostics from the TEQ smoothing pipeline."""

    alpha: torch.Tensor
    best_loss: float
    awq_loss: float | None
    trained: bool


def _iter_values(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_values(item)


def detect_batch_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    """Infer a replay batch size from the first tensor with a batch axis."""
    values = (*args, *kwargs.values())
    for value in _iter_values(values):
        if value.ndim >= 2:
            return value.shape[0]
    return None


def _slice_batch_value(value: Any, actual_batch: int, start: int, end: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim >= 2 and value.shape[0] == actual_batch:
            return value[start:end]
        return value
    if isinstance(value, tuple):
        return tuple(_slice_batch_value(item, actual_batch, start, end) for item in value)
    if isinstance(value, list):
        return [_slice_batch_value(item, actual_batch, start, end) for item in value]
    if isinstance(value, dict):
        return {key: _slice_batch_value(item, actual_batch, start, end) for key, item in value.items()}
    return value


def iter_batch_calls(
    stored_args: tuple[Any, ...],
    stored_kwargs: dict[str, Any],
    batch_size: int | None,
):
    """Yield full calls or deterministic micro-batches for parent replay."""
    actual_batch = detect_batch_size(stored_args, stored_kwargs)
    if batch_size is None or actual_batch is None or actual_batch <= batch_size:
        yield stored_args, stored_kwargs
        return
    for start in range(0, actual_batch, batch_size):
        end = min(start + batch_size, actual_batch)
        yield (
            tuple(_slice_batch_value(value, actual_batch, start, end) for value in stored_args),
            {key: _slice_batch_value(value, actual_batch, start, end) for key, value in stored_kwargs.items()},
        )


def detach_forward_value(value: Any, weight_dtype: torch.dtype | None = None) -> Any:
    """Detach a captured parent argument and move tensor leaves to host memory."""

    def detach(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if weight_dtype is not None and tensor.is_floating_point() and tensor.dtype != weight_dtype:
            tensor = tensor.to(weight_dtype)
        return tensor.to("cpu", copy=True)

    from auto_round.utils.model import map_nested_tensors

    if hasattr(value, "key_cache"):
        return None
    return map_nested_tensors(value, detach)


def _detect_sequence_length(values) -> int | None:
    for value in values:
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            return value.shape[1]
        if isinstance(value, (tuple, list)):
            nested = _detect_sequence_length(value)
            if nested is not None:
                return nested
        if isinstance(value, dict):
            nested = _detect_sequence_length(value.values())
            if nested is not None:
                return nested
    return None


def _slice_sequence_tensor(value: Any, actual_seq: int, seqlen: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 3 and value.shape[1] == actual_seq:
            return value[:, :seqlen]
        if value.ndim == 2 and value.shape[1] == actual_seq:
            return value[:, :seqlen]
        if value.ndim == 4 and value.shape[2] == actual_seq and value.shape[3] == actual_seq:
            return value[:, :, :seqlen, :seqlen]
        if value.ndim == 4 and value.shape[2] == actual_seq:
            return value[:, :, :seqlen]
        return value
    if isinstance(value, tuple):
        return tuple(_slice_sequence_tensor(item, actual_seq, seqlen) for item in value)
    if isinstance(value, list):
        return [_slice_sequence_tensor(item, actual_seq, seqlen) for item in value]
    if isinstance(value, dict):
        return {key: _slice_sequence_tensor(item, actual_seq, seqlen) for key, item in value.items()}
    return value


def truncate_forward_value(value: Any, seqlen: int | None) -> Any:
    """Cap sequence tokens using the same multidimensional shapes as AWQ."""
    if seqlen is None:
        return value
    actual_seq = _detect_sequence_length((value,))
    if actual_seq is None or actual_seq <= seqlen:
        return value
    return _slice_sequence_tensor(value, actual_seq, seqlen)


def _normalize_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def move_parent_value(value: Any, device: torch.device | str) -> Any:
    """Move captured replay inputs to the evaluation module's device."""
    return move_to_device(value, torch.device(device))


@torch.no_grad()
def run_parent_samples(
    owner: TEQTransform,
    parent: torch.nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    offload_to_cpu: bool = False,
) -> list[torch.Tensor]:
    """Replay FP parent calls and collect reference outputs."""
    param = next(parent.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    outputs = []
    for stored_args, stored_kwargs in samples:
        for micro_args, micro_kwargs in iter_batch_calls(stored_args, stored_kwargs, owner.teq_batch_size):
            call_args = tuple(move_parent_value(value, device) for value in micro_args)
            call_kwargs = {key: move_parent_value(value, device) for key, value in micro_kwargs.items()}
            output = _normalize_output(parent(*call_args, **call_kwargs)).detach()
            outputs.append(output.to("cpu", non_blocking=False) if offload_to_cpu else output)
    return outputs


@torch.no_grad()
def compute_parent_loss(
    owner: TEQTransform,
    parent: torch.nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    reference_outputs: list[torch.Tensor],
) -> float:
    """Compute the streamed parent-output MSE used by both TEQ phases."""
    param = next(parent.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    loss = torch.tensor(0.0, device=device)
    num_elements = torch.tensor(0, device=device, dtype=torch.long)
    output_idx = 0
    for stored_args, stored_kwargs in samples:
        for micro_args, micro_kwargs in iter_batch_calls(stored_args, stored_kwargs, owner.teq_batch_size):
            if output_idx >= len(reference_outputs):
                return float("inf")
            call_args = tuple(move_parent_value(value, device) for value in micro_args)
            call_kwargs = {key: move_parent_value(value, device) for key, value in micro_kwargs.items()}
            output = _normalize_output(parent(*call_args, **call_kwargs))
            reference = reference_outputs[output_idx].to(device, non_blocking=False)
            loss += F.mse_loss(output.float(), reference.float(), reduction="sum")
            num_elements += reference.numel()
            output_idx += 1
            del output, reference
    if output_idx != len(reference_outputs) or num_elements == 0:
        return float("inf")
    return (loss / num_elements).item()


def _prepare_wrapped_parent(owner: TEQTransform, mapping: ResolvedMapping, log_alpha: torch.nn.Parameter):
    original = dict(zip(mapping.balance_names, mapping.balance_layers))
    try:
        for name, layer in original.items():
            set_module(owner.model, name, owner._make_wrapper(layer, log_alpha))
        parent = owner.model.get_submodule(mapping.parent_name) if mapping.parent_name else owner.model
        return original, parent
    except Exception:
        _restore_layers(owner, original)
        raise


def _restore_layers(owner: TEQTransform, original: dict[str, torch.nn.Module]) -> None:
    for name, layer in original.items():
        set_module(owner.model, name, layer)


class TEQAWQInitStrategy:
    """AWQ-style scale initialization expressed as a SmoothEngine strategy."""

    def __init__(self, owner: TEQTransform, mapping: ResolvedMapping) -> None:
        self.owner = owner
        self.mapping = mapping
        self.calibration: TEQSmoothCalibration | None = None
        self._candidates = None

    def prepare(self, group: SmoothGroup, calibration: TEQSmoothCalibration) -> None:
        self.calibration = calibration
        if calibration.activation_mean is None:
            self._candidates = None
            return
        x_mean = calibration.activation_mean
        params = self.owner._resolve_params(self.mapping.balance_layers[0])
        group_size = self.owner._normalize_group_size(params["group_size"], -1)
        w_mean = compute_layer_means(self.mapping.balance_layers, group_size)
        n_grid = max(2, int(self.owner.awq_init_n_grid))

        def candidates():
            for index in range(n_grid):
                ratio = index / (n_grid - 1)
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
                scales = scales / (scales.max() * scales.min()).sqrt()
                scales = torch.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0)
                scales = scales.clamp(self.owner.min_scale, self.owner.max_scale)
                yield ratio, scales.detach().clone()

        self._candidates = candidates

    def candidates(self):
        if self._candidates is None:
            return iter(())
        return self._candidates()

    def score(self, candidate) -> float:
        _, alpha = candidate
        calibration = self.calibration
        log_alpha = torch.nn.Parameter(alpha.log())
        try:
            original, parent = _prepare_wrapped_parent(self.owner, self.mapping, log_alpha)
            loss = compute_parent_loss(
                self.owner,
                parent,
                calibration.samples,
                calibration.reference_outputs,
            )
        finally:
            if "original" in locals():
                _restore_layers(self.owner, original)
        return loss

    def apply(self, candidate) -> torch.Tensor:
        return candidate[1].detach().clone()

    def clear(self) -> None:
        self.calibration = None
        self._candidates = None


class TEQSmoothStrategy:
    """Gradient-trained TEQ policy with loss-checkpoint selection."""

    def __init__(self, owner: TEQTransform, mapping: ResolvedMapping) -> None:
        self.owner = owner
        self.mapping = mapping
        self.calibration: TEQSmoothCalibration | None = None
        self.best_alpha: torch.Tensor | None = None
        self.best_loss = float("inf")

    def prepare(self, group: SmoothGroup, calibration: TEQSmoothCalibration) -> None:
        self.calibration = calibration
        self.best_alpha = calibration.initial_alpha.detach().clone()

    def candidates(self):
        yield self.calibration.initial_alpha.detach().clone()

    def score(self, candidate) -> float:
        alpha = candidate.clamp(self.owner.min_scale, self.owner.max_scale)
        log_alpha = torch.nn.Parameter(alpha.log())
        if self.owner.optimizer_name == "lion":
            from auto_round.algorithms.transforms.teq.lion import Lion

            optimizer = Lion([log_alpha], lr=self.owner.lr, betas=(0.9, 0.99), weight_decay=0.0)
        else:
            optimizer = torch.optim.Adam([log_alpha], lr=self.owner.lr, betas=(0.9, 0.9), weight_decay=0.0)

        logger.info(
            "TEQ train '%s': optimizer=%s iters=%d lr=%.3e",
            self.mapping.smooth_name,
            self.owner.optimizer_name,
            self.owner.iters,
            self.owner.lr,
        )
        best_loss = float("inf")
        best_alpha = alpha.detach().clone()
        calibration = self.calibration
        try:
            original, parent = _prepare_wrapped_parent(self.owner, self.mapping, log_alpha)
            for step in range(self.owner.iters + 1):
                optimizer.zero_grad(set_to_none=True)
                has_grad_loss = False
                finite_loss = True
                output_idx = 0
                loss_sum = 0.0
                num_elements = 0
                param = next(parent.parameters(), None)
                device = param.device if param is not None else torch.device("cpu")
                for stored_args, stored_kwargs in calibration.samples:
                    for micro_args, micro_kwargs in iter_batch_calls(
                        stored_args, stored_kwargs, self.owner.teq_batch_size
                    ):
                        if output_idx >= len(calibration.reference_outputs):
                            finite_loss = False
                            break
                        call_args = tuple(move_parent_value(value, device) for value in micro_args)
                        call_kwargs = {key: move_parent_value(value, device) for key, value in micro_kwargs.items()}
                        ref_output = calibration.reference_outputs[output_idx]
                        output_idx += 1
                        with torch.enable_grad():
                            output = _normalize_output(parent(*call_args, **call_kwargs))
                            reference = ref_output.to(output.device).float()
                            loss = F.mse_loss(output.float(), reference, reduction="sum")
                        if loss.requires_grad and step < self.owner.iters:
                            has_grad_loss = True
                        if not torch.isfinite(loss):
                            logger.warning(
                                "TEQ: non-finite loss for '%s'; keeping best scale.",
                                self.mapping.smooth_name,
                            )
                            finite_loss = False
                            break
                        loss_sum += loss.detach().item()
                        num_elements += reference.numel()
                        if step < self.owner.iters and loss.requires_grad:
                            loss.backward()
                        del output, loss, reference, ref_output
                    if not finite_loss:
                        break
                if output_idx != len(calibration.reference_outputs):
                    finite_loss = False
                if not finite_loss:
                    break
                total_loss = loss_sum / max(1, num_elements)
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_alpha = log_alpha.detach().exp().clamp(self.owner.min_scale, self.owner.max_scale).clone()
                if step == self.owner.iters:
                    break
                if not has_grad_loss:
                    logger.warning(
                        "TEQ: loss for '%s' is not differentiable; keeping the current scale.",
                        self.mapping.smooth_name,
                    )
                    break
                for param_group in optimizer.param_groups:
                    for parameter in param_group["params"]:
                        if parameter.grad is not None:
                            parameter.grad.div_(max(1, num_elements))
                optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(math.log(self.owner.min_scale), math.log(self.owner.max_scale))
        finally:
            if "original" in locals():
                _restore_layers(self.owner, original)

        self.best_alpha = best_alpha
        self.best_loss = best_loss
        return best_loss

    def apply(self, candidate) -> torch.Tensor:
        scales = self.best_alpha if self.best_alpha is not None else candidate
        self.owner._apply_scales(self.mapping, scales)
        return scales.detach().clone()

    def clear(self) -> None:
        self.calibration = None


def select_awq_initial_alpha(
    owner: TEQTransform,
    mapping: ResolvedMapping,
    calibration: TEQSmoothCalibration,
) -> tuple[torch.Tensor, float | None]:
    """Run the optional AWQ grid through SmoothEngine."""
    if not owner.awq_init or calibration.activation_mean is None:
        return calibration.initial_alpha, None
    engine = SmoothEngine()
    group = smooth_group_from_mapping(mapping)
    strategy = TEQAWQInitStrategy(owner, mapping)
    try:
        result = engine.search(group, calibration, strategy, require_finite=False, log_search=False)
    except NoFiniteCandidateError:
        logger.warning("TEQ: AWQ-init grid search failed for '%s'; using default alpha.", mapping.smooth_name)
        return calibration.initial_alpha, None
    return result.candidate[1].detach().clone(), result.error


def run_teq_smoothing(
    owner: TEQTransform,
    mapping: ResolvedMapping,
    calibration: TEQSmoothCalibration,
) -> TEQSmoothOutcome:
    """Select an AWQ initialization, train TEQ, and fold the selected scale."""
    initial_alpha, awq_loss = select_awq_initial_alpha(owner, mapping, calibration)
    trained_calibration = replace(calibration, initial_alpha=initial_alpha)
    engine = SmoothEngine()
    strategy = TEQSmoothStrategy(owner, mapping)
    scales = engine.run(smooth_group_from_mapping(mapping), trained_calibration, strategy)
    best_loss = strategy.best_loss
    if awq_loss is not None:
        owner._log_scale_summary("awq_init", mapping, awq_loss, initial_alpha)
    owner._log_scale_summary("train", mapping, best_loss, scales)
    return TEQSmoothOutcome(scales, best_loss, awq_loss, owner.iters > 0)
