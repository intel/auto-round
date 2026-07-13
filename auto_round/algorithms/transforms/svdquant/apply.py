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

import torch

import auto_round.algorithms.transforms.svdquant.residual as residual_module
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.transforms.base import BaseWeightTransformer
from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
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


@register_pipeline_member(SVDQuantConfig)
class SVDQuantTransform(BaseWeightTransformer):
    """Apply SVDQuant decomposition before downstream RTN/SignRound quantization."""

    def __init__(self, config: SVDQuantConfig) -> None:
        super().__init__(config)
        self._act_max: dict[int, torch.Tensor] = {}

    @contextmanager
    def block_forward_hooks(self, ctx):
        handles = []
        if not self.config.smooth_enabled:
            yield handles
            return

        def collect_input_amax(module, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.shape[-1] != module.in_features:
                return
            amax = x.detach().abs().reshape(-1, x.shape[-1]).amax(dim=0).to(torch.float32).cpu()
            key = id(module)
            old = self._act_max.get(key)
            self._act_max[key] = amax if old is None else torch.maximum(old, amax)

        for name, module in ctx.block.named_modules():
            if self._is_target(name, module):
                handles.append(module.register_forward_hook(collect_input_amax))
        try:
            yield handles
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def pre_quantize_block(self, ctx) -> None:
        replacements = []
        for name, module in ctx.block.named_modules():
            if self._is_target(name, module):
                replacements.append((name, module))

        for name, module in replacements:
            _set_child_module(ctx.block, name, self._decompose_linear(module))

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
            residual_weight, down_weight, up_weight = self._iterate_residual(
                module, weight_hat, rank, low_rank_dtype
            )

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
        act = self._act_max.pop(id(module), None)
        if not self.config.smooth_enabled:
            return torch.ones(weight.shape[1], dtype=torch.float32, device=weight.device)
        if act is None:
            return torch.ones(weight.shape[1], dtype=torch.float32, device=weight.device)
        act = act.to(weight.device).clamp(min=self.config.smooth_eps)
        w = weight.abs().amax(dim=0).clamp(min=self.config.smooth_eps)
        alpha = self.config.smooth_alpha
        scale = (act.pow(alpha) / w.pow(1.0 - alpha)).clamp(min=self.config.smooth_eps)
        smooth = torch.reciprocal(scale)
        if not torch.isfinite(smooth).all():
            logger.warning_once("SVDQuant smooth scale contains non-finite values; falling back to identity smooth.")
            return torch.ones(weight.shape[1], dtype=torch.float32, device=weight.device)
        return smooth

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
