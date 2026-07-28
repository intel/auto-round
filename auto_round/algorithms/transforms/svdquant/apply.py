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

import torch

from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.transforms.base import BasePreprocessor
from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig
from auto_round.algorithms.transforms.svdquant.residual import (
    ResidualQuantScheme,
    iterate_residual_decomposition,
    truncated_svd,
)
from auto_round.algorithms.transforms.svdquant.smooth_adapters import SmoothSearchGroup, discover_svdquant_groups
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
_MXFP4_ALIASES = frozenset({"mx_fp", "mx_fp4", "mx_fp4e2m1"})


@register_pipeline_member(SVDQuantConfig)
class SVDQuantTransform(BasePreprocessor):
    """Split target linears into a quantized residual and an FP low-rank branch."""

    def __init__(self, config: SVDQuantConfig) -> None:
        super().__init__(config)
        self._configured_block_names: tuple[str, ...] = ()
        self._block_groups: dict[str, list[SmoothSearchGroup]] = {}

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
        for block_name in self._configured_block_names:
            block = self.model.get_submodule(block_name)
            self._block_groups[block_name] = discover_svdquant_groups(block, self._is_target)
        logger.info(
            "SVDQuant: resolved %d projection groups across %d blocks.",
            sum(len(groups) for groups in self._block_groups.values()),
            len(self._block_groups),
        )

    def register_fp_input_forward_hooks(self, block) -> list:
        if self.config.smooth_enabled:
            raise NotImplementedError("SVDQuant smooth calibration is not ported to the main architecture yet.")
        return []

    @torch.no_grad()
    def pre_quantize_block(self, ctx) -> None:
        if len(ctx.block_names) != 1:
            raise ValueError(f"SVDQuant requires one block at a time, got {ctx.block_names!r}.")
        if self.config.smooth_enabled:
            raise NotImplementedError("SVDQuant smooth calibration is not ported to the main architecture yet.")

        block_name = ctx.block_name
        block = ctx.model.get_submodule(block_name)
        groups = self._block_groups.get(block_name)
        if groups is None:
            groups = discover_svdquant_groups(block, self._is_target)
            self._block_groups[block_name] = groups

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
        self._block_groups.pop(ctx.block_name, None)

    def finalize_run(self) -> None:
        self._block_groups.clear()

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
    ) -> list[SVDQuantLinear]:
        wrappers = []
        rank = down.shape[0]
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
            wrappers.append(SVDQuantLinear(residual, lora_down, lora_up, smooth.to(projection.weight.dtype)))
        return wrappers

    def _shared_residual_scheme(self, group: SmoothSearchGroup) -> ResidualQuantScheme:
        schemes = tuple(self._residual_quant_scheme(projection) for projection in group.projections)
        if any(scheme != schemes[0] for scheme in schemes[1:]):
            raise ValueError(f"SVDQuant group {group.key!r} has inconsistent residual quantization schemes.")
        return schemes[0]

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
        if self.config.target_modules and not any(
            pattern in name or pattern in full_name for pattern in self.config.target_modules
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
        for attr in _SCHEME_ATTRS:
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
