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

from contextlib import contextmanager

import torch

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

        if rank == 0:
            low_rank = torch.zeros_like(weight_hat)
            down_weight = torch.empty((0, in_features), dtype=weight_hat.dtype, device=weight_hat.device)
            up_weight = torch.empty((out_features, 0), dtype=weight_hat.dtype, device=weight_hat.device)
        else:
            u, s, vh = torch.linalg.svd(weight_hat, full_matrices=False)
            down_weight = vh[:rank, :]
            up_weight = u[:, :rank] * s[:rank].reshape(1, -1)
            low_rank = up_weight @ down_weight

        residual = self._new_linear_like(module, weight_hat - low_rank, module.bias)
        low_rank_dtype = self._resolve_low_rank_dtype(module.weight.dtype)
        lora_down = torch.nn.Linear(in_features, rank, bias=False, dtype=low_rank_dtype, device=module.weight.device)
        lora_up = torch.nn.Linear(rank, out_features, bias=False, dtype=low_rank_dtype, device=module.weight.device)
        with torch.no_grad():
            lora_down.weight.copy_(down_weight.to(low_rank_dtype))
            lora_up.weight.copy_(up_weight.to(low_rank_dtype))

        self._mark_unquantized(lora_down)
        self._mark_unquantized(lora_up)
        self._copy_quant_attrs(module, residual, suffix=".residual_linear")
        return SVDQuantLinear(residual, lora_down, lora_up, smooth.to(module.weight.dtype))

    def _build_smooth(self, module: torch.nn.Linear, weight: torch.Tensor) -> torch.Tensor:
        act = self._act_max.get(id(module))
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
        parent = parent[int(part)] if part.isdigit() and isinstance(parent, torch.nn.Sequential) else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, torch.nn.Sequential):
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)
