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

import inspect
import re
from typing import TYPE_CHECKING

import torch

from auto_round.algorithms.transforms.smoothing.groups import SmoothGroup
from auto_round.algorithms.transforms.smoothing.replay import restore_weights
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

if TYPE_CHECKING:
    from auto_round.algorithms.transforms.awq.base import AWQTransform
    from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping


# Known normalization classes whose ``forward`` computes
# ``output = (1 + weight) * x_norm`` (Gemma-style "unit-offset" RMSNorm) rather
# than the standard ``output = weight * x_norm``. Folding an AWQ smoothing scale
# ``s`` into such a layer requires ``weight <- (1 + weight) / s - 1`` instead of
# ``weight <- weight / s``; using the wrong fold silently breaks AWQ's output
# invariance and severely degrades accuracy (e.g. Qwen3.5, Gemma2/3, Qwen3-Next).
_UNIT_OFFSET_RMSNORM_NAMES = frozenset(
    {
        "GemmaRMSNorm",
        "Gemma2RMSNorm",
        "Gemma3RMSNorm",
        "Gemma3TextRMSNorm",
        "Qwen3_5RMSNorm",
        "Qwen3_5MoeRMSNorm",
        "Qwen3NextRMSNorm",
    }
)

# Detects ``1 + self.weight`` / ``self.weight + 1`` in a norm's forward source.
_UNIT_OFFSET_SRC_RE = re.compile(r"1(\.0)?\s*\+\s*self\.weight|self\.weight(\.float\(\))?\s*\+\s*1")

# Cache the unit-offset decision per norm class to avoid repeated source parsing.
_unit_offset_cache: dict[type, bool] = {}


def _rmsnorm_has_unit_offset(module: torch.nn.Module) -> bool:
    """Return True if ``module`` applies a Gemma-style ``(1 + weight)`` gain.

    Uses a fast class-name allowlist, falling back to source inspection of the
    module's ``forward`` so newly-added Gemma-style norms are detected without a
    code change. Result is cached per class.
    """
    cls = type(module)
    cached = _unit_offset_cache.get(cls)
    if cached is not None:
        return cached
    result = cls.__name__ in _UNIT_OFFSET_RMSNORM_NAMES
    if not result:
        try:
            src = inspect.getsource(cls.forward)
            result = bool(_UNIT_OFFSET_SRC_RE.search(src))
        except (OSError, TypeError):
            result = False
    _unit_offset_cache[cls] = result
    return result


def smooth_group_from_mapping(mapping: ResolvedMapping) -> SmoothGroup:
    return SmoothGroup(
        key=mapping.smooth_name,
        projection_names=tuple(mapping.balance_names),
        projections=tuple(mapping.balance_layers),
        projection_input_key=mapping.activation_hook_target or mapping.balance_names[0],
        projection_input_module=mapping.balance_layers[0],
        evaluation_input_key=mapping.parent_name,
        evaluation_module=mapping.parent,
    )


class AWQSmoothStrategy:
    """AWQ statistics, ratio candidates, QDQ loss and scale folding policy."""

    def __init__(self, owner: AWQTransform, mapping: ResolvedMapping) -> None:
        self.owner = owner
        self.mapping = mapping
        self._candidates = None
        self._score = None

    def prepare(self, group: SmoothGroup, x_mean: torch.Tensor) -> None:
        mapping = self.mapping
        owner = self.owner
        device = mapping.balance_layers[0].weight.device
        x_mean = x_mean.to(device)

        bl_params = {bl: owner._qdq_tool.resolve_params(bl) for bl in mapping.balance_layers}
        group_size = owner._normalize_group_size(bl_params[mapping.balance_layers[0]]["group_size"], -1)
        if owner.duo_scaling is not False:
            w_mean = owner._compute_layer_means(mapping.balance_layers, group_size).to(device)

        parent_kwargs_list = owner._parent_args_cache.get(mapping.parent, [])
        use_parent_forward = len(parent_kwargs_list) > 0

        if use_parent_forward:
            fp16_outputs = owner._run_parent_samples(
                mapping.parent,
                parent_kwargs_list,
                offload_to_cpu=owner._smooth_batch_size is not None,
            )
            if not fp16_outputs or all(f.numel() == 0 for f in fp16_outputs):
                use_parent_forward = False

        orig_state = {bl: bl.weight.data.clone() for bl in mapping.balance_layers}
        if not use_parent_forward:
            orig_weights = orig_state  # same reference is fine

        # Resolve each balance layer's quant functions once, then reuse them in
        # the grid-search loop. Normal AWQ flow requires one mapping to have
        # compatible quant params, but keeping this per-layer avoids hidden
        # coupling to the first layer and makes direct calls robust.
        bl_quant_funcs = {bl: owner._qdq_tool.resolve_quant_funcs(bl_params[bl]) for bl in mapping.balance_layers}

        def candidates():
            for ratio, use_duo in owner._get_grid_search_params():
                if use_duo:
                    scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
                else:
                    scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                scales[torch.isinf(scales)] = 1
                scales[torch.isnan(scales)] = 1
                yield ratio, scales

        def score(candidate):
            _, scales = candidate
            scales_view = scales.view(1, -1).to(device)
            with restore_weights(orig_state):
                if use_parent_forward:
                    # Quantize each balance layer's smoothed weight and write the
                    # de-smoothed result back, so the parent forward below sees the
                    # weights the layer would actually compute with.
                    for bl in mapping.balance_layers:
                        quant_func, opt_quant_func = bl_quant_funcs[bl]
                        w_qdq = owner._qdq_tool.qdq(
                            orig_state[bl] * scales_view,
                            bl_params[bl],
                            quant_func=quant_func,
                            opt_quant_func=opt_quant_func,
                            imatrix=getattr(bl, "imatrix", None),
                        )
                        bl.weight.data = (w_qdq / scales_view).to(bl.weight.dtype)

                    total_loss = owner._compute_parent_loss(mapping.parent, parent_kwargs_list, fp16_outputs)
                else:
                    total_loss = 0.0
                    for bl in mapping.balance_layers:
                        quant_func, opt_quant_func = bl_quant_funcs[bl]
                        w_orig = orig_weights[bl].to(device)
                        w_qdq = owner._qdq_tool.qdq(
                            w_orig * scales_view,
                            bl_params[bl],
                            quant_func=quant_func,
                            opt_quant_func=opt_quant_func,
                            imatrix=getattr(bl, "imatrix", None),
                        )
                        total_loss += (w_orig - w_qdq / scales_view).pow(2).sum().item()

                return total_loss

        self._candidates = candidates
        self._score = score

    def candidates(self):
        return self._candidates()

    def score(self, candidate):
        return self._score(candidate)

    def apply(self, candidate) -> None:
        self.owner._apply_scales(self.mapping, candidate[1])

    def clear(self) -> None:
        # Drop closure-owned snapshots and outputs; shared parent calls remain
        # available to subsequent mappings and smoothing passes.
        self._candidates = None
        self._score = None


def get_grid_search_params(n_grid: int, duo_scaling: bool | str) -> list[tuple[float, bool]]:
    """Return (ratio, use_duo_scaling) tuples for the grid search."""
    match duo_scaling:
        case "both":
            n = max(int(n_grid / 2), 2)
            return [(idx / (n - 1), duo) for idx in range(n) for duo in [False, True]]
        case False:
            n = max(n_grid, 2)
            return [(idx / (n - 1), False) for idx in range(n)]
        case True:
            n = max(n_grid, 3)
            return [(0.0, False)] + [(idx / (n - 2), True) for idx in range(n - 1)]
        case _:
            raise ValueError(f"Unexpected duo_scaling value: {duo_scaling!r}")


def compute_layer_means(layers: list[torch.nn.Module], group_size: int) -> torch.Tensor:
    """Per-channel mean of normalised weights across all balance layers."""
    weight = torch.cat([m.weight.detach().float() for m in layers], dim=0)
    org_shape = weight.shape
    gs = group_size if group_size is not None and group_size > 0 else org_shape[1]
    weight, _, pad_len = reshape_pad_tensor_by_group_size(weight, gs)
    w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
    w_scale = revert_tensor_by_pad(w_scale, orig_shape=org_shape, pad_len=pad_len)
    return w_scale.mean(0)


@torch.no_grad()
def fold_scales_into_smooth_layer(smooth: torch.nn.Module, scales: torch.Tensor) -> None:
    """Divide a smooth layer's output by ``scales`` to offset balance scaling.

    Dispatches on the smooth layer's weight layout:

    * 1-D norm weight with a Gemma-style ``(1 + weight)`` gain: folded as
      ``weight <- (1 + weight) / s - 1`` to preserve output invariance.
    * 1-D standard norm weight: folded as ``weight <- weight / s``.
    * 2-D linear weight: its trailing ``s.numel()`` output rows are divided.

    Any bias is always divided by ``s``.
    """
    s = scales.to(smooth.weight.device)
    weight = smooth.weight.data
    if weight.ndim == 1:
        if _rmsnorm_has_unit_offset(smooth):
            weight.copy_((1.0 + weight) / s - 1.0)
        else:
            weight.div_(s)
    else:
        weight[-s.size(0) :].div_(s.view(-1, 1))

    if getattr(smooth, "bias", None) is not None:
        smooth.bias.data.div_(s)
