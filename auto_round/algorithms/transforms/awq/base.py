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
"""AWQ (Activation-Aware Weight Quantization) quantizer.

Algorithm:
1. Collect per-channel activation magnitudes during calibration.
2. For each smooth-balance mapping, perform a grid search over scaling ratios
   to find the one that minimises quantization error (output-based loss).
3. Apply the best channel-wise scaling:
   - balance_layer.weight *= scales
   - smooth_layer.weight /= scales (or smooth_layer.bias /= scales if 1-D)
4. Weight compression is delegated to the pipeline's block_quantizer.

Reference implementations:
  - AutoAWQ: https://github.com/casper-hansen/AutoAWQ
  - llm-compressor: https://github.com/vllm-project/llm-compressor
"""

from __future__ import annotations

import inspect
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from auto_round.algorithms.pipeline import (
    ActCalibPolicy,
    CalibTiming,
    InputSource,
)
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.awq.mappings import (
    ResolvedMapping,
    _extract_block_prefix,
    check_model_compatibility,
    resolve_mappings,
)
from auto_round.algorithms.transforms.awq.qdq import QDQTool
from auto_round.algorithms.transforms.base import BaseWeightTransformer
from auto_round.data_type.utils import (
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
)
from auto_round.logger import logger

if TYPE_CHECKING:
    from auto_round.algorithms.pipeline import BlockContext


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


@register_pipeline_member(AWQConfig)
class AWQTransform(BaseWeightTransformer):
    """AWQ transform: activation-aware weight smoothing pre-processor.

    Inherits :class:`~auto_round.algorithms.transforms.base.BaseWeightTransformer`.
    It smooths block weights in-place; actual weight compression (RTN /
    SignRound) is performed by the pipeline's ``block_quantizer``.
    """

    def __init__(self, config: AWQConfig) -> None:
        super().__init__(config)
        self.duo_scaling: bool | str = config.duo_scaling
        self.n_grid: int = config.n_grid
        self.smooth_iters: int = getattr(config, "smooth_iters", 1)

        # AWQ weight-clip options (search + hard-clamp after smoothing).
        self.apply_clip: bool = getattr(config, "apply_clip", False)
        self.clip_as_init: bool = getattr(config, "clip_as_init", False)
        self.clip_n_grid: int = getattr(config, "clip_n_grid", 20)
        self.clip_max_shrink: float = getattr(config, "clip_max_shrink", 0.5)
        self.clip_n_sample_token: int = getattr(config, "clip_n_sample_token", 512)

        # Single source of truth for "QDQ a candidate weight under the target
        # block-quantizer scheme", used as AWQ's grid-search / clip loss. AWQ
        # composes this instead of re-implementing block-quantizer dispatch;
        self._qdq_tool = QDQTool(
            bits=config.bits,
            group_size=config.group_size,
            sym=config.sym,
            data_type=config.data_type,
        )

        self._user_mappings: list[dict] | None = config.mappings

        # Set at runtime by the compressor's post_init() via ``pre.layer_config = self.layer_config``.
        self.layer_config: dict | None = None

        self._resolved_mappings: list[ResolvedMapping] = []
        self._block_mappings: dict[str, list[ResolvedMapping]] = {}

        self._activation_stats: dict[str, list] = {}
        self._parent_args_cache: dict[torch.nn.Module, list[dict]] = {}
        self._parent_signatures: dict[int, inspect.Signature] = {}
        # Per-mapping balance-layer input features captured for the clip search
        # (keyed by smooth_name). Only populated when ``apply_clip`` is set.
        self._clip_input_feat: dict[str, torch.Tensor] = {}

        self._finalized: bool = False

    # ── Algorithm Fusion: lifecycle hook implementations ──────────────────────

    def bind(self, compressor) -> None:
        """Wire shared state and force AWQ onto single-block scheduling."""
        super().bind(compressor)
        nblocks = getattr(compressor, "nblocks", 1)
        if nblocks > 1:
            logger.warning(
                "AWQ does not support nblocks > 1 (got nblocks=%s). " "Falling back to nblocks=1.",
                nblocks,
            )
            compressor.nblocks = 1

    def prepare_run(self, compressor) -> None:
        """Validate compatibility, resolve model-wide mappings, and group by block prefix."""
        model = compressor.model_context.model
        report = check_model_compatibility(model, self._user_mappings)
        for warning in report["warnings"]:
            logger.warning(warning)

        # ── Resolve all model-level mappings (name-only, no module caching) ──
        self._resolved_mappings = resolve_mappings(model, self._user_mappings)
        if not self._resolved_mappings:
            raise ValueError(
                "AWQ: no layer mappings were resolved for this model. "
                f"Model class: {type(model).__name__}. "
                "To add support, provide explicit 'mappings' in AWQConfig, or "
                "add an entry to auto_round/algorithms/transforms/awq/mappings.py."
            )

        # Group mappings by block prefix for O(1) lookup during block iteration.
        self._block_mappings = {}
        for m in self._resolved_mappings:
            prefix = _extract_block_prefix(m.smooth_name)
            self._block_mappings.setdefault(prefix, []).append(m)

        self._qdq_tool.configure(compressor)

        if compressor.compress_context is not None:
            compressor.compress_context.cache_device = torch.device("cpu")

        logger.info(
            "AWQ: resolved %d mappings across %d blocks.",
            len(self._resolved_mappings),
            len(self._block_mappings),
        )
        self._finalized = False

    def get_act_calib_policy(self, ctx: "BlockContext"):
        """AWQ W4A16 (weight-only): no activation calibration needed."""
        # AWQ pre-processing does not collect act-calib stats; that is the
        # block_quantizer's concern.  For W8A8/static activation, a post-smooth
        # forward may be needed — handled via the block_quantizer's policy.
        return ActCalibPolicy(when=CalibTiming.SKIP, source=InputSource.FP_CACHE)

    @contextmanager
    def block_forward_hooks(self, ctx: "BlockContext"):
        """Register AWQ activation-stats and parent-kwargs hooks.

        Hooks are registered on the *current block's* smooth sources and
        parent modules.  All handles are removed when this context manager
        exits (before ``__exit__`` returns), regardless of exceptions.
        """
        handles = []
        block_mappings = self._block_mappings.get(ctx.block_name, [])
        if block_mappings:
            handles = self._register_awq_hooks(ctx.model, ctx.block, ctx.block_name)
        try:
            yield handles
        finally:
            for h in handles:
                h.remove()
            handles.clear()

    def pre_quantize_block(self, ctx: "BlockContext") -> None:
        """Apply AWQ smoothing for this block and mark modified params.

        Called after the reference forward (activation stats collected) and
        before the block quantizer runs.
        """
        if len(ctx.block_names) != 1:
            raise ValueError(f"AWQ requires nblocks=1, got {len(ctx.block_names)} blocks: {ctx.block_names}.")
        block_name = ctx.block_names[0]
        block_mappings = self._block_mappings.get(block_name, [])
        if not block_mappings:
            logger.debug("AWQ: no mappings for block '%s', skipping.", block_name)
            return
        # The compressor sets ``layer_config`` after ``prepare_run``; keep the
        # QDQ service in sync before it is used for the grid-search / clip loss.
        self._qdq_tool.layer_config = self.layer_config
        self._smooth_block(block_name, block_mappings)
        if self.apply_clip:
            self._clip_block(block_name, block_mappings)
        modified = []
        for mapping in block_mappings:
            modified.extend(mapping.balance_names)
            modified.append(mapping.smooth_name)
        ctx.mark_modified_fp_params(modified)

    def post_quantize_block(self, ctx: "BlockContext") -> None:
        """Release per-block AWQ caches to free memory."""
        block_mappings = self._block_mappings.get(ctx.block_name, [])
        if not block_mappings:
            return
        for m in block_mappings:
            self._activation_stats.pop(m.smooth_name, None)
            self._clip_input_feat.pop(m.smooth_name, None)
            # Drop the transient clip attribute once the block quantizer has
            # consumed it (the persistent copy lives on the model context).
            for bl in m.balance_layers:
                if hasattr(bl, "awq_clip_max"):
                    delattr(bl, "awq_clip_max")
        seen_parents: set[int] = set()
        for m in block_mappings:
            pid = id(m.parent)
            if pid not in seen_parents:
                seen_parents.add(pid)
                self._parent_args_cache.pop(m.parent, None)

    def finalize_run(self, compressor) -> None:
        """Idempotent global teardown.  Safe to call inside try/finally."""
        if self._finalized:
            return
        self._activation_stats.clear()
        self._parent_args_cache.clear()
        self._parent_signatures.clear()
        self._clip_input_feat.clear()
        self._finalized = True
        logger.debug("AWQ: finalize_quantization complete.")

    # ── Hook registration ─────────────────────────────────────────────────────

    def _register_awq_hooks(
        self,
        model: torch.nn.Module,
        block: torch.nn.Module,
        block_name: str,
    ) -> list:
        """Register activation-stats and parent-kwargs hooks for one block."""
        handles = []
        mappings = self._block_mappings.get(block_name, [])
        smooth_names = {m.smooth_name for m in mappings}

        # ── Smooth-layer activation-stats hooks ───────────────────────────────
        # Priority: smooth source forward_hook (output stats).
        # Each smooth source is hooked exactly once (set de-duplication via name).
        for name, module in block.named_modules():
            full_name = f"{block_name}.{name}" if name else block_name
            if full_name not in smooth_names:
                continue

            def _make_stats_hook(layer_name: str):

                def hook_fn(mod, args, output):
                    x = output[0] if isinstance(output, tuple) else output
                    if x is None or x.numel() == 0:
                        return
                    channel_sum = x.detach().float().flatten(0, -2).abs().sum(dim=0).cpu()
                    count = x[..., 0].numel()
                    if layer_name not in self._activation_stats:
                        self._activation_stats[layer_name] = [
                            torch.zeros_like(channel_sum),
                            0,
                        ]
                    self._activation_stats[layer_name][0] += channel_sum
                    self._activation_stats[layer_name][1] += count

                return hook_fn

            h = module.register_forward_hook(_make_stats_hook(full_name))
            handles.append(h)

        # ── Clip input-feature hooks (only when apply_clip is enabled) ────────
        # The balance layers of a mapping share the same input; capture it once
        # per mapping (keyed by smooth_name) for the post-smooth clip search.
        if self.apply_clip:
            for mapping in mappings:
                if not mapping.balance_layers:
                    continue
                target_layer = mapping.balance_layers[0]

                def _make_clip_hook(smooth_name: str):

                    def hook_fn(mod, args):
                        x = args[0] if isinstance(args, tuple) else args
                        if x is None or not isinstance(x, torch.Tensor) or x.numel() == 0:
                            return
                        feat = x.detach().reshape(-1, x.shape[-1])
                        # Subsample tokens to bound memory.
                        if feat.shape[0] > self.clip_n_sample_token:
                            step = max(1, feat.shape[0] // self.clip_n_sample_token)
                            feat = feat[::step]
                        feat = feat.float().cpu()
                        prev = self._clip_input_feat.get(smooth_name)
                        if prev is None:
                            self._clip_input_feat[smooth_name] = feat
                        else:
                            self._clip_input_feat[smooth_name] = torch.cat([prev, feat], dim=0)

                    return hook_fn

                h = target_layer.register_forward_pre_hook(_make_clip_hook(mapping.smooth_name))
                handles.append(h)

        # One forward_pre_hook per unique parent module in the current block.
        parent_modules_hooked: set[int] = set()
        for mapping in mappings:
            parent = mapping.parent
            hook_target = mapping.activation_hook_target
            if hook_target:
                target_parent = dict(model.named_modules()).get(hook_target)
                if target_parent is None:
                    logger.warning(
                        "AWQ: activation_hook_target '%s' for '%s' was not found; using resolved parent '%s'.",
                        hook_target,
                        mapping.smooth_name,
                        mapping.parent_name,
                    )
                else:
                    parent = target_parent
            if id(parent) in parent_modules_hooked:
                continue
            parent_modules_hooked.add(id(parent))

            if parent not in self._parent_args_cache:
                self._parent_args_cache[parent] = []

            def _make_parent_hook(parent_module: torch.nn.Module):

                def hook_fn(mod, args, kwargs):
                    cls_id = id(type(mod))
                    if cls_id not in self._parent_signatures:
                        self._parent_signatures[cls_id] = inspect.signature(mod.forward)
                    sig = self._parent_signatures[cls_id]
                    try:
                        bound = sig.bind(*args, **kwargs)
                        bound.apply_defaults()
                    except TypeError:
                        return  # signature mismatch; skip this sample

                    param = next(mod.parameters(), None)
                    w_dtype = param.dtype if param is not None else None

                    stored: dict[str, Any] = {}
                    for k, v in bound.arguments.items():
                        if isinstance(v, torch.Tensor):
                            v = v.detach()
                            if w_dtype and v.is_floating_point() and v.dtype != w_dtype:
                                v = v.to(w_dtype)
                            stored[k] = v
                        elif isinstance(v, tuple) and any(isinstance(t, torch.Tensor) for t in v):
                            stored[k] = tuple(
                                (
                                    t.detach().to(w_dtype)
                                    if (w_dtype and isinstance(t, torch.Tensor) and t.is_floating_point())
                                    else (t.detach() if isinstance(t, torch.Tensor) else t)
                                )
                                for t in v
                            )
                        elif hasattr(v, "key_cache"):
                            stored[k] = None  # Null out KV cache objects
                        else:
                            stored[k] = v

                    self._parent_args_cache[parent_module].append(stored)

                return hook_fn

            h = parent.register_forward_pre_hook(_make_parent_hook(parent), with_kwargs=True)
            handles.append(h)

        return handles

    # ── Smoothing (grid search + scale apply) ─────────────────────────────────

    def _smooth_block(self, block_prefix: str, block_mappings: list) -> None:
        """Run grid search and apply AWQ scales for one block.

        When ``smooth_iters > 1`` the grid search + scale apply is repeated.
        Repeating refines the smoothing scale because the mx max_scale search
        and the AWQ alpha (ratio) search influence each other: each extra pass
        re-derives the max_scale from the freshly-smoothed weights and
        re-searches the ratio, accumulating the resulting scales.
        """
        n_passes = max(1, int(self.smooth_iters))
        for smooth_pass in range(n_passes):
            for mapping in block_mappings:
                if mapping.smooth_name not in self._activation_stats:
                    logger.warning(
                        "AWQ: no activation stats for '%s' in block '%s'; skipping.",
                        mapping.smooth_name,
                        block_prefix,
                    )
                    continue

                act_sum, act_count = self._activation_stats[mapping.smooth_name]
                if act_count == 0:
                    logger.warning(
                        "AWQ: zero activation count for '%s' in block '%s'; skipping.",
                        mapping.smooth_name,
                        block_prefix,
                    )
                    continue

                x_mean = (act_sum / act_count).to(torch.float32)
                del act_sum

                best_scales = self._grid_search_scales(mapping, x_mean)
                if best_scales is not None:
                    self._apply_scales(mapping, best_scales)

            if n_passes > 1:
                logger.debug("AWQ: completed smooth pass %d/%d for block '%s'", smooth_pass + 1, n_passes, block_prefix)

        # Release parent kwargs after ALL passes/mappings for this block are processed.
        seen_parents: set[int] = set()
        for mapping in block_mappings:
            pid = id(mapping.parent)
            if pid not in seen_parents:
                seen_parents.add(pid)
                self._parent_args_cache.pop(mapping.parent, None)

    def _get_grid_search_params(self) -> list[tuple[float, bool]]:
        """Return (ratio, use_duo_scaling) tuples for the grid search."""
        match self.duo_scaling:
            case "both":
                n = max(int(self.n_grid / 2), 2)
                return [(idx / (n - 1), duo) for idx in range(n) for duo in [False, True]]
            case False:
                n = max(self.n_grid, 2)
                return [(idx / (n - 1), False) for idx in range(n)]
            case True:
                n = max(self.n_grid, 3)
                return [(0.0, False)] + [(idx / (n - 2), True) for idx in range(n - 1)]
            case _:
                raise ValueError(f"Unexpected duo_scaling value: {self.duo_scaling!r}")

    @staticmethod
    def _normalize_group_size(group_size: int | None, fallback: int) -> int:
        """Return ``group_size`` if it denotes a real per-group size, else ``fallback``.

        A ``None``, ``0`` or negative ``group_size`` means "no grouping" (per-row),
        which each caller represents with its own sentinel (``-1`` for the quant
        funcs, the row width for weight reshaping).
        """
        return group_size if (group_size is not None and group_size > 0) else fallback

    @staticmethod
    def _compute_layer_means(layers: list[torch.nn.Module], group_size: int) -> torch.Tensor:
        """Per-channel mean of normalised weights across all balance layers."""
        weight = torch.cat([m.weight.detach().float() for m in layers], dim=0)
        org_shape = weight.shape
        gs = AWQTransform._normalize_group_size(group_size, org_shape[1])
        weight, _, pad_len = reshape_pad_tensor_by_group_size(weight, gs)
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        w_scale = revert_tensor_by_pad(w_scale, orig_shape=org_shape, pad_len=pad_len)
        return w_scale.mean(0)

    @torch.no_grad()
    def _grid_search_scales(
        self,
        mapping: ResolvedMapping,
        x_mean: torch.Tensor,
    ) -> torch.Tensor | None:
        """Find the best scaling ratio for *mapping* via output-based loss."""
        device = mapping.balance_layers[0].weight.device
        x_mean = x_mean.to(device)

        group_size = self._normalize_group_size(self._qdq_tool.group_size, -1)
        if self.duo_scaling is not False:
            w_mean = self._compute_layer_means(mapping.balance_layers, group_size).to(device)

        parent_kwargs_list = self._parent_args_cache.get(mapping.parent, [])
        use_parent_forward = len(parent_kwargs_list) > 0

        if use_parent_forward:
            fp16_outputs = self._run_parent_samples(mapping.parent, parent_kwargs_list)
            if not fp16_outputs or all(f.numel() == 0 for f in fp16_outputs):
                use_parent_forward = False

        orig_state = {bl: bl.weight.data.clone() for bl in mapping.balance_layers}
        if not use_parent_forward:
            orig_weights = orig_state  # same reference is fine

        # Resolve each balance layer's scheme once, then pre-resolve the quant
        # functions for the grid-search loop. ``opt_quant_func`` is non-None only
        # when the SignRoundV2 optimized init-scale path applies for this mapping.
        bl_params = {bl: self._qdq_tool.resolve_params(bl) for bl in mapping.balance_layers}
        cached_quant_func, opt_quant_func = self._qdq_tool.resolve_quant_funcs(bl_params[mapping.balance_layers[0]])

        best_error = float("inf")
        best_scales = None
        best_ratio = -1

        for ratio, use_duo in self._get_grid_search_params():
            if use_duo:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1
            scales_view = scales.view(1, -1).to(device)

            if use_parent_forward:
                # Quantize each balance layer's smoothed weight and write the
                # de-smoothed result back, so the parent forward below sees the
                # weights the layer would actually compute with.
                for bl in mapping.balance_layers:
                    w_qdq = self._qdq_tool.qdq(
                        orig_state[bl] * scales_view,
                        bl_params[bl],
                        quant_func=cached_quant_func,
                        opt_quant_func=opt_quant_func,
                        imatrix=getattr(bl, "imatrix", None),
                    )
                    bl.weight.data = (w_qdq / scales_view).to(bl.weight.dtype)

                int_w_outputs = self._run_parent_samples(mapping.parent, parent_kwargs_list)
                total_loss = self._compute_loss(fp16_outputs, int_w_outputs)
                del int_w_outputs
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl])
            else:
                total_loss = 0.0
                for bl in mapping.balance_layers:
                    w_orig = orig_weights[bl].to(device)
                    w_qdq = self._qdq_tool.qdq(
                        w_orig * scales_view,
                        bl_params[bl],
                        quant_func=cached_quant_func,
                        opt_quant_func=opt_quant_func,
                        imatrix=getattr(bl, "imatrix", None),
                    )
                    total_loss += (w_orig - w_qdq / scales_view).pow(2).sum().item()

            if total_loss < best_error:
                best_error = total_loss
                best_scales = scales.clone()
                best_ratio = ratio

        if best_ratio < 0:
            logger.warning("AWQ: grid search failed for '%s': no finite error.", mapping.smooth_name)
            return None

        logger.debug("AWQ '%s': best_ratio=%.2f, best_error=%.3e", mapping.smooth_name, best_ratio, best_error)
        return best_scales

    @torch.no_grad()
    def _run_parent_samples(
        self,
        parent: torch.nn.Module,
        kwargs_list: list[dict],
    ) -> list[torch.Tensor]:
        outputs = []
        for stored_kwargs in kwargs_list:
            out = parent(**stored_kwargs)
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out)
        return outputs

    @staticmethod
    @torch.no_grad()
    def _compute_loss(
        fp16_outputs: list[torch.Tensor],
        int_w_outputs: list[torch.Tensor],
    ) -> float:
        device = fp16_outputs[0].device
        loss = torch.tensor(0.0, device=device)
        num_elements = torch.tensor(0, device=device, dtype=torch.long)
        for fp16_out, int_w_out in zip(fp16_outputs, int_w_outputs):
            loss += torch.nn.functional.mse_loss(
                fp16_out.float(),
                int_w_out.to(fp16_out.device).float(),
                reduction="sum",
            )
            num_elements += fp16_out.numel()
        if num_elements == 0:
            return float("inf")
        return (loss / num_elements).item()

    @torch.no_grad()
    def _apply_scales(self, mapping: ResolvedMapping, scales: torch.Tensor) -> None:
        """Apply computed AWQ scales to smooth and balance layers in-place.

        Each balance layer's input channels are multiplied by ``scales`` while
        the upstream smooth layer's output is divided by the same factor, so the
        block's overall function is preserved and quantization difficulty is
        shifted off the balance weights.
        """
        for bl in mapping.balance_layers:
            bl.weight.data.mul_(scales.to(bl.weight.device).view(1, -1))

        self._fold_scales_into_smooth_layer(mapping.smooth_layer, scales)

        # Keep the captured clip input features consistent with the smoothing:
        # the balance-layer input is divided by the smooth scales, so the stored
        # features (used later by the clip search) must be divided too.
        if self.apply_clip:
            feat = self._clip_input_feat.get(mapping.smooth_name)
            if feat is not None:
                feat.div_(scales.detach().to(feat.device).view(1, -1))

    @staticmethod
    @torch.no_grad()
    def _fold_scales_into_smooth_layer(smooth: torch.nn.Module, scales: torch.Tensor) -> None:
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

    # ── Weight clipping (search best per-group clip + hard-clamp) ─────────────

    # Layers whose clipping is skipped: clipping q/k projections hurts the
    # attention score (q·kᵀ) precision, mirroring AutoAWQ's ``avoid_clipping``.
    _AVOID_CLIP_TOKENS = ("q_", "k_", "query", "key", "Wqkv", "wqkv")

    def _should_skip_clip(self, balance_name: str) -> bool:
        local = balance_name.rsplit(".", 1)[-1]
        return any(token in local for token in self._AVOID_CLIP_TOKENS)

    @torch.no_grad()
    def _clip_block(self, block_prefix: str, block_mappings: list) -> None:
        """Search per-group weight clip thresholds for one block.

        Runs after smoothing. The searched per-group clip magnitude is always
        recorded on the model context (and, in ``clip_as_init`` mode, on the
        balance layer) so it is kept for downstream use. Two modes:

        * ``clip_as_init=False`` (default): the clip is hard-clamped in place on
          the (already smoothed) balance-layer weights, so any downstream block
          quantizer (RTN / SignRound / SignRoundV2) re-derives its min/max
          range from the clipped weights.
        * ``clip_as_init=True``: the weights are left untouched and the clip is
          stored on the layer (``awq_clip_max``); the downstream SignRound /
          SignRoundV2 quantizer uses it to *initialize* its tunable weight range
          (capping ``weight_min``/``weight_max`` or clamping before the scale
          search) and then tunes ``min_scale``/``max_scale`` on top.
        """
        clip_store = getattr(self.model_context, "awq_clip_values", None)
        for mapping in block_mappings:
            feat = self._clip_input_feat.get(mapping.smooth_name)
            if feat is None:
                logger.warning(
                    "AWQ: no clip input features for '%s' in block '%s'; skipping clip.",
                    mapping.smooth_name,
                    block_prefix,
                )
                continue
            for bl, name in zip(mapping.balance_layers, mapping.balance_names):
                if self._should_skip_clip(name):
                    logger.debug("AWQ: skip clip for '%s' (avoid-clipping layer).", name)
                    continue
                max_val = self._compute_best_clip(bl, feat)
                if max_val is None:
                    continue
                key = getattr(bl, "global_name", None) or name
                if clip_store is not None:
                    clip_store[key] = max_val.detach().to("cpu")
                if self.clip_as_init:
                    # Keep the weights intact; hand the clip to the block
                    # quantizer as the initialization of its weight range.
                    bl.awq_clip_max = max_val.detach()
                else:
                    self._apply_clip(bl, max_val)

    @torch.no_grad()
    def _compute_best_clip(
        self,
        layer: torch.nn.Module,
        input_feat: torch.Tensor,
    ) -> torch.Tensor | None:
        """Search the per-group clip threshold that minimizes output MSE.

        Returns a ``[out_channels, n_group]`` tensor of clip magnitudes, or
        ``None`` if clipping is not applicable to this layer.
        """
        params = self._qdq_tool.resolve_params(layer)
        bits = params["bits"]
        if bits is None or bits >= 16:
            return None
        group_size = params["group_size"]

        device = layer.weight.device
        weight = layer.weight.detach().float()
        out_features, in_features = weight.shape
        gs = self._normalize_group_size(group_size, in_features)
        if in_features % gs != 0:
            logger.warning(
                "AWQ: in_features=%d not divisible by group_size=%d for clip; skipping '%s'.",
                in_features,
                gs,
                getattr(layer, "global_name", "") or "<layer>",
            )
            return None
        n_group = in_features // gs

        # Clip search is a flat per-group weight QDQ: substitute the normalized
        # group size and drop super-block (double-quant) params, which the clip
        # path does not apply.
        clip_params = {**params, "group_size": gs, "super_bits": None, "super_group_size": None}
        quant_func, _ = self._qdq_tool.resolve_quant_funcs(clip_params)
        if quant_func is None:
            return None

        feat = input_feat.to(device).reshape(-1, in_features)
        if feat.shape[0] > self.clip_n_sample_token:
            step = max(1, feat.shape[0] // self.clip_n_sample_token)
            feat = feat[::step]
        # [1, n_token, n_group, gs]
        feat = feat.reshape(1, feat.shape[0], n_group, gs)

        # [out_features, 1, n_group, gs]
        w = weight.reshape(out_features, 1, n_group, gs)

        # Batch output channels to bound peak memory.
        oc_batch = 256 if out_features % 256 == 0 else (64 if out_features % 64 == 0 else out_features)
        best_max_val_all = []
        n_steps = max(1, int(self.clip_max_shrink * self.clip_n_grid))

        for i_b in range(0, out_features, oc_batch):
            w_b = w[i_b : i_b + oc_batch]
            org_max_val = w_b.abs().amax(dim=-1, keepdim=True)  # [oc_b, 1, n_group, 1]
            best_max_val = org_max_val.clone()
            min_errs = torch.full_like(org_max_val, 1e9)
            org_out = (feat * w_b).sum(dim=-1)  # [oc_b, n_token, n_group]

            for i_s in range(n_steps):
                max_val = org_max_val * (1 - i_s / self.clip_n_grid)
                cur_w = torch.clamp(w_b, -max_val, max_val)
                cur_w_flat = cur_w.reshape(cur_w.shape[0], n_group * gs)
                q_w = self._qdq_tool.qdq(cur_w_flat, clip_params, quant_func=quant_func).reshape(cur_w.shape)
                cur_out = (feat * q_w).sum(dim=-1)
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                improved = err < min_errs
                min_errs[improved] = err[improved]
                best_max_val[improved] = max_val[improved]
                del cur_w, q_w, cur_out

            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        return best_max_val.squeeze(1)  # [out_features, n_group, 1]

    @torch.no_grad()
    def _apply_clip(self, layer: torch.nn.Module, max_val: torch.Tensor) -> None:
        """Hard-clamp the layer weight to ``[-max_val, max_val]`` per group."""
        org_dtype = layer.weight.dtype
        max_val = max_val.to(device=layer.weight.device, dtype=org_dtype)
        org_shape = layer.weight.shape
        w = layer.weight.data.reshape(*max_val.shape[:2], -1)
        w = torch.clamp(w, -max_val, max_val)
        layer.weight.data = w.reshape(org_shape).to(org_dtype)
