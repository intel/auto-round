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

import gc
import inspect
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
from auto_round.algorithms.transforms.base import BaseWeightTransformer
from auto_round.data_type.mxfp import search_mx_scale
from auto_round.data_type.utils import (
    get_quant_func,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
)
from auto_round.logger import logger

if TYPE_CHECKING:
    from auto_round.algorithms.pipeline import BlockContext, RunContext


@register_pipeline_member(AWQConfig)
class AWQQuantizer(BaseWeightTransformer):
    """AWQ quantizer: activation-aware weight smoothing pre-processor.

    Inherits :class:`~auto_round.algorithms.transforms.base.BaseWeightTransformer`.
    It smooths block weights in-place; actual weight compression (RTN /
    SignRound) is performed by the pipeline's ``block_quantizer``.
    """

    def __init__(self, config: AWQConfig):
        super().__init__(config)
        self.duo_scaling: bool | str = config.duo_scaling
        self.n_grid: int = config.n_grid

        self._user_mappings: list[dict] | None = config.mappings

        # Set at runtime by the compressor's post_init() via ``pre.layer_config = self.layer_config``.
        self.layer_config: dict | None = None

        self._resolved_mappings: list[ResolvedMapping] = []
        self._block_mappings: dict[str, list[ResolvedMapping]] = {}

        self._activation_stats: dict[str, list] = {}
        self._parent_args_cache: dict[torch.nn.Module, list[dict]] = {}
        self._parent_signatures: dict[int, inspect.Signature] = {}
        self._use_v2_mx_scale_search: bool = False

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

    def prepare_run(self, run_ctx: "RunContext") -> None:
        """Validate compatibility, resolve model-wide mappings, and group by block prefix."""
        report = check_model_compatibility(run_ctx.model, self._user_mappings)
        for warning in report["warnings"]:
            logger.warning(warning)

        # ── Resolve all model-level mappings (name-only, no module caching) ──
        self._resolved_mappings = resolve_mappings(run_ctx.model, self._user_mappings)
        if not self._resolved_mappings:
            raise ValueError(
                "AWQ: no layer mappings were resolved for this model. "
                f"Model class: {type(run_ctx.model).__name__}. "
                "To add support, provide explicit 'mappings' in AWQConfig, or "
                "add an entry to auto_round/algorithms/transforms/awq/mappings.py."
            )

        # Group mappings by block prefix for O(1) lookup during block iteration.
        self._block_mappings = {}
        for m in self._resolved_mappings:
            prefix = _extract_block_prefix(m.smooth_name)
            self._block_mappings.setdefault(prefix, []).append(m)

        self._use_v2_mx_scale_search = any(
            getattr(config, "_alg_cls", None) == "SignRoundV2Quantizer" for config in run_ctx.alg_configs
        ) and str(self.data_type).startswith("mx_fp")
        logger.info(f"AWQ: use_v2_mx_scale_search={self._use_v2_mx_scale_search}")
        # self._use_v2_mx_scale_search = False

        if run_ctx.compress_context is not None:
            run_ctx.compress_context.cache_device = torch.device("cpu")

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
        self._smooth_block(block_name, block_mappings)
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
        seen_parents: set[int] = set()
        for m in block_mappings:
            pid = id(m.parent)
            if pid not in seen_parents:
                seen_parents.add(pid)
                self._parent_args_cache.pop(m.parent, None)

    def finalize_run(self, run_ctx: "RunContext") -> None:
        """Idempotent global teardown.  Safe to call inside try/finally."""
        if self._finalized:
            return
        self._activation_stats.clear()
        self._parent_args_cache.clear()
        self._parent_signatures.clear()
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

        # ── Parent-kwargs hooks ───────────────────────────────────────────────
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
        """Run grid search and apply AWQ scales for one block."""
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

        # Release parent kwargs after ALL mappings for this block are processed.
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
    def _compute_layer_means(layers: list[torch.nn.Module], group_size: int) -> torch.Tensor:
        """Per-channel mean of normalised weights across all balance layers."""
        weight = torch.cat([m.weight.detach().float() for m in layers], dim=0)
        org_shape = weight.shape
        gs = group_size if group_size > 0 else org_shape[1]
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

        group_size = self.group_size if (self.group_size is not None and self.group_size > 0) else -1
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

        # Pre-resolve quant function once to avoid repeated dispatch in loop.
        ref_layer = mapping.balance_layers[0]
        ref_name = getattr(ref_layer, "global_name", None) or ""
        ref_cfg = (self.layer_config or {}).get(ref_name, {})
        try:
            cached_quant_func, _ = get_quant_func(
                ref_cfg.get("data_type", self.data_type),
                ref_cfg.get("bits", self.bits),
                ref_cfg.get("sym", self.sym),
                disable_opt_rtn=ref_cfg.get("disable_opt_rtn", True),
                group_size=ref_cfg.get("group_size", self.group_size),
                iters=0,
            )
        except Exception as exc:
            logger.debug("AWQ: failed to pre-resolve quant function for '%s': %s", ref_name, exc)
            cached_quant_func = None

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
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl] * scales_view)
                    w_qdq = self._quantize_dequantize_weight(bl, bl.weight.data.float(), quant_func=cached_quant_func)
                    if w_qdq is not None:
                        bl.weight.data = (w_qdq / scales_view).to(bl.weight.dtype)
                    else:
                        bl.weight.data.copy_(orig_state[bl])

                int_w_outputs = self._run_parent_samples(mapping.parent, parent_kwargs_list)
                total_loss = self._compute_loss(fp16_outputs, int_w_outputs)
                del int_w_outputs
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl])
            else:
                total_loss = 0.0
                for bl in mapping.balance_layers:
                    w_orig = orig_weights[bl].to(device)
                    w_qdq = self._quantize_dequantize_weight(bl, w_orig * scales_view, quant_func=cached_quant_func)
                    if w_qdq is None:
                        total_loss = float("inf")
                        break
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

    def _quantize_dequantize_weight(
        self,
        layer: torch.nn.Module,
        weight: torch.Tensor,
        quant_func=None,
    ) -> torch.Tensor | None:
        """Quantize-dequantize a weight tensor using the layer's config.

        Used internally for grid search loss calculation only.  Does NOT
        modify the layer's stored weights.
        """
        layer_name = getattr(layer, "global_name", None) or ""
        config = (self.layer_config or {}).get(layer_name, {})
        bits = config.get("bits", self.bits)
        group_size = config.get("group_size", self.group_size)
        sym = config.get("sym", self.sym)
        data_type = config.get("data_type", self.data_type)
        disable_opt_rtn = config.get("disable_opt_rtn", True)

        if quant_func is None:
            try:
                quant_func, _ = get_quant_func(
                    data_type,
                    bits,
                    sym,
                    disable_opt_rtn=disable_opt_rtn,
                    group_size=group_size,
                    iters=0,
                )
            except Exception as exc:
                logger.debug("AWQ: failed to resolve quant function for '%s': %s", layer_name, exc)
                return None

        if quant_func is None:
            return None

        try:
            quant_kwargs = {
                "bits": bits,
                "group_size": group_size,
                "data_type": data_type,
                "sym": sym,
            }
            if self._use_v2_mx_scale_search and str(data_type).startswith("mx_fp"):
                weight_reshape, _, _ = reshape_pad_tensor_by_group_size(weight, group_size)
                imatrix = self._reshape_imatrix_for_weight(layer, weight_reshape, group_size)
                quant_kwargs["init_scale"] = search_mx_scale(weight_reshape, bits, imatrix)
            qdq_weight, _, _ = quant_func(weight, **quant_kwargs)
            return qdq_weight
        except Exception as exc:
            logger.debug("AWQ: quantize-dequantize failed for '%s': %s", layer_name, exc)
            return None

    @staticmethod
    def _reshape_imatrix_for_weight(
        layer: torch.nn.Module,
        weight_reshape: torch.Tensor,
        group_size,
    ) -> torch.Tensor | float:
        """Match SignRoundV2's imatrix layout before MXFP scale search."""
        imatrix = getattr(layer, "imatrix", None)
        if imatrix is None:
            return 1.0
        imatrix = imatrix.reshape(1, -1)
        imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        imatrix = imatrix.expand(weight_reshape.numel() // imatrix.numel(), -1)
        return imatrix.reshape(weight_reshape.shape).to(weight_reshape.device)

    @torch.no_grad()
    def _apply_scales(self, mapping: ResolvedMapping, scales: torch.Tensor) -> None:
        """Apply computed AWQ scales to smooth and balance layers in-place."""
        for bl in mapping.balance_layers:
            s = scales.to(bl.weight.device).view(1, -1)
            bl.weight.data.mul_(s)

        smooth = mapping.smooth_layer
        s = scales.to(smooth.weight.device)
        if smooth.weight.ndim == 1:
            smooth.weight.data.div_(s)
        else:
            smooth.weight.data[-s.size(0) :].div_(s.view(-1, 1))

        if hasattr(smooth, "bias") and smooth.bias is not None:
            smooth.bias.data.div_(s)
