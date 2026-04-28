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

The algorithm:
1. Collects per-channel activation magnitudes during calibration.
2. For each smooth-balance mapping, performs a grid search over scaling ratios
   to find the one that minimizes quantization error (output-based loss).
3. Applies the best channel-wise scaling: balance_layer.weight *= scales,
   smooth_layer.weight /= scales (or smooth_layer.bias /= scales if 1-D).
4. Delegates to RTN quantization for the actual weight quantization.

Reference implementations:
  - AutoAWQ: https://github.com/casper-hansen/AutoAWQ
  - llm-compressor: https://github.com/vllm-project/llm-compressor
"""

from __future__ import annotations

import contextlib
import gc
import inspect

import torch
from tqdm import tqdm

from auto_round.algorithms.quantization.awq.config import AWQConfig
from auto_round.algorithms.quantization.awq.mappings import (
    ResolvedMapping,
    _extract_block_prefix,
    resolve_mappings,
)
from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer
from auto_round.data_type.utils import (
    get_quant_func,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
)
from auto_round.logger import logger


class AWQQuantizer(RTNQuantizer):
    """AWQ quantizer that applies activation-aware channel scaling before RTN.

    The AWQ algorithm pre-processes weights via channel-wise scaling derived from
    activation statistics, then delegates final quantization to RTN.
    """

    def __init__(self, config: AWQConfig):
        super().__init__(config)
        self.duo_scaling = config.duo_scaling
        self.n_grid = config.n_grid

        # Populated during calibration
        self._user_mappings = config.mappings
        self._resolved_mappings: list[ResolvedMapping] = []
        self._activation_stats: dict[str, list[torch.Tensor]] = {}
        # Parent module kwargs cache: parent_module → list of kwargs dicts
        self._parent_args_cache: dict[torch.nn.Module, list[dict]] = {}
        self._parent_signatures: dict[int, inspect.BoundArguments] = {}
        self._smoothing_applied: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve_all_mappings(self, model: torch.nn.Module) -> dict[str, list[ResolvedMapping]]:
        """Resolve all AWQ mappings and group by block prefix.

        Call this once before the block-by-block loop.  The resolved mappings
        are stored internally and used by ``register_activation_hooks`` and
        ``smooth_block`` for per-block filtering.

        Returns:
            Dictionary mapping block prefix (e.g. "model.layers.0") to the
            list of resolved mappings for that block.
        """
        self._resolved_mappings = resolve_mappings(model, self._user_mappings)
        block_groups: dict[str, list[ResolvedMapping]] = {}
        for m in self._resolved_mappings:
            prefix = _extract_block_prefix(m.smooth_name)
            block_groups.setdefault(prefix, []).append(m)
        return block_groups

    def apply_smoothing(
        self,
        model: torch.nn.Module,
        device: torch.device | str | None = None,
    ) -> None:
        """Apply AWQ smoothing to the model weights.

        This is the core AWQ step: for each mapping, find the best scaling
        factor and apply it to smooth/balance layers.

        When *device* is provided the model is assumed to reside on CPU and
        only the modules needed for each mapping are temporarily moved to
        *device* during the grid search.  This dramatically reduces peak VRAM
        (~14 GB savings for an 8B model).

        Should be called AFTER calibration data has been collected and BEFORE
        RTN quantization.
        """
        if self._smoothing_applied:
            return

        # Resolve mappings
        self._resolved_mappings = resolve_mappings(model, self._user_mappings)
        if not self._resolved_mappings:
            logger.warning("No AWQ mappings resolved; skipping smoothing.")
            self._smoothing_applied = True
            return

        if not self._activation_stats:
            logger.error("No activation statistics collected for AWQ smoothing.")

        logger.info(f"Applying AWQ smoothing to {len(self._resolved_mappings)} mappings...")

        for mapping in tqdm(self._resolved_mappings, desc="AWQ Smoothing"):
            if mapping.smooth_name not in self._activation_stats:
                logger.warning(f"No activation stats for '{mapping.smooth_name}', skipping this layer.")
                continue

            act_sum, act_count = self._activation_stats.pop(mapping.smooth_name)
            if act_count == 0:
                logger.warning(f"Zero activation count for '{mapping.smooth_name}', skipping this layer.")
                continue

            # Mean activation per channel (act_count is a plain int now,
            # avoiding a CPU tensor allocation per channel).
            x_mean = (act_sum / act_count).to(torch.float32)
            del act_sum

            with self._align_modules(mapping, device):
                best_scales = self._grid_search_scales(mapping, x_mean)

            if best_scales is not None:
                self._apply_scales(mapping, best_scales)

            # Eagerly free per-mapping parent cache to reduce RAM
            parent = mapping.parent
            self._parent_args_cache.pop(parent, None)

        self._activation_stats.clear()
        self._parent_args_cache.clear()
        self._parent_signatures.clear()
        self._smoothing_applied = True
        logger.info("AWQ smoothing complete.")

    def smooth_block(self, block_prefix: str) -> None:
        """Apply AWQ smoothing for one block's mappings.

        Assumes activation stats and parent kwargs have already been collected
        for this block via ``register_activation_hooks`` + block forward.
        The block's modules are expected to be on the compute device, so
        no device alignment is needed (unlike ``apply_smoothing`` which uses
        ``_align_modules`` for CPU-offloaded models).

        Args:
            block_prefix: Block prefix (e.g. "model.layers.0") identifying
                which mappings to smooth.
        """
        block_mappings = [m for m in self._resolved_mappings if _extract_block_prefix(m.smooth_name) == block_prefix]
        for mapping in block_mappings:
            if mapping.smooth_name not in self._activation_stats:
                logger.warning(f"No activation stats for '{mapping.smooth_name}', skipping.")
                continue

            act_sum, act_count = self._activation_stats.pop(mapping.smooth_name)
            if act_count == 0:
                logger.warning(f"Zero activation count for '{mapping.smooth_name}', skipping.")
                continue

            x_mean = (act_sum / act_count).to(torch.float32)
            del act_sum

            # Block is on the compute device — grid search runs in-place
            best_scales = self._grid_search_scales(mapping, x_mean)

            if best_scales is not None:
                self._apply_scales(mapping, best_scales)

        # Free parent kwargs cache after ALL mappings for this block are done.
        # (All mappings in a block typically share the same parent module;
        # popping inside the loop would break output-based loss for
        # subsequent mappings.)
        seen_parents = set()
        for mapping in block_mappings:
            pid = id(mapping.parent)
            if pid not in seen_parents:
                seen_parents.add(pid)
                self._parent_args_cache.pop(mapping.parent, None)

    # ── Module alignment (device onload/offload) ─────────────────────────────

    @contextlib.contextmanager
    def _align_modules(
        self,
        mapping,
        device: torch.device | str | None,
    ):
        """Temporarily move mapping-related modules to *device* for grid search.

        If *device* is None (model already on the compute device), this is a
        no-op passthrough.
        """
        if device is None:
            yield
            return

        modules_to_move = [mapping.parent, mapping.smooth_layer] + list(mapping.balance_layers)
        # Deduplicate (parent may overlap with smooth/balance)
        seen = set()
        unique = []
        for m in modules_to_move:
            if id(m) not in seen:
                seen.add(id(m))
                unique.append(m)

        # Onload
        for m in unique:
            m.to(device)

        try:
            yield
        finally:
            # Offload back to CPU
            for m in unique:
                m.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    # ── Activation statistics collection ──────────────────────────────────────

    def register_activation_hooks(
        self,
        model: torch.nn.Module,
        block_prefix: str | None = None,
    ) -> list:
        """Register activation hooks for AWQ and return hook handles.

        Hooks are registered for:
        1. Smooth layers: accumulate abs activation sums/counts for x_mean.
        2. Parent modules: cache forward kwargs for output-based loss

        Args:
            model: The model (hooks are registered on the actual module
                objects, which are shared with blocks obtained via
                ``get_module``).
            block_prefix: When provided, only register hooks for mappings
                whose smooth layer belongs to this block (e.g.
                "model.layers.0").  Used in the block-by-block pipeline.

        Should be called by the compressor before calibration, and handles
        should be removed after calibration.
        """
        if not self._resolved_mappings:
            self._resolved_mappings = resolve_mappings(model, self._user_mappings)

        mappings = self._resolved_mappings
        if block_prefix is not None:
            mappings = [m for m in mappings if _extract_block_prefix(m.smooth_name) == block_prefix]

        hooks = []
        smooth_names = {m.smooth_name for m in mappings}

        # Hook smooth layers for activation statistics
        for name, module in model.named_modules():
            if name not in smooth_names:
                continue

            def _make_hook(layer_name):
                def hook_fn(mod, args, output):
                    if isinstance(output, tuple):
                        x = output[0]
                    else:
                        x = output
                    if x is None or x.numel() == 0:
                        return

                    # Compute abs-sum directly on GPU without materialising a
                    # full float32 intermediate. flatten→abs→sum produces a
                    # single [channels] tensor; .cpu() immediately frees the
                    # GPU allocation (aligned with AR's "immediate CPU move"
                    # pattern from CalibCompressor hooks).
                    channel_sum = x.detach().float().flatten(0, -2).abs().sum(dim=0).cpu()
                    count = x.shape[:-1].numel()

                    if layer_name not in self._activation_stats:
                        self._activation_stats[layer_name] = [
                            torch.zeros_like(channel_sum),
                            0,
                        ]
                    self._activation_stats[layer_name][0] += channel_sum
                    self._activation_stats[layer_name][1] += count

                return hook_fn

            h = module.register_forward_hook(_make_hook(name))
            hooks.append(h)

        # Hook parent modules to cache kwargs for output-based loss
        parent_modules_hooked = set()
        for mapping in mappings:
            parent = mapping.parent
            if id(parent) in parent_modules_hooked:
                continue
            parent_modules_hooked.add(id(parent))

            self._parent_args_cache[parent] = []

            def _make_parent_hook(parent_module):
                def hook_fn(mod, args, kwargs):
                    # Cache ALL calibration batches for output-based grid
                    # search loss, use all calibration data without subsampling.
                    mod_cls_id = id(type(mod))
                    if mod_cls_id not in self._parent_signatures:
                        self._parent_signatures[mod_cls_id] = inspect.signature(mod.forward)
                    sig = self._parent_signatures[mod_cls_id]
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    stored = {}
                    for k, v in bound.arguments.items():
                        if isinstance(v, torch.Tensor):
                            stored[k] = v.detach()
                        elif isinstance(v, tuple) and any(isinstance(t, torch.Tensor) for t in v):
                            # Detach tensors in tuples (e.g. position_embeddings
                            # = (cos, sin)) to release computation graph refs.
                            stored[k] = tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in v)
                        elif hasattr(v, "key_cache"):
                            # Null out KV cache objects (DynamicCache etc.)
                            stored[k] = None
                        else:
                            stored[k] = v

                    self._parent_args_cache[parent_module].append(stored)

                return hook_fn

            h = parent.register_forward_pre_hook(_make_parent_hook(parent), with_kwargs=True)
            hooks.append(h)

        return hooks

    # ── Grid search ───────────────────────────────────────────────────────────

    def _get_grid_search_params(self) -> list[tuple[float, bool]]:
        """Get grid search parameters (ratio, duo_scaling).

        Returns:
            List of (ratio, use_duo_scaling) tuples for the grid search.
        """
        match self.duo_scaling:
            # "both": half grid with duo off, half with duo on
            case "both":
                n_grid = max(int(self.n_grid / 2), 2)
                return [
                    (grid_idx / (n_grid - 1), duo_scaling)
                    for grid_idx in range(n_grid)
                    for duo_scaling in [False, True]
                ]
            case False:
                return [(grid_idx / (self.n_grid - 1), False) for grid_idx in range(self.n_grid)]
            # True: include identity (0.0, False) as first, then duo points
            case True:
                return [(0.0, False)] + [(grid_idx / (self.n_grid - 2), True) for grid_idx in range(self.n_grid - 1)]
            case _:
                raise ValueError(f"Found unexpected duo_scaling configuration {self.duo_scaling}")

    @staticmethod
    def _compute_layer_means(layers: list[torch.nn.Module], group_size: int) -> torch.Tensor:
        """Compute per-channel mean of normalised weights.

        Within each quantization group, weights are normalized by their group max
        (so values are on a 0-1 scale), then averaged across all groups to get
        per-channel importance.

        Args:
            layers: Balance layers whose weights are concatenated.
            group_size: Quantization group size. If -1, uses full channel width.

        Returns:
            Per-channel mean of normalised weights [in_features].
        """
        # Concatenate all balance layer weights [total_out, in_features]
        weight = torch.cat([m.weight.detach().float() for m in layers], dim=0)
        org_shape = weight.shape

        # Determine effective group size
        gs = group_size if group_size > 0 else org_shape[1]

        # Pad when needed so AWQ works with layers whose input width is not a
        # multiple of group_size, matching grouped RTN quantization behavior.
        weight, _, pad_len = reshape_pad_tensor_by_group_size(weight, gs)
        # Normalize within each group: abs / max (0-1 scale per group)
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Remove padding, then take mean across output channels.
        w_scale = revert_tensor_by_pad(w_scale, orig_shape=org_shape, pad_len=pad_len)
        w_mean = w_scale.mean(0)
        return w_mean

    @torch.no_grad()
    def _grid_search_scales(
        self,
        mapping: ResolvedMapping,
        x_mean: torch.Tensor,
    ) -> torch.Tensor | None:
        """Find the best scaling ratio via grid search.

        Uses output-based error:
        ``L(s) = || fp16_output - Q(W*s) @ (X/s) ||^2``

        For each candidate scaling ratio, applies scales to balance layer
        weights, quantize-dequantizes them, runs all cached calibration
        batches through the parent module, and computes the output MSE
        against the fp16 baseline.

        Returns:
            Best scales tensor, or None if no valid scale was found.
        """
        device = mapping.balance_layers[0].weight.device
        x_mean = x_mean.to(device)

        # Compute normalised weight means
        group_size = self.group_size if self.group_size > 0 else -1
        if self.duo_scaling is not False:
            w_mean = self._compute_layer_means(mapping.balance_layers, group_size).to(device)

        # Try to run parent module forward for output-based loss
        parent_kwargs_list = self._parent_args_cache.get(mapping.parent, [])
        use_parent_forward = len(parent_kwargs_list) > 0

        if use_parent_forward:
            # Compute fp16 baseline outputs for loss computation
            fp16_outputs = self._run_parent_samples(mapping.parent, parent_kwargs_list)
            if not fp16_outputs or all(f.numel() == 0 for f in fp16_outputs):
                use_parent_forward = False

        if not use_parent_forward:
            orig_weights = {bl: bl.weight.data.clone() for bl in mapping.balance_layers}

        # Save original weights for restoration during grid search
        orig_state = {bl: bl.weight.data.clone() for bl in mapping.balance_layers}

        best_error = float("inf")
        best_scales = None
        best_ratio = -1

        # Pre-resolve quant function once.
        ref_layer = mapping.balance_layers[0]
        ref_name = getattr(ref_layer, "global_name", None) or ""
        ref_cfg = self.layer_config.get(ref_name, {})
        try:
            cached_quant_func, _ = get_quant_func(
                ref_cfg.get("data_type", self.data_type),
                ref_cfg.get("bits", self.bits),
                ref_cfg.get("sym", self.sym),
                disable_opt_rtn=True,
                group_size=ref_cfg.get("group_size", self.group_size),
                iters=0,
            )
        except Exception:
            cached_quant_func = None

        grid_params = self._get_grid_search_params()

        for ratio, use_duo in grid_params:
            # Compute scales
            if use_duo:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1
            scales_view = scales.view(1, -1).to(device)

            if use_parent_forward:
                # Q(W * s) / s: mutate balance layer weights
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl] * scales_view)
                    w_qdq = self._quantize_dequantize_weight(
                        bl,
                        bl.weight.data.float(),
                        quant_func=cached_quant_func,
                    )
                    if w_qdq is not None:
                        bl.weight.data = (w_qdq / scales_view).to(bl.weight.dtype)
                    else:
                        bl.weight.data.copy_(orig_state[bl])

                # Collect quantized outputs then compute loss
                int_w_outputs = self._run_parent_samples(mapping.parent, parent_kwargs_list)
                total_loss = self._compute_loss(fp16_outputs, int_w_outputs)
                del int_w_outputs

                # Restore original weights
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl])
            else:
                # Weight-only fallback: || W - Q(W*s)/s ||^2
                total_loss = 0.0
                for bl in mapping.balance_layers:
                    w_orig = orig_weights[bl].to(device)
                    w_scaled = w_orig * scales_view
                    w_qdq = self._quantize_dequantize_weight(
                        bl,
                        w_scaled,
                        quant_func=cached_quant_func,
                    )
                    if w_qdq is None:
                        total_loss = float("inf")
                        break
                    w_qdq_unscaled = w_qdq / scales_view
                    total_loss += (w_orig - w_qdq_unscaled).pow(2).sum().item()

            if total_loss < best_error:
                best_error = total_loss
                best_scales = scales.clone()
                best_ratio = ratio

        if best_ratio < 0:
            logger.warning(f"AWQ grid search failed for '{mapping.smooth_name}': " "no finite error found.")
            return None

        logger.debug(f"AWQ '{mapping.smooth_name}': best_ratio={best_ratio:.2f}, " f"best_error={best_error:.3e}")
        return best_scales

    # ── Parent module forward ─────────────────────────────

    @torch.no_grad()
    def _run_parent_samples(
        self,
        parent: torch.nn.Module,
        kwargs_list: list[dict],
    ) -> list[torch.Tensor]:
        """Run cached samples through the parent module.

        Feeds cached kwargs through the parent forward without any CUDA
        synchronisation between batches, so the GPU can pipeline all forwards.
        Outputs are kept on-device for loss computation.
        """
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
        """Compute normalised MSE between fp16 and quantized outputs."""
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

        Args:
            quant_func: Pre-resolved quantization function. When provided,
                skips the ``get_quant_func`` dispatch (avoids redundant
                lookups in the inner grid search loop).

        Returns the quantized-then-dequantized weight, or None on failure.
        """
        layer_name = getattr(layer, "global_name", None) or ""
        config = self.layer_config.get(layer_name, {})
        bits = config.get("bits", self.bits)
        group_size = config.get("group_size", self.group_size)
        sym = config.get("sym", self.sym)
        data_type = config.get("data_type", self.data_type)

        if quant_func is None:
            try:
                quant_func, _ = get_quant_func(
                    data_type,
                    bits,
                    sym,
                    disable_opt_rtn=True,  # AWQ always uses plain RTN
                    group_size=group_size,
                    iters=0,  # Route to rtn_int_sym, not int_sym
                )
            except Exception:
                return None

            if quant_func is None:
                return None

        try:
            qdq_weight, scale, zp = quant_func(
                weight,
                bits=bits,
                group_size=group_size,
            )
            return qdq_weight
        except Exception:
            return None

    # ── Apply scales ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _apply_scales(self, mapping: ResolvedMapping, scales: torch.Tensor) -> None:
        """Apply the computed AWQ scales to smooth and balance layers.

        - Balance layers (Linear): weight *= scales (along input channels)
        - Smooth layer (LayerNorm/RMSNorm): weight /= scales, bias /= scales
        """
        for bl in mapping.balance_layers:
            device = bl.weight.device
            s = scales.to(device).view(1, -1)
            bl.weight.data.mul_(s)

        smooth = mapping.smooth_layer
        device = smooth.weight.device
        s = scales.to(device)

        if smooth.weight.ndim == 1:
            # LayerNorm / RMSNorm: 1-D weight (per-channel)
            smooth.weight.data.div_(s)
        else:
            # Edge case: when smooth layer's out_features != balance layer's
            # in_features (e.g. fused qkv_proj smoothing o_proj). Scale the
            # last output features (aligned with AutoAWQ).
            # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/scale.py#L123
            smooth.weight.data[-s.size(0) :].div_(s.view(-1, 1))

        if hasattr(smooth, "bias") and smooth.bias is not None:
            smooth.bias.data.div_(s)
