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
1. Collect per-channel balance-layer input magnitudes during calibration.
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
from typing import TYPE_CHECKING, Any

import torch

from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.awq.mappings import (
    AWQ_DYNAMIC_MAPPING_REGISTRY,
    AWQ_MAPPING_REGISTRY,
    ResolvedMapping,
    _extract_block_prefix,
    resolve_mappings,
)
from auto_round.algorithms.transforms.awq.qdq import QDQTool
from auto_round.algorithms.transforms.base import BasePreprocessor
from auto_round.data_type.utils import (
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
)
from auto_round.logger import logger
from auto_round.utils.model import move_to_device

if TYPE_CHECKING:
    from auto_round.algorithms.composer import AlgorithmComposer, BlockContext


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


def _slice_seq_tensor(v: Any, actual_seq: int, seqlen: int) -> Any:
    """Slice a value's sequence dimension to *seqlen*, recursing into containers."""
    if isinstance(v, torch.Tensor):
        if v.ndim == 1 and v.shape[0] == actual_seq:
            return v[:seqlen]
        if v.ndim == 3 and v.shape[1] == actual_seq:
            return v[:, :seqlen]
        if v.ndim == 2 and v.shape[1] == actual_seq:
            return v[:, :seqlen]
        if v.ndim == 4 and v.shape[2] == actual_seq and v.shape[3] == actual_seq:
            return v[:, :, :seqlen, :seqlen]
        if v.ndim == 4 and v.shape[2] == actual_seq:
            return v[:, :, :seqlen]
        return v
    if isinstance(v, (tuple, list)):
        return type(v)(_slice_seq_tensor(t, actual_seq, seqlen) for t in v)
    if isinstance(v, dict):
        return {k: _slice_seq_tensor(t, actual_seq, seqlen) for k, t in v.items()}
    return v


def _detect_actual_seq(values) -> int | None:
    """Infer sequence length from the first 3-D tensor, including nested containers."""
    for v in values:
        if isinstance(v, torch.Tensor) and v.ndim == 3:
            return v.shape[1]
        if isinstance(v, (tuple, list)):
            nested = _detect_actual_seq(v)
            if nested is not None:
                return nested
        if isinstance(v, dict):
            nested = _detect_actual_seq(v.values())
            if nested is not None:
                return nested
    return None


def _truncate_args_kwargs(args: tuple, kwargs: dict, seqlen: int) -> tuple[tuple, dict]:
    """Truncate parent-forward positional args and kwargs to at most *seqlen* tokens."""
    actual_seq = _detect_actual_seq(list(args) + list(kwargs.values()))
    if actual_seq is None or actual_seq <= seqlen:
        return args, kwargs
    new_args = tuple(_slice_seq_tensor(v, actual_seq, seqlen) for v in args)
    new_kwargs = {k: _slice_seq_tensor(v, actual_seq, seqlen) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _truncate_awq_tensor(x: torch.Tensor, seqlen: int) -> torch.Tensor:
    """Truncate an AWQ activation tensor to at most *seqlen* tokens."""
    actual_seq = _detect_actual_seq((x,))
    if actual_seq is None or actual_seq <= seqlen:
        return x
    return _slice_seq_tensor(x, actual_seq, seqlen)


def _detect_actual_batch(values) -> int | None:
    """Infer batch size from the first tensor with an explicit batch axis."""
    for v in values:
        if isinstance(v, torch.Tensor) and v.ndim >= 2:
            return v.shape[0]
        if isinstance(v, (tuple, list)):
            nested = _detect_actual_batch(v)
            if nested is not None:
                return nested
        if isinstance(v, dict):
            nested = _detect_actual_batch(v.values())
            if nested is not None:
                return nested
    return None


def _detect_parent_batch(args: tuple, kwargs: dict) -> int | None:
    """Infer parent replay batch size, preferring positional model inputs."""
    actual_batch = _detect_actual_batch(args)
    if actual_batch is not None:
        return actual_batch
    return _detect_actual_batch(kwargs.values())


def _slice_batch_value(v: Any, actual_batch: int, start: int, end: int) -> Any:
    """Slice a value's leading batch dimension, recursing into containers."""
    if isinstance(v, torch.Tensor):
        if v.ndim >= 2 and v.shape[0] == actual_batch:
            return v[start:end]
        return v
    if isinstance(v, (tuple, list)):
        return type(v)(_slice_batch_value(t, actual_batch, start, end) for t in v)
    if isinstance(v, dict):
        return {k: _slice_batch_value(t, actual_batch, start, end) for k, t in v.items()}
    return v


def _slice_batch_args_kwargs(args: tuple, kwargs: dict, actual_batch: int, start: int, end: int) -> tuple[tuple, dict]:
    """Slice parent-forward positional args and kwargs to a batch microchunk."""
    new_args = tuple(_slice_batch_value(v, actual_batch, start, end) for v in args)
    new_kwargs = {k: _slice_batch_value(v, actual_batch, start, end) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _iter_block_names_for_mapping(model: torch.nn.Module) -> list[str]:
    """Return block names sorted from most-specific to least-specific for mapping lookup."""
    from auto_round.utils.common import flatten_list
    from auto_round.utils.model import get_block_names

    return sorted((name for name in flatten_list(get_block_names(model)) if name), key=len, reverse=True)


@register_pipeline_member(AWQConfig)
class AWQTransform(BasePreprocessor):
    """AWQ transform: activation-aware weight smoothing pre-processor.

    Inherits :class:`~auto_round.algorithms.transforms.base.BasePreprocessor`.
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

        # Cap sequence length consistently across AWQ calibration.
        # A value <= 0 disables truncation and uses the full calibration sequence.
        awq_seqlen = getattr(config, "awq_seqlen", 512)
        self._awq_seqlen: int | None = awq_seqlen if awq_seqlen > 0 else None
        smooth_batch_size = getattr(config, "smooth_batch_size", None)
        self._smooth_batch_size: int | None = smooth_batch_size if smooth_batch_size and smooth_batch_size > 0 else None

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
        self._skip_moe: bool = getattr(config, "skip_moe", False)

        # Set at runtime by the compressor's post_init() via ``pre.layer_config = self.layer_config``.
        self.layer_config: dict | None = None

        self._resolved_mappings: list[ResolvedMapping] = []
        self._block_mappings: dict[str, list[ResolvedMapping]] = {}

        self._activation_stats: dict[str, list] = {}
        self._parent_args_cache: dict[torch.nn.Module, list[tuple[tuple, dict]]] = {}
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
            logger.error(
                "AWQ does not support nblocks > 1 (got nblocks=%s). ",
                nblocks,
            )
            exit(-1)

    def can_compile_block_forward(self) -> bool:
        """AWQ installs per-block calibration hooks that trigger Dynamo recompiles."""
        return False

    def prepare_run(self, composer: "AlgorithmComposer" = None) -> None:
        """Resolve model-wide mappings and group them by transformer block."""
        model = self.model

        # ── Resolve all model-level mappings (name-only, no module caching) ──
        self._resolved_mappings = resolve_mappings(model, self._user_mappings, skip_moe=self._skip_moe)
        if not self._resolved_mappings:
            raise ValueError(
                "AWQ: no layer mappings were resolved for this model. "
                f"Model class: {type(model).__name__}. "
                "To add support, provide explicit 'mappings' in AWQConfig, or "
                "add an entry to auto_round/algorithms/transforms/awq/mappings.py."
            )

        cls_name = type(model).__name__
        if (
            self._user_mappings is None
            and cls_name not in AWQ_MAPPING_REGISTRY
            and cls_name not in AWQ_DYNAMIC_MAPPING_REGISTRY
        ):
            logger.warning(
                "AWQ: model class '%s' is not in any AWQ mapping registry; using "
                "default Llama-like mappings. If quantization quality is poor, "
                "provide explicit mappings via AWQConfig(mappings=[...]).",
                cls_name,
            )

        iter_block_names = _iter_block_names_for_mapping(model)

        self._block_mappings = {}
        for m in self._resolved_mappings:
            key = None
            for block_name in iter_block_names:
                if m.smooth_name == block_name or m.smooth_name.startswith(block_name + "."):
                    key = block_name
                    break
            if key is None:
                key = _extract_block_prefix(m.smooth_name)
            self._block_mappings.setdefault(key, []).append(m)

        if composer is not None:
            self._qdq_tool.configure(composer, awq_config=self.config)

        logger.info(
            "AWQ: resolved %d mappings across %d blocks.",
            len(self._resolved_mappings),
            len(self._block_mappings),
        )
        self._finalized = False

    def register_fp_input_forward_hooks(self, block) -> list:
        """Register AWQ activation-stats and parent-kwargs hooks.

        Hooks are registered on the *current block's* smooth sources and
        parent modules. Returns hook handles that the caller must remove.
        """
        # Need block_name from the block's global_name attribute
        block_name = getattr(block, "global_name", "")
        block_mappings = self._block_mappings.get(block_name, [])
        if block_mappings:
            return self._register_awq_hooks(self.model_context.model, block, block_name)
        return []

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
        active_mappings = [m for m in block_mappings if self._mapping_is_smoothable(m)]
        skipped = len(block_mappings) - len(active_mappings)
        if skipped:
            logger.warning_once(
                "AWQ: skipped %d smoothing mapping(s) in block '%s' that include "
                "ignore_layers / full-precision layers or incompatible per-layer quantization parameters.",
                skipped,
                block_name,
            )
        if not active_mappings:
            return
        self._smooth_block(block_name, active_mappings)
        if self.apply_clip:
            self._clip_block(block_name, active_mappings)
        modified = []
        for mapping in active_mappings:
            modified.extend(mapping.balance_names)
            modified.append(mapping.smooth_name)

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
                if hasattr(bl, "awq_clip_min"):
                    delattr(bl, "awq_clip_min")
                if hasattr(bl, "awq_clip_max"):
                    delattr(bl, "awq_clip_max")
        seen_parents: set[int] = set()
        for m in block_mappings:
            pid = id(m.parent)
            if pid not in seen_parents:
                seen_parents.add(pid)
                self._parent_args_cache.pop(m.parent, None)

    def finalize_run(self) -> None:
        """Idempotent global teardown.  Safe to call inside try/finally."""
        if self._finalized:
            return
        self._activation_stats.clear()
        self._parent_args_cache.clear()
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
        module_lookup = dict(model.named_modules())

        def _resolve_activation_hook_layer(mapping: ResolvedMapping) -> torch.nn.Module | None:
            if not mapping.balance_layers:
                return None

            hook_target = mapping.activation_hook_target
            if not hook_target:
                return mapping.balance_layers[0]

            target_layer = module_lookup.get(hook_target)
            if target_layer is None and mapping.parent_name:
                target_layer = module_lookup.get(f"{mapping.parent_name}.{hook_target}")
            if target_layer is None:
                try:
                    target_layer = mapping.parent.get_submodule(hook_target)
                except AttributeError:
                    target_layer = None
            if target_layer is None:
                logger.warning(
                    "AWQ: activation_hook_target '%s' for '%s' was not found; using first balance layer '%s'.",
                    hook_target,
                    mapping.smooth_name,
                    mapping.balance_names[0] if mapping.balance_names else "<unknown>",
                )
                return mapping.balance_layers[0]
            return target_layer

        # ── Balance-layer input activation hooks ─────────────────────────────
        # AWQ scales are derived from the tensor entering the balance layer.  For
        # gated MLPs, smooth-layer output (for example up_proj(x)) is not the same
        # tensor consumed by down_proj (act(gate_proj(x)) * up_proj(x)).
        for mapping in mappings:
            target_layer = _resolve_activation_hook_layer(mapping)
            if target_layer is None:
                continue

            def _make_activation_hook(smooth_name: str):

                def hook_fn(mod, args):
                    x = args[0] if isinstance(args, tuple) else args
                    if x is None or not isinstance(x, torch.Tensor) or x.numel() == 0:
                        return

                    feat = x.detach()
                    if self._awq_seqlen is not None:
                        feat = _truncate_awq_tensor(feat, self._awq_seqlen)
                    if feat.ndim == 1:
                        feat = feat.view(1, -1)
                    else:
                        feat = feat.flatten(0, -2)

                    channel_sum = feat.float().abs().sum(dim=0).cpu()
                    count = feat.shape[0]
                    if smooth_name not in self._activation_stats:
                        self._activation_stats[smooth_name] = [
                            torch.zeros_like(channel_sum),
                            0,
                        ]
                    self._activation_stats[smooth_name][0] += channel_sum
                    self._activation_stats[smooth_name][1] += count

                    if self.apply_clip:
                        clip_feat = feat
                        # Subsample tokens to bound memory.
                        if clip_feat.shape[0] > self.clip_n_sample_token:
                            step = max(1, clip_feat.shape[0] // self.clip_n_sample_token)
                            clip_feat = clip_feat[::step]
                        clip_feat = clip_feat.float().cpu()
                        prev = self._clip_input_feat.get(smooth_name)
                        if prev is None:
                            self._clip_input_feat[smooth_name] = clip_feat
                        else:
                            self._clip_input_feat[smooth_name] = torch.cat([prev, clip_feat], dim=0)

                return hook_fn

            h = target_layer.register_forward_pre_hook(_make_activation_hook(mapping.smooth_name))
            handles.append(h)

        # One forward_pre_hook per unique parent module in the current block.
        parent_modules_hooked: set[int] = set()
        for mapping in mappings:
            parent = mapping.parent
            if id(parent) in parent_modules_hooked:
                continue
            parent_modules_hooked.add(id(parent))

            if parent not in self._parent_args_cache:
                self._parent_args_cache[parent] = []

            def _make_parent_hook(parent_module: torch.nn.Module):

                def hook_fn(mod, args, kwargs):
                    param = next(mod.parameters(), None)
                    w_dtype = param.dtype if param is not None else None

                    def _proc(v):
                        if hasattr(v, "key_cache"):
                            return None
                        if isinstance(v, torch.Tensor):
                            v = v.detach()
                            if w_dtype and v.is_floating_point() and v.dtype != w_dtype:
                                v = v.to(w_dtype)
                            return v.to("cpu", non_blocking=False)
                        if isinstance(v, tuple):
                            return tuple(_proc(t) for t in v)
                        if isinstance(v, list):
                            return [_proc(t) for t in v]
                        if isinstance(v, dict):
                            return {k: _proc(t) for k, t in v.items()}
                        return v

                    proc_args = tuple(_proc(a) for a in args)
                    proc_kwargs = {k: _proc(v) for k, v in kwargs.items()}

                    if self._awq_seqlen is not None:
                        proc_args, proc_kwargs = _truncate_args_kwargs(proc_args, proc_kwargs, self._awq_seqlen)
                    self._parent_args_cache[parent_module].append((proc_args, proc_kwargs))

                return hook_fn

            h = parent.register_forward_pre_hook(_make_parent_hook(parent), with_kwargs=True)
            handles.append(h)

        return handles

    # ── Smoothing (grid search + scale apply) ─────────────────────────────────

    def _mapping_has_ignored_layer(self, mapping: ResolvedMapping) -> bool:
        """Return True if the smooth layer or any balance layer is kept full precision."""
        layer_config = self._qdq_tool.layer_config or {}
        if not layer_config:
            return False

        def _is_fp(layer: torch.nn.Module) -> bool:
            name = getattr(layer, "global_name", None)
            if not name or name not in layer_config:
                return False
            bits = layer_config[name].get("bits", None)
            return bits is not None and bits >= 16

        if _is_fp(mapping.smooth_layer):
            return True
        return any(_is_fp(bl) for bl in mapping.balance_layers)

    @staticmethod
    def _freeze_quant_param(value):
        if isinstance(value, list):
            return tuple(AWQTransform._freeze_quant_param(item) for item in value)
        if isinstance(value, tuple):
            return tuple(AWQTransform._freeze_quant_param(item) for item in value)
        return value

    def _balance_quant_signature(self, layer: torch.nn.Module) -> tuple:
        """Return the resolved quantization signature that must match within one AWQ mapping."""
        params = self._qdq_tool.resolve_params(layer)
        keys = ("bits", "group_size", "sym", "data_type", "super_bits", "super_group_size")
        return tuple((key, self._freeze_quant_param(params.get(key))) for key in keys)

    def _mapping_has_mixed_quant_params(self, mapping: ResolvedMapping) -> bool:
        """Return True when balance layers in one AWQ smoothing group do not share quant params."""
        if len(mapping.balance_layers) <= 1:
            return False

        signatures = [self._balance_quant_signature(layer) for layer in mapping.balance_layers]
        first = signatures[0]
        if all(signature == first for signature in signatures[1:]):
            return False

        details = {name: dict(signature) for name, signature in zip(mapping.balance_names, signatures)}
        logger.warning(
            "AWQ: skipping smoothing for '%s' because balance layers in the same mapping "
            "have different quantization parameters: %s.",
            mapping.smooth_name,
            details,
        )
        return True

    def _mapping_is_smoothable(self, mapping: ResolvedMapping) -> bool:
        """AWQ smoothing is all-or-nothing for layers sharing one smooth scale."""
        if self._mapping_has_ignored_layer(mapping):
            return False
        if self._mapping_has_mixed_quant_params(mapping):
            return False
        return True

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

        bl_params = {bl: self._qdq_tool.resolve_params(bl) for bl in mapping.balance_layers}
        group_size = self._normalize_group_size(bl_params[mapping.balance_layers[0]]["group_size"], -1)
        if self.duo_scaling is not False:
            w_mean = self._compute_layer_means(mapping.balance_layers, group_size).to(device)

        parent_kwargs_list = self._parent_args_cache.get(mapping.parent, [])
        use_parent_forward = len(parent_kwargs_list) > 0

        if use_parent_forward:
            fp16_outputs = self._run_parent_samples(
                mapping.parent,
                parent_kwargs_list,
                offload_to_cpu=self._smooth_batch_size is not None,
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
        bl_quant_funcs = {bl: self._qdq_tool.resolve_quant_funcs(bl_params[bl]) for bl in mapping.balance_layers}

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
                    quant_func, opt_quant_func = bl_quant_funcs[bl]
                    w_qdq = self._qdq_tool.qdq(
                        orig_state[bl] * scales_view,
                        bl_params[bl],
                        quant_func=quant_func,
                        opt_quant_func=opt_quant_func,
                        imatrix=getattr(bl, "imatrix", None),
                    )
                    bl.weight.data = (w_qdq / scales_view).to(bl.weight.dtype)

                total_loss = self._compute_parent_loss(mapping.parent, parent_kwargs_list, fp16_outputs)
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl])
            else:
                total_loss = 0.0
                for bl in mapping.balance_layers:
                    quant_func, opt_quant_func = bl_quant_funcs[bl]
                    w_orig = orig_weights[bl].to(device)
                    w_qdq = self._qdq_tool.qdq(
                        w_orig * scales_view,
                        bl_params[bl],
                        quant_func=quant_func,
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

    def _iter_parent_calls(self, stored_args: tuple, stored_kwargs: dict):
        """Yield full or microbatched parent-call args from one cached calibration batch."""
        actual_batch = _detect_parent_batch(stored_args, stored_kwargs)
        if self._smooth_batch_size is None or actual_batch is None or actual_batch <= self._smooth_batch_size:
            yield stored_args, stored_kwargs
            return

        for start in range(0, actual_batch, self._smooth_batch_size):
            end = min(actual_batch, start + self._smooth_batch_size)
            yield _slice_batch_args_kwargs(stored_args, stored_kwargs, actual_batch, start, end)

    @staticmethod
    def _normalize_parent_output(out: Any) -> torch.Tensor:
        """Extract the tensor output used by AWQ parent-output loss."""
        if isinstance(out, tuple):
            return out[0]
        return out

    @torch.no_grad()
    def _run_parent_samples(
        self,
        parent: torch.nn.Module,
        kwargs_list: list[tuple[tuple, dict]],
        offload_to_cpu: bool = False,
    ) -> list[torch.Tensor]:
        param = next(parent.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        outputs = []
        for stored_args, stored_kwargs in kwargs_list:
            for micro_args, micro_kwargs in self._iter_parent_calls(stored_args, stored_kwargs):
                call_args = tuple(move_to_device(a, device) for a in micro_args)
                call_kwargs = {k: move_to_device(v, device) for k, v in micro_kwargs.items()}
                out = self._normalize_parent_output(parent(*call_args, **call_kwargs)).detach()
                if offload_to_cpu:
                    out = out.to("cpu", non_blocking=False)
                outputs.append(out)
        return outputs

    @torch.no_grad()
    def _compute_parent_loss(
        self,
        parent: torch.nn.Module,
        kwargs_list: list[tuple[tuple, dict]],
        fp16_outputs: list[torch.Tensor],
    ) -> float:
        """Replay parent samples and stream MSE loss without storing candidate outputs."""
        param = next(parent.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        loss = torch.tensor(0.0, device=device)
        num_elements = torch.tensor(0, device=device, dtype=torch.long)
        output_idx = 0
        for stored_args, stored_kwargs in kwargs_list:
            for micro_args, micro_kwargs in self._iter_parent_calls(stored_args, stored_kwargs):
                if output_idx >= len(fp16_outputs):
                    return float("inf")
                call_args = tuple(move_to_device(a, device) for a in micro_args)
                call_kwargs = {k: move_to_device(v, device) for k, v in micro_kwargs.items()}
                out = self._normalize_parent_output(parent(*call_args, **call_kwargs))
                fp16_out = fp16_outputs[output_idx].to(device, non_blocking=False)
                loss += torch.nn.functional.mse_loss(
                    fp16_out.float(),
                    out.float(),
                    reduction="sum",
                )
                num_elements += fp16_out.numel()
                output_idx += 1
                del out, fp16_out

        if output_idx != len(fp16_outputs) or num_elements == 0:
            return float("inf")
        return (loss / num_elements).item()

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

        * ``clip_as_init=False`` (default): the clip range is hard-clamped in
          place on the (already smoothed) balance-layer weights, so any
          downstream block quantizer (RTN / SignRound / SignRoundV2) re-derives
          its min/max range from the clipped weights.
        * ``clip_as_init=True``: the weights are left untouched and the clip
          range is stored on the layer (``awq_clip_min`` / ``awq_clip_max``);
          the downstream SignRound / SignRoundV2 quantizer uses it to
          *initialize* its tunable weight range (capping
          ``weight_min``/``weight_max`` or clamping before the scale search) and
          then tunes ``min_scale``/``max_scale`` on top.
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
                clip_range = self._compute_best_clip(bl, feat)
                if clip_range is None:
                    continue
                min_val, max_val = clip_range
                key = getattr(bl, "global_name", None) or name
                if clip_store is not None:
                    if torch.allclose(min_val, -max_val):
                        clip_store[key] = max_val.detach().to("cpu")
                    else:
                        clip_store[key] = {
                            "min": min_val.detach().to("cpu"),
                            "max": max_val.detach().to("cpu"),
                        }
                if self.clip_as_init:
                    # Keep the weights intact; hand the clip to the block
                    # quantizer as the initialization of its weight range.
                    bl.awq_clip_min = min_val.detach()
                    bl.awq_clip_max = max_val.detach()
                else:
                    self._apply_clip(bl, min_val, max_val)

    @torch.no_grad()
    def _compute_best_clip(
        self,
        layer: torch.nn.Module,
        input_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Search the per-group clip threshold that minimizes output MSE.

        Returns ``(min_val, max_val)`` per-group tensors, or ``None`` if
        clipping is not applicable to this layer.
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
        use_asym_clip = params["sym"] is False
        best_min_val_all = []
        best_max_val_all = []
        n_steps = max(1, int(self.clip_max_shrink * self.clip_n_grid))

        for i_b in range(0, out_features, oc_batch):
            w_b = w[i_b : i_b + oc_batch]
            if use_asym_clip:
                org_min_val = w_b.amin(dim=-1, keepdim=True).clamp(max=0)
                org_max_val = w_b.amax(dim=-1, keepdim=True).clamp(min=0)
            else:
                org_max_val = w_b.abs().amax(dim=-1, keepdim=True)  # [oc_b, 1, n_group, 1]
                org_min_val = -org_max_val
            best_min_val = org_min_val.clone()
            best_max_val = org_max_val.clone()
            min_errs = torch.full_like(org_max_val, 1e9)
            org_out = (feat * w_b).sum(dim=-1)  # [oc_b, n_token, n_group]

            for i_s in range(n_steps):
                shrink = 1 - i_s / self.clip_n_grid
                min_val = org_min_val * shrink
                max_val = org_max_val * shrink
                cur_w = torch.clamp(w_b, min_val, max_val)
                cur_w_flat = cur_w.reshape(cur_w.shape[0], n_group * gs)
                q_w = self._qdq_tool.qdq(cur_w_flat, clip_params, quant_func=quant_func).reshape(cur_w.shape)
                cur_out = (feat * q_w).sum(dim=-1)
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                improved = err < min_errs
                min_errs[improved] = err[improved]
                best_min_val[improved] = min_val[improved]
                best_max_val[improved] = max_val[improved]
                del cur_w, q_w, cur_out

            best_min_val_all.append(best_min_val)
            best_max_val_all.append(best_max_val)

        best_min_val = torch.cat(best_min_val_all, dim=0)
        best_max_val = torch.cat(best_max_val_all, dim=0)
        return best_min_val.squeeze(1), best_max_val.squeeze(1)  # [out_features, n_group, 1]

    @torch.no_grad()
    def _apply_clip(self, layer: torch.nn.Module, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        """Hard-clamp the layer weight to ``[min_val, max_val]`` per group."""
        org_dtype = layer.weight.dtype
        min_val = min_val.to(device=layer.weight.device, dtype=org_dtype)
        max_val = max_val.to(device=layer.weight.device, dtype=org_dtype)
        org_shape = layer.weight.shape
        w = layer.weight.data.reshape(*max_val.shape[:2], -1)
        w = torch.clamp(w, min_val, max_val)
        layer.weight.data = w.reshape(org_shape).to(org_dtype)
