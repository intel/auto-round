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
"""Algorithm Fusion Pipeline abstraction.

This module defines the core data structures and utilities for composing
pre-processing algorithms (AWQ smooth, SmoothQuant, Rotation…) with a
block-quantization algorithm (RTN, SignRound/AutoRound…) into a single
shared-calibration bundle.

Design invariants (see AWQ_REFACTOR_PLAN.md §0.0 and §3.0):
- ``AlgorithmComposer`` is the *first-class abstraction*; it is NOT just
  AWQ's helper.
- All block-wise scheduling in ``Compressor`` operates against
  ``AlgorithmComposer``, never against a concrete ``AWQTransform``.
- Single-algorithm use is expressed as
  ``AlgorithmComposer(preprocessors=[], block_quantizer=q)``, which is
  semantically identical to the current direct-quantizer path.
"""
from __future__ import annotations
import torch

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    resolve_shared_config_values,
    split_quantization_configs,
)
from auto_round.compressors.utils import block_forward
from auto_round.logger import logger
from auto_round.utils.device_manager import device_manager

if TYPE_CHECKING:  # avoid circular imports at runtime
    from auto_round.algorithms.base import BaseAlgorithm
    from auto_round.algorithms.quantization.base import BaseQuantizer
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.transforms.base import BasePreprocessor


# ---------------------------------------------------------------------------
# Diffusion block output registry  (建议5 / Suggestion 5)
# ---------------------------------------------------------------------------

#: Maps block class name → ordered list of output tensor keys.
#: Register new diffusion architectures with :func:`register_diffusion_output`.
_DIFFUSION_OUTPUT_REGISTRY: dict[str, list[str]] = {}


def register_diffusion_output(block_cls_name: str, output_keys: list[str]) -> None:
    """Register the output key order for a diffusion transformer block class.

    Args:
        block_cls_name: The ``__class__.__name__`` of the diffusion block
            (e.g. ``"FluxTransformerBlock"``).
        output_keys: Ordered list of tensor keys returned by the block's
            forward pass (e.g. ``["encoder_hidden_states", "hidden_states"]``).
            ``"hidden_states"`` must be present.

    Example::

        register_diffusion_output("MyDiTBlock", ["hidden_states"])
    """
    _DIFFUSION_OUTPUT_REGISTRY[block_cls_name] = output_keys


# Built-in diffusion block registrations.
# Add new architectures here instead of editing BlockRunner internals.
register_diffusion_output("FluxTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("FluxSingleTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("OvisImageTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("OvisImageSingleTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("StableAudioDiTBlock", ["hidden_states"])
register_diffusion_output("WanTransformerBlock", ["hidden_states"])


# ---------------------------------------------------------------------------
# BlockRunner  (建议3 / Suggestion 3)
# ---------------------------------------------------------------------------

# TODO wenhuach better follow heng's imp to decouple llm/diffusion
class BlockForwardRunner:
    """Stateless block-forward execution engine shared across quantizer & compressor.

    Created **once** by the compressor at init time and shared with quantizers
    via :class:`QuantizationRunContext`.

    Usage::

        # Compressor creates once:
        self.block_forward = BlockForwardRunner.from_compressor(self)

        # Quantizer (via _run_ctx):
        output = self._run_ctx.block_forward_runner(block, inputs, others, indices)

    To register a new diffusion block output layout::

        from auto_round.algorithms.pipeline import register_diffusion_output
        register_diffusion_output("MyDiTBlock", ["hidden_states"])
    """

    # Class-level reference to the module-level registry — read-only view for
    # tests and introspection (e.g. ``BlockForwardRunner.DIFFUSION_OUTPUT_CONFIGS``).
    DIFFUSION_OUTPUT_CONFIGS = _DIFFUSION_OUTPUT_REGISTRY

    def __init__(
        self,
        batch_dim: int = 0,
        batch_size: int = 8,
        device: Union[str, "torch.device"] = "cpu",
        cache_device: Union[str, "torch.device"] = "cpu",
        amp: bool = True,
        amp_dtype: "torch.dtype" = None,
        is_diffusion: bool = False,
        shared_cache_keys: tuple = (),
        output_config: "list[str] | None" = None,
    ) -> None:
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.device = device
        self.cache_device = cache_device
        self.amp = amp
        self.amp_dtype = amp_dtype if amp_dtype is not None else torch.bfloat16
        self.is_diffusion = is_diffusion
        self.shared_cache_keys = shared_cache_keys
        self.output_config = output_config if output_config is not None else ["hidden_states"]

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_compressor(cls, compressor: Any) -> "BlockForwardRunner":
        """Create from a compressor instance (called once at compressor init)."""
        model_ctx = getattr(compressor, "model_context", None)
        is_diffusion = getattr(model_ctx, "is_diffusion", False) if model_ctx else False
        output_config = getattr(model_ctx, "output_config", None) if model_ctx else None

        return cls(
            batch_dim=getattr(compressor, "batch_dim", 0),
            batch_size=getattr(compressor, "batch_size", 8),
            device=device_manager.device,
            cache_device=getattr(compressor, "cache_device", "cpu"),
            amp=getattr(compressor, "amp", True),
            amp_dtype=getattr(compressor, "amp_dtype", torch.bfloat16),
            is_diffusion=is_diffusion,
            shared_cache_keys=getattr(compressor, "shared_cache_keys", ()),
            output_config=output_config,
        )

    # ── Core forward ─────────────────────────────────────────────────────────

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        block: "torch.nn.Module",
        inputs: list[torch.Tensor] | dict,
        input_others: dict,
        indices: torch.Tensor | None = None,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Run block forward with batching, output normalization, and cache transfer.

        Args:
            block:        The transformer block.
            inputs:       Cached inputs (list[Tensor] for LLM/MLLM, dict for diffusion).
            input_others: Auxiliary kwargs (attention_mask, position_ids, etc.).
            indices:      Sample indices to forward. None = all samples.

        Returns:
            if indices is not None, this func returns tensor, otherwise list
            Normalized output tensor on ``self.cache_device``.
        """
        is_returned_list = True
        if indices is not None:
            is_returned_list = False
        num_samples = self._count_samples(inputs)
        if isinstance(inputs, list):
            device = inputs[0].device
        elif isinstance(inputs, dict):
            first_val = next(iter(inputs.values()))
            device = first_val[0].device if isinstance(first_val, list) else first_val.device
        else:
            device = inputs.device

        if indices is None:
             indices = torch.arange(num_samples, dtype=torch.long, device=device)
        elif not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long, device=device)
        else:
            indices = indices.to(device=device)

        outputs = []

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_inputs, batch_others = self._select_batch(inputs, input_others, batch_indices)
            raw_output = self._forward_one_batch(block, batch_inputs, batch_others)
            output = self._normalize_output(raw_output, block)
            if is_returned_list and self.batch_size != 1:  # split  it to 1
                output = self.split_outputs(output)
            else:
                output = [output]
            outputs.extend(output)

        if not outputs:
            raise RuntimeError("BlockForwardRunner.forward: no outputs collected.")

        if is_returned_list:
            return outputs
        else:
            if self.batch_size == 1:
                outputs = [output.unsqueeze(dim=self.batch_dim).to(self.cache_device) for output in outputs]

            outputs = torch.cat(outputs, dim=self.batch_dim).to(self.cache_device)

        return outputs.to(self.cache_device)

    # ── Input selection ──────────────────────────────────────────────────────

    def select_batch(
        self,
        inputs: Any,
        input_others: dict,
        indices: torch.Tensor,
    ) -> tuple[Any, dict]:
        """Slice inputs and others by sample indices (public for custom loops)."""
        return self._select_batch(inputs, input_others, indices)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def split_outputs(self, output: torch.Tensor) -> list[torch.Tensor]:
        """Split a batched output back into per-sample tensors."""
        return list(torch.split(output, 1, dim=self.batch_dim))

    # ── Private ──────────────────────────────────────────────────────────────

    def _forward_one_batch(self, block, batch_inputs, batch_others) -> Any:
        """Forward one already-selected batch through the block (raw output)."""
        if isinstance(batch_inputs, dict):
            batch_inputs = dict(batch_inputs)
            batch_others = dict(batch_others)
            hidden_states = batch_inputs.pop("hidden_states")
            batch_others.update(batch_inputs)
            return block_forward(
                block,
                hidden_states,
                batch_others,
                self.amp,
                self.amp_dtype,
                self.device,
                None,
            )
        else:
            return block_forward(  # TODO torch compile wenhuach
                block,
                batch_inputs,
                batch_others,
                self.amp,
                self.amp_dtype,
                self.device,
                0,
            )

    def _count_samples(self, inputs: Any) -> int:
        if isinstance(inputs, dict):
            hs = inputs.get("hidden_states")
            return len(hs) if isinstance(hs, list) else hs.shape[self.batch_dim]
        elif isinstance(inputs, list):
            return len(inputs)
        else:
            return inputs.shape[self.batch_dim]

    def _normalize_output(self, output: Any, block: "torch.nn.Module" = None) -> torch.Tensor:
        """Normalize block output to a single tensor."""
        if isinstance(output, torch.Tensor):
            return output

        if not isinstance(output, (tuple, list)):
            raise TypeError(f"Block output must be tensor or tuple/list, got {type(output).__name__}.")

        if len(output) == 0:
            raise ValueError("Block output is an empty tuple/list.")

        if self.is_diffusion:
            # Look up per-block-type output config from the module-level registry;
            # fall back to instance-level output_config.
            block_cls_name = block.__class__.__name__ if block is not None else None
            oc = (
                _DIFFUSION_OUTPUT_REGISTRY.get(block_cls_name, self.output_config)
                if block_cls_name
                else self.output_config
            )
            idx = oc.index("hidden_states")
            if idx >= len(output):
                raise ValueError(f"Diffusion output has {len(output)} elements, but hidden_states index is {idx}.")
            hs = output[idx]
            if not isinstance(hs, torch.Tensor):
                raise TypeError(f"Expected hidden_states tensor, got {type(hs).__name__}.")
            return hs

        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
        raise TypeError(f"Block output[0] must be tensor, got {type(first).__name__}.")

    def _select_batch(self, inputs, input_others, indices):
        """Select a subset of inputs by indices."""
        batch_dim = self.batch_dim
        shared_cache_keys = self.shared_cache_keys

        if isinstance(inputs, dict):
            selected_inputs = {}
            for key, val in inputs.items():
                if key in shared_cache_keys:
                    if isinstance(val, list) and len(val) == 1:
                        selected_inputs[key] = val[0]
                    elif isinstance(val, list) and len(val) > 1:
                        idx = int(indices[0]) if len(indices) == 1 else 0
                        selected_inputs[key] = val[idx] if idx < len(val) else val[0]
                    else:
                        selected_inputs[key] = val
                else:
                    if isinstance(val, list):
                        selected_inputs[key] = torch.cat([val[i] for i in indices], dim=batch_dim)
                    elif isinstance(val, torch.Tensor):
                        selected_inputs[key] = torch.index_select(val, batch_dim, indices)
                    else:
                        selected_inputs[key] = val
        else:
            if isinstance(inputs, list):
                selected_inputs = torch.cat([inputs[i] for i in indices], dim=batch_dim)
            else:
                selected_inputs = torch.index_select(inputs, batch_dim, indices)

        selected_others = {"positional_inputs": input_others.get("positional_inputs")}

        for key, val in input_others.items():
            if "positional_inputs" in key:
                continue
            if key in shared_cache_keys:
                if isinstance(val, list) and len(val) == 1:
                    selected_others[key] = val[0]
                elif isinstance(val, list) and len(val) > 1:
                    idx = int(indices[0]) if len(indices) == 1 else 0
                    selected_others[key] = val[idx] if idx < len(val) else val[0]
                else:
                    selected_others[key] = val
            elif isinstance(val, list):
                batch_vals = [val[i] for i in indices]
                if len(batch_vals) == 1:
                    selected_others[key] = batch_vals[0]
                else:
                    selected_others[key] = torch.cat(batch_vals, dim=batch_dim)
            elif isinstance(val, torch.Tensor):
                selected_others[key] = torch.index_select(val, batch_dim, indices)
            elif isinstance(val, (str, bool, type(None))):
                selected_others[key] = val
            else:
                selected_others[key] = val
        return selected_inputs, selected_others


# ---------------------------------------------------------------------------
# Context dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BlockContext:
    """Per-block context threaded through the lifecycle hooks.

    Passed to lifecycle methods like ``block_forward_hooks``,
    ``pre_quantize_block``, ``post_quantize_block``, etc.

    ``block_names`` preserves the *scheduling group* (which may contain more
    than one block when ``nblocks > 1``).  Pre-processing algorithms that
    only support single-block operation (e.g. AWQ Phase-1) must check
    ``len(block_names) == 1`` in ``prepare_block_group`` and raise
    ``ValueError`` with a user-readable message.
    """

    model: "torch.nn.Module"
    # block: "torch.nn.Module"
    block_names: list[str]  # scheduling group; len > 1 when nblocks > 1
    block_name: str  # = block_names[0] for single-block; descriptive label for multi
    block_index: int  # 0-based index within the current all_blocks group
    bs: int = 1
    device: Union[str, "torch.device", None] = None
    is_mllm: bool = False  # fail-fast gate for algorithms that don't support MLLM
    is_diffusion: bool = False  # fail-fast gate for algorithms that don't support diffusion
    pbar: Any = None


    # def mark_modified_fp_params(self, param_names: list[str]) -> None:
    #     """Called by preprocessors to declare which FP params were modified in-place."""
    #     self.modified_fp_params.extend(param_names)


# ---------------------------------------------------------------------------
# AlgorithmComposer
# ---------------------------------------------------------------------------

class AlgorithmComposer:
    """An ordered composition of pre-processors + one block quantizer, built from
    a list of algorithm config objects and an optional compressor.

    The ``preprocessors`` list is order-sensitive: algorithms are applied in
    the listed order (e.g. ``[Rotation, AWQ]``).  There must be **exactly one**
    ``block_quantizer`` (the terminal weight-compression step).

    Usage::

        composer = AlgorithmComposer(configs, compressor=self)
    """

    def __init__(self, configs: list, compressor: Any = None) -> None:
        """Build the pipeline from a list of algorithm config instances.

        Resolution rules:

        1. If no ``QuantizationConfig`` with a ``BaseQuantizer`` is found in
           *configs*, a default :class:`RTNConfig` is appended automatically.
        2. ``BasePreprocessor`` instances go into ``preprocessors`` (in order).
        3. Exactly one ``BaseQuantizer`` becomes ``block_quantizer``.
        4. Multiple block-quantization configs raise ``ValueError``.
        5. If *compressor* is provided, every member is bound to it and
           ``block_forward`` / quantization metadata are extracted from it.

        Args:
            configs:    List of algorithm config objects (``QuantizationConfig``
                        subclasses such as :class:`RTNConfig`, :class:`SignRoundConfig`,
                        :class:`AWQConfig`, …).
            compressor: The :class:`~auto_round.compressors.base.BaseCompressor`
                        instance driving this pipeline.  When supplied, members are
                        bound to it and ``block_forward`` is taken from it.
        """
        from auto_round.algorithms.quantization.base import BaseQuantizer
        from auto_round.algorithms.quantization.config import QuantizationConfig
        from auto_round.algorithms.transforms.base import BasePreprocessor

        configs = list(configs)

        _, block_quantizer_configs = split_quantization_configs(configs)
        if not block_quantizer_configs:
            from auto_round.algorithms.quantization.rtn.config import RTNConfig
            configs = configs + [RTNConfig()]

        configs = resolve_shared_config_values(configs)

        preprocessors: list = []
        block_quantizers: list = []

        for cfg in configs:
            if not isinstance(cfg, QuantizationConfig):
                continue
            from auto_round.algorithms.registry import normalize_algorithm_config
            cfg = normalize_algorithm_config(cfg)
            alg_cls = get_algorithm_class(cfg)
            if alg_cls is None:
                raise ValueError(f"Unknown algorithm config type {type(cfg).__name__!r}.")
            q = alg_cls(cfg)
            if compressor is not None:
                q.bind(compressor)
            if isinstance(q, BasePreprocessor):
                preprocessors.append(q)
            elif isinstance(q, BaseQuantizer):
                block_quantizers.append(q)
            else:
                raise TypeError(
                    f"Algorithm class {type(q).__name__} must inherit either "
                    "BasePreprocessor or BaseQuantizer."
                )

        if len(block_quantizers) > 1:
            raise ValueError(
                f"AlgorithmComposer allows exactly one block-quantization config, "
                f"but got {len(block_quantizers)}: "
                f"{[type(q).__name__ for q in block_quantizers]}. "
                "Ensure only one of RTNConfig / SignRoundConfig / etc. is in the pipeline."
            )
        if len(block_quantizers) == 0:
            raise ValueError("No block quantizer found in configs.")

        seen = set()
        for pre in preprocessors:
            name = type(pre).__name__
            if name in seen:
                raise ValueError(
                    f"Duplicate preprocessor {name} in AlgorithmComposer. "
                    "Repeated instances of the same preprocessor are not supported yet."
                )
            seen.add(name)

        self.preprocessors = preprocessors
        #TODO wenhuach support multi quantizers
        self.block_quantizer = block_quantizers[0]

        # Bind compressor-level infrastructure (set before _build_quantizer is called).
        # TODO wenhuach refine
        self.block_forward = BlockForwardRunner.from_compressor(compressor) if compressor is not None else None
        # A little tricky
        self.block_quantizer.bind_block_forward_runner(self.block_forward)
        self.scheme = getattr(compressor, "scheme_context", None)

    # ── Internal hook helpers (act_max calibration) ───────────────────────────

    def _register_act_max_hooks(self, block: "torch.nn.Module") -> list:
        """Register per-module act_max tracking hooks for static activation quantization.

        Returns a list of hook handles that the caller must remove when done.
        """
        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size

        is_act_nv_fp = getattr(self.block_quantizer.config, "is_act_nv_fp", False)

        def collect_act_max(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            if input.numel() == 0:
                return
            input, _, _ = reshape_pad_tensor_by_group_size(input, module.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if is_act_nv_fp:
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                return
            act_max = act_max.to(module.act_max.device)
            if is_act_nv_fp:
                max_val = torch.max(act_max.max(), module.act_max.max())
                module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                module.act_max = torch.max(act_max, module.act_max)

        def should_collect(name, module):
            from auto_round.compressors.utils import check_need_act_calibration
            from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized

            if isinstance(module, tuple(SUPPORTED_LAYER_TYPES)):
                return (
                    hasattr(module, "act_dynamic")
                    and check_need_act_calibration(module.act_dynamic, module.act_data_type, module.act_bits)
                    and check_to_quantized(module)
                )
            if hasattr(module,"bits"):
                act_dynamic = getattr(module,"act_dynamic", True)
                act_data_type = getattr(module, "act_data_type", None)
                act_bits = getattr(module, "act_bits", 16)
                return (
                    module.bits <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(module)
                )
            return False

        handles = []
        if should_collect("", block):
            handles.append(block.register_forward_hook(collect_act_max))
            return handles
        for name, module in block.named_modules():
            if name and should_collect(name, module):
                handles.append(module.register_forward_hook(collect_act_max))
        return handles

    def _get_fp_act_hooks(self, block: "torch.nn.Module") -> list:
        """Register FP-input act_max + quantizer forward hooks."""
        handles = self._register_act_max_hooks(block)
        handles.extend(self.block_quantizer.register_fp_input_forward_hooks(block))
        return handles

    def _get_q_act_hooks(self, block: "torch.nn.Module") -> list:
        """Register Q-input act_max + quantizer forward hooks."""
        handles = self._register_act_max_hooks(block)
        handles.extend(self.block_quantizer.register_qinput_forward_hooks(block))
        return handles

    def _attach_act_max_for_outside_layer(self, layer: "torch.nn.Module", fp_input, q_input) -> None:
        """Compute and attach act_max for an outside-block layer from cached inputs.

        Mirrors the hook-based act_max collection done for in-block layers, but
        iterates over already-cached tensors directly instead of running a forward pass.

        Args:
            layer: The layer module to attach act_max to.
            fp_input: List of FP input tensors collected during calibration.
            q_input: Optional list of quantized input tensors; used instead of
                ``fp_input`` when provided.
        """
        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
        from auto_round.compressors.utils import is_nv_fp

        target_input = q_input if q_input else fp_input
        act_group_size = getattr(layer, "act_group_size", -1)
        act_data_type = getattr(layer, "act_data_type", None)
        is_act_nv_fp_flag = is_nv_fp(act_data_type) if act_data_type else False

        for inp in target_input:
            if isinstance(inp, (tuple, list)):
                inp = inp[0]
            if inp.numel() == 0:
                continue
            inp, _, _ = reshape_pad_tensor_by_group_size(inp, act_group_size)
            act_max = torch.max(torch.abs(inp), dim=-1).values

            if not hasattr(layer, "act_max") or layer.act_max.numel() == 0:
                layer.act_max = act_max
                if is_act_nv_fp_flag:
                    max_val = act_max.max()
                    layer.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                continue

            act_max = act_max.to(layer.act_max.device)
            if is_act_nv_fp_flag:
                max_val = torch.max(act_max.max(), layer.act_max.max())
                layer.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                layer.act_max = torch.max(act_max, layer.act_max)


    def compress_embedding_layer(self):
        self.block_quantizer.quantize_embedding_layer()

    # ── Per-block pipeline orchestration ─────────────────────────────────────

    def compress_block(
        self,
        block: "torch.nn.Module",
        input_ids,
        input_others: dict,
        ctx: "BlockContext",
        *,
        q_input=None,
        valid_token_mask=None,
    ) -> tuple:
        """Run the full per-block algorithm pipeline: calibration → quantization → collection.

        Orchestrates preprocessors, quantizer calibration, and block quantization in
        the canonical order.  Infrastructure concerns (device management, memory cleanup,
        q_input variable reassignment) remain the caller's responsibility.

        The interface mirrors :meth:`~auto_round.algorithms.quantization.base.BaseQuantizer.quantize_block`
        but covers the entire multi-step pipeline including preprocessor calibration and
        output collection.

        Args:
            block: The transformer block to process.
            input_ids: Full-precision (FP) cached inputs.
            input_others: Auxiliary kwargs (attention_mask, position_ids, …).
            ctx: Per-block lifecycle context (:class:`BlockContext`).
            q_input: Quantized-input tensors from the previous block, or ``None``.
            valid_token_mask: Optional mask for per-token loss weighting.

        Returns:
            ``(new_q_input, reference_output)``:

            - *new_q_input*: block output after quantization (``None`` when
              ``enable_quanted_input`` is ``False``).
            - *reference_output*: FP reference output collected before optimization.
        """
        block_forward_fn = self.block_forward

        # ── Step 1: Preprocessor calibration (e.g. AWQ activation stats) ──────
        with torch.no_grad():
            pre_hooks = []
            for pre in self.preprocessors:
                pre_hooks.extend(pre.register_fp_input_forward_hooks(block))
            if pre_hooks:
                block_forward_fn(block, input_ids, input_others)
            for h in pre_hooks:
                h.remove()

            pre_q_hooks = []
            for pre in self.preprocessors:
                if hasattr(pre, "register_qinput_forward_hooks"):
                    pre_q_hooks.extend(pre.register_qinput_forward_hooks(block))
            if pre_q_hooks:
                block_forward_fn(block, q_input if q_input is not None else input_ids, input_others)
            for h in pre_q_hooks:
                h.remove()

        # ── Step 2: pre_quantize_block (stats consolidation + weight transforms) ──
        for pre in self.preprocessors:
            pre.pre_quantize_block(ctx)

        # ── Step 3: Quantizer calibration (act_max, imatrix, etc.) ─────────────
        with torch.no_grad():
            quant_hooks = self._get_fp_act_hooks(block)
            reference_output = block_forward_fn(block, input_ids, input_others)
            for h in quant_hooks:
                h.remove()

            if self.block_quantizer.enable_quanted_input:
                q_hooks = self._get_q_act_hooks(block)
                if q_hooks:
                    block_forward_fn(block, q_input if q_input is not None else input_ids, input_others)
                    for h in q_hooks:
                        h.remove()

        # ── Step 3.5: MoE scale alignment + global scale update ─────────────────
        # Must run after calibration hooks (act_max collected) and before quantize_block.
        act_data_type = self.scheme.act_data_type if self.scheme else None
        act_dynamic = self.scheme.act_dynamic if (self.scheme and self.scheme.act_dynamic is not None) else True
        data_type = self.scheme.data_type if self.scheme else None
        group_size = self.scheme.group_size if self.scheme else -1
        if act_data_type is not None or not act_dynamic:
            from auto_round.compressors.utils import is_nv_fp
            from auto_round.data_type.utils import update_block_global_scale_if_needed
            from auto_round.utils import set_amax_for_all_moe_layers

            if is_nv_fp(act_data_type) or not act_dynamic:
                set_amax_for_all_moe_layers(block, attr_name="act_max")
            update_block_global_scale_if_needed(ctx.model, data_type, group_size)

        # ── Step 4: quantize_block ──────────────────────────────────────────────
        # When quantized input is available from the previous block, use it;
        # otherwise fall back to the FP input.
        effective_input = q_input if q_input is not None else input_ids
        self.block_quantizer.quantize_block(
            block, effective_input, input_others, reference_output, q_input, ctx,
            valid_token_mask=valid_token_mask,
        )

        # ── Step 5: post_quantize_block ─────────────────────────────────────────
        for pre in self.preprocessors:
            pre.post_quantize_block(ctx)

        # ── Step 6: Collect quantized-block outputs for the next block ──────────
        if self.block_quantizer.enable_quanted_input:
            with torch.no_grad():
                new_q_input = block_forward_fn(block, effective_input, input_others)
        else:
            new_q_input = None

        return new_q_input, reference_output

    def compress_layer_outside_block(
        self,
        layer: "torch.nn.Module",
        fp_input=None,
        q_input=None,
        valid_token_mask=None,
        disable_opt_rtn=None, #TODO wenhuach rename this to search_init_scale
    ) -> None:
        """Quantize a single layer that lives outside transformer blocks.

        Mirrors :meth:`compress_block` for the outside-block case: attaches
        act_max calibration when static activation quantization is required,
        then delegates to the block quantizer.

        Args:
            layer: The layer module to quantize. Must have a ``global_name``
                attribute.
            fp_input: Per-sample FP activations for calibration, or ``None``
                to fall back to zero-shot RTN.
            q_input: Optional quantized activations from a previous stage.
            valid_token_mask: Per-sample token masks for loss weighting.
            disable_opt_rtn: Override optimized-RTN; ``None`` defers to quantizer config.
        """
        # Attach act_max for static activation quantization when inputs are available.
        if fp_input is not None:
            from auto_round.compressors.utils import is_nv_fp

            act_data_type = getattr(layer, "act_data_type")
            if act_data_type is None:
                act_data_type = "fp"
            act_dynamic = getattr(layer, "act_dynamic", True)
            if is_nv_fp(act_data_type) or not act_dynamic:
                self._attach_act_max_for_outside_layer(layer, fp_input, q_input)

        # Infrastructure: move layer to the tuning device before handing off to the quantizer.
        device = getattr(layer, "tuning_device", device_manager.device)#TODO this should be handled by compressor
        layer = layer.to(device)

        self.block_quantizer.quantize_layer_outside_block(
            layer,
            fp_input=fp_input,
            q_input=q_input,
            valid_token_mask=valid_token_mask,
            disable_opt_rtn=disable_opt_rtn,
        )

    # ── Convenience act-calib helpers ────────────────────────────────────────

    def members(self) -> list:
        """Return all algorithm members: preprocessors followed by the block quantizer."""
        return list(self.preprocessors) + [self.block_quantizer]

    def dispatch_block(self, block: "torch.nn.Module", input_ids, input_others: dict):
        """Dispatch block to device(s) via the pipeline's algorithms.

        Iterates all members; if exactly one overrides the default dispatch_block,
        it is called. If multiple override, warns and uses the first one only.
        If none override, uses the block_quantizer's default (simple .to(device)).
        """
        from auto_round.algorithms.quantization.base import BaseQuantizer

        overriders = []
        for member in self.members():
            if not hasattr(member, "dispatch_block"):
                continue
            # Check if the member overrides the base default
            if type(member).dispatch_block is not BaseQuantizer.dispatch_block:
                overriders.append(member)

        if len(overriders) > 1:
            names = [type(m).__name__ for m in overriders]
            logger.warning(
                f"Multiple pipeline members override dispatch_block: {names}. "
                f"Only {names[0]} will be used; others are ignored."
            )

        if overriders:
            return overriders[0].dispatch_block(block, input_ids, input_others)
        return self.block_quantizer.dispatch_block(block, input_ids, input_others)

    def prepare_run(self):
        for alg in self.members():
            alg.prepare_run(self)


    def finalize_run(self):
        for alg in self.members():
            alg.finalize_run(self)
