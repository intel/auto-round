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
shared-calibration pipeline.

Design invariants (see AWQ_REFACTOR_PLAN.md §0.0 and §3.0):
- ``QuantizationPipeline`` is the *first-class abstraction*; it is NOT just
  AWQ's helper.
- All block-wise scheduling in ``DataDrivenCompressor`` operates against
  ``QuantizationPipeline``, never against a concrete ``AWQTransform``.
- Single-algorithm use is expressed as
  ``QuantizationPipeline(preprocessors=[], block_quantizer=q)``, which is
  semantically identical to the current direct-quantizer path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Union

import torch

from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    resolve_shared_config_values,
    split_quantization_configs,
)
from auto_round.compressors.utils import block_forward
from auto_round.logger import logger
from auto_round.utils.device_manager import device_manager

if TYPE_CHECKING:  # avoid circular imports at runtime
    import torch

    from auto_round.algorithms.base import BasePipelineMember
    from auto_round.algorithms.quantization.base import BaseQuantizer
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.transforms.base import BaseWeightTransformer

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class CalibTiming(IntEnum):
    """When to run the activation calibration hook forward pass.

    SKIP:
        No activation calibration needed (e.g. AWQ W4A16 pure weight-only smooth).
    WITH_REFERENCE:
        Register act-calib hook and run together with the FP reference forward
        (RTN / AutoRound default).  When ``source == InputSource.QUANTIZED_INPUT``, a
        separate hook-only forward is run with ``quantized_input`` instead.
    AFTER_PREPROCESS:
        Run a dedicated forward pass *after* all ``pre_quantize_block`` calls
        complete (i.e. after smooth/rotation transforms are applied).  Used for
        AWQ W8A8 / static-activation quantization where the activation
        distribution changes after smoothing.
    """

    SKIP = 0
    WITH_REFERENCE = 1
    AFTER_PREPROCESS = 2


class InputSource(IntEnum):
    """Which tensor to use as the block input for the act-calib forward pass.

    FP_CACHE:
        The original FP calibration activations cached at the start of the run
        (``enable_quanted_input=False`` path, RTN default).
    QUANTIZED_INPUT:
        The quantized output of the previous block used as the next block's input
        (``enable_quanted_input=True`` path, SignRound / AutoRound).
    """

    FP_CACHE = 0
    QUANTIZED_INPUT = 1


@dataclass
class ActCalibPolicy:
    """Activation calibration policy: when and from what inputs to collect act stats.

    Attributes:
        when:   ``CalibTiming`` — controls the scheduling phase.
        source: ``InputSource`` — which tensor feeds the calibration forward.
                Ignored when ``when == CalibTiming.SKIP``.
    """

    when: CalibTiming = CalibTiming.WITH_REFERENCE
    source: InputSource = InputSource.FP_CACHE

    def __post_init__(self) -> None:
        if not isinstance(self.when, CalibTiming):
            try:
                self.when = CalibTiming(self.when)
            except ValueError as exc:
                raise ValueError(f"ActCalibPolicy.when must be a CalibTiming, got {self.when!r}") from exc
        if not isinstance(self.source, InputSource):
            try:
                self.source = InputSource(self.source)
            except ValueError as exc:
                raise ValueError(f"ActCalibPolicy.source must be an InputSource, got {self.source!r}") from exc

    @classmethod
    def no_collection(cls) -> "ActCalibPolicy":
        """Convenience factory: no activation calibration."""
        return cls(when=CalibTiming.SKIP, source=InputSource.FP_CACHE)


def merge_policies(policies: list["ActCalibPolicy"]) -> "ActCalibPolicy":
    """Merge act-calib policies from multiple pipeline members.

    Rules:
    - ``when``:   take the *latest* timing (``SKIP < WITH_REFERENCE < AFTER_PREPROCESS``).
    - ``source``: when two policies share the same ``when`` but differ in ``source``,
                  that is a compatibility conflict → raise ``ValueError`` fail-fast.
    - Policies with ``when == CalibTiming.SKIP`` do **not** contribute a source
      constraint.

    Returns:
        A merged :class:`ActCalibPolicy`.
    """
    if not policies:
        return ActCalibPolicy.no_collection()

    merged_when = max(policies, key=lambda p: p.when).when

    if merged_when == CalibTiming.SKIP:
        return ActCalibPolicy.no_collection()

    contributing = [p for p in policies if p.when == merged_when]
    sources = {p.source for p in contributing}
    if len(sources) > 1:
        raise ValueError(
            f"Incompatible act-calib policies: multiple algorithms request "
            f"when={merged_when.name!r} but with different input sources: "
            f"{[s.name for s in sources]}. "
            "Use a compatible combination of algorithms or file an issue."
        )

    return ActCalibPolicy(when=merged_when, source=contributing[0].source)


# ---------------------------------------------------------------------------
# Context dataclasses
# ---------------------------------------------------------------------------


# TODO better to follow heng's imp to decouple llm/diffusion
@dataclass
class BlockForward:  # TODO override forward with
    """Stateless block-forward execution engine shared across quantizer & compressor.

    Created **once** by the compressor at init time. Quantizer accesses via
    ``self.compressor.block_forward``.

    Usage::

        # Compressor creates once:
        self.block_forward = BlockForward.from_compressor(self)

        # Quantizer (after bind):
        output = self.compressor.block_forward(block, inputs, others, indices)
    """

    batch_dim: int = 0
    batch_size: int = 8
    device: Union[str, torch.device] = "cpu"
    cache_device: Union[str, torch.device] = "cpu"
    amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    is_diffusion: bool = False
    shared_cache_keys: tuple = ()
    output_config: list[str] | None = None

    # Map block class name → list of output tensor keys returned by that block.
    # The order of keys must match the order of tensors in the block's return tuple.
    # Extend this dict to register new diffusion architectures.
    DIFFUSION_OUTPUT_CONFIGS: ClassVar[dict] = {
        "FluxTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "FluxSingleTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "OvisImageTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "OvisImageSingleTransformerBlock": ["encoder_hidden_states", "hidden_states"],
        "StableAudioDiTBlock": ["hidden_states"],
        "WanTransformerBlock": ["hidden_states"],
    }

    def __post_init__(self) -> None:
        if self.output_config is None:
            self.output_config = ["hidden_states"]

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_compressor(cls, compressor: Any) -> "BlockForward":
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
        inputs: list[torch.Tensor],
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
        device = inputs[0].device if isinstance(inputs, list) else inputs.device

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
            raise RuntimeError("BlockForward.forward: no outputs collected.")

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

    def _normalize_output(self, output: Any, block: "torch.nn.Module" = None) -> torch.Tensor:
        """Normalize block output to a single tensor."""
        if isinstance(output, torch.Tensor):
            return output

        if not isinstance(output, (tuple, list)):
            raise TypeError(f"Block output must be tensor or tuple/list, got {type(output).__name__}.")

        if len(output) == 0:
            raise ValueError("Block output is an empty tuple/list.")

        if self.is_diffusion:
            # Look up per-block-type output config; fall back to instance-level config.
            block_cls_name = block.__class__.__name__ if block is not None else None
            oc = (
                self.DIFFUSION_OUTPUT_CONFIGS.get(block_cls_name, self.output_config)
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

    def _count_samples(self, inputs: Any) -> int:
        if isinstance(inputs, dict):
            hs = inputs.get("hidden_states")
            return len(hs) if isinstance(hs, list) else hs.shape[self.batch_dim]
        elif isinstance(inputs, list):
            return len(inputs)
        else:
            return inputs.shape[self.batch_dim]

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
    block: "torch.nn.Module"
    block_names: list[str]  # scheduling group; len > 1 when nblocks > 1
    block_name: str  # = block_names[0] for single-block; descriptive label for multi
    block_index: int  # 0-based index within the current all_blocks group
    bs: int = 1
    device: Union[str, "torch.device", None] = None
    is_mllm: bool = False  # fail-fast gate for algorithms that don't support MLLM
    is_diffusion: bool = False  # fail-fast gate for algorithms that don't support diffusion
    pbar: Any = None
    modified_fp_params: list = field(default_factory=list)

    def mark_modified_fp_params(self, param_names: list[str]) -> None:
        """Called by preprocessors to declare which FP params were modified in-place."""
        self.modified_fp_params.extend(param_names)


# ---------------------------------------------------------------------------
# QuantizationPipeline
# ---------------------------------------------------------------------------


@dataclass
class QuantizationPipeline:
    """An ordered composition of pre-processing quantizers + one block quantizer.

    The ``preprocessors`` list is order-sensitive: algorithms are applied in
    the listed order (e.g. ``[Rotation, AWQ]``).  There must be **exactly one**
    ``block_quantizer`` (the terminal weight-compression step).

    Single-algorithm use:
        ``QuantizationPipeline(preprocessors=[], block_quantizer=rtn_quantizer)``
        is semantically equivalent to the current direct-quantizer path; the
        compressor's existing ``self.quantizer`` call-sites are transparently
        forwarded to ``block_quantizer`` via a ``@property``.
    """

    preprocessors: list["BaseWeightTransformer"] = field(default_factory=list)
    block_quantizer: "BaseQuantizer" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.block_quantizer is None:
            raise ValueError("QuantizationPipeline requires a non-None block_quantizer.")
        from auto_round.algorithms.quantization.base import BaseQuantizer
        from auto_round.algorithms.transforms.base import BaseWeightTransformer

        for q in self.preprocessors:
            if not isinstance(q, BaseWeightTransformer):
                raise TypeError(
                    f"{type(q).__name__} is listed as a preprocessor but does not " f"inherit BaseWeightTransformer."
                )
        if not isinstance(self.block_quantizer, BaseQuantizer):
            raise TypeError(
                f"{type(self.block_quantizer).__name__} is used as block_quantizer but does not "
                f"inherit BaseQuantizer."
            )

    def all(self) -> "list[BasePipelineMember]":
        """Return all members in pipeline order: preprocessors then block_quantizer."""
        return [*self.preprocessors, self.block_quantizer]

    @classmethod
    def from_configs(cls, configs: list, compressor: Any = None) -> "QuantizationPipeline":
        """Construct a ``QuantizationPipeline`` from a list of algorithm config instances.

        Resolution rules:
        1. If no ``QuantizationConfig`` with a ``BaseQuantizer`` is found in *configs*,
           a default :class:`RTNConfig` is appended automatically.
        2. Instances of ``BaseWeightTransformer`` go into ``preprocessors`` (in order).
        3. Exactly one ``BaseQuantizer`` becomes ``block_quantizer``.
        4. Multiple block-quantization configs raise ``ValueError``.
        5. If ``compressor`` is provided, every member is bound to it.
        """
        from auto_round.algorithms.quantization.base import BaseQuantizer
        from auto_round.algorithms.quantization.config import QuantizationConfig
        from auto_round.algorithms.transforms.base import BaseWeightTransformer

        configs = list(configs)

        # Ensure at least one terminal block quantizer is present; fall back to RTN.
        _, block_quantizer_configs = split_quantization_configs(configs)
        has_quantizer = bool(block_quantizer_configs)
        if not has_quantizer:
            from auto_round.algorithms.quantization.rtn.config import RTNConfig

            configs = list(configs) + [RTNConfig()]

        configs = resolve_shared_config_values(configs)

        def _resolve_cls(cfg):
            alg_cls = get_algorithm_class(cfg)
            if alg_cls is None:
                raise ValueError(f"Unknown algorithm config type {type(cfg).__name__!r}.")
            return alg_cls

        preprocessors = []
        block_quantizers = []

        for cfg in configs:
            if not isinstance(cfg, QuantizationConfig):
                continue
            from auto_round.algorithms.registry import normalize_algorithm_config

            cfg = normalize_algorithm_config(cfg)
            alg_cls = _resolve_cls(cfg)
            q = alg_cls(cfg)
            if compressor is not None:
                q.bind(compressor)
            if isinstance(q, BaseWeightTransformer):
                preprocessors.append(q)
            elif isinstance(q, BaseQuantizer):
                block_quantizers.append(q)
            else:
                raise TypeError(
                    f"Algorithm class {type(q).__name__} must inherit either " "BaseWeightTransformer or BaseQuantizer."
                )

        if len(block_quantizers) > 1:
            raise ValueError(
                f"QuantizationPipeline allows exactly one block-quantization config, "
                f"but got {len(block_quantizers)}: "
                f"{[type(q).__name__ for q in block_quantizers]}. "
                "Ensure only one of RTNConfig / SignRoundConfig / etc. is in the pipeline."
            )

        seen_preprocessors = set()
        for preprocessor in preprocessors:
            name = type(preprocessor).__name__
            if name in seen_preprocessors:
                raise ValueError(
                    f"Duplicate preprocessor {name} in QuantizationPipeline. "
                    "Repeated instances of the same preprocessor are not supported yet."
                )
            seen_preprocessors.add(name)

        return cls(preprocessors=preprocessors, block_quantizer=block_quantizers[0])

    # ── Convenience act-calib helpers ────────────────────────────────────────

    def dispatch_block(self, block: "torch.nn.Module", input_ids, input_others: dict):
        """Dispatch block to device(s) via the pipeline's algorithms.

        Iterates all members; if exactly one overrides the default dispatch_block,
        it is called. If multiple override, warns and uses the first one only.
        If none override, uses the block_quantizer's default (simple .to(device)).
        """
        from auto_round.algorithms.quantization.base import BaseQuantizer

        overriders = []
        for member in self.all():
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



