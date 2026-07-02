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

from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Union

import torch

from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    resolve_shared_config_values,
    split_quantization_configs,
)
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


@dataclass
class BlockIO:
    """Owns per-block calibration inputs, outputs, and batch forward mechanics."""

    _fp_inputs: Any
    _input_others: dict
    _quantized_inputs: Any = None
    _reference_outputs: Any = None
    _quantized_outputs: Any = None
    _active_source: InputSource = InputSource.FP_CACHE
    batch_dim: int = 0
    seqlen: int = 2048
    shared_cache_keys: tuple = ()
    _quantizer: Any = None
    _block: Any = None

    def _inputs_for(self, source: InputSource):
        if source == InputSource.QUANTIZED_INPUT:
            return self._quantized_inputs
        return self._fp_inputs

    def has_quantized_inputs(self) -> bool:
        return self._quantized_inputs is not None

    @property
    def num_samples(self) -> int:
        input_ids = self._inputs_for(self._active_source)
        if input_ids is None:
            return 0
        return self._num_samples(input_ids)

    def _release_references(
        self,
        *,
        keep_fp_inputs: bool = False,
        keep_quantized_inputs: bool = False,
        keep_reference_outputs: bool = False,
        keep_input_others: bool = False,
    ) -> None:
        """Drop large cached references once a block has finished using them."""
        if not keep_fp_inputs:
            self._fp_inputs = None
        if not keep_quantized_inputs:
            self._quantized_inputs = None
            self._quantized_outputs = None
        if not keep_reference_outputs:
            self._reference_outputs = None
        if not keep_input_others:
            self._input_others = {}

    def finish(self) -> None:
        self._release_references()
        self._quantizer = None
        self._block = None

    def _select_inputs_for_source(self, source: InputSource, indices):
        input_ids = self._inputs_for(source)
        if input_ids is None:
            raise ValueError(f"Input source {source.name} is unavailable for this block.")
        return self._select_inputs(input_ids, self._input_others, indices)

    def forward_block_batch(
        self,
        indices: torch.Tensor,
        *,
        device: Union[str, torch.device],
        cache_device: Union[str, torch.device, None] = None,
    ) -> Any:
        quantizer = self._quantizer
        block = self._block
        if quantizer is None or block is None:
            raise ValueError("BlockIO forward_batch requires bound quantizer and block.")

        input_ids, input_others = self._select_inputs_for_source(self._active_source, indices)
        output = self._run_block(block, quantizer, input_ids, input_others, device)
        output = self._normalize_output_for_loss(output)
        if cache_device is not None:
            output = output.to(cache_device)
        return output

    def _run_block(self, block, quantizer, input_ids, input_others, device):
        return quantizer._resolve_block_forward()(
            block,
            input_ids,
            input_others,
            quantizer.model_context.amp,
            quantizer.model_context.amp_dtype,
            device,
            0,
        )

    @torch.no_grad()
    def _collect_outputs(self, block, quantizer, *, source: InputSource, batch_size: int, save: bool = True):
        input_ids = self._inputs_for(source)
        if input_ids is None:
            raise ValueError(f"Input source {source.name} is unavailable for this block.")
        outputs = self._init_output_buffer()
        for start, end in self._iter_batch_ranges(input_ids, batch_size):
            indices = torch.arange(start, end).to(torch.long)
            previous_source = self._active_source
            self._active_source = source
            try:
                output = self.forward_block_batch(
                    indices, device=device_manager.device, cache_device=quantizer.compress_context.cache_device
                )
            finally:
                self._active_source = previous_source
            if save:
                self._append_output(outputs, output, quantizer)
        quantizer.compress_context.clear_memory()
        return outputs

    @torch.no_grad()
    def collect_reference(self, hooks=None) -> Any:
        _ = hooks
        outputs = self._collect_outputs(
            self._block,
            self._quantizer,
            source=InputSource.FP_CACHE,
            batch_size=self._quantizer.batch_size,
        )
        self._reference_outputs = outputs
        if self._active_source == InputSource.QUANTIZED_INPUT:
            self._fp_inputs = None
        return outputs

    @torch.no_grad()
    def collect_quantized_stats(self, hooks=None) -> Any:
        _ = hooks
        if not self.has_quantized_inputs():
            return None
        return self._collect_outputs(
            self._block,
            self._quantizer,
            source=InputSource.QUANTIZED_INPUT,
            batch_size=self._quantizer.batch_size,
            save=False,
        )

    @torch.no_grad()
    def collect_next_inputs(self) -> Any:
        if self._quantizer is None or not self._quantizer.enable_quanted_input:
            return None
        outputs = self._collect_outputs(
            self._block,
            self._quantizer,
            source=self._active_source,
            batch_size=self._quantizer.batch_size,
        )
        self._quantized_outputs = outputs
        return outputs

    def seed_reference(self, fp_inputs: Any, reference_outputs: Any) -> None:
        self._fp_inputs = fp_inputs
        self._reference_outputs = reference_outputs

    def get_reference_outputs(self, indices: torch.Tensor, *, device=None) -> torch.Tensor:
        if self._reference_outputs is None:
            raise ValueError("Reference outputs have not been collected for this block.")
        output = torch.cat([self._reference_outputs[i] for i in indices], dim=self.batch_dim)
        return output.to(device) if device is not None else output

    def _normalize_output_for_loss(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and len(output) == 1 and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(
            "BlockIO forward must return a tensor or a single-tensor tuple/list after normalization. "
            f"Got {type(output).__name__}."
        )

    def _count_input_elements(self, indices, *, source: InputSource | None = None) -> int:
        source = self._active_source if source is None else source
        input_ids = self._inputs_for(source)
        if isinstance(input_ids, dict):
            current_input_ids = [input_ids["hidden_states"][i] for i in indices]
        else:
            current_input_ids = [input_ids[i] for i in indices]
        return sum(t.numel() for t in current_input_ids)

    def count_batch_elements(self, indices: torch.Tensor) -> int:
        return self._count_input_elements(indices, source=self._active_source)

    def _preprocess_block_inputs(self, input_ids, input_others: dict, block):
        return input_ids, input_others

    def _iter_batch_ranges(self, input_ids, batch_size: int):
        for start in range(0, self._num_samples(input_ids), batch_size):
            end = min(self._num_samples(input_ids), start + batch_size)
            yield start, end

    def _num_samples(self, input_ids) -> int:
        return len(input_ids)

    def _init_output_buffer(self):
        return []

    def _append_output(self, outputs, output, quantizer) -> None:
        if quantizer.batch_size == 1:
            outputs.append(output)
        else:
            outputs.extend(list(torch.split(output, 1, dim=self.batch_dim)))

    def _select_inputs(self, input_ids, input_others: dict, indices):
        if isinstance(input_ids, list):
            current_input_ids = [input_ids[i] for i in indices]
            current_input_ids = torch.cat(current_input_ids, dim=self.batch_dim)
        elif isinstance(input_ids, dict):
            current_input_ids = {}
            for key in input_ids.keys():
                current_input_ids[key] = torch.cat([input_ids[key][i] for i in indices], dim=self.batch_dim)
        else:
            raise TypeError(f"Unsupported input container type: {type(input_ids).__name__}")

        current_input_others = {"positional_inputs": input_others["positional_inputs"]}
        for key in input_others.keys():
            if "positional_inputs" in key:
                continue
            if key in self.shared_cache_keys:
                val = input_others[key]
                if isinstance(val, list) and len(val) == 1:
                    current_input_others[key] = val[0]
                elif isinstance(val, list) and len(val) > 1:
                    idx = int(indices[0]) if len(indices) == 1 else 0
                    current_input_others[key] = val[idx] if idx < len(val) else val[0]
                else:
                    current_input_others[key] = val
            elif not isinstance(input_others[key], (str, bool, type(None))):
                current_input_others[key] = [input_others[key][i] for i in indices]
                if len(current_input_others[key]) == 1:
                    current_input_others[key] = current_input_others[key][0]
                else:
                    current_input_others[key] = torch.cat(current_input_others[key], dim=self.batch_dim)
            else:
                current_input_others[key] = input_others[key]
        return current_input_ids, current_input_others


class DiffusionBlockIO(BlockIO):
    """BlockIO variant for diffusion blocks with dict inputs and tuple outputs."""

    def __init__(self, *args, output_config: list[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_config = output_config or ["hidden_states"]

    def _preprocess_block_inputs(self, input_ids, input_others: dict, block):
        if not isinstance(input_ids, dict):
            return input_ids, input_others
        extra_keys = [key for key in list(input_ids.keys()) if key not in self.output_config]
        for key in extra_keys:
            input_others[key] = input_ids.pop(key)
        return input_ids, input_others

    def _run_block(self, block, quantizer, input_ids, input_others, device):
        if not isinstance(input_ids, dict):
            return super()._run_block(block, quantizer, input_ids, input_others, device)
        input_ids = dict(input_ids)
        input_others = dict(input_others)
        hidden_states = input_ids.pop("hidden_states")
        input_others.update(input_ids)
        return quantizer._resolve_block_forward()(
            block,
            hidden_states,
            input_others,
            quantizer.model_context.amp,
            quantizer.model_context.amp_dtype,
            device,
            None,
        )

    def _num_samples(self, input_ids) -> int:
        return len(input_ids["hidden_states"]) if isinstance(input_ids, dict) else len(input_ids)

    def _normalize_output_for_loss(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if not isinstance(output, (tuple, list)):
            raise TypeError(
                "DiffusionBlockIO forward must return a tensor or tuple/list of tensors. "
                f"Got {type(output).__name__}."
            )
        if "hidden_states" not in self.output_config:
            raise ValueError(
                "DiffusionBlockIO requires 'hidden_states' in output_config to normalize outputs for loss."
            )
        hidden_state_index = self.output_config.index("hidden_states")
        if hidden_state_index >= len(output):
            raise ValueError(
                f"Diffusion block output has {len(output)} tensors, but hidden_states index is {hidden_state_index}."
            )
        hidden_states = output[hidden_state_index]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("DiffusionBlockIO expected hidden_states to be a tensor after output normalization.")
        return hidden_states


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
    io: BlockIO
    bs: int = 1
    loss_device: Union[str, "torch.device", None] = None
    device: Union[str, "torch.device", None] = None
    mid_iter_mem_check: bool = False
    is_mllm: bool = False  # fail-fast gate for algorithms that don't support MLLM
    is_diffusion: bool = False  # fail-fast gate for algorithms that don't support diffusion
    pbar: Any = None
    # Names of FP parameters modified in-place by preprocessors (for example,
    # smooth source norm weights).  Populated by pre_quantize_block; read by Compressor
    # during immediate_save to persist FP param changes alongside packed weights.
    # Pipelines without FP-param preprocessors leave this empty, no behavior change.
    modified_fp_params: list = field(default_factory=list)

    def mark_modified_fp_params(self, param_names: list[str]) -> None:
        """Called by preprocessors to declare which FP params were modified in-place."""
        self.modified_fp_params.extend(param_names)

    def collect_reference(self, hooks=None) -> Any:
        return self.io.collect_reference(hooks)

    def collect_quantized_stats(self, hooks=None) -> Any:
        return self.io.collect_quantized_stats(hooks)

    def collect_next_inputs(self) -> Any:
        return self.io.collect_next_inputs()

    def has_quantized_inputs(self) -> bool:
        return self.io.has_quantized_inputs()

    @property
    def num_samples(self) -> int:
        return self.io.num_samples

    def count_batch_elements(self, indices: torch.Tensor) -> int:
        return self.io.count_batch_elements(indices)

    def get_reference_outputs(self, indices: torch.Tensor, *, device=None) -> torch.Tensor:
        return self.io.get_reference_outputs(indices, device=device)

    def forward_block_batch(
        self,
        indices: torch.Tensor,
        *,
        device: Union[str, torch.device],
        cache_device: Union[str, torch.device, None] = None,
    ) -> Any:
        return self.io.forward_block_batch(indices, device=device, cache_device=cache_device)

    def finish(self) -> None:
        self.io.finish()


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
        2. If ``compressor`` indicates a diffusion model, :class:`DiffusionMixin` is
           dynamically prepended to each algorithm class's MRO before instantiation,
           activating diffusion-aware method overrides without touching the class definitions.
        3. Instances of ``BaseWeightTransformer`` go into ``preprocessors`` (in order).
        4. Exactly one ``BaseQuantizer`` becomes ``block_quantizer``.
        5. Multiple block-quantization configs raise ``ValueError``.
        6. If ``compressor`` is provided, every member is bound to it.
        """
        from auto_round.algorithms.quantization.base import BaseQuantizer, DiffusionMixin
        from auto_round.algorithms.quantization.config import QuantizationConfig
        from auto_round.algorithms.transforms.base import BaseWeightTransformer

        is_diffusion = compressor is not None and getattr(compressor.model_context, "is_diffusion", False)
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
            if is_diffusion and not issubclass(alg_cls, DiffusionMixin):
                alg_cls = type(alg_cls.__name__, (DiffusionMixin, alg_cls), {})
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

    def get_merged_policy(self, ctx: "BlockContext") -> ActCalibPolicy:
        """Compute the merged act-calib policy for the current block."""
        policies = [q.get_act_calib_policy(ctx) for q in self.all()]
        return merge_policies(policies)

    def enter_block_forward_hooks(self, ctx: "BlockContext", fwd_stack: ExitStack) -> list:
        """Enter all pipeline members' ``block_forward_hooks`` into *fwd_stack*.

        Iterates over all members (preprocessors then block_quantizer) in order,
        entering each member's ``block_forward_hooks(ctx)`` context manager into
        the provided :class:`contextlib.ExitStack`.

        Returns the hook handles yielded by the terminal ``block_quantizer``
        so the caller can determine whether any act-calib hooks were registered
        (needed to decide whether a second forward with quantized inputs is required).
        """
        self.enter_preprocessor_hooks(ctx, fwd_stack)
        return self.enter_quantizer_hooks(ctx, fwd_stack)

    def enter_preprocessor_hooks(self, ctx: "BlockContext", fwd_stack: ExitStack) -> None:
        """Enter preprocessor hooks only.

        Preprocessor hooks collect stats from the FP reference forward.  They are
        intentionally separate from terminal quantizer hooks so quantizer stats
        can be collected from quantized inputs when required by policy.
        """
        for pre in self.preprocessors:
            fwd_stack.enter_context(pre.block_forward_hooks(ctx))

    def enter_quantizer_hooks(self, ctx: "BlockContext", fwd_stack: ExitStack) -> list:
        """Enter terminal block-quantizer hooks only and return their handles."""
        return fwd_stack.enter_context(self.block_quantizer.block_forward_hooks(ctx))
