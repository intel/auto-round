# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""Small public boundary between quantization algorithms and data types.

Algorithms describe *what* a layer needs quantized.  This module resolves the
requested datatype and delegates all datatype-specific work to that datatype's
class.  It deliberately contains no quantization math.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable

import torch

# Low-level tensor primitives remain functions.  Quantizer classes below own
# their lifecycle and are the only interface used by algorithms and wrappers.
QUANT_FUNC_WITH_DTYPE: dict[str, Callable] = {}
_QUANTIZER_CLASSES: dict[str, type] = {}
_CANONICAL_DATA_TYPES: dict[str, str] = {}


def _normalize_data_type(data_type: str) -> str:
    """Normalize spelling differences in datatype names for lookup."""
    return data_type.lower().replace("-", "").replace("_", "")


def register_dtype(names):
    """Register a low-level tensor quantization primitive by datatype name."""

    def register(function):
        for name in (names,) if isinstance(names, str) else names:
            QUANT_FUNC_WITH_DTYPE[name] = function
        return function

    return register


def register_quantizer(canonical_id: str, *, aliases: tuple[str, ...] = ()):
    """Register a datatype class under its canonical name and aliases.

    Datatype modules use this decorator at import time.  The mapping exists
    solely to turn user-facing aliases such as ``"int"`` into the class that
    owns their implementation.
    """

    def register(quantizer_class):
        names = (canonical_id, *aliases)
        normalized = tuple(dict.fromkeys(_normalize_data_type(name) for name in names))
        duplicate = next((name for name in normalized if name in _QUANTIZER_CLASSES), None)
        if duplicate is not None:
            raise ValueError(f"Datatype name {duplicate!r} is already registered")
        for name in normalized:
            _QUANTIZER_CLASSES[name] = quantizer_class
            _CANONICAL_DATA_TYPES[name] = canonical_id
        return quantizer_class

    return register


def canonical_data_type(data_type: str) -> str:
    """Return an alias's canonical name for datatype-dependent policy.

    This is intentionally separate from construction: callers such as loss
    policy and block preparation need to classify a datatype without creating
    a quantizer or allocating tensors.
    """
    try:
        return _CANONICAL_DATA_TYPES[_normalize_data_type(data_type)]
    except KeyError as error:
        raise LookupError(f"No datatype quantizer registered for {data_type!r}") from error


def _quantizer_class(data_type: str) -> type:
    """Look up the class that owns a datatype implementation."""
    try:
        return _QUANTIZER_CLASSES[_normalize_data_type(data_type)]
    except KeyError as error:
        raise LookupError(f"No datatype quantizer registered for {data_type!r}") from error


def create_weight_quantizer(data_type: str, spec: "WeightQuantizationSpec", tuning_options: "WeightTuningOptions"):
    """Create a datatype-owned weight quantizer from a resolved layer request.

    The algorithm supplies only generic request data.  The selected datatype
    class decides its own tuned, RTN, optimized, or block behavior.
    """
    canonical = canonical_data_type(data_type)
    quantizer_class = _quantizer_class(data_type)
    try:
        return quantizer_class.from_spec(spec, tuning_options, canonical)
    except AttributeError as error:
        raise LookupError(f"Datatype {data_type!r} supports activation quantization only") from error


def create_activation_quantizer(data_type: str, spec: "ActivationQuantizationSpec"):
    """Create the activation quantizer owned by ``data_type``.

    Keeping this lookup here prevents wrappers and algorithms from importing
    individual datatype modules or duplicating format-specific conditionals.
    """
    canonical_data_type(data_type)
    try:
        return _quantizer_class(data_type).create_activation(spec)
    except AttributeError as error:
        raise LookupError(f"Datatype {data_type!r} has no activation quantizer") from error


def prepare_data_type_block(data_type: str, block, layer_runtimes) -> None:
    """Run the optional block-level setup owned by one datatype class."""
    quantizer_class = _QUANTIZER_CLASSES.get(_normalize_data_type(data_type))
    prepare = getattr(quantizer_class, "prepare_block", None)
    if prepare is not None:
        prepare(block, layer_runtimes)


class WeightExecutionMode(str, Enum):
    """Existing weight-quantization behaviors selected by algorithm settings."""

    TUNED = "tuned"
    RTN = "rtn"
    OPTIMIZED_RTN = "optimized_rtn"


@dataclass(frozen=True)
class WeightTuningOptions:
    """Algorithm-controlled tuning switches passed to a datatype quantizer.

    ``clip_min`` and ``clip_max`` carry AWQ's precomputed clipping range; they
    cannot live on the datatype because AWQ owns how those bounds are learned.
    """

    mode: WeightExecutionMode
    enable_round_tuning: bool
    enable_minmax_tuning: bool
    clip_min: torch.Tensor | None = None
    clip_max: torch.Tensor | None = None


@dataclass(frozen=True)
class WeightQuantizationSpec:
    """Static properties needed to quantize one layer's weights."""

    data_type: str
    bits: int
    group_size: int | tuple[int, int]
    sym: bool
    scale_dtype: torch.dtype
    q_scale_thresh: float = 1e-5
    super_bits: int | None = None
    super_group_size: int | None = None


@dataclass(frozen=True)
class ActivationQuantizationSpec:
    """Static properties needed to quantize one layer's activations."""

    data_type: str
    bits: int
    group_size: int | tuple[int, int]
    sym: bool
    scale_dtype: torch.dtype
    dynamic: bool
    q_scale_thresh: float = 1e-5


@dataclass(frozen=True)
class WeightQuantizationResult:
    """Weight QDQ output plus the tensors required by export and inference."""

    weight: torch.Tensor
    scale: torch.Tensor | dict[str, torch.Tensor] | None = None
    zero_point: torch.Tensor | dict[str, torch.Tensor] | int | float | None = None
    metadata: object | None = None


def state_tunables(state: object) -> dict[str, torch.Tensor]:
    """Return a datatype state's trainable tensors using the common convention."""
    if isinstance(state, dict):
        return state.get("tunables", state)
    return state.tunables


class DataTypeQuantizer:
    """Run a datatype implementation through initialize, QDQ, and write-back.

    This is the only lifecycle wrapper shared by all data types.  Datatype
    classes keep their own state and numerical code; this class only manages
    the sequence and the trainable parameter dictionary.
    """

    def __init__(self, implementation: Any, options: WeightTuningOptions, apply_result):
        self._implementation = implementation
        self._options = options
        self._apply_result = apply_result
        self._state = None
        self.parameters: dict[str, torch.Tensor] = {}

    @classmethod
    def for_layer(cls, layer, *, iters=0, disable_opt_rtn=False, tune_rounding=False, tune_minmax=False):
        """Create the datatype quantizer requested by one resolved layer.

        This is the sole conversion from layer attributes to datatype request
        objects.  The rest of the codebase calls :func:`create_quantizer`.
        """
        if isinstance(layer, Mapping):
            layer = SimpleNamespace(**{"q_scale_thresh": 1e-5, **layer})
        spec = _weight_spec(layer)
        mode = (
            WeightExecutionMode.TUNED
            if iters > 0
            else (WeightExecutionMode.RTN if disable_opt_rtn else WeightExecutionMode.OPTIMIZED_RTN)
        )
        options = WeightTuningOptions(mode, tune_rounding, tune_minmax)
        implementation = create_weight_quantizer(spec.data_type, spec, options)
        return cls(implementation, options, implementation.apply_result)

    def initialize(self, weight: torch.Tensor, *, imatrix=None) -> None:
        """Initialize datatype state and expose its trainable tensors."""
        self._state = self._implementation.create_state(
            weight,
            imatrix=imatrix,
            tuning_options=self._options,
        )
        self.parameters = dict(state_tunables(self._state))

    def quantize(self, weight: torch.Tensor, **parameters) -> torch.Tensor:
        """Return QDQ weights for the current trainable parameter values."""
        self._ensure_initialized()
        return self._implementation.qdq(
            weight,
            self._state,
            tunables={**self.parameters, **parameters},
        ).weight

    def write_back(self, layer, weight: torch.Tensor, *, transpose=False, **parameters) -> None:
        """Materialize QDQ payload and store it on an exportable layer."""
        self._ensure_initialized()
        result = self._implementation.qdq(
            weight,
            self._state,
            tunables={**self.parameters, **parameters},
            materialize=True,
        )
        if transpose:
            result = WeightQuantizationResult(result.weight.t(), result.scale, result.zero_point, result.metadata)
        self._apply_result(layer, result)

    def _ensure_initialized(self) -> None:
        """Reject use before a datatype has created its per-layer state."""
        if self._state is None:
            raise RuntimeError("Call initialize() before quantize() or write_back().")


def _weight_spec(layer) -> WeightQuantizationSpec:
    """Extract the weight request from a resolved quantized layer."""
    scale_dtype = getattr(layer, "scale_dtype", torch.float32)
    return WeightQuantizationSpec(
        data_type=layer.data_type,
        bits=layer.bits,
        group_size=layer.group_size,
        sym=layer.sym,
        scale_dtype=scale_dtype,
        q_scale_thresh=getattr(layer, "q_scale_thresh", 1e-5),
        super_bits=getattr(layer, "super_bits", None),
        super_group_size=getattr(layer, "super_group_size", None),
    )


def _activation_spec(layer) -> ActivationQuantizationSpec:
    """Extract the activation request from a resolved quantized layer."""
    return ActivationQuantizationSpec(
        data_type=layer.act_data_type,
        bits=layer.act_bits,
        group_size=layer.act_group_size,
        sym=layer.act_sym,
        scale_dtype=getattr(layer, "scale_dtype", torch.float32),
        dynamic=layer.act_dynamic,
    )


def create_quantizer(layer, *, iters=None, disable_opt_rtn=False, tune_rounding=False, tune_minmax=False):
    """Create the datatype-owned weight quantizer for a resolved layer."""
    return DataTypeQuantizer.for_layer(
        layer,
        iters=getattr(layer, "iters", 0) if iters is None else iters,
        disable_opt_rtn=disable_opt_rtn,
        tune_rounding=tune_rounding,
        tune_minmax=tune_minmax,
    )


def cache_activation_quantizer(layer):
    """Create and cache a layer's activation quantizer when activation is quantized."""
    if getattr(layer, "act_bits", 16) > 8:
        return None
    if hasattr(layer, "_ar_activation_quantizer"):
        return layer._ar_activation_quantizer
    quantizer = create_activation_quantizer(layer.act_data_type, _activation_spec(layer))
    layer._ar_activation_quantizer = quantizer
    return quantizer


def activation_quantizer_for_layer(layer, *, scale_dtype=None):
    """Create the activation quantizer described by a layer's configuration."""
    spec = _activation_spec(layer)
    if scale_dtype is not None:
        from dataclasses import replace

        spec = replace(spec, scale_dtype=scale_dtype)
    return create_activation_quantizer(layer.act_data_type, spec)


def quantize_activation(tensor, config):
    """Apply dynamic activation quantization using a simple configuration mapping."""
    spec = ActivationQuantizationSpec(
        data_type=config["data_type"],
        bits=config["bits"],
        group_size=config["group_size"],
        sym=config["sym"],
        scale_dtype=tensor.dtype,
        dynamic=True,
    )
    return create_activation_quantizer(spec.data_type, spec).qdq(tensor)
