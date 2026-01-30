# Copyright (c) 2025 Intel Corporation
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

"""
Extensible Module Weight Type Conversion Framework.

This module provides a registry-based system for detecting and converting quantized
module weights to high precision.

================================================================================
                                TABLE OF CONTENTS
================================================================================

1. CORE FRAMEWORK (Lines ~60-180)
   - ModuleWeightType: Enum of supported weight types
   - WeightTypeHandler: Abstract base class for handlers
   - Registry functions: register_weight_type_handler, get_handler, etc.

2. PUBLIC API (Lines ~180-260)
   - detect_weight_type(): Detect weight type of a single layer
   - detect_model_weight_type(): Detect weight type in a model
   - is_quantized_layer(): Check if layer is quantized
   - is_quantized_model(): Check if model has quantized layers
   - convert_to_high_precision_if_necessary(): Main conversion function

3. HANDLER IMPLEMENTATIONS (Lines ~260+)
   - FP8BlockHandler: Fully implemented for FP8 block-wise quantization
   - MXFP8Handler: Placeholder (TODO)
   - MXFP4Handler: Placeholder (TODO)
   - NVFP4Handler: Placeholder (TODO)

================================================================================
                              QUICK START GUIDE
================================================================================

Usage - Detect and Convert:
    >>> from auto_round.utils.weight_type_conversion import (
    ...     convert_to_high_precision_if_necessary,
    ...     is_quantized_model,
    ... )
    >>> if is_quantized_model(model):
    ...     model = convert_to_high_precision_if_necessary(model)

Adding a New Weight Type Handler:
    1. Add new type to ModuleWeightType enum
    2. Create handler class inheriting from WeightTypeHandler
    3. Register with @register_weight_type_handler decorator

    Example:
        @register_weight_type_handler(ModuleWeightType.MY_NEW_TYPE)
        class MyNewTypeHandler(WeightTypeHandler):
            def detect_layer(self, module): ...
            def detect_model(self, model): ...
            def convert_layer(self, layer, dtype, device, to_cpu): ...
            def convert_model(self, model, dtype, device, to_cpu): ...

================================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import fields
from enum import Enum, auto
from typing import Callable, Dict, Optional, Type, Union

import torch

from auto_round.logger import logger

# ============================================================================
# Section 1: CORE FRAMEWORK - Types, Base Classes, and Registry
# ============================================================================


class ModuleWeightType(Enum):
    """Enumeration of supported quantized module weight types.

    When adding a new weight type:
    1. Add the new type here
    2. Create a handler class (see Section 3)
    3. Register with @register_weight_type_handler decorator
    """

    FP8_BLOCK = auto()  # FP8 with block-wise scaling (fully implemented)
    MXFP8 = auto()  # MX FP8 (placeholder)
    MXFP4 = auto()  # MX FP4 (placeholder)
    NVFP4 = auto()  # NV FP4 (placeholder)


class WeightTypeHandler(ABC):
    """Abstract base class for weight type detection and conversion handlers.

    Subclasses must implement:
        - detect_layer: Check if a single layer is of this weight type
        - detect_model: Check if a model contains layers of this weight type
        - convert_layer: Convert a single layer to high precision
        - convert_model: Convert all matching layers in a model
    """

    @abstractmethod
    def detect_layer(self, module: torch.nn.Module) -> bool:
        """Check if a single layer is of this weight type.

        Args:
            module: The module to check.

        Returns:
            True if the module is of this weight type, False otherwise.
        """
        pass

    @abstractmethod
    def detect_model(self, model: torch.nn.Module) -> bool:
        """Check if a model contains layers of this weight type.

        Args:
            model: The model to check.

        Returns:
            True if the model contains layers of this weight type, False otherwise.
        """
        pass

    @abstractmethod
    def convert_layer(
        self,
        layer: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert a single layer to high precision.

        Args:
            layer: The quantized layer to convert.
            dtype: Target dtype for the converted layer.
            device: Target device for the conversion.
            to_cpu: If True, move converted layer to CPU.

        Returns:
            A new high-precision layer with dequantized weights.
        """
        pass

    @abstractmethod
    def convert_model(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert all matching layers in a model to high precision.

        Args:
            model: The model containing quantized layers.
            dtype: Target dtype for the converted layers.
            device: Target device for the conversion.
            to_cpu: If True, move converted layers to CPU.

        Returns:
            The model with quantized layers converted to high precision.
        """
        pass


# --- Handler Registry ---

_WEIGHT_TYPE_HANDLERS: Dict[ModuleWeightType, WeightTypeHandler] = {}


def register_weight_type_handler(weight_type: ModuleWeightType):
    """Decorator to register a weight type handler.

    Args:
        weight_type: The ModuleWeightType this handler supports.

    Returns:
        Decorator function that registers the handler class.

    Example:
        @register_weight_type_handler(ModuleWeightType.MXFP4)
        class MXFP4Handler(WeightTypeHandler):
            ...
    """

    def decorator(handler_cls: Type[WeightTypeHandler]):
        if not issubclass(handler_cls, WeightTypeHandler):
            raise TypeError(f"Handler {handler_cls.__name__} must be a subclass of WeightTypeHandler")
        _WEIGHT_TYPE_HANDLERS[weight_type] = handler_cls()
        return handler_cls

    return decorator


def get_handler(weight_type: ModuleWeightType) -> Optional[WeightTypeHandler]:
    """Get the registered handler for a weight type.

    Args:
        weight_type: The ModuleWeightType to get the handler for.

    Returns:
        The registered handler, or None if not registered.
    """
    return _WEIGHT_TYPE_HANDLERS.get(weight_type)


def get_all_handlers() -> Dict[ModuleWeightType, WeightTypeHandler]:
    """Get all registered weight type handlers.

    Returns:
        Dictionary mapping weight types to their handlers.
    """
    return _WEIGHT_TYPE_HANDLERS.copy()


# ============================================================================
# Section 2: PUBLIC API - Detection and Conversion Functions
# ============================================================================
def detect_weight_type(module: torch.nn.Module) -> Optional[ModuleWeightType]:
    """Detect the weight type of a single module.

    Iterates through all registered handlers and returns the first matching type.

    Args:
        module: The module to check.

    Returns:
        The detected ModuleWeightType, or None if no match found.
    """
    for weight_type, handler in _WEIGHT_TYPE_HANDLERS.items():
        if handler.detect_layer(module):
            return weight_type
    return None


def detect_model_weight_type(model: torch.nn.Module) -> Optional[ModuleWeightType]:
    """Detect the weight type used in a model.

    Iterates through all registered handlers and returns the first matching type.

    Args:
        model: The model to check.

    Returns:
        The detected ModuleWeightType, or None if no match found.
    """
    for weight_type, handler in _WEIGHT_TYPE_HANDLERS.items():
        if handler.detect_model(model):
            return weight_type
    return None


def is_quantized_layer(module: torch.nn.Module) -> bool:
    """Check if a module is a quantized layer of any registered type.

    Args:
        module: The module to check.

    Returns:
        True if the module is a quantized layer, False otherwise.
    """
    return detect_weight_type(module) is not None


def is_quantized_model(model: torch.nn.Module) -> bool:
    """Check if a model contains quantized layers of any registered type.

    Args:
        model: The model to check.

    Returns:
        True if the model contains quantized layers, False otherwise.
    """
    return detect_model_weight_type(model) is not None


# --- Model Marking Functions ---
def check_and_mark_quantized_model(model: torch.nn.Module) -> Optional[ModuleWeightType]:
    """Check if model contains quantized layers and mark them accordingly.

    This function scans the model for quantized layers and marks them with
    `is_quantized_linear = True`. It also sets `quantized_weight_type` on the
    model to indicate the detected weight type.

    Args:
        model: The model to check and mark.

    Returns:
        The detected ModuleWeightType if quantized layers are found, None otherwise.
    """
    # Check if already marked
    if hasattr(model, "quantized_weight_type") and model.quantized_weight_type is not None:
        return model.quantized_weight_type

    for weight_type, handler in _WEIGHT_TYPE_HANDLERS.items():
        for n, m in model.named_modules():
            if handler.detect_layer(m):
                m.is_quantized_linear = True
                if not hasattr(model, "quantized_weight_type"):
                    model.quantized_weight_type = weight_type

    return getattr(model, "quantized_weight_type", None)


def is_quantized_input_model(model: torch.nn.Module) -> Optional[ModuleWeightType]:
    """Check if a model has quantized input weights and return the weight type.

    This checks for the `quantized_weight_type` attribute set by
    `check_and_mark_quantized_model`.

    Args:
        model: The model to check.

    Returns:
        The ModuleWeightType if the model has quantized weights, None otherwise.
    """
    if hasattr(model, "quantized_weight_type") and model.quantized_weight_type is not None:
        return model.quantized_weight_type
    return None


# --- Main Conversion Function ---
def convert_to_high_precision_if_necessary(
    model_or_layer: torch.nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    to_cpu: bool = False,
) -> torch.nn.Module:
    """Convert quantized layer(s) to high-precision Linear layer(s) if necessary.

    This function automatically detects the weight type and uses the appropriate
    handler for conversion. If the input is not quantized, it returns unchanged.

    Args:
        model_or_layer: Either a single layer, a block, or an entire model.
        dtype: Target dtype for the converted layer(s). Defaults to torch.bfloat16.
        device: Target device for the conversion. Defaults to "cpu".
        to_cpu: If True, move converted layers to CPU. Defaults to False.

    Returns:
        If input is a quantized layer: A new high-precision layer with dequantized weights.
        If input is a model containing quantized layers: The model with layers converted.
        If input has no quantized layers: Returns the input unchanged.
    """
    # Check if it's a single quantized layer
    layer_type = detect_weight_type(model_or_layer)
    if layer_type is not None:
        handler = get_handler(layer_type)
        return handler.convert_layer(model_or_layer, dtype, device, to_cpu)

    # Check if it's a model containing quantized layers
    model_type = detect_model_weight_type(model_or_layer)
    if model_type is not None:
        handler = get_handler(model_type)
        return handler.convert_model(model_or_layer, dtype, device, to_cpu)

    # No quantized layers, return unchanged
    return model_or_layer


# ============================================================================
# Section 3: HANDLER IMPLEMENTATIONS
# ============================================================================


# ----------------------------------------------------------------------------
# FP8 Block Handler - Full Implementation (Reference Example)
# ----------------------------------------------------------------------------
@register_weight_type_handler(ModuleWeightType.FP8_BLOCK)
class FP8BlockHandler(WeightTypeHandler):
    """Handler for FP8 block-wise quantized layers.

    This handler supports:
        - FP8Linear layers with block-wise scaling
        - CompressedLinear layers with FP8 compression
        - torch.nn.Linear layers with FP8 dtype weights
    """

    # Registry mapping FP8 layer class names to their dequantization handlers
    FP8_DEQUANT_REGISTRY: Dict[str, Callable] = {}

    @classmethod
    def register_fp8_layer(cls, layer_name: str):
        """Register a dequantization handler for an FP8 layer type.

        Args:
            layer_name: The class name of the FP8 layer type.

        Returns:
            Decorator function that registers the handler.
        """

        def decorator(fn: Callable):
            cls.FP8_DEQUANT_REGISTRY[layer_name] = fn
            return fn

        return decorator

    def _dequant_layer(self, layer: torch.nn.Module, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Dequantize an FP8 layer using the internal registry.

        Args:
            layer: The FP8 layer to dequantize.
            dtype: Target dtype for dequantized weights.
            device: Target device for dequantization.

        Returns:
            Dequantized weight tensor.

        Raises:
            NotImplementedError: If the layer type is not registered.
        """
        name = layer.__class__.__name__
        if name not in self.FP8_DEQUANT_REGISTRY:
            raise NotImplementedError(
                f"Unsupported FP8 layer type: {name}. " f"Supported types: {list(self.FP8_DEQUANT_REGISTRY.keys())}"
            )
        return self.FP8_DEQUANT_REGISTRY[name](layer, dtype=dtype, device=device)

    def detect_layer(self, module: torch.nn.Module) -> bool:
        """Check if a module is an FP8 linear layer."""
        # Check explicit marker
        if hasattr(module, "is_quantized_linear") and module.is_quantized_linear:
            return True

        # Check registry for supported FP8 layer types
        layer_name = module.__class__.__name__
        if layer_name in self.FP8_DEQUANT_REGISTRY:
            return True

        # Fallback: Check for FP8 dtype (for torch.nn.Linear with FP8 weights)
        if type(module) == torch.nn.Linear and module.weight is not None:
            if str(module.weight.dtype).startswith("torch.float8"):
                return True

        return False

    def detect_model(self, model: torch.nn.Module) -> bool:
        """Check if a model contains FP8 layers."""
        if hasattr(model, "quantized_weight_type"):
            return model.quantized_weight_type == ModuleWeightType.FP8_BLOCK
        return False

    def convert_layer(
        self,
        layer: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert a single FP8 layer to a standard Linear layer."""
        from auto_round.schemes import QuantizationScheme
        from auto_round.utils.device import is_gaudi2
        from auto_round.utils.model import set_module

        new_layer = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None, dtype=dtype)
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data.to(dtype=dtype))

        # Copy quantization scheme attributes
        scheme_keys = (f.name for f in fields(QuantizationScheme))
        for key in tuple(scheme_keys) + ("global_name", "scale_dtype"):
            setattr(new_layer, key, getattr(layer, key, None))

        # Handle Gaudi2 device compatibility
        if is_gaudi2():
            device = "cpu"

        # Use registry-based dequantization
        dq_weight = self._dequant_layer(layer, dtype=dtype, device=device)
        new_layer.weight.data.copy_(dq_weight.to(dtype=dtype))

        if to_cpu:
            new_layer = new_layer.to("cpu")

        return new_layer

    def convert_model(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert all FP8 layers in a model to high-precision Linear layers."""
        from auto_round.utils.device import clear_memory
        from auto_round.utils.model import set_module

        cnt = 0
        for n, m in model.named_modules():
            if self.detect_layer(m):
                new_module = self.convert_layer(m, dtype, device, to_cpu)
                set_module(model, n, new_module)
                cnt += 1
                if cnt % 10 == 0:
                    clear_memory()
        return model


# --- FP8 Layer Dequantization Handlers ---
# Register specific dequantization logic for different FP8 layer types


@FP8BlockHandler.register_fp8_layer("CompressedLinear")
def _dequant_compressed_linear(layer, dtype=torch.bfloat16, device: str = "cpu"):
    """Dequantize CompressedLinear layer using compressor."""
    layer = layer.to(device)
    return layer.compressor.decompress_module(layer)


@FP8BlockHandler.register_fp8_layer("FP8Linear")
def _dequant_fp8_linear(layer, dtype=torch.bfloat16, device: str = "cpu"):
    """Dequantize FP8Linear layer using block-based dequantization."""
    from auto_round.utils.model import dequant_block_fp8_weight

    layer = layer.to(device)
    weight_scale = layer.weight_scale if hasattr(layer, "weight_scale") else layer.weight_scale_inv
    data_type = getattr(layer, "data_type", None)
    return dequant_block_fp8_weight(layer.weight, weight_scale, layer.block_size, data_type=data_type)


# ----------------------------------------------------------------------------
# Placeholder Handlers for Unimplemented Types
# ----------------------------------------------------------------------------


def _create_placeholder_handler(weight_type_name: str) -> Type[WeightTypeHandler]:
    """Factory function to create placeholder handlers for unimplemented weight types."""

    class PlaceholderHandler(WeightTypeHandler):
        """Placeholder handler - detection always returns False."""

        def detect_layer(self, module: torch.nn.Module) -> bool:
            return False

        def detect_model(self, model: torch.nn.Module) -> bool:
            return False

        def convert_layer(self, layer, dtype=torch.bfloat16, device="cpu", to_cpu=False):
            raise NotImplementedError(f"{weight_type_name} layer conversion not yet implemented")

        def convert_model(self, model, dtype=torch.bfloat16, device="cpu", to_cpu=False):
            raise NotImplementedError(f"{weight_type_name} model conversion not yet implemented")

    PlaceholderHandler.__name__ = f"{weight_type_name}Handler"
    PlaceholderHandler.__doc__ = (
        f"Placeholder handler for {weight_type_name}. TODO: Implement detection and conversion."
    )
    return PlaceholderHandler


# Register placeholder handlers for unimplemented types
register_weight_type_handler(ModuleWeightType.MXFP8)(_create_placeholder_handler("MXFP8"))
register_weight_type_handler(ModuleWeightType.MXFP4)(_create_placeholder_handler("MXFP4"))
register_weight_type_handler(ModuleWeightType.NVFP4)(_create_placeholder_handler("NVFP4"))
