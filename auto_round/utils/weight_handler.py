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

Table of Contents:
    1. CORE FRAMEWORK
       - ModuleWeightType: Enum of supported weight types
       - WeightTypeHandler: Abstract base class for handlers
       - Registry functions: register_weight_type_handler, get_handler, etc.

    2. PUBLIC API
       - detect_weight_type(): Detect weight type of a layer or model
       - check_and_mark_quantized_module(): Check and mark quantized layers
       - is_quantized_input_module(): Check if model has quantized weights
       - convert_module_to_hp_if_necessary(): Main conversion function

    3. HANDLER IMPLEMENTATIONS
       - FP8Handler: Fully implemented for FP8 block-wise quantization
       - MXFP8Handler: CompressedLinear with MXFP8PackedCompressor
       - MXFP4Handler: CompressedLinear with MXFP4PackedCompressor
       - NVFP4Handler: CompressedLinear with NVFP4PackedCompressor

Quick Start Guide:
    Usage - Detect and Convert:
        >>> from auto_round.utils.weight_handler import (
        ...     check_and_mark_quantized_module,
        ...     convert_module_to_hp_if_necessary,
        ... )
        >>> check_and_mark_quantized_module(model)
        >>> model = convert_module_to_hp_if_necessary(model)

    Adding a New Weight Type Handler:
        1. Add new type to ModuleWeightType enum
        2. Create handler class inheriting from WeightTypeHandler
        3. Register with @register_weight_type_handler decorator

        Example:
            @register_weight_type_handler(ModuleWeightType.MY_NEW_TYPE)
            class MyNewTypeHandler(WeightTypeHandler):
                def detect_layer(self, module): ...
                def convert_layer(self, layer, dtype, device, to_cpu): ...
"""

import os
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from dataclasses import fields
from enum import Enum, auto
from typing import Callable, Dict, Optional, Set, Type

import psutil
import torch

from auto_round import envs
from auto_round.logger import logger

# ============================================================================
# Section 0: UTILITY HELPERS
# ============================================================================


def _pad_weight(weight: torch.Tensor, block_size: list) -> tuple[torch.Tensor, int, int]:
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode="constant", value=0)
    return padded_weight, M, N  # Return original dimensions for unpadding


def _unpad_weight(weight: torch.Tensor, original_M: int, original_N: int, keep_first_dim: bool = False) -> torch.Tensor:
    """Removes padding from the matrix to restore its original shape."""
    if (weight.shape[-2] == original_M) and (weight.shape[-1] == original_N):
        return weight
    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]


class with_thread_limits(ContextDecorator):
    """
    Context manager and decorator to temporarily set AR_OMP_NUM_THREADS and PyTorch threads.
    Inspired by vLLM's thread limit decorator.
    https://github.com/HabanaAI/vllm-fork/blob/f943a89a20e0e57bca64e1cca05469bfcaaec6f8/vllm/worker/hpu_model_runner.py#L1063-L1115
    """

    def __init__(self, div=1):
        self.div = div
        self.old_omp = None
        self.old_torch = None

    def __enter__(self):
        # Save original settings
        self.old_omp = envs.AR_OMP_NUM_THREADS
        self.old_torch = torch.get_num_threads()

        try:
            if psutil is not None:
                num_cores = len(psutil.Process().cpu_affinity() or [])
                if num_cores == 0:
                    num_cores = os.cpu_count() or 1
            else:
                num_cores = os.cpu_count() or 1

            # Set new limits
            new_threads = max(1, num_cores // self.div)
            os.environ["AR_OMP_NUM_THREADS"] = str(new_threads)
            torch.set_num_threads(new_threads)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original settings
        if self.old_omp is None:
            os.environ.pop("AR_OMP_NUM_THREADS", None)
        else:
            os.environ["AR_OMP_NUM_THREADS"] = self.old_omp
        torch.set_num_threads(self.old_torch)


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

    FP8 = auto()  # FP8 with block-wise scaling (FP8Linear/CompressedLinear from transformers)
    MXFP8 = auto()  # MX FP8 (CompressedLinear with MXFP8PackedCompressor)
    MXFP4 = auto()  # MX FP4 (CompressedLinear with MXFP4PackedCompressor)
    NVFP4 = auto()  # NV FP4 (CompressedLinear with NVFP4PackedCompressor)


class WeightTypeHandler(ABC):
    """Abstract base class for weight type detection and conversion handlers.

    Subclasses must implement:
        - detect_layer(): Check if a single layer is of this weight type
        - convert_layer(): Convert a single layer to high precision
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
    """Detect the weight type of a module or model.

    First checks if the module itself has a quantized_weight_type attribute.
    If not found, scans all submodules to find any with quantized_weight_type attribute.

    Args:
        module: The module or model to check.

    Returns:
        The detected ModuleWeightType, or None if no match found.
    """
    # Check if module itself is marked
    if hasattr(module, "quantized_weight_type") and module.quantized_weight_type is not None:
        return module.quantized_weight_type

    # Scan submodules for quantized_weight_type attribute
    for submodule in module.modules():
        if hasattr(submodule, "quantized_weight_type") and submodule.quantized_weight_type is not None:
            return submodule.quantized_weight_type

    return None


# --- Model Marking Functions ---
def check_and_mark_quantized_module(model: torch.nn.Module) -> Set[ModuleWeightType]:
    """Check if model contains quantized layers and mark them accordingly.

    This function scans the model (including the model itself) for quantized layers using
    handlers' detect_layer method (which checks actual characteristics). It marks detected
    layers with `quantized_weight_type`.

    Args:
        model: The model to check and mark.

    Returns:
        A set of detected ModuleWeightType values. Empty set if no quantized layers found.
    """
    detected_types: Set[ModuleWeightType] = set()
    for weight_type, handler in _WEIGHT_TYPE_HANDLERS.items():
        # Check model itself first
        if handler.detect_layer(model):
            model.quantized_weight_type = weight_type
            detected_types.add(weight_type)

        # Then check all submodules
        for n, m in model.named_modules():
            # Use handler to detect based on actual characteristics
            if handler.detect_layer(m):
                # Mark the layer itself
                m.quantized_weight_type = weight_type
                # Record detected types
                detected_types.add(weight_type)

    return detected_types


def is_quantized_input_module(model: torch.nn.Module) -> Optional[ModuleWeightType]:
    """Check if a model has quantized input weights and return the weight type.

    This traverses all submodules to check for the `quantized_weight_type` attribute
    set by `check_and_mark_quantized_module`.

    Args:
        model: The model to check.

    Returns:
        The ModuleWeightType if the model has quantized weights, None otherwise.
    """
    # Check model itself first
    if hasattr(model, "quantized_weight_type") and model.quantized_weight_type is not None:
        return model.quantized_weight_type

    # Traverse all submodules to find quantized layers
    for module in model.modules():
        if hasattr(module, "quantized_weight_type") and module.quantized_weight_type is not None:
            return module.quantized_weight_type

    return None


# --- Main Conversion Function ---
def convert_module_to_hp_if_necessary(
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
    from auto_round.utils.device import clear_memory
    from auto_round.utils.model import set_module

    # Check if it's a single quantized layer (has the attribute directly)
    if hasattr(model_or_layer, "quantized_weight_type") and model_or_layer.quantized_weight_type is not None:
        handler = get_handler(model_or_layer.quantized_weight_type)
        return handler.convert_layer(model_or_layer, dtype, device, to_cpu)

    # Otherwise, traverse model and convert all quantized layers
    # Get handler for each layer to support mixed quantization types
    cnt = 0
    for n, m in model_or_layer.named_modules():
        if hasattr(m, "quantized_weight_type") and m.quantized_weight_type is not None:
            handler = get_handler(m.quantized_weight_type)
            new_module = handler.convert_layer(m, dtype, device, to_cpu)
            set_module(model_or_layer, n, new_module)
            cnt += 1
            if cnt % 10 == 0:
                clear_memory()

    return model_or_layer


# ============================================================================
# Section 3: FP8 BLOCK DEQUANTIZATION HELPERS
# ============================================================================


def _pad_block_fp8_weight_naive(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_size: list
) -> tuple[torch.Tensor, int, int]:
    assert len(block_size) == 2

    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = _pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n

    return weight, orig_M, orig_N


@with_thread_limits()
def _dequant_fp8_linear_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_size: list = None, data_type: str = None
) -> torch.Tensor:
    """Core dequantization logic for block-wise FP8 weights."""
    dtype = torch.bfloat16
    if weight_scale is None:
        return weight

    # If weight is stored as uint8, view it as float8_e4m3fn
    if weight.element_size() == 1 and weight.dtype != torch.float8_e4m3fn:
        weight = weight.view(torch.float8_e4m3fn)

    if block_size is None:
        if weight_scale.numel() > 1 and weight_scale.shape != weight.shape:
            if weight_scale.numel() == weight.shape[0]:
                weight_scale = weight_scale.view(-1, 1)
            elif weight_scale.numel() == weight.shape[-1]:
                weight_scale = weight_scale.view(1, -1)
        return weight.to(dtype) * weight_scale.to(dtype)

    assert len(block_size) == 2

    weight, orig_M, orig_N = _pad_block_fp8_weight_naive(weight, weight_scale, block_size)

    weight_shape_len = len(weight.shape)
    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    dequant_weight = _unpad_weight(dequant_weight, orig_M, orig_N, keep_first_dim=keep_first_dim)

    return dequant_weight


# ============================================================================
# Section 4: HANDLER IMPLEMENTATIONS
# ============================================================================


# ----------------------------------------------------------------------------
# FP8 Block Handler - Full Implementation (Reference Example)
# ----------------------------------------------------------------------------
@register_weight_type_handler(ModuleWeightType.FP8)
class FP8Handler(WeightTypeHandler):
    """Handler for FP8 block-wise quantized layers.

    This handler supports:
        - FP8Linear layers with block-wise scaling
        - CompressedLinear layers with compressor
        - torch.nn.Linear layers with FP8 dtype weights
    """

    def detect_layer(self, module: torch.nn.Module) -> bool:
        """Check if a module is an FP8 linear layer based on actual characteristics."""
        # Check for CompressedLinear layer type (exclude MXFP4, MXFP8, NVFP4)
        if module.__class__.__name__ == "CompressedLinear":
            if hasattr(module, "compressor") and module.compressor is not None:
                compressor_name = module.compressor.__class__.__name__
                return "Float" in compressor_name
            return True

        # Check for FP8Linear layer type
        if module.__class__.__name__ == "FP8Linear":
            return True

        # Fallback: Check for FP8 dtype (for torch.nn.Linear with FP8 weights)
        if type(module) == torch.nn.Linear and module.weight is not None:
            if str(module.weight.dtype).startswith("torch.float8"):
                return True

        return False

    def convert_layer(
        self,
        layer: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert a single FP8/CompressedLinear layer to a standard Linear layer."""
        from auto_round.schemes import QuantizationScheme
        from auto_round.utils.device import is_gaudi2

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

        layer = layer.to(device)

        # Check if layer has compressor.decompress_module method
        if hasattr(layer, "compressor") and hasattr(layer.compressor, "decompress_module"):
            dq_weight = layer.compressor.decompress_module(layer)
        else:
            # Use FP8 block-based dequantization
            weight_scale = getattr(layer, "weight_scale", None)
            block_size = getattr(layer, "block_size", None)
            data_type = getattr(layer, "data_type", None)
            if weight_scale is None:
                weight_scale = getattr(layer, "weight_scale_inv", None)
            if weight_scale is None:
                raise AttributeError(
                    "FP8 layer is missing both 'weight_scale' and 'weight_scale_inv' "
                    "attributes required for dequantization."
                )
            dq_weight = _dequant_fp8_linear_weight(layer.weight, weight_scale, block_size, data_type=data_type)

        new_layer.weight.data.copy_(dq_weight.to(dtype=dtype))

        if to_cpu:
            new_layer = new_layer.to("cpu")

        return new_layer


# ----------------------------------------------------------------------------
# MXFP4 Handler - CompressedLinear with MXFP4PackedCompressor
# ----------------------------------------------------------------------------
@register_weight_type_handler(ModuleWeightType.MXFP4)
class MXFP4Handler(WeightTypeHandler):
    """Handler for MXFP4 quantized layers (CompressedLinear with MXFP4PackedCompressor)."""

    def detect_layer(self, module: torch.nn.Module) -> bool:
        """Check if a module is an MXFP4 CompressedLinear layer."""
        if module.__class__.__name__ != "CompressedLinear":
            return False
        if hasattr(module, "compressor") and module.compressor is not None:
            compressor_name = module.compressor.__class__.__name__
            return "MXFP4" in compressor_name
        return False

    def convert_layer(
        self,
        layer: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert an MXFP4 CompressedLinear layer to a standard Linear layer."""
        from auto_round.schemes import QuantizationScheme
        from auto_round.utils.device import is_gaudi2

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

        layer = layer.to(device)

        # MXFP4 dequantization using to_dtype from mxfp4_qdq_utils
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import to_dtype

        # Get packed weight and scale from layer
        # MXFP4 weights are stored as packed uint8 in weight_packed
        weight_packed = getattr(layer, "weight_packed", None)
        if weight_packed is None:
            weight_packed = getattr(layer, "weight", None)
        weight_scale = getattr(layer, "weight_scale", None)
        weight_scale = weight_scale.to(torch.uint8)

        if weight_packed is None or weight_scale is None:
            raise ValueError("MXFP4 layer must have weight_packed and weight_scale attributes")

        # Dequantize using to_dtype function
        dq_weight = to_dtype(
            data_lp=weight_packed,
            scale_e8m0=weight_scale,
            elem_dtype="fp4_e2m1",
            block_size=32,
            target_dtype=dtype,
        )
        new_layer.weight.data.copy_(dq_weight.to(dtype=dtype))

        if to_cpu:
            new_layer = new_layer.to("cpu")

        return new_layer


# ----------------------------------------------------------------------------
# MXFP8 Handler - CompressedLinear with MXFP8PackedCompressor
# ----------------------------------------------------------------------------
@register_weight_type_handler(ModuleWeightType.MXFP8)
class MXFP8Handler(WeightTypeHandler):
    """Handler for MXFP8 quantized layers (CompressedLinear with MXFP8PackedCompressor)."""

    def detect_layer(self, module: torch.nn.Module) -> bool:
        """Check if a module is an MXFP8 CompressedLinear layer."""
        if module.__class__.__name__ != "CompressedLinear":
            return False
        if hasattr(module, "compressor") and module.compressor is not None:
            compressor_name = module.compressor.__class__.__name__
            return "MXFP8" in compressor_name
        return False

    def convert_layer(
        self,
        layer: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert an MXFP8 CompressedLinear layer to a standard Linear layer."""
        from auto_round.schemes import QuantizationScheme
        from auto_round.utils.device import is_gaudi2

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

        layer = layer.to(device)

        # MXFP8 dequantization using dequant_mx_fp8 from mxfp8_qdq_utils
        from auto_round_extension.vllm_ext.mxfp8_qdq_utils import dequant_mx_fp8

        # Get weight and scale from layer
        weight = layer.weight
        weight_scale = getattr(layer, "weight_scale", None)

        if weight is None or weight_scale is None:
            raise ValueError("MXFP8 layer must have weight and weight_scale attributes")

        # Dequantize using dequant_mx_fp8 function
        dq_weight = dequant_mx_fp8(
            weight_fp8=weight,
            scale_e8m0=weight_scale,
            block_size=32,
            target_dtype=dtype,
        )
        new_layer.weight.data.copy_(dq_weight.to(dtype=dtype))

        if to_cpu:
            new_layer = new_layer.to("cpu")

        return new_layer


# ----------------------------------------------------------------------------
# NVFP4 Handler - CompressedLinear with NVFP4PackedCompressor
# ----------------------------------------------------------------------------
@register_weight_type_handler(ModuleWeightType.NVFP4)
class NVFP4Handler(WeightTypeHandler):
    """Handler for NVFP4 quantized layers (CompressedLinear with NVFP4PackedCompressor)."""

    def detect_layer(self, module: torch.nn.Module) -> bool:
        """Check if a module is an NVFP4 CompressedLinear layer."""
        if module.__class__.__name__ != "CompressedLinear":
            return False
        if hasattr(module, "compressor") and module.compressor is not None:
            compressor_name = module.compressor.__class__.__name__
            return "NVFP4" in compressor_name
        return False

    def convert_layer(
        self,
        layer: torch.nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        to_cpu: bool = False,
    ) -> torch.nn.Module:
        """Convert an NVFP4 CompressedLinear layer to a standard Linear layer."""
        from auto_round.schemes import QuantizationScheme
        from auto_round.utils.device import is_gaudi2

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

        layer = layer.to(device)

        # Use compressor.decompress_module for dequantization
        dq_weight = layer.compressor.decompress_module(layer)
        new_layer.weight.data.copy_(dq_weight.to(dtype=dtype))

        if to_cpu:
            new_layer = new_layer.to("cpu")

        return new_layer
