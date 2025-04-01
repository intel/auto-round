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

from logging import getLogger
from typing import Union

import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D

from auto_round.utils import (get_module, set_module, is_hpu_supported, get_block_names,
                              find_matching_blocks, get_layer_names_in_block, check_to_quantized)

from auto_round.inference.backend import get_layer_backend, dynamic_import_inference_linear, find_backend, BackendInfos

logger = getLogger(__name__)

supported_devices = ("cpu", "hpu", "xpu", "cuda")


def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if is_hpu_supported():
        devices.append("hpu")
    if torch.xpu.is_available():
        devices.append("xpu")
    devices.append("cpu")
    return devices


def parse_target_device_and_backend(target_backend: str):
    """
    Parse the target device and backend from the given target_backend string.

    Args:
        target_backend (str): A string specifying the target backend, which may include a device prefix.

    Returns:
        tuple: (target_device, backend), where:
            - target_device (str): The identified device (e.g., "cpu", "cuda", "auto").
            - backend (str): The backend string after removing the device prefix.
    """

    def remove_device_prefix(backend: str, prefix: str) -> str:
        """Removes the given device prefix from the backend string, if present."""
        return backend[len(prefix):].lstrip(":") if backend.startswith(prefix) else backend

    # Handle "auto" case explicitly
    if target_backend == "auto":
        return "auto", ""
    if target_backend.startswith("auto:"):
        return "auto", target_backend[5:].lstrip(":")

    # Define supported devices

    # Identify the target device and adjust the backend string accordingly
    for device in supported_devices:
        if target_backend.startswith(device):
            return device, remove_device_prefix(target_backend, device)

    # Default case: If no known device is found, assume "auto"
    return "auto", target_backend


def get_layer_config(model, quantization_config):
    """
    get a layer-wise quantization configuration for a given model.

    Args:
        model (torch.nn.Module): The model for which quantization settings are generated.
        quantization_config (object): An object containing quantization parameters, including:
            - bits (int): Default bit width for quantization.
            - group_size (int): Group size for weight quantization.
            - data_type (str, optional): Data type for quantization (default: "int").
            - sym (bool): Whether to use symmetric quantization.
            - quant_block_list (list, optional): Predefined list of blocks to quantize.
            - to_quant_block_names (list or str, optional): Blocks to quantize (if quant_block_list is None).
            - extra_config (dict, optional): Per-layer overrides for quantization settings.
            - modules_in_block_to_quantize (list, optional): Specific modules within a block for quantization.
            - modules_to_not_convert (list, optional): Layers excluded from quantization (AWQ format).

    Returns:
        dict: A dictionary mapping layer names to their quantization configurations, where each layer has:
            - "bits" (int): Bit width for quantization.
            - "group_size" (int): Group size for quantization.
            - "data_type" (str): Data type used for quantization.
            - "sym" (bool): Whether symmetric quantization is applied.
            - "clip" (bool): Whether weight clipping is enabled.
    """
    bits = quantization_config.bits
    group_size = quantization_config.group_size
    data_type = getattr(quantization_config, "data_type", "int")  # Default to "int" if not specified
    sym = quantization_config.sym

    # Determine the quantization block list
    quant_block_list = getattr(quantization_config, "quant_block_list", None)
    if quant_block_list is None:
        to_quant_block_names = getattr(quantization_config, "to_quant_block_names", None)
        if isinstance(to_quant_block_names, (list, tuple)):
            quant_block_list = to_quant_block_names
        elif isinstance(to_quant_block_names, str):
            # Generate quant block names based on the given layer names
            quant_block_list = [
                [f'{block}.{i}' for i in range(len(get_module(model, block)))]
                for block in to_quant_block_names.split(',')
            ]
        else:
            # Find matching blocks if no explicit names are provided
            all_blocks = get_block_names(model)
            quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)

    # Get layer names that will be quantized
    layer_names = get_layer_names_in_block(model, quant_block_list=quant_block_list)

    # Load extra configuration if available
    extra_config = getattr(quantization_config, "extra_config", {})

    # Process GPTQ format: identify modules that should be quantized
    if getattr(quantization_config, "modules_in_block_to_quantize", None):
        modules_in_block_to_quantize = sum(quantization_config.modules_in_block_to_quantize, [])  # Flatten the list
        for layer_name in layer_names:
            if not any(qname in layer_name for qname in modules_in_block_to_quantize):
                extra_config[layer_name] = {"bits": 16}  # Default to 16-bit for unquantized layers

    # Process AWQ format: exclude specified modules from quantization
    modules_to_not_convert = getattr(quantization_config, "modules_to_not_convert", [])
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    for layer_name in modules_to_not_convert:
        extra_config[layer_name] = {"bits": 16}

    # Merge and deduplicate layer names
    layer_names = list(set(layer_names).union(extra_config.keys()))

    # Construct final layer configuration
    layer_configs = {
        layer_name: {
            "bits": extra_config.get(layer_name, {}).get("bits", bits),
            "group_size": extra_config.get(layer_name, {}).get("group_size", group_size),
            "data_type": extra_config.get(layer_name, {}).get("data_type", data_type),
            "sym": extra_config.get(layer_name, {}).get("sym", sym),
            "clip": extra_config.get(layer_name, {}).get("clip", False),
        }
        for layer_name in layer_names
    }

    return layer_configs


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def _replace_by_quant_layers(module: nn.Module, layer_configs, target_backend, target_device, orig_backend,
                             available_devices):
    """
    Replaces linear layers in the given module with quantized layers.

    Args:
        module (nn.Module): The module containing layers to be quantized.
        layer_configs (dict): Configuration for each layer's quantization.
        target_backend (str): Backend for quantization (device and format).
        target_device (str): Device for execution ('cuda', 'cpu', 'hpu').
        orig_backend (str): Original backend of the model.
        available_devices (list): List of available devices.

    Returns:
        dict: Flags indicating which backends were used.
    """
    backend_flags = {"used_autogptq": False, "used_autoawq": False, "used_qbits": False, "used_ipex": False}
    must_use_target_backend = False
    if target_backend:
        must_use_target_backend = True

    target_backend = target_backend or orig_backend  # Default to original backend if not specified

    for layer_name, config in layer_configs.items():
        if not check_to_quantized(config):
            continue  # Skip layers that do not require quantization

        layer = get_module(module, layer_name)
        in_features, out_features = _get_layer_features(layer)
        if in_features is None:
            continue  # Skip unsupported layer types

        ##TODO cache bakend
        if must_use_target_backend:
            layer_backend = target_backend
            layer_backend = find_backend(layer_backend)
            devices = BackendInfos[layer_backend].device
            for device in devices:
                if device in available_devices or device in target_device:
                    target_device = device
                    break
        else:
            # Determine backend
            layer_backend = _get_layer_backend(target_device, available_devices, target_backend, orig_backend, config,
                                               in_features, out_features)
        if not layer_backend:
            raise ValueError(f"No compatible backend found for layer {layer_name} with config {config}")

        # Update backend usage flags
        _update_backend_flags(layer_backend, backend_flags)

        # Import ExLlamaV2 kernels if needed
        _import_exllamav2_kernels(layer_backend)

        # Create and replace layer
        new_layer = _create_quant_layer(layer, layer_backend, config, in_features, out_features)
        set_module(module, layer_name, new_layer)

    return backend_flags


def _get_layer_features(layer):
    """Extracts input and output feature dimensions for supported layers."""
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    elif isinstance(layer, Conv1D):  # TODO: Verify correctness
        return layer.weight.shape[0], layer.weight.shape[1]
    return None, None  # Unsupported layer type


def _get_layer_backend(target_device, available_devices, target_backend, orig_backend, config, in_features,
                       out_features):
    """Determines the best backend for a given layer."""
    devices = [target_device] + available_devices if target_device == "auto" else [target_device]
    for device in devices:
        backend = get_layer_backend(device, target_backend, orig_backend, config["bits"], config["group_size"],
                                    config["sym"], in_features, out_features)
        if backend:
            return backend
    return None


def _update_backend_flags(layer_backend, flags):
    """Updates backend flags based on detected backend type."""
    if "qbits" in layer_backend:
        flags["used_qbits"] = True
    elif "ipex" in layer_backend:
        flags["used_ipex"] = True
    elif "gptq" in layer_backend and "gptqmodel" not in layer_backend:
        flags["used_autogptq"] = True
    elif "awq" in layer_backend:
        flags["used_autoawq"] = True


def _import_exllamav2_kernels(layer_backend):
    """Attempts to import ExLlamaV2 kernels for performance optimization."""
    if "gptq" in layer_backend and "gptqmodel" not in layer_backend and "exllamav2" in layer_backend:
        try:
            from exllamav2_kernels import gemm_half_q_half, make_q_matrix  # pylint: disable=E0611, E0401
        except ImportError:
            logger.warning_once(
                "For better inference performance, install ExLlamaV2 kernel via: "
                "`pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@b8b4127`"
            )


def _create_quant_layer(layer, layer_backend, config, in_features, out_features):
    """Creates a quantized layer using the appropriate class."""
    QuantLinear = dynamic_import_inference_linear(layer_backend, config["bits"], config["group_size"], config["sym"])
    bias = layer.bias is not None

    # Special handling for AWQ layers
    from auto_round_extension.qbits.qbits_awq import QuantLinear as QBitsAWQQuantLinear
    if "awq" in layer_backend and isinstance(QuantLinear, QBitsAWQQuantLinear):
        return QuantLinear.from_linear(layer, config["bits"], config["group_size"], init_only=True,
                                       has_zero_points=not config["sym"])
    elif "awq" in layer_backend:
        return QuantLinear.from_linear(layer, config["bits"], config["group_size"], init_only=True)

    # Default quantized layer creation
    try:
        return QuantLinear(config["bits"], config["group_size"], in_features, out_features, bias,
                           weight_dtype=layer.weight.dtype, clip=config["clip"])
    except:  # Handle cases where `clip` is not a valid argument
        return QuantLinear(config["bits"], config["group_size"], in_features, out_features, bias,
                           weight_dtype=layer.weight.dtype)


def convert_hf_model(model: nn.Module):
    """Converts the given model to an AutoRound model by replacing its layers with quantized layers.

    This method extracts the quantization configuration from the model and adjusts its layers
    according to the specified quantization parameters. It supports different backends and
    ensures that the model's data type is compatible with the selected hardware.

    Args:
        model (nn.Module):
            The model to be converted into an AutoRound model.

    Returns:
        nn.Module:
            The converted AutoRound model with quantized layers.

    Raises:
        ValueError:
            If the quantization backend is not specified in the configuration.
    """

    quantization_config = model.config.quantization_config
    if not hasattr(quantization_config, "target_backend"):
        quantization_config.target_backend = quantization_config.backend

    ##target_backend could be None
    target_device, target_backend = parse_target_device_and_backend(quantization_config.target_backend)
    available_devices = get_available_devices()
    if target_device != "auto":
        if target_device not in available_devices:
            logger.error(f"Target device '{target_device}' is not supported. Available devices: {available_devices}")
        else:
            available_devices.pop(available_devices.index(target_device))
    else:
        if len(available_devices) == 1 and target_device == "auto":
            target_device = available_devices[0]

    if ("hpu" == target_device or "cpu" == target_device) and model.dtype != torch.bfloat16:
        logger.info(f"Change the dtype to `bfloat16` as {target_device.upper()} does not support float16")
        model = model.to(torch.bfloat16)

    if hasattr(quantization_config, "backend"):  # pragma: no cover
        backend = quantization_config.backend
    elif 'gptq' in quantization_config.quant_method:  # pragma: no cover
        backend = "gptq"
    elif "awq" in quantization_config.quant_method:
        backend = "awq"
    else:  # pragma: no cover
        backend = "gptq"
        logger.warning("Quantization backend must be specified. Set it to 'gptq' by default.")

    layer_configs = get_layer_config(model, quantization_config)

    backend = find_backend(backend)

    used_backend_info = _replace_by_quant_layers(model, layer_configs, target_backend, target_device, backend,
                                                 available_devices)
    return model, used_backend_info
