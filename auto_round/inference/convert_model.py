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

import re
from logging import getLogger
from typing import Union
from tqdm import tqdm
import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D

from auto_round.utils import (
    get_module, set_module, is_hpu_supported, get_block_names, find_matching_blocks,
    get_layer_names_in_block, check_to_quantized)

from auto_round.inference.backend import (
    get_layer_backend, dynamic_import_inference_linear, find_backend, BackendInfos, get_highest_priority_backend,
    process_requirement)

logger = getLogger(__name__)

supported_devices = ("cpu", "hpu", "xpu", "cuda")


def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def skip_not_convert_modules(model, quantization_config, layer_names, layer_configs):
    modules_to_not_convert = getattr(quantization_config, "modules_to_not_convert", [])
    try:  # transformers new api
        modules_to_not_convert = get_modules_to_not_convert(model, modules_to_not_convert, add_default_skips=True)
    except Exception:
        modules_to_not_convert = get_modules_to_not_convert(model, modules_to_not_convert)
    if modules_to_not_convert:
        for layer_name in layer_names:
            if any([re.search(re.compile(n), layer_name) for n in modules_to_not_convert]):
                layer_configs[layer_name] = {"bits": 16}
    return layer_configs


def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    from copy import deepcopy
    from accelerate.utils import find_tied_parameters
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # If there is not tied weights, we want to keep the lm_head（output_embedding) in full precision
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module

    # otherwise, no tied weights, no output embedding defined, simply keep the last module in full precision
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names


def _get_modules_to_not_convert(
        model,
        skip_modules=None,
        keep_in_fp32_modules=None,
        add_default_skips: bool = False,
):
    if skip_modules is None or add_default_skips:
        modules_to_not_convert = get_keys_to_not_convert(model)
    else:
        modules_to_not_convert = []

    if skip_modules is not None:
        modules_to_not_convert.extend(skip_modules)

    if keep_in_fp32_modules is not None:
        modules_to_not_convert.extend(keep_in_fp32_modules)

    return modules_to_not_convert


try:
    from transformers.quantizers.base import HfQuantizer

    get_modules_to_not_convert = HfQuantizer.get_modules_to_not_convert
except:
    get_modules_to_not_convert = _get_modules_to_not_convert


def get_available_devices():
    """
    Returns a list of available devices in the current environment.

    Returns:
        List[str]: A list of device identifiers like "cuda", "hpu", "xpu", "cpu".
    """
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    if is_hpu_supported():
        devices.append("hpu")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")

    devices.append("cpu")  # Always available

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
            all_blocks = get_block_names(model, quant_vision=True)
            quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)

    # Get layer names that will be quantized
    layer_names = get_layer_names_in_block(model, quant_block_list=quant_block_list)

    # Load extra configuration if available
    extra_config = getattr(quantization_config, "extra_config", {})

    # Process GPTQ format: identify modules that should be quantized
    if getattr(quantization_config, "modules_in_block_to_quantize", None):
        modules_in_block_to_quantize = flatten_list(
            quantization_config.modules_in_block_to_quantize)  # Flatten the list
        for layer_name in layer_names:
            if not any([re.search(re.compile(n), layer_name) is not None for n in modules_in_block_to_quantize]):
                extra_config[layer_name] = {"bits": 16}  # Default to 16-bit for unquantized layers

    # Process AWQ format: exclude specified modules from quantization
    extra_config = skip_not_convert_modules(model, quantization_config, layer_names, extra_config)

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


def _replace_by_quant_layers(module: nn.Module, layer_configs, target_backend, target_device, orig_backend):
    """
    Replaces linear layers in the given module with quantized layers.

    Args:
        module (nn.Module): The module containing layers to be quantized.
        layer_configs (dict): Configuration for each layer's quantization.
        target_backend (str): Backend for quantization (device and format).
        target_device (str): Device for execution ('cuda', 'cpu', 'hpu').
        orig_backend (str): Original backend of the model.

    Returns:
        dict: Flags indicating which backends were used.
    """

    used_backends = []
    must_use_target_backend = False
    if target_backend:
        must_use_target_backend = True
        layer_backend = target_backend
        layer_backend_must = find_backend(layer_backend, orig_backend)
        used_backends.append(layer_backend_must)
        if layer_backend_must is None:
            raise ValueError(
                f"{target_backend} is not compatible, please change backend to `auto` and retry")
        devices = BackendInfos[layer_backend_must].device
        if target_device not in devices:
            raise ValueError(f"{target_backend} does not support {target_device}, please change device or backend")
        ##check requirements
        requirements = BackendInfos[layer_backend_must].requirements
        requirements_info = process_requirement(requirements)
        if len(requirements_info) > 0:
            extra_info = ""
            for index, req in enumerate(requirements_info):
                extra_info += (f"`{req}`")
                if index != len(requirements_info) - 1:
                    extra_info += " and "
            raise ImportError(f"{target_backend} requires the follow libraries. Please install them via {extra_info}")

    target_backend = target_backend or orig_backend  # Default to original backend if not specified

    backend_cache = {}

    for layer_name, config in layer_configs.items():
        if not check_to_quantized(config):
            continue  # Skip layers that do not require quantization

        layer = get_module(module, layer_name)
        in_features, out_features = _get_layer_features(layer)
        if in_features is None:
            continue  # Skip unsupported layer types

        if must_use_target_backend:
            layer_backend = layer_backend_must
        else:
            key = f"{config['bits']}_{config['group_size']}_{config['sym']}_{in_features}_{out_features}"
            if key in backend_cache:
                layer_backend = backend_cache[key]
                if layer_backend not in used_backends:
                    used_backends.append(layer_backend)
            else:
                # Determine backend
                layer_backend = _get_layer_backend(target_device, target_backend, orig_backend, config,
                                                   in_features, out_features)
                backend_cache[key] = layer_backend

        if not layer_backend:
            raise ValueError(f"No compatible backend found for layer {layer_name} with config {config}")

        logger.debug(f"{layer_name}: {layer_backend} backend is used")  ##TODO delete

        # Create and replace layer
        new_layer = _create_quant_layer(layer, layer_backend, config, in_features, out_features)
        set_module(module, layer_name, new_layer)

    return used_backends


def _get_layer_features(layer):
    """Extracts input and output feature dimensions for supported layers."""
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    elif isinstance(layer, Conv1D):  # TODO: Verify correctness
        return layer.weight.shape[0], layer.weight.shape[1]
    return None, None  # Unsupported layer type


def _get_layer_backend(target_device, target_backend, orig_backend, config, in_features,
                       out_features):
    """Determines the best backend for a given layer."""

    backend = get_layer_backend(target_device, target_backend, orig_backend, config["bits"], config["group_size"],
                                config["sym"], in_features, out_features)

    return backend


def _import_exllamav2_kernels():
    """Attempts to import ExLlamaV2 kernels for performance optimization."""
    try:
        from exllamav2_kernels import gemm_half_q_half, make_q_matrix  # pylint: disable=E0611, E0401
    except ImportError:
        raise ImportError(
            "AutoGPTQ ExLlamaV2 has not been installed, Please install it using the following command: "
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
    elif "gptqmodel" in layer_backend:
        return QuantLinear(bits=config["bits"], group_size=config["group_size"], desc_act=False, sym=config["sym"],
                           in_features=in_features, out_features=out_features, bias=bias)
    # Default quantized layer creation
    try:
        return QuantLinear(config["bits"], config["group_size"], in_features, out_features, bias,
                           weight_dtype=layer.weight.dtype, clip=config["clip"])
    except:  # Handle cases where `clip` is not a valid argument
        return QuantLinear(config["bits"], config["group_size"], in_features, out_features, bias,
                           weight_dtype=layer.weight.dtype)


def infer_target_device(device_map=None):
    if device_map is None:
        target_device = "cpu"
    elif isinstance(device_map, dict):
        devices = set(device_map.values())
        target_device = "cpu"
        for device in devices:
            if device != "cpu" and device != "disk":
                if isinstance(device, int):
                    target_device = get_available_devices()[0]
                else:
                    target_device = str(device).split(":")[0]
    else:
        target_device = get_available_devices()[0]
    assert isinstance(target_device, str)
    return target_device


def post_init(model, used_backends):
    need_autogptq_init = False
    need_gptqmodel_init = False
    need_ipex_itrex_init = False
    used_gptq_exllamav2 = False
    for used_backend in used_backends:
        if used_backend.startswith("auto_gptq"):
            need_autogptq_init = True
            if used_backend == "auto_gptq:exllamav2":
                used_gptq_exllamav2 = True
        elif used_backend.startswith("gptqmodel"):
            need_gptqmodel_init = True
        elif used_backend.startswith("ipex"):
            need_ipex_itrex_init = True
        elif used_backend.startswith("qbit"):
            need_ipex_itrex_init = True
    if need_autogptq_init:
        from auto_gptq.modeling._utils import autogptq_post_init as gptq_post_init  # pylint: disable=E0401
        model = gptq_post_init(model, use_act_order=False)
    if need_gptqmodel_init:
        from gptqmodel.utils.model import hf_gptqmodel_post_init as gptq_post_init  # pylint: disable=E0401
        model = gptq_post_init(model, use_act_order=False)
    if need_ipex_itrex_init:
        message = "repacking to CPU/XPU format"
        layers = []  ## ipex post_init  will add one more layer
        for n, m in model.named_modules():
            if hasattr(m, "QUANT_TYPE") and ("qbits" in m.QUANT_TYPE or "ipex" in m.QUANT_TYPE):
                layers.append(m)

        for layer in tqdm(layers, desc=message, total=len(layers),
                          leave=True):
            layer.post_init()

    if used_gptq_exllamav2:
        _import_exllamav2_kernels()

    ## convert datatype
    data_types = []
    for used_backend in used_backends:
        backend = BackendInfos[used_backend]
        data_types.append(backend.dtype)
    common = set(data_types[0])
    for l in data_types[1:]:
        common &= set(l)
    common = list(common)
    if len(common) > 0 and str(model.dtype).split('.')[-1] not in common:
        if common[0] == "float16":
            model = model.to(torch.float16)
            logger.warning("force model to float16")
        elif common[0] == "bfloat16":
            model = model.to(torch.bfloat16)
            logger.warning("force model to bfloat16")


def convert_hf_model(model: nn.Module, target_device="cpu"):
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

    if hasattr(quantization_config, "desc_act") and quantization_config.desc_act == True:
        ##check static_group
        if (hasattr(quantization_config, "static_groups") and not quantization_config.static_groups) or (
                not hasattr(quantization_config, "static_groups")):
            raise NotImplementedError(
                "This GPTQ model may contain a non-dummy g_idx, which is not yet supported by AutoRound")

    if hasattr(quantization_config, "backend"):
        backend = quantization_config.backend
    else:
        backend = "auto"


    ##target_backend could be None
    _, backend = parse_target_device_and_backend(backend)

    if hasattr(quantization_config, "packing_format"):  # pragma: no cover
        packing_format = quantization_config.packing_format
    elif 'gptq' in quantization_config.quant_method:  # pragma: no cover
        packing_format = "auto_gptq"
    elif "awq" in quantization_config.quant_method:
        packing_format = "auto_awq"
    else:  # pragma: no cover
        packing_format = "auto_gptq"
        logger.warning("Quantization backend must be specified. Set it to 'auto_gptq' by default.")
    if packing_format == "auto":
        packing_format = "auto_gptq"

    layer_configs = get_layer_config(model, quantization_config)
    if packing_format.startswith("auto_round:") and ("gptq" in packing_format or "awq" in packing_format):
        packing_format = packing_format[len("auto_round:"):]
    orig_backend = find_backend(packing_format)

    if backend.startswith("auto_round:") and ("gptq" in packing_format or "awq" in packing_format):
        backend = backend[len("auto_round:"):]

    used_backends = _replace_by_quant_layers(model, layer_configs, backend, target_device, orig_backend)

    if backend == "auto" or backend == "":
        best_backend = get_highest_priority_backend(quantization_config.bits, quantization_config.sym,
                                                    quantization_config.group_size, target_device,
                                                    BackendInfos[orig_backend].packing_format)
        if best_backend is not None and best_backend not in used_backends:
            info = f"better backend is found, please install all the following requirements to enable it,"
            extra_info = info
            requirements = BackendInfos[best_backend].requirements
            requirements_info = process_requirement(requirements)
            for index, req in enumerate(requirements_info):
                extra_info += (f"`{req}`")
                if index != len(requirements_info) - 1:
                    extra_info += " and "

            logger.warning(extra_info)

    return model, used_backends
