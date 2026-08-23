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

import copy
import os
from collections.abc import Mapping
from typing import Callable, Union

import torch

from auto_round.export.utils import is_immediate_saving_mode, save_model, save_pretrained_artifact
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_major_device,
    get_module,
    set_module,
    unsupported_meta_device,
)
from auto_round.wrapper import WrapperWALayer


def _get_weight_scheme_strategy(group_size):
    if group_size == 0:
        return "tensor"
    if group_size == -1:
        return "channel"
    if isinstance(group_size, tuple):
        return "block"
    if isinstance(group_size, int):
        return "group"
    return None


def _get_act_scheme_strategy(group_size):
    if group_size == 0:
        return "tensor"
    if group_size == -1:
        return "token"
    if isinstance(group_size, int):
        return "group"
    return None


def _get_scheme_type(data_type):
    if "int" in data_type:
        return "int"
    if "fp" in data_type or "float" in data_type:
        return "float"
    raise NotImplementedError("only support `int` and `fp` data type")


def construct_ct_scheme(layer):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme  # pylint: disable=E0401

    weights_args = QuantizationArgs(
        num_bits=layer.bits,
        type=_get_scheme_type(layer.data_type),
        symmetric=layer.sym,
        dynamic=False,
        group_size=layer.group_size if _get_weight_scheme_strategy(layer.group_size) == "group" else None,
        strategy=_get_weight_scheme_strategy(layer.group_size),
        block_structure=layer.group_size if _get_weight_scheme_strategy(layer.group_size) == "block" else None,
    )
    # Weight-only quantization (W4A16, W8A16, etc.): no activation quantization
    if layer.act_bits >= 16 or layer.act_data_type is None:
        activations_args = None
    else:
        activations_args = QuantizationArgs(
            num_bits=layer.act_bits,
            type=_get_scheme_type(layer.act_data_type),
            symmetric=layer.act_sym,
            dynamic=layer.act_dynamic,
            group_size=layer.act_group_size if _get_act_scheme_strategy(layer.act_group_size) == "group" else None,
            strategy=_get_act_scheme_strategy(layer.act_group_size),
        )
    scheme = QuantizationScheme(
        targets=[layer.__class__.__name__],
        weights=weights_args,
        input_activations=activations_args,
    )
    return scheme


def _get_quant_format(model):
    for n, m in model.named_modules():
        if hasattr(m, "quantization_scheme") and hasattr(m.quantization_scheme, "format"):
            return m.quantization_scheme.format
    return None


def _canonicalize_quantization_value(value):
    """Convert a serialized quantization value into a deterministic hashable form."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonicalize_quantization_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonicalize_quantization_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_canonicalize_quantization_value(item) for item in value), key=repr))

    value = getattr(value, "value", value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _quant_args_signature(args):
    """Return a hashable signature of every serialized QuantizationArgs field.

    Two layers whose weight (and activation) args share this signature end up in
    the same config_group, so we use it to map real module names back onto the
    group that `QuantizationConfig.from_pretrained` built for them.
    """
    if args is None:
        return None

    if not isinstance(args, Mapping):
        args = args.model_dump(mode="json")
    return _canonicalize_quantization_value(args)


def _rewrite_config_group_targets(model, quantization_config_dict):
    """Fill each config_group's `targets` with the real module names it covers.

    `construct_ct_scheme` writes `targets=[layer.__class__.__name__]` (always
    "Linear") for every layer, so a mixed-precision model ends up with multiple
    config_groups that all say `targets=["Linear"]`. Inference engines (vLLM)
    then cannot tell which layer is 4-bit vs 8-bit and miss-unpack the weights,
    producing garbage output. Here we group the actually-quantized modules by
    their scheme signature and rewrite the matching group's `targets` to the
    concrete module names.

    Single-group (uniform) models are left untouched: their lone `["Linear"]`
    target is unambiguous and already loads correctly.
    """
    groups = quantization_config_dict.get("config_groups", {})
    if len(groups) <= 1:
        return quantization_config_dict

    from collections import defaultdict

    sig_to_names = defaultdict(list)
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None:
            continue
        sig = (
            _quant_args_signature(getattr(scheme, "weights", None)),
            _quant_args_signature(getattr(scheme, "input_activations", None)),
        )
        sig_to_names[sig].append(name)

    for group in groups.values():
        sig = (
            _quant_args_signature(group.get("weights")),
            _quant_args_signature(group.get("input_activations")),
        )
        names = sig_to_names.get(sig)
        if names:
            group["targets"] = sorted(names)
        else:
            logger.warning(
                "AutoRound llm_compressor export: no quantized module matched "
                "config_group %s; leaving its targets unchanged.",
                group.get("targets"),
            )

    return quantization_config_dict


def _compress_and_set_format(layer, scheme, device=None):
    """Compress a layer and set its quantization format.

    Compatible with multiple compressed_tensors versions.
    """
    try:
        # Newer compressed_tensors export path
        from compressed_tensors.compressors import compress_module as _compress_module  # pylint: disable=E0401
    except ImportError:
        try:
            # Older versions expose this from module path only
            from compressed_tensors.compressors.base import compress_module as _compress_module  # pylint: disable=E0401
        except ImportError as e:
            logger.error(
                "Unable to import compress_module from compressed_tensors "
                "(tried compressed_tensors.compressors and "
                "compressed_tensors.compressors.base). "
                "Please install/upgrade compressed-tensors."
            )
            raise ImportError(
                "compress_module not found in compressed_tensors. " "Install a compatible version."
            ) from e
    _compress_module(layer)


def pack_layer(name, model, device=None):
    from compressed_tensors.quantization import QuantizationStatus  # pylint: disable=E0401

    layer = get_module(model, name)
    if type(layer) not in SUPPORTED_LAYER_TYPES and not isinstance(layer, WrapperWALayer):  ##already packed
        return

    if hasattr(layer, "orig_layer"):  # revert WrapperWALayer for offline usage
        wp_layer = layer
        layer = wp_layer.orig_layer
        set_module(model, name, layer)

    if not check_to_quantized(layer):
        return

    if hasattr(layer, "quantization_status") and layer.quantization_status == QuantizationStatus.COMPRESSED:
        return

    # explicitly obtain the underlying device to prevent RuntimeError mismatched tensors
    weight_device = layer.weight.device

    scheme = construct_ct_scheme(layer)
    setattr(layer, "quantization_scheme", scheme)
    setattr(layer, "weight_scale", torch.nn.Parameter(layer.scale.to(weight_device)))
    if not isinstance(layer.zp, torch.Tensor):
        if layer.sym:
            zp = torch.full_like(layer.weight_scale, 0).to(torch.int8)
        else:
            zp = torch.full_like(layer.weight_scale, layer.zp).to(torch.int8)
    else:
        zp = layer.zp

    setattr(layer, "weight_zero_point", torch.nn.Parameter(zp.to(weight_device), requires_grad=False))
    delattr(layer, "scale")

    _compress_and_set_format(layer, scheme, device)


@torch.no_grad()
def save_quantized_as_llmcompressor(
    output_dir: str,
    model: torch.nn.Module = None,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace: bool = True,
    device: Union[str, torch.device] = "cpu",
    serialization_dict: dict = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Save a quantized model in the LLM-Compressor format.

    This function saves a quantized model, including its configuration, state dictionary,
    tokenizer, and processor, in the specified output directory. It supports inplace
    modification of the model or creating a deepcopy for saving. Currently, only NVFP
    and MXFP backends are supported for specific quantization formats.

    Args:
        output_dir (str): The directory where the quantized model will be saved.
        inplace (bool, optional): If True, modifies the model in place. Otherwise, creates a deepcopy of the model.
                                Default is True.
        **kwargs: Additional arguments, including:
            - model (torch.nn.Module): The model to be quantized and saved.
            - backend (str): The backend framework used for quantization.
            - tokenizer: The tokenizer associated with the model.
            - processor: The processor associated with the model.
            - safe_serialization (bool): Whether to use safe serialization when saving
                                         the model. Default is True.

    Returns:
        torch.nn.Module: The quantized model that was saved.
    """
    from compressed_tensors.quantization import QuantizationConfig  # pylint: disable=E0401

    safe_serialization = kwargs.get("safe_serialization", True)
    processor = kwargs.get("processor", None)
    immediate_saving = is_immediate_saving_mode(model, serialization_dict)
    if output_dir is not None and os.path.exists(output_dir) and not immediate_saving:
        logger.warning(f"{output_dir} already exists, this may cause model conflict")
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    # save tokenizer, processor
    save_pretrained_artifact(tokenizer, output_dir, artifact_name="tokenizer")
    if output_dir is not None and processor is not None:
        processor.save_pretrained(output_dir)

    # generate q_weight
    device = get_major_device(device)
    if not unsupported_meta_device(model):
        for n, m in model.named_modules():
            pack_layer(n, model, device)

    quant_format = _get_quant_format(model)
    quantization_config = QuantizationConfig.from_pretrained(model, format=quant_format)
    quantization_config_dict = quantization_config.to_dict()
    # from_pretrained groups layers by scheme correctly, but every group inherits
    # the placeholder targets=["Linear"] from construct_ct_scheme. For mixed
    # precision that makes groups indistinguishable to vLLM; rewrite targets with
    # the concrete module names each group covers.
    quantization_config_dict = _rewrite_config_group_targets(model, quantization_config_dict)
    model.config.quantization_config = quantization_config_dict

    if output_dir is None:
        return model

    # save model.config, model.state_dict()
    model.config.save_pretrained(output_dir)

    save_model(model, output_dir, safe_serialization=safe_serialization, immediate_saving=immediate_saving)

    try:
        copy_python_files_from_model_cache(model, output_dir)
    except Exception as e:
        logger.warning("Skipping source model Python file copy due to error: %s", e)

    return model
