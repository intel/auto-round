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
from typing import Callable, Union

import torch
from compressed_tensors.compressors import IntQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from compressed_tensors.utils import delete_offload_parameter, get_offloaded_device, register_offload_parameter

from auto_round.compressors.utils import is_static_wfp8afp8
from auto_round.export.utils import save_model
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    copy_python_files_from_model_cache,
    detect_device,
    get_module,
    set_module,
    unsupported_meta_device,
)
from auto_round.wrapper import WrapperWALayer


def construct_ct_scheme(layer):
    weights_args = QuantizationArgs(
        num_bits=layer.bits,
        type=layer.data_type.split("_")[-2],  # int_sym, rtn_int_sym
        symmetric=layer.sym,
        dynamic=False,
        group_size=layer.group_size if layer.group_size != 0 else None,
        strategy=None if layer.group_size != 0 else "tensor",
    )
    activations_args = QuantizationArgs(
        num_bits=layer.act_bits,
        type=layer.act_data_type.split("_")[-2],  # int_sym, rtn_int_sym
        symmetric=layer.act_sym,
        dynamic=layer.act_dynamic,
        group_size=layer.act_group_size if layer.group_size != 0 else None,
        strategy=None if layer.act_group_size != 0 else "tensor",
    )
    scheme = QuantizationScheme(
        targets=[layer.__class__.__name__],
        weights=weights_args,
        input_activations=activations_args,
    )
    return scheme


def pack_layer(name, model, device=None):
    layer = get_module(model, name)
    if type(layer) not in SUPPORTED_LAYER_TYPES and not isinstance(layer, WrapperWALayer):  ##already packed
        return

    if not check_to_quantized(layer):
        return

    if hasattr(layer, "quantization_status") and layer.quantization_status == QuantizationStatus.COMPRESSED:
        return

    scheme = construct_ct_scheme(layer)
    setattr(layer, "quantization_scheme", scheme)
    setattr(layer, "weight_scale", torch.nn.Parameter(layer.scale))
    if not isinstance(layer.zp, torch.Tensor):
        if layer.sym:
            zp = torch.full_like(layer.weight_scale, 0).to(torch.int8)
        else:
            zp = torch.full_like(layer.weight_scale, layer.zp).to(torch.int8)
    else:
        zp = layer.zp
    setattr(layer, "weight_zero_point", torch.nn.Parameter(zp, requires_grad=False))
    delattr(layer, "scale")

    compressor = IntQuantizationCompressor()
    q_state_dict = compressor.compress(layer.state_dict(), names_to_scheme={"": scheme}, show_progress=False)

    # remove any existing parameters
    offload_device = get_offloaded_device(layer)
    for name, _ in list(layer.named_parameters(recurse=False)):
        delete_offload_parameter(layer, name)

    # replace with compressed parameters
    for name, value in q_state_dict.items():
        param = torch.nn.Parameter(value, requires_grad=False)
        register_offload_parameter(layer, name, param, offload_device)

    layer.quantization_status = QuantizationStatus.COMPRESSED


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

    safe_serialization = kwargs.get("safe_serialization", True)
    processor = kwargs.get("processor", None)
    if output_dir is not None and os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    # save tokenizer, processor
    if output_dir is not None and tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)
    if output_dir is not None and processor is not None:
        processor.save_pretrained(output_dir)

    # generate q_weight
    device = detect_device(device)
    if not unsupported_meta_device(model):
        for n, m in model.named_modules():
            pack_layer(n, model, device)

    if output_dir is None:
        return model

    quantization_config = QuantizationConfig.from_pretrained(model)
    # save model.config, model.state_dict()
    model.config.quantization_config = quantization_config.to_dict()
    model.config.save_pretrained(output_dir)

    try:
        save_model(model, output_dir, safe_serialization=safe_serialization)
    except ValueError as e:
        if hasattr(model, "generation_config"):
            setattr(model.generation_config, "do_sample", True)
        save_model(model, output_dir, safe_serialization=safe_serialization)

    try:
        copy_python_files_from_model_cache(model, output_dir)
    except Exception as e:
        logger.warning("Skipping source model Python file copy due to error: %s", e)

    return model
