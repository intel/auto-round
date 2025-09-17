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

import os

import torch

from auto_round.export.export_to_llmcompressor.config import quantization_config
from auto_round.export.export_to_llmcompressor.export_to_fp import save_quantized_as_fp
from auto_round.export.export_to_llmcompressor.export_to_static_fp import save_quantized_as_static_fp
from auto_round.logger import logger
from auto_round.utils import (
    copy_python_files_from_model_cache,
    detect_device,
    get_module,
    is_mx_fp,
    is_nv_fp,
    is_standard_fp,
    is_static_wfp8afp8,
    set_module,
)
from auto_round.wrapper import WrapperWALayer


@torch.no_grad()
def recover_qweight(qdq_weight, scale):
    """
    Recover the quantized weight to its original floating-point representation.

    Args:
        qdq_weight (torch.Tensor): The quantized weight tensor.
        scale (float): The scale factor used for quantization.

    Returns:
        torch.Tensor: The recovered floating-point weight tensor.
    """
    return (qdq_weight / scale).to(torch.int8)


def pack_layer(layer_name, model, backend, device=None):
    """
    Packs a model layer for quantization based on its type and configuration.

    This function retrieves the specified layer from the model, checks its
    compatibility for quantization, and replaces it with a quantized version
    if applicable. The quantization process depends on the layer's bit-width,
    group size, symmetry, and activation bits.

    Args:
        layer_name (str): The name of the layer to be packed.
        model (torch.nn.Module): The model containing the layer.
        backend (str): The backend framework to be used for quantization.

    Returns:
        None: The function modifies the model in place.
    """
    if is_nv_fp(backend) or is_mx_fp(backend):
        from auto_round.export.export_to_llmcompressor.export_to_fp import pack_layer

        return pack_layer(layer_name, model, backend, device)

    if is_static_wfp8afp8(backend):
        from auto_round.export.export_to_llmcompressor.export_to_static_fp import pack_layer

        return pack_layer(layer_name, model, backend, device)

    ## passed as no other llm_compressor format is supported yet
    logger.warning("No other llm_compressor packing format(except NVFP&MXFP) is supported yet, skip packing")
    return


@torch.no_grad()
def save_quantized_as_llmcompressor(output_dir, **kwargs):
    backend = kwargs.get("backend", None)
    if is_nv_fp(backend) or is_mx_fp(backend):
        return save_quantized_as_fp(output_dir, **kwargs)

    if is_static_wfp8afp8(backend):
        return save_quantized_as_static_fp(output_dir, **kwargs)

    model = kwargs.get("model", None)
    safe_serialization = kwargs.get("safe_serialization", True)
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    if output_dir is not None and os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")

    # save tokenizer, processor
    if output_dir is not None and tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)
    if output_dir is not None and processor is not None:
        processor.save_pretrained(output_dir)

    # generate q_weight
    device = detect_device()
    for n, m in model.named_modules():
        if isinstance(m, WrapperWALayer):
            m = m.orig_layer
            q_weight = recover_qweight(m.weight.to(device), m.scale.to(device)).to("cpu")
            delattr(m, "weight")
            setattr(m, "weight", torch.nn.Buffer(q_weight))
            setattr(m, "weight_scale", torch.nn.Buffer(m.scale))

    # replace WrapperWALayer with orig_layer
    candidates = []
    for n, m in model.named_modules():
        if isinstance(m, WrapperWALayer):
            candidates.append(n)
    for n in candidates:
        set_module(model, n, get_module(model, n).orig_layer)

    # save model.config, model.state_dict()
    model.config.quantization_config = quantization_config
    model.config.save_pretrained(output_dir)
    try:
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    except ValueError as e:
        if hasattr(model, "generation_config"):
            setattr(model.generation_config, "do_sample", True)
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)

    try:
        copy_python_files_from_model_cache(model, output_dir)
    except Exception as e:
        logger.warning("Skipping source model Python file copy due to error: %s", e)
