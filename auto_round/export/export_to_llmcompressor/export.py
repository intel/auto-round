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

from auto_round.compressors.utils import is_mx_fp, is_nv_fp, is_standard_fp, is_static_wfp8afp8
from auto_round.export.export_to_llmcompressor.config import quantization_config
from auto_round.export.export_to_llmcompressor.export_to_fp import save_quantized_as_fp
from auto_round.export.export_to_llmcompressor.export_to_static_fp import save_quantized_as_static_fp
from auto_round.logger import logger
from auto_round.utils import (
    copy_python_files_from_model_cache,
    detect_device,
    get_module,
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

    return model
