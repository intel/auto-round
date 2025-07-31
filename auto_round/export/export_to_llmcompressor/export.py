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
from auto_round.utils import detect_device, get_module, logger, set_module
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
def save_quantized_as_llmcompressor(output_dir, model=None, **kwargs):
    safe_serialization = kwargs.get("safe_serialization", True)
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    if output_dir is not None and os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")

    # save tokenizer, processor
    if output_dir is not None and tokenizer is not None:
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
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
