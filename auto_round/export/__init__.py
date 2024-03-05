# Copyright (c) 2023 Intel Corporation
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
import torch
from ..utils import (
    logger,
    get_block_names,
    get_module,
    check_to_quantized,
)

EXPORT_FORMAT = {}


def register_format(name):
    """Class decorator to register a EXPORT subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the export type.

    Returns:
        cls: The class of register.
    """

    def register(format):
        EXPORT_FORMAT[name] = format
        return format

    return register


@register_format("auto_gptq")
def save_quantized_as_autogptq(output_dir, use_triton=False, inplace=True, **kwargs):
    """Export the model to autogptq format to easily leverage cuda kernel."""
    model = kwargs["model"]
    weight_config = kwargs["weight_config"]
    sym = kwargs["scheme"] == "sym"
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    iters = kwargs["iters"]
    lr = kwargs["lr"]
    minmax_lr = kwargs["minmax_lr"]
    enable_minmax_tuning = kwargs["enable_minmax_tuning"]
    use_quant_input = kwargs["use_quant_input"]
    scale_dtype = kwargs["scale_dtype"]
    tokenizer = kwargs["tokenizer"]
    supported_types = kwargs["supported_types"]

    logger.info("Saving quantized model to autogptq format, this may take a while...")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    ##check module quantized in block, this may have bug for mixed precision quantization
    block_name = get_block_names(model)[0]
    first_block = get_module(model, block_name)
    all_to_quantized = True
    modules_in_block_to_quantize = []
    for n, m in first_block.named_modules():
        is_supported_type = False
        for supported_type in supported_types:
            if isinstance(m, supported_type):
                is_supported_type = True
                break
        if not is_supported_type:
            continue
        if not check_to_quantized(m):
            all_to_quantized = False
        else:
            modules_in_block_to_quantize.append(n)
    modules_in_block_to_quantize = [modules_in_block_to_quantize]  ##align with autogptq
    if all_to_quantized:
        modules_in_block_to_quantize = None

    if inplace:
        compressed_model = model.to("cpu")
    else:
        compressed_model = copy.deepcopy(model.to("cpu"))

    from auto_gptq.modeling._utils import pack_model

    if bits == 3 or use_triton is False:
        if bits == 3 and use_triton is True:
            logger.warning("triton does not support 3 bits, reset it to False")
        quantizers = {}
        for key in weight_config:
            info = weight_config[key]
            if not check_to_quantized(info):
                continue
            quantizers[key] = (None, info["scale"], info["zp"], info["g_idx"])
        pack_model(
            compressed_model,
            quantizers,
            bits,
            group_size,
            use_cuda_fp16=True,
            desc_act=False,
            force_layer_back_to_cpu=True,
            use_triton=False,
        )
    else:
        quantizers = {}
        for key in weight_config:
            info = weight_config[key]
            if not check_to_quantized(info):
                continue
            info["zp"] = info["zp"].to(torch.float32)
            quantizers[key] = (None, info["scale"].to(torch.float32), info["zp"], info["g_idx"])
        pack_model(
            compressed_model,
            quantizers,
            bits,
            group_size,
            use_cuda_fp16=True,
            desc_act=False,
            force_layer_back_to_cpu=True,
            use_triton=True,
        )
    from auto_round import save_quantized_to_autogptq

    save_quantized_to_autogptq(
        compressed_model,
        output_dir,
        bits=bits,
        group_size=group_size,
        sym=sym,
        iters=iters,
        lr=lr,
        minmax_lr=minmax_lr,
        enable_minmax_tuning=enable_minmax_tuning,
        use_quant_input=use_quant_input,
        scale_dtype=scale_dtype,
        use_safetensors=True,
        modules_in_block_to_quantize=modules_in_block_to_quantize,
    )


@register_format("itrex")
def save_quantized_as_itrex(output_dir, inplace=True, **kwargs):
    """Save configure file and weights for CPU backend inference."""
    model = kwargs["model"]
    weight_config = kwargs["weight_config"]
    sym = kwargs["scheme"] == "sym"
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    iters = kwargs["iters"]
    lr = kwargs["lr"]
    minmax_lr = kwargs["minmax_lr"]
    enable_minmax_tuning = kwargs["enable_minmax_tuning"]
    use_quant_input = kwargs["use_quant_input"]
    scale_dtype = kwargs["scale_dtype"]
    tokenizer = kwargs["tokenizer"]

    from auto_round.export.export_to_itrex import compress_model, QuantConfig

    compressed_model = compress_model(model, weight_config, inplace=inplace)
    quantize_config = QuantConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        iters=iters,
        lr=lr,
        minmax_lr=minmax_lr,
        enable_minmax_tuning=enable_minmax_tuning,
        use_quant_input=use_quant_input,
        scale_dtype=str(scale_dtype),
    )
    if quantize_config is not None:
        config = compressed_model.config
        setattr(config, "quantization_config", quantize_config.to_dict())
        config.save_pretrained(output_dir)
        quantize_config.save_pretrained(output_dir)
    try:
        compressed_model.save_pretrained(output_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        logger.info("Saved config file and weights of quantized model to {}.".format(output_dir))
    except IOError as e:  # pragma: no cover
        logger.error("Fail to save configure file and weights due to {}.".format(e))
