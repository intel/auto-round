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
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import threadpoolctl as tctl
import torch
import transformers
from tqdm import tqdm

from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
from auto_round.export.export_to_autoround.export_to_fp8 import FP8QLinear
from auto_round.export.export_to_llmcompressor.config import check_compressed_tensors_supported
from auto_round.export.utils import save_model
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    filter_quantization_config,
    get_module,
    get_packing_device,
    logger,
    set_module,
)


def pack_layer(layer_name: str, model: torch.nn.Module, data_type: str, device: str = None) -> None:
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
    packing_device = get_packing_device(device)
    layer = get_module(model, layer_name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if type(layer) not in SUPPORTED_LAYER_TYPES:  ##already packed
        return

    if not check_to_quantized(layer):
        return

    orig_device = layer.weight.device
    scale = layer.scale.view(-1)
    zp = layer.zp
    weight = layer.weight
    weight, orig_shape, pad_len = reshape_pad_tensor_by_group_size(weight, layer.group_size)
    act_scale = layer.act_scale.view(-1) if hasattr(layer, "act_scale") else None
    torch_dtype = torch.float8_e4m3fn
    if "fp8_e5m2" in data_type:
        torch_dtype = torch.float8_e5m2
    info = torch.finfo(torch_dtype)
    if zp is not None:
        if isinstance(zp, torch.Tensor):
            zp = zp.to(packing_device)
        q_weight = weight.to(packing_device) / scale.to(packing_device).unsqueeze(-1) + zp
    else:
        q_weight = weight.to(packing_device) / scale.to(packing_device).unsqueeze(-1)
    q_weight = revert_tensor_by_pad(q_weight, orig_shape=orig_shape, pad_len=pad_len)
    q_weight = torch.clamp(q_weight, info.min, info.max)
    q_weight = q_weight.to(torch_dtype)
    if type(layer) == torch.nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    bias = layer.bias
    my_linear = FP8QLinear(
        in_features,
        out_features,
        weight=q_weight,
        weight_scale=scale,
        bias=bias,
        weight_zp=zp,
        input_scale=act_scale,
        dtype=model.dtype,
    )
    if len(my_linear.weight_scale.shape) and my_linear.weight_scale.shape[0] != 1:
        my_linear.weight_scale = my_linear.weight_scale.reshape(-1, 1)

    my_linear.to(orig_device)
    set_module(model, layer_name, my_linear)


def save_quantized_as_static_fp(output_dir: str, inplace: bool = True, **kwargs) -> torch.nn.Module:
    """
    Saves a quantized model of FP8_STATIC scheme in the llm-compressor format.

    Args:
        output_dir (str): The directory where the quantized model will be saved.
        inplace (bool, optional): If True, modifies the model in place. Otherwise, creates a deepcopy of the model.
                                Default is True.
        backend (str, optional): The backend to be used for quantization.
                                  Default is "autoround:exllamav2".
        **kwargs: Additional keyword arguments including:
            - model (nn.Module): The model to be quantized.
            - layer_config (dict): The layer configuration for each layer.
            - serialization_dict (dict): The serialization configuration.
            - tokenizer (Tokenizer, optional): The tokenizer to be saved.

    Returns:
        None

    Raises:
        ValueError: If the backend is not supported.
    """
    model = kwargs["model"]
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    layer_config = kwargs["layer_config"]
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)
    device = kwargs.get("device", None)
    image_processor = kwargs.get("image_processor", None)

    names = list(layer_config.keys())
    max_workers = 1
    if not torch.cuda.is_available() and not torch.xpu.is_available():
        max_workers = 2  ## 2 with cuda packing will cause hang occasionally
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(names), leave=True) as pbar:

            def wrapper(name):
                pbar.set_description(f"packing {name}")
                with tctl.threadpool_limits(limits=1):
                    pack_layer(name, model, kwargs.get("data_type", "fp8"), device)
                pbar.update(1)

            for _ in executor.map(wrapper, names):
                pass

    # Get llm-compressor format config
    check_compressed_tensors_supported()
    from compressed_tensors.quantization import (  # pylint: disable=E0401
        QuantizationArgs,
        QuantizationConfig,
        QuantizationScheme,
        QuantizationStatus,
        QuantizationStrategy,
        QuantizationType,
    )

    group_size = kwargs["serialization_dict"]["group_size"]
    if group_size == -1:
        strategy = QuantizationStrategy.CHANNEL
    elif group_size == 0:
        strategy = QuantizationStrategy.TENSOR
    else:
        strategy = QuantizationStrategy.GROUP
    if kwargs["serialization_dict"]["act_group_size"] != 0:
        logger.error(
            f"scheme FP8_STATIC export to llm_compressor format only support for act_group_size 0,"
            f" but got {kwargs['serialization_dict']['act_group_size']}, please check."
        )
        sys.exit(-1)
    scheme_args = dict(
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=strategy,
            symmetric=True,
            dynamic=False,
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
            dynamic=False,
        ),
    )
    targets = ["Linear"]
    ignore = []
    for layer_name in layer_config:
        if not check_to_quantized(layer_config[layer_name]):
            ignore.append(layer_name)
    config_groups = {}
    scheme = QuantizationScheme(targets=targets, **scheme_args)
    config_groups["group_0"] = scheme
    quantization_config = QuantizationConfig(
        config_groups=config_groups,
        kv_cache_scheme=None,
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=ignore,
    )
    quantization_config = quantization_config.to_dict()
    quantization_config["provider"] = "auto-round"
    quantization_config["format"] = "float-quantized"
    if group_size > 0:
        quantization_config["config_groups"]["group_0"]["weights"]["group_size"] = group_size
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config

    if output_dir is None:
        return model

    if os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    if processor is not None:
        processor.save_pretrained(output_dir)

    if image_processor is not None:
        image_processor.save_pretrained(output_dir)

    dtype = None
    save_model(model, output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model
