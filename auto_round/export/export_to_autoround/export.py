# Copyright (c) 2024 Intel Corporation
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
import functools
import inspect
import json
import os
import re
from dataclasses import fields
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.compressors.utils import is_mx_fp, is_nv_fp, is_standard_fp
from auto_round.export.export_to_autoround.utils import check_neq_config
from auto_round.export.utils import (
    filter_quantization_config,
    get_autogptq_packing_qlinear,
    release_layer_safely,
    resolve_pipeline_export_layout,
    save_model,
)
from auto_round.formats import AutoRoundExportFormat
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    set_module,
    to_standard_regex,
    unsupported_meta_device,
)

_NORM_CLASS_SUFFIX = re.compile(r".*Norm(?:\d+d)?\Z")

_DTYPE_STRING_ALIASES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "f32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "f16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _is_norm_module(module: nn.Module) -> bool:
    return _NORM_CLASS_SUFFIX.match(type(module).__name__) is not None


def _resolve_dtype_spec(dtype_spec) -> Optional[torch.dtype]:
    if dtype_spec is None:
        return None
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    if isinstance(dtype_spec, str):
        key = dtype_spec.strip().lower()
        if key in _DTYPE_STRING_ALIASES:
            return _DTYPE_STRING_ALIASES[key]
        raise ValueError(
            f"Unsupported dtype string: {dtype_spec!r}. " f"Supported: {sorted(set(_DTYPE_STRING_ALIASES))}"
        )
    raise TypeError(f"Expected torch.dtype or str, got {type(dtype_spec).__name__}: {dtype_spec!r}")


def _cast_norm_modules(model: nn.Module, dtype: Optional[torch.dtype]) -> int:
    """Upcast norm module parameters/buffers to ``dtype`` for residual precision."""
    if dtype is None:
        return 0
    count = 0
    skipped_meta = 0
    for name, module in model.named_modules():
        if not _is_norm_module(module):
            continue
        try:
            first_param = next(module.parameters(), None)
        except Exception:
            first_param = None
        if first_param is not None and first_param.device.type == "meta":
            skipped_meta += 1
            continue
        try:
            module.to(dtype)
            count += 1
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning("Skipping norm cast for %s (%s): %s", name, type(module).__name__, exc)
    if count:
        logger.info("Cast %d norm module(s) to %s for residual-precision export.", count, dtype)
    if skipped_meta:
        logger.warning(
            "Skipped %d norm module(s) on meta device; their dtype was not changed.",
            skipped_meta,
        )
    return count


@functools.lru_cache(maxsize=None)
def _quant_linear_accepts(qlinear_cls, kwarg_name: str) -> bool:
    target_cls = qlinear_cls
    if isinstance(target_cls, functools.partial):
        target_cls = target_cls.func
    try:
        sig = inspect.signature(target_cls.__init__)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if kwarg_name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def dynamic_import_quant_linear_for_packing(backend, bits, group_size, sym, act_bits=16):
    """
    Dynamically imports and returns the appropriate QuantLinear class based on the specified backend and parameters.

    Args:
        backend (str): The backend to be used for quantization. Supported values include "auto_round" "awq" and "gptq".
        bits (int): The number of bits for quantization.
        group_size (int): The group size for quantization.
        sym (bool): Flag indicating whether to use symmetric quantization.

    Returns:
        class: The dynamically imported QuantLinear class configured according to the specified parameters.

    Raises:
        ValueError: If the backend is not supported.
    """
    if "auto_round" in backend and "awq" not in backend and "gptq" not in backend:
        if act_bits <= 8:  ##easily have bug for other configuration, need to refine code later
            import auto_round.export.export_to_autoround.qlinear_triton_act

            return auto_round.export.export_to_autoround.qlinear_triton_act.QuantLinear
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        return QuantLinear
    elif "gptqmodel" in backend:
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        return functools.partial(QuantLinear, g_idx=True)
    elif "auto_round" in backend and "gptq" in backend and "gptqmodel" not in backend:
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

        return QuantLinear
    elif "awq" in backend:
        from ..export_to_awq.utils import WQLinear_GEMM

        return WQLinear_GEMM
    elif "gptq" in backend and "gptqmodel" not in backend:  ## have g_idx
        return get_autogptq_packing_qlinear(backend, bits, group_size, sym)
    else:
        raise ValueError(f"unsupported backend: '{backend}'. Supported backends are: {', '.join(SUPPORTED_FORMATS)}")


def pack_qact_layer(name, model):
    layer = get_module(model, name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if layer.bits > 8:
        return

    device = layer.weight.device
    bits = layer.bits
    group_size = layer.group_size
    act_bits = layer.act_bits

    act_scale = layer.act_scale if hasattr(layer, "act_scale") else None
    w_bf16_to_fp8_scale = layer.w_bf16_to_fp8_scale if hasattr(layer, "w_bf16_to_fp8_scale") else None
    scale = layer.scale
    zp = layer.zp
    import auto_round.export.export_to_autoround.qlinear_triton_act

    QuantLinear = auto_round.export.export_to_autoround.qlinear_triton_act.QuantLinear

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    bias = layer.bias is not None
    use_pc = False
    extra_kwargs: dict[str, "torch.dtype"] = {}
    if isinstance(scale, torch.Tensor) and _quant_linear_accepts(QuantLinear, "scale_dtype"):
        extra_kwargs["scale_dtype"] = scale.dtype
    new_layer = QuantLinear(  ##pylint: disable=E1123
        bits,
        group_size,
        in_features,
        out_features,
        bias,
        weight_dtype=layer.weight.dtype,
        use_pc=use_pc,
        **extra_kwargs,
    )
    new_layer.device = device
    set_module(model, name, new_layer)
    qlayer = new_layer

    qlayer.to("cpu")

    qlayer.pack(layer, scale, zp, act_scale, w_bf16_to_fp8_scale, device)
    qlayer.to(device)


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
    layer = get_module(model, layer_name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if type(layer) not in SUPPORTED_LAYER_TYPES:  ##already packed
        return

    if int(layer.act_bits) <= 8:
        return pack_qact_layer(layer_name, model)

    if not check_to_quantized(layer):
        return

    orig_device = layer.weight.device
    bits = layer.bits
    group_size = layer.group_size
    sym = layer.sym
    act_bits = layer.act_bits

    scale = layer.scale
    zp = layer.zp
    QuantLinear = dynamic_import_quant_linear_for_packing(backend, bits, group_size, sym, act_bits)

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    bias = layer.bias is not None

    extra_kwargs: dict[str, "torch.dtype"] = {}
    if isinstance(scale, torch.Tensor) and _quant_linear_accepts(QuantLinear, "scale_dtype"):
        extra_kwargs["scale_dtype"] = scale.dtype
    new_layer = QuantLinear(  ##pylint: disable=E1123
        bits, group_size, in_features, out_features, bias=bias, weight_dtype=layer.weight.dtype, **extra_kwargs
    )
    new_layer.device = orig_device
    set_module(model, layer_name, new_layer)
    qlayer = new_layer
    import auto_round_extension.torch.qlinear_torch

    if (
        sym
        and isinstance(zp, torch.Tensor)
        and isinstance(QuantLinear, (auto_round_extension.torch.qlinear_torch.QuantLinear))
    ):
        zp = int(zp.flatten()[0])

    qlayer.to("cpu")
    # Force to float32 to be compatible with torch 2.0
    sig = inspect.signature(qlayer.pack)
    param_count = len(sig.parameters)
    if param_count == 2:
        qlayer.pack(layer, scale, device=device)
    else:
        qlayer.pack(layer, scale, zp, None, device=device)
    qlayer.to(orig_device)
    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)


def save_quantized_as_autoround(
    output_dir: str,
    model: torch.nn.Module,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace=True,
    backend="auto_round:exllamav2",
    device: Union[str, torch.device] = "cpu",
    serialization_dict: dict = None,
    **kwargs,
):
    """
    Saves a quantized model in the auto-round format.

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
    # IF using sym, we change to gptq sym kernel to avoid compiling from auto_round source
    if (
        (serialization_dict.get("sym") is None or serialization_dict.get("sym"))
        and ("gptq" not in backend and "awq" not in backend)
        and (AutoRoundExportFormat.FP8_STATIC.value not in backend)
    ):
        backend = backend.replace("auto_round", "auto_round:auto_gptq")

    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    quantization_config = serialization_dict
    quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
    quantization_config["quant_method"] = "auto-round"
    quantization_config["packing_format"] = backend

    processor = kwargs.get("processor", None)
    image_processor = kwargs.get("image_processor", None)

    extra_config = {}
    block_name_to_quantize = quantization_config["block_name_to_quantize"]
    if isinstance(block_name_to_quantize, str):
        block_name_to_quantize = block_name_to_quantize.split(",")
    elif isinstance(block_name_to_quantize, list):
        for i in range(len(block_name_to_quantize)):
            block_name_to_quantize[i] = os.path.commonprefix(block_name_to_quantize[i]).rstrip(".")

    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    for layer_name, cfg in layer_config.items():
        if not cfg["in_blocks"] and cfg["bits"] <= 8:  # lm head
            extra_config[layer_name] = {key: cfg.get(key) for key in scheme_keys}
        elif cfg["in_blocks"] or (
            block_name_to_quantize is not None and check_start_with_block_name(layer_name, block_name_to_quantize)
        ):
            neq_keys = check_neq_config(cfg, **{k: quantization_config[k] for k in scheme_keys})
            if len(neq_keys) > 0:
                extra_config[layer_name] = {}
                for key in neq_keys:
                    if cfg.get(key) is not None:
                        extra_config[layer_name][key] = cfg[key]

    regex_config = quantization_config.pop("regex_config")
    if regex_config is not None:
        for name, cfg in regex_config.items():
            regex_name = to_standard_regex(name)
            neq_keys = check_neq_config(cfg, **{k: quantization_config[k] for k in scheme_keys})
            if len(neq_keys) > 0:
                extra_config[regex_name] = {}
                for key in neq_keys:
                    if cfg.get(key) is not None:
                        extra_config[regex_name][key] = cfg[key]

    if len(extra_config) > 0:
        quantization_config["extra_config"] = extra_config

    names = list(layer_config.keys())
    if not unsupported_meta_device(model):
        for name in tqdm(names, desc="packing", leave=True):
            pack_layer(name, model, backend, device)
    norm_dtype = _resolve_dtype_spec(kwargs.get("norm_dtype"))
    if norm_dtype is not None:
        _cast_norm_modules(model, norm_dtype)
        quantization_config["norm_dtype"] = str(norm_dtype).split(".")[-1]
    filter_quantization_config(quantization_config)
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    if output_dir is None:
        return model

    if output_dir is None:
        model.tokenizer = tokenizer
        return model
    # if os.path.exists(output_dir):
    #     logger.info(f"{output_dir} already exists, this may cause model conflict")
    model_output_dir = output_dir
    processor_output_dir = output_dir
    if output_dir:
        model_output_dir, processor_output_dir, _ = resolve_pipeline_export_layout(model, output_dir)

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(processor_output_dir)

    if processor is not None:
        processor.save_pretrained(processor_output_dir)
    if image_processor is not None:
        image_processor.save_pretrained(processor_output_dir)
    if quantization_config.get("act_bits", 16) <= 8:
        dtype = torch.bfloat16
    elif "awq" in quantization_config.get("packing_format", "auto_round:auto_gptq"):
        dtype = torch.float16  ## awq vllm kernel only supports float16 on cuda
    else:
        dtype = None
    save_model(model, model_output_dir, safe_serialization=safe_serialization, dtype=dtype)

    return model
