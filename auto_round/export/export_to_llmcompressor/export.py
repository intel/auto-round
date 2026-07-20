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

from auto_round.export.utils import is_immediate_saving_mode, save_model, save_pretrained_artifact

# Lazy import: vLLM LinearBase (None when vLLM is not installed).
_VLLMLinearBase = None
try:
    from vllm.model_executor.layers.linear import LinearBase as _VLLMLinearBase
except Exception:
    pass
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


def _unfuse_integer_vllm_layer(name: str, model: torch.nn.Module, layer, output_sizes) -> bool:
    """Split a compressed-tensors integer-packed fused vLLM layer
    (qkv_proj / gate_up_proj) into separate nn.Linear sub-layers stored
    under the corresponding HF-checkpoint key names.

    For qkv_proj   → injects q_proj / k_proj / v_proj into the parent module.
    For gate_up_proj → injects gate_proj / up_proj into the parent module.

    Returns True if the layer was unfused, False otherwise.
    """
    layer_base = name.rsplit(".", 1)[-1]

    if layer_base == "qkv_proj":
        sub_names = ["q_proj", "k_proj", "v_proj"]
    elif layer_base == "gate_up_proj":
        sub_names = ["gate_proj", "up_proj"]
    else:
        return False

    if len(output_sizes) != len(sub_names):
        logger.warning(
            "Cannot unfuse integer vLLM layer %s: output_sizes length mismatch "
            "(expected %d, got %d). Saving as fused.",
            name,
            len(sub_names),
            len(output_sizes),
        )
        return False

    # Compute cumulative offsets
    offsets = [0]
    for s in output_sizes:
        offsets.append(offsets[-1] + s)

    # After _compress_and_set_format: weight becomes weight_packed (int4 → uint8)
    # or remains as weight for non-pack-quantized formats.
    if hasattr(layer, "weight_packed"):
        fused_weight = layer.weight_packed
        weight_attr = "weight_packed"
    else:
        fused_weight = layer.weight
        weight_attr = "weight"

    fused_scale = layer.weight_scale
    # weight_zero_point may be deleted by compress_module for symmetric quantization
    fused_zp = getattr(layer, "weight_zero_point", None)
    fused_bias = getattr(layer, "bias", None)
    q_scheme = layer.quantization_scheme

    # Determine in_features for creating sub nn.Linear shells
    if weight_attr == "weight_packed":
        in_features = fused_weight.shape[1] * 2
    else:
        in_features = fused_weight.shape[1]

    # Resolve parent module
    parts = name.rsplit(".", 1)
    parent_path = parts[0] if len(parts) == 2 else ""
    parent = get_module(model, parent_path) if parent_path else model

    for i, sub_name in enumerate(sub_names):
        row_start, row_end = offsets[i], offsets[i + 1]
        sub_out = row_end - row_start

        # Slice all attributes along output dimension (dim=0)
        sub_weight = fused_weight[row_start:row_end, :].contiguous()
        sub_scale = fused_scale[row_start:row_end, :].contiguous()
        sub_bias = fused_bias[row_start:row_end].contiguous() if fused_bias is not None else None

        # Create a minimal nn.Linear shell with compressed-tensors attributes
        sub_layer = torch.nn.Linear(in_features, sub_out, bias=sub_bias is not None)
        # Remove default float weight (will be replaced by packed tensor)
        del sub_layer.weight
        setattr(sub_layer, weight_attr, torch.nn.Parameter(sub_weight, requires_grad=False))
        sub_layer.weight_scale = torch.nn.Parameter(sub_scale, requires_grad=False)
        if fused_zp is not None:
            sub_zp = fused_zp[row_start:row_end, :].contiguous()
            sub_layer.weight_zero_point = torch.nn.Parameter(sub_zp, requires_grad=False)
        sub_layer.quantization_scheme = q_scheme
        if sub_bias is not None:
            sub_layer.bias = torch.nn.Parameter(sub_bias, requires_grad=False)

        setattr(parent, sub_name, sub_layer)

    delattr(parent, layer_base)
    logger.info("Unfused integer vLLM layer %s → %s", name, sub_names)
    return True


def _vllm_to_linear(name: str, model: torch.nn.Module, vllm_layer) -> torch.nn.Linear:
    """Replace a vLLM LinearBase layer in the model with an equivalent nn.Linear.

    compress_module from compressed-tensors produces 'pack-quantized' format for
    nn.Linear but falls back to 'dense' for unrecognised types (such as vLLM's
    LinearBase subclasses).  Converting to nn.Linear first guarantees the correct
    format and targets=["Linear"] in quantization_config.json.

    The returned nn.Linear has the same weight/bias data and all quantization
    attributes (bits, scale, zp, …) copied from the original vLLM layer.
    """
    in_features = getattr(vllm_layer, "input_size", vllm_layer.weight.shape[1])
    out_features = getattr(vllm_layer, "output_size", vllm_layer.weight.shape[0])
    has_bias = isinstance(getattr(vllm_layer, "bias", None), torch.Tensor)

    linear = torch.nn.Linear(in_features, out_features, bias=has_bias)
    # Replace the randomly-initialised weight with the actual (fake-quantised) weight
    linear.weight = torch.nn.Parameter(vllm_layer.weight.data.clone())
    if has_bias:
        linear.bias = torch.nn.Parameter(vllm_layer.bias.data.clone())

    # Copy all quantisation attributes expected by construct_ct_scheme / pack_layer
    for attr in (
        "bits",
        "sym",
        "group_size",
        "data_type",
        "act_bits",
        "act_data_type",
        "act_sym",
        "act_dynamic",
        "act_group_size",
        "scale",
        "zp",
    ):
        if hasattr(vllm_layer, attr):
            setattr(linear, attr, getattr(vllm_layer, attr))

    set_module(model, name, linear)
    return linear


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
    _is_vllm_linear = _VLLMLinearBase is not None and isinstance(layer, _VLLMLinearBase)
    if type(layer) not in SUPPORTED_LAYER_TYPES and not isinstance(layer, WrapperWALayer) and not _is_vllm_linear:
        return  ##already packed

    if hasattr(layer, "orig_layer"):  # revert WrapperWALayer for offline usage
        wp_layer = layer
        layer = wp_layer.orig_layer
        set_module(model, name, layer)
        _is_vllm_linear = _VLLMLinearBase is not None and isinstance(layer, _VLLMLinearBase)

    if not check_to_quantized(layer):
        return

    if hasattr(layer, "quantization_status") and layer.quantization_status == QuantizationStatus.COMPRESSED:
        return

    # Save vLLM fused-layer output_sizes and convert to nn.Linear before packing.
    # compress_module produces 'pack-quantized' for nn.Linear but falls back to
    # 'dense' for unrecognised types (vLLM LinearBase subclasses), so the
    # conversion must happen before _compress_and_set_format is called.
    vllm_output_sizes = None
    if _is_vllm_linear:
        if hasattr(layer, "output_sizes"):
            vllm_output_sizes = list(layer.output_sizes)
        layer = _vllm_to_linear(name, model, layer)

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

    # Unfuse vLLM fused layers (qkv_proj → q_proj/k_proj/v_proj, gate_up_proj → gate_proj/up_proj)
    if _is_vllm_linear and vllm_output_sizes is not None:
        _unfuse_integer_vllm_layer(name, model, layer, vllm_output_sizes)


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
    model.config.quantization_config = quantization_config.to_dict()

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
