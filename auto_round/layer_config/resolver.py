# Copyright (c) 2026 Intel Corporation
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
import re
import sys
from dataclasses import asdict, fields
from typing import Union

import torch

from auto_round.layer_config.special_cases import apply_layer_config_special_cases
from auto_round.logger import logger
from auto_round.planning import ResolvedScheme
from auto_round.planning.contracts import LayerConfig, freeze_mapping
from auto_round.schemes import QuantizationScheme, get_gguf_scheme
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    compress_layer_names,
    infer_bits_by_data_type,
    to_standard_regex,
)


def _get_safetensor_layer_names_not_in_model(model, all_module_names: list) -> list:
    """Collect layer names from safetensor files that are not loaded into the model.

    Some tensors (e.g. MTP layers) exist in the original checkpoint but are not
    instantiated by ``transformers``.  This function discovers them so that regex
    patterns in ``layer_config`` can still match them.

    Returns:
        List of layer names (the path without the ``.weight`` suffix) for weight
        tensors present in the safetensor files but absent from *all_module_names*.
    """
    name_or_path = None
    if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
        name_or_path = model.config.name_or_path
    if not name_or_path:
        return []

    if not os.path.isdir(name_or_path):
        # Resolution is intentionally local-only: model downloads belong to model loading,
        # not to the pure layer-config planning phase.
        return []

    try:
        from safetensors import safe_open
    except ImportError:
        return []

    # Build tensor-name list from the safetensors index or single file
    source_index_file = os.path.join(name_or_path, "model.safetensors.index.json")
    source_single_file = os.path.join(name_or_path, "model.safetensors")

    tensor_names: list = []
    if os.path.exists(source_index_file):
        with open(source_index_file) as f:
            src_index = json.load(f)
        tensor_names = list(src_index["weight_map"].keys())
    elif os.path.exists(source_single_file):
        with safe_open(source_single_file, framework="pt", device="cpu") as f:
            tensor_names = list(f.keys())
    else:
        return []

    module_name_set = set(all_module_names)
    extra_layer_names = []
    for tensor_name in tensor_names:
        if not tensor_name.endswith(".weight"):
            continue
        layer_name = tensor_name[: -len(".weight")]
        if layer_name not in module_name_set:
            extra_layer_names.append(layer_name)
    return extra_layer_names


def _resolve_layer_config_presets(
    layer_config, model, ignore_layers, default_scheme, default_scale_dtype, fill_default_value
) -> tuple[dict, dict, tuple[str, ...], set[str]]:
    """Steps 1-4 of layer-config resolution: ignore-layer parsing, normalization,
    bits-inference, and default-filling."""
    from auto_round.schemes import QuantizationScheme, preset_name_to_scheme

    def normalize_item(item: Union[str, dict, "QuantizationScheme"], layer_name: str) -> dict:
        """Convert config entry into dict and validate keys."""
        if isinstance(item, str):
            config = asdict(preset_name_to_scheme(item.upper()))
        elif isinstance(item, QuantizationScheme):
            config = asdict(item)
        elif isinstance(item, dict):
            # "in_blocks" is an internal bookkeeping key injected by LLM-Compressor;
            # silently drop it before validation.
            item = {k: v for k, v in item.items() if k != "in_blocks"}
            scheme_name = item.pop("scheme", None)
            config = asdict(preset_name_to_scheme(scheme_name.upper())) if scheme_name is not None else {}
            invalid = set(item) - set(scheme_keys + ("fixed_by_user", "scale_dtype"))
            if invalid:
                raise ValueError(
                    f"Invalid keys {invalid} in layer_config for '{layer_name}'. " f"Allowed keys: {scheme_keys}"
                )
            config.update(item)
        else:
            raise TypeError(
                f"Unsupported type for layer_config[{layer_name}]: {type(item)}. "
                f"Expected str, dict, or QuantizationScheme."
            )
        # Clean up
        config = {k: v for k, v in config.items() if v is not None}
        config["fixed_by_user"] = True
        return config

    extra_scheme_keys = ("scale_dtype",)
    scheme_keys = tuple(f.name for f in fields(QuantizationScheme)) + ("scale_dtype",)
    layer_config = copy.deepcopy(layer_config) or {}
    ignore_layer_patterns = set()
    if ignore_layers:
        ignore_layers = ignore_layers.replace(" ", "").split(",")
        ignore_layers = [name + "." if name[-1].isdigit() else name for name in ignore_layers]
        ignore_layer_patterns = set(ignore_layers)

    # 1. ignore_layers -> force 16
    for name in get_fp_layer_names(model, ignore_layers):
        layer_config[name] = {
            "bits": 16,
            "act_bits": 16,
            "data_type": "float",
            "act_data_type": "float",
            "fixed_by_user": True,
        }

    # 2. normalize
    layer_config = {k: normalize_item(v, k) for k, v in layer_config.items()}

    # 3. infer missing bits
    for cfg in layer_config.values():
        if "data_type" in cfg and "bits" not in cfg:
            if (b := infer_bits_by_data_type(cfg["data_type"])) is not None:
                cfg["bits"] = b
        if "act_data_type" in cfg and "act_bits" not in cfg:
            if (b := infer_bits_by_data_type(cfg["act_data_type"])) is not None:
                cfg["act_bits"] = b

    # 4. fill defaults
    if isinstance(default_scheme, str):
        default_dict = asdict(preset_name_to_scheme(default_scheme.upper()))
    else:
        default_dict = asdict(default_scheme)
    default_dict["scale_dtype"] = default_scale_dtype

    # In AutoScheme with mixed gguf:q4_k_m, the super_group_size of gguf:q8_0 layer is None,
    # which should not be filled by default q4km again
    for cfg in layer_config.values():
        for key in scheme_keys:
            if fill_default_value:
                cfg.setdefault(key, copy.deepcopy(default_dict.get(key)))
            else:
                if key in extra_scheme_keys:
                    cfg.setdefault(key, copy.deepcopy(default_dict.get(key)))
                else:
                    cfg.setdefault(key, None)

    return layer_config, default_dict, scheme_keys, ignore_layer_patterns


def _traverse_and_expand_layer_config(
    layer_config, model, supported_types, inner_supported_types, ignore_layer_patterns, scheme_keys, gguf_name
) -> tuple[dict, dict, list[str], tuple]:
    """Steps 5-6 of layer-config resolution: supported-module collection (incl. the
    GGUF embedding-type-detection branch) and regex-config expansion."""
    from auto_round.utils.model import get_module

    # 5. collect supported modules
    embedding_types = (torch.nn.Embedding,)
    if gguf_name:
        if torch.nn.Embedding not in supported_types:
            supported_types = (*supported_types, torch.nn.Embedding)

        # for some Embedding which type() is not torch.nn.Embedding
        # for example: transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding
        model_module_name = model.__class__.__module__
        module_cls = sys.modules[model_module_name]
        for name in module_cls.__dict__:
            if name.endswith("Embedding") and not name.endswith("RotaryEmbedding"):
                embedding_types = (*embedding_types, getattr(module_cls, name))
        supported_types = (*supported_types, *embedding_types)

    all_supported_layer_names, embedding_layer_names = [], []
    all_module_names = []
    for n, m in model.named_modules():
        all_module_names.append(n)
        if type(m) not in supported_types and m.__class__.__name__ not in inner_supported_types:
            continue
        all_supported_layer_names.append(n)
        if isinstance(m, embedding_types) or m.__class__.__name__.endswith("Embedding"):
            embedding_layer_names.append(n)

    # Also include layer names from safetensor files not loaded into the model
    # (e.g. MTP layers that transformers does not instantiate).
    safetensor_only_names = _get_safetensor_layer_names_not_in_model(model, all_module_names)

    # 6. expand regex configs
    regex_config = {}
    for name in list(layer_config.keys()):
        if name in all_supported_layer_names:
            continue
        if name in all_module_names:
            m = get_module(model, name)
            if len(list(m.children())) == 0 and type(m) not in supported_types:
                val = layer_config.pop(name)
                if name in ignore_layer_patterns:
                    # Keep unsupported ignore_layers entries so export can serialize
                    # them into regex-based extra_config for loaders like vLLM INC.
                    regex_config[name] = val
                else:
                    logger.warning(
                        f"'{name}' exists in the model but is not a supported quantization target "
                        f"in the current scheme, ignoring its setting in `layer_config`"
                    )
                continue

        regex = re.compile(to_standard_regex(name))
        matched = [ln for ln in all_supported_layer_names if regex.search(ln)]
        safetensor_only_matched = [ln for ln in safetensor_only_names if regex.search(ln)]
        # skip it for mtp layers not loaded in transformers
        if not matched and not safetensor_only_matched:
            # type(mlp.gate) is Qwen3VLMoeTextTopKRouter instead of Linear
            logger.warning_once(
                f"Layer name or regex '{name}' in layer_config does not match any supported layers. "
                + "Please check for typos or update the regex pattern, ignore it for now"
            )
        val = layer_config.pop(name)
        regex_config[name] = val  # keep regex config
        for match in matched:
            layer_config[match] = val

    return layer_config, regex_config, embedding_layer_names, supported_types


def get_fp_layer_names(model: torch.nn.Module, ignore_layers: str):
    """Identifies and returns layers in the model to exclude from quantization.

    This function processes a comma-separated list of fully precision (FP) layers,
    matches them to the names of layers in the model, and returns a list of such
    layers to exclude from quantization.

    Args:
        model (torch.nn.Module): The model whose layers will be inspected.
        ignore_layers (str): A comma-separated string of layer names to be excluded
            from quantization. Whitespace is ignored in this string.

    Returns:
        list: A list of layer names that match the specified FP layers or are
        subcomponents of those layers.
    """
    from auto_round.utils import SUPPORTED_LAYER_TYPES

    if not ignore_layers:
        return []

    all_layer_names = []
    for n, m in model.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:
            all_layer_names.append(n)
    not_to_quantized_layers = []

    for fp_layer in ignore_layers:
        if fp_layer == "":
            continue
        if fp_layer in all_layer_names:
            not_to_quantized_layers.append(fp_layer)
            continue
        for name in all_layer_names:
            if fp_layer in name:
                not_to_quantized_layers.append(name)
    not_to_quantized_layers.extend(ignore_layers)  # keep regex name for later use
    if not_to_quantized_layers:
        logger.info(f"Ignored layers: {compress_layer_names(not_to_quantized_layers)}")
    return not_to_quantized_layers


def resolve_layer_config(
    *,
    model,
    scheme: ResolvedScheme,
    layer_config,
    scale_dtype=None,
    supported_types=None,
    inner_supported_types=None,
    quant_block_list=None,
    ignore_layers: str = "",
    quant_lm_head: bool = False,
    enable_gguf_official_mixed: bool = True,
    is_mllm: bool = False,
    fill_default_value: bool = True,
    layer_policy=None,
) -> LayerConfig:
    """Resolve final per-layer configuration without writing model attributes."""
    supported_types = tuple(SUPPORTED_LAYER_TYPES if supported_types is None else supported_types)
    inner_supported_types = tuple(
        INNER_SUPPORTED_LAYER_TYPES if inner_supported_types is None else inner_supported_types
    )
    scheme_value = scheme.value
    gguf_name = scheme.preset_name if (scheme.preset_name or "").startswith("gguf:") else get_gguf_scheme(scheme_value)

    resolved, default_dict, scheme_keys, ignore_patterns = _resolve_layer_config_presets(
        layer_config,
        model,
        ignore_layers,
        scheme_value,
        scale_dtype,
        fill_default_value,
    )
    resolved, _, embedding_names, supported_types = _traverse_and_expand_layer_config(
        resolved,
        model,
        supported_types,
        inner_supported_types,
        ignore_patterns,
        scheme_keys,
        gguf_name,
    )
    resolved, has_outside, lm_head_name, tied = apply_layer_config_special_cases(
        resolved,
        model,
        default_dict,
        supported_types,
        inner_supported_types,
        quant_block_list,
        quant_lm_head,
        gguf_name,
    )
    if layer_policy is not None:
        resolved = layer_policy.apply(
            model=model,
            scheme=scheme,
            layer_config=resolved,
            gguf_name=gguf_name,
            lm_head_name=lm_head_name,
            tie_word_embeddings=tied,
            embedding_layer_names=embedding_names,
            default_scale_dtype=scale_dtype,
            enable_gguf_official_mixed=enable_gguf_official_mixed,
            is_mllm=is_mllm,
            has_qlayer_outside_block=has_outside,
        )
    return freeze_mapping(resolved)


def extract_regex_config(
    *,
    model,
    scheme: ResolvedScheme,
    layer_config,
    scale_dtype=None,
    supported_types=None,
    inner_supported_types=None,
    ignore_layers: str = "",
    fill_default_value: bool = True,
) -> LayerConfig:
    """Resolve only the regex entries retained for export metadata."""
    supported_types = tuple(SUPPORTED_LAYER_TYPES if supported_types is None else supported_types)
    inner_supported_types = tuple(
        INNER_SUPPORTED_LAYER_TYPES if inner_supported_types is None else inner_supported_types
    )
    scheme_value = scheme.value
    gguf_name = scheme.preset_name if (scheme.preset_name or "").startswith("gguf:") else get_gguf_scheme(scheme_value)
    normalized, _, scheme_keys, ignore_patterns = _resolve_layer_config_presets(
        layer_config,
        model,
        ignore_layers,
        scheme_value,
        scale_dtype,
        fill_default_value,
    )
    _, regex_config, _, _ = _traverse_and_expand_layer_config(
        normalized,
        model,
        supported_types,
        inner_supported_types,
        ignore_patterns,
        scheme_keys,
        gguf_name,
    )
    return freeze_mapping(regex_config)


def has_quantized_layer_outside_blocks(layer_config: LayerConfig) -> bool:
    """Derive the block-outside flag from a final layer mapping."""
    return any(
        not config.get("in_blocks", False) and check_to_quantized(dict(config)) for config in layer_config.values()
    )
