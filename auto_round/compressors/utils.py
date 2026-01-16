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
import random
import re
import sys
from dataclasses import asdict, fields
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import transformers
from torch.amp import autocast

from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, GGUF_CONFIG, GGUF_INNER_CONFIG, QK_K, ModelType
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, get_gguf_scheme, preset_name_to_scheme
from auto_round.utils import check_to_quantized, copy_python_files_from_model_cache, is_fp8_linear, is_fp8_model


class BackendDataType(str, Enum):
    STANDARD_FP = "fp"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"


def is_standard_fp(backend):
    backend = backend.lower()
    return BackendDataType.STANDARD_FP in backend and not is_mx_fp(backend) and not is_nv_fp(backend)


def is_mx_fp(backend):
    backend = backend.lower()
    return BackendDataType.MX_FP in backend


def is_nv_fp(backend):
    backend = backend.lower()
    return BackendDataType.NV_FP in backend


def _is_weight_fp8_activation_static_fp8(
    bit: int, group_size: int, sym: bool, data_type: str, act_dynamic: bool
) -> bool:
    return bit == 8 and group_size == -1 and sym and data_type == "fp" and not act_dynamic


def is_wfp8afp8(ar):
    if (
        ("fp8" in ar.act_data_type or ("fp" in ar.act_data_type and ar.act_bits == 8))
        and ("fp8" in ar.data_type or ("fp" in ar.data_type and ar.bits == 8))
        and is_standard_fp(ar.act_data_type)
        and is_standard_fp(ar.data_type)
    ):
        return True
    else:
        return False


def is_static_wfp8afp8(ar_or_format: Union[str, Callable]) -> bool:
    if isinstance(ar_or_format, str):
        return "fp8_static" in ar_or_format
    if ar_or_format.act_dynamic:
        return False
    if is_wfp8afp8(ar_or_format):
        return True
    return False


def block_forward(
    block: torch.nn.Module,
    input_ids: torch.Tensor,
    input_others: dict,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cpu"),
    output_return_id: int = 0,
) -> Union[torch.Tensor, dict]:
    """Performs a forward pass through a block with the given inputs.

    Args:
    block: The block to perform the forward pass on.
    input_ids: The input IDs.
    input_others: A dictionary containing other input data.
    amp: A boolean indicating whether to use automatic mixed precision.
    amp_dtype: The data type for automatic mixed precision.
    device: The target device.
    output_return_id: if the output has more than one tenor, return the specified idx tensor.

    Returns:
    output: The output of the forward pass.
    """
    from auto_round.utils.model import to_device

    if input_ids.device != device:
        input_ids = to_device(input_ids, device)
        input_others = to_device(input_others, device)
    input_tuple = input_others.pop("positional_inputs", None)
    if "alibi" in input_others.keys() and input_others["alibi"] is not None:
        alibi = input_others["alibi"]
        input_others["alibi"] = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
    if amp:
        with autocast(device_type=str(device).split(":")[0], dtype=amp_dtype):  # pragma: no cover
            output = block(input_ids, *input_tuple, **input_others)
    else:
        output = block(input_ids, *input_tuple, **input_others)
    if isinstance(output_return_id, int) and (isinstance(output, list) or isinstance(output, tuple)):
        output = output[output_return_id]
    return output


def check_and_mark_fp8_model(model: torch.nn.Module) -> bool:
    if is_fp8_model(model):
        return True
    for n, m in model.named_modules():
        if is_fp8_linear(m):
            m.is_fp8_linear = True
            if not hasattr(model, "is_fp8"):
                model.is_fp8 = True
    if hasattr(model, "is_fp8"):
        return True
    return False


def check_skippable_keywords(key):
    """
    Prints a reminder if a key is not stored during quantization fine-tuning.
    """
    skippable_cache_keys = ("past_key_value",)
    for cache_key in skippable_cache_keys:
        if cache_key not in key:
            return True
    return False


def check_need_act_calibration(
    is_act_dynamic: Union[bool, None],
    act_data_type: Union[str, None] = None,
    act_bits: Union[int, None] = 16,
    static_kv_dtype: Union[str, None] = None,
    static_attention_dtype: Union[str, None] = None,
) -> bool:
    if static_kv_dtype is not None or static_attention_dtype is not None:
        return True
    if act_bits is None or act_bits > 8:
        return False
    # None is dynamic
    if is_act_dynamic is not None and not is_act_dynamic:
        return True
    if act_data_type is not None and "static" in act_data_type:
        return True
    return False


def collect_best_params(block, cache_device="cpu"):
    """Collect the best parameters from the block to the specified device."""
    params = {}
    if hasattr(block, "orig_layer"):
        for key in block.params.keys():
            params[key] = block.params[key].data.to(cache_device, copy=True)
    else:
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                params[n] = {}
                for key in m.params.keys():
                    params[n][key] = m.params[key].data.to(cache_device, copy=True)
    return params


def infer_bits_by_data_type(data_type: str):
    """Infer bits by data_type

    Args:
        data_type (str): data_type

    Returns:
        int: bits inferred by data_type, None means cannot infer correct bits by data_type
    """
    from auto_round.utils import SUPPORTED_DTYPES

    if data_type is None:
        return 16
    for supported_dtype in SUPPORTED_DTYPES:
        if data_type.startswith(supported_dtype) and len(data_type) > len(supported_dtype):
            ##first check the following two bits
            suc_2str = data_type[len(supported_dtype) : len(supported_dtype) + 2]
            if str.isdigit(suc_2str):
                return int(suc_2str)
            if str.isdigit(data_type[len(supported_dtype)]):
                return int(data_type[len(supported_dtype)])
    return None


def set_layer_config(
    model: torch.nn.Module,
    layer_config: dict[str, Union[str, dict, "QuantizationScheme"]],
    default_scheme: Union[str, "QuantizationScheme"],
    default_scale_dtype: torch.dtype | str,
    supported_types: tuple,
    inner_supported_types: tuple,
    quant_block_list=None,
    ignore_layers: str = "",
    quant_lm_head: bool = False,
    enable_gguf_official_mixed: bool = True,
    is_mllm: bool = False,
    fill_default_value=True,
) -> tuple[dict, bool, dict]:
    """
    Normalize, validate, and expand layer-specific quantization configs.
    Returns (final_layer_config, has_quant_layer_outside_block)
    """

    from auto_round.schemes import get_gguf_scheme
    from auto_round.utils.model import get_layer_names_in_block, get_lm_head_name, get_module, is_separate_lm_head

    # ---- helpers -------------------------------------------------
    def dispatch_layer_config(layer_config: dict[str, dict]) -> None:
        """Assign scheme values as attributes to matched modules."""
        for layer_name, scheme in layer_config.items():
            module = get_module(model, layer_name)
            for attr, value in scheme.items():
                setattr(module, attr, value)

    def normalize_item(item: Union[str, dict, "QuantizationScheme"], layer_name: str) -> dict:
        """Convert config entry into dict and validate keys."""
        if isinstance(item, str):
            config = asdict(preset_name_to_scheme(item.upper()))
        elif isinstance(item, QuantizationScheme):
            config = asdict(item)
        elif isinstance(item, dict):
            invalid = set(item) - set(scheme_keys + ("fixed_by_user", "scale_dtype"))
            if invalid:
                raise ValueError(
                    f"Invalid keys {invalid} in layer_config for '{layer_name}'. " f"Allowed keys: {scheme_keys}"
                )
            config = dict(item)
        else:
            raise TypeError(
                f"Unsupported type for layer_config[{layer_name}]: {type(item)}. "
                f"Expected str, dict, or QuantizationScheme."
            )
        # Clean up
        config = {k: v for k, v in config.items() if v is not None}
        config["fixed_by_user"] = True
        return config

    # ---- main logic ----------------------------------------------
    extra_scheme_keys = ("scale_dtype",)
    scheme_keys = tuple(f.name for f in fields(QuantizationScheme)) + ("scale_dtype",)
    layer_config = copy.deepcopy(layer_config) or {}

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

    # 5. collect supported modules
    embedding_types = (torch.nn.Embedding,)
    gguf_name = get_gguf_scheme(default_scheme)
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
        # cleanup stale attributes
        for key in scheme_keys:
            if hasattr(m, key):
                delattr(m, key)
        if type(m) not in supported_types and m.__class__.__name__ not in inner_supported_types:
            continue
        all_supported_layer_names.append(n)
        if isinstance(m, embedding_types) or m.__class__.__name__.endswith("Embedding"):
            embedding_layer_names.append(n)

    # 6. expand regex configs
    regex_config = {}
    for name in list(layer_config.keys()):
        if name in all_supported_layer_names:
            continue
        if name in all_module_names:
            m = get_module(model, name)
            if len(list(m.children())) == 0 and type(m) not in supported_types:
                layer_config.pop(name)
                logger.warning(f"{name} is not supported in current scheme, ignoring its setting in `layer_config`")
                continue

        regex = re.compile(name)
        matched = [ln for ln in all_supported_layer_names if regex.search(ln)]
        if not matched:
            raise ValueError(f"Invalid '{name}' in layer_config, no match found.")
        val = layer_config.pop(name)
        regex_config[name] = val  # keep regex config
        for match in matched:
            layer_config[match] = val
    # regex_config = None if len(regex_config)==0 else regex_config

    # 7. lm_head
    lm_head_name = get_lm_head_name(model)
    tie_word_embeddings = False
    if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
        tie_word_embeddings = model.config.tie_word_embeddings

    if lm_head_name in layer_config:
        quant_lm_head = True

    if quant_lm_head and tie_word_embeddings and not gguf_name:
        quant_lm_head = False
        logger.warning(
            "reset `quant_lm_head` to false as quantizing " "lm_head with tied weights has not been supported currently"
        )

    if lm_head_name not in layer_config and quant_lm_head:
        layer_config[lm_head_name] = copy.deepcopy(default_dict)

    if not quant_lm_head and not gguf_name:
        layer_config.pop(lm_head_name, None)

    # 8. enforce shape divisibility for int weight-only
    if default_dict["data_type"] == "int" and default_dict["act_bits"] >= 16 and not gguf_name:
        for n, m in model.named_modules():
            if type(m) in supported_types or m.__class__.__name__ in inner_supported_types:
                if m.weight.shape[0] % 32 or m.weight.shape[1] % 32:
                    layer_config.setdefault(n, copy.deepcopy(default_dict))
                    layer_config[n].update({"bits": 16, "data_type": "fp", "fixed_by_user": True})
                    logger.warning_once(f"{n} skipped quantization (shape not divisible by 32).")
    # enforce shape divisibility for mxfp/nvfp
    if (is_nv_fp(default_dict["data_type"]) or is_mx_fp(default_dict["data_type"])) and not gguf_name:
        for n, m in model.named_modules():
            if type(m) in supported_types or m.__class__.__name__ in inner_supported_types:
                if m.weight.shape[1] % default_dict["group_size"]:
                    layer_config.setdefault(n, copy.deepcopy(default_dict))
                    layer_config[n].update(
                        {"bits": 16, "data_type": "fp", "act_bits": 16, "act_data_type": "fp", "fixed_by_user": True}
                    )
                    logger.warning_once(
                        f"{n} skipped quantization (shape not divisible by {default_dict['group_size']})."
                    )

    # 9. block layers: mark as in_blocks=True
    for name in get_layer_names_in_block(model, supported_types, quant_block_list, inner_supported_types):
        if name not in layer_config:
            layer_config[name] = copy.deepcopy(default_dict)
            layer_config[name]["fixed_by_user"] = False
        layer_config[name]["in_blocks"] = True

    # ---- restore: ensure missing in_blocks are set to False and compute flag ----
    has_qlayer_outside_block = False
    for cfg in layer_config.values():
        if "in_blocks" not in cfg:
            cfg["in_blocks"] = False
        # mark layer outside block
        if not cfg["in_blocks"] and check_to_quantized(cfg):
            has_qlayer_outside_block = True

    # 10. GGUF handling
    if not gguf_name:
        dispatch_layer_config(layer_config)
        return layer_config, has_qlayer_outside_block, regex_config

    # embed + lm_head defaults for gguf
    tie_word_embeddings &= not is_separate_lm_head(model)
    if lm_head_name not in layer_config and not tie_word_embeddings:
        cfg = GGUF_INNER_CONFIG[GGUF_CONFIG[gguf_name.lower()]["lm_head"]]
        cfg = {**cfg, "fixed_by_user": False, "scale_dtype": default_scale_dtype}
        layer_config[lm_head_name] = cfg
        has_qlayer_outside_block = True
    for emd_name in embedding_layer_names:
        if emd_name in layer_config:
            continue
        if not tie_word_embeddings:
            cfg = GGUF_INNER_CONFIG[GGUF_CONFIG[gguf_name.lower()]["embedding"]]
        else:
            cfg = GGUF_INNER_CONFIG[GGUF_CONFIG[gguf_name.lower()]["lm_head"]]
        cfg = {**cfg, "fixed_by_user": False, "scale_dtype": default_scale_dtype}
        layer_config[emd_name] = cfg

    if enable_gguf_official_mixed:
        model_type = ModelType.MMPROJ if is_mllm else ModelType.TEXT
        layer_config, _ = get_layer_config_by_gguf_format(layer_config, gguf_name.lower(), model, model_type)

    dispatch_layer_config(layer_config)
    return layer_config, has_qlayer_outside_block, regex_config


def _use_more_bits(i_layer: int, n_layer: int):
    return (i_layer < n_layer // 8) or (i_layer >= 7 * n_layer // 8) or ((i_layer - n_layer // 8) % 3 == 2)


def _search_gguf_type(gguf_type):
    if gguf_type in GGUF_INNER_CONFIG:
        return gguf_type
    pattern = re.compile("gguf:q([0-9]{1,})_[01k]")
    bits = re.search(pattern, gguf_type)
    if not bits:
        raise KeyError(f"{gguf_type} is not a correct gguf type, please check")

    for suffix in ["_k", "_0", "_1"]:
        if gguf_type.endswith(suffix):
            continue
        if (tmp_type := re.sub("_[01k]", suffix, gguf_type)) in GGUF_INNER_CONFIG:
            return tmp_type
    return None


def gguf_type_fallback(gguf_type: str) -> str:
    gguf_type = gguf_type.lower()
    if gguf_type in ("gguf:q2_k", "gguf:q3_k", "gguf:q4_k"):
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q5_k":
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q6_k":
        gguf_type = "gguf:q8_0"
    return gguf_type


def get_gguf_qtype_by_layer_config(layer_config):
    import gguf  # pylint: disable=E0401

    if layer_config["bits"] >= 16:
        return None
    bits = layer_config["bits"]
    super_bits = layer_config.get("super_bits", None)
    sym = layer_config["sym"]
    group_size = layer_config.get("group_size", None)
    super_group_size = layer_config.get("super_group_size", None)
    if bits == 2 and super_bits == 4 and not sym and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q2_K
    if bits == 3 and super_bits == 6 and sym and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q3_K
    if bits == 4:
        if super_bits is not None and super_bits == 6 and not sym and group_size == 32 and super_group_size == 8:
            return gguf.GGMLQuantizationType.Q4_K
        if super_bits is None and sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q4_0
        if super_bits is None and not sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q4_1
    if bits == 5:
        if super_bits == 6 and not sym and group_size == 32 and super_group_size == 8:
            return gguf.GGMLQuantizationType.Q5_K
        if super_bits is None and sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q5_0
        if super_bits is None and not sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q5_1
    if bits == 6 and super_bits == 8 and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q6_K
    if bits == 8 and sym and group_size == 32:
        return gguf.GGMLQuantizationType.Q8_0
    raise ValueError("Unknown layer config")


def _get_digital_in_layer_name(layer_name):
    pattern = re.compile(r"([a-zA-Z]+\.){1,}(\d+)")
    res = re.search(pattern, layer_name)
    if res:
        return int(res[2])
    else:
        return None


def _gguf_type_fallback(gguf_type: str) -> str:
    gguf_type = gguf_type.lower()
    if gguf_type in ("gguf:q2_k", "gguf:q3_k", "gguf:q4_k"):
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q5_k":
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q6_k":
        gguf_type = "gguf:q8_0"
    return gguf_type


##https://github.com/ggml-org/llama.cpp/blob/9e31bec4fd53634c9e5b04650488a09a055f5dab/src/llama-quant.cpp#L129
def get_layer_config_by_gguf_format(layer_config, target_gguf_format: str, model, model_type=ModelType.TEXT):
    # # TODO: support for other format later
    # target_gguf_format = next((fmt for fmt in gguf_format if fmt != "fake"), None)

    import gguf  # pylint: disable=E0401

    from auto_round.utils.common import MM_KEYS, LazyImport
    from auto_round.utils.model import get_lm_head_name, get_module

    # from auto_round.export.export_to_gguf.convert import ModelBase, get_model_architecture
    convert_hf_to_gguf = LazyImport("auto_round.export.export_to_gguf.convert_hf_to_gguf")

    model_architecture = convert_hf_to_gguf.get_model_architecture(
        hparams=model.config.to_dict(), model_type=model_type
    )
    try:
        if model_type != ModelType.TEXT:
            model_class_vision = convert_hf_to_gguf.ModelBase.from_model_architecture(
                model_architecture, model_type=model_type
            )
        model_class = convert_hf_to_gguf.ModelBase.from_model_architecture(
            model_architecture, model_type=ModelType.TEXT
        )

    except NotImplementedError:
        return layer_config, {}

    n_layer = None
    if model_type != ModelType.TEXT:
        n_layer_vision = None
    for name in ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth"]:
        if hasattr(model.config, name):
            n_layer = getattr(model.config, name)
        if model_type != ModelType.TEXT:
            if n_layer is not None and hasattr(model.config, "text_config"):
                if hasattr(getattr(model.config, "text_config"), name):
                    n_layer = getattr(getattr(model.config, "text_config"), name)
            for config_name in ["vision_config", "vision_encoder"]:
                if hasattr(model.config, config_name):
                    if hasattr(getattr(model.config, config_name), name):
                        n_layer_vision = getattr(getattr(model.config, config_name), name)
                        break
            if n_layer and n_layer_vision:
                break

    if n_layer is None:
        return layer_config, {}

    tensor_map = gguf.get_tensor_name_map(model_class.model_arch, n_layer)
    if model_type != ModelType.TEXT:
        tensor_map_vision = gguf.get_tensor_name_map(model_class_vision.model_arch, n_layer_vision)

    def _set_config(config, target_config):
        for k, v in target_config.items():
            if isinstance(config, dict):
                config[k] = v
            else:
                setattr(config, k, v)
        return config

    gguf_format_config = {}
    lm_head_name = get_lm_head_name(model)
    inner_gguf_format = GGUF_CONFIG[target_gguf_format]["mostly"]
    # ggml_type =  getattr(gguf.GGMLQuantizationType,inner_gguf_format.split(":")[-1].upper())
    block_size = GGML_QUANT_SIZES[inner_gguf_format.split(":")[-1].lower()][0]
    tie_word_embeddings = True
    if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
        tie_word_embeddings = model.config.tie_word_embeddings

    n_gqa = 1
    if (
        hasattr(model, "config")
        and hasattr(model.config, "num_attention_heads")
        and hasattr(model.config, "num_key_value_heads")
    ):
        n_gqa = model.config.num_attention_heads // model.config.num_key_value_heads
    n_expert = 0
    for name in ["num_experts", "num_local_experts", "n_routed_experts"]:
        if hasattr(model.config, name):
            n_expert = getattr(model.config, name)

    i_attention_wv = 0
    i_ffn_down = 0
    layer_config_copy = copy.deepcopy(layer_config)
    target_bits = None
    if inner_gguf_format.startswith("gguf:q") and len(inner_gguf_format) >= 7 and (inner_gguf_format[6]).isdigit():
        target_bits = int(inner_gguf_format[6])

    for layer_name, config in layer_config_copy.items():
        if not check_to_quantized(config):
            continue
        new_type = GGUF_CONFIG[target_gguf_format]["mostly"]
        layer = get_module(model, layer_name)
        if type(layer) == transformers.pytorch_utils.Conv1D:
            input_features = layer.weight.shape[0]
        else:
            input_features = layer.weight.shape[-1]
        i_layer = _get_digital_in_layer_name(layer_name)

        if lm_head_name is not None and layer_name == lm_head_name:
            target_bits = int(re.search("gguf:q([0-9]{1,})_[01k]", GGUF_CONFIG[target_gguf_format]["lm_head"]).group(1))
        if isinstance(layer, torch.nn.Embedding):
            target_bits = int(
                re.search("gguf:q([0-9]{1,})_[01k]", GGUF_CONFIG[target_gguf_format]["embedding"]).group(1)
            )

        if model_type != ModelType.TEXT and any([key in layer_name for key in MM_KEYS]):
            gguf_name = tensor_map_vision.get_name(layer_name)
            if gguf_name is None:
                for key in MM_KEYS:
                    gguf_name = tensor_map_vision.get_name(layer_name.replace(f".{key}", ""))
                    if gguf_name is not None:
                        break
        else:
            gguf_name = tensor_map.get_name(layer_name)
            if gguf_name is None:
                gguf_name = tensor_map.get_name(layer_name.replace(".language_model", ""))
        bits_index = 6
        if config.get("fixed_by_user", False):
            if "bits" not in config:
                logger.warning(
                    f"Setting layer_config requires providing bits, {layer_name} has not bits,"
                    f" using bits={target_bits} instead."
                )
                new_type = new_type[:bits_index] + target_bits + new_type[bits_index + 1 :]
            else:
                config_tmp = config.copy()
                scheme_keys = [f.name for f in fields(QuantizationScheme)]
                for key in config.keys():
                    if key not in scheme_keys:
                        config_tmp.pop(key, None)
                matched_scheme = get_gguf_scheme(QuantizationScheme.from_dict(config_tmp))  # check matched
                if not matched_scheme:
                    if config.get("super_group_size", None) is not None or config.get("super_bits", None) is not None:
                        new_type = new_type[:bits_index] + str(config["bits"]) + "_k"
                    if new_type not in GGUF_INNER_CONFIG:
                        prefix_idx = 0 if config.get("sym", True) else 1
                        new_type = new_type[:bits_index] + str(config["bits"]) + f"_{prefix_idx}"
                        if new_type not in GGUF_INNER_CONFIG:
                            new_type = new_type[:bits_index] + str(config["bits"]) + f"_{1-prefix_idx}"
                    if new_type not in GGUF_INNER_CONFIG:
                        raise ValueError(
                            f"the setting in layer_config {layer_name} "
                            f"could not match any supported gguf format, please have a check."
                        )

                new_type = new_type[:bits_index] + str(config["bits"]) + new_type[bits_index + 1 :]
            new_type = _search_gguf_type(new_type)
            if new_type is None:
                raise ValueError(f"invalid bit setting for {layer_name}")
        elif target_bits is not None and "bits" in config and config["bits"] != target_bits:
            new_type = new_type[:bits_index] + str(config["bits"]) + new_type[bits_index + 1 :]
            new_type = _search_gguf_type(new_type)
            if new_type is None:
                raise ValueError(f"invalid bit setting for {layer_name}")
        elif lm_head_name is not None and layer_name == lm_head_name and not tie_word_embeddings:
            if gguf.MODEL_ARCH.FALCON == model_class.model_arch or input_features % block_size != 0:
                new_type = "gguf:q8_0"
            elif "lm_head" in GGUF_CONFIG[target_gguf_format]:
                new_type = GGUF_CONFIG[target_gguf_format]["lm_head"]
            elif new_type != "gguf:q8_0":
                new_type = "gguf:q6_k"
        elif lm_head_name is not None and layer_name == lm_head_name and tie_word_embeddings:
            # new_type = GGUF_CONFIG[target_gguf_format]["lm_head"]
            continue
        elif isinstance(layer, torch.nn.Embedding):
            if "embedding" in GGUF_CONFIG[target_gguf_format]:
                new_type = GGUF_CONFIG[target_gguf_format]["embedding"]
        elif gguf_name is None:
            pass
        # attn_v
        elif "attn_v" in gguf_name:
            if target_gguf_format == "gguf:q2_k":
                new_type = "gguf:q4_k" if n_gqa >= 4 else "gguf:q3_k"
            elif target_gguf_format == "gguf:q2_k_s" and n_gqa >= 4:
                new_type = "gguf:q4_k"
            elif target_gguf_format == "gguf:q3_k_m":
                new_type = "gguf:q5_k" if i_attention_wv < 2 else "gguf:q4_k"
            elif target_gguf_format == "gguf:q3_k_l":
                new_type = "gguf:q5_k"
            elif (target_gguf_format == "gguf:q4_k_m" or target_gguf_format == "gguf:q5_k_m") and _use_more_bits(
                i_layer, n_layer
            ):
                new_type = "gguf:q6_k"
            elif target_gguf_format == "gguf:q4_k_s" and i_attention_wv < 4:
                new_type = "gguf:q5_k"
            ##TODO check which models are be grouped into to LLM_TYPE_70B
            # if (qs.model.type == LLM_TYPE_70B) {
            # // In the 70B model we have 8 heads sharing the same attn_v weights.
            # As a result, the attn_v.weight tensor is
            # // 8x smaller compared to attn_q.weight.Hence, we can get a nice boost in quantization accuracy with
            # // nearly negligible increase in model size by quantizing this tensor with more bits:
            #     if
            # (new_type == GGML_TYPE_Q3_K | | new_type == GGML_TYPE_Q4_K)
            # new_type = GGML_TYPE_Q5_K;
            # }
            if n_expert == 8:
                new_type = "gguf:q8_k"
            i_attention_wv += 1

        elif "attn_k" in gguf_name:
            if n_expert == 8:
                new_type = "gguf:q8_0"
        # ffn_down
        elif "ffn_down" in gguf_name:
            if target_gguf_format == "gguf:q2_k":
                new_type = "gguf:q3_k"
            elif target_gguf_format == "gguf:q2_k_s":
                if i_layer < n_layer / 8:
                    new_type = "gguf:q4_k"
            elif target_gguf_format == "gguf:q3_k_m":
                if i_layer < n_layer / 16:
                    new_type = "gguf:q5_k"
                elif gguf.MODEL_ARCH.FALCON == model_class.model_arch or _use_more_bits(i_layer, n_layer):
                    new_type = "gguf:q4_k"
                else:
                    new_type = "gguf:q3_k"
            elif target_gguf_format == "gguf:q3_k_l":
                if gguf.MODEL_ARCH.FALCON == model_class.model_arch:
                    new_type = "gguf:q4_k"
                else:
                    new_type = "gguf:q5_k"
            elif target_gguf_format == "gguf:q4_k_m":
                if gguf.MODEL_ARCH.FALCON == model_class.model_arch:
                    if i_layer < n_layer // 16:
                        new_type = "gguf:q6_k"
                    elif _use_more_bits(i_layer, n_layer):
                        new_type = "gguf:q5_k"
                    else:
                        new_type = "gguf:q4_k"
                else:
                    if _use_more_bits(i_layer, n_layer):
                        new_type = "gguf:q6_k"
            elif target_gguf_format == "gguf:q5_k_m" and _use_more_bits(i_layer, n_layer):
                new_type = "gguf:q6_k"
            elif (
                target_gguf_format == "gguf:q4_k_s"
                and model_class.model_arch != gguf.MODEL_ARCH.FALCON
                and i_layer < n_layer / 8
            ):
                new_type = "gguf:q5_k"
            elif (target_gguf_format == "gguf:q4_0" or target_gguf_format == "gguf:q5_0") and i_layer < n_layer / 8:
                if target_gguf_format == "gguf:q4_0":
                    new_type = "gguf:q4_1"
                else:
                    new_type = "gguf:q5_1"
            i_ffn_down += 1

        # attn_output
        elif "attn_output" in gguf_name:
            if gguf.MODEL_ARCH.FALCON != model_class.model_arch:
                if n_expert == 8:
                    if target_gguf_format in (
                        "gguf:q2_k",
                        "gguf:q3_k_s",
                        "gguf:q3_k_m",
                        "gguf:q4_k_s",
                        "gguf:q4_k_m",
                        "gguf:q5_k",
                    ):
                        new_type = "gguf:q5_k"
                    elif target_gguf_format == "gguf:q2_k":
                        new_type = "gguf:q3_k"
                    elif target_gguf_format == "gguf:q3_k_m":
                        new_type = "gguf:q4_k"
                    elif target_gguf_format == "gguf:q3_k_l":
                        new_type = "gguf:q5_k"
            else:
                if target_gguf_format == "gguf:q3_k_l":
                    new_type = "gguf:q4_k"
        # attn_qkv
        elif "attn_qkv" in gguf_name:
            if target_gguf_format in ("gguf:q3_k_m", "gguf:q3_k_l"):
                new_type = "gguf:q4_k"
            elif target_gguf_format == "gguf:q4_k_m":
                new_type = "gguf:q5_k"
            elif target_gguf_format == "gguf:q5_k_m":
                new_type = "gguf:q5_k"
        new_block_size = GGML_QUANT_SIZES[new_type.split(":")[-1].lower()][0]
        if input_features % new_block_size != 0:
            new_type = _gguf_type_fallback(new_type)
            new_block_size = GGML_QUANT_SIZES[new_type.split(":")[-1].lower()][0]
            if input_features % new_block_size != 0:
                new_type = "gguf:bf16"
            logger.warning(
                f"fallback {layer_name} to {new_type}, "
                f"because input_features({input_features}) % block_size({block_size}) != 0"
            )
        # for deepseek v2
        if layer_name.endswith("kv_b_proj") and new_type.endswith("_k") and "Deepseek" in model.config.architectures[0]:
            fallback = False

            # calc if need fallback
            qk_nope_head_dim = model.config.qk_nope_head_dim
            kv_b_shape = get_module(model, layer_name).weight.shape

            if (
                qk_nope_head_dim < QK_K
                or qk_nope_head_dim % QK_K != 0
                or kv_b_shape[-1] < QK_K
                or kv_b_shape[-1] % QK_K != 0
            ):
                fallback = True
            if fallback:
                tmp_type = _gguf_type_fallback(new_type)
                logger.warning_once(
                    f"self_attn.kv_b_proj does not support the use of {new_type}, replace it with {tmp_type}"
                )
                new_type = tmp_type

        target_config = GGUF_INNER_CONFIG[new_type]

        _set_config(layer_config[layer_name], target_config)
        _set_config(layer, target_config)
        gguf_format_config[layer_name] = new_type

    return layer_config, gguf_format_config


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
    ignore_layers = ignore_layers.replace(" ", "").split(",")
    all_layer_names = []
    for n, m in model.named_modules():
        # if type(m) in SUPPORTED_LAYER_TYPES:
        if type(m) in SUPPORTED_LAYER_TYPES or "Linear" in str(type(m)):
            all_layer_names.append(n)
    not_to_quantized_layers = []

    for fp_layer in ignore_layers:
        if fp_layer == "":
            continue
        if fp_layer in all_layer_names:
            not_to_quantized_layers.append(fp_layer)
            continue
        if fp_layer[-1].isdigit():
            fp_layer = fp_layer + "."  ##tricky setting
        for name in all_layer_names:
            if fp_layer in name:
                not_to_quantized_layers.append(name)
    logger.trace(f"not_to_quantized_layers: {not_to_quantized_layers}")
    return not_to_quantized_layers


def get_shared_keys(model):
    """
    Retrieves shared keys from the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to retrieve shared keys from.

    Returns:
        tuple: tuple of shared keys.
    """
    from auto_round.special_model_handler import SPECIAL_SHARED_CACHE_KEYS
    from auto_round.utils import SHARED_CACHE_KEYS

    shared_keys = SHARED_CACHE_KEYS
    shared_keys += SPECIAL_SHARED_CACHE_KEYS.get(model.__class__.__name__, ())
    return shared_keys


def init_cache(positional_inputs, inputs):
    """
    Initializes special model inputs by adding positional inputs if missing.

    Args:
        positional_inputs (list): List of positional inputs to add to inputs.
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Adds "positional_inputs" key if not present.
    """
    from auto_round.utils.model import to_device

    if "positional_inputs" not in inputs:  # for chatglm Series
        inputs["positional_inputs"] = []
    for idx, item in enumerate(positional_inputs):
        inputs["positional_inputs"] = to_device(positional_inputs)


def reset_params(inputs):
    """
    Resets specific input parameters to avoid saving the key-value cache during fine-tuning.

    Args:
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Sets "use_cache" to False if the key is present.
    """
    if "use_cache" in inputs.keys():  # Not storing kv cache
        inputs["use_cache"] = False


def immediate_saving(rounder: object, m: torch.nn.Module, name: str = None, last_group: bool = False):
    """
    Shard-saves the parameters of a model block (or group of blocks) immediately into disk,
    accumulating tensors into size-limited shards, optionally finalizing all remaining
    model weights when processing the last group.

    Args:
        rounder (object): The object of compressor.
        m (torch.nn.Module): The current block (or composite module) whose parameters will be added to the shard set.
        name (str): Override module name used as prefix for saved parameter keys. If None, falls back to m.tmp_name.
        last_group (bool): If True, triggers final pass over the entire model to include unsaved weights,
            writes shard index, renames shard files, copies source files, and releases temporary state.
    """
    import json
    from collections import OrderedDict

    from auto_round.utils import clear_memory, get_module

    # User configurable (can be preset on rounder)
    max_shard_size = getattr(rounder, "max_shard_size", "5GB")
    safe_serialization = getattr(rounder, "safe_serialization", True)
    if not hasattr(rounder, "quantized_layer_names_outside_blocks"):
        rounder.quantized_layer_names_outside_blocks = rounder._get_quantized_layer_names_outside_blocks()
    layer_names = rounder.quantized_layer_names_outside_blocks
    if len(layer_names) > 0 and name != layer_names[-1]:
        last_group = False

    def _parse_size(size_str: str) -> int:
        s = size_str.strip().upper()
        if s.endswith("GB"):
            return int(s[:-2]) * (1024**3)
        if s.endswith("MB"):
            return int(s[:-2]) * (1024**2)
        if s.endswith("KB"):
            return int(s[:-2]) * 1024
        return int(s)

    # Init global accumulators (once)
    if not hasattr(rounder, "_shard_init_done"):
        rounder._shard_init_done = True
        rounder._max_shard_bytes = _parse_size(str(max_shard_size))
        rounder._use_safetensors = False
        if safe_serialization:
            try:
                from safetensors.torch import save_file as _sf  # noqa

                rounder._use_safetensors = True
            except ImportError:
                logger.warning("safe_serialization=True but safetensors not installed; fallback to torch.save.")
        rounder._current_shard_tensors = OrderedDict()
        rounder._current_shard_size = 0
        rounder._shard_meta = []  # list of dicts: {file, params}
        rounder._global_weight_map = {}  # param_name -> final shard file (filled after finalize)
        rounder._shard_counter = 0
        rounder._shard_suffix = "safetensors" if rounder._use_safetensors else "bin"
        # new global counters
        rounder._total_param_elems = 0
        rounder._total_param_size_bytes = 0
        # Directory
        rounder._packed_blocks_root = os.path.join(rounder._get_save_folder_name(rounder.formats[0]), "")
        os.makedirs(rounder._packed_blocks_root, exist_ok=True)

    # Collect tensors directly from current (multi)block `m`
    flat_tensors = OrderedDict()
    for k, v in m.state_dict().items():
        tmp_name = name if name is not None else m.tmp_name
        if isinstance(v, torch.Tensor):
            flat_tensors[f"{tmp_name}.{k}"] = v

    # Append tensors into the running shard(s)
    def _flush_current_shard():
        if len(rounder._current_shard_tensors) == 0:
            return
        rounder._shard_counter += 1
        tmp_name = f"model-shard-{rounder._shard_counter:05d}.{rounder._shard_suffix}"  # temporary name
        tmp_path = os.path.join(rounder._packed_blocks_root, tmp_name)
        if rounder._use_safetensors:
            from safetensors.torch import save_file

            save_file(rounder._current_shard_tensors, tmp_path)
        else:
            torch.save(rounder._current_shard_tensors, tmp_path)
        params = list(rounder._current_shard_tensors.keys())
        rounder._shard_meta.append({"tmp_file": tmp_name, "params": params})
        for param in params:
            free_module_name = param.rsplit(".", 1)[0]
            free_module = get_module(rounder.model, free_module_name)

            # free module only when all its parameters have been saved
            free_flag = True
            free_module_state_dict = free_module.state_dict()
            already_saved_name = []
            for _meta in rounder._shard_meta:
                already_saved_name += _meta.get("params", [])
            for free_module_key in free_module_state_dict:
                free_module_key_full_name = f"{free_module_name}.{free_module_key}"
                if free_module_key_full_name not in already_saved_name:
                    free_flag = False
            if free_flag:
                free_module.to("meta")
                del rounder._current_shard_tensors[param]
        rounder._current_shard_tensors = OrderedDict()
        rounder._current_shard_size = 0

    for pname, tensor in flat_tensors.items():
        t_elems = tensor.numel()
        t_size = t_elems * tensor.element_size()
        # accumulate global stats
        rounder._total_param_elems += t_elems
        rounder._total_param_size_bytes += t_size
        if t_size > rounder._max_shard_bytes:
            _flush_current_shard()
            rounder._current_shard_tensors[pname] = tensor
            rounder._current_shard_size = t_size
            _flush_current_shard()
            continue
        if rounder._current_shard_size + t_size > rounder._max_shard_bytes and rounder._current_shard_size > 0:
            _flush_current_shard()
        rounder._current_shard_tensors[pname] = tensor
        rounder._current_shard_size += t_size

    if last_group:

        # 1) Add the remaining (unsaved) model weights into new shard(s),
        # do not overwrite the already saved weights.
        try:
            full_sd = rounder.model.state_dict()
        except Exception as e:
            logger.warning(f"failed to obtain full state_dict for remaining weights: {e}")
            full_sd = {}
        tie_word_embeddings: bool = getattr(getattr(rounder.model, "config", None), "tie_word_embeddings", True)
        for pname, tensor in full_sd.items():
            if "lm_head" in pname and tie_word_embeddings:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            # Check whether pname already stored in previous shards via _shard_meta
            already_saved = False
            for _meta in rounder._shard_meta:
                if pname in _meta.get("params", []):
                    already_saved = True
                    break
            if already_saved:
                continue  # skip weights already saved
            # Size accounting
            t_elems = tensor.numel()
            t_size = t_elems * tensor.element_size()

            # Update global stats (these counters may already include earlier packed weights)
            rounder._total_param_elems += t_elems
            rounder._total_param_size_bytes += t_size

            # If this tensor alone exceeds shard size -> dedicated shard
            if t_size > rounder._max_shard_bytes:
                _flush_current_shard()
                rounder._current_shard_tensors[pname] = tensor.detach().cpu()
                rounder._current_shard_size = t_size
                _flush_current_shard()
                continue

            # If adding this tensor would overflow current shard -> flush current first
            if rounder._current_shard_size + t_size > rounder._max_shard_bytes and rounder._current_shard_size > 0:
                _flush_current_shard()

            # Add to current shard
            rounder._current_shard_tensors[pname] = tensor.detach().cpu()
            rounder._current_shard_size += t_size

        # 2) Flush any remaining unsaved leftover tensors
        _flush_current_shard()

        # 3) Finalize: rename temp shard files to HF-style names and build index
        total_shards = rounder._shard_counter
        if total_shards == 0:
            logger.warning("no tensors saved across all blocks")
        else:
            final_names = []
            for idx, meta in enumerate(rounder._shard_meta, start=1):
                old_tmp = meta["tmp_file"]
                old_path = os.path.join(rounder._packed_blocks_root, old_tmp)
                if total_shards == 1:
                    new_name = f"model.{rounder._shard_suffix}"
                else:
                    new_name = f"model-{idx:05d}-of-{total_shards:05d}.{rounder._shard_suffix}"
                new_path = os.path.join(rounder._packed_blocks_root, new_name)
                os.rename(old_path, new_path)
                final_names.append(new_name)
                for p in meta["params"]:
                    rounder._global_weight_map[p] = new_name

            index_fname = "model.safetensors.index.json" if rounder._use_safetensors else "model.bin.index.json"
            index_path = os.path.join(rounder._packed_blocks_root, index_fname)
            index_data = {
                "metadata": {
                    "format": "safetensors" if rounder._use_safetensors else "pytorch",
                    "total_shards": total_shards,
                    "total_parameters": int(getattr(rounder, "_total_param_elems", 0)),
                    "total_size": int(getattr(rounder, "_total_param_size_bytes", 0)),
                },
                "weight_map": rounder._global_weight_map,
            }
            if total_shards > 1:
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(index_data, f, indent=2)
            logger.info(
                f"saved {total_shards} shard(s) (HF-style, including remaining unsaved weights) to "
                f"{rounder._packed_blocks_root} (index: {index_fname})"
            )

            try:
                copy_python_files_from_model_cache(rounder.model, rounder._get_save_folder_name(rounder.formats[0]))
            except Exception as e:
                logger.warning("Skipping source model Python file copy due to error: %s", e)

        # 4) Cleanup attributes to release memory after final shard is written
        try:
            attrs_to_cleanup = [
                "_current_shard_tensors",
                "_current_shard_size",
                "_shard_counter",
                "_max_shard_bytes",
                "_use_safetensors",
                "_shard_suffix",
                "_packed_blocks_root",
                "_total_param_elems",
                "_total_param_size_bytes",
                "_shard_init_done",
                "_shard_meta",
                "_global_weight_map",
            ]
            for _attr in attrs_to_cleanup:
                if hasattr(rounder, _attr):
                    delattr(rounder, _attr)
        except Exception as _cleanup_err:
            logger.warning(f"shard cleanup warning: {_cleanup_err}")


class IndexSampler:
    """A cyclic sampler that returns shuffled index batches.

    This sampler maintains internal state so that each call to `next_batch()`
    continues from where it left off. When the remaining number of samples is
    less than `batch_size`, the sampler reshuffles all indices and starts from
    the beginning, discarding the last incomplete batch.

    Attributes:
        nsamples (int): Total number of samples.
        batch_size (int): Number of indices to return in each batch.
        index (int): Current position in the index list.
        indices (List[int]): Shuffled list of indices.
    """

    def __init__(self, nsamples: int, batch_size: int) -> None:
        """Initializes the sampler.

        Args:
            nsamples (int): Total number of samples (must be >= batch_size).
            batch_size (int): Number of indices per batch.

        Raises:
            ValueError: If batch_size is not in the range (0, nsamples].
        """
        if batch_size <= 0 or batch_size > nsamples:
            raise ValueError("batch_size must be > 0 and <= nsamples")

        self.nsamples: int = nsamples
        self.batch_size: int = batch_size
        self.index: int = 0

        self.indices: list[int] = list(range(nsamples))
        random.shuffle(self.indices)

    def next_batch(self) -> list[int]:
        """Returns the next batch of shuffled indices.

        If the remaining indices are fewer than `batch_size`, the sampler
        reshuffles the entire list and starts from the beginning.

        Returns:
            list[int]: A list of size `batch_size` containing sample indices.
        """
        if self.index + self.batch_size > self.nsamples:
            random.shuffle(self.indices)
            self.index = 0

        batch = self.indices[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return batch
