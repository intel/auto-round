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
import re
import sys
from dataclasses import asdict, fields
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import transformers
from torch.amp import autocast

from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, GGUF_CONFIG, GGUF_INNER_CONFIG, QK_K, ModelType
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, get_gguf_scheme, preset_name_to_scheme


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
    from auto_round.utils.model_utils import to_device

    if input_ids.device != device:
        input_ids = to_device(input_ids, device)
        input_others = to_device(input_others, device)
    input_tuple = input_others.pop("positional_inputs", None)
    if "alibi" in input_others.keys() and input_others["alibi"] is not None:
        alibi = input_others["alibi"]
        input_others["alibi"] = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
    if amp:
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype):  # pragma: no cover
            output = block(input_ids, *input_tuple, **input_others)
    else:
        output = block(input_ids, *input_tuple, **input_others)
    if isinstance(output_return_id, int) and (isinstance(output, list) or isinstance(output, tuple)):
        output = output[output_return_id]
    return output


def collect_best_params(block):
    params = {}
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            params[n] = {}
            for key in m.params.keys():
                params[n][key] = copy.deepcopy(m.params[key].data)
    return params


def infer_bits_by_data_type(data_type: str):
    """Infer bits by data_type

    Args:
        data_type (str): data_type

    Returns:
        int: bits inferred by data_type, None means cannot infer correct bits by data_type
    """
    from auto_round.utils.constants import SUPPORTED_DTYPES

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


def check_to_quantized(config):
    """Checks if the configuration is valid for quantization.

    Args:
        config (dict or object): The configuration to check. It can be either a
            dictionary with a 'bits' key or an object with a 'bits' attribute.

    Returns:
        bool: True if the configuration is valid for quantization (bits <= 8),
            False otherwise.
    """

    if isinstance(config, (dict, QuantizationScheme)):
        bits = int(config.get("bits", 16))
        act_bits = int(config.get("act_bits", 16))
    elif hasattr(config, "orig_layer"):
        bits = int(config.orig_layer.bits) if hasattr(config.orig_layer, "bits") else 16
        act_bits = int(config.orig_layer.act_bits) if hasattr(config.orig_layer, "act_bits") else 16
    else:
        bits = int(config.bits) if hasattr(config, "bits") else 16
        act_bits = int(config.act_bits) if hasattr(config, "act_bits") else 16

    return bits <= 8 or act_bits <= 8


def set_layer_config(
    model: torch.nn.Module,
    layer_config: dict[str, Union[str, dict, "QuantizationScheme"]],
    default_scheme: Union[str, "QuantizationScheme"],
    default_scale_dtype: torch.dtype | str,
    supported_types: tuple,
    inner_supported_types: tuple,
    quant_block_list=None,
    fp_layers: str = "",
    quant_lm_head: bool = False,
    enable_gguf_official_mixed: bool = True,
    is_mllm: bool = False,
) -> tuple[dict, bool, dict]:
    """
    Normalize, validate, and expand layer-specific quantization configs.
    Returns (final_layer_config, has_quant_layer_outside_block)
    """

    from auto_round.schemes import get_gguf_scheme
    from auto_round.utils.dtype_utils import is_mx_fp, is_nv_fp
    from auto_round.utils.model_utils import get_layer_names_in_block, get_lm_head_name, get_module

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
    scheme_keys = tuple(f.name for f in fields(QuantizationScheme)) + ("scale_dtype",)
    layer_config = copy.deepcopy(layer_config) or {}

    # 1. fp_layers -> force 16
    for name in get_fp_layer_names(model, fp_layers):
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
    for cfg in layer_config.values():
        for key in scheme_keys:
            cfg.setdefault(key, copy.deepcopy(default_dict.get(key)))

    # 5. collect supported modules
    gguf_name = get_gguf_scheme(default_scheme)
    if gguf_name and torch.nn.Embedding not in supported_types:
        supported_types = (*supported_types, torch.nn.Embedding)

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
        if isinstance(m, torch.nn.Embedding):
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

    if quant_lm_head and tie_word_embeddings:
        quant_lm_head = False
        logger.warning(
            "reset `quant_lm_head` to false as quantizing " "lm_head with tied weights has not been supported currently"
        )

    if lm_head_name not in layer_config and quant_lm_head:
        layer_config[lm_head_name] = copy.deepcopy(default_dict)

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


##https://github.com/ggml-org/llama.cpp/blob/9e31bec4fd53634c9e5b04650488a09a055f5dab/src/llama-quant.cpp#L129
def get_layer_config_by_gguf_format(layer_config, target_gguf_format: str, model, model_type=ModelType.TEXT):
    # # TODO: support for other format later
    # target_gguf_format = next((fmt for fmt in gguf_format if fmt != "fake"), None)
    import gguf  # pylint: disable=E0401

    from auto_round.utils.misc_utils import LazyImport
    from auto_round.utils.model_utils import _get_digital_in_layer_name, get_lm_head_name, get_module

    # from auto_round.export.export_to_gguf.convert import ModelBase, get_model_architecture
    convert_hf_to_gguf = LazyImport("auto_round.export.export_to_gguf.convert_hf_to_gguf")

    model_architecture = convert_hf_to_gguf.get_model_architecture(
        hparams=model.config.to_dict(), model_type=model_type
    )
    try:
        model_class = convert_hf_to_gguf.ModelBase.from_model_architecture(model_architecture, model_type=model_type)
    except NotImplementedError:
        return layer_config, {}

    n_layer = None
    for name in ["n_layers", "num_hidden_layers", "n_layer", "num_layers"]:
        sub_attr = "text_config" if model_type == ModelType.TEXT else "vision_config"
        if hasattr(model.config, name):
            n_layer = getattr(model.config, name)
            break
        if hasattr(model.config, sub_attr):
            if hasattr(getattr(model.config, sub_attr), name):
                n_layer = getattr(getattr(model.config, sub_attr), name)
                break
    if n_layer is None:
        return layer_config, {}

    tensor_map = gguf.get_tensor_name_map(model_class.model_arch, n_layer)

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

        gguf_name = tensor_map.get_name(layer_name)
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
                    if config.get("super_group_size", None) is not None:
                        new_type = new_type[:bits_index] + str(config["bits"]) + "_k"
                    if config.get("super_group_size", None) is None or new_type not in GGUF_INNER_CONFIG:
                        prefix_idx = 0 if config.get("sym", True) else 1
                        new_type = new_type[:bits_index] + str(config["bits"]) + f"_{prefix_idx}"
                        if new_type not in GGUF_INNER_CONFIG:
                            new_type = new_type[:bits_index] + str(config["bits"]) + f"_{1-prefix_idx}"
                    if new_type not in GGUF_INNER_CONFIG:
                        raise ValueError(
                            f"the setting in layer_config {layer_name} "
                            f"could not match any supported gguf format, please have a check."
                        )
                    else:
                        logger.warning_once(
                            f"the setting in layer_config {layer_name} "
                            f"could not match any supported gguf format, reset to {new_type}"
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
            pass
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


def check_awq_gemm_compatibility(model, bits, group_size, sym, layer_configs=None):
    """Checks if a model is compatible with the AutoAWQ GEMM kernel.

    Args:
        model: The model object to evaluate, typically a PyTorch model.
        bits (int): The number of bits for quantization (must be 4 for compatibility).
        group_size (int): The group size for quantization.
        sym (bool): Whether symmetric quantization is used (not utilized in the current function logic).
        layer_configs (dict, optional): A dictionary mapping layer names to configurations, where each
            configuration can specify a custom number of bits for the layer.

    Returns:
        tuple: A tuple containing:
            - bool: `True` if the model is compatible, `False` otherwise.
            - str: An error message describing why the model is incompatible, or an empty string if compatible.
    """
    from auto_round.utils.model_utils import get_layer_names_in_block, get_module

    if bits != 4:
        return False, "AutoAWQ GEMM kernel only supports 4 bits"
    for n, m in model.named_modules():
        if type(m) == transformers.pytorch_utils.Conv1D:
            return False, "AutoAWQ GEMM kernel does not support conv1d"

    layer_names = get_layer_names_in_block(model)
    for layer_name in layer_names:
        if (
            layer_configs is not None
            and layer_name in layer_configs.keys()
            and layer_configs[layer_name].get("bits", bits) > 8
        ):
            continue

        layer = get_module(model, layer_name)
        if layer.in_features % group_size != 0:
            return False, f"Layer {layer_name} in_features is not multiple of group_size {group_size}"
        if layer.out_features % (32 // bits) != 0:
            return False, f"Layer {layer_name} out_features is not multiple of 32 // bits"

    return True, ""


def check_need_act_calibration(
    is_act_dynamic: Union[bool, None], act_data_type: Union[str, None] = None, act_bits: Union[int, None] = 16
) -> bool:
    if act_bits is None or act_bits > 8:
        return False
    # None is dynamic
    if is_act_dynamic is not None and not is_act_dynamic:
        return True
    if act_data_type is not None and "static" in act_data_type:
        return True
    return False


def is_autoround_exllamav2_available():
    """Checks if the AutoRound ExLlamaV2 kernels are available.

    Returns:
        bool:
            True if the AutoRound ExLlamaV2 kernels are available, False otherwise.
    """
    res = True
    try:
        from autoround_exllamav2_kernels import gemm_half_q_half, make_q_matrix
    except ImportError as e:
        res = False
    return res


def get_autogptq_packing_qlinear(backend, bits=4, group_size=128, sym=False):
    """
    Configures and returns a QuantLinear class based on the specified backend and parameters.

    Args:
        backend (str): The backend to be used for quantization. Supported values include "qigen", "triton", "marlin",
                       "exllama", and "cuda".
        bits (int, optional): The number of bits for quantization. Default is 4.
        group_size (int, optional): The group size for quantization. Default is 128.
        sym (bool, optional): Flag indicating whether to use symmetric quantization. Default is False.

    Returns:
        class: The dynamically imported QuantLinear class configured according to the specified parameters.
    """
    use_triton = True
    if bits not in [2, 4, 8]:
        use_triton = False
    disable_exllamav2 = True
    disable_exllamav1 = False
    disable_marlin = True
    use_qigen = False
    if "qigen" in backend:
        use_triton = False
        use_qigen = True
    elif "triton" in backend:
        use_triton = True
    elif "marlin" in backend and sym:
        use_triton = False
        disable_marlin = False
    elif "exllama" in backend:  ##need v1 code to export
        use_triton = True  ##same with triton
        disable_marlin = True
    elif "cuda" in backend:
        use_triton = False
        disable_marlin = True
        disable_exllamav2 = True
        disable_exllamav1 = True
    if use_triton:
        from auto_round.export.export_to_autogptq.qlinear_triton import QuantLinear

        return QuantLinear
    try:
        import auto_gptq  # pylint: disable=E0401
    except:
        logger.error(f"please install auto_gptq via 'pip install auto-gptq' to support exporting to {backend}")
        exit()

    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear  # pylint: disable=E0401

    from auto_round.utils.misc_utils import get_library_version

    version = get_library_version("auto_gptq")
    from packaging.version import Version

    if Version(version) < Version("0.7.2"):
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=use_triton,
            desc_act=False,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllamav1,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen,
            disable_marlin=disable_marlin,
        )
    else:
        QuantLinear = dynamically_import_QuantLinear(  # pylint: disable=E1123
            use_triton=use_triton,
            desc_act=False,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllamav1,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen,
            use_marlin=not disable_marlin,
        )
    return QuantLinear


def get_reciprocal(tensor):
    if torch.dtype is torch.float16:
        tensor = torch.sign(tensor) * torch.clamp(torch.abs(tensor), min=1e-5)
    else:
        tensor = torch.where(torch.abs(tensor) < 1e-30, 0, tensor)
    return torch.where(tensor != 0, 1 / tensor, torch.zeros_like(tensor))


def check_seqlen_compatible(input_seqlen, tokenizer=None, model=None):
    """
    Check whether the input sequence length is within the limits defined
    by the tokenizer and the model configuration.

    Args:
        input_seqlen (int): The length of the input sequence.
        tokenizer: Optional, a HuggingFace tokenizer object.
        model: Optional, a HuggingFace model object.

    Returns:
        ValueError: if the input length is not valid, riase Error.
    """
    if model is not None and hasattr(model, "config"):
        model_config = model.config
        if hasattr(model_config, "max_position_embeddings") and input_seqlen > model_config.max_position_embeddings:
            raise ValueError(
                f"seqlen({input_seqlen}) exceeds model.config.max_position_embeddings("
                f"{model_config.max_position_embeddings}). Please lowering '--seqlen'"
            )
    if tokenizer is not None and hasattr(tokenizer, "model_max_length") and input_seqlen > tokenizer.model_max_length:
        raise ValueError(
            f"seqlen({input_seqlen}) exceeds tokenizer.model_max_length({tokenizer.model_max_length}). "
            "Please oncider Consider lowering the '--seqlen' or increasing tokenizer.model_max_length."
        )


def filter_quantization_config(quantization_config):
    default_dict = {
        "amp": True,
        "batch_size": 8,
        "data_type": int,
        "dataset": "NeelNanda/pile-10k",
        "enable_minmax_tuning": True,
        "enable_norm_bias_tuning": False,
        "enable_quanted_input": True,
        "gradient_accumulate_steps": 1,
        "iters": 200,
        "low_gpu_mem_usage": False,
        "nsamples": 128,
        "scale_dtype": "torch.float16",
        "seqlen": 2048,
    }
    iters = quantization_config.get("iters", 200)

    default_dict["lr"] = 1.0 / iters if iters > 0 else 5e-3
    default_dict["minmax_lr"] = default_dict["lr"]

    for key in default_dict:
        if key in quantization_config and default_dict[key] == quantization_config[key]:
            quantization_config.pop(key)
    for k in list(quantization_config.keys()):
        if quantization_config[k] is None:
            quantization_config.pop(k)

    if quantization_config.get("act_bits", 16) >= 16:
        quantization_config.pop("act_bits", None)
        quantization_config.pop("act_data_type", None)
        quantization_config.pop("act_dynamic", None)
        quantization_config.pop("act_sym", None)
        quantization_config.pop("act_group_size", None)


def get_fp_layer_names(model: torch.nn.Module, fp_layers: str):
    """Identifies and returns layers in the model to exclude from quantization.

    This function processes a comma-separated list of fully precision (FP) layers,
    matches them to the names of layers in the model, and returns a list of such
    layers to exclude from quantization.

    Args:
        model (torch.nn.Module): The model whose layers will be inspected.
        fp_layers (str): A comma-separated string of layer names to be excluded
            from quantization. Whitespace is ignored in this string.

    Returns:
        list: A list of layer names that match the specified FP layers or are
        subcomponents of those layers.
    """
    from auto_round.utils.constants import SUPPORTED_LAYER_TYPES

    if not fp_layers:
        return []
    fp_layers = fp_layers.replace(" ", "").split(",")
    all_layer_names = []
    for n, m in model.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:
            all_layer_names.append(n)
    not_to_quantized_layers = []

    for fp_layer in fp_layers:
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


def _gguf_type_fallback(gguf_type: str) -> str:
    gguf_type = gguf_type.lower()
    if gguf_type in ("gguf:q2_k", "gguf:q3_k", "gguf:q4_k"):
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q5_k":
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q6_k":
        gguf_type = "gguf:q8_0"
    return gguf_type


def gguf_args_check(args_or_ar, formats: list[str] = None, model_type=ModelType.TEXT):
    import argparse

    from auto_round.export.export_to_gguf.convert import download_convert_file
    from auto_round.logger import logger
    from auto_round.utils.model_utils import download_hf_model, get_gguf_architecture

    formats = sorted(formats, key=lambda x: len(x))
    export_gguf = False
    for f in formats:
        if f.startswith("gguf"):
            export_gguf = True

        if f.startswith("gguf") and f not in GGUF_CONFIG:
            logger.error(f"{f} is not supported, please check.")

    redownload = False
    if export_gguf:
        try:
            from auto_round.export.export_to_gguf.convert_hf_to_gguf import (  # pylint: disable=E0401
                ModelBase,
                ModelType,
                get_model_architecture,
            )

            if isinstance(args_or_ar.model, str):
                model_path = args_or_ar.model
            else:
                model_path = args_or_ar.model.name_or_path
            if not os.path.isdir(model_path):
                model_path = download_hf_model(model_path)
            model_architecture = get_gguf_architecture(model_path, model_type=ModelType.TEXT)
            if model_architecture not in ModelBase._model_classes[ModelType.TEXT]:
                logger.warning(
                    f"Current version of gguf export does not support for {model_architecture},"
                    " will re-download dependency file."
                )
                redownload = True
        except ModuleNotFoundError as e:
            if "convert_hf_to_gguf" in str(e):
                logger.warning("GGUF export dependency file is not found, download from github.")
                redownload = True
        except AttributeError as e:
            raise ImportError(
                "Please use the latest gguf-py, you can use the following command to install it:\n"
                "git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py && pip install ."
            )
        download_convert_file(redownload)

        try:
            from auto_round.export.export_to_gguf.convert_hf_to_gguf import (  # pylint: disable=E0401
                ModelBase,
                ModelType,
            )
        except ImportError as e:
            raise ImportError(
                "Please use the latest gguf-py, you can use the following command to install it:\n"
                "git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py && pip install ."
            )
        if isinstance(args_or_ar.model, str):
            model_path = args_or_ar.model
        else:
            model_path = args_or_ar.model.name_or_path
        if not os.path.isdir(model_path):
            model_path = download_hf_model(model_path)
        model_architecture = get_gguf_architecture(model_path, model_type=ModelType.TEXT)
        if model_architecture not in ModelBase._model_classes[ModelType.TEXT]:
            logger.error(f"Model {model_architecture} is not supported to export gguf format.")
            sys.exit(1)

    pattern = re.compile(r"q\d_k")
    pre_dq_format = ""
    unsupported_list, reset_list = [], []
    for format in GGUF_CONFIG:
        if format in formats:
            if format == "q6_k_s":
                logger.warning("Please note that q6_k_s is q6_k.")

            if re.search(pattern, format):
                if pre_dq_format and re.search(pattern, format).group() not in pre_dq_format:
                    logger.error(f"Cannot export {pre_dq_format} and {format} at the same time.")
                    sys.exit(-1)
                else:
                    pre_dq_format = format

            unsupported_list, reset_list = [], []
            gguf_config = GGUF_CONFIG[format]
            for k, v in gguf_config.items():
                if not hasattr(args_or_ar, k):
                    continue
                if k == "data_type":
                    if re.search(r"q\d_1", format) and len(formats) > 1:
                        v = "int"
                if k == "sym" and isinstance(args_or_ar, argparse.Namespace):
                    k = "asym"
                    v = not v
                if getattr(args_or_ar, k) != v:
                    unsupported_list.append(f"{k}={getattr(args_or_ar, k)}")
                    reset_list.append(f"{k}={v}")
                    setattr(args_or_ar, k, v)
            if len(unsupported_list) > 0:
                logger.info(
                    f"format {format} does not support for {', '.join(unsupported_list)},"
                    f" reset to {', '.join(reset_list)}."
                )
    # Removed obsolete commented-out block for improved readability and maintainability.
    return args_or_ar


def get_layer_features(layer):
    """Extracts input and output feature dimensions for supported layers."""
    from auto_round.utils.constants import LinearAllreduce, LinearLayer, deepspeed_exists

    if type(layer) == torch.nn.Linear:
        return layer.in_features, layer.out_features
    elif type(layer) == transformers.pytorch_utils.Conv1D:  # TODO: Verify correctness
        return layer.weight.shape[0], layer.weight.shape[1]
    elif isinstance(layer, torch.nn.Embedding):
        return layer.num_embeddings, layer.embedding_dim
    elif deepspeed_exists and type(layer) in (LinearLayer, LinearAllreduce):
        return layer.weight.shape[1], layer.weight.shape[0]  # (input_dim, output_dim)
    elif "FP8Linear" in layer.__class__.__name__:
        return layer.in_features, layer.out_features
    return None, None  # Unsupported layer type


def get_common_prefix(paths):
    # Split each path into components and find the common prefix
    split_paths = [path.split(".") for path in paths]
    common_prefix = split_paths[0]
    for path in split_paths[1:]:
        common_prefix = [comp for comp, other in zip(common_prefix, path) if comp == other]
    return ".".join(common_prefix)


def extract_block_names_to_str(quant_block_list):
    if not isinstance(quant_block_list, (list, tuple)):
        return None
    # Extract common prefix for each list
    prefixes = [get_common_prefix(blocks) for blocks in quant_block_list]
    # Join prefixes into a single string
    return ",".join(prefixes)


def find_matching_blocks(model, all_blocks, to_quant_block_names):
    """
    Find and return matching blocks in the model based on to_quant_block_names.

    Args:
        model: The model (not used in this specific function but kept for completeness).
        all_blocks: List of lists, where each inner list contains full block names in the model.
        to_quant_block_names: Comma-separated string of target block names to match.

    Returns:
        target_blocks: List of lists containing full paths of matching blocks in the model.
    """
    if not to_quant_block_names:
        return all_blocks
    to_quant_block_list = to_quant_block_names
    if isinstance(to_quant_block_names, list) or isinstance(to_quant_block_names, tuple):
        return to_quant_block_names
    if isinstance(to_quant_block_names, str):
        to_quant_block_list = [name.strip() for name in to_quant_block_names.split(",")]
    target_blocks = []
    for block_list in all_blocks:
        matched_sublist = []
        for name in to_quant_block_list:
            matches = [block for block in block_list if re.search(name, block)]
            if matches:
                matched_sublist.extend(matches)
        if matched_sublist:
            target_blocks.append(matched_sublist)
    if not target_blocks:
        raise ValueError(
            "No block names matched. Please check the input for to_quant_block_name,"
            "or set to_quant_block_name to None to automatically match quantizable blocks."
        )
    return target_blocks


def get_scale_shape(weight, group_size):
    """Computes the shape of the scale tensor for quantization based on the weight tensor and group size.

    Args:
      weight (torch.Tensor): The weight tensor of the layer.
      group_size (int): The size of the groups for quantization.

    Returns:
      The shape of the scale tensor to be used for quantization.
    """
    if group_size == 0:
        return 1
    elif group_size == -1 or weight.shape[1] < group_size:
        shape = weight.shape[0]
    else:
        shape = weight.shape[0] * ((weight.shape[1] + group_size - 1) // group_size)

    return shape


def init_cache(positional_inputs, inputs):
    """
    Initializes special model inputs by adding positional inputs if missing.

    Args:
        positional_inputs (list): List of positional inputs to add to inputs.
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Adds "positional_inputs" key if not present.
    """
    from auto_round.utils.model_utils import to_device

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


def check_skippable_keywords(key):
    """
    Prints a reminder if a key is not stored during quantization fine-tuning.
    """
    skippable_cache_keys = ("past_key_value",)
    for cache_key in skippable_cache_keys:
        if cache_key not in key:
            return True
    return False
