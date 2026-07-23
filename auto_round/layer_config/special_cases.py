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

import torch

from auto_round.logger import logger
from auto_round.schemes import is_mx_fp, is_nv_fp
from auto_round.utils import check_to_quantized, compress_layer_names, get_layer_names_in_block


def apply_layer_config_special_cases(
    layer_config,
    model,
    default_dict,
    supported_types,
    inner_supported_types,
    quant_block_list,
    quant_lm_head,
    gguf_name,
) -> tuple[dict, bool, str | None, bool]:
    """Steps 7-9 of layer-config resolution: lm_head handling, shape-divisibility
    enforcement (int weight-only + mxfp/nvfp), and block-membership marking."""
    from auto_round.utils.model import get_lm_head_name

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
                    # logger.warning_once(f"{n} skipped quantization (shape not divisible by 32).")
    # enforce shape divisibility for mxfp/nvfp
    if (is_nv_fp(default_dict["data_type"]) or is_mx_fp(default_dict["data_type"])) and not gguf_name:
        skipped_layers = []
        for n, m in model.named_modules():
            if type(m) in supported_types or m.__class__.__name__ in inner_supported_types:
                if m.weight.shape[1] % default_dict["group_size"]:
                    layer_config.setdefault(n, copy.deepcopy(default_dict))
                    layer_config[n].update(
                        {"bits": 16, "data_type": "fp", "act_bits": 16, "act_data_type": "fp", "fixed_by_user": True}
                    )
                    skipped_layers.append(n)

        compressed_skipped_layers = compress_layer_names(skipped_layers)
        if compressed_skipped_layers:
            logger.warning_once(
                f"some layers are skipped quantization (shape not divisible by {default_dict['group_size']}): "
                f"{compressed_skipped_layers}"
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

    return layer_config, has_qlayer_outside_block, lm_head_name, tie_word_embeddings
