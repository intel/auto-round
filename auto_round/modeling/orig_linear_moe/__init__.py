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
from auto_round import envs
import transformers
import importlib
from packaging import version
from transformers import AutoConfig

from auto_round.logger import logger
import torch

MODEL_CONFIG = {
    "qwen3_moe": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": None,
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
                "auto_round.modeling.orig_linear_moe.qwen3_moe.LinearQwen3MoeSparseMoeBlock",
            )
        ],
    },
}


def get_checkpoint_conversion_mapping_ar(model_type):
    from transformers import conversion_mapping

    if not hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping"):
        conversion_mapping.orig_get_checkpoint_conversion_mapping = conversion_mapping.get_checkpoint_conversion_mapping

    cfg = MODEL_CONFIG.get(model_type)
    if cfg:
        return cfg.get("checkpoint_mapping", [])

    from transformers import conversion_mapping

    return conversion_mapping.orig_get_checkpoint_conversion_mapping(model_type)

def get_file_path_via_model_name(model_or_path: str, file_name):
    from huggingface_hub import hf_hub_download
    # 1) local folder
    if os.path.isdir(model_or_path):
        index_path = os.path.join(model_or_path, file_name)

    # 2) HF model name
    elif not envs.AR_USE_MODELSCOPE:
        index_path = hf_hub_download(
            repo_id=model_or_path,
            filename=file_name,
            repo_type="model",
        )
    elif envs.AR_USE_MODELSCOPE:
        index_path = None # TODO

    return index_path

def pre_check_config(model_name):
    if isinstance(model_name,str):
        config = AutoConfig.from_pretrained(model_name)
    elif isinstance(model_name, torch.nn.Module):
        config = getattr(model_name, "config", None)
        if config is None:
            return False

    model_type = getattr(config, "model_type", None)
    if model_type is None or model_type not in MODEL_CONFIG:
        return False

    cfg = MODEL_CONFIG[model_type]

    min_ver = cfg.get("min_transformers_version")
    max_ver = cfg.get("max_transformers_version")
    tf_ver = version.parse(transformers.__version__)
    if min_ver and tf_ver < version.parse(min_ver):
        return False
    if max_ver and tf_ver > version.parse(max_ver):
        return False
        # Check keys
    try:
        file_path = get_file_path_via_model_name(model_name,"model.safetensors.index.json")
        if os.path.exists(file_path):
            import json

            with open(file_path, "r") as f:
                index_data = json.load(f)
            model_keys = set(index_data.get("weight_map", {}).keys())
            for key in model_keys:
                if "gate_up_proj" in key:
                    return False

    except:
        return True
    return True

# This is for model checkpoint with linear definition
def apply_model_monkey_patches(model_name:str) -> bool:
    res = pre_check_config(model_name)
    if not res:
        return False
    # patch blocks
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type")

    cfg = MODEL_CONFIG[model_type]
    for orig_path, custom_path in cfg.get("block_patch", []):
        orig_module_path, orig_class_name = orig_path.rsplit(".", 1)
        custom_module_path, custom_class_name = custom_path.rsplit(".", 1)

        try:
            orig_module = importlib.import_module(orig_module_path)
            custom_module = importlib.import_module(custom_module_path)
            custom_class = getattr(custom_module, custom_class_name)
            setattr(orig_module, orig_class_name, custom_class)

            if version.parse(transformers.__version__) >= version.parse("5.0.0"):
                from transformers import conversion_mapping

                if not hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping"):
                    conversion_mapping.orig_get_checkpoint_conversion_mapping = (
                        conversion_mapping.get_checkpoint_conversion_mapping
                    )

                conversion_mapping.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping_ar
                transformers.modeling_utils.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping_ar
            logger.info(f"Patched {orig_path} -> {custom_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to patch {orig_path}: {e}")
            return False
    return False


def apply_modeling_patch(model: torch.nn.Module) -> bool:
    if hasattr(model, "config"):
        model_type = model.config.model_type
    else:
        return False
    if model_type not in MODEL_CONFIG:
        return False

    cfg = MODEL_CONFIG[model_type]


    min_ver = cfg.get("min_transformers_version")
    max_ver = cfg.get("max_transformers_version")
    tf_ver = version.parse(transformers.__version__)
    if min_ver and tf_ver < version.parse(min_ver):
        return False
    if max_ver and tf_ver > version.parse(max_ver):
        return False

    # patch blocks
    for orig_path, custom_path in cfg.get("block_patch", []):
        orig_module_path, orig_class_name = orig_path.rsplit(".", 1)
        custom_module_path, custom_class_name = custom_path.rsplit(".", 1)

        try:
            orig_module = importlib.import_module(orig_module_path)
            custom_module = importlib.import_module(custom_module_path)
            custom_class = getattr(custom_module, custom_class_name)
            orig_class = getattr(orig_module, orig_class_name)
            names = []
            for n, m in model.named_modules():
                if isinstance(m, orig_class):
                    names.append(n)
            for n in names:
                model.set_submodule(n, custom_class(model.config), True)
            logger.info(f"Patched module {n} : {orig_path} -> {custom_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to patch {orig_path}: {e}")
            return False
    return False
