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
import transformers
import importlib
from packaging import version
from auto_round.logger import logger

MODEL_CONFIG = {
    "qwen3_moe": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": None,
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
                "auto_round.modeling.qwen3_moe.Qwen3MoeSparseMoeBlock"
            )
        ]
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

def apply_model_patches(model_type):
    if model_type not in MODEL_CONFIG:
        return

    cfg = MODEL_CONFIG[model_type]

    # 版本检查
    min_ver = cfg.get("min_transformers_version")
    max_ver = cfg.get("max_transformers_version")
    tf_ver = version.parse(transformers.__version__)
    if min_ver and tf_ver < version.parse(min_ver):
        return
    if max_ver and tf_ver > version.parse(max_ver):
        return

    # patch blocks
    for orig_path, custom_path in cfg.get("block_patch", []):
        orig_module_path, orig_class_name = orig_path.rsplit(".", 1)
        custom_module_path, custom_class_name = custom_path.rsplit(".", 1)

        try:
            orig_module = importlib.import_module(orig_module_path)
            custom_module = importlib.import_module(custom_module_path)
            custom_class = getattr(custom_module, custom_class_name)
            setattr(orig_module, orig_class_name, custom_class)
            logger.info(f"Patched {orig_path} -> {custom_path}")
            from transformers import conversion_mapping

            if not hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping"):
                conversion_mapping.orig_get_checkpoint_conversion_mapping = conversion_mapping.get_checkpoint_conversion_mapping

            conversion_mapping.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping_ar
            transformers.modeling_utils.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping_ar

        except Exception as e:
            logger.warning(f"Failed to patch {orig_path}: {e}")

#TODO change on demand
for key in MODEL_CONFIG.keys():
    apply_model_patches(key)
