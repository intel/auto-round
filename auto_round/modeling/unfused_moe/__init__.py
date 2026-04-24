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
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
                "auto_round.modeling.unfused_moe.qwen3_moe.LinearQwen3MoeSparseMoeBlock",
            )
        ],
    },
    "glm4_moe_lite": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite.Glm4MoeLiteMoE",
                "auto_round.modeling.unfused_moe.glm_moe_light.LinearGlm4MoeLiteMoE",
            )
        ],
    },
    "glm4_moe": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.glm4_moe.modeling_glm4_moe.Glm4MoeMoE",
                "auto_round.modeling.unfused_moe.glm_moe.LinearGlm4MoeMoE",
            )
        ],
    },
    "glm_moe_dsa": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaMoE",
                "auto_round.modeling.unfused_moe.glm_moe_dsa.LinearGlmMoeDsaMoE",
            )
        ],
    },
    "qwen3_next": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock",
                "auto_round.modeling.unfused_moe.qwen3_next.LinearQwen3NextSparseMoeBlock",
            )
        ],
    },
    "deepseek_v3": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3MoE",
                "auto_round.modeling.unfused_moe.deepseek_v3.LinearDeepseekV3MoE",
            )
        ],
    },
    "ernie4_5_moe": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "block_patch": [
            (
                "transformers.models.ernie4_5_moe.modeling_ernie4_5_moe.Ernie4_5_MoeSparseMoeBlock",
                "auto_round.modeling.unfused_moe.ernie4_5_moe.LinearErnie4_5_MoeSparseMoeBlock",
            )
        ],
    },
    "nemotron_h": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        "checkpoint_mapping": [],
        "preserve_upstream_conversion_mapping": True,
        "drop_conversion_target_patterns": [
            "mixer.experts.up_proj",
            "mixer.experts.down_proj",
            "embeddings.weight",
        ],
        "post_load_fn": "auto_round.modeling.unfused_moe.nemotron_h.apply_nemotron_h_post_load",
        "default_layer_config_patterns_fn": (
            "auto_round.modeling.unfused_moe.nemotron_h.nemotron_h_default_layer_config_patterns"
        ),
        "block_patch": [
            (
                "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMoE",
                "auto_round.modeling.unfused_moe.nemotron_h.LinearNemotronHMoE",
            )
        ],
        "dispatch_dict_patch": [
            (
                "transformers.models.nemotron_h.modeling_nemotron_h",
                "MIXER_TYPES",
                "moe",
                "auto_round.modeling.unfused_moe.nemotron_h.LinearNemotronHMoE",
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
        # Preserve upstream renames; filter out drop_conversion_target_patterns.
        if cfg.get("preserve_upstream_conversion_mapping"):
            try:
                upstream = conversion_mapping.orig_get_checkpoint_conversion_mapping(model_type)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    f"Failed to fetch upstream checkpoint conversion mapping for "
                    f"model_type={model_type!r}: {exc}. Falling back to AutoRound default mapping."
                )
                return cfg.get("checkpoint_mapping", [])
            drop_targets = cfg.get("drop_conversion_target_patterns", []) or []
            if not drop_targets:
                return upstream
            filtered = []
            for rule in upstream:
                target_patterns = getattr(rule, "target_patterns", []) or []
                if any(pat in target_patterns for pat in drop_targets):
                    continue
                filtered.append(rule)
            return filtered
        return cfg.get("checkpoint_mapping", [])

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
        from modelscope import snapshot_download  # pylint: disable=E0401

        # ModelSCOPE is different, it returns the folder path
        folder = snapshot_download(model_or_path, allow_patterns=[file_name])
        index_path = os.path.join(folder, file_name)
    else:
        index_path = None

    return index_path


def pre_check_config(model_name: str | torch.nn.Module, trust_remote_code: bool = True):
    if isinstance(model_name, str):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
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
    try:
        file_path = get_file_path_via_model_name(model_name, "model.safetensors.index.json")
        if os.path.exists(file_path):
            import json

            with open(file_path, "r") as f:
                index_data = json.load(f)
            model_keys = list(index_data.get("weight_map", {}).keys())
            for key in model_keys:
                if "gate_up_proj" in key:
                    return False
    except:
        return True
    return True


# This is for model checkpoint with linear definition
def apply_model_monkey_patches(model_name: str, trust_remote_code: bool = True) -> bool:
    res = pre_check_config(model_name, trust_remote_code=trust_remote_code)
    if not res:
        return False
    # patch blocks
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type")

    cfg = MODEL_CONFIG[model_type]
    for module_path, dict_name, dict_key, replacement_path in cfg.get("dispatch_dict_patch", []):
        try:
            target_module = importlib.import_module(module_path)
            dispatch = getattr(target_module, dict_name, None)
            if not isinstance(dispatch, dict) or dict_key not in dispatch:
                logger.warning(f"dispatch_dict_patch skipped: {module_path}.{dict_name}[{dict_key!r}] not found")
                continue
            replacement_module_path, replacement_class_name = replacement_path.rsplit(".", 1)
            replacement_module = importlib.import_module(replacement_module_path)
            dispatch[dict_key] = getattr(replacement_module, replacement_class_name)
            logger.info(f"Patched dispatch dict: {module_path}.{dict_name}[{dict_key!r}] -> {replacement_path}")
        except Exception as e:
            logger.warning(f"Failed to patch dispatch dict {module_path}.{dict_name}[{dict_key!r}]: {e}")

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


def _resolve_dotted(path: str):
    """Import ``module.attr`` given ``'module.submodule.attr'``."""
    module_path, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _resolve_registered_fn(model, cfg_key, caller):
    """Resolve (model_type, fn_path, callable) for a registered function on model_type.

    Returns (None, None, None) if no config, no model_type, no cfg entry, or no
    registered path. Emits a warning tagged with ``caller`` if resolution fails.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None, None, None
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return None, None, None
    cfg = MODEL_CONFIG.get(model_type)
    if not cfg:
        return None, None, None
    fn_path = cfg.get(cfg_key)
    if not fn_path:
        return None, None, None
    try:
        fn = _resolve_dotted(fn_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "%s: failed to resolve %r for model_type=%r: %s",
            caller,
            fn_path,
            model_type,
            exc,
        )
        return None, None, None
    return model_type, fn_path, fn


def apply_post_load_fixups(model: torch.nn.Module, **overrides) -> dict:
    """Invoke the registered ``post_load_fn`` for model_type, or no-op."""
    _, fn_path, fn = _resolve_registered_fn(model, "post_load_fn", "apply_post_load_fixups")
    if fn is None:
        return {}
    try:
        return fn(model, **overrides) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "apply_post_load_fixups: %s raised %s — continuing without post-load patches.",
            fn_path,
            exc,
        )
        return {}


def get_default_layer_config_patterns(model: torch.nn.Module) -> dict:
    """Return registered default layer_config patterns for model_type."""
    _, fn_path, fn = _resolve_registered_fn(
        model, "default_layer_config_patterns_fn", "get_default_layer_config_patterns"
    )
    if fn is None:
        return {}
    try:
        patterns = fn() or {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "get_default_layer_config_patterns: %s raised %s — skipping defaults.",
            fn_path,
            exc,
        )
        return {}
    if not isinstance(patterns, dict):
        logger.warning(
            "get_default_layer_config_patterns: %s returned %s, expected dict",
            fn_path,
            type(patterns).__name__,
        )
        return {}
    return patterns


def apply_modeling_patch(model: torch.nn.Module) -> bool:
    res = pre_check_config(model)
    if not res:
        return False
    model_type = getattr(model.config, "model_type")
    cfg = MODEL_CONFIG[model_type]
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
            logger.info(f"Patched module: {orig_path} -> {custom_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to patch {orig_path}: {e}")
            return False
    return False
