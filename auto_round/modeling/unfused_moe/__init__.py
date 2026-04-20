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
    # Nemotron-H is a Mamba2 + Attention + MoE hybrid. ``NemotronHBlock``
    # resolves its mixer via a module-level dispatch dict
    # (``MIXER_TYPES``), so the standard setattr-based ``block_patch`` is
    # not sufficient — we also rewrite the dict entry, which is the
    # purpose of ``dispatch_dict_patch`` (see ``apply_model_monkey_patches``).
    "nemotron_h": {
        "min_transformers_version": "5.0.0",
        "max_transformers_version": "6.0.0",
        # Nemotron-H stores weights on disk under ``backbone.*`` and
        # ``embedding.weight``, but the in-memory ``NemotronHForCausalLM``
        # exposes them as ``model.*`` and ``embeddings.weight``.  The
        # upstream conversion mapping renames these on load.  Returning an
        # empty list here would silently drop those renames — every
        # ``backbone.*`` tensor would fail to match any module parameter
        # and the entire backbone (router, attention, Mamba2, MLP) would
        # remain at random init while quantization happily proceeded on
        # garbage.  ``preserve_upstream_conversion_mapping`` keeps the
        # upstream rules; ``drop_conversion_target_patterns`` removes only
        # the expert-bundling ``WeightConverter`` entries that fan
        # individual on-disk per-expert tensors back into a single
        # bundled Parameter (the very thing the unfused handler exists
        # to avoid).
        "checkpoint_mapping": [],
        "preserve_upstream_conversion_mapping": True,
        "drop_conversion_target_patterns": [
            "mixer.experts.up_proj",
            "mixer.experts.down_proj",
            # Upstream ships a legacy ``embedding.weight → embeddings.weight``
            # WeightRenaming that is a no-op at load time for the current
            # on-disk checkpoint (already plural), but is applied in reverse
            # at save time — turning the saved tensor into singular
            # ``backbone.embedding.weight``. That creates a name mismatch
            # against the source checkpoint's ``backbone.embeddings.weight``
            # which ``copy_missing_tensors_from_source`` then resolves by
            # RTN-quantizing the source embedding and emitting a phantom
            # ``backbone.embeddings.qweight`` alongside the unquantized BF16
            # copy. Dropping the rule keeps load/save symmetric (plural on
            # both sides) and prevents the double-store.
            "embeddings.weight",
        ],
        # Architecture-specific defaults applied automatically by
        # ``apply_post_load_fixups`` and ``set_layer_config`` — see
        # ``nemotron_h_setup.py`` for the rationale of each item.
        "post_load_fn": "auto_round.modeling.unfused_moe.nemotron_h_setup.apply_nemotron_h_post_load",
        "default_layer_config_patterns_fn": (
            "auto_round.modeling.unfused_moe.nemotron_h_setup.nemotron_h_default_layer_config_patterns"
        ),
        "block_patch": [
            (
                "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMoE",
                "auto_round.modeling.unfused_moe.nemotron_h.LinearNemotronHMoE",
            )
        ],
        # (module_path, dict_attribute_name, key_to_overwrite, replacement_class_path)
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
        # Some models (notably Nemotron-H) require the upstream rename rules
        # — e.g. ``backbone.*`` → ``model.*`` — to remain in effect or
        # nothing in the backbone will load and the in-memory parameters
        # silently keep their random init values.  When
        # ``preserve_upstream_conversion_mapping`` is set, fetch the
        # upstream mapping and only filter out the explicit
        # ``drop_conversion_target_patterns`` (used to suppress
        # expert-bundling rules that conflict with the unfused module
        # layout).
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
    # Some hybrid models (e.g. Nemotron-H) resolve their mixer/MoE block
    # via a module-level dispatch dict instead of a direct symbol
    # reference. ``setattr(orig_module, orig_class_name, custom_class)``
    # alone leaves the dict entry pointing at the original class. Apply
    # the configured dispatch-dict patches first so any subsequent model
    # load picks up the replacement.
    for module_path, dict_name, dict_key, replacement_path in cfg.get("dispatch_dict_patch", []):
        try:
            target_module = importlib.import_module(module_path)
            dispatch = getattr(target_module, dict_name, None)
            if not isinstance(dispatch, dict) or dict_key not in dispatch:
                logger.warning(
                    f"dispatch_dict_patch skipped: {module_path}.{dict_name}[{dict_key!r}] not found"
                )
                continue
            replacement_module_path, replacement_class_name = replacement_path.rsplit(".", 1)
            replacement_module = importlib.import_module(replacement_module_path)
            dispatch[dict_key] = getattr(replacement_module, replacement_class_name)
            logger.info(
                f"Patched dispatch dict: {module_path}.{dict_name}[{dict_key!r}] -> {replacement_path}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to patch dispatch dict {module_path}.{dict_name}[{dict_key!r}]: {e}"
            )

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


def apply_post_load_fixups(model: torch.nn.Module, **overrides) -> dict:
    """Apply architecture-specific post-load patches for *model*.

    Looks up ``model.config.model_type`` in :data:`MODEL_CONFIG` and, if
    a ``post_load_fn`` is registered there, invokes it with *model*
    plus any caller-supplied ``overrides`` (forwarded as kwargs).

    This is a no-op for models whose ``model_type`` is not registered
    in :data:`MODEL_CONFIG` or that don't ship a ``post_load_fn`` —
    making it safe to call unconditionally from the quantization
    pipeline.

    Typical post-load concerns (see e.g.
    ``nemotron_h_setup.apply_nemotron_h_post_load``) include setting
    attributes that HF ``low_cpu_mem_usage`` skipped via
    ``init_empty_weights``, or reloading specific tensors from the
    source checkpoint at higher precision.
    """

    config = getattr(model, "config", None)
    if config is None:
        return {}
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return {}
    cfg = MODEL_CONFIG.get(model_type)
    if not cfg:
        return {}
    fn_path = cfg.get("post_load_fn")
    if not fn_path:
        return {}
    try:
        fn = _resolve_dotted(fn_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "apply_post_load_fixups: failed to resolve %r for model_type=%r: %s",
            fn_path,
            model_type,
            exc,
        )
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
    """Return the architecture-specific default ``layer_config``
    patterns for *model* (a mapping of regex string → overlay dict),
    or an empty dict when none are registered.

    The patterns are merged into the user-supplied ``layer_config``
    by ``set_layer_config`` (per-pattern, user entries take
    precedence).  No-op for non-registered models.
    """

    config = getattr(model, "config", None)
    if config is None:
        return {}
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        return {}
    cfg = MODEL_CONFIG.get(model_type)
    if not cfg:
        return {}
    fn_path = cfg.get("default_layer_config_patterns_fn")
    if not fn_path:
        return {}
    try:
        fn = _resolve_dotted(fn_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "get_default_layer_config_patterns: failed to resolve %r: %s",
            fn_path,
            exc,
        )
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
