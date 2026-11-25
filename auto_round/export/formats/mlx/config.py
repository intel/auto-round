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

"""
MLX Format Exporter for AutoRound

Exports quantized models to MLX-compatible format that can be loaded
directly by mlx-lm for inference on Apple Silicon.

MLX QuantizedLinear dequantization (affine mode):
    w_float = scale * w_int + bias

where w_int is packed into uint32, each holding (32 // bits) elements,
and scale/bias (called "biases" in MLX) have shape [out_features, num_groups].

Supported schemes: W2A16, W3A16, W4A16, W5A16, W6A16, W8A16
"""

import copy
import json
import os
from collections.abc import Callable

import torch
import torch.nn as nn
import transformers

from auto_round.export.common.serialization import save_pretrained_artifact, unsupported_meta_device
from auto_round.export.formats.mlx.packing import _MLXPackedLayer
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    get_packing_device,
    set_module,
)
from auto_round.utils.common import MM_KEYS

SUPPORTED_LAYER_TYPES = [nn.Linear, nn.Conv2d, transformers.pytorch_utils.Conv1D]


def _is_mlx_quantizable(module: nn.Module, group_size: int) -> bool:
    """Predicate matching mlx-lm's default quantization predicate.

    A layer is considered quantizable by mlx-lm if its inner dim is a multiple
    of ``group_size`` and its outer dim is a multiple of 64. Embeddings are also
    eligible. Layers that fail this predicate are skipped at load time, so we do
    NOT need to emit ``false`` entries for them.
    """
    if isinstance(module, nn.Linear):
        in_dim, out_dim = module.in_features, module.out_features
    elif isinstance(module, nn.Embedding):
        in_dim, out_dim = module.embedding_dim, module.num_embeddings
    else:
        return False
    return (in_dim % group_size == 0) and (out_dim % 64 == 0)


# Sub-modules whose layers we never want to auto-quantize on the mlx side.
# Vision / audio towers in VLMs are conventionally kept in fp16 by mlx-vlm,
# so emitting per-layer ``false`` for hundreds of vision linears just bloats
# the config. We simply skip these subtrees.
_DEFAULT_SKIP_PREFIXES = tuple(MM_KEYS)


def _build_mlx_quantization_config(
    model: nn.Module,
    default_bits: int,
    default_group_size: int,
    skip_prefixes: tuple = _DEFAULT_SKIP_PREFIXES,
) -> dict:
    """Build the ``quantization`` dict for an MLX ``config.json``.

    Mirrors the mlx-community mixed-bit format (see e.g.
    ``mlx-community/GLM-4.7-REAP-50-mixed-3-4-bits``)::

        "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 3},
            "model.layers.0.mlp.down_proj": false,
            ...
        }

    * Top-level ``group_size`` / ``bits`` give the default scheme.
    * Per-layer ``{group_size, bits}`` overrides differing layers.
    * Per-layer ``false`` marks an mlx-quantizable layer that should be kept
      in fp16 (e.g. ``lm_head`` left at 16 bits in our ``layer_config``).

    For VLMs, ``skip_prefixes`` lists sub-trees (vision/audio towers, projectors)
    whose un-quantized layers we do NOT emit ``false`` entries for. Quantized
    layers inside those sub-trees are still recorded faithfully.

    Note: AutoRound may leave non-quantized layers wrapped in ``WrapperLinear``
    (``.orig_layer`` points to the real ``nn.Linear``); we unwrap before testing
    so that e.g. ``lm_head`` with ``bits=16`` is correctly emitted as ``false``.
    """
    quant_cfg: dict = {"group_size": default_group_size, "bits": default_bits}

    def _is_skipped(layer_name: str) -> bool:
        return any(layer_name == p or layer_name.startswith(p + ".") for p in skip_prefixes)

    for name, module in model.named_modules():
        if isinstance(module, _MLXPackedLayer):
            # Always record packed layers, even if inside a skipped sub-tree
            # (user explicitly asked for them to be quantized).
            quant_cfg[name] = {
                "group_size": int(module.group_size) if module.group_size is not None else default_group_size,
                "bits": int(module.bits) if module.bits is not None else default_bits,
            }
            continue

        if _is_skipped(name):
            continue

        # Unwrap auto-round WrapperLinear (and similar) before the predicate test.
        target = module
        while hasattr(target, "orig_layer") and target.orig_layer is not None:
            target = target.orig_layer

        if _is_mlx_quantizable(target, default_group_size):
            # mlx-lm would auto-quantize this layer using the default scheme,
            # but since we deliberately left it un-quantized, mark it explicitly.
            quant_cfg[name] = False
    return quant_cfg


# Sub-config keys commonly used by HF multi-modal / dual-tower configs that
# carry their own ``rope_parameters`` and may want a ``quantization`` block.
_TEXT_SUB_CONFIG_KEYS = ("text_config", "language_config", "thinker_config")


def _flatten_rope_parameters_recursive(cfg: dict) -> None:
    """Pop ``rope_parameters`` and surface its keys to the same level.

    mlx-lm / mlx-vlm both expect ``rope_theta`` (and friends) at the same level
    as the rest of the model config — i.e. on the top-level config for LLMs and
    on ``text_config`` for VLMs. We walk every nested dict so both cases work.

    Two ``rope_parameters`` shapes appear in the wild and both are handled:
      1. ``{"rope_theta": 1e6, "rope_type": "default", ...}`` (flat)
      2. ``{"default": {"rope_theta": 1e6, "rope_type": "default", ...}, ...}``
         (newer HF "by-mode" layout — we surface keys from the ``"default"``
         entry, falling back to the first sub-dict if absent).
    """
    if not isinstance(cfg, dict):
        return
    rope_params = cfg.pop("rope_parameters", None)
    if isinstance(rope_params, dict):
        # If the dict is nested by mode, prefer "default", else any sub-dict.
        sub_dicts = {k: v for k, v in rope_params.items() if isinstance(v, dict)}
        if sub_dicts:
            chosen = sub_dicts.get("default") or next(iter(sub_dicts.values()))
            for k, v in chosen.items():
                cfg.setdefault(k, v)
        else:
            for k, v in rope_params.items():
                cfg.setdefault(k, v)
    for v in cfg.values():
        if isinstance(v, dict):
            _flatten_rope_parameters_recursive(v)


def _extract_rope_theta_from_obj(config_obj) -> "float | None":
    """Best-effort extraction of ``rope_theta`` from a HF config object.

    Newer transformers versions stop exposing ``rope_theta`` as a direct
    attribute on the config and only expose ``rope_parameters`` (which can be
    a dict, a typed object, or a by-mode mapping like
    ``{"default": <obj-or-dict>}``). This helper tries every shape we've
    encountered in the wild.
    """
    if config_obj is None:
        return None
    direct = getattr(config_obj, "rope_theta", None)
    if direct is not None:
        return direct

    rope_params = getattr(config_obj, "rope_parameters", None)
    if rope_params is None:
        return None

    # Case A: dict-like
    if isinstance(rope_params, dict):
        if "rope_theta" in rope_params:
            return rope_params["rope_theta"]
        # by-mode: {"default": {...}, ...}
        sub_dicts = [v for v in rope_params.values() if isinstance(v, (dict,)) or hasattr(v, "rope_theta")]
        chosen = None
        if "default" in rope_params and (
            isinstance(rope_params["default"], dict) or hasattr(rope_params["default"], "rope_theta")
        ):
            chosen = rope_params["default"]
        elif sub_dicts:
            chosen = sub_dicts[0]
        if chosen is not None:
            return _extract_rope_theta_from_obj(chosen) if not isinstance(chosen, dict) else chosen.get("rope_theta")
        return None

    # Case B: object with attributes
    inner = getattr(rope_params, "rope_theta", None)
    if inner is not None:
        return inner
    # by-mode object: try .default
    inner_default = getattr(rope_params, "default", None)
    if inner_default is not None:
        return _extract_rope_theta_from_obj(inner_default)
    return None


def _ensure_rope_theta_from_config_obj(cfg: dict, config_obj) -> None:
    """Backfill ``rope_theta`` on every level by probing the live HF config object.

    HF model configs may carry ``rope_theta`` only inside ``rope_parameters`` (or
    not at all on the serialized JSON when ``rope_parameters`` is nested by mode
    and the JSON-side flatten couldn't reach a usable value). Walk the config
    object alongside the JSON dict and copy ``rope_theta`` over wherever it's
    still missing — at top level for LLMs, and on
    ``text_config`` / ``language_config`` / ``thinker_config`` for VLMs.
    """
    if not isinstance(cfg, dict) or config_obj is None:
        return
    if "rope_theta" not in cfg:
        rope_theta = _extract_rope_theta_from_obj(config_obj)
        if rope_theta is not None:
            cfg["rope_theta"] = rope_theta
    for sub_key in _TEXT_SUB_CONFIG_KEYS + ("vision_config", "audio_config"):
        sub_cfg = cfg.get(sub_key)
        sub_obj = getattr(config_obj, sub_key, None)
        if isinstance(sub_cfg, dict) and sub_obj is not None:
            _ensure_rope_theta_from_config_obj(sub_cfg, sub_obj)


# Sub-config keys whose ``model_type`` we want to preserve verbatim from the
# *original* checkpoint's ``config.json``. Newer ``transformers`` versions
# have a bug where they rewrite a sub-config's ``model_type`` on save (e.g.
# ``vision_config.model_type`` becomes ``qwen3_5_vision`` while the original
# checkpoint had ``qwen3_5``). We snapshot the original JSON and restore.
_PRESERVE_MODEL_TYPE_SUB_KEYS = (
    "text_config",
    "language_config",
    "thinker_config",
    "vision_config",
    "audio_config",
)


def _load_original_config_json(model: nn.Module) -> "dict | None":
    """Best-effort load of the *raw* ``config.json`` from the original checkpoint.

    HF's in-memory ``model.config`` is unreliable for preserving ``model_type``
    because newer ``transformers`` versions auto-rewrite sub-config
    ``model_type`` values based on the registered child config class
    (e.g. an original ``qwen3_5`` becomes ``qwen3_5_text`` /
    ``qwen3_5_vision`` once the model is instantiated). The on-disk
    ``config.json`` from the source checkpoint is the only faithful source.

    We try ``model.config._name_or_path`` first, then ``model.name_or_path``.
    Returns the parsed JSON dict, or ``None`` if it can't be located/read.
    """
    candidates = []
    cfg_obj = getattr(model, "config", None)
    if cfg_obj is not None:
        nop = getattr(cfg_obj, "_name_or_path", None) or getattr(cfg_obj, "name_or_path", None)
        if nop:
            candidates.append(nop)
    nop2 = getattr(model, "name_or_path", None)
    if nop2 and nop2 not in candidates:
        candidates.append(nop2)

    for path in candidates:
        try:
            if isinstance(path, str) and os.path.isdir(path):
                cfg_path = os.path.join(path, "config.json")
            elif isinstance(path, str) and os.path.isfile(path) and path.endswith(".json"):
                cfg_path = path
            else:
                continue
            if not os.path.isfile(cfg_path):
                continue
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[mlx-export] failed to read original config.json at {path!r}: {e}")
    return None


def _snapshot_original_model_types(model: nn.Module) -> "dict | None":
    """Snapshot ``model_type`` for top-level and known sub-configs.

    Prefer the original on-disk ``config.json`` (the only faithful source),
    falling back to the in-memory ``model.config`` if the file isn't reachable.
    Newer ``transformers`` versions rewrite sub-config ``model_type`` values
    on instantiation (e.g. ``qwen3_5`` -> ``qwen3_5_text`` /
    ``qwen3_5_vision``), so the in-memory copy can't be trusted for this.

    Returns a plain dict shaped like::

        {"model_type": "qwen3_5",
         "vision_config": {"model_type": "qwen3_5"},
         ...}

    or ``None`` if no config is available.
    """
    cfg_dict = _load_original_config_json(model)
    source = "on-disk config.json"
    if cfg_dict is None:
        cfg_obj = getattr(model, "config", None)
        if cfg_obj is None:
            logger.warning("[mlx-export] model has no .config; skipping model_type preservation")
            return None
        try:
            cfg_dict = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else dict(cfg_obj.__dict__)
            source = "in-memory model.config (fallback; may be HF-rewritten)"
        except Exception as e:
            logger.warning(f"[mlx-export] failed to snapshot model.config: {e}")
            return None

    snapshot: dict = {}
    top_model_type = cfg_dict.get("model_type")
    if top_model_type is not None:
        snapshot["model_type"] = top_model_type
    # Force every known sub-config to share the top-level ``model_type``.
    # Different ``transformers`` versions disagree on what sub-config
    # ``model_type`` should be (some leave it equal to the parent, some append
    # ``_text`` / ``_vision`` / ``_audio``). To keep the exported MLX config
    # stable across versions we just propagate the top-level value down.
    for sub_key in _PRESERVE_MODEL_TYPE_SUB_KEYS:
        sub = cfg_dict.get(sub_key)
        if isinstance(sub, dict) and top_model_type is not None:
            snapshot[sub_key] = {"model_type": top_model_type}
    logger.info(f"[mlx-export] snapshotted original model_types from {source}: {snapshot!r}")
    return snapshot


def _preserve_original_model_types(new_cfg: dict, orig_cfg: "dict | None") -> None:
    """Make ``new_cfg``'s ``model_type`` fields exactly match ``orig_cfg``.

    For the top-level config and each known sub-config (``text_config``,
    ``vision_config``, etc.):
      * If the original had a ``model_type`` value, write it back.
      * If the original did **not** have ``model_type`` at all, delete the key
        from the new config (so HF can't sneak in a renamed default like
        ``qwen3_5`` → ``qwen3_5_vision``).
      * If the original sub-config doesn't exist in the new config, nothing to do.

    No-op if ``orig_cfg`` is ``None``.
    """
    if not isinstance(new_cfg, dict):
        logger.warning("[mlx-export] new config is not a dict; skip model_type preservation")
        return
    if not isinstance(orig_cfg, dict):
        # Already warned in _load_original_config_json.
        return

    def _align(sub_new: dict, sub_orig: dict, label: str) -> None:
        new_val = sub_new.get("model_type", "<missing>")
        if "model_type" in sub_orig:
            orig_val = sub_orig["model_type"]
            if new_val != orig_val:
                logger.info(f"[mlx-export] restoring {label}.model_type: {new_val!r} -> {orig_val!r}")
                sub_new["model_type"] = orig_val
        else:
            if "model_type" in sub_new:
                logger.info(
                    f"[mlx-export] removing {label}.model_type ({new_val!r}); " f"original config did not declare one"
                )
                sub_new.pop("model_type", None)

    _align(new_cfg, orig_cfg, "top-level")
    for sub_key in _PRESERVE_MODEL_TYPE_SUB_KEYS:
        sub_new = new_cfg.get(sub_key)
        sub_orig = orig_cfg.get(sub_key)
        if isinstance(sub_new, dict) and isinstance(sub_orig, dict):
            _align(sub_new, sub_orig, sub_key)


def _strip_prefix(name: str, prefix: str) -> str:
    """Return ``name`` with ``prefix.`` removed, or ``name`` if no match."""
    if name == prefix:
        return name
    pref = prefix + "."
    return name.removeprefix(pref)


def _build_text_subconfig_quantization(quant_cfg: dict, text_prefix: str) -> dict:
    """Re-key a top-level quantization dict for placement under ``text_config``.

    mlx-vlm's language-model loader queries ``text_config["quantization"]`` with
    keys *relative to the language model* (e.g. ``model.layers.0...``), so we
    strip the outer ``language_model.`` / ``text_model.`` prefix from each entry.
    Entries that don't belong to the language model are dropped.
    """
    sub: dict = {"group_size": quant_cfg["group_size"], "bits": quant_cfg["bits"]}
    pref = text_prefix + "."
    for k, v in quant_cfg.items():
        if k in ("group_size", "bits"):
            continue
        if k == text_prefix or k.startswith(pref):
            sub[_strip_prefix(k, text_prefix)] = v
    return sub


def _detect_text_module_prefix(model: nn.Module) -> str:
    """Best-effort detection of the language-model sub-module name in a VLM.

    Returns an empty string if the model is text-only.
    """
    for cand in ("language_model", "text_model", "thinker"):
        if hasattr(model, cand):
            return cand
    return ""
