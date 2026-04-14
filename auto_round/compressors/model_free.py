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

"""Model-free RTN quantization.

This module performs weight-only quantization (WOQ) using RTN (Round-To-Nearest)
**without** loading the full model into memory.  It downloads safetensors files
directly (from a Hugging Face repo or a local directory), quantizes eligible
``nn.Linear`` weight tensors shard-by-shard, and writes the packed result to
the output directory.

Key features
------------
* **No model object required** – only ``config.json`` and safetensors files are
  needed.
* **Low-disk-memory mode** (``--low_disk_mem_usage``) – downloads, quantizes,
  and deletes each safetensors shard one at a time so that only one copy of a
  shard is ever on disk.
* **Per-layer configuration** – honours ``--layer_config`` overrides and
  ``--ignore_layers`` for fine-grained control.
* **Predefined ignore layers** – inspects
  :func:`~auto_round.special_model_handler.get_predefined_ignore_layers_from_config`
  so that model-specific quirks (e.g. MoE gates, MTP layers) are respected even
  without a live model object.

Usage (CLI)
-----------
::

    auto_round facebook/opt-125m \\
        --model_free \\
        --low_disk_mem_usage \\
        --scheme W4A16 \\
        --output_dir int4-125m

Usage (API)
-----------
::

    from auto_round.compressors.model_free import model_free_quantize
    model_free_quantize(
        model_name_or_path="facebook/opt-125m",
        scheme="W4A16",
        output_dir="./int4-125m",
        low_disk_mem_usage=True,
    )
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from dataclasses import asdict, fields
from typing import Optional, Union

import torch

from auto_round.logger import logger
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme, preset_name_to_scheme
from auto_round.utils.common import compress_layer_names
from auto_round.utils.missing_tensors import quantize_weight_rtn

# ------------------------------------------------------------------ #
#  Layers that should never be RTN-quantized (e.g. routing gates)     #
# ------------------------------------------------------------------ #
_BLOCK_NAME_TO_IGNORE = ["shared_expert_gate.", "mlp.gate.", "g_proj."]

# ------------------------------------------------------------------ #
#  Predefined ignore-layer rules (mirrors special_model_handler.py)   #
# ------------------------------------------------------------------ #
# Each entry: (matcher_fn(config) -> bool, ignore_layers: list[str])
# matcher_fn operates on the config dict (no model object needed).

_CONFIG_BASED_IGNORE_RULES: list[tuple] = []


def _register_config_ignore_rule(matcher, ignore_layers: list[str]):
    """Register a config-based ignore-layer rule for model-free mode."""
    _CONFIG_BASED_IGNORE_RULES.append((matcher, ignore_layers))


def _arch_contains(config: dict, pattern: str) -> bool:
    archs = config.get("architectures", [])
    archs_str = ",".join(archs) if isinstance(archs, list) else str(archs)
    return pattern in archs_str


def _model_type_eq(config: dict, model_type: str) -> bool:
    return config.get("model_type", "") == model_type


def _model_type_contains(config: dict, model_type: str) -> bool:
    return model_type in config.get("model_type", "")


# Longcat – skip classifier
_register_config_ignore_rule(
    lambda cfg: _arch_contains(cfg, "Longcat"),
    ["classifier"],
)


# Glm4MoeLite – skip first_k_dense_replace mlp layers
def _glm_flash_ignore(config: dict) -> list[str]:
    num_dense = config.get("first_k_dense_replace", 1)
    return [f"layers.{i}.mlp" for i in range(num_dense)]


_register_config_ignore_rule(
    lambda cfg: _arch_contains(cfg, "Glm4MoeLite"),
    [],  # placeholder, resolved dynamically
)


# glm_moe_dsa – skip dense layers + weights_proj
_register_config_ignore_rule(
    lambda cfg: _model_type_eq(cfg, "glm_moe_dsa"),
    ["weights_proj"],
)


# step3p5 – skip g_proj, moe.gate, MTP layers
_register_config_ignore_rule(
    lambda cfg: _model_type_eq(cfg, "step3p5"),
    ["g_proj", "moe.gate", "eh_proj", "shared_head", "layers.45"],
)


def get_predefined_ignore_layers_from_config(config: dict) -> list[str]:
    """Return layers to ignore based on the model's config.json (no model object needed).

    This mirrors :func:`auto_round.special_model_handler.get_predefined_ignore_layers`
    but operates purely on the config dictionary.
    """
    layers: list[str] = []
    for matcher, ignore_list in _CONFIG_BASED_IGNORE_RULES:
        try:
            if matcher(config):
                if _arch_contains(config, "Glm4MoeLite"):
                    layers.extend(_glm_flash_ignore(config))
                elif _model_type_eq(config, "glm_moe_dsa"):
                    layers.extend(_glm_flash_ignore(config))
                    layers.extend(ignore_list)  # adds "weights_proj"
                else:
                    layers.extend(ignore_list)
                break
        except Exception:
            continue

    # Fallback: for MoE models, ignore .gate layers
    if not layers and _is_moe_config(config):
        layers.append(".gate")

    return list(dict.fromkeys(layers))


def _is_moe_config(config: dict) -> bool:
    """Check if config indicates a MoE model."""
    return (
        config.get("num_local_experts") is not None
        or config.get("num_experts") is not None
        or config.get("num_experts_per_tok") is not None
        or "moe" in config.get("model_type", "").lower()
        or "moe" in ",".join(config.get("architectures", [])).lower()
    )


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _resolve_source_dir(model_name_or_path: str) -> str:
    """Resolve model_name_or_path to a local directory (download if needed)."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name_or_path)


def _load_config(source_dir: str) -> dict:
    """Load config.json from model directory."""
    config_path = os.path.join(source_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {source_dir}")
    with open(config_path) as f:
        return json.load(f)


def _list_safetensor_shards(source_dir: str) -> list[str]:
    """Return list of safetensors shard filenames in order."""
    index_file = os.path.join(source_dir, "model.safetensors.index.json")
    single_file = os.path.join(source_dir, "model.safetensors")

    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        # Unique shards in order
        seen = set()
        shards = []
        for shard_file in index["weight_map"].values():
            if shard_file not in seen:
                seen.add(shard_file)
                shards.append(shard_file)
        return shards
    elif os.path.exists(single_file):
        return ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors files found in {source_dir}")


def _get_tensor_to_shard_map(source_dir: str) -> dict[str, str]:
    """Return mapping from tensor name to shard filename."""
    index_file = os.path.join(source_dir, "model.safetensors.index.json")
    single_file = os.path.join(source_dir, "model.safetensors")

    if os.path.exists(index_file):
        with open(index_file) as f:
            return json.load(f)["weight_map"]
    elif os.path.exists(single_file):
        from safetensors import safe_open

        with safe_open(single_file, framework="pt", device="cpu") as f:
            return {name: "model.safetensors" for name in f.keys()}
    else:
        raise FileNotFoundError(f"No safetensors files found in {source_dir}")


def _should_ignore_layer(tensor_name: str, ignore_patterns: list[str]) -> bool:
    """Check if a tensor should be ignored (kept in full precision)."""
    layer_name = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name
    for pattern in ignore_patterns:
        if pattern.endswith("."):
            # Prefix match: e.g. "layers.45." matches "model.layers.45.xxx"
            # but not "model.layers.450.xxx"
            stripped = pattern.rstrip(".")
            if stripped + "." in layer_name or layer_name.endswith(stripped):
                return True
        elif pattern in layer_name:
            return True
    return False


def _should_skip_quantization(tensor_name: str) -> bool:
    """Check if a tensor should be skipped (routing gates, etc.)."""
    for block_name in _BLOCK_NAME_TO_IGNORE:
        if block_name in tensor_name:
            return True
    return False


def _resolve_layer_scheme(
    tensor_name: str,
    layer_config: dict,
    default_scheme: dict,
) -> dict | None:
    """Resolve quantization config for a specific tensor.

    Returns None if the layer should be kept in full precision (bits >= 16).
    """
    layer_name = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name

    # Check exact match first
    if layer_name in layer_config:
        cfg = layer_config[layer_name]
        if cfg.get("bits", default_scheme["bits"]) >= 16:
            return None
        merged = {**default_scheme, **cfg}
        return merged

    # Check regex/fuzzy match
    for pattern, cfg in layer_config.items():
        try:
            if re.search(pattern, layer_name):
                if cfg.get("bits", default_scheme["bits"]) >= 16:
                    return None
                merged = {**default_scheme, **cfg}
                return merged
        except re.error:
            if pattern in layer_name:
                if cfg.get("bits", default_scheme["bits"]) >= 16:
                    return None
                merged = {**default_scheme, **cfg}
                return merged

    return default_scheme


def _is_eligible_weight(tensor_name: str, tensor: torch.Tensor) -> bool:
    """Check if a tensor is eligible for quantization (2D Linear weight)."""
    return tensor_name.endswith(".weight") and tensor.dim() == 2


# ------------------------------------------------------------------ #
#  Download helpers for low_disk_mem_usage                             #
# ------------------------------------------------------------------ #


def _download_single_shard(
    model_name_or_path: str,
    shard_filename: str,
    local_dir: str,
) -> str:
    """Download a single safetensors shard file. Returns the local path."""
    local_path = os.path.join(local_dir, shard_filename)
    if os.path.exists(local_path):
        logger.info(f"Shard '{shard_filename}' already exists at '{local_path}', skipping download.")
        return local_path

    if os.path.isdir(model_name_or_path):
        src = os.path.join(model_name_or_path, shard_filename)
        if os.path.exists(src):
            shutil.copy2(src, local_path)
            return local_path
        raise FileNotFoundError(f"{shard_filename} not found in {model_name_or_path}")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=model_name_or_path,
        filename=shard_filename,
        local_dir=local_dir,
    )
    return downloaded


def _download_metadata_files(
    model_name_or_path: str,
    local_dir: str,
) -> str:
    """Download config.json, tokenizer files, and index file. Returns local dir."""
    os.makedirs(local_dir, exist_ok=True)
    metadata_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "model.safetensors.index.json",
    ]

    if os.path.isdir(model_name_or_path):
        for fname in metadata_files:
            src = os.path.join(model_name_or_path, fname)
            dst = os.path.join(local_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        # Also copy any .py files (modeling code for trust_remote_code)
        for fname in os.listdir(model_name_or_path):
            if fname.endswith(".py"):
                src = os.path.join(model_name_or_path, fname)
                dst = os.path.join(local_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        return local_dir

    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        repo_files = list_repo_files(model_name_or_path)
    except Exception:
        repo_files = []

    for fname in metadata_files:
        if repo_files and fname not in repo_files:
            continue
        try:
            hf_hub_download(
                repo_id=model_name_or_path,
                filename=fname,
                local_dir=local_dir,
            )
        except Exception:
            pass  # Not all files are mandatory

    # Download .py files for trust_remote_code
    for fname in repo_files:
        if fname.endswith(".py"):
            try:
                hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=fname,
                    local_dir=local_dir,
                )
            except Exception:
                pass

    return local_dir


# ------------------------------------------------------------------ #
#  Core: process a single shard                                        #
# ------------------------------------------------------------------ #


def _process_shard(
    shard_path: str,
    default_scheme: dict,
    layer_config: dict,
    ignore_patterns: list[str],
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Quantize eligible weights in a single safetensors shard.

    Returns:
        (output_tensors, quantized_layer_names, ignored_layer_names)
    """
    from safetensors import safe_open

    output_tensors: dict[str, torch.Tensor] = {}
    quantized_layers: list[str] = []
    ignored_layers: list[str] = []

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        tensor_names = list(f.keys())

        for tensor_name in tensor_names:
            tensor = f.get_tensor(tensor_name)

            # Check if this is an eligible weight for quantization
            if not _is_eligible_weight(tensor_name, tensor):
                output_tensors[tensor_name] = tensor.contiguous()
                continue

            layer_name = tensor_name.rsplit(".", 1)[0]

            # Check ignore patterns
            if _should_ignore_layer(tensor_name, ignore_patterns):
                logger.debug(f"Ignoring (user-specified): {layer_name}")
                output_tensors[tensor_name] = tensor.contiguous()
                ignored_layers.append(layer_name)
                continue

            # Check predefined skip patterns (routing gates, etc.)
            if _should_skip_quantization(tensor_name):
                logger.debug(f"Skipping (predefined): {layer_name}")
                output_tensors[tensor_name] = tensor.contiguous()
                ignored_layers.append(layer_name)
                continue

            # Resolve per-layer scheme
            scheme = _resolve_layer_scheme(tensor_name, layer_config, default_scheme)
            if scheme is None:
                logger.debug(f"Keeping full precision: {layer_name}")
                output_tensors[tensor_name] = tensor.contiguous()
                ignored_layers.append(layer_name)
                continue

            bits = scheme["bits"]
            group_size = scheme["group_size"]
            sym = scheme.get("sym", True)

            if bits >= 16:
                output_tensors[tensor_name] = tensor.contiguous()
                ignored_layers.append(layer_name)
                continue

            # Quantize with RTN
            try:
                qweight, qzeros, scales = quantize_weight_rtn(
                    weight=tensor,
                    bits=bits,
                    group_size=group_size,
                    sym=sym,
                    device=torch.device(device) if device != "cpu" else None,
                )

                # Store packed tensors with auto_gptq naming convention
                output_tensors[f"{layer_name}.qweight"] = qweight.contiguous()
                output_tensors[f"{layer_name}.qzeros"] = qzeros.contiguous()
                output_tensors[f"{layer_name}.scales"] = scales.contiguous()

                # Store g_idx as empty (not needed for RTN without reordering)
                in_features = tensor.shape[1]
                if in_features % group_size != 0:
                    in_features = in_features + (group_size - in_features % group_size)
                g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
                output_tensors[f"{layer_name}.g_idx"] = g_idx.contiguous()

                quantized_layers.append(layer_name)
                logger.debug(f"Quantized: {layer_name} (bits={bits}, group_size={group_size}, sym={sym})")

            except Exception as e:
                logger.warning(f"Failed to quantize {layer_name}: {e}. Keeping original weight.")
                output_tensors[tensor_name] = tensor.contiguous()
                ignored_layers.append(layer_name)

    return output_tensors, quantized_layers, ignored_layers


# ------------------------------------------------------------------ #
#  Build quantization config for config.json                           #
# ------------------------------------------------------------------ #


def _build_quantization_config(
    default_scheme: dict,
    layer_config: dict,
    ignore_patterns: list[str],
    quantized_layers: list[str],
    ignored_layers: list[str],
    block_name_to_quantize: Optional[list[str]] = None,
) -> dict:
    """Build a quantization_config dict compatible with auto-round format."""
    from auto_round.version import __version__

    scheme_keys = [f.name for f in fields(QuantizationScheme)]

    qconfig = {
        "quant_method": "auto-round",
        "packing_format": "auto_round:auto_gptq",
        "bits": default_scheme["bits"],
        "group_size": default_scheme["group_size"],
        "sym": default_scheme.get("sym", True),
        "data_type": default_scheme.get("data_type", "int"),
        "iters": 0,
        "model_free": True,
        "autoround_version": __version__,
    }

    if block_name_to_quantize:
        qconfig["block_name_to_quantize"] = block_name_to_quantize

    # Build extra_config for layers with non-default settings
    extra_config = {}
    for layer_name, cfg in layer_config.items():
        if cfg.get("bits", default_scheme["bits"]) >= 16:
            extra_config[layer_name] = {k: cfg.get(k) for k in scheme_keys if cfg.get(k) is not None}
            continue
        # Check if config differs from default
        differs = False
        for key in ("bits", "group_size", "sym"):
            if cfg.get(key) is not None and cfg[key] != default_scheme.get(key):
                differs = True
                break
        if differs:
            extra_config[layer_name] = {k: cfg.get(k) for k in scheme_keys if cfg.get(k) is not None}

    # Add ignored layers to extra_config
    unique_ignored = list(dict.fromkeys(ignored_layers))
    for layer_name in unique_ignored:
        if layer_name not in extra_config:
            extra_config[layer_name] = {
                "bits": 16,
                "data_type": "float",
            }

    if extra_config:
        qconfig["extra_config"] = extra_config

    return qconfig


# ------------------------------------------------------------------ #
#  Write output shard and update index                                 #
# ------------------------------------------------------------------ #


def _write_output_shard(
    output_dir: str,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str],
):
    """Write a single output shard and update the weight_map."""
    from safetensors.torch import save_file

    shard_path = os.path.join(output_dir, shard_name)
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    save_file(tensors, shard_path)
    for tensor_name in tensors:
        weight_map[tensor_name] = shard_name


def _write_index_file(output_dir: str, weight_map: dict[str, str]):
    """Write model.safetensors.index.json."""
    if len(set(weight_map.values())) <= 1:
        # Single shard – rename to model.safetensors if needed
        shard_names = list(set(weight_map.values()))
        if shard_names and shard_names[0] != "model.safetensors":
            src = os.path.join(output_dir, shard_names[0])
            dst = os.path.join(output_dir, "model.safetensors")
            if os.path.exists(src):
                os.rename(src, dst)
            weight_map = {k: "model.safetensors" for k in weight_map}
        return

    index = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


# ------------------------------------------------------------------ #
#  Main entry point                                                    #
# ------------------------------------------------------------------ #


def model_free_quantize(
    model_name_or_path: str,
    output_dir: str,
    scheme: Union[str, QuantizationScheme] = "W4A16",
    layer_config: Optional[dict] = None,
    ignore_layers: str = "",
    format: str = "auto_round",
    device: str = "cpu",
    low_disk_mem_usage: bool = False,
    trust_remote_code: bool = True,
) -> str:
    """Perform model-free RTN quantization.

    Downloads safetensors directly, applies RTN quantization to each eligible
    Linear weight, and saves the packed model in auto-round format.

    Args:
        model_name_or_path: HuggingFace model ID or local directory path.
        output_dir: Directory to save the quantized model.
        scheme: Quantization scheme name (e.g. ``"W4A16"``) or a
            :class:`~auto_round.schemes.QuantizationScheme` instance.
        layer_config: Per-layer quantization overrides.  Keys are layer names
            or regex patterns; values are dicts with ``bits``, ``group_size``,
            ``sym``, etc.
        ignore_layers: Comma-separated list of layer name patterns to keep in
            full precision.
        format: Output format (only ``"auto_round"`` is supported in
            model-free mode).
        device: Device for quantization computation (``"cpu"`` or ``"cuda"``).
        low_disk_mem_usage: If True, download and process one shard at a time,
            deleting the source shard after processing.
        trust_remote_code: Whether to trust remote code when downloading.

    Returns:
        Path to the output directory.
    """
    from safetensors.torch import save_file

    # ---- Validate format ----
    supported_formats = ("auto_round",)
    format_lower = format.lower().replace(" ", "").split(",")[0]
    if format_lower not in supported_formats:
        raise ValueError(
            f"Model-free mode only supports {supported_formats} format, got '{format}'. "
            f"Please use --format auto_round."
        )

    # ---- Parse scheme ----
    if isinstance(scheme, str):
        scheme_name = scheme.upper()
        if scheme_name not in PRESET_SCHEMES:
            raise ValueError(f"Unknown scheme '{scheme}'. Available: {list(PRESET_SCHEMES.keys())}")
        scheme_obj = preset_name_to_scheme(scheme_name)
    elif isinstance(scheme, QuantizationScheme):
        scheme_obj = scheme
    else:
        raise TypeError(f"Unsupported scheme type: {type(scheme)}")

    default_scheme = asdict(scheme_obj)
    default_scheme = {k: v for k, v in default_scheme.items() if v is not None}

    # ---- Parse layer_config ----
    layer_config = copy.deepcopy(layer_config) if layer_config else {}
    # Normalize layer_config values
    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    for key, val in list(layer_config.items()):
        if isinstance(val, str):
            parsed = asdict(preset_name_to_scheme(val.upper()))
            layer_config[key] = {k: v for k, v in parsed.items() if v is not None}
        elif isinstance(val, QuantizationScheme):
            layer_config[key] = {k: v for k, v in asdict(val).items() if v is not None}
        elif isinstance(val, dict):
            pass  # already a dict
        else:
            raise TypeError(f"Unsupported layer_config value type for '{key}': {type(val)}")

    # ---- Parse ignore_layers ----
    ignore_patterns: list[str] = []
    if ignore_layers:
        ignore_patterns = [p.strip() for p in ignore_layers.replace(" ", "").split(",") if p.strip()]

    # ---- Setup output directory ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- Resolve source and load config ----
    if low_disk_mem_usage:
        # In low-disk mode, only download metadata first
        work_dir = os.path.join(output_dir, "_model_free_tmp")
        _download_metadata_files(model_name_or_path, work_dir)
        config = _load_config(work_dir)
    else:
        source_dir = _resolve_source_dir(model_name_or_path)
        config = _load_config(source_dir)

    # ---- Get predefined ignore layers from config ----
    predefined_ignore = get_predefined_ignore_layers_from_config(config)
    if predefined_ignore:
        compressed = compress_layer_names(predefined_ignore)
        logger.info(f"Using predefined ignore_layers from config: {compressed}")
        ignore_patterns.extend(predefined_ignore)

    # ---- Discover shards ----
    if low_disk_mem_usage:
        # Get shard list from index file or by checking the repo
        index_file = os.path.join(work_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                index_data = json.load(f)
            seen = set()
            shard_names = []
            for shard_file in index_data["weight_map"].values():
                if shard_file not in seen:
                    seen.add(shard_file)
                    shard_names.append(shard_file)
        else:
            shard_names = ["model.safetensors"]
    else:
        shard_names = _list_safetensor_shards(source_dir)

    logger.info(
        f"Model-free quantization: {model_name_or_path}\n"
        f"  Scheme: {scheme_obj}\n"
        f"  Output: {output_dir}\n"
        f"  Shards: {len(shard_names)}\n"
        f"  Low-disk mode: {low_disk_mem_usage}"
    )

    # ---- Process each shard ----
    all_quantized_layers: list[str] = []
    all_ignored_layers: list[str] = []
    output_weight_map: dict[str, str] = {}

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    shard_iter = _tqdm(shard_names, desc="Processing shards", unit="shard") if _tqdm else shard_names

    for shard_idx, shard_name in enumerate(shard_iter):
        if low_disk_mem_usage:
            # Download this specific shard
            shard_path = _download_single_shard(model_name_or_path, shard_name, work_dir)
        else:
            shard_path = os.path.join(source_dir, shard_name)

        if not os.path.exists(shard_path):
            logger.warning(f"Shard not found: {shard_path}, skipping")
            continue

        # Process the shard
        output_tensors, quantized, ignored = _process_shard(
            shard_path=shard_path,
            default_scheme=default_scheme,
            layer_config=layer_config,
            ignore_patterns=ignore_patterns,
            device=device,
        )

        all_quantized_layers.extend(quantized)
        all_ignored_layers.extend(ignored)

        # Write output shard
        out_shard_name = f"model-{shard_idx + 1:05d}-of-{len(shard_names):05d}.safetensors"
        _write_output_shard(output_dir, out_shard_name, output_tensors, output_weight_map)

        # Free memory
        del output_tensors

        # In low-disk mode, delete the source shard to save disk space
        if low_disk_mem_usage:
            try:
                os.remove(shard_path)
                logger.debug(f"Deleted source shard: {shard_path}")
            except OSError as e:
                logger.warning(f"Could not delete source shard {shard_path}: {e}")

    # ---- Write index file ----
    _write_index_file(output_dir, output_weight_map)

    # ---- Write config.json ----
    quantization_config = _build_quantization_config(
        default_scheme=default_scheme,
        layer_config=layer_config,
        ignore_patterns=ignore_patterns,
        quantized_layers=all_quantized_layers,
        ignored_layers=all_ignored_layers,
    )

    # Copy and update config.json
    config["quantization_config"] = quantization_config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Write standalone quantization_config.json
    qconfig_path = os.path.join(output_dir, "quantization_config.json")
    with open(qconfig_path, "w") as f:
        json.dump(quantization_config, f, indent=2)

    # ---- Copy tokenizer files ----
    if low_disk_mem_usage:
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "generation_config.json",
        ]
        for fname in tokenizer_files:
            src = os.path.join(work_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        # Copy .py files
        for fname in os.listdir(work_dir):
            if fname.endswith(".py"):
                src = os.path.join(work_dir, fname)
                dst = os.path.join(output_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        # Clean up temp dir
        try:
            shutil.rmtree(work_dir)
        except OSError:
            pass
    else:
        # Copy tokenizer and other files from source
        copy_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "generation_config.json",
        ]
        for fname in copy_files:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        # Copy .py files
        for fname in os.listdir(source_dir):
            if fname.endswith(".py"):
                src = os.path.join(source_dir, fname)
                dst = os.path.join(output_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # ---- Summary ----
    compressed_quantized = compress_layer_names(all_quantized_layers)
    compressed_ignored = compress_layer_names(list(dict.fromkeys(all_ignored_layers)))
    logger.info(
        f"\nModel-free quantization complete.\n"
        f"  Output directory: {output_dir}\n"
        f"  Quantized layers ({len(all_quantized_layers)}): {compressed_quantized}\n"
        f"  Ignored layers ({len(set(all_ignored_layers))}): {compressed_ignored}\n"
    )

    return output_dir
