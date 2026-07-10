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

"""Model-free RTN quantization (class-based).

This module performs weight-only quantization (WOQ) using RTN (Round-To-Nearest)
**without** loading the full model into memory.  It reads safetensors files
(from a Hugging Face repo or a local directory), quantizes eligible
``nn.Linear`` weight tensors shard-by-shard, and writes the packed result to
the output directory.

The main entry point is the :class:`ModelFreeCompressor` class.

Supported schemes
-----------------
Model-free mode supports the following quantization families:

**Integer weight-only** (packed in ``auto_round:auto_gptq`` format):

* Preset names: ``W2A16``, ``W2A16G32``, ``W2A16G64``, ``W4A16``,
  ``W4A16_MIXED``, ``W8A16``.
* Custom :class:`~auto_round.schemes.QuantizationScheme` instances with
  ``data_type="int"``, ``bits in {2, 4, 8}``, ``act_bits >= 16``, and any
  symmetric / asymmetric configuration.

**MXFP (Microscaling Floating Point)** (packed in ``mxfp4-pack-quantized`` or
``mxfp8-quantized`` format, compatible with llm-compressor / compressed-tensors):

* Preset names: ``MXFP4``, ``MXFP8``.
* ``data_type="mx_fp"``, ``group_size=32``, ``bits in {4, 8}``.

Schemes that require special packing (FP8, NVFP4, GGUF, INT8_W8A8,
BF16, FPW8A16, ...) are **not** supported in model-free mode and will raise
``ValueError``.  Use the standard AutoRound flow for those.

Output formats
--------------
* **INT schemes** → ``auto_round:auto_gptq`` packing format, ``quant_method="auto-round"``.
* **MXFP schemes** → ``mxfp4-pack-quantized`` or ``mxfp8-quantized`` format,
  ``quant_method="compressed-tensors"``, compatible with vLLM / llm-compressor.

Usage (CLI)
-----------
::

    # Integer WOQ
    auto_round facebook/opt-125m \\
        --model_free \\
        --scheme W4A16 \\
        --output_dir int4-125m

    # MXFP4
    auto_round facebook/opt-125m \\
        --model_free \\
        --scheme MXFP4 \\
        --output_dir mxfp4-125m

Usage (API)
-----------
::

    from auto_round import AutoRound

    # Integer WOQ
    AutoRound(
        model="facebook/opt-125m",
        scheme="W4A16",
        model_free=True,
    ).quantize_and_save("./int4-125m")

    # MXFP4
    AutoRound(
        model="facebook/opt-125m",
        scheme="MXFP4",
        model_free=True,
    ).quantize_and_save("./mxfp4-125m")
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from typing import Any, Callable, Optional, Union

import torch

from auto_round import envs
from auto_round.compressors.utils import is_mx_fp
from auto_round.logger import logger
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme, preset_name_to_scheme
from auto_round.utils.common import AUDIO_MM_KEYS, VISION_MM_KEYS, compress_layer_names, to_standard_regex
from auto_round.utils.device import clear_memory, memory_monitor
from auto_round.utils.missing_tensors import quantize_weight_rtn, split_fused_expert_tensors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# add "embed", "conv" in case of auto detection failure in _check_conv1d_and_embedding
_BLOCK_NAME_TO_IGNORE = ["shared_expert_gate.", ".gate.", "embed", "conv"]

# Preset schemes that model-free mode can produce.
# INT presets use ``auto_round:auto_gptq`` packing; MXFP presets use
# ``mxfp4-pack-quantized`` or ``mxfp8-quantized`` (compressed-tensors) packing.
#
# Note: ``W3A16`` (3-bit) is intentionally excluded.  3-bit packing requires
# in_features to be padded to a multiple of pack_factor=10, which the current
# ``quantize_weight_rtn`` implementation does not handle correctly.
SUPPORTED_PRESET_SCHEMES: tuple[str, ...] = (
    "W2A16",
    "W2A16G32",
    "W2A16G64",
    "W4A16",
    "W4A16_MIXED",
    "W8A16",
    "MXFP4",
    "MXFP8",
)

# Allowed ``bits`` values for integer WOQ.
# 3-bit is excluded — see note above.
_SUPPORTED_INT_BITS: tuple[int, ...] = (2, 4, 8)

# Allowed ``bits`` values for MXFP weight quantization.
_SUPPORTED_MXFP_BITS: tuple[int, ...] = (4, 8)

# Multimodal keywords kept in full precision by default.
_NONTEXT_KEYWORDS: tuple[str, ...] = VISION_MM_KEYS + AUDIO_MM_KEYS


# ---------------------------------------------------------------------------
# Predefined ignore-layer rules
# ---------------------------------------------------------------------------


def get_predefined_ignore_layers_from_config(config: dict) -> list[str]:
    """Return layers to ignore based on the model's config.json.

    Delegates to the same rules registered via
    :func:`~auto_round.special_model_handler.register_ignore_layers` by
    wrapping the config dict in a lightweight pseudo-model object, so there
    is no need to duplicate ignore-layer rule registrations here.
    """
    import types

    from auto_round.special_model_handler import _PRE_DEFINED_IGNORE_LAYERS

    # Build a pseudo-model whose .config attribute exposes the config fields.
    cfg_ns = types.SimpleNamespace(**config)
    wrapper = types.SimpleNamespace(config=cfg_ns)

    layers: list[str] = []
    for rule in _PRE_DEFINED_IGNORE_LAYERS:
        if all(m(wrapper) for m in rule.matchers):
            for ignore_layer in rule.ignore_layers:
                if isinstance(ignore_layer, str):
                    layers.append(ignore_layer)
                else:
                    # callable (e.g. get_glm_flash_ignore_layers)
                    res = ignore_layer(wrapper)
                    if isinstance(res, str):
                        layers.append(res)
                    elif isinstance(res, list):
                        layers.extend(res)

    return list(dict.fromkeys(layers))


# ---------------------------------------------------------------------------
# I/O helpers (model resolution, shard discovery, downloads)
# ---------------------------------------------------------------------------


def _is_model_cached(model_name_or_path: str) -> bool:
    """Return True if the model is already available locally or in HF cache."""
    if os.path.isdir(model_name_or_path):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(model_name_or_path, "config.json")
        return isinstance(result, str)
    except Exception:
        return False


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


def _list_weight_shards(source_dir: str) -> list[str]:
    """Return list of weight shard filenames in order.

    Safetensors shards are preferred.  When no safetensors files are
    found the function falls back to PyTorch ``.bin`` shards.

    Handles both standard naming (``model.safetensors``,
    ``pytorch_model.bin``) and custom prefixes such as
    ``diffusion_pytorch_model-XXXXX-of-XXXXX.safetensors`` by scanning
    all ``*.safetensors.index.json`` / ``*.bin.index.json`` index files
    in the directory when no standard index is found.
    """

    def _shards_from_index(index_path: str) -> list[str]:
        with open(index_path) as f:
            index = json.load(f)
        seen: set[str] = set()
        shards: list[str] = []
        for shard_file in index["weight_map"].values():
            if shard_file not in seen:
                seen.add(shard_file)
                shards.append(shard_file)
        return shards

    # --- safetensors: standard index ---
    st_index = os.path.join(source_dir, "model.safetensors.index.json")
    if os.path.exists(st_index):
        return _shards_from_index(st_index)

    # --- safetensors: custom-prefix index (e.g. diffusion_pytorch_model.safetensors.index.json) ---
    for fname in sorted(os.listdir(source_dir)):
        if fname.endswith(".safetensors.index.json"):
            return _shards_from_index(os.path.join(source_dir, fname))

    # --- safetensors: single file or index-less multi-file shards ---
    st_files = sorted(f for f in os.listdir(source_dir) if f.endswith(".safetensors"))
    if len(st_files) >= 1:
        return st_files

    # --- pytorch .bin: standard index ---
    bin_index = os.path.join(source_dir, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index):
        return _shards_from_index(bin_index)

    # --- pytorch .bin: custom-prefix index ---
    for fname in sorted(os.listdir(source_dir)):
        if fname.endswith(".bin.index.json"):
            return _shards_from_index(os.path.join(source_dir, fname))

    # --- pytorch .bin: single file ---
    bin_single = os.path.join(source_dir, "pytorch_model.bin")
    if os.path.exists(bin_single):
        return ["pytorch_model.bin"]

    # --- pytorch .bin: any single .bin file ---
    bin_files = sorted(f for f in os.listdir(source_dir) if f.endswith(".bin"))
    if len(bin_files) >= 1:
        return bin_files


def _is_eligible_weight(tensor_name: str, tensor: torch.Tensor) -> bool:
    """Check if a tensor is eligible for quantization (2D Linear weight)."""
    return tensor_name.endswith(".weight") and tensor.dim() == 2


def _is_weight_shard(fname: str) -> bool:
    """Return True if *fname* is a weight shard (safetensors or .bin).

    Excludes index files (``*.index.json``) so that they are copied to the
    output directory as normal metadata.
    """
    if fname.endswith(".index.json"):
        return False
    return fname.endswith(".safetensors") or fname.endswith(".bin")


# Keep old name as an alias for backward compatibility.
_is_safetensors_shard = _is_weight_shard


def _download_single_shard(
    model_name_or_path: str,
    shard_filename: str,
    local_dir: str,
) -> str:
    """Download a single safetensors shard file. Returns the local path."""
    os.makedirs(local_dir, exist_ok=True)
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

    return hf_hub_download(
        repo_id=model_name_or_path,
        filename=shard_filename,
        local_dir=local_dir,
    )


def _download_metadata_files(
    model_name_or_path: str,
    local_dir: str,
) -> str:
    """Download all non-safetensors files from a model repo. Returns local dir."""
    os.makedirs(local_dir, exist_ok=True)

    if os.path.isdir(model_name_or_path):
        for fname in os.listdir(model_name_or_path):
            if _is_weight_shard(fname):
                continue
            src = os.path.join(model_name_or_path, fname)
            dst = os.path.join(local_dir, fname)
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
            elif os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        return local_dir

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_name_or_path,
        local_dir=local_dir,
        ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.pt"],
    )
    return local_dir


# ---------------------------------------------------------------------------
# Pattern matcher
# ---------------------------------------------------------------------------


class _PatternMatcher:
    """Precompiled pattern matcher with result caching.

    Merges *ignore_patterns* and ``_BLOCK_NAME_TO_IGNORE`` into single
    compiled regexes, precompiles ``layer_config`` regex patterns, and
    caches all match results so that repeated lookups (common across
    shards) are O(1) dict hits.
    """

    __slots__ = (
        "_ignore_re",
        "_skip_re",
        "_layer_config",
        "_default_scheme",
        "_compiled_lc",
        "_ignore_cache",
        "_scheme_cache",
    )

    def __init__(
        self,
        ignore_patterns: list[str],
        layer_config: dict[str, dict],
        default_scheme: dict,
    ) -> None:
        self._default_scheme = default_scheme
        self._layer_config = layer_config

        self._ignore_re: re.Pattern | None = self._build_ignore_regex(ignore_patterns)

        skip_parts = [re.escape(b) for b in _BLOCK_NAME_TO_IGNORE]
        self._skip_re: re.Pattern | None = re.compile("|".join(skip_parts)) if skip_parts else None

        # Each entry: (compiled_regex | None, plain_string | None, cfg_dict)
        self._compiled_lc: list[tuple[re.Pattern | None, str | None, dict]] = []
        for pattern, cfg in layer_config.items():
            try:
                self._compiled_lc.append((re.compile(to_standard_regex(pattern)), None, cfg))
            except re.error:
                self._compiled_lc.append((None, pattern, cfg))

        self._ignore_cache: dict[str, bool] = {}
        self._scheme_cache: dict[str, dict | None] = {}

    @staticmethod
    def _build_ignore_regex(patterns: list[str]) -> re.Pattern | None:
        """Merge ignore patterns into one compiled regex.

        Uses :func:`~auto_round.utils.common.to_standard_regex` so that
        plain names are automatically wrapped with ``.*`` on both sides
        (substring matching) and regex meta-characters in user patterns
        are preserved — consistent with ``set_layer_config``.
        """
        if not patterns:
            return None
        parts: list[str] = []
        for p in patterns:
            if p.endswith("."):
                std = to_standard_regex(p.rstrip("."))
                std = std.removesuffix(".*")
                parts.append(f"{std}(?:\\.|$)")
            else:
                parts.append(to_standard_regex(p))
        return re.compile("|".join(parts))

    def should_ignore(self, tensor_name: str) -> bool:
        """Check user-specified ignore patterns (merged regex + cache)."""
        cached = self._ignore_cache.get(tensor_name)
        if cached is not None:
            return cached
        layer_name = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name
        result = bool(self._ignore_re and self._ignore_re.search(layer_name))
        self._ignore_cache[tensor_name] = result
        return result

    def should_skip(self, tensor_name: str) -> bool:
        """Check predefined skip patterns (routing gates, embeddings, etc.)."""
        return bool(self._skip_re and self._skip_re.search(tensor_name))

    def resolve_scheme(self, tensor_name: str) -> dict | None:
        """Resolve quantization scheme for *tensor_name* (cached).

        Returns ``None`` when the layer should stay in full precision.
        """
        if tensor_name in self._scheme_cache:
            return self._scheme_cache[tensor_name]
        result = self._resolve_uncached(tensor_name)
        self._scheme_cache[tensor_name] = result
        return result

    def _resolve_uncached(self, tensor_name: str) -> dict | None:
        layer_name = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name
        default = self._default_scheme

        if layer_name in self._layer_config:
            cfg = self._layer_config[layer_name]
            if cfg.get("bits", default.get("bits", 4)) >= 16:
                return None
            return {**default, **cfg}

        for compiled, plain, cfg in self._compiled_lc:
            if compiled is not None:
                if compiled.search(layer_name):
                    if cfg.get("bits", default.get("bits", 4)) >= 16:
                        return None
                    return {**default, **cfg}
            elif plain is not None and plain in layer_name:
                if cfg.get("bits", default.get("bits", 4)) >= 16:
                    return None
                return {**default, **cfg}

        return default


# ---------------------------------------------------------------------------
# Per-tensor / per-shard helpers
# ---------------------------------------------------------------------------


def _quantize_weight_mxfp(
    weight: torch.Tensor,
    layer_name: str,
    bits: int,
    group_size: int,
    data_type: str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Quantize a 2D weight tensor to MXFP4 / MXFP8 and return packed outputs.

    Reuses :func:`auto_round.data_type.mxfp.quant_mx` to derive the per-block
    shared exponent (E8M0 scale), and :class:`auto_round.export.export_to_autoround.qlinear_fp.QuantLinear`
    to perform the same packing as :func:`auto_round.export.export_to_llmcompressor.export_to_fp.pack_layer`.

    Returns a dict with one of:
      * MXFP8: ``{layer_name+'.weight': float8_e4m3fn, layer_name+'.weight_scale': uint8}``
      * MXFP4: ``{layer_name+'.weight_packed': uint8, layer_name+'.weight_scale': uint8}``
    """
    import torch.nn as nn

    from auto_round.data_type.mxfp import quant_mx
    from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

    if not is_mx_fp(data_type):
        data_type = "mx_fp"

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} for layer '{layer_name}' is not divisible "
            f"by MXFP group_size={group_size}; cannot pack."
        )

    weight_dev = weight.to(device)
    # quant_mx returns (qdq_tensor, shared_exp, None).  We only need shared_exp
    # (the per-block log2 scale).  The element-wise rounding to the FP4/FP8 grid
    # is performed inside QuantLinear.pack via dtype casts / pack_fp4_to_uint8.
    weight_dev, shared_exp, _ = quant_mx(weight_dev, bits=bits, group_size=group_size, data_type=data_type)
    # Reshape to (out_features, n_groups) so the on-disk weight_scale matches
    # the llm-compressor convention (and QuantLinear's registered buffer shape).
    shared_exp = shared_exp.reshape(out_features, in_features // group_size)
    # Ensure shared_exp is a numeric float (not a storage-specific dtype like
    # float8) — QuantLinear.pack performs `2 ** scales` which dispatches to
    # torch.pow; some backends do not implement pow for float8 dtypes. Cast to
    # float32 here to avoid runtime errors like "pow_cuda not implemented for
    # 'Float8_e4m3fn'" while preserving numeric values.
    shared_exp = shared_exp.to(torch.float32)

    # Build a lightweight nn.Linear holding the original weight so we can
    # delegate packing to the existing QuantLinear.pack implementation.
    fake_linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        fake_linear.weight = nn.Parameter(weight_dev, requires_grad=False)

    qlayer = QuantLinear(
        bits=bits,
        group_size=group_size,
        infeatures=in_features,
        outfeatures=out_features,
        bias=False,
        data_type="mx_fp4" if bits == 4 else "mx_fp8e4m3",
        sym=True,
        act_bits=bits,
    )
    qlayer.pack(fake_linear, shared_exp, device=device)

    if bits == 8:
        return {
            f"{layer_name}.weight": qlayer.weight.to("cpu"),
            f"{layer_name}.weight_scale": qlayer.weight_scale.to("cpu"),
        }
    return {
        f"{layer_name}.weight_packed": qlayer.weight_packed.to("cpu"),
        f"{layer_name}.weight_scale": qlayer.weight_scale.to("cpu"),
    }


def _quantize_single_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    matcher: "_PatternMatcher",
    device: str = "cpu",
) -> tuple[str, dict[str, torch.Tensor], str | None, str | None]:
    """Quantize one eligible weight tensor and return packed outputs.

    Returns:
        (layer_name, output_tensors_dict, quantized_layer_or_None, ignored_layer_or_None)
    """
    layer_name = tensor_name.rsplit(".", 1)[0]

    if not _is_eligible_weight(tensor_name, tensor):
        if tensor_name.endswith(".weight"):
            return layer_name, {tensor_name: tensor}, None, layer_name
        return layer_name, {tensor_name: tensor}, None, None

    if matcher.should_ignore(tensor_name):
        logger.debug(f"Ignoring (user-specified): {layer_name}")
        return layer_name, {tensor_name: tensor}, None, layer_name

    if matcher.should_skip(tensor_name):
        logger.debug(f"Skipping (predefined): {layer_name}")
        return layer_name, {tensor_name: tensor}, None, layer_name

    scheme = matcher.resolve_scheme(tensor_name)
    if scheme is None:
        logger.debug(f"Keeping full precision: {layer_name}")
        return layer_name, {tensor_name: tensor}, None, layer_name

    bits = scheme["bits"]
    group_size = scheme["group_size"]
    sym = scheme.get("sym", True)
    data_type = (scheme.get("data_type") or "int").lower()

    if bits >= 16:
        return layer_name, {tensor_name: tensor}, None, layer_name

    # ---- MXFP path (MXFP4 / MXFP8) ----
    if is_mx_fp(data_type):
        try:
            out = _quantize_weight_mxfp(
                weight=tensor,
                layer_name=layer_name,
                bits=bits,
                group_size=group_size,
                data_type=data_type,
                device=device,
            )
            logger.debug(f"Quantized (MXFP): {layer_name} (bits={bits}, group_size={group_size})")
            return layer_name, out, layer_name, None
        except Exception as e:
            logger.warning(f"Failed to MXFP-quantize {layer_name}: {e}. Keeping original weight.")
            return layer_name, {tensor_name: tensor}, None, layer_name

    # ---- Integer WOQ path ----
    try:
        qweight, qzeros, scales = quantize_weight_rtn(
            weight=tensor,
            bits=bits,
            group_size=group_size,
            sym=sym,
            device=device,
        )

        out: dict[str, torch.Tensor] = {
            f"{layer_name}.qweight": qweight,
            f"{layer_name}.qzeros": qzeros,
            f"{layer_name}.scales": scales,
        }

        logger.debug(f"Quantized: {layer_name} (bits={bits}, group_size={group_size}, sym={sym})")
        return layer_name, out, layer_name, None

    except Exception as e:
        logger.warning(f"Failed to quantize {layer_name}: {e}. Keeping original weight.")
        return layer_name, {tensor_name: tensor}, None, layer_name


def _collect_mxfp_source_entries(raw_tensors: dict[str, torch.Tensor]) -> list[tuple[str, str, str, int]]:
    """Collect MXFP source tensors present in a shard.

    Returns entries as ``(layer_name, weight_key, scale_key, bits)`` where
    ``bits`` is 8 for ``.weight`` (float8) and 4 for ``.weight_packed``.
    """
    entries: list[tuple[str, str, str, int]] = []
    for name, tensor in raw_tensors.items():
        if name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn:
            layer_name = name[: -len(".weight")]
            scale_key = f"{layer_name}.weight_scale"
            if scale_key in raw_tensors:
                entries.append((layer_name, name, scale_key, 8))
        elif name.endswith(".weight_packed") and tensor.dtype in (torch.int8, torch.uint8):
            layer_name = name[: -len(".weight_packed")]
            scale_key = f"{layer_name}.weight_scale"
            if scale_key in raw_tensors:
                entries.append((layer_name, name, scale_key, 4))
    return entries


def _is_out_of_memory_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _dequantize_with_device_fallback(
    *,
    dequant_device: str,
    shard_prefix: str,
    op_name: str,
    tensor_label: str,
    on_device: Callable[[], torch.Tensor],
    on_cpu: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Run dequantization on ``dequant_device`` and fall back to CPU on errors."""
    if dequant_device != "cpu":
        try:
            return on_device()
        except Exception as e:
            if _is_out_of_memory_error(e):
                logger.warning(
                    f"{shard_prefix}{op_name} on {dequant_device} ran OOM for {tensor_label}: {e}. "
                    "Clearing accelerator memory and falling back to CPU for this tensor."
                )
                clear_memory()
            else:
                logger.warning(
                    f"{shard_prefix}{op_name} on {dequant_device} failed for {tensor_label}: {e}. "
                    "Falling back to CPU for this tensor."
                )
    return on_cpu()


def _dequant_mxfp_tensors(
    raw_tensors: dict[str, torch.Tensor],
    device: str = "cpu",
    shard_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Dequantize llm-compressor MXFP8 / MXFP4 weight tensors to bfloat16.

    Detection is purely by *name* and *dtype*, reusing the dequant kernels in
    :mod:`auto_round_extension.vllm_ext`:

    * ``<layer>.weight`` (``float8_e4m3fn``) + ``<layer>.weight_scale`` → MXFP8,
      dequantized via :func:`~auto_round_extension.vllm_ext.mxfp8_qdq_utils.dequant_mx_fp8`.
    * ``<layer>.weight_packed`` (``uint8``) + ``<layer>.weight_scale`` → MXFP4,
      dequantized via :func:`~auto_round_extension.vllm_ext.mxfp4_qdq_utils.to_dtype`.

    The dequantized weight is written back under ``<layer>.weight`` and the
    scale (and any ``weight_packed``) tensor is removed, so the downstream RTN
    path can requantize the layer to the requested target scheme.
    """
    from auto_round_extension.vllm_ext.mxfp4_qdq_utils import to_dtype
    from auto_round_extension.vllm_ext.mxfp8_qdq_utils import dequant_mx_fp8

    # Tuple layout: (layer_name, weight_key, scale_key, bits)
    entries = _collect_mxfp_source_entries(raw_tensors)

    if not entries:
        return raw_tensors

    n_mxfp8 = sum(1 for _layer_name, _weight_key, _scale_key, bits in entries if bits == 8)
    n_mxfp4 = len(entries) - n_mxfp8
    dequant_device = str(device or "cpu")
    shard_prefix = f"[{shard_name}] " if shard_name else ""
    logger.info(
        f"{shard_prefix}Dequantizing MXFP tensor(s) to bfloat16 on {dequant_device}: "
        f"MXFP8={n_mxfp8}, MXFP4={n_mxfp4}, total={len(entries)}."
    )

    for layer_name, weight_key, scale_key, bits in entries:
        weight = raw_tensors.pop(weight_key)
        scale = raw_tensors.pop(scale_key).view(torch.uint8)
        if bits == 8:
            dq_weight = _dequantize_with_device_fallback(
                dequant_device=dequant_device,
                shard_prefix=shard_prefix,
                op_name="MXFP dequant",
                tensor_label=layer_name,
                on_device=lambda: dequant_mx_fp8(
                    weight_fp8=weight.to(dequant_device, non_blocking=True),
                    scale_e8m0=scale.to(dequant_device, non_blocking=True),
                    block_size=32,
                    target_dtype=torch.bfloat16,
                ).to("cpu"),
                on_cpu=lambda: dequant_mx_fp8(
                    weight_fp8=weight,
                    scale_e8m0=scale,
                    block_size=32,
                    target_dtype=torch.bfloat16,
                ),
            )
        else:
            dq_weight = _dequantize_with_device_fallback(
                dequant_device=dequant_device,
                shard_prefix=shard_prefix,
                op_name="MXFP dequant",
                tensor_label=layer_name,
                on_device=lambda: to_dtype(
                    data_lp=weight.view(torch.uint8).contiguous().to(dequant_device, non_blocking=True),
                    scale_e8m0=scale.to(dequant_device, non_blocking=True),
                    elem_dtype="fp4_e2m1",
                    block_size=32,
                    target_dtype=torch.bfloat16,
                ).to("cpu"),
                on_cpu=lambda: to_dtype(
                    data_lp=weight.view(torch.uint8).contiguous(),
                    scale_e8m0=scale,
                    elem_dtype="fp4_e2m1",
                    block_size=32,
                    target_dtype=torch.bfloat16,
                ),
            )
        raw_tensors[f"{layer_name}.weight"] = dq_weight

    return raw_tensors


def _handle_mxfp_source_tensors(
    raw_tensors: dict[str, torch.Tensor],
    matcher: "_PatternMatcher",
    source_state: dict[str, int] | None = None,
    device: str = "cpu",
    shard_name: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    """Handle llm-compressor-style MXFP8/MXFP4 source tensors.

    Detects llm-compressor MXFP tensors purely by *name + dtype*:

    * ``<layer>.weight`` (``float8_e4m3fn``) + ``<layer>.weight_scale`` → MXFP8.
    * ``<layer>.weight_packed`` (``uint8``) + ``<layer>.weight_scale`` → MXFP4.

    For each detected layer the effective target scheme is resolved via *matcher*:

    * If the target is the **same MXFP format** (``data_type='mx_fp'``, matching
      ``bits``), the tensors are emitted directly as a passthrough — no
      dequantization is performed and the layer is recorded as already quantized.
    * Otherwise the tensors are dequantized to ``bfloat16`` via
      :func:`_dequant_mxfp_tensors` so the downstream RTN path can re-quantize
      them to the requested target scheme.

    Returns:
        ``(raw_tensors, passthrough_tensors, passthrough_layers)``.
    """
    entries = _collect_mxfp_source_entries(raw_tensors)
    if not entries:
        return raw_tensors, {}, []

    source_state = source_state or {}

    passthrough_tensors: dict[str, torch.Tensor] = {}
    passthrough_layers: list[str] = []
    n_dequant = 0

    for layer_name, weight_key, scale_key, bits in entries:
        scheme = matcher.resolve_scheme(f"{layer_name}.weight")
        target_is_same_mxfp = (
            scheme is not None and is_mx_fp((scheme.get("data_type") or "").lower()) and scheme.get("bits") == bits
        )
        if target_is_same_mxfp:
            passthrough_tensors[weight_key] = raw_tensors.pop(weight_key).to("cpu")
            passthrough_tensors[scale_key] = raw_tensors.pop(scale_key).to("cpu")
            passthrough_layers.append(layer_name)
        else:
            n_dequant += 1

    if n_dequant:
        raw_tensors = _dequant_mxfp_tensors(raw_tensors, device=device, shard_name=shard_name)

    parts: list[str] = []
    if passthrough_layers:
        parts.append(f"{len(passthrough_layers)} passthrough")
    if n_dequant:
        parts.append(f"{n_dequant} dequantized to bfloat16")
    if source_state:
        parts.append(f"{len(source_state)} model_type-normalized")
    logger.info(f"Handling MXFP source tensor(s): {', '.join(parts)}.")

    return raw_tensors, passthrough_tensors, passthrough_layers


def _dequant_fp8_tensors(
    raw_tensors: dict[str, torch.Tensor],
    block_size: list | None = None,
    device: str = "cpu",
    shard_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """Dequantize DeepSeek-V3-style FP8 weight tensors to bfloat16.

    Handles the **DeepSeek-V3 FP8** convention: weight dtype ``float8_e4m3fn``
    paired with a ``.weight_scale_inv`` tensor (per-block float32 scales, NOT
    E8M0).  The weights are converted to ``bfloat16`` so downstream RTN
    quantization can proceed normally.

    MXFP sources are handled separately by
    :func:`_preprocess_model_type_source_tensors` / :func:`_handle_mxfp_source_tensors`.
    """
    from auto_round.utils.weight_handler import _dequant_fp8_linear_weight

    quant_entries: list[tuple[str, str]] = []
    for name, tensor in raw_tensors.items():
        if not name.endswith(".weight"):
            continue
        if tensor.dtype != torch.float8_e4m3fn and tensor.element_size() != 1:
            continue
        # DeepSeek-V3 style: .weight_scale_inv (per-block float32 scales).
        scale_inv_name = name.replace(".weight", ".weight_scale_inv")
        if scale_inv_name in raw_tensors:
            quant_entries.append((name, scale_inv_name))

    if not quant_entries:
        return raw_tensors

    # device has already been resolved by the caller; use it directly here.
    dequant_device = str(device or "cpu")
    shard_prefix = f"[{shard_name}] " if shard_name else ""

    logger.info(
        f"{shard_prefix}Dequantizing {len(quant_entries)} FP8 weight tensor(s) to bfloat16 on {dequant_device}."
    )

    for weight_name, scale_name in quant_entries:
        weight = raw_tensors[weight_name]
        scale = raw_tensors.pop(scale_name)

        # Dequantize on GPU for throughput, then move back to CPU to keep
        # per-shard memory usage bounded before per-layer quantization.
        raw_tensors[weight_name] = _dequantize_with_device_fallback(
            dequant_device=dequant_device,
            shard_prefix=shard_prefix,
            op_name="FP8 dequant",
            tensor_label=weight_name,
            on_device=lambda: _dequant_fp8_linear_weight(
                weight.to(dequant_device, non_blocking=True),
                scale.to(dequant_device, non_blocking=True),
                block_size=block_size,
            ).to("cpu"),
            on_cpu=lambda: _dequant_fp8_linear_weight(weight, scale, block_size=block_size),
        )

    return raw_tensors


def _process_shard(
    shard_path: str,
    default_scheme: dict = None,
    layer_config: dict = None,
    ignore_patterns: list[str] = None,
    device: str = "cpu",
    *,
    shard_name: str | None = None,
    matcher: "_PatternMatcher | None" = None,
    fp8_block_size: list | None = None,
    model_type: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Quantize eligible weights in a single safetensors shard.

    Returns:
        (output_tensors, quantized_layer_names, ignored_layer_names)

    ``ignored_layer_names`` is derived by comparing the set of input ``.weight``
    layer names (collected after fused-expert splitting) with the final set of
    quantized layer names.  Any layer that had a ``.weight`` tensor in the input
    but was NOT quantized is reported as ignored — this correctly captures
    user-ignored layers, predefined-skipped layers, non-eligible weights, and
    any other pass-through case without separate per-tensor tracking.
    """
    if matcher is None:
        matcher = _PatternMatcher(
            ignore_patterns if ignore_patterns is not None else [],
            layer_config if layer_config is not None else {},
            default_scheme if default_scheme is not None else {},
        )

    output_tensors: dict[str, torch.Tensor] = {}
    quantized_layers: list[str] = []

    if shard_path.endswith(".bin"):
        # PyTorch pickle checkpoint — load with weights_only where supported.
        try:
            raw_tensors = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:
            # weights_only not available in older PyTorch versions
            raw_tensors = torch.load(shard_path, map_location="cpu")  # nosec
        # Flatten nested state-dict wrappers if present.
        if not isinstance(raw_tensors, dict):
            raise ValueError(f"Expected a dict from {shard_path}, got {type(raw_tensors)}")
    else:
        from safetensors import safe_open

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            raw_tensors = {name: f.get_tensor(name) for name in f.keys()}

    raw_tensors = split_fused_expert_tensors(raw_tensors)

    # Snapshot eligible weight layer names *before* any preprocessing so that
    # the ignored-layer list can be derived by dict comparison at the end.
    input_weight_layers: list[str] = list(
        dict.fromkeys(k.rsplit(".", 1)[0] for k in raw_tensors if k.endswith(".weight"))
    )

    # Preserve original tensors for ignored/skipped layers so that already-
    # quantized weights (FP8, FP4-packed, etc.) are NOT dequantized.
    # Check both ".weight" and ".weight_packed" so that layers whose primary
    # tensor uses non-standard naming (e.g. already-quantized FP4-packed layers
    # stored as ".weight_packed") are correctly captured.
    preserved_prefixes: set[str] = set()
    for tname in raw_tensors:
        if (tname.endswith(".weight") or tname.endswith(".weight_packed") or tname.endswith(".qweight")) and (
            matcher.should_ignore(tname) or matcher.should_skip(tname)
        ):
            preserved_prefixes.add(tname.rsplit(".", 1)[0])

    preserved_tensors: dict[str, torch.Tensor] = {}
    if preserved_prefixes:
        for key in list(raw_tensors.keys()):
            prefix = key.rsplit(".", 1)[0]
            if prefix in preserved_prefixes:
                preserved_tensors[key] = raw_tensors.pop(key)

    # 1) model-type-specific preprocessing (format conversion only)
    raw_tensors, source_state = _preprocess_model_type_source_tensors(raw_tensors, model_type=model_type)

    # 2) generic MXFP handling for both preprocessed and normal source models
    raw_tensors, passthrough_tensors, passthrough_layers = _handle_mxfp_source_tensors(
        raw_tensors,
        matcher,
        source_state=source_state,
        device=device,
        shard_name=shard_name,
    )
    output_tensors.update(passthrough_tensors)
    quantized_layers.extend(passthrough_layers)

    raw_tensors = _dequant_fp8_tensors(
        raw_tensors,
        block_size=fp8_block_size,
        device=device,
        shard_name=shard_name,
    )
    raw_tensors.update(preserved_tensors)

    for tensor_name in list(raw_tensors.keys()):
        tensor = raw_tensors.pop(tensor_name)
        _layer_name, out_dict, q_layer, _ig_layer = _quantize_single_tensor(
            tensor_name,
            tensor,
            matcher,
            device,
        )
        output_tensors.update(out_dict)
        if q_layer:
            quantized_layers.append(q_layer)

    # Derive ignored layers by comparing input weight layers with quantized set.
    quantized_set = set(quantized_layers)
    ignored_layers: list[str] = [l for l in input_weight_layers if l not in quantized_set]

    return output_tensors, quantized_layers, ignored_layers


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _build_mxfp_quantization_config(
    default_scheme: dict,
    quantized_layers: list[str],
    ignored_layers: list[str],
    layer_config: dict | None = None,
) -> dict:
    """Build a compressed-tensors / llm-compressor style quantization_config
    dict for MXFP4 / MXFP8 model-free output, including mixed-precision cases.

    When *layer_config* contains layers that override the default bits (e.g.
    some layers are MXFP8 while the default is MXFP4), the function creates
    one ``config_group`` per distinct bit-width.  Override groups list their
    layers explicitly; the default-bits group uses ``targets=["Linear"]`` as a
    catch-all.  The top-level ``"format"`` is set to ``"mixed-precision"``
    when more than one group is produced.

    Mirrors the per-group format produced by
    :mod:`auto_round.export.export_to_llmcompressor.export_to_fp`.
    """
    from auto_round.export.export_to_llmcompressor.config import (
        check_compressed_tensors_supported,
        initialize_quantization,
    )

    check_compressed_tensors_supported(raise_error=True)

    bits = default_scheme["bits"]
    if bits not in _SUPPORTED_MXFP_BITS:
        raise ValueError(f"Unsupported MXFP bits={bits} for model-free output.")

    # Default ignore list: any layer present in ignored_layers (deduped) that
    # was NOT quantized.
    ignore = list(dict.fromkeys(ignored_layers))
    quant_set = set(quantized_layers)
    ignore = [n for n in ignore if n not in quant_set]

    # Resolve each quantized layer's effective bits using layer_config overrides.
    scheme_groups: dict[int, list[str]] = {}  # bits -> [layer_names]
    if layer_config:
        temp_matcher = _PatternMatcher(
            ignore_patterns=[],
            layer_config=layer_config,
            default_scheme=default_scheme,
        )
        for layer in quantized_layers:
            scheme = temp_matcher.resolve_scheme(f"{layer}.weight")
            layer_bits = scheme.get("bits", bits) if scheme is not None else bits
            scheme_groups.setdefault(layer_bits, []).append(layer)
    else:
        scheme_groups[bits] = list(quantized_layers)

    if len(scheme_groups) <= 1:
        # Single scheme — existing behavior.
        scheme_name = "MXFP4" if bits == 4 else "MXFP8"
        fmt = "mxfp4-pack-quantized" if bits == 4 else "mxfp8-quantized"
        qconfig = initialize_quantization(scheme=scheme_name, ignore=ignore)
        qconfig = qconfig.to_dict()
        qconfig["format"] = fmt
        qconfig["provider"] = "auto-round"
        return qconfig

    # Mixed MXFP: build one config_group per distinct bit-width.
    # Override groups (non-default bits) come first, default group last,
    # ordered by descending bit-width within each partition so that the
    # higher-precision group gets the lower group index.
    override_items = sorted(
        [(b, layers) for b, layers in scheme_groups.items() if b != bits],
        key=lambda x: x[0],
        reverse=True,
    )
    default_item = (bits, scheme_groups[bits]) if bits in scheme_groups else None
    ordered = override_items + ([default_item] if default_item else [])

    config_groups: dict = {}
    group_formats: dict[str, str] = {}
    for idx, (group_bits, layer_names) in enumerate(ordered):
        group_name = f"group_{idx}"
        scheme_name = "MXFP4" if group_bits == 4 else "MXFP8"
        fmt = "mxfp4-pack-quantized" if group_bits == 4 else "mxfp8-quantized"
        is_default_group = group_bits == bits
        targets = ["Linear"] if is_default_group else layer_names
        # vLLM MoE: prepend RoutedExperts so vLLM's routed-expert matcher
        # takes priority when this explicit group contains expert layers.
        if not is_default_group and any(".experts." in n for n in layer_names):
            targets = ["RoutedExperts"] + targets
        tmp_qconfig = initialize_quantization(scheme=scheme_name, ignore=ignore)
        group_scheme = tmp_qconfig.config_groups["group_0"]
        group_scheme.targets = targets
        config_groups[group_name] = group_scheme
        group_formats[group_name] = fmt

    full_qconfig = initialize_quantization(scheme=None, config_groups=config_groups, ignore=ignore)
    full_dict = full_qconfig.to_dict()
    full_dict["format"] = "mixed-precision"
    for group_name, fmt in group_formats.items():
        full_dict["config_groups"][group_name]["format"] = fmt
    full_dict["provider"] = "auto-round"
    return full_dict


def _build_quantization_config(
    default_scheme: dict,
    layer_config: dict,
    ignore_patterns: list[str],
    quantized_layers: list[str],
    ignored_layers: list[str],
    block_name_to_quantize: Optional[list[str]] = None,
) -> dict:
    """Build a quantization_config dict compatible with auto-round format."""
    # MXFP (mx_fp) uses the llm-compressor / compressed-tensors style config.
    if is_mx_fp((default_scheme.get("data_type") or "int").lower()):
        return _build_mxfp_quantization_config(
            default_scheme=default_scheme,
            quantized_layers=quantized_layers,
            ignored_layers=ignored_layers,
            layer_config=layer_config,
        )

    from auto_round.version import __version__

    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    # vllm only support auto_round:auto_gptq, but transformers cannot load it correctly when sym=False.
    # So we keep auto_round for asymmetric quantization to maintain compatibility with both.
    packing_format = "auto_round:auto_gptq" if default_scheme.get("sym", True) else "auto_round"

    qconfig = {
        "quant_method": "auto-round",
        "packing_format": packing_format,
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

    extra_config = {}
    for layer_name, cfg in layer_config.items():
        if cfg.get("bits", default_scheme["bits"]) >= 16:
            extra_config[layer_name] = {k: cfg.get(k) for k in scheme_keys if cfg.get(k) is not None}
            continue
        differs = False
        for key in ("bits", "group_size", "sym"):
            if cfg.get(key) is not None and cfg[key] != default_scheme.get(key):
                differs = True
                break
        if differs:
            extra_config[layer_name] = {k: cfg.get(k) for k in scheme_keys if cfg.get(k) is not None}

    # Filter out non-Linear ops (embed, conv) that don't need to be recorded in config.
    # Routing gates and other predefined patterns are still recorded.
    non_linear_ops = ["embed", "conv"]
    non_linear_re = re.compile("|".join(re.escape(op) for op in non_linear_ops))

    unique_ignored = list(dict.fromkeys(ignored_layers))
    for layer_name in unique_ignored:
        if layer_name not in extra_config:
            # Skip non-Linear ops (embed, conv) since they're not Linear layers
            if non_linear_re.search(layer_name):
                continue
            extra_config[layer_name] = {"bits": 16, "data_type": "float"}

    quantized_layer_set = set(quantized_layers)
    if "lm_head" in quantized_layer_set and "lm_head" not in extra_config:
        lm_head_cfg = layer_config.get("lm_head", default_scheme)
        extra_config["lm_head"] = {k: lm_head_cfg.get(k) for k in scheme_keys if lm_head_cfg.get(k) is not None}

    if extra_config:
        qconfig["extra_config"] = extra_config

    return qconfig


def _write_output_shard(
    output_dir: str,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str],
):
    """Write a single output shard and update the weight_map."""
    from safetensors.torch import save_file

    shard_path = os.path.join(output_dir, shard_name)

    # Detect shared-storage tensors (e.g. tie_word_embeddings: wte ↔ lm_head).
    # safetensors refuses to serialise them as-is; clone the duplicates so each
    # tensor occupies its own memory region.  The first occurrence keeps the
    # original storage; subsequent aliases are cloned.
    seen_data_ptrs: set[int] = set()
    deduped: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if not v.is_contiguous():
            v = v.contiguous()
        ptr = v.data_ptr()
        if ptr in seen_data_ptrs:
            v = v.clone()
        else:
            seen_data_ptrs.add(ptr)
        deduped[k] = v

    save_file(deduped, shard_path)
    for tensor_name in tensors:
        weight_map[tensor_name] = shard_name


def _write_index_file(output_dir: str, weight_map: dict[str, str]):
    """Write model.safetensors.index.json (or rename single shard)."""
    if len(set(weight_map.values())) <= 1:
        shard_names = list(set(weight_map.values()))
        if shard_names and shard_names[0] != "model.safetensors":
            src = os.path.join(output_dir, shard_names[0])
            dst = os.path.join(output_dir, "model.safetensors")
            if os.path.exists(src):
                os.rename(src, dst)
            weight_map = {k: "model.safetensors" for k in weight_map}
        return

    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


def _prefetch_shard(
    model_name_or_path: str,
    shard_name: str,
    work_dir: str,
    source_dir: str,
    streaming: bool,
) -> str | None:
    """Return the local path of the next shard (download if needed)."""
    try:
        if streaming:
            # Keep source shards in a dedicated cache directory to avoid
            # colliding with quantized output shard names in output_dir.
            shard_cache_dir = os.path.join(work_dir, ".cache", "model_free_source_shards")
            return _download_single_shard(model_name_or_path, shard_name, shard_cache_dir)
        path = os.path.join(source_dir, shard_name)
        return path if os.path.exists(path) else None
    except Exception as e:  # pragma: no cover
        logger.warning(f"Prefetch failed for {shard_name}: {e}")
        return None


def _process_single_shard_task(
    shard_idx: int,
    shard_name: str,
    *,
    model_name_or_path: str,
    work_dir: str,
    source_dir: str,
    is_streaming: bool,
    device: str,
    default_scheme: dict,
    layer_config: dict,
    ignore_patterns: list[str],
    fp8_block_size: list | None,
    model_type: str | None,
    quant_output_dir: str,
    total_shards: int,
) -> tuple[int, str, str | None, str | None, list[str] | None, list[str] | None, list[str] | None]:
    """Process one shard in an isolated subprocess task.

    Each worker builds its own matcher/cache via ``_process_shard`` to avoid
    cross-shard shared state.
    """
    shard_path = _prefetch_shard(
        model_name_or_path,
        shard_name,
        work_dir,
        source_dir,
        is_streaming,
    )
    if shard_path is None or not os.path.exists(shard_path):
        return shard_idx, shard_name, None, None, None, None, None

    output_tensors, quantized, ignored = _process_shard(
        shard_path=shard_path,
        shard_name=shard_name,
        default_scheme=default_scheme,
        layer_config=layer_config,
        ignore_patterns=ignore_patterns,
        device=device,
        fp8_block_size=fp8_block_size,
        model_type=model_type,
    )

    out_shard_name = f"model-{shard_idx + 1:05d}-of-{total_shards:05d}.safetensors"
    local_weight_map: dict[str, str] = {}
    _write_output_shard(
        quant_output_dir,
        out_shard_name,
        output_tensors,
        local_weight_map,
    )
    tensor_names = list(local_weight_map.keys())
    clear_memory()

    if is_streaming:
        try:
            os.remove(shard_path)
        except OSError:
            pass

    # Return only lightweight metadata to avoid IPC transfer of tensor storages.
    return shard_idx, shard_name, shard_path, out_shard_name, tensor_names, quantized, ignored


def _force_cleanup_process_pool(pool: ProcessPoolExecutor | None) -> None:
    """Best-effort cleanup for process-pool workers.

    On interruption (Ctrl+C / SIGTERM) or executor failures, worker processes
    may survive briefly. This helper force-terminates workers before shutting
    the executor down.
    """
    if pool is None:
        return

    # Accessing _processes is intentionally best-effort for robust cleanup.
    # pylint: disable=protected-access
    processes = getattr(pool, "_processes", None)
    if isinstance(processes, dict):
        for proc in processes.values():
            if proc is not None and proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass

    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scheme validation
# ---------------------------------------------------------------------------


def _normalize_scheme(scheme: Union[str, QuantizationScheme]) -> QuantizationScheme:
    """Convert *scheme* to a :class:`QuantizationScheme` instance.

    Raises ``ValueError`` for unknown preset names and ``TypeError`` for
    unsupported types.
    """
    if isinstance(scheme, str):
        scheme_name = scheme.upper()
        if scheme_name not in PRESET_SCHEMES:
            raise ValueError(f"Unknown scheme '{scheme}'. Available: {list(PRESET_SCHEMES.keys())}")
        return preset_name_to_scheme(scheme_name)
    if isinstance(scheme, QuantizationScheme):
        return scheme
    raise TypeError(f"Unsupported scheme type: {type(scheme)}")


def _apply_scheme_overrides(
    scheme: Union[str, QuantizationScheme],
    scheme_overrides: Optional[dict] = None,
) -> QuantizationScheme:
    """Return the effective scheme after applying non-None overrides."""
    scheme_obj = copy.deepcopy(_normalize_scheme(scheme))
    if not scheme_overrides:
        return scheme_obj

    valid_fields = {field.name for field in fields(QuantizationScheme)}
    for key, value in scheme_overrides.items():
        if key in valid_fields and value is not None:
            setattr(scheme_obj, key, value)
    return scheme_obj


def _validate_supported_scheme(
    scheme_obj: QuantizationScheme,
    scheme_input: Union[str, QuantizationScheme],
) -> None:
    """Raise ``ValueError`` if *scheme_obj* is not supported by model-free.

    Model-free supports:

    * Integer weight-only quantization (sym/asym), ``bits ∈ {2, 4, 8}``,
      packed in the ``auto_round:auto_gptq`` format.
    * MXFP weight quantization (``data_type='mx_fp'``), ``bits ∈ {4, 8}``,
      ``group_size=32``, packed in ``mxfp4-pack-quantized`` / ``mxfp8-quantized``
      format (compressed-tensors compatible).
    """
    data_type = (scheme_obj.data_type or "int").lower()
    bits = scheme_obj.bits
    act_bits = scheme_obj.act_bits if scheme_obj.act_bits is not None else 16

    # MXFP weight-only path: accept mx_fp data type with bits in {4, 8}.
    # Activation quantization for MXFP is dynamic at inference time, so the
    # weight-only RTN path here is independent of act_bits.
    if is_mx_fp(data_type):
        # Restrict to the two explicitly supported MXFP presets when a string
        # name is provided.  Variants such as MXFP4_RCEIL / MXFP8_RCEIL use a
        # different activation format; silently mapping them to "MXFP4" /
        # "MXFP8" in the output config would misrepresent the requested scheme.
        if isinstance(scheme_input, str) and scheme_input not in ("MXFP4", "MXFP8"):
            raise ValueError(
                f"Model-free mode only supports MXFP preset names 'MXFP4' and 'MXFP8', "
                f"but got '{scheme_input}'. "
                f"Supported preset schemes: {list(SUPPORTED_PRESET_SCHEMES)}."
            )
        if bits is None or bits not in _SUPPORTED_MXFP_BITS:
            raise ValueError(
                f"Model-free mode supports MXFP bits in {_SUPPORTED_MXFP_BITS}, "
                f"but '{scheme_input}' requests bits={bits}. "
                f"Supported preset schemes: {list(SUPPORTED_PRESET_SCHEMES)}."
            )
        group_size = scheme_obj.group_size
        if group_size not in (None, 32):
            raise ValueError(
                f"Model-free mode supports MXFP only with group_size=32, "
                f"but '{scheme_input}' requests group_size={group_size}."
            )
        return

    if act_bits < 16:
        raise ValueError(
            f"Model-free mode only supports weight-only quantization (WOQ) schemes "
            f"where act_bits >= 16, but '{scheme_input}' has act_bits={act_bits}. "
            f"Supported preset schemes: {list(SUPPORTED_PRESET_SCHEMES)}."
        )

    if data_type != "int":
        raise ValueError(
            f"Model-free mode only supports integer weight quantization "
            f"(data_type='int') or MXFP (data_type='mx_fp'), but '{scheme_input}' "
            f"has data_type='{data_type}'. FP8 / NVFP / GGUF / BF16 schemes require "
            f"the standard AutoRound flow.  Supported preset schemes: "
            f"{list(SUPPORTED_PRESET_SCHEMES)}."
        )

    if bits is None or bits not in _SUPPORTED_INT_BITS:
        raise ValueError(
            f"Model-free mode supports bits in {_SUPPORTED_INT_BITS}, "
            f"but '{scheme_input}' requests bits={bits}. "
            f"Supported preset schemes: {list(SUPPORTED_PRESET_SCHEMES)}."
        )


def is_model_free_supported_scheme(
    scheme: Union[str, QuantizationScheme],
    scheme_overrides: Optional[dict] = None,
) -> bool:
    """Return True if *scheme* can be quantized via model-free mode.

    Useful for CLI auto-routing logic.  Never raises.
    """
    try:
        scheme_obj = _apply_scheme_overrides(scheme, scheme_overrides)
        _validate_supported_scheme(scheme_obj, scheme)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# AutoScheme support (two-phase: delta-loss selection + model-free packing)
# ---------------------------------------------------------------------------


def _looks_like_auto_scheme(scheme: Any) -> bool:
    """Duck-typed check for an :class:`~auto_round.auto_scheme.AutoScheme`.

    Avoids importing ``AutoScheme`` at module scope (it pulls in exporter /
    compressor modules that would create an import cycle with this file).
    """
    return hasattr(scheme, "options") and hasattr(scheme, "avg_bits")


def _validate_auto_scheme_options(auto_scheme: Any) -> str:
    """Validate that every AutoScheme option is model-free-packable.

    Returns the single data-type family shared by all options
    (``"int"`` or ``"mx_fp"``).  Raises ``ValueError`` when any option is
    unsupported or when INT and MXFP options are mixed (they use different
    packing formats and cannot be produced in one model-free run).
    """
    options = list(getattr(auto_scheme, "options", []) or [])
    if not options:
        raise ValueError("AutoScheme.options must be non-empty for model-free mode.")

    families: set[str] = set()
    unsupported: list[Any] = []
    for opt in options:
        # Preserve original string validation semantics so preset-name
        # restrictions (e.g. MXFP4/MXFP8 only) are enforced.
        if isinstance(opt, str):
            try:
                scheme_obj = _normalize_scheme(opt)
            except (ValueError, TypeError):
                scheme_obj = None
        elif isinstance(opt, QuantizationScheme):
            scheme_obj = opt
        else:
            scheme_obj = None

        # GGUF k-quants carry super_bits and are not packable by the model-free
        # RTN kernel even though their data_type is nominally "int".
        if scheme_obj is None or getattr(scheme_obj, "super_bits", None) is not None:
            unsupported.append(opt)
            continue
        if not is_model_free_supported_scheme(opt):
            unsupported.append(opt)
            continue

        data_type = (scheme_obj.data_type or "int").lower()
        families.add("mx_fp" if is_mx_fp(data_type) else "int")

    if unsupported:
        raise ValueError(
            f"Model-free + AutoScheme received unsupported option(s): {unsupported}. "
            f"Model-free supports INT WOQ (bits in {_SUPPORTED_INT_BITS}) and MXFP "
            f"(bits in {_SUPPORTED_MXFP_BITS}); GGUF / NVFP4 / FP8 options are not "
            f"packable in model-free mode. Remove the unsupported options or pass "
            f"disable_model_free=True to use the regular flow."
        )
    if len(families) > 1:
        raise ValueError(
            "Model-free + AutoScheme cannot mix INT and MXFP options in a single run "
            f"(got families {sorted(families)}); INT and MXFP use different packing "
            "formats. Use a single data-type family, or pass disable_model_free=True."
        )
    return families.pop()


def _convert_auto_scheme_layer_config(
    generated: dict[str, dict],
) -> tuple[QuantizationScheme, dict[str, dict], list[str]]:
    """Convert an AutoScheme-generated ``layer_config`` into model-free inputs.

    Returns ``(base_scheme, per_layer_overrides, fp16_layers)`` where:

    * ``base_scheme`` is the most common quantized scheme across layers, used
      as the model-free default (top-level config.json ``bits``/``group_size``).
    * ``per_layer_overrides`` maps every quantized layer name to its resolved
      :class:`QuantizationScheme` fields.
    * ``fp16_layers`` lists layers AutoScheme kept at >= 16 bits (added to the
      model-free ignore list so they stay in full precision).
    """
    from collections import Counter

    scheme_keys = {f.name for f in fields(QuantizationScheme)}
    per_layer: dict[str, dict] = {}
    fp16_layers: list[str] = []
    counter: "Counter[tuple]" = Counter()

    for name, cfg in generated.items():
        if not isinstance(cfg, dict):
            continue
        bits = cfg.get("bits")
        if bits is None:
            continue
        clean = {k: cfg[k] for k in scheme_keys if cfg.get(k) is not None}
        if bits >= 16:
            fp16_layers.append(name)
            continue
        data_type = (clean.get("data_type") or "int").lower()
        per_layer[name] = clean
        counter[(clean.get("bits"), clean.get("group_size"), bool(clean.get("sym", True)), data_type)] += 1

    if not counter:
        raise ValueError("AutoScheme did not assign any quantizable layers for model-free mode.")

    (base_bits, base_group_size, base_sym, base_dtype), _ = counter.most_common(1)[0]
    base_scheme = QuantizationScheme(
        bits=base_bits,
        group_size=base_group_size,
        sym=base_sym,
        data_type=base_dtype,
    )
    return base_scheme, per_layer, fp16_layers


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class _ModelFreeCompressorCore:
    """Class-based driver for model-free RTN quantization.

    The lifecycle is:

    1. ``__init__`` — store user inputs.
    2. :meth:`run` — perform validation, IO, quantization and writing.

    Internal helpers are split into focused methods so that the flow is
    readable end-to-end.

    Args:
        model_name_or_path: HuggingFace model ID or local directory path.
        output_dir: Directory to save the quantized model.
        scheme: Quantization scheme name (e.g. ``"W4A16"``, ``"MXFP4"``,
            ``"MXFP8"``) or a :class:`QuantizationScheme` instance.
        layer_config: Per-layer quantization overrides.  Keys are layer
            names or regex patterns; values are dicts with ``bits``,
            ``group_size``, ``sym`` etc.
        ignore_layers: Comma-separated list of layer name patterns to keep
            in full precision.  Ignored layers that are already quantized
            (e.g. FP8) are preserved in their original format.
        format: Output format.  Supported: ``"auto_round"``,
            ``"auto_round:auto_gptq"``, ``"llm_compressor"``,
            ``"auto_round:llm_compressor"``.  The packing format is
            auto-selected based on the scheme (INT→auto_gptq,
            MXFP→compressed-tensors).
        device: Device for quantization computation (``"cpu"`` or
            ``"cuda"``).
        quant_lm_head: If True, quantize ``lm_head`` as well.  By default
            ``lm_head`` and any layer containing ``embed`` are kept in
            full precision.
        quant_nontext_module: If True, quantize non-text modules
            (vision/audio/image) as well.  By default these multimodal
            modules are kept in full precision.
    """

    SUPPORTED_FORMATS: tuple[str, ...] = (
        "auto_round",
        "auto_round:auto_gptq",
        "llm_compressor",
        "auto_round:llm_compressor",
    )

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        scheme: Union[str, QuantizationScheme] = "W4A16",
        layer_config: Optional[dict] = None,
        ignore_layers: str = "",
        format: str = "auto_round",
        device: str = "cpu",
        quant_lm_head: bool = False,
        quant_nontext_module: bool = False,
    ) -> None:
        # --- raw inputs ---
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.scheme_input = scheme
        self.layer_config_input = layer_config
        self.ignore_layers_input = ignore_layers
        self.format = format
        self.device = device
        self.quant_lm_head = quant_lm_head
        self.quant_nontext_module = quant_nontext_module

        # --- derived state populated during run() ---
        self.scheme_obj: QuantizationScheme | None = None
        self.default_scheme: dict = {}
        self.layer_config: dict = {}
        self.ignore_patterns: list[str] = []
        self.config: dict = {}
        self.fp8_block_size: list | None = None
        self.model_type: str = ""
        self.is_streaming: bool = False
        self.is_diffusion_model: bool = False
        self.diffusion_root_dir: str = ""
        self.work_dir: str = ""
        self.source_dir: str = ""
        self.shard_names: list[str] = []
        self.all_quantized_layers: list[str] = []
        self.all_ignored_layers: list[str] = []
        self.output_weight_map: dict[str, str] = {}
        self.shard_parallelism: int = 1

    # -------------------------------------------------------------------
    # Validation / parsing
    # -------------------------------------------------------------------

    def _validate_format(self) -> None:
        format_lower = self.format.lower().replace(" ", "").split(",")[0]
        if format_lower not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Model-free mode only supports {self.SUPPORTED_FORMATS} format, "
                f"got '{self.format}'. Please use --format auto_round."
            )

    def _parse_scheme(self) -> None:
        scheme_in = self.scheme_input
        if isinstance(scheme_in, str) and scheme_in.upper() == "W4A16_MIXED":
            # Match regular-flow mixed recipe behavior in model-free mode:
            # default non-expert linear layers use 8-bit; expert overrides are
            # injected in _parse_layer_config.
            self.scheme_obj = _normalize_scheme("W8A16")
        else:
            self.scheme_obj = _normalize_scheme(scheme_in)
        _validate_supported_scheme(self.scheme_obj, self.scheme_input)
        ds = asdict(self.scheme_obj)
        self.default_scheme = {k: v for k, v in ds.items() if v is not None}

    def _parse_layer_config(self) -> None:
        lc = copy.deepcopy(self.layer_config_input) if self.layer_config_input else {}

        if isinstance(self.scheme_input, str) and self.scheme_input.upper() == "W4A16_MIXED":
            # Keep shared experts at 8-bit while routing experts to 4-bit.
            # User-provided layer_config entries (if any) still take priority.
            if "shared_expert" not in lc:
                lc[".shared_expert."] = {"bits": 8, "data_type": "int"}
            if "expert" not in lc:
                lc[".experts."] = {"bits": 4, "data_type": "int"}
                lc[".moe."] = {"bits": 4, "data_type": "int"}

        # Append '.' only for keys ending with ".<digits>" to avoid partial
        # numeric matches (e.g. layer.1 should not match layer.10).
        # Keep plain names like "fc2" untouched.
        for key in list(lc.keys()):
            if re.search(r"\.\d+$", key):
                lc[key + "."] = lc.pop(key)

        # Normalize values to dicts.
        for key, val in list(lc.items()):
            if isinstance(val, str):
                parsed = asdict(preset_name_to_scheme(val.upper()))
                lc[key] = {k: v for k, v in parsed.items() if v is not None}
            elif isinstance(val, QuantizationScheme):
                lc[key] = {k: v for k, v in asdict(val).items() if v is not None}
            elif isinstance(val, dict):
                # Resolve 'scheme' key inside dict values, e.g. {'scheme': 'W2A16'}
                if "scheme" in val:
                    scheme_val = val.pop("scheme")
                    if isinstance(scheme_val, str):
                        parsed = asdict(preset_name_to_scheme(scheme_val.upper()))
                        resolved = {k: v for k, v in parsed.items() if v is not None}
                    elif isinstance(scheme_val, QuantizationScheme):
                        resolved = {k: v for k, v in asdict(scheme_val).items() if v is not None}
                    else:
                        resolved = {}
                    # Explicit keys in val override the resolved scheme values
                    resolved.update(val)
                    lc[key] = resolved
            else:
                raise TypeError(f"Unsupported layer_config value type for '{key}': {type(val)}")

        self.layer_config = lc

    def _build_ignore_patterns(self) -> None:
        ignore_patterns: list[str] = []
        if self.ignore_layers_input:
            ignore_patterns = [p.strip() for p in self.ignore_layers_input.replace(" ", "").split(",") if p.strip()]
            ignore_patterns = [p + "." if re.search(r"\.\d+$", p) else p for p in ignore_patterns]

        if not self.quant_lm_head and "lm_head" not in ignore_patterns:
            ignore_patterns.append("lm_head")
            ignore_patterns.append("head")  # for deepseek v4

        if not self.quant_nontext_module:
            for kw in _NONTEXT_KEYWORDS:
                if kw not in ignore_patterns:
                    ignore_patterns.append(kw)

        self.ignore_patterns = ignore_patterns

    # -------------------------------------------------------------------
    # Source resolution and discovery
    # -------------------------------------------------------------------

    def _resolve_source(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_streaming = not _is_model_cached(self.model_name_or_path)
        if self.is_streaming:
            logger.info("Model not found locally or in cache — using streaming download mode.")
            self.work_dir = self.output_dir
            _download_metadata_files(self.model_name_or_path, self.work_dir)
            transformer_work_dir = os.path.join(self.work_dir, "transformer")
            if (
                not os.path.exists(os.path.join(self.work_dir, "config.json"))
                and os.path.isdir(transformer_work_dir)
                and os.path.exists(os.path.join(transformer_work_dir, "config.json"))
            ):
                self.is_diffusion_model = True
                self.diffusion_root_dir = self.work_dir
                self.work_dir = transformer_work_dir
                logger.info(
                    "Detected diffusion model (no root config.json, found transformer/ subfolder). "
                    "Only the transformer component will be quantized; other sub-components are skipped."
                )
            self.config = _load_config(self.work_dir)
        else:
            self.source_dir = _resolve_source_dir(self.model_name_or_path)
            transformer_source_dir = os.path.join(self.source_dir, "transformer")
            if (
                not os.path.exists(os.path.join(self.source_dir, "config.json"))
                and os.path.isdir(transformer_source_dir)
                and os.path.exists(os.path.join(transformer_source_dir, "config.json"))
            ):
                self.is_diffusion_model = True
                self.diffusion_root_dir = self.source_dir
                self.source_dir = transformer_source_dir
                logger.info(
                    "Detected diffusion model (no root config.json, found transformer/ subfolder). "
                    "Only the transformer component will be quantized; other sub-components are skipped."
                )
            self.config = _load_config(self.source_dir)

    def _check_conv1d_and_embedding(self) -> None:
        """Detect Conv1d and embedding layers and automatically add them to the ignore list."""
        local_dir = self.work_dir if self.is_streaming else self.source_dir
        if not local_dir or not os.path.isdir(local_dir):
            return

        try:
            from auto_round.utils.model import find_layers_from_config

            incompatible = find_layers_from_config(local_dir, class_names=["Embedding", "Conv1d", "Conv1D"])

            if incompatible:
                # Group by class for a cleaner warning message
                incompatible_layers = []
                for cls, layers in incompatible.items():
                    incompatible_layers.extend(layers)
                summary = ", ".join(f"{cls}({len(layers)})" for cls, layers in sorted(incompatible.items()))
                self.ignore_patterns.extend(incompatible_layers)
                logger.warning(
                    f"Detected {len(incompatible)} layer(s) incompatible with model-free RTN"
                    f": {compress_layer_names(incompatible_layers)}.\n"
                    f"These layers have been automatically added to ignore_layers "
                    f"and will be kept in full precision.\n"
                    f"To override, pass --ignore_layers explicitly or disable "
                    f"model-free mode (--disable_model_free)."
                )

        except Exception as exc:
            logger.warning(
                f"Could not check model architecture for incompatible layers: {exc}.\n"
                f"Models with Embedding or Conv1d layers may be incorrectly quantized "
                f"in model-free mode (non-2D weights cannot be packed by the RTN kernel).\n"
                f"If affected, either disable model-free mode (remove --model_free) or "
                f"add those layers to --ignore_layers."
            )

    def _apply_predefined_ignore_layers(self) -> None:
        predefined = get_predefined_ignore_layers_from_config(self.config)
        if predefined:
            logger.info(f"Using predefined ignore_layers from config: " f"{compress_layer_names(predefined)}")
            self.ignore_patterns.extend(predefined)

    def _detect_fp8_source(self) -> None:
        quant_config = self.config.get("quantization_config", {})
        is_fp8 = (
            quant_config.get("quant_method") == "fp8"
            or quant_config.get("quantization_type") == "fp8"
            or quant_config.get("fmt", "").startswith("e4m3")
        )
        if is_fp8:
            self.fp8_block_size = quant_config.get("weight_block_size")
            logger.info(
                f"Detected FP8 source model (block_size={self.fp8_block_size}, "
                f"scale_fmt={quant_config.get('scale_fmt', 'N/A')}). "
                f"FP8 weights will be dequantized before quantization."
            )

    def _resolve_model_type(self) -> None:
        """Resolve and log model_type for model-specific preprocessing hooks."""
        self.model_type = str(self.config.get("model_type", "")).lower()
        if self.model_type:
            logger.info(f"Detected source model_type='{self.model_type}'.")

    def _discover_shards(self) -> None:
        search_dir = self.work_dir if self.is_streaming else self.source_dir
        self.shard_names = _list_weight_shards(search_dir)

    def _resolve_shard_parallelism(self) -> tuple[int, str]:
        shard_count = len(self.shard_names)
        # Auto policy: shard_count // 4, capped at 10, minimum 1.
        default_parallelism = max(1, min(shard_count // 4, 10))
        env_name = "AR_MODEL_FREE_SHARD_PARALLELISM"
        if not envs.is_set(env_name):
            return min(default_parallelism, shard_count or 1), f"auto(default={default_parallelism})"

        try:
            configured = envs.AR_MODEL_FREE_SHARD_PARALLELISM
        except ValueError as e:
            logger.warning(f"{e}; using auto default {default_parallelism}.")
            raw_value = os.environ.get(env_name, "")
            return min(default_parallelism, shard_count or 1), f"invalid({raw_value!r})"

        if configured is None:
            return min(default_parallelism, shard_count or 1), f"auto(default={default_parallelism})"

        effective = min(configured, shard_count or 1)
        return effective, f"env={configured}"

    @property
    def _quant_output_dir(self) -> str:
        """Effective output directory for quantized weight shards and config.

        For diffusion models the quantized transformer component is written
        to ``<output_dir>/transformer/``; for all other models the top-level
        ``output_dir`` is used directly.
        """
        if self.is_diffusion_model:
            return os.path.join(self.output_dir, "transformer")
        return self.output_dir

    # -------------------------------------------------------------------
    # Shard processing pipeline
    # -------------------------------------------------------------------

    def _process_all_shards(self) -> None:
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        if not self.shard_names:
            return

        os.makedirs(self._quant_output_dir, exist_ok=True)

        worker_count = max(1, min(self.shard_parallelism, len(self.shard_names)))
        futures = []
        pool: ProcessPoolExecutor | None = None
        try:
            pool = ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn"))
            for shard_idx, shard_name in enumerate(self.shard_names):
                futures.append(
                    pool.submit(
                        _process_single_shard_task,
                        shard_idx,
                        shard_name,
                        model_name_or_path=self.model_name_or_path,
                        work_dir=self.work_dir,
                        source_dir=self.source_dir,
                        is_streaming=self.is_streaming,
                        device=self.device,
                        default_scheme=self.default_scheme,
                        layer_config=self.layer_config,
                        ignore_patterns=self.ignore_patterns,
                        fp8_block_size=self.fp8_block_size,
                        model_type=self.model_type,
                        quant_output_dir=self._quant_output_dir,
                        total_shards=len(self.shard_names),
                    )
                )

            shard_iter = (
                _tqdm(as_completed(futures), total=len(futures), desc="Processing shards", unit="shard")
                if _tqdm
                else as_completed(futures)
            )

            for future in shard_iter:
                shard_idx, shard_name, shard_path, out_shard_name, tensor_names, quantized, ignored = future.result()

                if (
                    shard_path is None
                    or out_shard_name is None
                    or tensor_names is None
                    or quantized is None
                    or ignored is None
                ):
                    logger.warning(f"Shard not found: {shard_name}, skipping")
                    continue

                memory_monitor.update()
                clear_memory()
                if len(self.shard_names) > 1:
                    logger.info(f"Memory usage: {memory_monitor.get_summary()}")

                compressed_quantized = compress_layer_names(quantized)
                compressed_ignored = compress_layer_names(ignored)
                logger.info(
                    f"Shard {shard_idx + 1}/{len(self.shard_names)} ({shard_name}):\n"
                    f"  Quantized layers ({len(quantized)}): {compressed_quantized}\n"
                    f"  Ignored layers ({len(ignored)}): {compressed_ignored}"
                )

                self.all_quantized_layers.extend(quantized)
                self.all_ignored_layers.extend(ignored)

                for tensor_name in tensor_names:
                    self.output_weight_map[tensor_name] = out_shard_name
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; terminating model-free shard worker processes.")
            _force_cleanup_process_pool(pool)
            raise
        except Exception:
            _force_cleanup_process_pool(pool)
            raise
        finally:
            _force_cleanup_process_pool(pool)

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------

    def _write_index(self) -> None:
        _write_index_file(self._quant_output_dir, self.output_weight_map)

    def _write_config_files(self) -> None:
        quantization_config = _build_quantization_config(
            default_scheme=self.default_scheme,
            layer_config=self.layer_config,
            ignore_patterns=self.ignore_patterns,
            quantized_layers=self.all_quantized_layers,
            ignored_layers=self.all_ignored_layers,
        )

        self.config["quantization_config"] = quantization_config
        os.makedirs(self._quant_output_dir, exist_ok=True)
        with open(os.path.join(self._quant_output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

        with open(os.path.join(self._quant_output_dir, "quantization_config.json"), "w") as f:
            json.dump(quantization_config, f, indent=2)

    def _copy_metadata_files(self) -> None:
        if self.is_streaming:
            # Metadata was downloaded directly to output_dir (or output_dir/transformer/
            # for diffusion models) — nothing to copy or clean up.
            return

        if self.is_diffusion_model:
            # For diffusion models, copy root-level metadata files and
            # sub-component directories (vae, scheduler, tokenizer, …) to
            # output_dir.  The quantized transformer component is already
            # written to output_dir/transformer/ by the pipeline, so
            # copytree's ``not os.path.exists(dst)`` guard prevents
            # overwriting it.
            for fname in os.listdir(self.diffusion_root_dir):
                src = os.path.join(self.diffusion_root_dir, fname)
                dst = os.path.join(self.output_dir, fname)
                if os.path.isdir(src):
                    if not os.path.exists(dst):
                        shutil.copytree(src, dst)
                elif os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            return

        for fname in os.listdir(self.source_dir):
            if _is_weight_shard(fname):
                continue
            src = os.path.join(self.source_dir, fname)
            dst = os.path.join(self.output_dir, fname)
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
            elif os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    def _cleanup_streaming_shard_cache(self) -> None:
        """Remove temporary streaming shard cache under output_dir/.cache."""
        if not self.is_streaming:
            return

        cache_dir = os.path.join(self.work_dir, ".cache", "model_free_source_shards")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

        # Best-effort prune for empty .cache directory created by this flow.
        try:
            os.rmdir(os.path.join(self.work_dir, ".cache"))
        except OSError:
            pass

    def _log_summary(self, total_time: float) -> None:
        compressed_quantized = compress_layer_names(self.all_quantized_layers)
        compressed_ignored = compress_layer_names(list(dict.fromkeys(self.all_ignored_layers)))
        logger.info(
            f"\nModel-free quantization complete.\n"
            f"  Output directory: {self.output_dir}\n"
            f"  Total time: {total_time:.2f} seconds\n"
            f"  Memory usage: {memory_monitor.get_summary()}\n"
            f"  Quantized layers ({len(self.all_quantized_layers)}): "
            f"{compressed_quantized}\n"
            f"  Ignored layers ({len(set(self.all_ignored_layers))}): "
            f"{compressed_ignored}\n"
        )

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------

    def run(self) -> str:
        """Execute the full model-free quantization pipeline.

        Returns:
            Absolute path to the output directory.
        """
        # ---- AutoScheme: resolve per-layer schemes before anything else ----
        if _looks_like_auto_scheme(self.scheme_input):
            resolver = getattr(self, "_resolve_auto_scheme", None)
            if not callable(resolver):
                raise ValueError(
                    "AutoScheme schemes are only supported through the "
                    "AutoRound(model_free=True) API, not the low-level "
                    "_ModelFreeCompressorCore driver."
                )
            resolver()  # pylint: disable=E1102

        # ---- preflight ----
        self._validate_format()
        self._parse_scheme()
        self._parse_layer_config()
        self._build_ignore_patterns()

        # ---- source resolution ----
        self._resolve_source()
        self._check_conv1d_and_embedding()
        self._apply_predefined_ignore_layers()
        self._detect_fp8_source()
        self._resolve_model_type()
        self._discover_shards()
        self.shard_parallelism, shard_parallelism_source = self._resolve_shard_parallelism()

        # Determine the output packing format based on scheme data type
        data_type = (self.default_scheme.get("data_type") or "int").lower()
        if is_mx_fp(data_type):
            bits = self.default_scheme.get("bits", 4)
            packing_format = "mxfp4-pack-quantized" if bits == 4 else "mxfp8-quantized"
        else:
            packing_format = "auto_round:auto_gptq"

        logger.info(
            f"Model-free quantization: {self.model_name_or_path}\n"
            f"  Scheme: {self.scheme_obj}\n"
            f"  Packing format: {packing_format}\n"
            f"  Output: {self.output_dir}\n"
            f"  Shards: {len(self.shard_names)}\n"
            f"  Shard parallelism: {self.shard_parallelism} ({shard_parallelism_source}, "
            f"env AR_MODEL_FREE_SHARD_PARALLELISM)\n"
            f"  Streaming download: {self.is_streaming}\n"
            f"  Diffusion model: {self.is_diffusion_model}\n"
            f"  Quant lm_head: {self.quant_lm_head}\n"
            f"  Quant nontext module: {self.quant_nontext_module}\n"
            f"  Device: {self.device}"
        )

        start_time = time.time()
        memory_monitor.reset()

        # ---- main loop ----
        self._process_all_shards()

        # ---- write outputs ----
        self._write_index()
        self._write_config_files()
        self._copy_metadata_files()
        self._cleanup_streaming_shard_cache()

        self._log_summary(time.time() - start_time)
        return self.output_dir


# ---------------------------------------------------------------------------
# AutoRound-compatible compressor: ModelFreeCompressor doubles as the
# compressor object returned by AutoRound.__new__ when model-free mode is
# selected.  It owns both the quantization pipeline (run()) AND the
# AutoRound-facing interface (quantize_and_save()).
# ---------------------------------------------------------------------------


class ModelFreeCompressor(_ModelFreeCompressorCore):
    """Model-free RTN quantizer that also acts as an AutoRound compressor.

    When constructed via ``AutoRound(model_free=True, ...)`` the instance is
    returned directly from ``AutoRound.__new__``.  The caller then invokes
    :meth:`quantize_and_save` exactly as they would on any other compressor.

    When used as a pure-quantization driver (CLI / functional API) call
    :meth:`run` instead.

    Args:
        model_name_or_path: HuggingFace model ID or local directory path.
            In the AutoRound compressor role this is the ``model`` argument.
        output_dir: Where to write the quantized model.  May be ``None``
            when used as a compressor (output_dir is passed to
            :meth:`quantize_and_save` later).
        scheme: Quantization scheme name or :class:`QuantizationScheme`.
        layer_config: Per-layer overrides.
        ignore_layers: Comma-separated layer name patterns to skip.
        format: Output format (only ``"auto_round"`` is supported).
        device: Compute device.
        quant_lm_head: Whether to quantize ``lm_head``.
        quant_nontext_module: Whether to quantize non-text modules.
        **kwargs: When called from ``AutoRound.__new__`` the full AutoRound
            kwargs are forwarded here.  Unknown kwargs are silently ignored
            so that calibration-only parameters (``nsamples``, ``iters``,
            ``dataset``, …) do not cause errors.
    """

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: Optional[str] = None,
        scheme: Union[str, QuantizationScheme] = "W4A16",
        layer_config: Optional[dict] = None,
        ignore_layers: str = "",
        format: str = "auto_round",
        device: str = "cpu",
        quant_lm_head: bool = False,
        quant_nontext_module: bool = False,
        # --- AutoRound compressor-role aliases ---
        tokenizer: Any = None,
        device_map: Any = None,
        **kwargs,
    ) -> None:
        import copy
        from dataclasses import fields as dc_fields

        fallback_kwargs = dict(kwargs)

        # Collect per-field scheme overrides forwarded from AutoRound
        # (e.g. bits=4, sym=False passed as individual kwargs).
        self.user_scheme_overrides: dict = {}
        for field in dc_fields(QuantizationScheme):
            if field.name in kwargs:
                val = kwargs.pop(field.name)
                if val is not None:
                    self.user_scheme_overrides[field.name] = val

        # Resolve device: AutoRound passes device_map; the core API uses device.
        if device_map is not None:
            from auto_round.utils import get_major_device

            device = get_major_device(device_map)

        # Initialise the core quantizer
        super().__init__(
            model_name_or_path=model_name_or_path,
            output_dir=output_dir or "tmp_autoround",
            scheme=scheme,
            layer_config=layer_config,
            ignore_layers=ignore_layers,
            format=format,
            device=device,
            quant_lm_head=quant_lm_head,
            quant_nontext_module=quant_nontext_module,
        )

        # Compressor-role state (mirrors BaseCompressor attributes used by
        # AutoRound's post-processing code)
        self._output_dir_override: Optional[str] = None  # set by quantize_and_save
        self.model = None
        self.tokenizer = tokenizer
        self.model_free = True
        self.model_free_path = model_name_or_path
        self.iters = 0
        self.disable_opt_rtn = True
        self.formats = None
        self.quantized = False
        self._fallback_compressor = None
        # Start from the remaining user kwargs and explicitly set/override
        # known compressor init parameters for clarity.
        fallback_init = dict(fallback_kwargs)
        # Route-control kwargs are only meaningful for the initial entry
        # selection. Strip them so fallback always re-enters the regular flow
        # with a single explicit disable_model_free=True override.
        fallback_init.pop("model_free", None)
        fallback_init.pop("disable_model_free", None)
        fallback_init.update(
            model=model_name_or_path,
            iters=0,
            disable_opt_rtn=True,
            tokenizer=tokenizer,
            scheme=copy.deepcopy(scheme),
            layer_config=copy.deepcopy(layer_config),
            ignore_layers=ignore_layers,
            device_map=device_map,
            quant_lm_head=quant_lm_head,
        )

        self._fallback_init_kwargs = fallback_init
        if quant_nontext_module:
            self._fallback_init_kwargs["quant_nontext_module"] = quant_nontext_module
        # remaining kwargs intentionally consumed/ignored

        # AutoScheme (two-phase delta-loss selection) state.
        self._auto_scheme_resolved = False
        self._auto_scheme_family: Optional[str] = None

    def _fallback_to_base_compressor(self):
        from auto_round.autoround import AutoRound

        logger.info(
            "Format '%s' is not supported by model-free mode; falling back to the regular AutoRound flow.",
            format,
        )
        logger.info(
            "fallbacked_init_kwargs: %s",
            self._fallback_init_kwargs,
        )
        compressor = AutoRound(**self._fallback_init_kwargs, disable_model_free=True)
        self._fallback_compressor = compressor

    def _fallback_to_quantize_and_save(
        self,
        output_dir: str,
        format: str,
        inplace: bool,
        **kwargs,
    ):
        self._fallback_to_base_compressor()
        return self._fallback_compressor.quantize_and_save(  # pylint: disable=E1101
            output_dir=output_dir, format=format, inplace=inplace, **kwargs
        )

    def quantize(
        self,
    ) -> Any:
        """fallback to base compressor's quantize."""
        self._fallback_to_base_compressor()
        return self._fallback_compressor.quantize()  # pylint: disable=E1101

    def __getattribute__(self, name: str):
        """Prefer attributes from the fallback compressor when available.

        Once model-free flow falls back to the regular AutoRound compressor,
        external attribute reads on this wrapper should observe the fallback
        compressor's state first.
        """
        local_only_names = {
            "_fallback_compressor",
            "_fallback_init_kwargs",
            "_fallback_to_base_compressor",
            "__dict__",
            "__class__",
            "__getattribute__",
            "__setattr__",
            "__delattr__",
        }

        if name in local_only_names or name.startswith("__"):
            return super().__getattribute__(name)

        fallback = super().__getattribute__("__dict__").get("_fallback_compressor")
        if fallback is not None:
            if name == "compressor":
                return fallback
            try:
                return getattr(fallback, name)
            except AttributeError:
                pass

        return super().__getattribute__(name)

    # ------------------------------------------------------------------
    # AutoScheme (two-phase: delta-loss selection + model-free packing)
    # ------------------------------------------------------------------

    def _run_auto_scheme_selection(self, auto_scheme: Any) -> dict[str, dict]:
        """Run AutoScheme delta-loss selection to obtain a per-layer config.

        The model is loaded temporarily (via the regular AutoRound flow) so
        that delta-loss scoring can run its forward/backward passes, then it is
        released before the model-free shard-by-shard packing begins.
        """
        from auto_round.autoround import AutoRound

        init_kwargs = dict(self._fallback_init_kwargs)
        init_kwargs["scheme"] = auto_scheme

        compressor = AutoRound(**init_kwargs, disable_model_free=True)
        try:
            # post_init() (outside inference_mode) runs the delta-loss scheme
            # selection and populates ``compressor.layer_config``.
            post_init = getattr(compressor, "post_init", None)
            if not callable(post_init):
                raise RuntimeError("AutoScheme fallback compressor has no callable post_init().")
            post_init()  # pylint: disable=E1102
            layer_config = copy.deepcopy(getattr(compressor, "layer_config", {}) or {})
        finally:
            # Release the model that was loaded only for scoring so the
            # packing phase keeps model-free's low memory footprint.
            try:
                model_context = getattr(compressor, "model_context", None)
                if model_context is not None and hasattr(model_context, "model"):
                    model_context.model = None
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            del compressor
            clear_memory()

        if not layer_config:
            raise RuntimeError("AutoScheme did not produce a layer_config for model-free mode.")
        return layer_config

    def _resolve_auto_scheme(self) -> None:
        """Resolve an ``AutoScheme`` scheme into concrete model-free inputs.

        Idempotent.  Validates the options, runs delta-loss selection, then
        rewrites ``scheme_input`` / ``layer_config_input`` / ``ignore_layers_input``
        so the standard model-free pipeline can proceed unchanged.
        """
        if self._auto_scheme_resolved:
            return

        auto_scheme = self.scheme_input
        family = _validate_auto_scheme_options(auto_scheme)
        logger.info(
            "Model-free + AutoScheme: generating a per-layer scheme via delta-loss. "
            "The model is loaded temporarily for scoring, then released before "
            "shard-by-shard packing."
        )

        generated = self._run_auto_scheme_selection(auto_scheme)
        base_scheme, per_layer, fp16_layers = _convert_auto_scheme_layer_config(generated)

        # Merge the generated per-layer overrides; any user-provided
        # layer_config entries take priority.
        merged_lc: dict = dict(per_layer)
        if self.layer_config_input:
            merged_lc.update(copy.deepcopy(self.layer_config_input))
        self.layer_config_input = merged_lc

        # Keep AutoScheme's 16-bit layers in full precision.
        if fp16_layers:
            extra = ",".join(fp16_layers)
            self.ignore_layers_input = f"{self.ignore_layers_input},{extra}" if self.ignore_layers_input else extra

        self.scheme_input = base_scheme
        self._auto_scheme_family = family
        self._auto_scheme_resolved = True

        logger.info(
            "Model-free + AutoScheme resolved: base scheme %s, %d per-layer override(s), "
            "%d layer(s) kept at 16-bit.",
            base_scheme,
            len(per_layer),
            len(fp16_layers),
        )

    # ------------------------------------------------------------------
    # AutoRound compressor interface
    # ------------------------------------------------------------------

    def quantize_and_save(
        self,
        output_dir: str = "tmp_autoround",
        format: str = "auto_round",
        inplace: bool = True,
        **kwargs,
    ) -> Any:
        """Quantize and save — AutoRound compressor entry point."""
        # AutoScheme: run delta-loss selection first so the effective scheme /
        # data-type family (which drives the accepted export formats) is known.
        if _looks_like_auto_scheme(self.scheme_input):
            self._resolve_auto_scheme()

        # Accept the standard auto_round formats.
        _accepted_formats = {
            "auto_round",
            "auto_round:auto_gptq",
        }
        # MXFP only supports the llm_compressor format (INT string preset,
        # or an AutoScheme run whose options resolved to the MXFP family).
        if self.scheme_input in ["MXFP4", "MXFP8"] or self._auto_scheme_family == "mx_fp":
            _accepted_formats = ["llm_compressor"]
        if format not in _accepted_formats:
            logger.warning(
                f"Format '{format}' is not supported by model-free mode for scheme '{self.scheme_input}'; "
                f"falling back to the regular AutoRound flow."
            )
            return self._fallback_to_quantize_and_save(output_dir=output_dir, format=format, inplace=inplace, **kwargs)

        # Apply user scheme overrides before running
        if self.user_scheme_overrides:
            self.scheme_input = _apply_scheme_overrides(self.scheme_input, self.user_scheme_overrides)

        # Temporarily point output_dir at what the caller requested
        orig = self.output_dir
        self.output_dir = output_dir
        out_path = self.run()
        self.output_dir = orig
        self.quantized = True
        return None, out_path


# ---------------------------------------------------------------------------
# Model-Type Specific Preprocessing Hooks (Extension Point)
# ---------------------------------------------------------------------------
#
# Keep model-specific source-format adaptation functions at the end of this
# file so the core quantization pipeline remains easy to read and maintain.
# Add new model handlers here, keyed by `model_type`, and keep dequant/passthrough
# decisions in the generic MXFP handlers above.


def _expand_e8m0_block_scale(
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Expand a coarse 2D E8M0 block scale to the llm-compressor per-group layout.

    deepseek_v4 stores the per-block shared exponent in a *coarse* 2D shape
    ``[out_features // block_h, in_features // block_w]`` (e.g. ``[12, 56]`` for
    a ``[1536, 7168]`` weight, i.e. 128x128 blocks).  llm-compressor expects a
    per-group scale of shape ``[out_features, in_features // group_size]``
    (e.g. ``[1536, 224]`` for ``group_size=32``).

    Because every fine MXFP group lies entirely inside a single coarse block,
    the expansion is a pure ``repeat_interleave`` along both axes (no
    interpolation).  The returned tensor is ``uint8`` (raw E8M0 bytes), matching
    the ``U8`` dtype used by llm-compressor ``weight_scale`` tensors.
    """
    scale = scale.view(torch.uint8)
    if scale.dim() != 2:
        raise ValueError(f"Expected a 2D E8M0 block scale, got shape {tuple(scale.shape)}.")

    target_rows = out_features
    target_cols = in_features // group_size
    rows, cols = scale.shape

    if target_rows % rows != 0 or target_cols % cols != 0:
        raise ValueError(
            f"Cannot expand E8M0 block scale {tuple(scale.shape)} to "
            f"({target_rows}, {target_cols}); shapes are not divisible."
        )

    if target_rows != rows:
        scale = scale.repeat_interleave(target_rows // rows, dim=0)
    if target_cols != cols:
        scale = scale.repeat_interleave(target_cols // cols, dim=1)
    return scale.contiguous()


def _preprocess_model_type_source_tensors(
    raw_tensors: dict[str, torch.Tensor],
    model_type: str | None,
    group_size: int = 32,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Apply model-type-specific source tensor normalization.

    This step is intentionally limited to *format conversion* and does not do
    passthrough / dequant decisions. It marks converted layers in the returned
    ``source_state`` so downstream generic MXFP handling can treat them exactly
    like normal llm-compressor MXFP sources.

    Returns:
        ``(raw_tensors, source_state)`` where ``source_state[layer]`` is the
        source MXFP bits (4 or 8) for model-type preprocessed layers.
    """
    if (model_type or "").lower() != "deepseek_v4":
        return raw_tensors, {}

    entries: list[tuple[str, str, bool]] = []  # (weight_name, scale_name, is_fp8)
    for name, tensor in raw_tensors.items():
        if not name.endswith(".weight"):
            continue
        scale_name = name[: -len(".weight")] + ".scale"
        if scale_name not in raw_tensors:
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            entries.append((name, scale_name, True))
        elif tensor.dtype in (torch.int8, torch.uint8):
            entries.append((name, scale_name, False))

    if not entries:
        return raw_tensors, {}

    source_state: dict[str, int] = {}
    n_fp8 = 0
    n_fp4 = 0
    for weight_name, scale_name, is_fp8 in entries:
        layer_name = weight_name[: -len(".weight")]
        weight = raw_tensors.pop(weight_name)
        scale = raw_tensors.pop(scale_name)

        if is_fp8:
            out_features, in_features = weight.shape
            weight_key = f"{layer_name}.weight"
            source_state[layer_name] = 8
            n_fp8 += 1
        else:
            out_features = weight.shape[0]
            in_features = weight.shape[1] * 2
            weight = weight.view(torch.uint8).contiguous()
            weight_key = f"{layer_name}.weight_packed"
            source_state[layer_name] = 4
            n_fp4 += 1

        weight_scale = _expand_e8m0_block_scale(scale, out_features, in_features, group_size=group_size)
        raw_tensors[weight_key] = weight
        raw_tensors[f"{layer_name}.weight_scale"] = weight_scale

    logger.info(
        "Applied model_type preprocessing for deepseek_v4: "
        f"{n_fp8} MXFP8 layer(s), {n_fp4} MXFP4 layer(s) converted to llm-compressor naming."
    )
    return raw_tensors, source_state
