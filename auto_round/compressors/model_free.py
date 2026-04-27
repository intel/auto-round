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
Model-free mode currently supports **integer weight-only** quantization
schemes packed in the ``auto_round:auto_gptq`` format only.  Specifically:

* Preset names: ``W2A16``, ``W2A16G32``, ``W2A16G64``, ``W3A16``, ``W4A16``,
  ``W8A16``.
* Custom :class:`~auto_round.schemes.QuantizationScheme` instances with
  ``data_type="int"``, ``bits in {2, 3, 4, 8}``, ``act_bits >= 16``, and any
  symmetric / asymmetric configuration.

Schemes that require special packing (FP8, MXFP4, NVFP4, GGUF, INT8_W8A8,
BF16, FPW8A16, ...) are **not** supported in model-free mode and will raise
``ValueError``.  Use the standard AutoRound flow for those.

Usage (CLI)
-----------
::

    auto_round facebook/opt-125m \\
        --model_free \\
        --scheme W4A16 \\
        --output_dir int4-125m

Usage (API)
-----------
::

    from auto_round import AutoRound

    AutoRound(
        model="facebook/opt-125m",
        scheme="W4A16",
        model_free=True,
    ).quantize_and_save("./int4-125m")
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from typing import Optional, Union

import torch

from auto_round import envs
from auto_round.logger import logger
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme, preset_name_to_scheme
from auto_round.utils.common import compress_layer_names, to_standard_regex
from auto_round.utils.device import clear_memory, memory_monitor
from auto_round.utils.missing_tensors import quantize_weight_rtn, split_fused_expert_tensors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# add "embed", "conv" in case of auto detection failure in _check_conv1d_and_embedding
_BLOCK_NAME_TO_IGNORE = ["shared_expert_gate.", "mlp.gate.", "embed", "conv"]

# Integer WOQ preset schemes that model-free mode can produce.
# Other presets (FP8/MX/NV/GGUF/BF16/INT8_W8A8/FPW8A16) require different
# packing kernels not implemented by ``quantize_weight_rtn``.
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
)

# Allowed ``bits`` values for integer WOQ.
# 3-bit is excluded — see note above.
_SUPPORTED_INT_BITS: tuple[int, ...] = (2, 4, 8)

# Multimodal keywords kept in full precision by default.
_NONTEXT_KEYWORDS: tuple[str, ...] = (
    "vision",
    "visual",
    "image",
    "img",
    "audio",
    "speech",
    "wav",
    "waveform",
)


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
        try:
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
                break
        except Exception:
            continue

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

    # --- safetensors: any single .safetensors file ---
    st_files = sorted(f for f in os.listdir(source_dir) if f.endswith(".safetensors"))
    if len(st_files) == 1:
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
    if len(bin_files) == 1:
        return bin_files

    # --- safetensors: single file ---
    return ["model.safetensors"]


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
    ):
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

    if bits >= 16:
        return layer_name, {tensor_name: tensor}, None, layer_name

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


def _dequant_fp8_tensors(
    raw_tensors: dict[str, torch.Tensor],
    block_size: list | None = None,
) -> dict[str, torch.Tensor]:
    """Dequantize FP8 / FP4-packed weight tensors and remove their scale tensors.

    Handles three storage conventions:

    1. **DeepSeek-V3 FP8** — weight dtype ``float8_e4m3fn`` + ``weight_scale_inv``
       (per-block float32 scales, NOT E8M0).
    2. **FP8 + UE8M0 scale** — weight dtype ``float8_e4m3fn`` + ``.scale``
       (F8_E8M0 / ``ue8m0`` format, e.g. ``scale_fmt=ue8m0``).
    3. **FP4-packed I8 + UE8M0 scale** — weight dtype ``torch.int8`` where each
       byte stores two FP4 E2M1 nibbles, paired with a ``.scale`` tensor in
       F8_E8M0 format.  Shape relationship: ``weight[rows, cols/2]`` and
       ``scale[rows, (cols/2)*2/block_size]`` where ``block_size`` is inferred
       from the ratio ``weight.shape[1] * 2 / scale.shape[1]``.

    All cases are converted to ``bfloat16`` so downstream RTN quantization can
    proceed normally.
    """
    from auto_round.utils.weight_handler import _dequant_fp8_linear_weight

    E8M0_EXPONENT_BIAS = 127

    def _e8m0_to_float(scale_tensor: torch.Tensor) -> torch.Tensor:
        """Convert E8M0 (power-of-2 exponent) scale tensor to float."""
        raw = scale_tensor.view(torch.uint8).to(torch.int16) - E8M0_EXPONENT_BIAS
        return torch.pow(2.0, raw.to(torch.float32)).to(torch.bfloat16)

    def _dequant_fp4_packed_weight(
        weight_i8: torch.Tensor,
        scale_e8m0: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize FP4-packed int8 weight with E8M0 block scales to bfloat16.

        Each ``int8`` byte stores two FP4 E2M1 values: the low nibble is the
        first element and the high nibble is the second.  Block size is inferred
        from the shape ratio ``weight.shape[1] * 2 / scale.shape[1]``.

        Args:
            weight_i8:   Packed weight tensor, shape ``[rows, cols_packed]``,
                         dtype ``torch.int8``.  ``cols_packed = cols / 2``.
            scale_e8m0:  Per-block scale tensor, shape ``[rows, n_blocks]``,
                         stored as raw uint8 bytes in F8_E8M0 (ue8m0) format.

        Returns:
            Dequantized weight, shape ``[rows, cols]``, dtype ``bfloat16``.
        """
        # FP4 E2M1 lookup table — index is the 4-bit pattern (0..15)
        # Layout: positive range at indices 0-7, negative at 8-15.
        FP4_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.bfloat16,
            device=weight_i8.device,
        )
        rows = weight_i8.shape[0]

        # Re-interpret int8 storage as uint8 for bit operations.
        weight_u8 = weight_i8.view(torch.uint8)
        lo = (weight_u8 & 0x0F).to(torch.long)  # low  nibble → first  element
        hi = ((weight_u8 >> 4) & 0x0F).to(torch.long)  # high nibble → second element
        # Interleave lo/hi along the last dim: [rows, cols_packed, 2] → [rows, cols]
        unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, -1)
        cols = unpacked.shape[1]

        # Decode 4-bit patterns to bfloat16 float values via the lookup table.
        decoded = FP4_LUT[unpacked]  # [rows, cols]

        # Convert E8M0 per-block scales to float32 → bfloat16.
        scale_f = _e8m0_to_float(scale_e8m0)  # [rows, n_blocks]
        n_blocks = scale_f.shape[1]
        block_size_inner = cols // n_blocks  # inferred block size (e.g. 32)

        # Multiply each block of decoded values by its corresponding scale.
        decoded = decoded.reshape(rows, n_blocks, block_size_inner)
        return (decoded * scale_f.unsqueeze(-1)).reshape(rows, cols)

    # ------------------------------------------------------------------
    # Collect quantized weight entries.
    # Tuple layout: (weight_name, scale_name, is_e8m0, is_fp4_packed)
    #   is_e8m0      – scale tensor is in F8_E8M0 format (needs _e8m0_to_float)
    #   is_fp4_packed – weight is int8-packed FP4; use _dequant_fp4_packed_weight
    # ------------------------------------------------------------------
    quant_entries: list[tuple[str, str, bool, bool]] = []

    for name, tensor in raw_tensors.items():
        if not name.endswith(".weight"):
            continue

        is_fp4_packed = tensor.dtype == torch.int8
        is_fp8 = tensor.dtype == torch.float8_e4m3fn
        # Also catch other 1-byte dtypes (e.g. float8_e5m2) by element size.
        if not is_fp4_packed and not is_fp8:
            if tensor.element_size() != 1:
                continue
            is_fp8 = True  # treat remaining 1-byte dtypes as FP8

        # Convention 1: .weight_scale_inv (DeepSeek-V3 style, FP8 only)
        scale_inv_name = name.replace(".weight", ".weight_scale_inv")
        if scale_inv_name in raw_tensors and not is_fp4_packed:
            quant_entries.append((name, scale_inv_name, False, False))
            continue

        # Convention 2: .scale in F8_E8M0 / ue8m0 format
        scale_name = name.replace(".weight", ".scale")
        if scale_name in raw_tensors:
            quant_entries.append((name, scale_name, True, is_fp4_packed))

    if not quant_entries:
        return raw_tensors

    fp4_count = sum(1 for e in quant_entries if e[3])
    fp8_count = len(quant_entries) - fp4_count
    parts: list[str] = []
    if fp8_count:
        parts.append(f"{fp8_count} FP8")
    if fp4_count:
        parts.append(f"{fp4_count} FP4-packed (int8)")
    logger.info(f"Dequantizing {' and '.join(parts)} weight tensor(s) to bfloat16.")

    for weight_name, scale_name, is_e8m0, is_fp4_packed in quant_entries:
        weight = raw_tensors[weight_name]
        scale = raw_tensors.pop(scale_name)
        if is_fp4_packed:
            # FP4 E2M1 packed in int8 with UE8M0 per-block scale.
            raw_tensors[weight_name] = _dequant_fp4_packed_weight(weight, scale)
        else:
            if is_e8m0:
                scale = _e8m0_to_float(scale)
            raw_tensors[weight_name] = _dequant_fp8_linear_weight(weight, scale, block_size=block_size)

    return raw_tensors


def _process_shard(
    shard_path: str,
    default_scheme: dict = None,
    layer_config: dict = None,
    ignore_patterns: list[str] = None,
    device: str = "cpu",
    *,
    matcher: "_PatternMatcher | None" = None,
    fp8_block_size: list | None = None,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Quantize eligible weights in a single safetensors shard.

    Returns:
        (output_tensors, quantized_layer_names, ignored_layer_names)
    """
    if matcher is None:
        matcher = _PatternMatcher(
            ignore_patterns if ignore_patterns is not None else [],
            layer_config if layer_config is not None else {},
            default_scheme if default_scheme is not None else {},
        )

    output_tensors: dict[str, torch.Tensor] = {}
    quantized_layers: list[str] = []
    ignored_layers: list[str] = []

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
    raw_tensors = _dequant_fp8_tensors(raw_tensors, block_size=fp8_block_size)

    for tensor_name in list(raw_tensors.keys()):
        tensor = raw_tensors.pop(tensor_name)
        _layer_name, out_dict, q_layer, ig_layer = _quantize_single_tensor(
            tensor_name,
            tensor,
            matcher,
            device,
        )
        output_tensors.update(out_dict)
        if q_layer:
            quantized_layers.append(q_layer)
        if ig_layer:
            ignored_layers.append(ig_layer)

    return output_tensors, quantized_layers, ignored_layers


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


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

    unique_ignored = list(dict.fromkeys(ignored_layers))
    for layer_name in unique_ignored:
        if layer_name not in extra_config:
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
            return _download_single_shard(model_name_or_path, shard_name, work_dir)
        path = os.path.join(source_dir, shard_name)
        return path if os.path.exists(path) else None
    except Exception as e:  # pragma: no cover
        logger.warning(f"Prefetch failed for {shard_name}: {e}")
        return None


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

    Model-free only supports integer weight-only quantization (sym/asym),
    packed in the ``auto_round:auto_gptq`` format.
    """
    act_bits = scheme_obj.act_bits if scheme_obj.act_bits is not None else 16
    if act_bits < 16:
        raise ValueError(
            f"Model-free mode only supports weight-only quantization (WOQ) schemes "
            f"where act_bits >= 16, but '{scheme_input}' has act_bits={act_bits}. "
            f"Supported preset schemes: {list(SUPPORTED_PRESET_SCHEMES)}."
        )

    data_type = (scheme_obj.data_type or "int").lower()
    if data_type != "int":
        raise ValueError(
            f"Model-free mode only supports integer weight quantization "
            f"(data_type='int'), but '{scheme_input}' has data_type='{data_type}'. "
            f"FP8 / MXFP / NVFP / GGUF / BF16 schemes require the standard "
            f"AutoRound flow.  Supported preset schemes: "
            f"{list(SUPPORTED_PRESET_SCHEMES)}."
        )

    bits = scheme_obj.bits
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
        _validate_supported_scheme(scheme_obj, scheme_obj)
        return True
    except (ValueError, TypeError):
        return False


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
        scheme: Quantization scheme name (e.g. ``"W4A16"``) or a
            :class:`QuantizationScheme` instance.
        layer_config: Per-layer quantization overrides.  Keys are layer
            names or regex patterns; values are dicts with ``bits``,
            ``group_size``, ``sym`` etc.
        ignore_layers: Comma-separated list of layer name patterns to keep
            in full precision.
        format: Output format (only ``"auto_round"`` is supported).
        device: Device for quantization computation (``"cpu"`` or
            ``"cuda"``).
        quant_lm_head: If True, quantize ``lm_head`` as well.  By default
            ``lm_head`` and any layer containing ``embed`` are kept in
            full precision.
        quant_nontext_module: If True, quantize non-text modules
            (vision/audio/image) as well.  By default these multimodal
            modules are kept in full precision.
    """

    SUPPORTED_FORMATS: tuple[str, ...] = ("auto_round",)

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
    ):
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
        self.matcher: _PatternMatcher | None = None
        self.config: dict = {}
        self.fp8_block_size: list | None = None
        self.is_streaming: bool = False
        self.is_diffusion_model: bool = False
        self.diffusion_root_dir: str = ""
        self.work_dir: str = ""
        self.source_dir: str = ""
        self.shard_names: list[str] = []
        self.all_quantized_layers: list[str] = []
        self.all_ignored_layers: list[str] = []
        self.output_weight_map: dict[str, str] = {}

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
        self.scheme_obj = _normalize_scheme(self.scheme_input)
        _validate_supported_scheme(self.scheme_obj, self.scheme_input)
        ds = asdict(self.scheme_obj)
        self.default_scheme = {k: v for k, v in ds.items() if v is not None}

    def _parse_layer_config(self) -> None:
        lc = copy.deepcopy(self.layer_config_input) if self.layer_config_input else {}

        # Append '.' to keys ending with a digit to avoid partial numeric matches.
        for key in list(lc.keys()):
            if key and key[-1].isdigit():
                lc[key + "."] = lc.pop(key)

        # Normalize values to dicts.
        for key, val in list(lc.items()):
            if isinstance(val, str):
                parsed = asdict(preset_name_to_scheme(val.upper()))
                lc[key] = {k: v for k, v in parsed.items() if v is not None}
            elif isinstance(val, QuantizationScheme):
                lc[key] = {k: v for k, v in asdict(val).items() if v is not None}
            elif isinstance(val, dict):
                pass
            else:
                raise TypeError(f"Unsupported layer_config value type for '{key}': {type(val)}")

        self.layer_config = lc

    def _build_ignore_patterns(self) -> None:
        ignore_patterns: list[str] = []
        if self.ignore_layers_input:
            ignore_patterns = [p.strip() for p in self.ignore_layers_input.replace(" ", "").split(",") if p.strip()]
            ignore_patterns = [p + "." if p and p[-1].isdigit() else p for p in ignore_patterns]

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
                    f"model-free mode (remove --model_free)."
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

    def _discover_shards(self) -> None:
        search_dir = self.work_dir if self.is_streaming else self.source_dir
        self.shard_names = _list_weight_shards(search_dir)

    def _build_matcher(self) -> None:
        self.matcher = _PatternMatcher(
            self.ignore_patterns,
            self.layer_config,
            self.default_scheme,
        )

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

        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        write_pool = ThreadPoolExecutor(max_workers=1)
        prefetch_future = None
        write_future = None

        if self.shard_names:
            prefetch_future = prefetch_pool.submit(
                _prefetch_shard,
                self.model_name_or_path,
                self.shard_names[0],
                self.work_dir,
                self.source_dir,
                self.is_streaming,
            )

        shard_iter = (
            _tqdm(
                enumerate(self.shard_names),
                total=len(self.shard_names),
                desc="Processing shards",
                unit="shard",
            )
            if _tqdm
            else enumerate(self.shard_names)
        )

        for shard_idx, shard_name in shard_iter:
            shard_path = prefetch_future.result() if prefetch_future else None

            # Kick off prefetch of the next shard.
            if shard_idx + 1 < len(self.shard_names):
                prefetch_future = prefetch_pool.submit(
                    _prefetch_shard,
                    self.model_name_or_path,
                    self.shard_names[shard_idx + 1],
                    self.work_dir,
                    self.source_dir,
                    self.is_streaming,
                )
            else:
                prefetch_future = None

            if shard_path is None or not os.path.exists(shard_path):
                logger.warning(f"Shard not found: {shard_name}, skipping")
                continue

            output_tensors, quantized, ignored = _process_shard(
                shard_path=shard_path,
                device=self.device,
                matcher=self.matcher,
                fp8_block_size=self.fp8_block_size,
            )
            memory_monitor.update()
            clear_memory()
            if len(self.shard_names) > 1:
                logger.info(f"Memory usage: {memory_monitor.get_summary()}")

            self.all_quantized_layers.extend(quantized)
            self.all_ignored_layers.extend(ignored)

            if write_future is not None:
                write_future.result()

            os.makedirs(self._quant_output_dir, exist_ok=True)
            out_shard_name = f"model-{shard_idx + 1:05d}-of-{len(self.shard_names):05d}.safetensors"
            write_future = write_pool.submit(
                _write_output_shard,
                self._quant_output_dir,
                out_shard_name,
                output_tensors,
                self.output_weight_map,
            )

            if self.is_streaming:
                try:
                    os.remove(shard_path)
                    logger.debug(f"Deleted source shard: {shard_path}")
                except OSError as e:
                    logger.warning(f"Could not delete source shard {shard_path}: {e}")

        prefetch_pool.shutdown(wait=False)
        if write_future is not None:
            write_future.result()
        write_pool.shutdown(wait=True)

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
            # For diffusion models, copy only root-level non-weight metadata files
            # (model_index.json, tokenizer files, etc.) to output_dir.
            # Sub-components other than transformer (vae, scheduler, …) are skipped
            # per the "only quantize transformer" policy; the quantized transformer
            # component is already written to output_dir/transformer/ by the pipeline.
            for fname in os.listdir(self.diffusion_root_dir):
                src = os.path.join(self.diffusion_root_dir, fname)
                dst = os.path.join(self.output_dir, fname)
                if os.path.isdir(src):
                    continue
                if os.path.isfile(src) and not os.path.exists(dst):
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
        # ---- preflight ----
        self._validate_format()
        self._parse_scheme()
        self._parse_layer_config()
        self._build_ignore_patterns()

        # ---- source resolution ----
        self._resolve_source()
        self._check_conv1d_and_embedding()
        self._apply_predefined_ignore_layers()
        self._build_matcher()
        self._detect_fp8_source()
        self._discover_shards()

        logger.info(
            f"Model-free quantization: {self.model_name_or_path}\n"
            f"  Scheme: {self.scheme_obj}\n"
            f"  Output: {self.output_dir}\n"
            f"  Shards: {len(self.shard_names)}\n"
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
        tokenizer=None,
        device_map=None,
        **kwargs,
    ):
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
            from auto_round.utils import detect_device

            device = detect_device(device_map)

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
        self._fallback_init_kwargs = {
            **fallback_kwargs,
            "model": model_name_or_path,
            "tokenizer": tokenizer,
            "scheme": copy.deepcopy(scheme),
            "layer_config": copy.deepcopy(layer_config),
            "ignore_layers": ignore_layers,
            "device_map": device_map,
            "device": device,
            "quant_lm_head": quant_lm_head,
            "quant_nontext_module": quant_nontext_module,
        }
        # remaining kwargs intentionally consumed/ignored

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
        return self._fallback_compressor.quantize_and_save(
            output_dir=output_dir, format=format, inplace=inplace, **kwargs
        )  # pylint: disable=E1101

    def quantize(
        self,
    ):
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
    # AutoRound compressor interface
    # ------------------------------------------------------------------

    def quantize_and_save(
        self,
        output_dir: str = "tmp_autoround",
        format: str = "auto_round",
        inplace: bool = True,
        **kwargs,
    ):
        """Quantize and save — AutoRound compressor entry point."""
        if format not in ["auto_round", "auto_round:auto_gptq"]:
            return self._fallback_to_base_compressor(output_dir=output_dir, format=format, inplace=inplace, **kwargs)

        # Apply user scheme overrides before running
        if self.user_scheme_overrides:
            self.scheme_input = _apply_scheme_overrides(self.scheme_input, self.user_scheme_overrides)

        # Temporarily point output_dir at what the caller requested
        orig = self.output_dir
        self.output_dir = output_dir
        out_path = self.run()
        self.output_dir = orig
        self.quantized = True
        return None, [out_path]
