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

**NVFP4 E5M3** (``auto_round`` or ``fake`` format):

* Preset name: ``NVFP4_E5M3``.
* ``data_type="nvfp4_v2"``, ``group_size=16``, with high-precision QDQ weights.

Schemes that require special packing (FP8, standard NVFP4, GGUF, INT8_W8A8,
BF16, FPW8A16, ...) are **not** supported in model-free mode and will raise
``ValueError``.  Use the standard AutoRound flow for those.

Output formats
--------------
* **INT schemes** → ``auto_round:auto_gptq`` packing format, ``quant_method="auto-round"``.
* **MXFP schemes** → ``mxfp4-pack-quantized`` or ``mxfp8-quantized`` format,
  ``quant_method="compressed-tensors"``, compatible with vLLM / llm-compressor.
* **NVFP4_E5M3** → AutoRound format with packed ``.weight_packed`` and
    ``.weight_scale`` tensors; use ``format="fake"`` explicitly for high-precision
    QDQ ``.weight`` tensors.

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
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, fields
from typing import Any, Optional, Union

import torch

from auto_round import envs
from auto_round.compressors.config_resolution import thaw_mapping
from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme
from auto_round.utils.common import AUDIO_MM_KEYS, VISION_MM_KEYS, compress_layer_names
from auto_round.utils.device import clear_memory, memory_monitor
from auto_round.utils.device_manager import default_enable_torch_compile
from auto_round.utils.model_free_utils import (
    _LM_HEAD_PATTERNS,
    _apply_scheme_overrides,
    _build_cross_shard_pairs_from_weight_map,
    _build_quantization_config,
    _convert_auto_scheme_layer_config,
    _download_metadata_files,
    _download_single_shard,
    _get_model_cache_status,
    _is_full_precision_default,
    _is_weight_shard,
    _layer_config_has_mxfp,
    _layer_config_has_nvfp4,
    _list_weight_shards,
    _load_config,
    _load_weight_map_from_index,
    _looks_like_auto_scheme,
    _normalize_scheme,
    _PatternMatcher,
    _process_shard,
    _resolve_source_dir,
    _validate_auto_scheme_options,
    _validate_supported_scheme,
    _write_index_file,
    _write_output_shard,
    handle_model_type_low_precision_source_tensors,
    is_model_free_supported_scheme,
    preprocess_model_type_source_tensors,
)

# Backward-compat aliases for internal/private helper names used in tests and
# downstream imports.
_preprocess_model_type_source_tensors = preprocess_model_type_source_tensors
_handle_model_type_low_precision_source_tensors = handle_model_type_low_precision_source_tensors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Preset schemes that model-free mode can produce.
# INT presets use ``auto_round:auto_gptq`` packing; MXFP presets use
# ``mxfp4-pack-quantized`` or ``mxfp8-quantized`` (compressed-tensors) packing.
# BF16 acts as a full-precision default — all layers stay in BF16 unless
# overridden by layer_config.
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
    "NVFP4_E5M3",
    "BF16",
)

# Allowed ``bits`` values for integer WOQ.
# 3-bit is excluded — see note above.
_SUPPORTED_INT_BITS: tuple[int, ...] = (2, 4, 8)

# Allowed ``bits`` values for MXFP weight quantization.
_SUPPORTED_MXFP_BITS: tuple[int, ...] = (4, 8)

_NVFP4_E5M3_DATA_TYPE = "nvfp4_v2"

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


# Per-tensor / per-shard helpers
# ---------------------------------------------------------------------------


# Output writers
# ---------------------------------------------------------------------------


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
    source_quantization_config: dict | None = None,
    quant_output_dir: str,
    total_shards: int,
    enable_torch_compile: bool = False,
    disable_opt_rtn: bool = False,
    donor_tensors_to_exclude: list[str] | None = None,
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

    return _quantize_local_shard_task(
        shard_idx,
        shard_name,
        shard_path=shard_path,
        device=device,
        default_scheme=default_scheme,
        layer_config=layer_config,
        ignore_patterns=ignore_patterns,
        fp8_block_size=fp8_block_size,
        model_type=model_type,
        source_quantization_config=source_quantization_config,
        quant_output_dir=quant_output_dir,
        total_shards=total_shards,
        enable_torch_compile=enable_torch_compile,
        disable_opt_rtn=disable_opt_rtn,
        cleanup_source_shard=is_streaming,
        donor_tensors_to_exclude=donor_tensors_to_exclude,
    )


def _quantize_local_shard_task(
    shard_idx: int,
    shard_name: str,
    *,
    shard_path: str,
    device: str,
    default_scheme: dict,
    layer_config: dict,
    ignore_patterns: list[str],
    fp8_block_size: list | None,
    model_type: str | None,
    source_quantization_config: dict | None,
    quant_output_dir: str,
    total_shards: int,
    enable_torch_compile: bool = False,
    disable_opt_rtn: bool = False,
    cleanup_source_shard: bool = False,
    donor_tensors_to_exclude: list[str] | None = None,
    index_dir: str | None = None,
    donor_shard_dir: str | None = None,
) -> tuple[int, str, str | None, str | None, list[str] | None, list[str] | None, list[str] | None]:
    """Quantize one already-downloaded shard and write the output shard.

    Returns lightweight metadata only so IPC does not transfer tensor storages.
    """
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
        source_quantization_config=source_quantization_config,
        enable_torch_compile=enable_torch_compile,
        disable_opt_rtn=disable_opt_rtn,
        index_dir=index_dir,
        donor_shard_dir=donor_shard_dir,
        donor_tensors_to_exclude=set(donor_tensors_to_exclude) if donor_tensors_to_exclude else None,
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

    if cleanup_source_shard:
        try:
            os.remove(shard_path)
        except OSError:
            pass

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
        "fake",
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
        enable_torch_compile: Optional[bool] = None,
        disable_opt_rtn: bool = False,
    ) -> None:
        # --- raw inputs ---
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.scheme_input = scheme
        self.layer_config_input = layer_config
        self.ignore_layers_input = ignore_layers
        self.format = format or "auto_round"
        self.device = device
        self.quant_lm_head = quant_lm_head
        self.quant_nontext_module = quant_nontext_module
        self.enable_torch_compile = (
            default_enable_torch_compile(device, platform_name=sys.platform)
            if enable_torch_compile is None
            else enable_torch_compile
        )
        self.disable_opt_rtn = disable_opt_rtn

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
        # Cross-shard FP8 scale_inv dependency maps built from index.json.
        # cross_shard_deps: {recipient_shard: {donor_shard: [scale_inv_names]}}
        # donor_shard_tensors: {donor_shard: set(scale_inv names it donates)}
        self.cross_shard_deps: dict[str, dict[str, list[str]]] = {}
        self.donor_shard_tensors: dict[str, set[str]] = {}
        # Reference-counting state so donor shard cache files can be deleted as
        # soon as they are no longer needed, instead of waiting until the end
        # of the whole run. A donor file is safe to delete once (a) its own
        # quantization task has read it, and (b) every recipient shard that
        # depends on it has finished (and thus finished hydrating from it).
        self._donor_remaining_recipients: dict[str, int] = {}
        self._donor_self_consumed: dict[str, bool] = {}
        self._donor_shard_paths: dict[str, str] = {}

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
        scheme_overrides = getattr(self, "user_scheme_overrides", None)
        if scheme_overrides:
            scheme_in = _apply_scheme_overrides(scheme_in, scheme_overrides)
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
        self.default_scheme["_output_format"] = self.format

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

        if not self.quant_lm_head:
            layer_config_keys = set(self.layer_config or {})
            # Skip each known lm_head name variant unless the user has explicitly listed it in layer_config.
            # Each pattern is checked independently so a user-specified key takes priority over the default.
            for lm_head_pattern in _LM_HEAD_PATTERNS:
                pattern_in_config = any(
                    key == lm_head_pattern or key.startswith(lm_head_pattern + ".") for key in layer_config_keys
                )
                if not pattern_in_config and lm_head_pattern not in ignore_patterns:
                    ignore_patterns.append(lm_head_pattern)

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
        cached, cache_reason = _get_model_cache_status(self.model_name_or_path)
        self.is_streaming = not cached
        logger.info(
            "Model-free source decision: %s (%s)",
            "streaming" if self.is_streaming else "local/snapshot",
            cache_reason,
        )
        if self.is_streaming:
            logger.info(
                "Path selected: streaming mode. Will download metadata first "
                "(snapshot_download with weight ignore patterns), then fetch "
                "weight shards on demand (hf_hub_download)."
            )
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
            if os.path.isdir(self.model_name_or_path):
                logger.info("Path selected: local directory mode. Reading shards from local path.")
            else:
                logger.info("Path selected: snapshot/local-cache mode. Resolving source dir via snapshot_download.")
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

    def _build_cross_shard_deps(self) -> None:
        """Build cross-shard FP8 scale_inv dependency map from index.json.

        Identifies triples (recipient_shard, donor_shard, scale_inv_tensor) by
        comparing ``.weight`` and ``.weight_scale_inv`` entries in weight_map.
        Stored in ``self.cross_shard_deps`` and ``self.donor_shard_tensors`` for
        use by the shard scheduler and quantization workers.
        """
        search_dir = self.work_dir if self.is_streaming else self.source_dir
        # Find the first *.safetensors.index.json in search_dir.
        index_path: str | None = None
        st_std = os.path.join(search_dir, "model.safetensors.index.json")
        if os.path.exists(st_std):
            index_path = st_std
        else:
            candidates = sorted(
                os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.endswith(".safetensors.index.json")
            )
            if candidates:
                index_path = candidates[0]

        if not index_path:
            self.cross_shard_deps = {}
            self.donor_shard_tensors = {}
            return

        try:
            weight_map = _load_weight_map_from_index(index_path)
        except Exception as e:
            logger.warning(f"Could not load weight_map for cross-shard dep analysis: {e}")
            self.cross_shard_deps = {}
            self.donor_shard_tensors = {}
            return

        self.cross_shard_deps, self.donor_shard_tensors = _build_cross_shard_pairs_from_weight_map(weight_map)
        if self.cross_shard_deps:
            n_pairs = sum(len(t) for donors in self.cross_shard_deps.values() for t in donors.values())
            logger.info(
                f"Cross-shard FP8 dependencies: {n_pairs} scale_inv tensor(s), "
                f"{len(self.cross_shard_deps)} recipient shard(s), "
                f"{len(self.donor_shard_tensors)} donor shard(s)."
            )

        # Build reverse mapping (donor -> distinct recipient shards) so donor
        # cache files can be reference-counted and deleted as soon as every
        # dependent recipient has finished, instead of only at the very end.
        donor_recipients: dict[str, set[str]] = {}
        for recipient, donors in self.cross_shard_deps.items():
            for donor in donors:
                donor_recipients.setdefault(donor, set()).add(recipient)
        self._donor_remaining_recipients = {donor: len(recipients) for donor, recipients in donor_recipients.items()}
        self._donor_self_consumed = {donor: False for donor in self.donor_shard_tensors}

    def _reorder_shards_by_dependency(self) -> None:
        """Reorder shard_names via topological sort so donors precede recipients.

        A single-pass "donors first, recipients last" partition is not enough:
        a shard can simultaneously be a *recipient* (it needs a scale_inv from
        another shard) and a *donor* (it donates a scale_inv to a third shard).
        Such transitive chains (X donates to Y, Y donates to Z) must be
        resolved with a proper topological sort, otherwise Y could still be
        scheduled after Z, i.e. after one of its own dependents.

        Ties are broken by original shard order (stable) so scheduling stays
        as close as possible to the natural shard sequence. This ensures:
        - Non-streaming: workers start on shards that need no cross-shard data.
        - Streaming: donor shards are downloaded before recipient shards, so the
          donor cache files are available when recipient workers run hydration.
        """
        if not self.cross_shard_deps:
            return

        original_index = {name: i for i, name in enumerate(self.shard_names)}
        # recipient -> set(donors it depends on)
        depends_on: dict[str, set[str]] = {
            recipient: set(donors) for recipient, donors in self.cross_shard_deps.items()
        }
        # in-degree = number of unresolved donor dependencies for each shard.
        in_degree: dict[str, int] = {name: len(depends_on.get(name, ())) for name in self.shard_names}
        # donor -> shards that depend on it (only counting donors that are
        # themselves known shards; unknown donors can't be waited on anyway).
        dependents: dict[str, list[str]] = {}
        for recipient, donors in depends_on.items():
            for donor in donors:
                if donor in original_index:
                    dependents.setdefault(donor, []).append(recipient)
                elif recipient in in_degree:
                    # Donor shard is unknown/missing from discovery; it can
                    # never be satisfied here, so don't block scheduling on it.
                    in_degree[recipient] -= 1

        import heapq

        ready = [original_index[name] for name, deg in in_degree.items() if deg == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        visited: set[str] = set()
        index_to_name = {i: name for name, i in original_index.items()}

        while ready:
            idx = heapq.heappop(ready)
            name = index_to_name[idx]
            if name in visited:
                continue
            visited.add(name)
            ordered.append(name)
            for dependent in dependents.get(name, ()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heapq.heappush(ready, original_index[dependent])

        if len(ordered) != len(self.shard_names):
            # Cycle detected (shouldn't happen with well-formed indices) —
            # fall back to appending any unresolved shards in original order.
            remaining = [name for name in self.shard_names if name not in visited]
            logger.warning(
                f"Shard dependency graph has a cycle or unresolved node(s) "
                f"({len(remaining)} shard(s)); appending them in original order."
            )
            ordered.extend(remaining)

        n_recipients = len(self.cross_shard_deps)
        if n_recipients:
            logger.info(
                f"Shard scheduling: topologically sorted {len(self.shard_names)} shard(s) "
                f"({n_recipients} recipient shard(s) with cross-shard dependencies)."
            )
        self.shard_names = ordered

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
        if self.is_streaming:
            self._process_all_shards_streaming_pipeline()
            return

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
                donor_tensors = list(self.donor_shard_tensors.get(shard_name, ())) or None
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
                        source_quantization_config=self.config.get("quantization_config", {}),
                        enable_torch_compile=self.enable_torch_compile,
                        disable_opt_rtn=self.disable_opt_rtn,
                        quant_output_dir=self._quant_output_dir,
                        total_shards=len(self.shard_names),
                        donor_tensors_to_exclude=donor_tensors,
                    )
                )

            shard_iter = (
                _tqdm(as_completed(futures), total=len(futures), desc="Processing shards", unit="shard")
                if _tqdm
                else as_completed(futures)
            )

            for future in shard_iter:
                result = future.result()
                self._merge_shard_task_result(result)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; terminating model-free shard worker processes.")
            _force_cleanup_process_pool(pool)
            raise
        except Exception:
            _force_cleanup_process_pool(pool)
            raise
        finally:
            _force_cleanup_process_pool(pool)

    def _merge_shard_task_result(
        self,
        result: tuple[int, str, str | None, str | None, list[str] | None, list[str] | None, list[str] | None],
    ) -> None:
        """Merge one shard-task result into global stats and weight map."""
        shard_idx, shard_name, shard_path, out_shard_name, tensor_names, quantized, ignored = result
        if shard_path is None or out_shard_name is None or tensor_names is None or quantized is None or ignored is None:
            logger.warning(f"Shard not found: {shard_name}, skipping")
            return

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

        if self.is_streaming:
            self._release_donor_dependency(shard_name)

    def _release_donor_dependency(self, completed_shard_name: str) -> None:
        """Update donor reference counts after *completed_shard_name* finishes.

        Donor shard cache files are kept alive only as long as needed: once a
        donor's own quantization task has read it AND every recipient shard
        that depends on it has finished (and thus hydrated from it), the
        cached file is deleted immediately rather than waiting for the whole
        run to finish via :meth:`_cleanup_streaming_shard_cache`.
        """
        if completed_shard_name in self._donor_self_consumed:
            self._donor_self_consumed[completed_shard_name] = True
            self._maybe_cleanup_donor_shard(completed_shard_name)

        for donor in self.cross_shard_deps.get(completed_shard_name, {}):
            if donor in self._donor_remaining_recipients:
                self._donor_remaining_recipients[donor] = max(0, self._donor_remaining_recipients[donor] - 1)
                self._maybe_cleanup_donor_shard(donor)

    def _maybe_cleanup_donor_shard(self, donor_name: str) -> None:
        """Delete *donor_name*'s cached shard file once it is no longer needed."""
        if not self._donor_self_consumed.get(donor_name, False):
            return
        if self._donor_remaining_recipients.get(donor_name, 0) > 0:
            return
        shard_path = self._donor_shard_paths.pop(donor_name, None)
        # Remove tracking regardless of whether the file still exists so we
        # never attempt to clean this donor up twice.
        self._donor_self_consumed.pop(donor_name, None)
        self._donor_remaining_recipients.pop(donor_name, None)
        if shard_path and os.path.exists(shard_path):
            try:
                os.remove(shard_path)
                logger.debug(f"Removed donor shard cache file (no longer needed): {donor_name}")
            except OSError as e:
                logger.warning(f"Failed to remove donor shard cache file {donor_name}: {e}")

    def _process_all_shards_streaming_pipeline(self) -> None:
        """Streaming-mode shard pipeline with dedicated downloader and quant workers.

        Design:
        - one downloader worker serializes network bandwidth usage;
        - N quant workers consume ready shards independently;
        - shards are assigned for quantization as soon as download completes.
        """
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        if not self.shard_names:
            return

        os.makedirs(self._quant_output_dir, exist_ok=True)

        worker_count = max(1, min(self.shard_parallelism, len(self.shard_names)))
        prefetch_depth = max(2, worker_count)
        total_shards = len(self.shard_names)

        download_pool: ThreadPoolExecutor | None = None
        quant_pool: ProcessPoolExecutor | None = None
        download_futures: dict = {}
        quant_futures = set()
        next_download_idx = 0
        completed_quant = 0

        def _submit_next_download() -> bool:
            nonlocal next_download_idx
            if next_download_idx >= total_shards:
                return False
            shard_idx = next_download_idx
            shard_name = self.shard_names[shard_idx]
            future = download_pool.submit(
                _prefetch_shard,
                self.model_name_or_path,
                shard_name,
                self.work_dir,
                self.source_dir,
                self.is_streaming,
            )
            download_futures[future] = (shard_idx, shard_name)
            next_download_idx += 1
            return True

        try:
            download_pool = ThreadPoolExecutor(max_workers=1)
            quant_pool = ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn"))

            for _ in range(min(prefetch_depth, total_shards)):
                _submit_next_download()

            progress = _tqdm(total=total_shards, desc="Processing shards", unit="shard") if _tqdm else None

            while completed_quant < total_shards:
                wait_set = set(download_futures.keys()) | set(quant_futures)
                if not wait_set:
                    break

                done, _ = wait(wait_set, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in download_futures:
                        shard_idx, shard_name = download_futures.pop(future)
                        shard_path = future.result()
                        if shard_path is None or not os.path.exists(shard_path):
                            logger.warning(f"Prefetch failed for shard {shard_name}, skipping")
                            completed_quant += 1
                            if progress is not None:
                                progress.update(1)
                        else:
                            # Donor shards must stay on disk until every recipient
                            # that depends on them has been processed (tracked via
                            # _donor_remaining_recipients / _release_donor_dependency),
                            # so that recipient shards can still read scale_inv
                            # tensors from them via
                            # _hydrate_missing_fp8_scales_from_index. They are
                            # deleted as soon as that's no longer needed; any
                            # stragglers are swept up at the end by
                            # _cleanup_streaming_shard_cache.
                            shard_cache_dir = os.path.join(self.work_dir, ".cache", "model_free_source_shards")
                            is_donor = shard_name in self.donor_shard_tensors
                            if is_donor:
                                self._donor_shard_paths[shard_name] = shard_path
                            donor_tensors = list(self.donor_shard_tensors.get(shard_name, ())) or None
                            qf = quant_pool.submit(
                                _quantize_local_shard_task,
                                shard_idx,
                                shard_name,
                                shard_path=shard_path,
                                device=self.device,
                                default_scheme=self.default_scheme,
                                layer_config=self.layer_config,
                                ignore_patterns=self.ignore_patterns,
                                fp8_block_size=self.fp8_block_size,
                                model_type=self.model_type,
                                source_quantization_config=self.config.get("quantization_config", {}),
                                quant_output_dir=self._quant_output_dir,
                                total_shards=total_shards,
                                enable_torch_compile=self.enable_torch_compile,
                                # Keep donor shards alive for recipient hydration.
                                cleanup_source_shard=not is_donor,
                                donor_tensors_to_exclude=donor_tensors,
                                # Pass the work_dir as index_dir so workers can
                                # find the index.json that was downloaded during
                                # metadata fetch (not in the shard cache sub-dir).
                                index_dir=self.work_dir,
                                donor_shard_dir=shard_cache_dir,
                            )
                            quant_futures.add(qf)

                        while len(download_futures) < prefetch_depth and _submit_next_download():
                            pass
                    elif future in quant_futures:
                        quant_futures.remove(future)
                        result = future.result()
                        self._merge_shard_task_result(result)
                        completed_quant += 1
                        if progress is not None:
                            progress.update(1)

            if progress is not None:
                progress.close()
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; terminating model-free shard workers.")
            _force_cleanup_process_pool(quant_pool)
            raise
        except Exception:
            _force_cleanup_process_pool(quant_pool)
            raise
        finally:
            _force_cleanup_process_pool(quant_pool)
            if download_pool is not None:
                try:
                    download_pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------

    def _write_index(self) -> None:
        _write_index_file(self._quant_output_dir, self.output_weight_map)

    def _write_config_files(self) -> None:
        block_prefixes = []
        for layer_name in self.all_quantized_layers:
            parts = layer_name.split(".")
            for index, part in enumerate(parts):
                if part.isdigit() and index > 0:
                    block_prefixes.append(".".join(parts[:index]))
                    break
        block_name_to_quantize = ",".join(dict.fromkeys(block_prefixes)) or None

        os.makedirs(self._quant_output_dir, exist_ok=True)
        quantization_config = _build_quantization_config(
            default_scheme=self.default_scheme,
            layer_config=self.layer_config,
            ignore_patterns=self.ignore_patterns,
            quantized_layers=self.all_quantized_layers,
            ignored_layers=self.all_ignored_layers,
            block_name_to_quantize=block_name_to_quantize,
            format=self.format,
        )

        self.config["quantization_config"] = quantization_config
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
        self._build_cross_shard_deps()
        self._reorder_shards_by_dependency()
        self.shard_parallelism, shard_parallelism_source = self._resolve_shard_parallelism()

        # Determine the output packing format based on scheme data type
        data_type = (self.default_scheme.get("data_type") or "int").lower()
        if is_mx_fp(data_type):
            bits = self.default_scheme.get("bits", 4)
            packing_format = "mxfp4-pack-quantized" if bits == 4 else "mxfp8-quantized"
        elif data_type == _NVFP4_E5M3_DATA_TYPE:
            packing_format = "fake" if self.format == "fake" else "auto_round:llm_compressor_nvfp4_e5m3"
        else:
            packing_format = "auto_round:auto_gptq"
        if is_mx_fp(data_type) or _layer_config_has_mxfp(self.layer_config):
            if not self.disable_opt_rtn:
                logger.info(
                    "MXFP optimized RTN is enabled: evaluating the baseline E8M0 scale, "
                    "2x scale, and 0.5x scale independently for each group. "
                    "Pass --disable_opt_rtn to use plain RTN."
                )
        else:
            logger.info(
                "Integer WOQ model-free quantization uses plain RTN "
                "(opt_rtn is disabled for INT WOQ to preserve accuracy)."
            )

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
            f"  Torch compile: {self.enable_torch_compile}\n"
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
        low_cpu_mem_usage: bool = True,
        enable_torch_compile: Optional[bool] = None,
        disable_opt_rtn: bool = False,
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
            enable_torch_compile=enable_torch_compile,
            disable_opt_rtn=disable_opt_rtn,
        )

        # Compressor-role state (mirrors BaseCompressor attributes used by
        # AutoRound's post-processing code)
        self._output_dir_override: Optional[str] = None  # set by quantize_and_save
        self.model = None
        self.tokenizer = tokenizer
        self.model_free = True
        self.model_free_path = model_name_or_path
        self.iters = 0
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
            disable_opt_rtn=disable_opt_rtn,
            tokenizer=tokenizer,
            scheme=copy.deepcopy(scheme),
            layer_config=copy.deepcopy(layer_config),
            ignore_layers=ignore_layers,
            device_map=device_map,
            quant_lm_head=quant_lm_head,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        # Scheme fields are consumed above from ``kwargs``. Preserve them when
        # a later format check falls back to the regular AutoRound flow.
        fallback_init.update(self.user_scheme_overrides)

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
            self.format,
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
            layer_config = thaw_mapping(getattr(compressor, "layer_config", {}) or {})
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
        preferred_base_scheme = None
        for option in auto_scheme.options:
            option_scheme = _normalize_scheme(option)
            act_bits = option_scheme.act_bits if option_scheme.act_bits is not None else 16
            if (option_scheme.bits or 0) < 16 or act_bits < 16:
                preferred_base_scheme = option_scheme
                break
        base_scheme, per_layer, fp16_layers = _convert_auto_scheme_layer_config(
            generated,
            preferred_base_scheme=preferred_base_scheme,
        )

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

    def _precheck_auto_scheme_fallback(self, format: str) -> bool:
        """Early fallback check for model-free + AutoScheme.

        This avoids spending time on delta-loss AutoScheme selection when the
        requested export format is known to be incompatible with the resolved
        AutoScheme option family.
        """
        if not _looks_like_auto_scheme(self.scheme_input):
            return False

        family = _validate_auto_scheme_options(self.scheme_input)
        accepted_formats = {"auto_round", "auto_round:auto_gptq"}
        if family == "mx_fp":
            accepted_formats = {"llm_compressor", "auto_round", "auto_round:auto_gptq"}

        if format not in accepted_formats:
            logger.warning(
                "Format '%s' is incompatible with model-free + AutoScheme (family=%s); "
                "fallback before running AutoScheme scoring.",
                format,
                family,
            )
            return True
        return False

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
        # Early fallback gate for model-free + AutoScheme: avoid running
        # costly delta-loss selection when format is known incompatible.
        if self._precheck_auto_scheme_fallback(format):
            return self._fallback_to_quantize_and_save(output_dir=output_dir, format=format, inplace=inplace, **kwargs)

        # AutoScheme: run delta-loss selection first so the effective scheme /
        # data-type family (which drives the accepted export formats) is known.
        if _looks_like_auto_scheme(self.scheme_input):
            self._resolve_auto_scheme()

        # Accept the standard auto_round formats.
        _accepted_formats = {
            "auto_round",
            "auto_round:auto_gptq",
        }
        # MXFP supports both llm_compressor (compressed-tensors) and auto_round formats.
        # The only difference is the quantization_config metadata; on-disk weights are identical.
        normalized_scheme = (
            _normalize_scheme(self.scheme_input) if not _looks_like_auto_scheme(self.scheme_input) else None
        )
        if (
            normalized_scheme is not None and is_mx_fp((normalized_scheme.data_type or "").lower())
        ) or self._auto_scheme_family == "mx_fp":
            _accepted_formats = {"llm_compressor", "auto_round", "auto_round:auto_gptq"}
        elif normalized_scheme is not None and (normalized_scheme.data_type or "").lower() == _NVFP4_E5M3_DATA_TYPE:
            _accepted_formats = {"fake", "llm_compressor", "auto_round", "auto_round:auto_gptq"}
        elif _is_full_precision_default(self.scheme_input) and _layer_config_has_mxfp(self.layer_config_input):
            # BF16 default with MXFP layer_config overrides.
            _accepted_formats = {"llm_compressor", "auto_round", "auto_round:auto_gptq"}
        elif _is_full_precision_default(self.scheme_input) and _layer_config_has_nvfp4(self.layer_config_input):
            # BF16 default with NVFP4_E5M3 layer_config overrides.
            _accepted_formats = {"fake", "llm_compressor", "auto_round", "auto_round:auto_gptq"}
        if format not in _accepted_formats:
            logger.warning(
                f"Format '{format}' is not supported by model-free mode for scheme '{self.scheme_input}'; "
                f"falling back to the regular AutoRound flow."
            )
            return self._fallback_to_quantize_and_save(output_dir=output_dir, format=format, inplace=inplace, **kwargs)

        # Apply user scheme overrides before running
        if self.user_scheme_overrides:
            self.scheme_input = _apply_scheme_overrides(self.scheme_input, self.user_scheme_overrides)

        # Temporarily point output_dir and format at what the caller requested
        orig_dir = self.output_dir
        orig_fmt = self.format
        self.output_dir = output_dir
        self.format = format
        try:
            out_path = self.run()
        finally:
            self.output_dir = orig_dir
            self.format = orig_fmt
        self.quantized = True
        return None, out_path
