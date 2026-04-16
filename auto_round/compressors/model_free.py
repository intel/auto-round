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
* **Streaming mode** (``--streaming``) – downloads, quantizes,
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
    )
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
from typing import Optional, Union

import torch

from auto_round import envs
from auto_round.logger import logger
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme, preset_name_to_scheme
from auto_round.utils.common import compress_layer_names
from auto_round.utils.device import memory_monitor
from auto_round.utils.missing_tensors import quantize_weight_rtn, split_fused_expert_tensors

_BLOCK_NAME_TO_IGNORE = ["shared_expert_gate.", "mlp.gate.", "embed"]
_GPU_CONCURRENT_WORKER = 64


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

    # MoE fallback: no module iteration available, use the common gate pattern
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


def _is_eligible_weight(tensor_name: str, tensor: torch.Tensor) -> bool:
    """Check if a tensor is eligible for quantization (2D Linear weight)."""
    return tensor_name.endswith(".weight") and tensor.dim() == 2


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

        # ---- Combined ignore regex (user-specified patterns) ----
        self._ignore_re: re.Pattern | None = self._build_ignore_regex(ignore_patterns)

        # ---- Combined skip regex (_BLOCK_NAME_TO_IGNORE) ----
        skip_parts = [re.escape(b) for b in _BLOCK_NAME_TO_IGNORE]
        self._skip_re: re.Pattern | None = re.compile("|".join(skip_parts)) if skip_parts else None

        # ---- Precompile layer_config patterns ----
        # Each entry: (compiled_regex | None, plain_string | None, cfg_dict)
        self._compiled_lc: list[tuple[re.Pattern | None, str | None, dict]] = []
        for pattern, cfg in layer_config.items():
            try:
                self._compiled_lc.append((re.compile(pattern), None, cfg))
            except re.error:
                self._compiled_lc.append((None, pattern, cfg))

        # ---- Result caches (thread-safe under CPython GIL) ----
        self._ignore_cache: dict[str, bool] = {}
        self._scheme_cache: dict[str, dict | None] = {}

    # ---- helpers ----

    @staticmethod
    def _build_ignore_regex(patterns: list[str]) -> re.Pattern | None:
        """Merge ignore patterns into one compiled regex."""
        if not patterns:
            return None
        parts: list[str] = []
        for p in patterns:
            if p.endswith("."):
                escaped = re.escape(p.rstrip("."))
                # "stripped." as substring OR "stripped" at end-of-string
                parts.append(f"{escaped}(?:\\.|$)")
            else:
                parts.append(re.escape(p))
        return re.compile("|".join(parts))

    # ---- public API ----

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

        # Exact match
        if layer_name in self._layer_config:
            cfg = self._layer_config[layer_name]
            if cfg.get("bits", default.get("bits", 4)) >= 16:
                return None
            return {**default, **cfg}

        # Precompiled pattern match
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


def _is_safetensors_shard(fname: str) -> bool:
    """Return True if *fname* is a safetensors weight shard (not the index)."""
    return fname.endswith(".safetensors") and not fname.endswith(".index.json")


def _download_metadata_files(
    model_name_or_path: str,
    local_dir: str,
) -> str:
    """Download all non-safetensors files from a model repo. Returns local dir."""
    os.makedirs(local_dir, exist_ok=True)

    if os.path.isdir(model_name_or_path):
        for fname in os.listdir(model_name_or_path):
            if _is_safetensors_shard(fname):
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
        ignore_patterns=["*.safetensors"],
    )
    return local_dir


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

    # ---- eligibility / skip checks ----
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

    # ---- RTN quantization ----
    try:
        orig_in_features = tensor.shape[1]
        use_cuda = device != "cpu" and torch.cuda.is_available()
        if use_cuda:
            # Dedicated CUDA stream per worker: allows H2D / compute / D2H
            # overlap across concurrent workers in the ThreadPoolExecutor.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                tensor_gpu = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
                qweight, qzeros, scales = quantize_weight_rtn(
                    weight=tensor_gpu,
                    bits=bits,
                    group_size=group_size,
                    sym=sym,
                    device=None,  # already on target device
                )
            stream.synchronize()
            del tensor, tensor_gpu
        else:
            qweight, qzeros, scales = quantize_weight_rtn(
                weight=tensor,
                bits=bits,
                group_size=group_size,
                sym=sym,
                device=None,
            )

        out: dict[str, torch.Tensor] = {
            f"{layer_name}.qweight": qweight,
            f"{layer_name}.qzeros": qzeros,
            f"{layer_name}.scales": scales,
        }

        in_features = orig_in_features
        if in_features % group_size != 0:
            in_features += group_size - in_features % group_size
        out[f"{layer_name}.g_idx"] = torch.arange(in_features, dtype=torch.int32) // group_size

        logger.debug(f"Quantized: {layer_name} (bits={bits}, group_size={group_size}, sym={sym})")
        return layer_name, out, layer_name, None

    except Exception as e:
        logger.warning(f"Failed to quantize {layer_name}: {e}. Keeping original weight.")
        return layer_name, {tensor_name: tensor}, None, layer_name


def _process_shard(
    shard_path: str,
    default_scheme: dict = None,
    layer_config: dict = None,
    ignore_patterns: list[str] = None,
    device: str = "cpu",
    *,
    matcher: "_PatternMatcher | None" = None,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Quantize eligible weights in a single safetensors shard.

    When *matcher* is provided it is used directly (preferred for
    cross-shard cache reuse).  Otherwise a new :class:`_PatternMatcher`
    is built from *default_scheme*, *layer_config*, and *ignore_patterns*.

    Args:
        shard_path: Path to the safetensors file.
        default_scheme: Default quantization parameters.
        layer_config: Per-layer overrides.
        ignore_patterns: Layer name patterns to skip.
        device: Computation device (``"cpu"`` or ``"cuda"``).
        matcher: Precompiled :class:`_PatternMatcher` instance.

    Returns:
        (output_tensors, quantized_layer_names, ignored_layer_names)
    """
    if matcher is None:
        matcher = _PatternMatcher(
            ignore_patterns if ignore_patterns is not None else [],
            layer_config if layer_config is not None else {},
            default_scheme if default_scheme is not None else {},
        )

    from safetensors import safe_open

    output_tensors: dict[str, torch.Tensor] = {}
    quantized_layers: list[str] = []
    ignored_layers: list[str] = []

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        raw_tensors = {name: f.get_tensor(name) for name in f.keys()}

    # Split fused expert tensors (3-D) into per-expert 2-D tensors
    raw_tensors = split_fused_expert_tensors(raw_tensors)

    use_gpu_concurrent = device != "cpu" and torch.cuda.is_available()

    if use_gpu_concurrent:
        # Each worker owns a dedicated CUDA stream (created inside
        # _quantize_single_tensor), enabling H2D / compute / D2H overlap.
        futures: dict = {}
        tensor_names = list(raw_tensors.keys())
        with ThreadPoolExecutor(max_workers=_GPU_CONCURRENT_WORKER) as pool:
            for tensor_name in tensor_names:
                tensor = raw_tensors.pop(tensor_name)
                fut = pool.submit(
                    _quantize_single_tensor,
                    tensor_name,
                    tensor,
                    matcher,
                    device,
                )
                futures[fut] = tensor_name

            for fut in as_completed(futures):
                _layer_name, out_dict, q_layer, ig_layer = fut.result()
                output_tensors.update(out_dict)
                if q_layer:
                    quantized_layers.append(q_layer)
                if ig_layer:
                    ignored_layers.append(ig_layer)
            memory_monitor.update()
        torch.cuda.empty_cache()
    else:
        # Serial path (CPU)
        tensor_names = list(raw_tensors.keys())
        for tensor_name in tensor_names:
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


def _write_output_shard(
    output_dir: str,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str],
):
    """Write a single output shard and update the weight_map."""
    from safetensors.torch import save_file

    shard_path = os.path.join(output_dir, shard_name)
    tensors = {k: v if v.is_contiguous() else v.contiguous() for k, v in tensors.items()}
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


def _prefetch_shard(
    model_name_or_path: str,
    shard_name: str,
    work_dir: str,
    source_dir: str,
    streaming: bool,
) -> str | None:
    """Return the local path of the next shard (download if needed).

    Runs in a background thread so the I/O overlaps with quantization
    of the current shard.
    """
    try:
        if streaming:
            return _download_single_shard(model_name_or_path, shard_name, work_dir)
        path = os.path.join(source_dir, shard_name)
        return path if os.path.exists(path) else None
    except Exception as e:  # pragma: no cover
        logger.warning(f"Prefetch failed for {shard_name}: {e}")
        return None


def model_free_quantize(
    model_name_or_path: str,
    output_dir: str,
    scheme: Union[str, QuantizationScheme] = "W4A16",
    layer_config: Optional[dict] = None,
    ignore_layers: str = "",
    format: str = "auto_round",
    device: str = "cpu",
    quant_lm_head: bool = False,
) -> str:
    """Perform model-free RTN quantization.

    Downloads safetensors directly, applies RTN quantization to each eligible
    Linear weight, and saves the packed model in auto-round format.

    The function automatically decides the I/O strategy:

    * If the model is already available locally (a directory path or
      previously cached by ``huggingface_hub``), all shards are read
      directly from disk.
    * Otherwise, shards are downloaded one at a time and deleted after
      quantization to minimise disk usage.

    On **GPU** (``device="cuda"``), multiple weight tensors inside each
    shard are quantized concurrently so that H2D transfers and GPU
    compute overlap.

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
        quant_lm_head: If True, quantize ``lm_head`` as well.  By default
            ``lm_head`` and any layer containing ``embed`` are kept in full
            precision.

    Returns:
        Path to the output directory.
    """
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

    # By default keep lm_head in full precision; embed layers are always
    # skipped via _BLOCK_NAME_TO_IGNORE regardless of quant_lm_head
    if not quant_lm_head:
        if "lm_head" not in ignore_patterns:
            ignore_patterns.append("lm_head")

    # ---- Setup output directory ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- Decide I/O strategy ----
    # If the model is already local or in HF cache, read directly;
    # otherwise download one shard at a time to save disk.
    is_streaming = not _is_model_cached(model_name_or_path)
    if is_streaming:
        logger.info("Model not found locally or in cache — using streaming download mode.")

    # ---- Resolve source and load config ----
    work_dir: str = ""
    source_dir: str = ""
    if is_streaming:
        # Download only metadata first
        work_dir = os.path.join(envs.AR_WORK_SPACE, "_model_free_tmp")
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

    # ---- Build precompiled pattern matcher (shared across all shards) ----
    matcher = _PatternMatcher(ignore_patterns, layer_config, default_scheme)

    # Discover shards
    if is_streaming:
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
        f"  Streaming download: {is_streaming}\n"
        f"  Quant lm_head: {quant_lm_head}\n"
        f"  Device: {device}"
    )

    # ---- Process each shard ----
    all_quantized_layers: list[str] = []
    all_ignored_layers: list[str] = []
    output_weight_map: dict[str, str] = {}

    start_time = time.time()
    memory_monitor.reset()

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    # ---- I/O pipeline: prefetch next shard while quantizing current ----
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    write_pool = ThreadPoolExecutor(max_workers=1)
    prefetch_future = None
    write_future = None

    # Kick off prefetch of the very first shard
    if shard_names:
        prefetch_future = prefetch_pool.submit(
            _prefetch_shard,
            model_name_or_path,
            shard_names[0],
            work_dir,
            source_dir,
            is_streaming,
        )

    shard_iter = (
        _tqdm(enumerate(shard_names), total=len(shard_names), desc="Processing shards", unit="shard")
        if _tqdm
        else enumerate(shard_names)
    )

    for shard_idx, shard_name in shard_iter:
        # Wait for the prefetched shard path
        shard_path = prefetch_future.result() if prefetch_future else None

        # Kick off prefetch of the *next* shard immediately
        if shard_idx + 1 < len(shard_names):
            prefetch_future = prefetch_pool.submit(
                _prefetch_shard,
                model_name_or_path,
                shard_names[shard_idx + 1],
                work_dir,
                source_dir,
                is_streaming,
            )
        else:
            prefetch_future = None

        if shard_path is None or not os.path.exists(shard_path):
            logger.warning(f"Shard not found: {shard_name}, skipping")
            continue

        # Quantize (GPU concurrent or CPU serial, chosen inside _process_shard)
        output_tensors, quantized, ignored = _process_shard(
            shard_path=shard_path,
            device=device,
            matcher=matcher,
        )
        memory_monitor.update()
        mem_summary = memory_monitor.get_summary()
        logger.info(
            f"Finished quantizing shard {shard_name}: {len(quantized)} layers"
        )
        logger.info(
            f"Memory usage: {mem_summary}"
        )

        all_quantized_layers.extend(quantized)
        all_ignored_layers.extend(ignored)

        # Wait for previous async write before starting a new one
        if write_future is not None:
            write_future.result()

        out_shard_name = f"model-{shard_idx + 1:05d}-of-{len(shard_names):05d}.safetensors"
        write_future = write_pool.submit(
            _write_output_shard, output_dir, out_shard_name, output_tensors, output_weight_map
        )

        # In streaming mode, delete the source shard to save disk space
        if is_streaming:
            try:
                os.remove(shard_path)
                logger.debug(f"Deleted source shard: {shard_path}")
            except OSError as e:
                logger.warning(f"Could not delete source shard {shard_path}: {e}")

    prefetch_pool.shutdown(wait=False)

    # Wait for the last async write to complete
    if write_future is not None:
        write_future.result()
    write_pool.shutdown(wait=True)

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

    # ---- Copy non-safetensors files (tokenizer, config, modeling code, etc.) ----
    copy_src_dir = work_dir if is_streaming else source_dir
    for fname in os.listdir(copy_src_dir):
        if _is_safetensors_shard(fname):
            continue
        src = os.path.join(copy_src_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        elif os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    if is_streaming:
        # Clean up temp dir
        try:
            shutil.rmtree(work_dir)
        except OSError:
            pass

    end_time = time.time()
    total_time = end_time - start_time
    # ---- Summary ----
    compressed_quantized = compress_layer_names(all_quantized_layers)
    compressed_ignored = compress_layer_names(list(dict.fromkeys(all_ignored_layers)))
    mem_summary = memory_monitor.get_summary()
    logger.info(
        f"\nModel-free quantization complete.\n"
        f"  Output directory: {output_dir}\n"
        f"  Total time: {total_time:.2f} seconds\n"
        f"  Memory usage: {mem_summary}\n"
        f"  Quantized layers ({len(all_quantized_layers)}): {compressed_quantized}\n"
        f"  Ignored layers ({len(set(all_ignored_layers))}): {compressed_ignored}\n"
    )

    return output_dir
