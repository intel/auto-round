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

"""Utility helpers for model-free quantization.

This module hosts generic helpers used by model-free flow and keeps
model-special handlers at the end of the file.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import warnings
from dataclasses import fields
from functools import lru_cache
from typing import Any, Callable, Optional, Union

import torch

from auto_round.compressors.utils import is_mx_fp, is_nv_fp
from auto_round.logger import logger
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme, preset_name_to_scheme
from auto_round.utils.common import to_standard_regex
from auto_round.utils.device import clear_memory, compile_func
from auto_round.utils.missing_tensors import quantize_weight_rtn, split_fused_expert_tensors

_NVFP4_E5M3_DATA_TYPE = "nvfp4_v2"
_BLOCK_NAME_TO_IGNORE = ("shared_expert_gate.", ".gate.", "embed", "conv")
_SUPPORTED_MXFP_BITS = (4, 8)
_SUPPORTED_INT_BITS = (2, 4, 8)
# Known lm_head layer name variants across model families:
#   "lm_head"  – most models (LLaMA, Mistral, Qwen, …)
#   "head"     – DeepSeek v4
#   "embed_out" – Pythia / Dolly
#   "output"   – some InternLM variants
_LM_HEAD_PATTERNS: tuple[str, ...] = ("lm_head", "head", "embed_out", "output")
SUPPORTED_PRESET_SCHEMES = (
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


# ---------------------------------------------------------------------------
# Generic Utility Helpers
# ---------------------------------------------------------------------------


# Pattern Matching and Layer Scheme Resolution
# ---------------------------------------------------------------------------


class _PatternMatcher:
    """Precompile ignore and layer-config patterns for shard processing."""

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
        self._skip_re = re.compile("|".join(re.escape(name) for name in _BLOCK_NAME_TO_IGNORE))

        self._compiled_lc: list[tuple[re.Pattern | None, str | None, dict]] = []
        for pattern, config in layer_config.items():
            try:
                self._compiled_lc.append((re.compile(to_standard_regex(pattern)), None, config))
            except re.error:
                self._compiled_lc.append((None, pattern, config))

        self._ignore_cache: dict[str, bool] = {}
        self._scheme_cache: dict[str, dict | None] = {}

    @staticmethod
    def _build_ignore_regex(patterns: list[str]) -> re.Pattern | None:
        if not patterns:
            return None
        normalized_patterns: list[str] = []
        for pattern in patterns:
            if pattern.endswith("."):
                normalized = to_standard_regex(pattern.rstrip(".")).removesuffix(".*")
                normalized_patterns.append(f"{normalized}(?:\\.|$)")
            else:
                normalized_patterns.append(to_standard_regex(pattern))
        return re.compile("|".join(normalized_patterns))

    def should_ignore(self, tensor_name: str) -> bool:
        cached = self._ignore_cache.get(tensor_name)
        if cached is not None:
            return cached
        layer_name = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name
        # Explicit layer_config entries take priority: never ignore a layer the user has explicitly configured.
        if layer_name in self._layer_config:
            self._ignore_cache[tensor_name] = False
            return False
        result = bool(self._ignore_re and self._ignore_re.search(layer_name))
        self._ignore_cache[tensor_name] = result
        return result

    def should_skip(self, tensor_name: str) -> bool:
        return bool(self._skip_re.search(tensor_name))

    def resolve_scheme(self, tensor_name: str) -> dict | None:
        if tensor_name not in self._scheme_cache:
            self._scheme_cache[tensor_name] = self._resolve_uncached(tensor_name)
        return self._scheme_cache[tensor_name]

    def _resolve_uncached(self, tensor_name: str) -> dict | None:
        layer_name = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name
        default = self._default_scheme
        if layer_name in self._layer_config:
            config = self._layer_config[layer_name]
            return None if config.get("bits", default.get("bits", 4)) >= 16 else {**default, **config}

        for compiled, plain, config in self._compiled_lc:
            if (compiled is not None and compiled.search(layer_name)) or (plain is not None and plain in layer_name):
                return None if config.get("bits", default.get("bits", 4)) >= 16 else {**default, **config}
        return default


# ---------------------------------------------------------------------------
# Source Tensor Detection and Normalization
# ---------------------------------------------------------------------------


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
            if scale_key in raw_tensors and raw_tensors[scale_key].dtype == torch.uint8:
                entries.append((layer_name, name, scale_key, 8))
        elif name.endswith(".weight_packed") and tensor.dtype in (torch.int8, torch.uint8):
            layer_name = name[: -len(".weight_packed")]
            if (
                f"{layer_name}.weight_global_scale" in raw_tensors
                or f"{layer_name}.input_global_scale" in raw_tensors
                or f"{layer_name}.weight_scale_2" in raw_tensors
                or f"{layer_name}.input_scale" in raw_tensors
            ):
                continue
            scale_key = f"{layer_name}.weight_scale"
            if scale_key in raw_tensors and raw_tensors[scale_key].dtype == torch.uint8:
                entries.append((layer_name, name, scale_key, 4))
    return entries


def _normalize_nvfp4_source_tensors(
    raw_tensors: dict[str, torch.Tensor],
    shard_name: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Normalize legacy NVFP4 source naming to llm-compressor naming."""

    def _reciprocal_global_scale(scale: torch.Tensor) -> torch.Tensor:
        return (1.0 / scale.float()).to(torch.float32).reshape([1])

    converted_layers: list[str] = []
    candidates: list[str] = []
    for name, tensor in list(raw_tensors.items()):
        if name.endswith(".weight"):
            layer_name = name[: -len(".weight")]
        elif name.endswith(".weight_packed"):
            layer_name = name[: -len(".weight_packed")]
        else:
            continue

        if tensor.dtype not in (torch.uint8, torch.int8):
            continue
        if f"{layer_name}.weight_scale" not in raw_tensors:
            continue

        has_legacy_global = f"{layer_name}.weight_scale_2" in raw_tensors or f"{layer_name}.input_scale" in raw_tensors
        has_new_global = (
            f"{layer_name}.weight_global_scale" in raw_tensors or f"{layer_name}.input_global_scale" in raw_tensors
        )
        has_new_packed = f"{layer_name}.weight_packed" in raw_tensors
        if has_legacy_global or has_new_global or has_new_packed:
            candidates.append(layer_name)

    if not candidates:
        return raw_tensors, converted_layers

    for layer_name in candidates:
        weight_key = f"{layer_name}.weight"
        weight_packed_key = f"{layer_name}.weight_packed"
        weight_scale_2_key = f"{layer_name}.weight_scale_2"
        input_scale_key = f"{layer_name}.input_scale"
        weight_global_scale_key = f"{layer_name}.weight_global_scale"
        input_global_scale_key = f"{layer_name}.input_global_scale"

        if weight_packed_key not in raw_tensors and weight_key in raw_tensors:
            raw_tensors[weight_packed_key] = raw_tensors.pop(weight_key).view(torch.uint8).contiguous()

        if weight_scale_2_key in raw_tensors and weight_global_scale_key not in raw_tensors:
            raw_tensors[weight_global_scale_key] = raw_tensors.pop(weight_scale_2_key)
        elif weight_scale_2_key in raw_tensors:
            raw_tensors.pop(weight_scale_2_key)

        if input_scale_key in raw_tensors and input_global_scale_key not in raw_tensors:
            raw_tensors[input_global_scale_key] = raw_tensors.pop(input_scale_key)
        elif input_scale_key in raw_tensors:
            raw_tensors.pop(input_scale_key)

        if weight_global_scale_key in raw_tensors:
            raw_tensors[weight_global_scale_key] = _reciprocal_global_scale(raw_tensors[weight_global_scale_key])
        if input_global_scale_key in raw_tensors:
            raw_tensors[input_global_scale_key] = _reciprocal_global_scale(raw_tensors[input_global_scale_key])

        converted_layers.append(layer_name)

    if converted_layers:
        shard_prefix = f"[{shard_name}] " if shard_name else ""
        logger.info(f"{shard_prefix}Normalized {len(converted_layers)} legacy NVFP4 layer(s) to llm-compressor naming.")
    return raw_tensors, converted_layers


def _handle_nvfp4_source_tensors(
    raw_tensors: dict[str, torch.Tensor],
    matcher: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    """Passthrough NVFP4 source tensors when target scheme for the layer is NVFP4."""
    passthrough_tensors: dict[str, torch.Tensor] = {}
    passthrough_layers: list[str] = []

    for name, tensor in list(raw_tensors.items()):
        if not name.endswith(".weight_packed") or tensor.dtype not in (torch.uint8, torch.int8):
            continue

        layer_name = name[: -len(".weight_packed")]
        scale_key = f"{layer_name}.weight_scale"
        if scale_key not in raw_tensors:
            continue

        scheme = matcher.resolve_scheme(f"{layer_name}.weight")
        if scheme is None:
            continue
        scheme_bits = scheme.get("bits")
        scheme_data_type = (scheme.get("data_type") or "").lower()
        if not (scheme_bits == 4 and (is_nv_fp(scheme_data_type) or scheme_data_type == _NVFP4_E5M3_DATA_TYPE)):
            continue

        keys_to_move = [name, scale_key]
        weight_global_scale_key = f"{layer_name}.weight_global_scale"
        input_global_scale_key = f"{layer_name}.input_global_scale"
        if weight_global_scale_key in raw_tensors:
            keys_to_move.append(weight_global_scale_key)
        if input_global_scale_key in raw_tensors:
            keys_to_move.append(input_global_scale_key)

        for key in keys_to_move:
            value = raw_tensors.pop(key).to("cpu")
            if key.endswith("_global_scale"):
                value = value.reshape([1])
            passthrough_tensors[key] = value
        passthrough_layers.append(layer_name)

    if passthrough_layers:
        logger.info(f"Handling NVFP4 source tensor(s): {len(passthrough_layers)} passthrough layer(s).")

    return raw_tensors, passthrough_tensors, passthrough_layers


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


# ---------------------------------------------------------------------------
# Shard Tensor Processing
# ---------------------------------------------------------------------------


def _is_eligible_weight(tensor_name: str, tensor: torch.Tensor) -> bool:
    """Check if a tensor is eligible for quantization (2D Linear weight)."""
    return tensor_name.endswith(".weight") and tensor.dim() == 2


def _is_moe_fused_expert_weight(tensor_name: str, tensor: torch.Tensor) -> bool:
    """Check if *tensor* is a 3-D fused per-layer stacked MoE expert weight
    (e.g. ``experts.w13_weight`` / ``experts.w2_weight``, shape
    ``[num_experts, out, in]``).

    Such tensors are produced when :func:`split_fused_expert_tensors
    <auto_round.utils.missing_tensors.split_fused_expert_tensors>` skips
    unfusing for the source model's ``model_type`` (see
    ``_KEEP_FUSED_EXPERT_MODEL_TYPES``) because the target inference engine's
    loader expects the fused layout preserved rather than split per expert.

    Matches ``<prefix>.experts.<proj_name>`` only -- ``shared_experts`` (never
    a quantization target) is intentionally excluded by requiring the
    immediate parent segment to be exactly ``experts``.
    """
    if tensor.dim() != 3:
        return False
    parts = tensor_name.split(".")
    return len(parts) >= 2 and parts[-2] == "experts"


def _quantize_moe_fused_expert_weight(
    tensor_name: str,
    tensor: torch.Tensor,
    matcher: "_PatternMatcher",
    device: str = "cpu",
    disable_opt_rtn: bool = False,
) -> tuple[str, dict[str, torch.Tensor], str | None, str | None]:
    """Quantize a 3-D fused per-layer stacked MoE expert weight in place.

    Each expert's 2-D slice (``tensor[i]``) is quantized independently via
    :func:`_quantize_weight_mxfp` and the packed outputs are re-stacked along
    a new leading dimension, preserving the tensor's original fused layout
    (e.g. ``experts.w13_weight.weight_packed`` stays 3-D instead of being
    unfused into per-expert 2-D tensors). RTN/MXFP quantization groups values
    along the last (``in_features``) dimension on a per-row basis, so this is
    numerically equivalent to quantizing whatever 2-D weight each row
    originally came from -- any gate/up row interleaving within a slice does
    not affect correctness.

    Only the MXFP4/MXFP8 path is currently supported for this layout; other
    schemes fall back to keeping the original (unquantized) weight.

    Returns:
        (layer_name, output_tensors_dict, quantized_layer_or_None, ignored_layer_or_None)
    """
    layer_name = tensor_name

    if matcher.should_ignore(tensor_name) or matcher.should_skip(tensor_name):
        return layer_name, {tensor_name: tensor}, None, layer_name

    scheme = matcher.resolve_scheme(tensor_name)
    if scheme is None:
        return layer_name, {tensor_name: tensor}, None, layer_name

    bits = scheme["bits"]
    if bits >= 16:
        return layer_name, {tensor_name: tensor}, None, layer_name

    data_type = (scheme.get("data_type") or "int").lower()
    group_size = scheme["group_size"]

    if not is_mx_fp(data_type):
        logger.warning_once(
            f"3-D fused MoE weight '{tensor_name}' (shape={list(tensor.shape)}) is only "
            "supported for MXFP schemes in model_free mode; keeping original weight."
        )
        return layer_name, {tensor_name: tensor}, None, layer_name

    try:
        packed_parts: dict[str, list[torch.Tensor]] = {}
        for i in range(tensor.shape[0]):
            slice_out = _quantize_weight_mxfp(
                weight=tensor[i],
                layer_name=f"{layer_name}.{i}",
                bits=bits,
                group_size=group_size,
                data_type=data_type,
                device=device,
                disable_opt_rtn=disable_opt_rtn,
            )
            prefix = f"{layer_name}.{i}"
            for key, value in slice_out.items():
                suffix = key[len(prefix) :]  # e.g. ".weight_packed" / ".weight_scale"
                packed_parts.setdefault(suffix, []).append(value)

        out = {f"{layer_name}{suffix}": torch.stack(values, dim=0) for suffix, values in packed_parts.items()}
        logger.debug(
            f"Quantized (MXFP, fused 3-D): {layer_name} "
            f"(bits={bits}, group_size={group_size}, num_experts={tensor.shape[0]})"
        )
        return layer_name, out, layer_name, None
    except Exception as e:
        logger.warning(f"Failed to MXFP-quantize fused 3-D MoE weight {layer_name}: {e}. Keeping original weight.")
        return layer_name, {tensor_name: tensor}, None, layer_name


def _quantize_weight_mxfp(
    weight: torch.Tensor,
    layer_name: str,
    bits: int,
    group_size: int,
    data_type: str,
    device: str = "cpu",
    disable_opt_rtn: bool = False,
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

    from auto_round.data_type.utils import get_quant_func
    from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

    if not is_mx_fp(data_type):
        data_type = "mx_fp4" if bits == 4 else "mx_fp8"

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} for layer '{layer_name}' is not divisible "
            f"by MXFP group_size={group_size}; cannot pack."
        )

    weight_dev = weight.to(device)
    # Use get_quant_func (same as WrapperLinear) so that all registered MXFP
    # variants automatically get opt_rtn support via the QUANT_FUNC_WITH_DTYPE
    # registry (e.g. "opt_rtn_mx_fp4" -> quant_mx_opt_rtn, "mx_fp4" -> quant_mx).
    quant_func, _ = get_quant_func(data_type, bits, sym=True, disable_opt_rtn=disable_opt_rtn, iters=0)
    weight_dev, shared_exp, _ = quant_func(weight_dev, bits=bits, group_size=group_size, data_type=data_type)
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


def _quantize_weight_nvfp4_e5m3(
    weight: torch.Tensor,
    layer_name: str,
    group_size: int = 16,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Fake-quantize a 2D weight tensor to NVFP4 E5M3 and return its high-precision QDQ weight."""
    from auto_round.data_type.nvfp import nvfp4_v2

    out_features, in_features = weight.shape
    if group_size != 16:
        raise ValueError(f"NVFP4_E5M3 requires group_size=16, got {group_size} for layer '{layer_name}'.")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} for layer '{layer_name}' is not divisible "
            f"by NVFP4_E5M3 group_size={group_size}; cannot quantize."
        )

    weight_dev = weight.to(device)
    qdq_weight, _, _ = nvfp4_v2(weight_dev, bits=4, group_size=group_size)
    return {f"{layer_name}.weight": qdq_weight.to(dtype=weight.dtype, device="cpu")}


def _pack_weight_nvfp4_e5m3(
    weight: torch.Tensor,
    layer_name: str,
    group_size: int = 16,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Pack FP4 E2M1 weights with unsigned E5M3 block scales."""
    from auto_round.data_type.nvfp import nvfp4_v2
    from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear

    out_features, in_features = weight.shape
    if group_size != 16 or in_features % group_size != 0:
        raise ValueError(
            f"NVFP4_E5M3 requires in_features divisible by group_size=16, got {in_features} for '{layer_name}'."
        )
    weight_dev = weight.to(device)
    _, scale, _ = nvfp4_v2(weight_dev, bits=4, group_size=group_size)
    # nvfp4_v2 may return a flattened per-group scale layout (e.g. [N, 1]);
    # normalize to [out_features, in_features // group_size] before packing
    # so serialized .weight_scale keeps the expected 2D shape.
    scale = scale.reshape(out_features, in_features // group_size).to(torch.float32)
    linear = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=weight.dtype)
    linear.weight = torch.nn.Parameter(weight_dev, requires_grad=False)
    qlayer = QuantLinear(
        4, group_size, in_features, out_features, False, data_type="nvfp4_v2", act_bits=4, act_data_type="nvfp4_v2"
    )
    qlayer.pack(linear, scale, device=device)
    return {
        f"{layer_name}.weight_packed": qlayer.weight_packed.to("cpu"),
        f"{layer_name}.weight_scale": qlayer.weight_scale.to("cpu"),
    }


def _quantize_single_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    matcher: "_PatternMatcher",
    device: str = "cpu",
    quantize_func: Callable = quantize_weight_rtn,
    disable_opt_rtn: bool = False,
) -> tuple[str, dict[str, torch.Tensor], str | None, str | None]:
    """Quantize one eligible weight tensor and return packed outputs.

    Returns:
        (layer_name, output_tensors_dict, quantized_layer_or_None, ignored_layer_or_None)
    """
    if _is_moe_fused_expert_weight(tensor_name, tensor):
        return _quantize_moe_fused_expert_weight(tensor_name, tensor, matcher, device, disable_opt_rtn)

    layer_name = tensor_name.rsplit(".", 1)[0]

    if not _is_eligible_weight(tensor_name, tensor):
        ignored_layer = layer_name if tensor_name.endswith(".weight") and tensor.dim() > 1 else None
        return layer_name, {tensor_name: tensor}, None, ignored_layer

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
                disable_opt_rtn=disable_opt_rtn,
            )
            logger.debug(f"Quantized (MXFP): {layer_name} (bits={bits}, group_size={group_size})")
            return layer_name, out, layer_name, None
        except Exception as e:
            logger.warning(f"Failed to MXFP-quantize {layer_name}: {e}. Keeping original weight.")
            return layer_name, {tensor_name: tensor}, None, layer_name

    # ---- NVFP4 E5M3 fake-quantization path ----
    if data_type == _NVFP4_E5M3_DATA_TYPE:
        try:
            quantize_e5m3 = (
                _quantize_weight_nvfp4_e5m3 if scheme.get("_output_format") == "fake" else _pack_weight_nvfp4_e5m3
            )
            out = quantize_e5m3(
                weight=tensor,
                layer_name=layer_name,
                group_size=group_size,
                device=device,
            )
            logger.debug(f"Quantized (NVFP4_E5M3): {layer_name} (bits=4, group_size={group_size})")
            return layer_name, out, layer_name, None
        except Exception as e:
            logger.warning(f"Failed to NVFP4_E5M3-quantize {layer_name}: {e}. Keeping original weight.")
            return layer_name, {tensor_name: tensor}, None, layer_name

    # ---- Integer WOQ path ----
    # opt_rtn is always disabled for integer WOQ in model-free mode because
    # the scale search does not improve accuracy for INT quantization here.
    try:
        qweight, qzeros, scales = quantize_func(
            weight=tensor,
            bits=bits,
            group_size=group_size,
            sym=sym,
            device=device,
            disable_opt_rtn=True,
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


@lru_cache(maxsize=32)
def _load_weight_map_from_index(index_path: str) -> dict[str, str]:
    """Load weight_map from an index file with a small process-local cache."""
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    return weight_map if isinstance(weight_map, dict) else {}


def _build_cross_shard_pairs_from_weight_map(
    weight_map: dict[str, str],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, set[str]]]:
    """Identify cross-shard FP8 (weight, weight_scale_inv) pairs from weight_map.

    A *cross-shard pair* exists when ``<layer>.weight`` and
    ``<layer>.weight_scale_inv`` reside in **different** shards.  The shard
    that holds ``weight_scale_inv`` is the *donor*; the shard that holds
    ``weight`` is the *recipient*.

    Returns:
        recipient_to_donors:
            ``{recipient_shard: {donor_shard: [scale_inv_tensor_names]}}``
        donor_shard_tensors:
            ``{donor_shard: set(scale_inv tensor names it donates to other shards)}``
    """
    recipient_to_donors: dict[str, dict[str, list[str]]] = {}
    donor_shard_tensors: dict[str, set[str]] = {}

    for tensor_name, shard in weight_map.items():
        if not tensor_name.endswith(".weight_scale_inv"):
            continue
        # "layer.weight_scale_inv" -> "layer.weight"
        weight_name = tensor_name[: -len("_scale_inv")]
        weight_shard = weight_map.get(weight_name)
        if not weight_shard or weight_shard == shard:
            continue
        # `shard` (donor) donates `tensor_name` to `weight_shard` (recipient)
        recipient_to_donors.setdefault(weight_shard, {}).setdefault(shard, []).append(tensor_name)
        donor_shard_tensors.setdefault(shard, set()).add(tensor_name)

    return recipient_to_donors, donor_shard_tensors


def _hydrate_missing_fp8_scales_from_index(
    raw_tensors: dict[str, torch.Tensor],
    shard_path: str,
    *,
    shard_name: str | None = None,
    index_dir: str | None = None,
    donor_shard_dir: str | None = None,
) -> dict[str, torch.Tensor]:
    """Populate missing ``.weight_scale_inv`` tensors from sibling shards.

    Some checkpoints shard FP8 weight and its corresponding scale tensor into
    different ``.safetensors`` files. Model-free processing is shard-local, so
    this helper hydrates missing ``<layer>.weight_scale_inv`` tensors by looking
    up ``weight_map`` in ``*.safetensors.index.json`` and loading only the
    needed tensors from referenced shards.

    Args:
        index_dir: Directory that contains ``*.safetensors.index.json``.  When
            ``None`` (default) the directory of *shard_path* is used, which is
            correct for non-streaming mode.  Streaming mode should pass
            ``work_dir`` here because downloaded shards live in a
            ``.cache/model_free_source_shards/`` sub-directory that does not
            contain the index file.
        donor_shard_dir: Directory where donor shard files can be found.  When
            ``None`` (default) the same directory as *index_dir* is used.
            Streaming mode should pass the local shard cache directory so that
            already-downloaded donor shards are resolved correctly.
    """
    if not shard_path.endswith(".safetensors"):
        return raw_tensors

    weight_to_scale: dict[str, str] = {}
    for name, tensor in raw_tensors.items():
        if not name.endswith(".weight"):
            continue
        if tensor.dtype != torch.float8_e4m3fn:
            continue
        scale_inv_name = f"{name[: -len('.weight')]}.weight_scale_inv"
        if scale_inv_name not in raw_tensors:
            weight_to_scale[name] = scale_inv_name

    if not weight_to_scale:
        return raw_tensors

    shard_dir = os.path.dirname(shard_path)
    # Resolve the directory where index.json lives.  In streaming mode the
    # shard lives in a .cache/ sub-directory that has no index file; callers
    # pass index_dir=work_dir to point at the correct location.
    idx_dir = index_dir if index_dir is not None else shard_dir
    # Donor shards are looked up in donor_shard_dir (defaults to idx_dir).
    donor_dir = donor_shard_dir if donor_shard_dir is not None else idx_dir

    index_path = os.path.join(idx_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        candidates = sorted(
            os.path.join(idx_dir, f) for f in os.listdir(idx_dir) if f.endswith(".safetensors.index.json")
        )
        if not candidates:
            return raw_tensors
        index_path = candidates[0]

    try:
        weight_map = _load_weight_map_from_index(index_path)
    except Exception:
        return raw_tensors

    current_shard = os.path.basename(shard_path)
    scales_by_shard: dict[str, list[str]] = {}
    for scale_name in weight_to_scale.values():
        target_shard = weight_map.get(scale_name)
        if not target_shard or target_shard == current_shard:
            continue
        scales_by_shard.setdefault(target_shard, []).append(scale_name)

    if not scales_by_shard:
        return raw_tensors

    from safetensors import safe_open

    hydrated = 0
    shard_prefix = f"[{shard_name}] " if shard_name else ""
    for target_shard, scale_names in scales_by_shard.items():
        target_path = os.path.join(donor_dir, target_shard)
        if not os.path.exists(target_path):
            logger.warning(
                f"{shard_prefix}Donor shard '{target_shard}' not found in '{donor_dir}' while hydrating "
                f"{len(scale_names)} FP8 scale_inv tensor(s); the affected weight(s) will remain in "
                f"float8_e4m3fn and may fail downstream quantization. This usually indicates a shard "
                f"scheduling/ordering issue (donor shard processed/downloaded after its recipient)."
            )
            continue
        try:
            with safe_open(target_path, framework="pt", device="cpu") as sf:
                for scale_name in scale_names:
                    if scale_name in raw_tensors:
                        continue
                    try:
                        raw_tensors[scale_name] = sf.get_tensor(scale_name)
                        hydrated += 1
                    except Exception:
                        # Tensor may be absent in this shard; skip lazily.
                        continue
        except Exception:
            continue

    if hydrated:
        logger.info(f"{shard_prefix}Hydrated {hydrated} FP8 scale tensor(s) from sibling shard(s) using index mapping.")

    return raw_tensors


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
        is_ignored = matcher.should_ignore(f"{layer_name}.weight")
        target_is_same_mxfp = (
            scheme is not None and is_mx_fp((scheme.get("data_type") or "").lower()) and scheme.get("bits") == bits
        )
        if target_is_same_mxfp and not is_ignored:
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
    shard_path: str | None = None,
    *,
    index_dir: str | None = None,
    donor_shard_dir: str | None = None,
) -> dict[str, torch.Tensor]:
    """Dequantize DeepSeek-V3-style FP8 weight tensors to bfloat16.

    Handles the **DeepSeek-V3 FP8** convention: weight dtype ``float8_e4m3fn``
    paired with a ``.weight_scale_inv`` tensor (per-block float32 scales, NOT
    E8M0).  The weights are converted to ``bfloat16`` so downstream RTN
    quantization can proceed normally.

    MXFP sources are handled separately by
    :func:`preprocess_model_type_source_tensors` / :func:`_handle_mxfp_source_tensors`.
    """
    from auto_round.utils.weight_handler import _dequant_fp8_linear_weight

    if shard_path:
        raw_tensors = _hydrate_missing_fp8_scales_from_index(
            raw_tensors,
            shard_path,
            shard_name=shard_name,
            index_dir=index_dir,
            donor_shard_dir=donor_shard_dir,
        )

    quant_entries: list[tuple[str, str]] = []
    for name, tensor in raw_tensors.items():
        if not name.endswith(".weight"):
            continue
        if tensor.dtype != torch.float8_e4m3fn:
            continue
        # DeepSeek-V3 style: .weight_scale_inv (per-block float32 scales).
        scale_inv_name = f"{name[: -len('.weight')]}.weight_scale_inv"
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
    source_quantization_config: dict | None = None,
    enable_torch_compile: bool = False,
    disable_opt_rtn: bool = False,
    index_dir: str | None = None,
    donor_shard_dir: str | None = None,
    donor_tensors_to_exclude: set[str] | None = None,
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

    Args:
        index_dir: Directory containing the model's ``*.safetensors.index.json``.
            Used to resolve cross-shard FP8 scale_inv tensors.  Defaults to the
            directory of *shard_path* (correct for non-streaming mode).
        donor_shard_dir: Directory where donor shards are cached.  Defaults to
            *index_dir*.  In streaming mode pass the local shard cache directory.
        donor_tensors_to_exclude: If provided, these ``weight_scale_inv`` tensor
            names are removed from the final output tensors.  Used when this
            shard is a donor: its scale_inv tensors have already been consumed
            by the recipient shard and must not appear in the quantized output.
    """
    if matcher is None:
        matcher = _PatternMatcher(
            ignore_patterns if ignore_patterns is not None else [],
            layer_config if layer_config is not None else {},
            default_scheme if default_scheme is not None else {},
        )

    output_tensors: dict[str, torch.Tensor] = {}
    quantized_layers: list[str] = []
    quantize_func = compile_func(quantize_weight_rtn, device) if enable_torch_compile else quantize_weight_rtn

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

    raw_tensors = split_fused_expert_tensors(raw_tensors, model_type=model_type)

    # Hydrate cross-shard FP8 weight_scale_inv tensors *before* any
    # preprocessing below. Otherwise a weight whose scale lives in a sibling
    # shard would miss the model_type-specific "expand-scale" passthrough
    # handling (step 1) simply because its scale wasn't loaded yet, and would
    # incorrectly fall through to the generic dequant-to-bf16 + RTN-requantize
    # path instead of being treated identically to same-shard weights.
    if shard_path:
        raw_tensors = _hydrate_missing_fp8_scales_from_index(
            raw_tensors,
            shard_path,
            shard_name=shard_name,
            index_dir=index_dir,
            donor_shard_dir=donor_shard_dir,
        )

    # Snapshot candidate weight layer names *before* any preprocessing. 1D
    # weights (for example LayerNorm) are not quantization targets, while 3D
    # weights remain tracked so unsupported layouts are visible as ignored.
    # Fused 3-D MoE expert weights (e.g. ``experts.w13_weight``, kept fused
    # for model_type architectures like Inkling) have no ``.weight`` suffix
    # of their own, so they are tracked separately here.
    input_weight_layers: list[str] = list(
        dict.fromkeys(
            name.rsplit(".", 1)[0] if name.endswith(".weight") else name
            for name, tensor in raw_tensors.items()
            if (name.endswith(".weight") and tensor.dim() > 1) or _is_moe_fused_expert_weight(name, tensor)
        )
    )

    # Preserve original tensors for predefined skipped layers so that already-
    # quantized weights (FP8, FP4-packed, etc.) are NOT dequantized.
    # User-specified ignore layers should still flow through the dequant path
    # so the saved model exports them in full precision.
    preserved_prefixes: set[str] = set()
    for tname in raw_tensors:
        if (
            tname.endswith(".weight") or tname.endswith(".weight_packed") or tname.endswith(".qweight")
        ) and matcher.should_skip(tname):
            preserved_prefixes.add(tname.rsplit(".", 1)[0])

    preserved_tensors: dict[str, torch.Tensor] = {}
    if preserved_prefixes:
        for key in list(raw_tensors.keys()):
            prefix = key.rsplit(".", 1)[0]
            if prefix in preserved_prefixes:
                preserved_tensors[key] = raw_tensors.pop(key)

    # 1) model-type-specific preprocessing (format conversion only)
    raw_tensors, source_state = preprocess_model_type_source_tensors(
        raw_tensors,
        model_type=model_type,
        quantization_config=source_quantization_config,
        shard_name=shard_name,
    )

    # 1.5) normalize legacy NVFP4 names to llm-compressor naming.
    raw_tensors, _converted_nvfp4_layers = _normalize_nvfp4_source_tensors(raw_tensors, shard_name=shard_name)

    # 1.5) model-type-specific low-precision dequantization (e.g. kimi_k25 INT4)
    raw_tensors = handle_model_type_low_precision_source_tensors(
        raw_tensors,
        model_type=model_type,
        source_quant_config=source_quantization_config,
        device=device,
        shard_name=shard_name,
        dequantize_with_device_fallback=_dequantize_with_device_fallback,
    )

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

    # 3) NVFP4 passthrough for layers already stored in packed format.
    raw_tensors, nvfp_passthrough_tensors, nvfp_passthrough_layers = _handle_nvfp4_source_tensors(
        raw_tensors,
        matcher,
    )
    output_tensors.update(nvfp_passthrough_tensors)
    quantized_layers.extend(nvfp_passthrough_layers)

    raw_tensors = _dequant_fp8_tensors(
        raw_tensors,
        block_size=fp8_block_size,
        device=device,
        shard_name=shard_name,
        shard_path=shard_path,
        index_dir=index_dir,
        donor_shard_dir=donor_shard_dir,
    )
    raw_tensors.update(preserved_tensors)

    for tensor_name in list(raw_tensors.keys()):
        tensor = raw_tensors.pop(tensor_name)
        _layer_name, out_dict, q_layer, _ig_layer = _quantize_single_tensor(
            tensor_name,
            tensor,
            matcher,
            device,
            quantize_func,
            disable_opt_rtn,
        )
        output_tensors.update(out_dict)
        if q_layer:
            quantized_layers.append(q_layer)

    # Remove scale_inv tensors that this shard donates to other shards.
    # These tensors have no corresponding weight in this shard; keeping them
    # would pollute the quantized output with raw FP8 scale metadata.
    if donor_tensors_to_exclude:
        removed = {t for t in donor_tensors_to_exclude if t in output_tensors}
        for t in removed:
            del output_tensors[t]
        if removed:
            shard_prefix = f"[{shard_name}] " if shard_name else ""
            logger.debug(
                f"{shard_prefix}Excluded {len(removed)} donated cross-shard scale_inv "
                f"tensor(s) from quantized shard output."
            )

    # Derive ignored layers by comparing input weight layers with quantized set.
    quantized_set = set(quantized_layers)
    ignored_layers: list[str] = [l for l in input_weight_layers if l not in quantized_set]

    return output_tensors, quantized_layers, ignored_layers


# ---------------------------------------------------------------------------
# Model Source I/O and Shard Discovery
# ---------------------------------------------------------------------------


def _get_model_cache_status(model_name_or_path: str) -> tuple[bool, str]:
    """Return cache decision and a short reason string.

    Cached means local dir, or HF cache contains config plus at least one
    weight entry file (index or single-file checkpoint).
    """
    if os.path.isdir(model_name_or_path):
        return True, "input is an existing local directory"

    try:
        from huggingface_hub import try_to_load_from_cache

        # ``config.json`` alone is not enough: many workflows prefetch only
        # config/tokenizer files, which would otherwise route to full
        # ``snapshot_download``.
        config_cached = isinstance(try_to_load_from_cache(model_name_or_path, "config.json"), str)
        if not config_cached:
            return False, "HF cache miss: config.json not found"

        weight_entry_candidates = (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "pytorch_model.bin",
        )
        hit_entries: list[str] = []
        for fname in weight_entry_candidates:
            if isinstance(try_to_load_from_cache(model_name_or_path, fname), str):
                hit_entries.append(fname)

        if hit_entries:
            return True, f"HF cache hit: config.json + {', '.join(hit_entries)}"
        return (
            False,
            "HF cache partial hit: config.json exists but no weight entry file found",
        )
    except Exception as e:
        return False, f"cache probe failed: {type(e).__name__}: {e}"


def _is_model_cached(model_name_or_path: str) -> bool:
    """Return True if the model is already available locally or in HF cache."""
    cached, _ = _get_model_cache_status(model_name_or_path)
    return cached


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
# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Quantization Configuration
# ---------------------------------------------------------------------------


def _get_llm_compressor_metadata() -> dict[str, str]:
    """Return AutoRound provenance for model-free llm-compressor output."""
    # Keep metadata deterministic and installation-source agnostic.
    return {"provider": "auto-round"}


def _build_nvfp4_e5m3_quantization_config(ignored_layers: list[str]) -> dict:
    """Build compressed-tensors metadata for NVFP4 E5M3 without global scales."""
    from auto_round.export.export_to_llmcompressor.config import initialize_nvfp4_e5m3_quantization

    qconfig = initialize_nvfp4_e5m3_quantization(ignore=ignored_layers)
    qconfig.update(_get_llm_compressor_metadata())
    return qconfig


# TODO: Remove when the issue is fixed
# https://github.com/vllm-project/llm-compressor/issues/3069
def _add_routed_experts_if_moe(targets: list[str], layer_names: list[str]) -> list[str]:
    """Append ``"RoutedExperts"`` to *targets* when *layer_names* indicates a MoE model.

    LLM-Compressor has a known bug where MoE expert projections (e.g. ``w1``/
    ``w3`` or layers inside ``block_sparse_moe``) are not matched by the generic
    ``"Linear"`` target and require the explicit ``"RoutedExperts"`` target to be
    present in the config group.  This helper detects that situation and injects
    the extra target so quantization is applied correctly.

    A model is considered MoE if any layer name contains:

    * ``.w1`` or ``.w3`` (typical Mixtral / DeepSeek gating projections), or
    * ``block_sparse_moe`` (Mixtral-style naming).
    """
    if "RoutedExperts" in targets:
        return targets
    moe_re = re.compile(r"\.w[13](?:\.|$)|block_sparse_moe")
    if any(moe_re.search(name) for name in layer_names):
        return list(targets) + ["RoutedExperts"]
    return targets


def _get_mxfp_group_scheme_and_format(
    group_bits: int,
    group_data_type: str,
    ignore: list[str],
):
    """Return ``(QuantizationScheme, format_str)`` for a single (bits, data_type) group.

    Handles MXFP (mx_fp), NVFP4 (nv_fp), and NVFP4_E5M3 (nvfp4_v2) groups.
    """
    from auto_round.export.export_to_llmcompressor.config import (
        initialize_nvfp4_e5m3_quantization,
        initialize_quantization,
    )

    if is_nv_fp(group_data_type):
        # Standard NVFP4 uses global scales and has a dedicated preset in
        # compressed-tensors. Reuse that preset so metadata matches tensor
        # layout when model-free does NVFP4 passthrough.
        tmp_qconfig = initialize_quantization(scheme="NVFP4", ignore=ignore)
        return tmp_qconfig.config_groups["group_0"], "nvfp4-pack-quantized"

    if group_data_type == _NVFP4_E5M3_DATA_TYPE:
        # NVFP4_E5M3 is not a compressed-tensors preset; use its dedicated builder.
        nvfp4_dict = initialize_nvfp4_e5m3_quantization(ignore=ignore)
        # Extract the QuantizationScheme object from the dict config.
        from compressed_tensors.quantization import QuantizationScheme  # pylint: disable=E0401

        raw_scheme = nvfp4_dict["config_groups"]["group_0"]
        scheme_obj = QuantizationScheme.model_validate(raw_scheme)
        return scheme_obj, "nvfp4-e5m3-pack-quantized"
    else:
        scheme_name = "MXFP4" if group_bits == 4 else "MXFP8"
        fmt = "mxfp4-pack-quantized" if group_bits == 4 else "mxfp8-quantized"
        tmp_qconfig = initialize_quantization(scheme=scheme_name, ignore=ignore)
        return tmp_qconfig.config_groups["group_0"], fmt


def _build_mxfp_quantization_config(
    default_scheme: dict,
    quantized_layers: list[str],
    ignored_layers: list[str],
    layer_config: dict | None = None,
) -> dict:
    """Build a compressed-tensors / llm-compressor style quantization_config
    dict for MXFP4 / MXFP8 model-free output, including mixed-precision cases.

    When *layer_config* contains layers that override the default bits or
    data_type (e.g. some layers are MXFP8 while the default is MXFP4, or
    some layers use NVFP4_E5M3 while the default is MXFP8), the function
    creates one ``config_group`` per distinct ``(bits, data_type)`` pair.
    Override groups list their layers explicitly; the default group uses
    ``targets=["Linear"]`` as a catch-all.  The top-level ``"format"`` is
    set to ``"mixed-precision"`` when more than one group is produced.

    Mirrors the per-group format produced by
    :mod:`auto_round.export.export_to_llmcompressor.export_to_fp`.
    """
    from auto_round.export.export_to_llmcompressor.config import (
        initialize_quantization,
    )

    bits = default_scheme.get("bits", 4)
    default_data_type = (default_scheme.get("data_type") or "mx_fp").lower()
    is_fp_default = (bits or 0) >= 16  # BF16/FP16 full-precision default

    if not is_fp_default and bits not in _SUPPORTED_MXFP_BITS and default_data_type != _NVFP4_E5M3_DATA_TYPE:
        raise ValueError(f"Unsupported MXFP bits={bits} for model-free output.")

    # Default ignore list: any layer present in ignored_layers (deduped) that
    # was NOT quantized.
    ignore = list(dict.fromkeys(ignored_layers))
    quant_set = set(quantized_layers)
    ignore = [n for n in ignore if n not in quant_set]

    # Resolve each quantized layer's effective (bits, data_type) using layer_config overrides.
    # Key: (bits, data_type) to distinguish e.g. MXFP4 from NVFP4_E5M3 (both 4-bit).
    scheme_groups: dict[tuple[int, str], list[str]] = {}  # (bits, data_type) -> [layer_names]
    if layer_config:
        temp_matcher = _PatternMatcher(
            ignore_patterns=[],
            layer_config=layer_config,
            default_scheme=default_scheme,
        )
        for layer in quantized_layers:
            scheme = temp_matcher.resolve_scheme(f"{layer}.weight")
            layer_bits = scheme.get("bits", bits) if scheme is not None else bits
            layer_dt = (
                (scheme.get("data_type") or default_data_type).lower() if scheme is not None else default_data_type
            )
            scheme_groups.setdefault((layer_bits, layer_dt), []).append(layer)
    else:
        if not is_fp_default:
            scheme_groups[(bits, default_data_type)] = list(quantized_layers)
        # else: BF16 default with no layer_config → no MXFP layers; scheme_groups stays {}

    if len(scheme_groups) <= 1:
        # Single scheme — use the actual (bits, data_type) from the group.
        if scheme_groups:
            actual_bits, actual_dt = next(iter(scheme_groups.keys()))
        else:
            actual_bits, actual_dt = bits, default_data_type
        if actual_dt != _NVFP4_E5M3_DATA_TYPE and actual_bits not in _SUPPORTED_MXFP_BITS:
            raise ValueError(f"Unsupported MXFP bits={actual_bits} for model-free output.")
        group_scheme, fmt = _get_mxfp_group_scheme_and_format(actual_bits, actual_dt, ignore)
        if actual_dt == _NVFP4_E5M3_DATA_TYPE:
            from auto_round.export.export_to_llmcompressor.config import initialize_nvfp4_e5m3_quantization

            qconfig = initialize_nvfp4_e5m3_quantization(ignore=ignore)
            if is_fp_default and scheme_groups:
                targets = _add_routed_experts_if_moe(list(quantized_layers), quantized_layers)
                qconfig["config_groups"]["group_0"]["targets"] = targets
            qconfig["format"] = fmt
            qconfig.update(_get_llm_compressor_metadata())
            return qconfig
        from auto_round.export.export_to_llmcompressor.config import initialize_quantization as _init_q

        scheme_name = "MXFP4" if actual_bits == 4 else "MXFP8"
        qconfig = _init_q(scheme=scheme_name, ignore=ignore)
        if is_fp_default and scheme_groups:
            targets = _add_routed_experts_if_moe(list(quantized_layers), quantized_layers)
            qconfig.config_groups["group_0"].targets = targets
        qconfig = qconfig.to_dict()
        qconfig["format"] = fmt
        qconfig.update(_get_llm_compressor_metadata())
        return qconfig

    # Mixed precision: build one config_group per distinct (bits, data_type).
    # Override groups (non-default key) come first, default group last,
    # ordered by descending bit-width within each partition so that the
    # higher-precision group gets the lower group index.
    default_key = (bits, default_data_type)
    override_items = sorted(
        [(key, layers) for key, layers in scheme_groups.items() if key != default_key],
        key=lambda x: x[0][0],
        reverse=True,
    )
    default_item = (default_key, scheme_groups[default_key]) if default_key in scheme_groups else None
    ordered = override_items + ([default_item] if default_item else [])

    config_groups: dict = {}
    group_formats: dict[str, str] = {}
    for idx, ((group_bits, group_dt), layer_names) in enumerate(ordered):
        group_name = f"group_{idx}"
        is_default_group = (group_bits, group_dt) == default_key
        targets = ["Linear"] if is_default_group else layer_names
        targets = _add_routed_experts_if_moe(targets, layer_names)
        group_scheme, fmt = _get_mxfp_group_scheme_and_format(group_bits, group_dt, ignore)
        group_scheme.targets = targets
        config_groups[group_name] = group_scheme
        group_formats[group_name] = fmt

    full_qconfig = initialize_quantization(scheme=None, config_groups=config_groups, ignore=ignore)
    full_dict = full_qconfig.to_dict()
    full_dict["format"] = "mixed-precision"
    for group_name, fmt in group_formats.items():
        full_dict["config_groups"][group_name]["format"] = fmt
    full_dict.update(_get_llm_compressor_metadata())
    return full_dict


def _build_mxfp_autoround_quantization_config(
    default_scheme: dict,
    quantized_layers: list[str],
    ignored_layers: list[str],
    layer_config: dict | None = None,
    block_name_to_quantize: Optional[str] = None,
) -> dict:
    """Build an auto-round style quantization_config for MXFP4 / MXFP8.

    Unlike :func:`_build_mxfp_quantization_config` which produces a
    compressed-tensors / llm-compressor style config, this function produces
    an ``auto-round`` style config (``quant_method="auto-round"``,
    ``packing_format="auto_round:llm_compressor"``).  The on-disk weight
    tensors are identical; only the ``quantization_config`` metadata differs.

    The generated config mirrors the layout of the regular AutoRound MXFP
    export, including activation quantization fields (``act_bits`` /
    ``act_data_type`` / …) from the scheme, ``enable_quanted_input``, and
    ``lm_head`` in ``extra_config`` when it was quantized.
    """
    from collections import Counter

    from auto_round.version import __version__

    bits = default_scheme.get("bits", 4)
    group_size = default_scheme.get("group_size", 32)
    is_fp_default = (bits or 0) >= 16

    # For BF16 default + MXFP layer_config overrides, derive the dominant
    # MXFP bit-width from the layers that were actually quantized.
    if is_fp_default and layer_config and quantized_layers:
        temp_matcher = _PatternMatcher(
            ignore_patterns=[],
            layer_config=layer_config,
            default_scheme=default_scheme,
        )
        mxfp_counter: Counter = Counter()
        for layer in quantized_layers:
            scheme = temp_matcher.resolve_scheme(f"{layer}.weight")
            if scheme is None:
                continue
            lb = scheme.get("bits")
            ldt = (scheme.get("data_type") or "").lower()
            if lb and lb < 16 and is_mx_fp(ldt):
                mxfp_counter[lb] += 1
        if mxfp_counter:
            bits, _ = mxfp_counter.most_common(1)[0]
            group_size = 32

    if (bits or 0) < 1 or bits not in _SUPPORTED_MXFP_BITS:
        bits = 4  # safe fallback

    qconfig: dict = {
        "quant_method": "auto-round",
        "packing_format": "auto_round:llm_compressor",
        "bits": bits,
        "group_size": group_size or 32,
        "sym": True,  # MXFP is always symmetric
        "data_type": "mx_fp",
        "iters": 0,
        "model_free": True,
        "autoround_version": __version__,
        "enable_quanted_input": False,
    }
    if block_name_to_quantize:
        qconfig["block_name_to_quantize"] = block_name_to_quantize

    # Carry activation quantization fields from the scheme (e.g. MXFP4 has
    # act_bits=4, act_data_type="mx_fp", act_dynamic=True, act_group_size=32,
    # act_sym=True).  Including these keeps the config consistent with the
    # output produced by the regular AutoRound MXFP export flow.
    for act_key in ("act_bits", "act_data_type", "act_dynamic", "act_group_size", "act_sym"):
        val = default_scheme.get(act_key)
        if val is not None:
            qconfig[act_key] = val

    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    extra_config: dict = {}
    non_linear_ops = ["embed", "conv"]
    non_linear_re = re.compile("|".join(re.escape(op) for op in non_linear_ops))

    if layer_config:
        from auto_round.export.export_to_autoround.utils import check_neq_config

        expected_scheme = {key: qconfig.get(key) for key in scheme_keys}
        for layer_name, cfg in layer_config.items():
            if not isinstance(cfg, dict):
                continue
            neq_keys = check_neq_config(cfg, **expected_scheme)
            if neq_keys:
                extra_config[layer_name] = {key: cfg[key] for key in neq_keys if cfg.get(key) is not None}

    quantized_set = set(quantized_layers)
    unique_ignored = list(dict.fromkeys(ignored_layers))
    for layer_name in unique_ignored:
        if layer_name in quantized_set or layer_name in extra_config:
            continue
        if non_linear_re.search(layer_name):
            continue
        extra_config[layer_name] = {
            "bits": 16,
            "data_type": "float",
            "act_bits": 16,
            "act_data_type": "float",
        }

    # lm_head variants: when explicitly quantized, record each variant's full scheme in extra_config
    # (mirrors the regular AutoRound MXFP export behavior).
    for lm_head_name in _LM_HEAD_PATTERNS:
        if lm_head_name in quantized_set and lm_head_name not in extra_config:
            lm_head_cfg = (layer_config or {}).get(lm_head_name, default_scheme)
            extra_config[lm_head_name] = {k: lm_head_cfg.get(k) for k in scheme_keys if lm_head_cfg.get(k) is not None}

    if extra_config:
        qconfig["extra_config"] = extra_config

    return qconfig


def _layer_config_has_mxfp(layer_config: dict | None) -> bool:
    """Return True if any layer_config entry requests MXFP quantization.

    Handles values that are plain strings (preset names), dicts (possibly
    with a ``"scheme"`` key or a ``"data_type"`` key), or
    :class:`QuantizationScheme` instances.
    """
    if not layer_config:
        return False
    for val in layer_config.values():
        if isinstance(val, str):
            try:
                s = _normalize_scheme(val.upper())
                if is_mx_fp((s.data_type or "").lower()):
                    return True
            except Exception:
                pass
        elif isinstance(val, dict):
            # Direct data_type field
            if is_mx_fp((val.get("data_type") or "").lower()):
                return True
            # Nested 'scheme' key (e.g. {"scheme": "MXFP4"})
            scheme_val = val.get("scheme")
            if isinstance(scheme_val, str):
                try:
                    s = _normalize_scheme(scheme_val.upper())
                    if is_mx_fp((s.data_type or "").lower()):
                        return True
                except Exception:
                    pass
        elif isinstance(val, QuantizationScheme):
            if is_mx_fp((val.data_type or "").lower()):
                return True
    return False


def _layer_config_has_nvfp4(layer_config: dict | None) -> bool:
    """Return True if any layer_config entry requests NVFP4 quantization.

    Detects both the standard NVFP4 (``data_type='nv_fp'``) and the
    global-scale-free variant NVFP4_E5M3 (``data_type='nvfp4_v2'``).

    Handles values that are plain strings (preset names), dicts (possibly
    with a ``"scheme"`` key or a ``"data_type"`` key), or
    :class:`QuantizationScheme` instances.
    """
    if not layer_config:
        return False
    for val in layer_config.values():
        if isinstance(val, str):
            try:
                s = _normalize_scheme(val.upper())
                dt = (s.data_type or "").lower()
                if dt == _NVFP4_E5M3_DATA_TYPE or is_nv_fp(dt):
                    return True
            except Exception:
                pass
        elif isinstance(val, dict):
            dt = (val.get("data_type") or "").lower()
            if dt == _NVFP4_E5M3_DATA_TYPE or is_nv_fp(dt):
                return True
            scheme_val = val.get("scheme")
            if isinstance(scheme_val, str):
                try:
                    s = _normalize_scheme(scheme_val.upper())
                    dt = (s.data_type or "").lower()
                    if dt == _NVFP4_E5M3_DATA_TYPE or is_nv_fp(dt):
                        return True
                except Exception:
                    pass
        elif isinstance(val, QuantizationScheme):
            dt = (val.data_type or "").lower()
            if dt == _NVFP4_E5M3_DATA_TYPE or is_nv_fp(dt):
                return True
    return False


def _get_layer_config_nvfp4_dt(layer_config: dict | None) -> str | None:
    """Return the first NVFP4 data_type found in *layer_config*.

    Returns ``"nv_fp"`` for standard NVFP4, ``"nvfp4_v2"`` for NVFP4_E5M3,
    or ``None`` when no NVFP4 entry is present.
    """
    if not layer_config:
        return None
    for val in layer_config.values():
        if isinstance(val, str):
            try:
                s = _normalize_scheme(val.upper())
                dt = (s.data_type or "").lower()
                if dt == _NVFP4_E5M3_DATA_TYPE:
                    return _NVFP4_E5M3_DATA_TYPE
                if is_nv_fp(dt):
                    return "nv_fp"
            except Exception:
                pass
        elif isinstance(val, dict):
            dt = (val.get("data_type") or "").lower()
            if dt == _NVFP4_E5M3_DATA_TYPE:
                return _NVFP4_E5M3_DATA_TYPE
            if is_nv_fp(dt):
                return "nv_fp"
            scheme_val = val.get("scheme")
            if isinstance(scheme_val, str):
                try:
                    s = _normalize_scheme(scheme_val.upper())
                    dt = (s.data_type or "").lower()
                    if dt == _NVFP4_E5M3_DATA_TYPE:
                        return _NVFP4_E5M3_DATA_TYPE
                    if is_nv_fp(dt):
                        return "nv_fp"
                except Exception:
                    pass
        elif isinstance(val, QuantizationScheme):
            dt = (val.data_type or "").lower()
            if dt == _NVFP4_E5M3_DATA_TYPE:
                return _NVFP4_E5M3_DATA_TYPE
            if is_nv_fp(dt):
                return "nv_fp"
    return None


def _is_full_precision_default(scheme_input: Any) -> bool:
    """Return True when *scheme_input* represents a full-precision default (bits >= 16).

    Used to detect BF16/FP16 schemes where no weights are quantized by
    default but layer_config overrides are still applied.
    """
    try:
        s = _normalize_scheme(scheme_input)
        return (s.bits or 0) >= 16 and (s.act_bits or 16) >= 16
    except Exception:
        return False


def _derive_dominant_int_scheme(
    quantized_layers: list[str],
    layer_config: dict,
    fallback: dict,
) -> dict | None:
    """Infer the dominant INT scheme from layers that were actually quantized.

    Used when the default scheme is full-precision (BF16/FP16) to produce a
    quantization_config where the most common INT scheme becomes the top-level
    default and the BF16 layers appear as ``bits=16`` exceptions in
    ``extra_config``.  This matches the layout expected by AutoRound loaders
    (dominant quantized bits at the top, full-precision overrides below).

    Returns ``None`` when no INT layers were found in *quantized_layers*.
    """
    from collections import Counter

    temp_matcher = _PatternMatcher(
        ignore_patterns=[],
        layer_config=layer_config,
        default_scheme=fallback,
    )
    counter: "Counter[tuple]" = Counter()
    for layer in quantized_layers:
        scheme = temp_matcher.resolve_scheme(f"{layer}.weight")
        if scheme is None:
            continue
        bits = scheme.get("bits")
        if bits is None or bits >= 16:
            continue
        data_type = (scheme.get("data_type") or "int").lower()
        if is_mx_fp(data_type):
            continue  # MXFP is handled by the dedicated path
        key = (bits, scheme.get("group_size") or fallback.get("group_size"), bool(scheme.get("sym", True)), data_type)
        counter[key] += 1

    if not counter:
        return None

    (bits, group_size, sym, data_type), _ = counter.most_common(1)[0]
    return {
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "data_type": data_type,
    }


def _build_quantization_config(
    default_scheme: dict,
    layer_config: dict,
    ignore_patterns: list[str],
    quantized_layers: list[str],
    ignored_layers: list[str],
    block_name_to_quantize: Optional[str] = None,
    format: str = "auto_round",
) -> dict:
    """Build a quantization_config dict compatible with auto-round format."""
    # MXFP (mx_fp) supports two output styles depending on *format*:
    #   - "llm_compressor" → compressed-tensors / llm-compressor style config
    #     (produced by _build_mxfp_quantization_config).
    #   - "auto_round" / "auto_round:auto_gptq" → auto-round style config
    #     (produced by _build_mxfp_autoround_quantization_config).
    # Also route to MXFP config when the default is full-precision (BF16/FP16)
    # but layer_config contains MXFP overrides.
    data_type = (default_scheme.get("data_type") or "int").lower()
    default_bits = default_scheme.get("bits", 4)
    is_fp_default = (default_bits or 0) >= 16 and not is_mx_fp(data_type)
    if data_type == _NVFP4_E5M3_DATA_TYPE and format == "llm_compressor":
        warnings.warn(
            "LLMC/llm-compressor does not currently support the NVFP4_E5M3 scheme. "
            "Please refer to docs/nvfp4_e5m3.md for the recommended export and usage path.",
            UserWarning,
            stacklevel=2,
        )
        return _build_nvfp4_e5m3_quantization_config(ignored_layers)
    # BF16/FP16 default with NVFP4 layer_config overrides and llm_compressor format:
    # build an NVFP4 config that targets only the explicitly quantized layers.
    if is_fp_default and _layer_config_has_nvfp4(layer_config) and format == "llm_compressor":
        nvfp4_dt = _get_layer_config_nvfp4_dt(layer_config)
        targets = list(dict.fromkeys(quantized_layers)) if quantized_layers else ["Linear"]
        if quantized_layers:
            targets = _add_routed_experts_if_moe(targets, quantized_layers)
        if nvfp4_dt == _NVFP4_E5M3_DATA_TYPE:
            warnings.warn(
                "LLMC/llm-compressor does not currently support the NVFP4_E5M3 scheme. "
                "Please refer to docs/nvfp4_e5m3.md for the recommended export and usage path.",
                UserWarning,
                stacklevel=2,
            )
            # Global-scale-free variant (NVFP4_E5M3)
            from auto_round.export.export_to_llmcompressor.config import initialize_nvfp4_e5m3_quantization

            qconfig = initialize_nvfp4_e5m3_quantization(ignore=[])
            qconfig["config_groups"]["group_0"]["targets"] = targets
            qconfig["format"] = "nvfp4-e5m3-pack-quantized"
        else:
            # Standard NVFP4 (nv_fp, scale_dtype=float8_e4m3fn)
            from auto_round.export.export_to_llmcompressor.config import initialize_quantization

            qconfig_obj = initialize_quantization(scheme="NVFP4", ignore=[])
            qconfig = qconfig_obj.to_dict()
            qconfig["config_groups"]["group_0"]["targets"] = targets
            qconfig["format"] = "nvfp4-pack-quantized"
        qconfig.update(_get_llm_compressor_metadata())
        return qconfig
    if is_mx_fp(data_type) or (is_fp_default and _layer_config_has_mxfp(layer_config)):
        if format in ("auto_round", "auto_round:auto_gptq"):
            return _build_mxfp_autoround_quantization_config(
                default_scheme=default_scheme,
                quantized_layers=quantized_layers,
                ignored_layers=ignored_layers,
                layer_config=layer_config,
                block_name_to_quantize=block_name_to_quantize,
            )
        return _build_mxfp_quantization_config(
            default_scheme=default_scheme,
            quantized_layers=quantized_layers,
            ignored_layers=ignored_layers,
            layer_config=layer_config,
        )

    # When the default is full-precision (BF16/FP16) but INT layers were
    # quantized via layer_config, derive the dominant INT scheme and use it as
    # the top-level defaults so the output config reads as
    # "INT-quantized with BF16 exceptions" rather than the inverse.
    if is_fp_default and quantized_layers:
        dominant = _derive_dominant_int_scheme(quantized_layers, layer_config, default_scheme)
        if dominant is not None:
            default_scheme = dominant

    from auto_round.version import __version__

    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    # vllm only support auto_round:auto_gptq, but transformers cannot load it correctly when sym=False.
    # So we keep auto_round for asymmetric quantization to maintain compatibility with both.
    if data_type == _NVFP4_E5M3_DATA_TYPE:
        packing_format = "auto_round:fake" if format == "fake" else "auto_round:llm_compressor_nvfp4_e5m3"
    else:
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

    if data_type == _NVFP4_E5M3_DATA_TYPE:
        for act_key in ("act_bits", "act_data_type", "act_group_size", "act_sym", "act_dynamic"):
            value = default_scheme.get(act_key)
            if value is not None:
                qconfig[act_key] = value

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
            if data_type == _NVFP4_E5M3_DATA_TYPE:
                extra_config[layer_name].update({"act_bits": 16, "act_data_type": "float"})

    quantized_layer_set = set(quantized_layers)
    for lm_head_name in _LM_HEAD_PATTERNS:
        if lm_head_name in quantized_layer_set and lm_head_name not in extra_config:
            lm_head_cfg = layer_config.get(lm_head_name, default_scheme)
            extra_config[lm_head_name] = {k: lm_head_cfg.get(k) for k in scheme_keys if lm_head_cfg.get(k) is not None}

    if extra_config:
        qconfig["extra_config"] = extra_config

    return qconfig


# ---------------------------------------------------------------------------
# Scheme Validation and Override Resolution
# ---------------------------------------------------------------------------


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

    # Full-precision (BF16/FP16) default: all layers stay in full precision
    # unless layer_config provides lower-bit overrides.
    if (bits or 0) >= 16 and act_bits >= 16:
        return

    # MXFP weight-only path: accept mx_fp data type with bits in {4, 8}.
    # Activation quantization for MXFP is dynamic at inference time, so the
    # weight-only RTN path here is independent of act_bits.
    if is_mx_fp(data_type):
        if scheme_obj.act_data_type not in (None, "mx_fp"):
            raise ValueError(
                "Model-free MXFP supports only act_data_type='mx_fp', " f"but got '{scheme_obj.act_data_type}'."
            )
        # Restrict to the two explicitly supported MXFP presets when a string
        # name is provided.  Variants such as MXFP4_RCEIL / MXFP8_RCEIL use a
        # different activation format; silently mapping them to "MXFP4" /
        # "MXFP8" in the output config would misrepresent the requested scheme.
        if isinstance(scheme_input, str) and scheme_input.upper() not in ("MXFP4", "MXFP8"):
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

    if data_type == _NVFP4_E5M3_DATA_TYPE:
        if bits != 4 or scheme_obj.group_size != 16 or act_bits != 4:
            raise ValueError(
                f"Model-free NVFP4_E5M3 requires bits=4, group_size=16, and act_bits=4, "
                f"but '{scheme_input}' requests bits={bits}, group_size={scheme_obj.group_size}, "
                f"act_bits={act_bits}."
            )
        if (scheme_obj.act_data_type or "").lower() != _NVFP4_E5M3_DATA_TYPE or scheme_obj.act_group_size != 16:
            raise ValueError(
                f"Model-free NVFP4_E5M3 requires act_data_type='nvfp4_v2' and act_group_size=16, "
                f"but '{scheme_input}' requests act_data_type='{scheme_obj.act_data_type}', "
                f"act_group_size={scheme_obj.act_group_size}."
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
    """Validate AutoScheme options for model-free packing.

    Returns the single quantized data-type family shared by all options
    (``"int"`` or ``"mx_fp"``). ``BF16`` (or any >=16-bit no-op scheme) is
    allowed and treated as "keep layer in full precision" during conversion,
    so it does not contribute to the returned family.

    Raises ``ValueError`` when any non-BF16 option is unsupported or when INT
    and MXFP options are mixed (they use different packing formats and cannot
    be produced in one model-free run).
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

        # AutoScheme may include BF16/no-op options. In model-free flow these
        # layers become ignore_layers entries (full precision), so they are
        # allowed and excluded from family checks.
        if scheme_obj is not None:
            act_bits = scheme_obj.act_bits if scheme_obj.act_bits is not None else 16
            if (scheme_obj.bits or 0) >= 16 and act_bits >= 16:
                continue

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
    preferred_base_scheme: Union[str, QuantizationScheme, None] = None,
) -> tuple[QuantizationScheme, dict[str, dict], list[str]]:
    """Convert an AutoScheme-generated ``layer_config`` into model-free inputs.

    Returns ``(base_scheme, per_layer_overrides, fp16_layers)`` where:

        * ``base_scheme`` is ``preferred_base_scheme`` when provided, matching the
            primary scheme selected by the regular AutoRound path. Otherwise the
            most common quantized scheme is used as a fallback.
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
        data_type_raw = (cfg.get("data_type") or "").strip()

        # Infer bits from compact dtype aliases that embed bit-width in the name,
        # e.g. "mxfp8" → 8, "MXFP4" → 4, "mx_fp4" → 4.
        if bits is None and data_type_raw:
            dt_lower = data_type_raw.lower()
            for prefix in ("mxfp", "mx_fp", "nvfp", "nv_fp"):
                tail = dt_lower[len(prefix) :]
                if dt_lower.startswith(prefix) and tail.isdigit():
                    bits = int(tail)
                    break

        if bits is None:
            continue

        clean = {k: cfg[k] for k in scheme_keys if cfg.get(k) is not None}
        clean["bits"] = bits  # honour inferred value when not explicit

        # Normalise compact dtype aliases to canonical forms understood by
        # the quantization kernels ("mxfp8" / "MXFP4" → "mx_fp").
        if data_type_raw:
            dt_lower = data_type_raw.lower()
            if dt_lower.startswith("mxfp") or dt_lower.startswith("mx_fp"):
                clean["data_type"] = "mx_fp"
            elif dt_lower.startswith("nvfp") or dt_lower.startswith("nv_fp"):
                clean["data_type"] = "nv_fp"

        if bits >= 16:
            fp16_layers.append(name)
            continue
        data_type = (clean.get("data_type") or "int").lower()
        per_layer[name] = clean
        counter[(clean.get("bits"), clean.get("group_size"), bool(clean.get("sym", True)), data_type)] += 1

    if not counter:
        raise ValueError("AutoScheme did not assign any quantizable layers for model-free mode.")

    if preferred_base_scheme is not None:
        base_scheme = copy.deepcopy(_normalize_scheme(preferred_base_scheme))
    else:
        (base_bits, base_group_size, base_sym, base_dtype), _ = counter.most_common(1)[0]
        base_scheme = QuantizationScheme(
            bits=base_bits,
            group_size=base_group_size,
            sym=base_sym,
            data_type=base_dtype,
        )
    return base_scheme, per_layer, fp16_layers


# ---------------------------------------------------------------------------
# Model-Special Handlers (Keep at End)
# ---------------------------------------------------------------------------


def _read_int_like(value: Any) -> int | None:
    """Best-effort conversion for int-like config values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return int(v)
    return None


def _read_bool_like(value: Any) -> bool | None:
    """Best-effort conversion for bool-like config values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return None


def _resolve_kimi_k25_int4_params(source_quant_config: dict | None) -> tuple[int | None, bool]:
    """Resolve default (group_size, sym) for kimi_k25 INT4 source dequant."""
    group_size: int | None = None
    sym = True

    if not isinstance(source_quant_config, dict):
        return group_size, sym

    candidates = [source_quant_config]
    weights_cfg = source_quant_config.get("weights")
    if isinstance(weights_cfg, dict):
        candidates.append(weights_cfg)

    for cfg in candidates:
        if group_size is None:
            for key in ("group_size", "weight_group_size"):
                maybe = _read_int_like(cfg.get(key))
                if maybe is not None and maybe > 0:
                    group_size = maybe
                    break

        sym_val = _read_bool_like(cfg.get("sym"))
        if sym_val is not None:
            sym = sym_val

        zp_val = _read_bool_like(cfg.get("zero_point"))
        if zp_val is not None:
            sym = not zp_val

    return group_size, sym


def _collect_kimi_k25_int4_source_entries(
    raw_tensors: dict[str, torch.Tensor],
) -> list[tuple[str, str, str, str | None]]:
    """Collect kimi_k25 INT4 source tensors."""
    entries: list[tuple[str, str, str, str | None]] = []
    for name, packed in raw_tensors.items():
        if not name.endswith(".weight_packed"):
            continue
        if packed.dtype not in (torch.uint8, torch.int8, torch.int32):
            continue

        layer_name = name[: -len(".weight_packed")]
        scale_key = f"{layer_name}.weight_scale"
        if scale_key not in raw_tensors:
            continue

        scale = raw_tensors[scale_key]
        if scale.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            continue
        if packed.dim() != 2 or scale.dim() != 2:
            continue

        zp_key = None
        for candidate in (f"{layer_name}.weight_zero_point", f"{layer_name}.zero_point"):
            if candidate in raw_tensors:
                zp_key = candidate
                break

        entries.append((layer_name, name, scale_key, zp_key))

    return entries


def _unpack_int4_weight_packed(packed: torch.Tensor) -> torch.Tensor:
    """Unpack packed int4 bytes/ints into int16 ``[out, in]`` nibbles."""
    if packed.dtype == torch.int32:
        packed_i32 = packed.view(torch.int32)
        rows, cols = packed_i32.shape
        nibble_list = [(packed_i32 >> (4 * i)) & 0xF for i in range(8)]
        out = torch.stack(nibble_list, dim=-1).reshape(rows, cols * 8).to(torch.int16)
        return out

    packed_u8 = packed.view(torch.uint8)
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F

    out = torch.empty(
        (packed_u8.shape[0], packed_u8.shape[1] * 2),
        dtype=torch.int16,
        device=packed_u8.device,
    )
    out[:, 0::2] = lo.to(torch.int16)
    out[:, 1::2] = hi.to(torch.int16)
    return out


def _dequant_kimi_k25_int4_tensors(
    raw_tensors: dict[str, torch.Tensor],
    source_quant_config: dict | None = None,
    device: str = "cpu",
    shard_name: str | None = None,
    dequantize_with_device_fallback: Callable[..., torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Dequantize kimi_k25-style INT4 packed weights to bfloat16 ``.weight``."""
    if dequantize_with_device_fallback is None:
        raise ValueError("dequantize_with_device_fallback callback is required")

    entries = _collect_kimi_k25_int4_source_entries(raw_tensors)
    if not entries:
        return raw_tensors

    default_group_size, default_sym = _resolve_kimi_k25_int4_params(source_quant_config)
    dequant_device = str(device or "cpu")
    shard_prefix = f"[{shard_name}] " if shard_name else ""
    logger.info(
        f"{shard_prefix}Dequantizing kimi_k25 INT4 tensor(s) to bfloat16 on {dequant_device}: total={len(entries)}."
    )

    for layer_name, packed_key, scale_key, zp_key in entries:
        packed = raw_tensors.pop(packed_key)
        scale = raw_tensors.pop(scale_key)
        zero = raw_tensors.pop(zp_key) if zp_key is not None else None
        raw_tensors.pop(f"{layer_name}.weight_shape", None)

        nibbles_per_element = 8 if packed.dtype == torch.int32 else 2
        in_features = packed.shape[1] * nibbles_per_element
        groups = scale.shape[1]
        inferred_group_size = (in_features // groups) if groups > 0 and in_features % groups == 0 else None
        group_size = inferred_group_size or default_group_size
        if group_size is None or group_size <= 0:
            raise ValueError(
                f"Cannot resolve group_size for INT4 tensor '{layer_name}': "
                f"packed_shape={tuple(packed.shape)}, scale_shape={tuple(scale.shape)}"
            )
        if in_features % group_size != 0:
            raise ValueError(
                f"Invalid group_size={group_size} for INT4 tensor '{layer_name}': "
                f"in_features={in_features} is not divisible by group_size."
            )

        if (
            inferred_group_size is not None
            and default_group_size is not None
            and inferred_group_size != default_group_size
        ):
            logger.warning(
                f"{shard_prefix}k25 INT4 group_size mismatch for {layer_name}: "
                f"config={default_group_size}, inferred={inferred_group_size}. "
                "Using inferred value from tensor shapes."
            )

        n_groups = in_features // group_size
        if scale.shape[1] != n_groups:
            raise ValueError(
                f"Scale shape mismatch for INT4 tensor '{layer_name}': "
                f"scale.shape={tuple(scale.shape)}, expected second dim {n_groups}."
            )

        def _dequant_impl(target_device: str) -> torch.Tensor:
            q = _unpack_int4_weight_packed(packed.to(target_device, non_blocking=True))
            sc = scale.to(target_device, non_blocking=True).to(torch.float32)

            if zero is not None:
                zp = zero.to(target_device, non_blocking=True)
                if zp.dim() == 2 and zp.shape == packed.shape and zp.dtype in (torch.int8, torch.uint8):
                    zp = _unpack_int4_weight_packed(zp)
                elif zp.dim() == 2 and zp.shape[1] != n_groups:
                    raise ValueError(f"Unsupported zero-point shape for INT4 tensor '{layer_name}': {tuple(zp.shape)}")
                if zp.dim() == 2 and zp.shape[1] == n_groups:
                    zp = zp.to(torch.int16).repeat_interleave(group_size, dim=1)
                zp = zp[:, : q.shape[1]].to(torch.int16)
                q = q - zp
            else:
                if default_sym:
                    q = torch.where(q >= 8, q - 16, q)
                else:
                    q = q - 8

            sc = sc.repeat_interleave(group_size, dim=1)[:, : q.shape[1]]
            return (q.to(torch.float32) * sc).to(torch.bfloat16)

        dq_weight = dequantize_with_device_fallback(
            dequant_device=dequant_device,
            shard_prefix=shard_prefix,
            op_name="kimi_k25 INT4 dequant",
            tensor_label=layer_name,
            on_device=lambda: _dequant_impl(dequant_device).to("cpu"),
            on_cpu=lambda: _dequant_impl("cpu"),
        )
        raw_tensors[f"{layer_name}.weight"] = dq_weight

    return raw_tensors


def _expand_e8m0_block_scale(
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int = 32,
) -> torch.Tensor:
    """Expand a coarse 2D E8M0 block scale to per-group llm-compressor layout."""
    scale = scale.view(torch.uint8)
    if scale.dim() != 2:
        raise ValueError(f"Expected a 2D E8M0 block scale, got shape {tuple(scale.shape)}.")

    target_rows = out_features
    target_cols = in_features // group_size
    rows, cols = scale.shape

    if target_rows % rows == 0 and target_cols % cols == 0:
        if target_rows != rows:
            scale = scale.repeat_interleave(target_rows // rows, dim=0)
        if target_cols != cols:
            scale = scale.repeat_interleave(target_cols // cols, dim=1)
        return scale.contiguous()

    coarse_block = 128
    expected_rows = (out_features + coarse_block - 1) // coarse_block
    expected_cols = (in_features + coarse_block - 1) // coarse_block
    if rows == expected_rows and cols == expected_cols:
        if coarse_block % group_size != 0:
            raise ValueError(
                f"Cannot expand DeepSeek E8M0 block scale with group_size={group_size}; "
                f"{coarse_block} is not divisible by group_size."
            )
        groups_per_block_col = coarse_block // group_size
        scale = scale.repeat_interleave(coarse_block, dim=0)[:target_rows]
        scale = scale.repeat_interleave(groups_per_block_col, dim=1)[:, :target_cols]
        return scale.contiguous()

    raise ValueError(
        f"Cannot expand E8M0 block scale {tuple(scale.shape)} to "
        f"({target_rows}, {target_cols}); unsupported coarse/block layout."
    )


def preprocess_model_type_source_tensors(
    raw_tensors: dict[str, torch.Tensor],
    model_type: str | None,
    group_size: int = 32,
    quantization_config: dict | None = None,
    shard_name: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Apply model-type-specific source tensor normalization."""
    model_type = (model_type or "").lower()
    quantization_config = quantization_config or {}
    is_deepseek_v4 = model_type == "deepseek_v4"
    is_deepseek_v32_ue8m0 = (
        model_type == "deepseek_v32"
        and quantization_config.get("quant_method") == "fp8"
        and str(quantization_config.get("fmt", "")).lower() == "e4m3"
        and str(quantization_config.get("scale_fmt", "")).lower() == "ue8m0"
    )
    if not is_deepseek_v4 and not is_deepseek_v32_ue8m0:
        return raw_tensors, {}

    entries: list[tuple[str, str, bool]] = []
    for name, tensor in raw_tensors.items():
        if not name.endswith(".weight"):
            continue
        layer_name = name[: -len(".weight")]
        scale_candidates = [f"{layer_name}.scale"]
        if is_deepseek_v32_ue8m0:
            scale_candidates.extend((f"{layer_name}.weight_scale", f"{layer_name}.weight_scale_inv"))
        scale_name = next((candidate for candidate in scale_candidates if candidate in raw_tensors), None)
        if scale_name is None:
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

            if scale.dtype == torch.float32:
                shard_prefix = f"[{shard_name}] " if shard_name else ""
                logger.warning_once(
                    f"[{model_type}] Scale tensor pattern has dtype float32 "
                    f"with UE8M0 encoding (only the 8-bit exponent is significant). "
                    f"Extracting uint8 E8M0 exponent bytes from fp32 representation."
                )
                scale = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
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

    shard_prefix = f"[{shard_name}] " if shard_name else ""
    logger.info(
        f"{shard_prefix}Applied model_type preprocessing for {model_type}: "
        f"{n_fp8} MXFP8 layer(s), {n_fp4} MXFP4 layer(s) converted to llm-compressor naming."
    )
    return raw_tensors, source_state


def handle_model_type_low_precision_source_tensors(
    raw_tensors: dict[str, torch.Tensor],
    model_type: str | None,
    source_quant_config: dict | None = None,
    device: str = "cpu",
    shard_name: str | None = None,
    dequantize_with_device_fallback: Callable[..., torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Handle model-type-specific low-precision source tensors."""
    if dequantize_with_device_fallback is None:
        dequantize_with_device_fallback = _dequantize_with_device_fallback

    if (model_type or "").lower() == "kimi_k25":
        return _dequant_kimi_k25_int4_tensors(
            raw_tensors,
            source_quant_config=source_quant_config,
            device=device,
            shard_name=shard_name,
            dequantize_with_device_fallback=dequantize_with_device_fallback,
        )
    return raw_tensors
