# Copyright (c) 2026 Intel Corporation
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

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Iterable

import torch

from auto_round.formats.backends.gguf import GGUFFormat, GGUFLayerPolicy
from auto_round.formats.base import OutputFormat
from auto_round.planning import CompressionIntent, FormatCompatibilityError, FormatResolution, ResolvedScheme
from auto_round.schemes import get_gguf_scheme
from auto_round.utils import SUPPORTED_FORMATS, logger


def _deduplicate(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _normalize_format_names(value: str, bits: int) -> list[str]:
    names = value.lower().replace("q*_", f"q{bits}_").replace(" ", "").split(",")
    return _deduplicate(name for name in names if name)


def _validate_format_combination(names: list[str]) -> None:
    has_gguf = any(name.startswith("gguf") for name in names)
    real_companions = [name for name in names if name != "fake" and not name.startswith("gguf")]
    if has_gguf and real_companions:
        raise FormatCompatibilityError(
            f"GGUF format is not compatible with other formats, but got {names}, please choose only one of them"
        )


def _apply_scheme_format_constraint(names: list[str], scheme: ResolvedScheme) -> list[str]:
    preset_name = scheme.preset_name or ""
    gguf_name = preset_name.lower() if preset_name.lower().startswith("gguf:") else get_gguf_scheme(scheme.value)
    if not gguf_name:
        return names
    if gguf_name.endswith("_mixed"):
        gguf_name = gguf_name.replace("_mixed", "_s")
    if all(name in {"fake", gguf_name} for name in names):
        return names
    if any(name.startswith("gguf") for name in names if name != "fake"):
        logger.warning(
            "scheme %s is GGUF, but format %s specifies a different GGUF type. "
            "The scheme-driven per-layer quantization may differ from the file-level GGUF format type.",
            gguf_name,
            ",".join(names),
        )
        return names
    reset_names = [gguf_name]
    if "fake" in names:
        reset_names.append("fake")
    logger.warning(
        "reset format %s to %s since scheme %s can only be exported to format %s or fake",
        ",".join(names),
        ",".join(reset_names),
        gguf_name,
        gguf_name,
    )
    return reset_names


def _precise_gguf_name(formats: list[OutputFormat]) -> str | None:
    for output_format in formats:
        if not output_format.is_gguf():
            continue
        backend = getattr(output_format, "backend", None)
        backend_name = getattr(backend, "output_format", None)
        precise = backend_name if backend_name and backend_name != "gguf" else output_format.output_format
        if precise and precise != "gguf":
            return precise.lower()
    return None


def resolve_formats(intent: CompressionIntent, scheme: ResolvedScheme, *, model=None) -> FormatResolution:
    """Resolve format selection using caller-isolated state and no exporter capability checks."""
    scheme_value = scheme.value
    names = _normalize_format_names(intent.format or "auto_round", scheme_value.bits)
    _validate_format_combination(names)
    names = _deduplicate(_apply_scheme_format_constraint(names, scheme))

    for name in names:
        if name not in SUPPORTED_FORMATS:
            raise ValueError(f"{name} is not supported, we only support {SUPPORTED_FORMATS}")

    layer_config = {name: copy.deepcopy(dict(config)) for name, config in intent.layer_config.items()}
    quant_block_list = (
        [list(group) for group in intent.quant_block_list] if intent.quant_block_list is not None else None
    )
    context = SimpleNamespace(
        model=model,
        layer_config=layer_config,
        scale_dtype=intent.scale_dtype,
        mllm=intent.mllm,
        iters=intent.iters,
        enable_alg_ext=intent.enable_alg_ext,
        quant_nontext_module=intent.quant_nontext_module,
        quant_block_list=quant_block_list,
        platform=intent.platform,
        is_auto_scheme=intent.is_auto_scheme,
    )

    selected = []
    requested = intent.format or "auto_round"
    for name in names:
        if name.startswith("gguf:"):
            output_format, scheme_value, context.layer_config = GGUFFormat.build(name, scheme_value, context)
        elif name in OutputFormat._format_list:
            output_format = OutputFormat._format_list[name](name, scheme_value, context)
        else:
            raise KeyError(f"Unsupported format {name}, please choose from {SUPPORTED_FORMATS}")

        new_name, scheme_value, context.layer_config, context.quant_block_list = output_format.check_and_reset_format(
            scheme_value, context
        )
        if new_name is not None:
            if new_name in requested:
                continue
            output_format = OutputFormat._format_list[new_name](new_name, scheme_value, context)
        selected.append(output_format)

    if any(output_format.is_gguf() for output_format in selected) and any(
        output_format.is_fake() for output_format in selected
    ):
        selected.sort(key=lambda output_format: 0 if output_format.is_fake() else 1)

    scale_dtype = context.scale_dtype
    if len(selected) == 1 and selected[0].is_gguf() and scale_dtype != torch.float32:
        scale_dtype = torch.float32
        logger.info("change `scale_dtype` to `torch.float32` for gguf format")

    precise_gguf_name = _precise_gguf_name(selected)
    resolved_scheme = ResolvedScheme.from_scheme(
        scheme_value,
        preset_name=precise_gguf_name or scheme.preset_name,
    )
    return FormatResolution(
        formats=tuple(selected),
        scheme=resolved_scheme,
        layer_policy=GGUFLayerPolicy() if precise_gguf_name is not None else None,
        layer_config_patch=context.layer_config,
        scale_dtype=scale_dtype,
        quant_block_list=context.quant_block_list,
    )
