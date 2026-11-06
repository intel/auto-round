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

"""Entry-level scheme resolution layer.

This module sits between the public entry points (``auto_round.AutoRound``,
the CLI) and the core scheme parser (:func:`auto_round.schemes.parse_scheme`).
It owns everything needed to turn user-facing scheme input — a single
``scheme``, a candidate sequence, or the legacy ``options``/``avg_bits``
kwargs — into the canonical scheme object the compressor pipeline consumes
(``str`` / ``dict`` / ``QuantizationScheme`` / ``AutoScheme``), plus the
routing preview/validation helpers the entry uses before a compressor exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from auto_round.logger import logger
from auto_round.schemes import parse_scheme

if TYPE_CHECKING:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme


def preview_resolved_attrs(scheme_overrides=None, scheme=None, format=None) -> dict:
    """Resolve scheme attributes for routing without mutating algorithm configs."""
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    overrides = {key: value for key, value in (scheme_overrides or {}).items() if value is not None}
    if isinstance(scheme, AutoScheme):
        return overrides
    try:
        _, _, final_attrs = parse_scheme(scheme, overrides, format=format)
        return final_attrs
    except Exception as error:
        logger.warning_once(
            "Scheme preview failed (%s: %s); routing falls back to explicit overrides.",
            type(error).__name__,
            error,
        )
        return overrides


def eager_validate_scheme(scheme_overrides=None, scheme=None, format=None) -> None:
    """Validate fixed scheme constraints before loading a model."""
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.schemes import QuantizationScheme

    if isinstance(scheme, AutoScheme):
        return
    overrides = {key: value for key, value in (scheme_overrides or {}).items() if value is not None}
    try:
        _, _, final_attrs = parse_scheme(scheme, overrides, format=format)
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        return
    QuantizationScheme.from_dict(final_attrs).validate()


def is_weight_scheme(scheme: str | dict | object) -> bool:
    if isinstance(scheme, str):
        return scheme.upper().startswith("W")
    if isinstance(scheme, dict):
        return all(isinstance(s, str) and s.upper().startswith("W") for s in scheme.values())
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    if isinstance(scheme, AutoScheme):
        opts = scheme.options
        if isinstance(opts, (list, tuple)):
            return all(isinstance(s, str) and s.upper().startswith("W") for s in opts)
        if isinstance(opts, str):
            return opts.upper().startswith("W")
    return False


def is_gguf_k_target(value: str | AutoScheme | object) -> bool:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized.startswith("gguf:") and "_k" in normalized
    if isinstance(value, AutoScheme):
        opts = value.options
        if isinstance(opts, str):
            opts = [opts]
        if isinstance(opts, (list, tuple)):
            return any(isinstance(opt, str) and is_gguf_k_target(opt) for opt in opts)
    return False


def resolve_entry_scheme(scheme, direct_kwargs):
    """Resolve a fixed scheme, candidate list/tuple, or AutoScheme instance.

    With candidates, ``bits`` is the required average target. With an
    AutoScheme instance, supplied AutoScheme fields override a copy, leaving
    the caller's instance untouched. Fixed schemes keep integer weight bits.
    """
    from dataclasses import replace

    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    if "schemes" in direct_kwargs:
        raise TypeError("`schemes` is not supported; pass candidate schemes through `scheme` instead.")
    legacy_options = direct_kwargs.pop("options", None)
    legacy_avg_bits = direct_kwargs.pop("avg_bits", None)

    if legacy_options is not None:
        if scheme is not None and (not isinstance(scheme, str) or scheme.upper() != "W4A16"):
            raise ValueError("`scheme` and `options` cannot be used together.")
        scheme = legacy_options.split(",") if isinstance(legacy_options, str) else legacy_options
    if legacy_avg_bits is not None:
        if "bits" in direct_kwargs and direct_kwargs["bits"] is not None and direct_kwargs["bits"] != legacy_avg_bits:
            raise ValueError("`bits` and `avg_bits` disagree, please use only `bits`")
        direct_kwargs["bits"] = legacy_avg_bits

    if not isinstance(scheme, (list, tuple, AutoScheme)):
        bits = direct_kwargs.get("bits")
        if bits is not None and not float(bits).is_integer():
            raise ValueError("`bits` must be an integer unless `scheme` is a list, tuple, or AutoScheme")
        if bits is not None:
            direct_kwargs["bits"] = int(bits)
        return scheme, direct_kwargs

    auto_scheme_kwargs = {}
    bits = direct_kwargs.pop("bits", None)
    if bits is not None:
        auto_scheme_kwargs["avg_bits"] = bits
    for key in ("shared_layers", "ignore_scale_zp_bits"):
        if direct_kwargs.get(key) is not None:
            auto_scheme_kwargs[key] = direct_kwargs.pop(key)
    if isinstance(scheme, AutoScheme):
        resolved = replace(scheme, **auto_scheme_kwargs) if auto_scheme_kwargs else scheme
    else:
        if not scheme:
            raise ValueError("`scheme` candidates must not be empty.")
        if bits is None:
            raise ValueError("`bits` is required when `scheme` is a list or tuple (AutoScheme average target bits).")
        resolved = AutoScheme(options=scheme, **auto_scheme_kwargs)
    return resolved, direct_kwargs
