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
``scheme``, multiple ``schemes``, or the legacy ``options``/``avg_bits``
kwargs — into the canonical scheme object the compressor pipeline consumes
(``str`` / ``dict`` / ``QuantizationScheme`` / ``AutoScheme``), plus the
routing preview/validation helpers the entry uses before a compressor exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from auto_round.logger import logger
from auto_round.schemes import parse_scheme

if TYPE_CHECKING:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme


def collect_config_scheme_overrides(config) -> dict:
    """Return the config's explicitly-set scheme fields as a ``{field: value}`` dict.

    These are exactly the per-field overrides layered on top of ``scheme=`` — the
    single mechanism through which ``bits`` / ``act_bits`` / ``data_type`` etc.
    reach the resolved scheme. Fields left as ``None`` are omitted so the scheme's
    own value wins.
    """
    return {k: getattr(config, k) for k in config._scheme_fields if getattr(config, k, None) is not None}


def preview_resolved_attrs(config, scheme=None, format=None) -> dict:
    """Resolve scheme attributes without mutating config, for routing decisions.

    Called in ``AutoRound.__new__`` before the concrete compressor class is
    chosen.  ``SchemeMixin.resolve_scheme()`` will do the authoritative
    resolution later; this is just a lightweight preview so routing logic
    (``enable_imatrix``, ``needs_act_calib``, etc.) can use the correct values
    even when the user specified only ``scheme=`` without explicit bit/dtype args.

    This is the single source of resolved scheme fields for entry-level routing:
    callers read from the returned dict and never re-read raw ``config`` attrs.
    When the scheme cannot be previewed (``AutoScheme``, or a deferred parse
    error), the config's own explicitly-set scheme overrides are returned so the
    values still reflect what the user passed. ``format`` must match the
    authoritative parse so format-scoped policies (e.g. the 8-bit asym rule)
    resolve identically here and never turn a refusal into fabricated defaults.

    Returns:
        dict: resolved scheme attributes (config overrides when preview is skipped).
    """
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    config_overrides = collect_config_scheme_overrides(config)
    if isinstance(scheme, AutoScheme):
        # AutoScheme needs model info — cannot preview; fall back to raw config attrs.
        return config_overrides
    try:
        _, _, final_attrs = parse_scheme(scheme, config_overrides, format=format)
        return final_attrs
    except Exception as e:
        logger.warning_once(
            "Scheme preview failed (%s: %s); routing falls back to the config's explicit overrides.",
            type(e).__name__,
            e,
        )
        return config_overrides


def eager_validate_scheme(config, scheme=None, format=None) -> None:
    """Eagerly validate scheme/config constraints at construction time.

    Mirrors the old-arch ``_check_configs()`` call in ``BaseCompressor.__init__``.
    Raises ``ValueError`` or ``NotImplementedError`` immediately if the scheme
    contains config-only invalid combinations (e.g. tuple group_size with non-fp8
    weight dtype) so that callers get a fast failure rather than a deferred error
    buried inside ``post_init()``.

    ``AutoScheme`` is skipped because it requires model information.
    """
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    if isinstance(scheme, AutoScheme):
        return

    user_overrides = collect_config_scheme_overrides(config)
    try:
        _, _, final_attrs = parse_scheme(scheme, user_overrides, format=format)
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        return  # Other parse errors are deferred to post_init

    import copy

    temp_config = copy.copy(config)
    if hasattr(config, "scheme"):
        temp_config.scheme = config.scheme.copy()
        temp_config._user_set_scheme_fields = set(getattr(config, "_user_set_scheme_fields", set()))
    for key, value in final_attrs.items():
        setattr(temp_config, key, value)
    temp_config.check_config()  # raises ValueError / NotImplementedError if invalid


def is_weight_scheme(scheme: Union[str, dict, object]) -> bool:
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


def is_gguf_k_target(value: Union[str, "AutoScheme", object]) -> bool:
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


def resolve_entry_scheme(scheme, schemes, direct_kwargs):
    """Resolve the unified ``schemes``/``bits`` entry into an ``AutoScheme``.

    ``AutoRound(model, schemes=("W4A16", "W8A16"), bits=4.2)`` is the
    recommended AutoScheme entry: multiple schemes select the candidate
    options and ``bits`` is the average target bits. The legacy ``options``
    and ``avg_bits`` kwargs are still accepted for backward compatibility.

    Returns ``(scheme, direct_kwargs)``; when ``schemes`` is given, ``scheme``
    is replaced by an :class:`AutoScheme` and the consumed ``bits``/
    ``shared_layers``/``ignore_scale_zp_bits`` are removed from
    ``direct_kwargs`` so they no longer flow into the algorithm config.
    """
    legacy_options = direct_kwargs.pop("options", None)
    legacy_avg_bits = direct_kwargs.pop("avg_bits", None)

    if legacy_options is not None:
        logger.warning_once("`options` is deprecated, please use `schemes` instead")
        if schemes is not None:
            raise ValueError("`schemes` and `options` cannot be used together, please use `schemes`")
        schemes = legacy_options
    if legacy_avg_bits is not None:
        logger.warning_once("`avg_bits` is deprecated, please use `bits` instead")
        if "bits" in direct_kwargs and direct_kwargs["bits"] is not None and direct_kwargs["bits"] != legacy_avg_bits:
            raise ValueError("`bits` and `avg_bits` disagree, please use only `bits`")
        direct_kwargs["bits"] = legacy_avg_bits

    if schemes is None:
        # Without schemes, `bits` is a plain weight bit width and must be an integer.
        bits = direct_kwargs.get("bits")
        if bits is not None and not float(bits).is_integer():
            raise ValueError("`bits` must be an integer unless `schemes` is provided (AutoScheme average target bits)")
        if bits is not None:
            direct_kwargs["bits"] = int(bits)
        return scheme, direct_kwargs

    if scheme not in (None, "W4A16"):
        raise ValueError("`scheme` and `schemes` are mutually exclusive, please pass only `schemes` for AutoScheme")
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    auto_scheme_kwargs = {}
    bits = direct_kwargs.pop("bits", None)
    if bits is not None:
        auto_scheme_kwargs["avg_bits"] = bits
    for key in ("shared_layers", "ignore_scale_zp_bits"):
        if direct_kwargs.get(key) is not None:
            auto_scheme_kwargs[key] = direct_kwargs.pop(key)
    return AutoScheme(options=schemes, **auto_scheme_kwargs), direct_kwargs
