# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from auto_round.algorithms.base import BasePipelineMember


@dataclass
class AlgRegistryEntry:
    name: str
    aliases: tuple[str, ...] = ()
    config_factory: Callable[[], object] | type | None = None
    cli_handler: type | None = None
    summary: str = ""
    alias_factories: dict[str, Callable[[], object]] = field(default_factory=dict)


_ALG_REGISTRY: dict[str, AlgRegistryEntry] = {}
_ALIAS_TO_NAME: dict[str, str] = {}
_CONFIG_IMPL_REGISTRY: dict[type, type["BasePipelineMember"]] = {}
_builtin_algorithms_registered = False
_pipeline_members_registered = False


def _ensure_builtin_algorithms_registered() -> None:
    global _builtin_algorithms_registered
    if _builtin_algorithms_registered:
        return
    importlib.import_module("auto_round.cli.algorithms")
    _builtin_algorithms_registered = True


def _ensure_pipeline_members_registered() -> None:
    global _pipeline_members_registered
    if _pipeline_members_registered:
        return
    for module_name in (
        "auto_round.algorithms.quantization.rtn.quantizer",
        "auto_round.algorithms.quantization.sign_round.quantizer",
        "auto_round.algorithms.quantization.sign_roundv2.quantizer",
        "auto_round.algorithms.quantization.adam_round.adam",
        "auto_round.algorithms.transforms.awq.base",
    ):
        importlib.import_module(module_name)
    _pipeline_members_registered = True


def _register_algorithm_entry(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    config_factory: Callable[[], object] | type | None = None,
    cli_handler: type | None = None,
    summary: str = "",
    alias_factories: dict[str, Callable[[], object]] | None = None,
) -> None:
    key = name.strip().lower()
    entry = _ALG_REGISTRY.get(key)
    if entry is None:
        entry = AlgRegistryEntry(name=key)
        _ALG_REGISTRY[key] = entry

    merged_aliases = tuple(
        dict.fromkeys((entry.aliases or ()) + tuple(a.strip().lower() for a in aliases if a.strip()))
    )
    if config_factory is not None:
        entry.config_factory = config_factory
    if cli_handler is not None:
        entry.cli_handler = cli_handler
    if summary:
        entry.summary = summary
    if alias_factories:
        entry.alias_factories.update({k.strip().lower(): v for k, v in alias_factories.items()})
    entry.aliases = merged_aliases

    _ALIAS_TO_NAME[key] = key
    for alias in merged_aliases:
        _ALIAS_TO_NAME[alias] = key


def resolve_algorithm_alias(alias: str) -> str | None:
    _ensure_builtin_algorithms_registered()
    return _ALIAS_TO_NAME.get(alias.strip().lower())


def get_algorithm_entry(name: str) -> AlgRegistryEntry:
    _ensure_builtin_algorithms_registered()
    canonical = resolve_algorithm_alias(name)
    if canonical is None or canonical not in _ALG_REGISTRY:
        raise KeyError(name)
    return _ALG_REGISTRY[canonical]


def iter_algorithm_entries() -> list[AlgRegistryEntry]:
    _ensure_builtin_algorithms_registered()
    return list(_ALG_REGISTRY.values())


def resolve_alg_config(alias: str) -> object:
    _ensure_builtin_algorithms_registered()
    canonical = resolve_algorithm_alias(alias)
    if canonical is None:
        raise ValueError(
            f"Unknown algorithm alias '{alias}'. Supported aliases: {sorted(_ALIAS_TO_NAME.keys())}. "
            "If you are adding a new algorithm, register its CLI/config entry in auto_round.cli.algorithms."
        )

    entry = _ALG_REGISTRY[canonical]
    factory = entry.alias_factories.get(alias.strip().lower(), entry.config_factory)
    if factory is None:
        raise ValueError(f"Algorithm alias '{alias}' is registered but has no config factory.")
    return factory() if callable(factory) else factory


def list_registered_algorithms() -> list[str]:
    _ensure_builtin_algorithms_registered()
    return sorted(_ALIAS_TO_NAME.keys())


def register_algorithm(config_cls: type):
    def _decorator(member_cls: type["BasePipelineMember"]) -> type["BasePipelineMember"]:
        raw_names = getattr(member_cls, "algorithm_names", None)
        if raw_names is None:
            raw_names = getattr(member_cls, "algorithm_name", None)
        if raw_names is None:
            raise TypeError(
                f"{member_cls.__name__} must define 'algorithm_names' (str or tuple/list[str])."
            )
        if isinstance(raw_names, str):
            names = (raw_names,)
        elif isinstance(raw_names, (tuple, list)) and all(isinstance(n, str) and n.strip() for n in raw_names):
            names = tuple(n.strip() for n in raw_names)
        else:
            raise TypeError(
                f"{member_cls.__name__}.algorithm_names must be str or tuple/list[str], got {type(raw_names).__name__}."
            )
        member_cls.algorithm_names = names
        _CONFIG_IMPL_REGISTRY[config_cls] = member_cls
        return member_cls

    return _decorator


def resolve_quantizer_by_config(config: object) -> type["BasePipelineMember"]:
    _ensure_pipeline_members_registered()
    normalized_config = normalize_algorithm_config(config)
    config_type = type(normalized_config)
    for candidate_cls in inspect.getmro(config_type):
        member_cls = _CONFIG_IMPL_REGISTRY.get(candidate_cls)
        if member_cls is not None:
            return member_cls
    raise ValueError(f"Unknown algorithm config type {config_type.__name__!r}.")


def resolve_quantizer_by_name(name: str) -> type["BasePipelineMember"]:
    config = resolve_alg_config(name)
    return resolve_quantizer_by_config(config)


# Backward-compatible aliases during migration.
register_pipeline_member = register_algorithm
resolve_pipeline_member = resolve_quantizer_by_config


def coerce_config_class(config: object, target_cls: type) -> object:
    if type(config) is target_cls:
        return config
    new_config = copy.copy(config)
    if hasattr(config, "scheme") and getattr(config, "scheme", None) is not None:
        new_config.scheme = config.scheme.copy()
    if hasattr(config, "_user_set_scheme_fields"):
        new_config._user_set_scheme_fields = set(getattr(config, "_user_set_scheme_fields", set()))
    new_config.__class__ = target_cls
    return new_config


def normalize_algorithm_config(config: object) -> object:
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
    from auto_round.algorithms.quantization.sign_round.config import AdamRoundConfig, SignRoundConfig, SignRoundV2Config

    if type(config) is RTNConfig and not getattr(config, "disable_opt_rtn", False):
        return coerce_config_class(config, OptimizedRTNConfig)
    if type(config) is SignRoundConfig:
        if getattr(config, "enable_adam", False):
            return coerce_config_class(config, AdamRoundConfig)
        if getattr(config, "enable_alg_ext", False):
            return coerce_config_class(config, SignRoundV2Config)
    return config
