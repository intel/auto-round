# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Data contracts for recommendation definitions and resolved plans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class _Unset:
    def __repr__(self):
        return "UNSET"


UNSET = _Unset()


def calibration_defaults() -> dict[str, Any]:
    return {"batch_size": 8, "seqlen": 2048, "nsamples": 128}


@dataclass(frozen=True)
class RecipeContext:
    """Available metadata; resolving a recipe never loads a model."""

    scheme: Any = "W4A16"
    model_type: str | None = None
    model_config: Any = None


@dataclass(frozen=True)
class AlgorithmSpec:
    """A stable stage name, registered algorithm name, and constructor parameters."""

    name: str
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)

    def build_config(self, overrides: dict[str, Any] | None = None):
        """Apply constructor overrides while preserving callable-factory defaults."""
        import copy
        import inspect

        from auto_round.algorithms.registry import get_algorithm_entry, resolve_alg_config

        entry = get_algorithm_entry(self.algorithm)
        factory = entry.alias_factories.get(self.algorithm.strip().lower(), entry.config_factory)
        params = copy.deepcopy({**self.params, **(overrides or {})})
        if isinstance(factory, type):
            return factory(**params)
        config = resolve_alg_config(self.algorithm)
        if not params:
            return config
        fields = set(getattr(type(config), "model_fields", {}))
        for cls in type(config).__mro__:
            init = cls.__dict__.get("__init__")
            if init is None:
                continue
            fields.update(
                name
                for name, parameter in inspect.signature(init).parameters.items()
                if name != "self"
                and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            )
        defaults = {name: copy.deepcopy(getattr(config, name)) for name in fields if hasattr(config, name)}
        result = type(config)(**{**defaults, **params})
        if hasattr(config, "_user_set_scheme_fields"):
            result._user_set_scheme_fields.update(config._user_set_scheme_fields)
        return result


@dataclass(frozen=True)
class RecipeOverlay:
    name: str
    matches: Callable[[RecipeContext], bool]
    pipeline: tuple[AlgorithmSpec, ...] | None = None
    stage_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    layer_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recipe:
    name: str
    pipeline: tuple[AlgorithmSpec, ...]
    calibration: dict[str, Any] = field(default_factory=dict)
    layer_config: dict[str, Any] = field(default_factory=dict)
    overlays: tuple[RecipeOverlay, ...] = ()
    description: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedRecipe:
    name: str
    pipeline: tuple[AlgorithmSpec, ...]
    calibration: dict[str, Any]
    layer_config: dict[str, Any]
    applied_overlays: tuple[str, ...] = ()

    def build_configs(self) -> list:
        """Construct fresh configs using the existing algorithm registry."""
        return [stage.build_config() for stage in self.pipeline]
