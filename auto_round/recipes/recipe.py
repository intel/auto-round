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
        import copy

        from auto_round.algorithms.registry import get_algorithm_entry, resolve_alg_config

        configs = []
        for stage in self.pipeline:
            entry = get_algorithm_entry(stage.algorithm)
            factory = entry.alias_factories.get(stage.algorithm.strip().lower(), entry.config_factory)
            if isinstance(factory, type):
                config = factory(**copy.deepcopy(stage.params))
            else:
                config = resolve_alg_config(stage.algorithm)
                if stage.params:
                    config = type(config)(**copy.deepcopy(stage.params))
            configs.append(config)
        return configs
