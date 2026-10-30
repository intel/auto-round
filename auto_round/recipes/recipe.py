# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Recipe definitions and built-in recommendations."""

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


@dataclass
class Recipe:
    """Defaults with an optional process(recipe, *, scheme, model_config) function.

    The function edits an isolated copy using ordinary Python conditionals.
    It may replace the pipeline or modify calibration and layer defaults.
    """

    name: str
    pipeline: tuple[AlgorithmSpec, ...]
    calibration: dict[str, Any] = field(default_factory=dict)
    layer_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    aliases: tuple[str, ...] = ()
    process: Callable[..., None] | None = None

    def build_configs(self) -> list:
        """Construct fresh configs using the existing algorithm registry."""
        return [stage.build_config() for stage in self.pipeline]


def builtin_recipes() -> tuple[Recipe, ...]:
    """Return built-in recommendations; numerical schemes remain in schemes.py."""
    calibration = calibration_defaults()
    return (
        Recipe(
            "ar",
            (AlgorithmSpec("quantizer", "auto_round", {"iters": 200}),),
            dict(calibration),
            description="Balanced quantization cost and accuracy.",
            aliases=("default",),
        ),
        Recipe(
            "ar-light",
            (AlgorithmSpec("quantizer", "auto_round", {"iters": 50, "lr": 5e-3}),),
            dict(calibration),
            description="Lower-cost iterative quantization.",
            aliases=("light",),
        ),
        Recipe(
            "ar-best",
            (AlgorithmSpec("quantizer", "auto_round", {"iters": 1000}),),
            {**calibration, "nsamples": 512},
            description="Accuracy-oriented iterative quantization.",
            aliases=("best",),
        ),
        Recipe(
            "rtn",
            (AlgorithmSpec("quantizer", "rtn", {"disable_opt_rtn": True}),),
            {**calibration, "nsamples": 1},
            description="Plain round-to-nearest quantization.",
        ),
        Recipe(
            "opt-rtn",
            (AlgorithmSpec("quantizer", "rtn", {"disable_opt_rtn": False}),),
            dict(calibration),
            description="Optimized round-to-nearest quantization.",
            aliases=("opt_rtn",),
        ),
        Recipe(
            "awq",
            (AlgorithmSpec("smooth", "awq"), AlgorithmSpec("quantizer", "rtn", {"disable_opt_rtn": True})),
            dict(calibration),
            description="AWQ smoothing followed by plain RTN.",
        ),
    )
