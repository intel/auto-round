# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure recommendation resolution, independent of model loading and execution."""

import copy

from auto_round.recipes.recipe import Recipe, calibration_defaults
from auto_round.recipes.registry import get_recipe


def merge_layer_defaults(defaults: dict, overrides: dict | None) -> dict:
    """Merge per-pattern fields; scalar scheme entries replace the whole entry."""
    result = copy.deepcopy(defaults)
    for name, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name].update(copy.deepcopy(value))
        else:
            result[name] = copy.deepcopy(value)
    return result


def resolve_recipe(
    recipe: str | Recipe = "ar", *, scheme="W4A16", model=None, model_config=None, trust_remote_code=True
) -> Recipe:
    """Apply a recipe's processing function to an isolated copy of its defaults."""
    definition = get_recipe(recipe) if isinstance(recipe, str) else copy.deepcopy(recipe)
    if not isinstance(definition, Recipe):
        raise TypeError("recipe must be a registered name or Recipe instance.")
    definition.calibration = {**calibration_defaults(), **definition.calibration}
    if definition.process is not None:
        if model_config is None:
            model_config = getattr(model, "config", None)
        if model_config is None and isinstance(model, str):
            from transformers import AutoConfig

            from auto_round.logger import logger

            try:
                model_config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
            except (OSError, ValueError) as error:
                logger.debug("Could not load recipe model metadata for %s: %s", model, error)
        definition.process(
            definition,
            scheme=scheme.upper() if isinstance(scheme, str) else scheme,
            model_config=copy.deepcopy(model_config),
        )
    pipeline = definition.pipeline
    calibration = definition.calibration
    if not pipeline or len({stage.name for stage in pipeline}) != len(pipeline):
        raise ValueError("Recipe pipeline must have nonempty, unique stage names and at least one stage.")
    if any(not stage.name for stage in pipeline):
        raise ValueError("Recipe stage names must be nonempty.")
    invalid_calibration = set(calibration) - {"dataset", "seqlen", "nsamples", "batch_size"}
    if invalid_calibration:
        raise ValueError(f"Unsupported recipe calibration fields: {sorted(invalid_calibration)}")
    return definition
