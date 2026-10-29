# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pure recommendation resolution, independent of model loading and execution."""

import copy
from dataclasses import replace

from auto_round.recipes.recipe import Recipe, RecipeContext, ResolvedRecipe, calibration_defaults
from auto_round.recipes.registry import get_recipe


def recipe_context(recipe: Recipe, model, scheme, *, trust_remote_code=True) -> RecipeContext:
    """Obtain metadata only for contextual recipes, without materializing weights."""
    if not isinstance(recipe, Recipe):
        raise TypeError("recipe must be a Recipe instance when building context.")
    model_config = getattr(model, "config", None)
    if model_config is None and isinstance(model, str) and recipe.overlays:
        from transformers import AutoConfig

        from auto_round.logger import logger

        try:
            model_config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        except (OSError, ValueError) as error:
            logger.debug("Could not load recipe model metadata for %s: %s", model, error)
    return RecipeContext(
        scheme=scheme.upper() if isinstance(scheme, str) else scheme,
        model_type=getattr(model_config, "model_type", None),
        model_config=model_config,
    )


def merge_layer_defaults(defaults: dict, overrides: dict | None) -> dict:
    """Merge per-pattern fields; scalar scheme entries replace the whole entry."""
    result = copy.deepcopy(defaults)
    for name, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name].update(copy.deepcopy(value))
        else:
            result[name] = copy.deepcopy(value)
    return result


def resolve_recipe(recipe: str | Recipe = "ar", context: RecipeContext | None = None) -> ResolvedRecipe:
    """Resolve ordered overlays into an isolated plan without executing algorithms."""
    definition = get_recipe(recipe) if isinstance(recipe, str) else copy.deepcopy(recipe)
    if not isinstance(definition, Recipe):
        raise TypeError("recipe must be a registered name or Recipe instance.")
    context = context or RecipeContext()
    pipeline = definition.pipeline
    calibration = {**calibration_defaults(), **definition.calibration}
    layer_config = definition.layer_config
    applied = []
    for overlay in definition.overlays:
        if not overlay.matches(context):
            continue
        if overlay.pipeline is not None:
            pipeline = overlay.pipeline
        stage_names = {stage.name for stage in pipeline}
        unknown = set(overlay.stage_params) - stage_names
        if unknown:
            raise ValueError(f"Recipe overlay {overlay.name!r} references missing stages: {sorted(unknown)}")
        pipeline = tuple(
            replace(stage, params={**stage.params, **overlay.stage_params.get(stage.name, {})}) for stage in pipeline
        )
        calibration.update(overlay.calibration)
        layer_config = merge_layer_defaults(layer_config, overlay.layer_config)
        applied.append(overlay.name)
    if not pipeline or len({stage.name for stage in pipeline}) != len(pipeline):
        raise ValueError("Recipe pipeline must have nonempty, unique stage names and at least one stage.")
    if any(not stage.name for stage in pipeline):
        raise ValueError("Recipe stage names must be nonempty.")
    invalid_calibration = set(calibration) - {"dataset", "seqlen", "nsamples", "batch_size"}
    if invalid_calibration:
        raise ValueError(f"Unsupported recipe calibration fields: {sorted(invalid_calibration)}")
    return ResolvedRecipe(definition.name, pipeline, calibration, layer_config, tuple(applied))
