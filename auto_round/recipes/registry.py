# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Discovery with isolated recipe definitions and unambiguous aliases."""

import copy

from auto_round.recipes.presets import builtin_recipes
from auto_round.recipes.recipe import Recipe

_RECIPES: dict[str, Recipe] = {}
_ALIASES: dict[str, str] = {}


def register_recipe(recipe: Recipe) -> None:
    """Register an isolated definition; reject colliding names and aliases."""
    names = (recipe.name, *recipe.aliases)
    keys = [name.strip().lower() for name in names]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Recipe names and aliases must be nonempty and unique.")
    conflicts = set(keys) & _ALIASES.keys()
    if conflicts:
        raise ValueError(f"Recipe names/aliases already registered: {sorted(conflicts)}")
    canonical = keys[0]
    _RECIPES[canonical] = copy.deepcopy(recipe)
    _ALIASES.update({key: canonical for key in keys})


def get_recipe(name: str) -> Recipe:
    """Return an independent definition using a case-insensitive name or alias."""
    canonical = _ALIASES.get(name.strip().lower())
    if canonical is None:
        raise ValueError(f"Unknown recipe {name!r}. Available recipes: {', '.join(list_recipes())}")
    return copy.deepcopy(_RECIPES[canonical])


def list_recipes() -> list[str]:
    """List canonical recipe names in registration order."""
    return list(_RECIPES)


for _recipe in builtin_recipes():
    register_recipe(_recipe)
