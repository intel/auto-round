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
"""Algorithm alias → config class registry.

This is the **only** place in the codebase where algorithm name strings are
mapped to algorithm/transform config objects.  ``auto_round/compressors/entry.py``
calls :func:`resolve_alg_config` to convert user-supplied string aliases into
concrete config instances; the rest of the compressor stack operates only on
config objects, never on raw strings.

Design invariant (AWQ_REFACTOR_PLAN.md §0.0 Rule 1):
    Algorithm-name strings must not appear outside this module and the CLI
    argument definitions.  ``compressors/`` code may not contain ``algorithm ==
    "awq"``-style branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from auto_round.algorithms.alg_config import AlgConfig

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Maps lower-cased alias → zero-argument factory that returns a config object.
# Using a factory (callable) rather than a class reference lets us defer
# imports and provide sensible defaults without importing every submodule
# at module-load time.
_REGISTRY: dict[str, Callable[[], "AlgConfig"]] = {}


def register_alg(alias: str, factory: Callable[[], "AlgConfig"]) -> None:
    """Register an algorithm alias with the given config factory.

    Args:
        alias:   Lower-cased string alias (e.g. ``"awq"``, ``"rtn"``).
        factory: Zero-argument callable that returns a fresh ``AlgConfig``
                 instance with default parameters.
    """
    _REGISTRY[alias.lower()] = factory


def resolve_alg_config(alias: str) -> "AlgConfig":
    """Resolve a string alias to a fresh ``AlgConfig`` instance.

    Args:
        alias: Case-insensitive algorithm name (e.g. ``"awq"``, ``"rtn"``).

    Returns:
        A fresh ``AlgConfig`` instance produced by the registered factory.

    Raises:
        ValueError: If *alias* is not registered.
    """
    key = alias.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown algorithm alias '{alias}'. "
            f"Supported aliases: {sorted(_REGISTRY.keys())}. "
            "If you are adding a new algorithm, register it via "
            "auto_round.algorithms.quantization.registry.register_alg()."
        )
    return _REGISTRY[key]()


def list_registered_algorithms() -> list[str]:
    """Return all registered algorithm aliases."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------
# These are the built-in algorithm aliases that ship with auto-round.
# Additional algorithms (AWQ, SmoothQuant, …) register themselves when their
# subpackage is imported (see each algorithm's ``__init__.py``).


def _register_builtins() -> None:
    from auto_round.algorithms.quantization.awq.config import AWQConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.rotation.config import RotationConfig

    register_alg("rtn", RTNConfig)
    register_alg("sign_round", SignRoundConfig)
    register_alg("signround", SignRoundConfig)
    register_alg("auto_round", SignRoundConfig)
    register_alg("autoround", SignRoundConfig)
    register_alg("awq", AWQConfig)
    register_alg("hadamard", RotationConfig)
    register_alg("random_hadamard", lambda: RotationConfig(hadamard_type="random_hadamard"))
    register_alg("quarot_hadamard", lambda: RotationConfig(hadamard_type="quarot_hadamard"))


_register_builtins()
