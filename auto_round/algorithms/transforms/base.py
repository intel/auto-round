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
"""Base classes and utilities for weight/activation rotation algorithms.

All rotation algorithms (Hadamard, SpinQuant, QuaRot, …) must subclass
``BaseRotation`` and declare a corresponding ``BaseRotationConfig``.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Config base
# ---------------------------------------------------------------------------


@dataclass
class BaseRotationConfig:
    """Minimal base for all rotation algorithm configs.

    Every concrete config subclass should be a ``dataclass`` so it is
    trivially serialisable / comparable.
    """

    #: Human-readable algorithm name, must be unique across all subclasses.
    algorithm: str = "base"


# ---------------------------------------------------------------------------
# Algorithm base
# ---------------------------------------------------------------------------


class BaseRotation(ABC):
    """Unified interface for all weight/activation rotation transforms.

    Concrete subclasses implement :meth:`apply_to_model` for their specific
    mathematical transform (Hadamard rotation, random rotation, …).

    Example
    -------
    >>> from auto_round.algorithms.transforms import apply_rotation
    >>> model = apply_rotation(model, config={"algorithm": "hadamard", ...})
    """

    # Registry populated by subclasses via ``BaseRotation.register``.
    _REGISTRY: dict[str, type["BaseRotation"]] = {}

    def __init__(self, config: BaseRotationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_to_model(
        self,
        model: torch.nn.Module,
        need_calibration: bool = False,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Apply this rotation to *model* and return the (possibly mutated) model.

        Args:
            model: The model to transform.
            need_calibration: When ``True``, monkey-patch training-time wrappers
                (``WrapperLinear``, ``WrapperWALayer``) so the transform is
                re-applied each forward pass during calibration.  When
                ``False``, fuse the transform eagerly into the weight tensor.
            **kwargs: Algorithm-specific extra arguments.

        Returns:
            The transformed model.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, algorithm_name: str):
        """Class decorator to register a ``BaseRotation`` subclass.

        Usage::

            @BaseRotation.register("hadamard")
            class HadamardRotation(BaseRotation):
                ...
        """

        def _decorator(subclass: type[BaseRotation]) -> type[BaseRotation]:
            cls._REGISTRY[algorithm_name] = subclass
            return subclass

        return _decorator

    @classmethod
    def from_config(cls, config: BaseRotationConfig) -> "BaseRotation":
        """Instantiate the correct ``BaseRotation`` subclass for *config*.

        The algorithm is looked up by ``config.algorithm`` in the registry.
        Sub-packages are imported lazily on first access so that optional
        dependencies (e.g. ``pydantic``) are not required unless actually used.
        """
        # Lazy-load all sub-packages to populate the registry.
        _ensure_registry_populated()

        name = getattr(config, "algorithm", None)
        if name not in cls._REGISTRY:
            raise ValueError(f"No rotation algorithm registered under {name!r}. " f"Available: {sorted(cls._REGISTRY)}")
        return cls._REGISTRY[name](config)


# ---------------------------------------------------------------------------
# Scheme compatibility check
# ---------------------------------------------------------------------------

#: Quantization schemes that support (and require) rotation transforms.
ROTATION_SUPPORTED_SCHEMES: list[str] = ["MXFP4"]


def check_supported_schemes(scheme: str) -> None:
    """Raise ``ValueError`` if *scheme* does not support rotation transforms."""
    if scheme not in ROTATION_SUPPORTED_SCHEMES:
        raise ValueError(
            f"Rotation transforms are not supported for scheme {scheme!r}. "
            f"Currently supported schemes: {ROTATION_SUPPORTED_SCHEMES}"
        )


# ---------------------------------------------------------------------------
# Lazy registry population
# ---------------------------------------------------------------------------

_registry_populated = False


def _ensure_registry_populated() -> None:
    """Import all known sub-packages so their ``@BaseRotation.register`` calls run."""
    global _registry_populated
    if _registry_populated:
        return
    # Import each sub-package here.  Add new entries as more algorithms land.
    import importlib

    for sub in ("hadamard",):
        try:
            importlib.import_module(f"auto_round.algorithms.transforms.{sub}")
        except ImportError:
            pass
    _registry_populated = True
