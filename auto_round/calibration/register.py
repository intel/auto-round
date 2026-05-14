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
"""Registry for ``Calibrator`` strategies.

Mirrors the same pattern used elsewhere in the codebase (e.g.
``auto_round/data_type/register.py``).  Add a new calibration strategy by
decorating its class with ``@register_calibrator("my_kind")``.
"""

from typing import Type

from auto_round.calibration.base import Calibrator

CALIBRATORS: dict[str, Type[Calibrator]] = {}


def register_calibrator(name: str):
    """Class decorator: register a ``Calibrator`` subclass under ``name``."""

    def _wrap(cls: Type[Calibrator]) -> Type[Calibrator]:
        if not issubclass(cls, Calibrator):
            raise TypeError(f"{cls.__name__} must subclass auto_round.calibration.base.Calibrator")
        cls.name = name
        if name in CALIBRATORS and CALIBRATORS[name] is not cls:
            raise ValueError(f"Calibrator '{name}' already registered as {CALIBRATORS[name].__name__}")
        CALIBRATORS[name] = cls
        return cls

    return _wrap


def get_calibrator(name: str) -> Type[Calibrator]:
    """Look up a registered calibrator class by name."""
    if name not in CALIBRATORS:
        raise KeyError(f"No calibrator registered under '{name}'. " f"Known: {sorted(CALIBRATORS.keys())}")
    return CALIBRATORS[name]
