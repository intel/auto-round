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

# Lazy imports to avoid circular dependencies
# Users should import from specific modules instead of this __init__.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor
    from auto_round.compressors.config import (
        DiffusionExtraConfig,
        ExtraConfig,
        MLLMExtraConfig,
        SchemeExtraConfig,
        TuningExtraConfig,
    )
    from auto_round.compressors.data_driven import CalibratedRTNCompressor, DataDrivenCompressor
    from auto_round.compressors.entry import AutoRoundCompatible, AutoRound
    from auto_round.compressors.model_free import ModelFreeCompressor
    from auto_round.compressors.zero_shot import ZeroShotCompressor

__all__ = [
    "AutoRound",
    "BaseCompressor",
    "DataDrivenCompressor",
    "CalibratedRTNCompressor",
    "ZeroShotCompressor",
    "AutoRoundCompatible",
    "ModelFreeCompressor",
    "ExtraConfig",
    "TuningExtraConfig",
    "SchemeExtraConfig",
    "MLLMExtraConfig",
    "DiffusionExtraConfig",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "AutoRound" or name == "AutoRoundCompatible":
        from auto_round.compressors.entry import AutoRound, AutoRoundCompatible

        if name == "AutoRound":
            return AutoRound
        return AutoRoundCompatible
    elif name == "BaseCompressor":
        from auto_round.compressors.base import BaseCompressor

        return BaseCompressor
    elif name in ("DataDrivenCompressor", "CalibratedRTNCompressor"):
        from auto_round.compressors.data_driven import DataDrivenCompressor, CalibratedRTNCompressor

        return {
            "DataDrivenCompressor": DataDrivenCompressor,
            "CalibratedRTNCompressor": CalibratedRTNCompressor,
        }[name]
    elif name == "ZeroShotCompressor":
        from auto_round.compressors.zero_shot import ZeroShotCompressor

        return ZeroShotCompressor
    elif name == "ModelFreeCompressor":
        from auto_round.compressors.model_free import ModelFreeCompressor

        return ModelFreeCompressor
    elif name in ("ExtraConfig", "TuningExtraConfig", "SchemeExtraConfig", "MLLMExtraConfig", "DiffusionExtraConfig"):
        from auto_round.compressors.config import (
            DiffusionExtraConfig,
            ExtraConfig,
            MLLMExtraConfig,
            SchemeExtraConfig,
            TuningExtraConfig,
        )

        return {
            "ExtraConfig": ExtraConfig,
            "TuningExtraConfig": TuningExtraConfig,
            "SchemeExtraConfig": SchemeExtraConfig,
            "MLLMExtraConfig": MLLMExtraConfig,
            "DiffusionExtraConfig": DiffusionExtraConfig,
        }[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
