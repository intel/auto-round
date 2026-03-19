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
    from auto_round.compressors_new.calib import CalibCompressor, CalibratedRTNCompressor
    from auto_round.compressors_new.entry import AutoRound, Compressor
    from auto_round.compressors_new.zero_shot import ZeroShotCompressor

__all__ = [
    "Compressor",
    "CalibCompressor",
    "CalibratedRTNCompressor",
    "ZeroShotCompressor",
    "AutoRound",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "Compressor" or name == "AutoRound":
        from auto_round.compressors_new.entry import Compressor, AutoRound

        if name == "Compressor":
            return Compressor
        return AutoRound
    elif name in ("CalibCompressor", "CalibratedRTNCompressor"):
        from auto_round.compressors_new.calib import CalibCompressor, CalibratedRTNCompressor

        return {
            "CalibCompressor": CalibCompressor,
            "CalibratedRTNCompressor": CalibratedRTNCompressor,
        }[name]
    elif name == "ZeroShotCompressor":
        from auto_round.compressors_new.zero_shot import ZeroShotCompressor

        return ZeroShotCompressor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
