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
"""Calibration sub-package."""

from auto_round.calibration.base import Calibrator
from auto_round.calibration.state import CalibrationState
from auto_round.calibration.register import CALIBRATORS, get_calibrator, register_calibrator

# Importing the strategy modules triggers their ``@register_calibrator`` decorators.
from auto_round.calibration import llm as _llm  # noqa: F401
from auto_round.calibration import mllm as _mllm  # noqa: F401
from auto_round.calibration import diffusion as _diffusion  # noqa: F401

__all__ = [
    "Calibrator",
    "CalibrationState",
    "CALIBRATORS",
    "get_calibrator",
    "register_calibrator",
]
