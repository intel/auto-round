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
"""Hadamard rotation sub-package for ``algorithms/rotation``."""

from auto_round.algorithms.rotation.hadamard.apply import HadamardRotation, apply_hadamard_transform
from auto_round.algorithms.rotation.hadamard.config import HadamardConfig, normalize_hadamard_config
from auto_round.algorithms.rotation.hadamard.transforms import (
    HADAMARDS,
    HadamardTransform,
    RandomHadamardTransform,
    build_hadamard_transform,
)

__all__ = [
    # Algorithm class
    "HadamardRotation",
    # Config
    "HadamardConfig",
    "normalize_hadamard_config",
    # Transform modules
    "HadamardTransform",
    "RandomHadamardTransform",
    "HADAMARDS",
    "build_hadamard_transform",
    # One-shot convenience
    "apply_hadamard_transform",
]
