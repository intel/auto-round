# Copyright (c) 2023 Intel Corporation
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
from auto_round.autoround import AutoRound

# support for old api
from auto_round.autoround import AutoRoundLLM, AutoRoundMLLM, AutoRoundAdam, AutoRoundDiffusion
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.quantization.sign_round.config import (
    AdamRoundConfig,
    SignRoundConfig,
    SignRoundV2Config,
)
from auto_round.algorithms.transforms.awq.config import AWQConfig
from auto_round.algorithms.transforms.hadamard.config import RotationConfig
from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig
from auto_round.schemes import QuantizationScheme
from auto_round.auto_scheme import AutoScheme
from auto_round.utils import LazyImport
from auto_round.utils import monkey_patch

monkey_patch()

from .version import __version__

__all__ = [
    "__version__",
    "AutoRound",
    "AutoRoundLLM",
    "AutoRoundMLLM",
    "AutoRoundAdam",
    "AutoRoundDiffusion",
    "AutoScheme",
    "QuantizationScheme",
    "RTNConfig",
    "OptimizedRTNConfig",
    "SignRoundConfig",
    "AdamRoundConfig",
    "SignRoundV2Config",
    "AWQConfig",
    "RotationConfig",
    "SpinQuantConfig",
]
