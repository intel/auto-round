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
from auto_round.autoround import AutoRound, AutoRoundAdam, AutoRoundDiffusion, AutoRoundLLM, AutoRoundMLLM
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.quantization.rrq.config import RRQConfig
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
    "RRQConfig",
    "SignRoundConfig",
    "AdamRoundConfig",
    "SignRoundV2Config",
    "AWQConfig",
    "RotationConfig",
    "SpinQuantConfig",
    "load_rrq_model",
]


def __getattr__(name):
    """Lazy import for heavy submodules to avoid import cycles / slow startup.

    Currently exposes ``load_rrq_model`` (combined base + residual RRQ loading),
    which lives in ``auto_round.inference.rrq_model`` and pulls in the full
    export path.
    """
    if name == "load_rrq_model":
        from auto_round.inference.rrq_model import load_rrq_model

        return load_rrq_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
