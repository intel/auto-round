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

from auto_round.algorithms.base import BasePipelineMember
from auto_round.algorithms.quantization.base import BaseQuantizer, DiffusionMixin, RTNLayerFallbackMixin
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.pipeline import (
    ActCalibPolicy,
    CalibTiming,
    InputSource,
    BlockContext,
    QuantizationPipeline,
    merge_policies,
)
from auto_round.algorithms.quantization.sign_round.config import AdamRoundConfig, SignRoundConfig, SignRoundV2Config
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.quantization.sign_roundv2 import SignRoundV2Quantizer
from auto_round.algorithms.quantization.adam_round.adam import AdamRoundQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.quantization.rtn.quantizer import RTNQuantizer, OptimizedRTNQuantizer
from auto_round.algorithms.transforms.base import BaseWeightTransformer


def __getattr__(name):
    if name == "AWQConfig":
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        return AWQConfig
    if name == "AWQTransform":
        from auto_round.algorithms.transforms.awq.base import AWQTransform

        return AWQTransform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
