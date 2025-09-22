# Copyright (c) 2025 Intel Corporation
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
from __future__ import annotations

from typing import TYPE_CHECKING

from auto_round.quantizers.base import QuantizerType
from auto_round.quantizers.mode import TuningQuantizer, RTNQuantizer
from auto_round.quantizers.model_type import LLMQuantizer
from auto_round.quantizers.data_type import GGUFQuantizer

if TYPE_CHECKING:
    from auto_round.compressors import BaseCompressor


def create_quantizer(cls: "BaseCompressor"):
    # example
    quantizers = {
        # QuantizerType.DATA_TYPE: GGUFQuantizer,
        # QuantizerType.MODEL_TYPE: LLMQuantizer,
        QuantizerType.MODE: RTNQuantizer if cls.iters == 0 else TuningQuantizer,
    }

    dynamic_quantizer = type("AutoRoundQuantizer", tuple(quantizers.values()), {})
    return dynamic_quantizer(cls)
