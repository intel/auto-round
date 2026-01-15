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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor

from auto_round.quantizers.algs.auto_round import ARQuantizer
from auto_round.quantizers.algs.rtn import OptRTNQuantizer, RTNQuantizer


class AutoRoundQuantizer:
    def __new__(cls, compressor: "BaseCompressor", dynamic_quantizers: dict = None):
        assert dynamic_quantizers is not None, "Please provide dynamic_quantizers dict."
        quantizer_cls = type("AutoRoundQuantizer", tuple(dynamic_quantizers.values()), {})
        return quantizer_cls(compressor)


class Quantizers:
    def __init__(self, quantizers: list[AutoRoundQuantizer]):
        self.quantizers = quantizers

    def quantize(self, *args, **kwargs):
        for quantizer in self.quantizers:
            quantizer.pre_quantize(*args, **kwargs)
            model, layer_config = quantizer.quantize(*args, **kwargs)
            quantizer.post_quantize(*args, **kwargs)
        return model, layer_config


def create_quantizers(compressor: "BaseCompressor"):

    alg_cls = None
    if compressor.iters > 0:
        alg_cls = ARQuantizer
    else:
        alg_cls = OptRTNQuantizer if compressor.disable_opt_rtn is False else RTNQuantizer

    dynamic_quantizers = {"algs": alg_cls}
    return Quantizers(
        quantizers=[
            AutoRoundQuantizer(compressor, dynamic_quantizers=dynamic_quantizers),
        ]
    )
