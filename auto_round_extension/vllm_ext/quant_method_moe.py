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

from typing import Optional

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig

from auto_round.schemes import QuantizationScheme
from auto_round_extension.vllm_ext.utils import _is_mxfp4_w4a4, _is_mxfp8_w8a8, get_scheme, need_quantize

logger = init_logger(__name__)


QMOE_METHODS_DISPATCH_TABLE = {}


class AutoRoundMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @staticmethod
    def get_moe_method(
        quant_config: AutoRoundConfig,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "AutoRoundMoEMethod":

        def get_impl(scheme: QuantizationScheme):
            if not need_quantize(scheme.bits):
                from vllm.model_executor.layers.fused_moe.layer import (
                    UnquantizedFusedMoEMethod,
                )

                return UnquantizedFusedMoEMethod(layer.moe_config)

            elif _is_mxfp4_w4a4(scheme):
                from auto_round_extension.vllm_ext.moe_impl_mxfp4 import AutoRoundMoEMethodMXFp4Impl

                return AutoRoundMoEMethodMXFp4Impl(quant_config, layer.moe_config)

            elif _is_mxfp8_w8a8(scheme):
                from auto_round_extension.vllm_ext.moe_impl_mxfp8 import AutoRoundMoEMethodMXFp8Impl

                return AutoRoundMoEMethodMXFp8Impl(quant_config, layer.moe_config)

            raise ValueError(f"Unsupported FusedMoe scheme: {scheme}")

        layer_scheme = get_scheme(quant_config, prefix)
        impl = get_impl(layer_scheme)
        layer._prefix = prefix
        logger.debug("Apply %s to %s", impl.__class__.__name__, prefix)
        return impl

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        return self.impl.get_fused_moe_quant_config(layer)
