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
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig

from auto_round.schemes import QuantizationScheme
from auto_round_extension.vllm_ext.utils import _is_mxfp4_w4a4, _is_mxfp8_w8a8

logger = init_logger(__name__)


QLINEAR_METHODS_DISPATCH_TABLE = {}


class AutoRoundQuantLinearMethod(LinearMethodBase):

    def __init__(self, impl, config=None, scheme=None):
        self.config = config
        self.impl = impl
        self.scheme = scheme

    @staticmethod
    def get_method(
        quant_config: AutoRoundConfig,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "AutoRoundQuantLinearMethod":

        def get_scheme(quant_config: AutoRoundConfig, prefix: str):
            # Check extra_config first
            layer_schemes = quant_config.layer_schemes
            # FIXME: make more robust
            for name, scheme in layer_schemes.items():
                if prefix.startswith(name):
                    return scheme
            # If not found, use default
            return quant_config.quant_scheme

        def check_quantized(weight_bits: int) -> bool:
            return weight_bits < 16

        def get_impl(scheme: QuantizationScheme):
            if not check_quantized(scheme.bits):

                return UnquantizedLinearMethod()

            elif _is_mxfp8_w8a8(scheme):
                from auto_round_extension.vllm_ext.linear_impl_mxfp8 import AutoRoundMXFP8LinearImpl

                return AutoRoundMXFP8LinearImpl(quant_config)

            raise ValueError(f"Unsupported Linear scheme: {scheme}")

        layer_scheme = get_scheme(quant_config, prefix)
        impl = get_impl(layer_scheme)
        logger.debug("Apply %s to %s", impl.__class__.__name__, prefix)
        return AutoRoundQuantLinearMethod(impl=impl)

    @classmethod
    def get_min_capability(cls) -> int:
        return cls.impl.get_min_capability()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        return self.impl.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return self.impl.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        return self.impl.apply_weights(layer, x, bias=bias)
