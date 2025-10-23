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

from typing import Any

import torch
from quant_method_moe import AutoRoundMoEMethod
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig

from auto_round.schemes import QuantizationScheme

logger = init_logger(__name__)


class AutoRoundExtensionConfig(AutoRoundConfig):
    SUPPORTED_DTYPES = AutoRoundConfig.SUPPORTED_DTYPES.union({"mx_fp"})
    SUPPORTED_FORMATS = AutoRoundConfig.SUPPORTED_FORMATS.union({"auto_round:llm_compressor"})

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # FIXME: (yi) make it compatible with `AutoRoundConfig`
        if isinstance(layer, FusedMoE):
            quant_method = AutoRoundMoEMethod.get_moe_method(self, layer, prefix)
            return quant_method

        return super().get_quant_method(layer, prefix)

    @staticmethod
    def _parse_quant_scheme(config: dict):
        quant_scheme_attrs = QuantizationScheme.get_attributes()
        filter_config = {key: value for key, value in config.items() if key in quant_scheme_attrs}
        quant_scheme = QuantizationScheme.from_dict(filter_config)
        return quant_scheme

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AutoRoundConfig:
        ar_config = super().from_config(config)
        # TODO: (yi) refine below implementation
        quant_scheme = AutoRoundExtensionConfig._parse_quant_scheme(config)
        layer_schemes = {}
        layer_schemes = {}  # ensure dict
        extra_config = getattr(ar_config, "extra_config", None)
        if extra_config is not None:
            for layer_name, layer_config in extra_config.items():
                layer_schemes[layer_name] = AutoRoundExtensionConfig._parse_quant_scheme(layer_config)
        ar_config.quant_scheme = quant_scheme
        ar_config.layer_schemes = layer_schemes
        return ar_config
