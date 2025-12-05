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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig as _BaseAutoRoundConfig

from auto_round.schemes import QuantizationScheme
from auto_round_extension.vllm_ext.quant_method_linear import AutoRoundQuantLinearMethod
from auto_round_extension.vllm_ext.quant_method_moe import AutoRoundMoEMethod

logger = init_logger(__name__)


class AutoRoundExtensionConfig(_BaseAutoRoundConfig):
    SUPPORTED_DTYPES = _BaseAutoRoundConfig.SUPPORTED_DTYPES.union({"mx_fp"})
    SUPPORTED_FORMATS = _BaseAutoRoundConfig.SUPPORTED_FORMATS.union({"auto_round:llm_compressor"})

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # FIXME: (yi) make it compatible with `AutoRoundConfig`
        from vllm.attention.layer import Attention

        if isinstance(layer, Attention):
            from auto_round_extension.vllm_ext.kv_cache import AutoRoundKVCacheMethod

            return AutoRoundKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            quant_method = AutoRoundMoEMethod.get_moe_method(self, layer, prefix)
            return quant_method
        elif isinstance(layer, LinearBase):
            return AutoRoundQuantLinearMethod.get_method(self, layer, prefix)
        else:
            return None

    @staticmethod
    def _parse_quant_scheme(config: dict):
        quant_scheme_attrs = QuantizationScheme.get_attributes()
        filter_config = {key: value for key, value in config.items() if key in quant_scheme_attrs}
        quant_scheme = QuantizationScheme.from_dict(filter_config)
        return quant_scheme

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _BaseAutoRoundConfig:
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


# Patch vLLM’s AutoRoundConfig at import time
import vllm.model_executor.layers.quantization.auto_round as _auto_round_module

_auto_round_module.AutoRoundConfig = AutoRoundExtensionConfig
