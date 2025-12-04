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



from typing import TYPE_CHECKING, Any, Literal, Optional, cast
import torch
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.logger import init_logger

logger = init_logger(__name__)

class AutoRoundKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from compressed-tensors
    checkpoints.
    """

    def __init__(self, quant_config):
        self.validate_kv_cache_scheme(quant_config)
        super().__init__(quant_config)
    
    @staticmethod
    def validate_kv_cache_scheme(quant_config):
        # FIXME: parse from quant_config
        return True