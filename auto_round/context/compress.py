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
from typing import Callable, Optional, Union

import torch

from auto_round.context.base import BaseContext
from auto_round.utils.device import (
    clear_memory,
    clear_memory_if_reached_threshold,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)
from auto_round.utils.device_manager import device_manager

__all__ = ["CompressContext"]


class CompressContext(BaseContext):

    def __init__(
        self,
        low_cpu_mem_usage: bool = True,
        low_gpu_mem_usage: bool = False,
        enable_torch_compile: bool = False,
        is_immediate_packing: bool = False,
        is_immediate_saving: bool = False,
        formats: Union[list, str] = None,
        output_dir: str = "./compressed_models",
        static_kv_dtype: Optional[torch.dtype] = None,
        static_attention_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.formats = formats
        self.output_dir = output_dir
        # All device / device-list state lives on the process-wide DeviceManager
        # singleton, which is configured from ``device_map`` before this context is
        # created.  CompressContext just reads from it -- it never owns a copy.
        self.cache_device = torch.device("cpu") if low_gpu_mem_usage else device_manager.device

        self.enable_torch_compile = enable_torch_compile
        self.immediate_packing = is_immediate_packing
        self.is_immediate_packing = is_immediate_packing
        self.is_immediate_saving = is_immediate_saving
        self.static_kv_dtype = static_kv_dtype
        self.static_attention_dtype = static_attention_dtype

    def clear_memory(self, tensor=None):
        """Clear GPU/CPU memory only when ``low_gpu_mem_usage`` is enabled."""
        if self.low_gpu_mem_usage:
            clear_memory(tensor, device_list=device_manager.device_list)
