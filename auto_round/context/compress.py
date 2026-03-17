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
from typing import Any, Callable, Optional, Union

import torch

from auto_round.context.base import BaseContext
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    get_major_device,
    parse_available_devices,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)

__all__ = ["CompressContext"]


class CompressContext(BaseContext):
    def __init__(
        self,
        low_cpu_mem_usage: bool = True,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        is_immediate_packing: bool = False,
        is_immediate_saving: bool = False,
        formats: Union[list, str] = None,
        output_dir: str = "./compressed_models",
    ):
        super().__init__()
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.formats = formats
        self.output_dir = output_dir
        if device_map is None:
            device_map = 0
        self.device_map = device_map
        if isinstance(self.device_map, str):
            self.device_map = self.device_map.replace(" ", "")
        self.device_list = parse_available_devices(self.device_map)
        self.device = get_major_device(self.device_map)

        self.cache_device = torch.device("cpu") if low_gpu_mem_usage else self.device

        self.enable_torch_compile = enable_torch_compile
        self.immediate_packing = is_immediate_packing
        self.is_immediate_saving = is_immediate_saving
        self.formats = formats
