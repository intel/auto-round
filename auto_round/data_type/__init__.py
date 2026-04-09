# Copyright (c) 2024 Intel Corporation
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

"""Data type registration and utilities for auto-round quantization.

This package registers all supported quantization data types (int, fp8, mxfp,
nvfp, gguf, w4fp8) and exposes the quantization function registry together with
common tensor-manipulation helpers used across data-type implementations.
"""

import auto_round.data_type.int
import auto_round.data_type.mxfp
import auto_round.data_type.fp8
from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE
import auto_round.data_type.w4fp8
from auto_round.data_type.utils import (
    get_quant_func,
    reshape_pad_tensor_by_group_size,
    update_fused_layer_global_scales,
)
import auto_round.data_type.nvfp
import auto_round.data_type.gguf
