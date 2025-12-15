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

from typing import Callable, Optional

import torch
import vllm.envs as envs
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

from auto_round_extension.vllm_ext.mxfp8_qdq_utils import dequant_mx_fp8, quant_mx_fp8
from auto_round_extension.vllm_ext.quant_impl import AutoRoundQuantImpl


class AutoRoundMXFP8LinearImpl(AutoRoundQuantImpl):
    def __init__(self, quant_scheme):
        self.quant_scheme = quant_scheme
        self.strategy = "TENSOR_GROUP"
        self.out_dtype = torch.get_default_dtype()
        self.group_size = 32

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def process_weights_after_loading(self, layer) -> None:
        return

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        # maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,  # E8M0 for MXFP8 scale
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # dequant weight
        weight = layer.weight
        weight_scale = layer.weight_scale
        dequnat_weight = dequant_mx_fp8(
            weight_fp8=weight.data,
            scale_e8m0=weight_scale.data,
            block_size=self.group_size,
            target_dtype=x.dtype,
        )
        dequnat_weight = dequnat_weight.to(x.dtype)
        # if not envs.VLLM_AR_MXFP8_DISABLE_INPUT_QDQ:
        # q-dq input
        x_scale, x_quant = quant_mx_fp8(x)
        dequant_x = dequant_mx_fp8(
            weight_fp8=x_quant,
            scale_e8m0=x_scale,
            block_size=self.group_size,
            target_dtype=x.dtype,
        )
        x = dequant_x.to(x.dtype)

        out = x @ dequnat_weight.t()
        return out.to(x.dtype) + (bias if bias is not None else 0)
