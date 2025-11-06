# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.logger import init_logger


from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter, PerTensorScaleParameter

from auto_round_extension.vllm_ext.mxfp4_qdq_utils import (
    dequant_mxfp4_to_fp8,
    mxfp4_gemm_with_unpacked_weight,
    run_mxfp4_emulations,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["AutoRoundMXFP4LinearImpl"]

from auto_round_extension.vllm_ext.quant_impl import AutoRoundQuantImpl


class AutoRoundMXFP4LinearImpl(AutoRoundQuantImpl):
    def __init__(self, quant_scheme):
        self.quant_scheme = quant_scheme
        self.group_size = 32

    @classmethod
    def get_min_capability(cls) -> int:
        if envs.VLLM_USE_MXFP4_CT_EMULATIONS:
            return 80
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition // 2, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                # dtype=torch.uint8,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        # FIXME: may dequant to bf16
        if envs.VLLM_MXFP4_PRE_UNPACK_WEIGHTS:
            from auto_round_extension.vllm_ext.mxfp4_qdq_utils import (
                dequant_mxfp4_to_fp8,
                mxfp4_gemm_with_unpacked_weight,
                run_mxfp4_emulations,
            )

            weight_fp8, scale_bf16 = dequant_mxfp4_to_fp8(
                data_lp=layer.weight_packed,
                scale_e8m0=layer.weight_scale,
            )
            del layer.weight_packed
            del layer.weight_scale
            layer.weight_packed = None
            layer.weight_scale = None
            layer.register_parameter(
                "weight_unpacked_fp8",
                torch.nn.Parameter(
                    weight_fp8,
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                "weight_scale_bf16",
                torch.nn.Parameter(
                    scale_bf16,
                    requires_grad=False,
                ),
            )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not envs.VLLM_MXFP4_PRE_UNPACK_WEIGHTS:
            out = run_mxfp4_emulations(x=x, weight=layer.weight_packed, weight_scale=layer.weight_scale)
            if bias is not None:
                out = out + bias
            return out
        else:
            out = mxfp4_gemm_with_unpacked_weight(
                x=x,
                weight_fp8=layer.weight_unpacked_fp8,
                weight_scale_bf16=layer.weight_scale_bf16,
                bias=bias,
            )
            return out

