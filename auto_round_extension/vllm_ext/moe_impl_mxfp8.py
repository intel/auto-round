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

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

from auto_round_extension.vllm_ext.quant_method_moe import AutoRoundMoEMethod

logger = init_logger(__name__)


def apply_act(local_w1_out: torch.Tensor, local_w3_out: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "silu":
        act_fn = F.silu
        w13_out = act_fn(local_w1_out) * local_w3_out
    elif activation == "swigluoai":
        limit = 7.0
        alpha = 1.702
        local_w1_out = local_w1_out.clamp(min=None, max=limit)
        local_w3_out = local_w3_out.clamp(min=-limit, max=limit)
        glu = (local_w1_out) * F.sigmoid(local_w1_out * alpha)
        w13_out = (local_w3_out + 1) * glu
    else:
        raise NotImplementedError(f"Activation {activation} is not implemented.")
    return w13_out


class AutoRoundMoEMethodMXFp8Impl(AutoRoundMoEMethod):
    def __init__(
        self,
        quant_config: "AutoRoundConfig",  # type: ignore # noqa E501
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.use_marlin = False
        self.group_size = 32
        self.quant_config = quant_config
        self.has_bias = self.moe.has_bias


    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.uint8,  # E8M0 for MXFP8 scale
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        # w2
        w2_weight_scale = torch.nn.Parameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,  # E8M0 for MXFP8 scale
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-TENSORGROUP quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)



    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        from vllm.model_executor.layers.fused_moe.config import (
            ocp_mx_moe_quant_config,
        )

        self.input_dtype = "mxfp8_e4m3"
        self.weight_dtype = "mxfp8_e4m3"
        return ocp_mx_moe_quant_config(
            quant_dtype=self.input_dtype,
            weight_dtype=self.weight_dtype,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            w1_bias=layer.w13_bias if self.has_bias else None,
            w2_bias=layer.w2_bias if self.has_bias else None,
            block_shape=None,
        )
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return 

    @torch.inference_mode()
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        topk_weights, topk_ids, _ = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
        assert self.fused_experts is None

        from vllm.model_executor.layers.fused_moe import fused_experts

        w1 = layer.w13_weight
        w2 = layer.w2_weight
        out = fused_experts(
            x,
            w1,
            w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            quant_config=self.moe_quant_config,
        )
        return out
