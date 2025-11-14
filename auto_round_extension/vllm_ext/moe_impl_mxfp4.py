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
import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs

import auto_round_extension.vllm_ext.mxfp4_qdq_utils as mxfp4_utils
from auto_round_extension.vllm_ext.mxfp4_qdq_utils import (
    dequant_mxfp4_to_fp8,
    mxfp4_gemm_with_unpacked_weight,
    run_mxfp4_emulations,
)
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


class AutoRoundMoEMethodMXFp4Impl(AutoRoundMoEMethod):
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
        self.mask_weights_buffer = None
        self.experts_mask_buffer = None
        self.num_all_tokens_threshold = 16 * 1024
        # self.num_all_tokens_threshold = 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        E = num_experts
        H = hidden_size
        IN = intermediate_size_per_partition
        if self.has_bias:
            # TODO: yiliu30 use the dtype in CK
            bias_dtype = torch.bfloat16
            w13_bias = torch.nn.Parameter(torch.zeros(E, 2 * IN, dtype=bias_dtype), requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(E, H, dtype=bias_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        if envs.VLLM_AR_MXFP4_MODULAR_MOE:
            from vllm.model_executor.layers.fused_moe.config import (
                ocp_mx_moe_quant_config,
            )

            if envs.VLLM_MXFP4_PRE_UNPACK_TO_FP8:
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

            self.input_dtype = "mxfp4"
            self.weight_dtype = "mxfp4"
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

    def _dequant_fp4_to_fp8(self, layer):
        weight_name_lst = ["w13_weight", "w2_weight"]

        for weight_name_prefix in weight_name_lst:
            weight_name = f"{weight_name_prefix}_packed"
            weight = getattr(layer, weight_name)
            weight_scale_name = f"{weight_name_prefix}_scale"
            weight_scale = getattr(layer, weight_scale_name)
            new_weight_name = f"{weight_name_prefix}_unpacked"
            new_scale_name = weight_scale_name
            num_experts, _, _ = weight.shape
            unpacked_weight_lst = []
            scale_list = []
            for expert_index in range(num_experts):
                weight_fp8, scale_bf16 = dequant_mxfp4_to_fp8(
                    data_lp=weight[expert_index],
                    scale_e8m0=weight_scale[expert_index],
                )

                unpacked_weight_lst.append(weight_fp8)
                scale_list.append(scale_bf16)
            unpacked_weight_fp8 = torch.stack(unpacked_weight_lst, dim=0)
            scale_bf16 = torch.stack(scale_list, dim=0)
            assert unpacked_weight_fp8.shape[0] == num_experts, (
                f"Expected {num_experts} unpacked weights, got " f"{unpacked_weight_fp8.shape[0]}"
            )
            delattr(layer, weight_name)
            delattr(layer, weight_scale_name)
            layer.register_parameter(
                new_weight_name,
                torch.nn.Parameter(
                    unpacked_weight_fp8,
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                new_scale_name,
                torch.nn.Parameter(
                    scale_bf16,
                    requires_grad=False,
                ),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.debug(f"Processing weights after loading for layer: {layer._prefix}")
        if envs.VLLM_ENABLE_STATIC_MOE:
            if envs.VLLM_MXFP4_PRE_UNPACK_WEIGHTS:
                self._dequant_fp4_to_fp8(layer)
                return
        elif envs.VLLM_AR_MXFP4_MODULAR_MOE:
            if envs.VLLM_MXFP4_PRE_UNPACK_TO_FP8:
                self._dequant_fp4_to_fp8(layer)
                return

            def revert_interleaved_bias(bias):
                """
                Convert from blocked bias format to interleaved format.

                Args:
                    bias: Tensor of shape [E, intermediate_size*2] where the first half contains
                        w1 biases and second half contains w3 biases for each expert

                Returns:
                    Tensor with interleaved w1 and w3 biases
                """
                # loaded bias[0]: [e0_w1_bias_0, e0_w1_bias_1, ..., e0_w3_bias_0, e0_w3_bias_1...]
                # Expected bias[0]: [e0_w1_bias_0, e0_w3_bias_0, e0_w1_bias_1, e0_w3_bias_1...]
                # The expected bias order is used in triton kernel https://github.com/vllm-project/vllm/pull/22508
                revert_bias = torch.zeros_like(bias, device=bias.device)
                E, two_IN = bias.shape

                # Verify the shape is as expected
                if two_IN % 2 != 0:
                    raise ValueError(f"Expected even number of bias elements, got {two_IN}")

                revert_bias[..., ::2] = bias[..., : two_IN // 2]
                revert_bias[..., 1::2] = bias[..., two_IN // 2 :]

                return revert_bias

            # breakpoint()
            if self.has_bias:
                if envs.VLLM_AR_POST_PROCESS_GPTOSS:
                    w13_bias_swapped = revert_interleaved_bias(layer.w13_bias)
                    layer.w13_bias.data.copy_(w13_bias_swapped)

            if envs.VLLM_MXFP4_PRE_UNPACK_WEIGHTS:

                w1 = layer.w13_weight_packed
                w1_scale = layer.w13_weight_scale
                w1 = mxfp4_utils.to_dtype(
                    data_lp=w1,
                    scale_e8m0=w1_scale,
                    elem_dtype="fp4_e2m1",
                    block_size=32,
                    target_dtype=torch.bfloat16,
                )

                def revert_interleaved_w1(w1):
                    new_w1 = torch.zeros_like(w1)
                    E, N, H = w1.shape
                    new_w1[:, ::2, :] = w1[:, : N // 2, :]
                    new_w1[:, 1::2, :] = w1[:, N // 2 :, :]
                    return new_w1

                if envs.VLLM_AR_POST_PROCESS_GPTOSS:
                    w1 = revert_interleaved_w1(w1)

                w1_scale = None
                w2 = layer.w2_weight_packed
                w2_scale = layer.w2_weight_scale
                w2 = mxfp4_utils.to_dtype(
                    data_lp=w2,
                    scale_e8m0=w2_scale,
                    elem_dtype="fp4_e2m1",
                    block_size=32,
                    target_dtype=torch.bfloat16,
                )
                w2_scale = None
                del layer.w13_weight_packed
                del layer.w13_weight_scale
                del layer.w2_weight_packed
                del layer.w2_weight_scale
                layer.w13_weight_scale = None
                layer.w2_weight_scale = None
                layer.register_parameter(
                    "w13_weight_unpacked",
                    torch.nn.Parameter(
                        w1,
                        requires_grad=False,
                    ),
                )
                layer.register_parameter(
                    "w2_weight_unpacked",
                    torch.nn.Parameter(
                        w2,
                        requires_grad=False,
                    ),
                )

        else:
            raise NotImplementedError("process_weights_after_loading is not implemented for now.")

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

        # There are three implementations:

        if envs.VLLM_AR_MXFP4_MODULAR_MOE:
            from vllm.model_executor.layers.fused_moe import fused_experts

            if envs.VLLM_MXFP4_PRE_UNPACK_TO_FP8:
                w1 = layer.w13_weight_unpacked
                w2 = layer.w2_weight_unpacked
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
            if envs.VLLM_MXFP4_PRE_UNPACK_WEIGHTS:
                w1 = layer.w13_weight_unpacked
                w2 = layer.w2_weight_unpacked
            else:
                w1 = layer.w13_weight_packed
                w2 = layer.w2_weight_packed
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

        num_all_tokens, hidden_dim = x.shape
        num_experts = layer.local_num_experts
        total_num_experts = router_logits.size(-1)
        if (
            self.mask_weights_buffer is None
            or self.mask_weights_buffer.dtype != x.dtype
            or self.mask_weights_buffer.device != x.device
            or self.mask_weights_buffer.shape[0] < num_all_tokens
            or self.mask_weights_buffer.shape[1] < total_num_experts
        ):
            if num_all_tokens > self.num_all_tokens_threshold:
                mask_weights = torch.zeros((num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device)
                if self.mask_weights_buffer is None and self.num_all_tokens_threshold != 0:
                    self.mask_weights_buffer = torch.zeros(
                        (self.num_all_tokens_threshold, total_num_experts),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    self.experts_mask_buffer = torch.zeros(
                        (self.num_all_tokens_threshold, total_num_experts),
                        dtype=x.dtype,
                        device=x.device,
                    )
            else:
                self.mask_weights_buffer = torch.zeros(
                    (num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device
                )
                self.experts_mask_buffer = torch.zeros(
                    (num_all_tokens, total_num_experts), dtype=x.dtype, device=x.device
                )
                mask_weights = self.mask_weights_buffer
                experts_mask = self.experts_mask_buffer
        else:
            self.mask_weights_buffer.zero_()
            mask_weights = self.mask_weights_buffer
            self.experts_mask_buffer.zero_()
            experts_mask = self.experts_mask_buffer

        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)
        experts_mask.scatter_(-1, topk_ids, topk_weights)
        mask_weights.scatter_(-1, topk_ids, 1)
        mask_weights = mask_weights[:num_all_tokens, :total_num_experts]
        mask_weights = mask_weights.transpose(0, 1)
        experts_mask = experts_mask[:num_all_tokens, :total_num_experts]
        experts_mask = experts_mask.transpose(0, 1)
        # Note: ep_size equal tp_size
        ep_rank = get_tensor_model_parallel_rank() if expert_map is not None else 0
        ep_shift = ep_rank * num_experts

        if envs.VLLM_ENABLE_STATIC_MOE and not envs.VLLM_MXFP4_PRE_UNPACK_WEIGHTS:
            num_experts, intermediate_size_per_partition_x2, _ = layer.w13_weight_packed.shape
            intermediate_size_per_partition = intermediate_size_per_partition_x2 // 2
            for expert_index in range(num_experts):
                mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
                current_state_static = x * mask_weight

                local_w13_packed = layer.w13_weight_packed[expert_index]
                local_w13_scale = layer.w13_weight_scale[expert_index]
                local_w2_packed = layer.w2_weight_packed[expert_index]
                local_w2_scale = layer.w2_weight_scale[expert_index]

                local_w1_packed = local_w13_packed[:intermediate_size_per_partition, ...]
                local_w1_scale = local_w13_scale[:intermediate_size_per_partition, ...]

                local_w3_packed = local_w13_packed[intermediate_size_per_partition:, ...]
                local_w3_scale = local_w13_scale[intermediate_size_per_partition:, ...]

                local_w1_bias = None
                local_w2_bias = None
                local_w3_bias = None
                if self.has_bias:
                    local_w13_bias = layer.w13_bias[expert_index]
                    local_w1_bias = local_w13_bias[:intermediate_size_per_partition]
                    local_w3_bias = local_w13_bias[intermediate_size_per_partition:]
                    local_w2_bias = layer.w2_bias[expert_index]

                local_w1_out = run_mxfp4_emulations(
                    x=current_state_static,
                    weight=local_w1_packed,
                    weight_scale=local_w1_scale,
                    bias=local_w1_bias,
                )
                local_w3_out = run_mxfp4_emulations(
                    x=current_state_static,
                    weight=local_w3_packed,
                    weight_scale=local_w3_scale,
                    bias=local_w3_bias,
                )

                w13_out = apply_act(local_w1_out, local_w3_out, activation)

                local_w2_out = run_mxfp4_emulations(
                    x=w13_out,
                    weight=local_w2_packed,
                    weight_scale=local_w2_scale,
                    bias=local_w2_bias,
                )
                padded_weight = experts_mask[expert_index + ep_shift].unsqueeze(1)
                local_w2_out = local_w2_out * padded_weight
                if expert_index == 0:
                    final_hidden_states = local_w2_out
                else:
                    final_hidden_states += local_w2_out
            return final_hidden_states
        if envs.VLLM_ENABLE_STATIC_MOE and envs.VLLM_MXFP4_PRE_UNPACK_WEIGHTS:
            num_experts, intermediate_size_per_partition_x2, _ = layer.w13_weight_unpacked.shape
            intermediate_size_per_partition = intermediate_size_per_partition_x2 // 2

            for expert_index in range(num_experts):
                mask_weight = mask_weights[expert_index + ep_shift].unsqueeze(1)
                current_state_static = x * mask_weight

                local_unpacked_w13 = layer.w13_weight_unpacked[expert_index]
                local_w13_scale = layer.w13_weight_scale[expert_index]

                local_unpacked_w2 = layer.w2_weight_unpacked[expert_index]
                local_w2_scale = layer.w2_weight_scale[expert_index]

                local_unpacked_w1 = local_unpacked_w13[:intermediate_size_per_partition, ...]
                half_scale = local_w13_scale.shape[0] // 2
                local_w1_scale = local_w13_scale[:half_scale, ...]
                local_unpacked_w3 = local_unpacked_w13[intermediate_size_per_partition:, ...]
                local_w3_scale = local_w13_scale[half_scale:, ...]

                local_w1_bias = None
                local_w2_bias = None
                local_w3_bias = None
                if self.has_bias:
                    local_w13_bias = layer.w13_bias[expert_index]
                    local_w1_bias = local_w13_bias[:intermediate_size_per_partition]
                    local_w3_bias = local_w13_bias[intermediate_size_per_partition:]
                    local_w2_bias = layer.w2_bias[expert_index]

                local_w1_out = mxfp4_gemm_with_unpacked_weight(
                    x=current_state_static,
                    weight_fp8=local_unpacked_w1,
                    weight_scale_bf16=local_w1_scale,
                    bias=local_w1_bias,
                )
                local_w3_out = mxfp4_gemm_with_unpacked_weight(
                    x=current_state_static,
                    weight_fp8=local_unpacked_w3,
                    weight_scale_bf16=local_w3_scale,
                    bias=local_w3_bias,
                )

                w13_out = apply_act(local_w1_out, local_w3_out, activation)

                local_w2_out = mxfp4_gemm_with_unpacked_weight(
                    x=w13_out,
                    weight_fp8=local_unpacked_w2,
                    weight_scale_bf16=local_w2_scale,
                    bias=local_w2_bias,
                )

                padded_weight = experts_mask[expert_index + ep_shift].unsqueeze(1)
                local_w2_out = local_w2_out * padded_weight
                if expert_index == 0:
                    final_hidden_states = local_w2_out
                else:
                    final_hidden_states += local_w2_out
            return final_hidden_states
        raise NotImplementedError("Not implemented for now.")
