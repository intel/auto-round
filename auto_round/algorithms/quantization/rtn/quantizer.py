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
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import accelerate
import torch

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.compressors.utils import (
    IndexSampler,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    infer_bits_by_data_type,
    init_cache,
    reset_params,
    set_layer_config,
)
from auto_round.data_type.utils import update_block_global_scale_if_needed
from auto_round.utils import (
    check_to_quantized,
    get_module,
    set_amax_for_all_moe_layers,
    set_module,
)


@register_pipeline_member(RTNConfig)
class RTNQuantizer(BaseQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        BaseQuantizer.__init__(self, config)

    @torch.no_grad()
    def quantize_block(self, block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, **kwargs) -> dict:
        """Apply zero-shot RTN quantization to a block.

        Returns:
            dict: Empty dict (zero-shot RTN has no tunable parameters to return).
        """
        if (
            self.config.is_act_nv_fp
            or self.config.is_static_afp8
            or (self.config.is_wfp8afp8 and not self.config.act_dynamic)
        ):
            # For FP8 static / NVFP paths, expert input scales are derived during
            # layer quantization from the current act_max. Unify MoE input-proj
            # act_max values before quantizing each expert so exported input_scale
            # stays aligned across experts.
            set_amax_for_all_moe_layers(block, attr_name="act_max")  # TODO wenhuach should move to compressor

        for _name, m in block.named_modules():
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer(m.global_name)
        return {}

    @torch.no_grad()
    def quantize_layer(self, name: str, dtype: torch.dtype = None) -> None:
        if dtype is not None:
            layer = get_module(self.model, name)
            set_module(self.model, name, layer.to(dtype))
        self.quantize_layer_via_rtn(name, disable_opt_rtn=True)


@register_pipeline_member(OptimizedRTNConfig)
class OptimizedRTNQuantizer(RTNQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        BaseQuantizer.__init__(self, config)
        self.data_type = config.data_type
        self.group_size = config.group_size
        self.infer_bs_coeff = config.infer_bs_coeff
        self.enable_imatrix = getattr(config, "enable_imatrix", False)

        self.enable_alg_ext = True

    def is_support_compile_block(self):
        return False

    def register_fp_input_forward_hooks(self, block):
        """Register FP-input hooks: act_max (from base) + imatrix."""
        handles = super().register_fp_input_forward_hooks(block)
        if self.enable_imatrix:
            handles.extend(self._register_imatrix_hooks(block, with_count=True))
        return handles


    def _register_imatrix_hooks(self, model, *, with_count: bool = False):
        def collect_imatrix(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

            if not hasattr(module, "imatrix"):
                module.imatrix = squared
                if with_count:
                    module.imatrix_cnt = input.shape[0]
                return
            module.imatrix += squared.to(module.imatrix.device)
            if with_count:
                module.imatrix_cnt += input.shape[0]

        handles = []
        for _, module in model.named_modules():
            if isinstance(module, self.supported_types) and check_to_quantized(module):
                handles.append(module.register_forward_hook(collect_imatrix))
        return handles

    @torch.no_grad()
    def quantize_block(self, block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, **kwargs):
        """Apply imatrix-informed RTN quantization to a block."""
        update_block_global_scale_if_needed(
            block, self.data_type, self.group_size
        )  # TODO move this to compressor, wenhuach
        if (
            self.config.is_act_nv_fp
            or self.config.is_static_afp8
            or (self.config.is_wfp8afp8 and not self.config.act_dynamic)
        ):
            # enable moe experts act_max automatic generation for Linear
            set_amax_for_all_moe_layers(block, attr_name="act_max")
        # Normalize imatrix and quantize layers
        for name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer_outside_block(m.global_name)
