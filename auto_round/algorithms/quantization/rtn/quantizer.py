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

from auto_round.algorithms.quantization.base import BaseQuantizers
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.quantization.utils import register_imatrix_hooks
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
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    get_lm_head_name,
    htcore,
    is_auto_device_mapping,
    is_hpex_available,
    memory_monitor,
    set_amax_for_all_moe_layers,
)
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    get_major_device,
    parse_available_devices,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)
from auto_round.wrapper import WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block


class RTNQuantizer(BaseQuantizers):

    def __init__(self, config: RTNConfig):
        BaseQuantizers.__init__(self, config)

    @torch.no_grad()
    def quantize_block(
        self, block: torch.nn.Module, input_ids=None, input_others=None, reference_output=None, **kwargs
    ) -> dict:
        """Apply zero-shot RTN quantization to a block.

        Pure-algorithm entry point.  Infrastructure (materialize, shard writing,
        device cleanup) is handled by the Compressor before/after this call.

        Args:
            block: Module already materialized and placed on the correct device.
            input_ids: Unused for zero-shot RTN (accepted for interface consistency).
            input_others: Unused for zero-shot RTN.
            reference_output: Unused for zero-shot RTN.

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
            set_amax_for_all_moe_layers(block, attr_name="act_max")

        for _name, m in block.named_modules():
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer(m.global_name)
        return {}

    @torch.no_grad()
    def quantize_layer(self, name: str, dtype: torch.dtype = None) -> None:
        self.quantize_layer_via_rtn(name, dtype=dtype)


class OptimizedRTNQuantizer(RTNQuantizer):

    def __init__(self, config: RTNConfig):
        BaseQuantizers.__init__(self, config)
        self.data_type = config.data_type
        self.group_size = config.group_size
        self.infer_bs_coeff = config.infer_bs_coeff
        self.enable_imatrix = getattr(config, "enable_imatrix", False)

        self.enable_alg_ext = True

    def register_calibration_hooks(self, model, *, act_max: bool = True, imatrix: bool = True):
        hook_handles = super().register_calibration_hooks(model, act_max=act_max, imatrix=imatrix)
        if imatrix and self.enable_imatrix:
            hook_handles.extend(register_imatrix_hooks(self, model, with_count=True))
        return hook_handles

    @torch.no_grad()
    def quantize_block(
        self, block: torch.nn.Module, input_ids=None, input_others=None, reference_output=None, **kwargs
    ):
        """Apply imatrix-informed RTN quantization to a block.

        Pure-algorithm entry point.  Device placement and cleanup are handled
        by the Compressor; act-max and imatrix hook registration are owned by
        the quantizer hook helpers before this method is called.

        Args:
            block: Module already placed on the correct device(s) with act_max
                attributes populated by the Compressor's hook pass.
            input_ids: Unused for optimized RTN; accepted for interface consistency.
            input_others: Unused for optimized RTN.
            reference_output: Unused for optimized RTN.
        """
        update_block_global_scale_if_needed(block, self.data_type, self.group_size)
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
