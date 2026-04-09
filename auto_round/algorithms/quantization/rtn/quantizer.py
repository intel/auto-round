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
import traceback
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import accelerate
import torch

from auto_round.algorithms.quantization.base import BaseQuantizers
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.compressors_new.shard_writer import ShardWriter
from auto_round.compressors_new.utils import (
    IndexSampler,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    immediate_pack,
    infer_bits_by_data_type,
    init_cache,
    reset_params,
    set_layer_config,
)
from auto_round.data_type.utils import update_block_global_scale_if_needed
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    convert_module_to_hp_if_necessary,
    get_lm_head_name,
    get_module,
    htcore,
    is_auto_device_mapping,
    is_hpex_available,
    memory_monitor,
    set_amax_for_all_moe_layers,
    set_module,
)
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    get_major_device,
    parse_available_devices,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block


class RTNQuantizer(BaseQuantizers):

    def __init__(self, config: RTNConfig):
        BaseQuantizers.__init__(self, config)

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

    def quantize_layer(self, name: str, dtype: torch.dtype = None) -> None:
        """Quantizes a layer using RTN (Round-To-Nearest) if available.

        This function attempts to quantize a layer by switching its data type to a
        `rtn_*` version if supported, then wraps and unwraps the module to apply
        quantization. If GPU memory is insufficient, it falls back to CPU.

        If packing is enabled (`immediate_packing`), the function will also export
        the quantized layer to the appropriate backend format.

        Args:
            name (str): Name of the layer to quantize.

        Raises:
            RuntimeError: If quantization fails for reasons unrelated to memory.
        """

        m = get_module(self.model, name)
        if dtype is not None:
            m = m.to(dtype)

        m = convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, self.compress_context.device)
        set_module(self.model, name, m)
        tuning_device = m.tuning_device if hasattr(m, "tuning_device") else self.compress_context.device
        # Step 1: let gguf merge layers or rename module first and we will handle the RTN is gguf specific logic
        if (
            self.compress_context.is_immediate_packing
            and self.compress_context.formats[0].is_gguf()
            and not getattr(self.config, "disable_opt_rtn", False)
        ):
            m = m.to(tuning_device)
            m.scale = None
            m.zp = None
        else:
            try:
                disable_opt_rtn = False
                if (
                    self.config.orig_disable_opt_rtn is None
                    and self.model_context.is_moe_model
                    and "expert" in m.global_name
                    and "shared_expert" not in m.global_name
                    and self.config.super_bits is None  # GGUF still uses the optimized RTN for MoE layers
                ):
                    disable_opt_rtn = True
                    logger.warning_once(
                        "MoE layer detected: optimized RTN is disabled for efficiency. "
                        "Use `--enable_opt_rtn` to force-enable it for MoE layers."
                    )
                m = m.to(tuning_device)
                m = WrapperLinear(
                    m,
                    device=tuning_device,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                    enable_torch_compile=self.compress_context.enable_torch_compile,
                    disable_opt_rtn=disable_opt_rtn,
                    enable_rtn=True,
                )
                m = m.unwrapper({})
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                m = m.orig_layer if hasattr(m, "orig_layer") else m
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU.")
                    m.to("cpu")
                    m = WrapperLinear(
                        m,
                        enable_minmax_tuning=False,
                        enable_norm_bias_tuning=False,
                        enable_round_tuning=False,
                        enable_torch_compile=self.compress_context.enable_torch_compile,
                        enable_rtn=True,
                    )
                    m = m.unwrapper({})
                except Exception as e:
                    raise

        set_module(self.model, name, m)
        self._immediate_pack_and_save_module(name)

    def _immediate_pack_and_save_module(self, module_name):
        shard_writer = ShardWriter.get_shard_writer()
        to_cpu = self.compress_context.low_gpu_mem_usage
        module = get_module(self.model, module_name)
        if self.compress_context.is_immediate_packing:  # For gguf, packing conducts on block level
            immediate_pack(module_name, self.layer_config)
            if to_cpu:
                module = module.to("cpu")
                packed_module = get_module(self.model, module_name)
                set_module(self.model, module_name, packed_module.to("cpu"))
        else:
            if to_cpu:
                module = module.to("cpu")
            set_module(self.model, module_name, module)
        if self.compress_context.is_immediate_saving:
            module = get_module(self.model, module_name)
            module.to("cpu")
            shard_writer.write(module, module_name, False)
            # Free RAM immediately: the data is now in the shard-writer buffer
            # (and will be flushed to disk).  Keeping it also in the model tree
            # causes linear RAM growth for large models.
            module.to("meta")


class OptimizedRTNQuantizer(RTNQuantizer):

    def __init__(self, config: RTNConfig):
        BaseQuantizers.__init__(self, config)
        self.batch_size = config.batch_size
        self.batch_dim = config.batch_dim
        self.data_type = config.data_type
        self.group_size = config.group_size
        self.infer_bs_coeff = config.infer_bs_coeff

        self.enable_alg_ext = True

    def quantize_layer_outside_block(self, *args, **kwargs):
        return self.quantize_layer(*args, **kwargs)

    def quantize_block(
        self, block: torch.nn.Module, input_ids=None, input_others=None, reference_output=None, **kwargs
    ):
        """Apply imatrix-informed RTN quantization to a block.

        Pure-algorithm entry point.  All infrastructure (device placement,
        act-max hook registration, imatrix collection, cleanup) is handled
        by the Compressor before calling this method.

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

    # _get_block_outputs and _sampling_inputs are defined in BaseQuantizers and inherited.
