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
import copy
import traceback
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import accelerate
import torch

from auto_round.algorithms.quantization.auto_round.quantizer import ARQuantizer
from auto_round.algorithms.quantization.base import BaseQuantizers
from auto_round.algorithms.quantization.rtn.config import RTNConfig
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
from auto_round.modeling.fused_moe.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.utils import (
    check_to_quantized,
    clear_memory,
    convert_module_to_hp_if_necessary,
    get_lm_head_name,
    get_module,
    htcore,
    is_auto_device_mapping,
    is_hpex_available,
    memory_monitor,
    mv_module_from_gpu,
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

    def quantize_block(self, block_name: str, **kwargs):
        block = get_module(self.model, block_name)
        shard_writer = ShardWriter.get_shard_writer()

        tied_weights_keys = getattr(self.model, "_tied_weights_keys", [])
        if tied_weights_keys is None:
            tied_weights_keys = []
        if isinstance(tied_weights_keys, dict):
            tied_weights_values = list(tied_weights_keys.values())
        else:
            tied_weights_values = list(tied_weights_keys)
        tied_weights_layers = [".".join(val.split(".")[:-1]) for val in tied_weights_values]  # rm weight/bias
        # In fact, we should detect whether it is is_separate_lm_head, to simplify, we don't do it
        if getattr(self.compress_context, "formats", None) and self.compress_context.formats[0].is_gguf():
            lm_head_name = get_lm_head_name(self.model)
            if lm_head_name is not None:
                tied_weights_layers.append(lm_head_name)

        materialize_model_(block)
        for name, m in block.named_modules():
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer(m.global_name)
            elif (
                not any(m.children())
                and len(m.state_dict()) > 0
                and m.global_name not in tied_weights_layers
                and self.compress_context.is_immediate_saving
            ):
                set_module(self.model, m.global_name, copy.deepcopy(m))
                if self.compress_context.is_immediate_saving:
                    shard_writer.write(name=m.global_name)
                    copied_m = get_module(self.model, m.global_name)
                    copied_m.to("meta")
                m.to("meta")

        # Move remaining GPU tensors to CPU; offload to disk if low_cpu_mem_usage.
        # This mirrors _quantize_via_rtn_blockwise's post-block cleanup.
        if not self.compress_context.is_immediate_saving:
            mv_module_from_gpu(block)
        else:
            # Save once at block scope to capture tensors that are not saved
            # in per-layer branch (e.g., custom module-level params/buffers).
            shard_writer.write(name=block_name)
            block.to("meta")

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
        if self.compress_context.is_immediate_packing and self.compress_context.formats[0].is_gguf():
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
        self.seqlen = config.seqlen
        self.nsamples = config.nsamples
        self.batch_dim = config.batch_dim
        self.data_type = config.data_type
        self.group_size = config.group_size
        self.infer_bs_coeff = config.infer_bs_coeff

        self.enable_alg_ext = True

    def quantize_block(self, block_name: str, input_ids: Union[list[torch.Tensor], dict], input_others: dict, **kwargs):
        block = get_module(self.model, block_name)
        materialize_model_(block)
        block.to("cpu")

        block = convert_module_to_hp_if_necessary(
            block, dtype=self.model_context.amp_dtype, device=self.compress_context.device
        )
        update_block_global_scale_if_needed(block, self.data_type, self.group_size)
        self._register_act_max_hook(block)
        if is_auto_device_mapping(self.compress_context.device_map) and len(self.compress_context.device_list) > 1:
            set_auto_device_map_for_block_with_tuning(
                block,
                self.compress_context.device_map,
                input_ids,
                self.compress_context.low_gpu_mem_usage,
                self.batch_size,
                self.compress_context.device,
            )
        # Dispatch model if needed
        if len(self.compress_context.device_list) > 1:
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module

            for _, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                add_hook_to_module(m, hook, True)
        else:
            block = block.to(self.compress_context.device)
        input_ids = self._get_block_outputs(
            block,
            input_ids,
            input_others,
            self.batch_size * self.infer_bs_coeff,
        )

        if len(self.compress_context.device_list) > 1:
            accelerate.hooks.remove_hook_from_submodules(block)

        if self.config.is_act_nv_fp or self.config.is_static_afp8:
            # enable moe experts act_max automatic generation for Linear
            set_amax_for_all_moe_layers(block, attr_name="act_max")
        # Normalize imatrix and quantize layers
        if self.compress_context.low_gpu_mem_usage:
            block.to("cpu")
            clear_memory(device_list=self.compress_context.device_list)

        for name, m in block.named_modules():
            # fix issue: Ling-flash-2.0-q2_k_s fail infer on cuda but well on cpu
            # https://huggingface.co/Intel/Ling-flash-2.0-gguf-q2ks-mixed-AutoRound/discussions/1
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer(m.global_name)

        mv_module_from_gpu(block)

    @torch.no_grad()
    def _get_block_outputs(
        self,
        block: torch.nn.Module,
        input_ids: torch.Tensor | list[torch.Tensor],
        input_others: torch.Tensor | dict,
        bs: int,
        save_output: bool = True,
    ):
        """Compute the output of a given block of the model for a given input.

        Args:
        block: The block of the model.
        input_ids: The input tensor containing tokenized input ids.
        input_others: A dictionary containing additional input data.
        bs: The batch size for computing the output.
        device: The device for computation.
        cache_device: The device for storing the output.
        batch_dim: The batch dimension of the output tensor.

        Returns:
        The output tensor of the block.
        """

        self.block_forward = block_forward

        output = []
        nsamples = len(input_ids)
        for i in range(0, nsamples, bs):
            end_index = min(nsamples, i + bs)
            indices = torch.arange(i, end_index).to(torch.long)
            tmp_input_ids, tmp_input_others = self._sampling_inputs(
                input_ids,
                input_others,
                indices,
                self.seqlen,
                self.batch_dim,
                share_cache_keys=self.model_context.shared_cache_keys,
            )
            tmp_output = self.block_forward(
                block,
                tmp_input_ids,
                tmp_input_others,
                self.model_context.amp,
                self.model_context.amp_dtype,
                self.compress_context.device,
            ).to(self.compress_context.cache_device)
            if save_output:
                if self.batch_size == 1:
                    output.append(tmp_output)
                else:
                    output.extend(list(torch.split(tmp_output, 1, dim=self.batch_dim)))
        if self.compress_context.low_gpu_mem_usage:
            clear_memory(device_list=self.compress_context.device_list)

        return output

    @classmethod
    @torch.no_grad()
    def _sampling_inputs(
        cls,
        input_ids: Union[list[torch.Tensor], dict],
        input_others: dict,
        indices: list[int] | torch.Tensor,
        seqlen: int,
        batch_dim: int = 0,
        share_cache_keys: tuple = (),
    ):
        """Samples inputs based on the given indices and sequence length.

        Args:
        input_ids: The list of input tensor containing  input_ids.
        input_others: A dictionary containing other input data.
        indices: The indices to sample from the input.
        seqlen: The sequence length.

        Returns:
        current_input_ids: The sampled input IDs.
        current_input_others: The sampled other input data.
        """
        if isinstance(input_ids, list):
            current_input_ids = [input_ids[i] for i in indices]
            current_input_ids = torch.cat(current_input_ids, dim=batch_dim)
        elif isinstance(input_ids, dict):
            current_input_ids = defaultdict(list)
            for k in input_ids.keys():
                current_input_ids[k].extend([input_ids[k][i] for i in indices])
                current_input_ids[k] = torch.cat(current_input_ids[k], dim=batch_dim)

        current_input_others = {"positional_inputs": input_others["positional_inputs"]}
        for key in input_others.keys():
            if "positional_inputs" in key:
                continue
            # Shared cache keys (e.g. position_embeddings, position_ids, cache_position) are stored
            # directly as-is (not wrapped in a per-sample list) when batch_size > 1.  Indexing such
            # values by sample index would incorrectly decompose them (e.g. (cos, sin)[0] == cos).
            # Always pass them through unchanged.
            if key in share_cache_keys or isinstance(input_others[key], (str, bool, type(None))):
                current_input_others[key] = input_others[key]
            elif input_others[key] is not None:
                current_input_others[key] = [input_others[key][i] for i in indices]
                if len(indices) == 1:
                    current_input_others[key] = current_input_others[key][0]
                else:
                    try:
                        current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                    except TypeError as err:
                        logger.warning_once("Please check the model cache inputs or try setting batch_size to 1.")
            else:
                current_input_others[key] = None

        return current_input_ids, current_input_others
