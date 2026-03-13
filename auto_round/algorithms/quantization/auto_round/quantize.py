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
from typing import Any, Callable, Optional, Union

import accelerate
import torch

from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig
from auto_round.compressors_new.utils import (
    IndexSampler,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    infer_bits_by_data_type,
    init_cache,
    is_nv_fp,
    reset_params,
    set_layer_config,
)
from auto_round.context.compress_context import CompressContext
from auto_round.context.model_context import ModelContext
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.utils import (
    clear_memory,
    convert_module_to_hp_if_necessary,
    is_auto_device_mapping,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    to_device,
)
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    get_major_device,
    parse_available_devices,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)
from auto_round.utils.distributed import setup_ddp_if_needed_
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block


class ARQuantizer:

    def __init__(self, config: AutoRoundConfig):
        self.config = AutoRoundConfig

    def quantize_block(
        self,
        block: torch.nn.Module,
        input_ids: Union[list[torch.Tensor], dict],
        input_others: dict,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload=True,
    ):
        """Quantize the weights of a given block of the model.

        Args:
        block: The block of the model to be quantized.
        input_ids: The input tensor containing tokenized input ids.
        input_others: A dictionary containing additional input data.
        q_input: The quantized input tensor.
        device: The device for quantization.

        Returns:
        Tuple: (q_outputs, output) if self.enable_quanted_input is True, else (None, output)
        """
        model_context = ModelContext.get_context()
        compress_context = CompressContext.get_context()

        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, model_context._amp_dtype, device)

        if auto_offload:
            # card_0_in_high_risk indicates that card_0 memory is already in high usage (90%) w/o any weights
            # loss_device is used to calculate loss on the second device if available and card_0_in_high_risk
            if is_auto_device_mapping(compress_context.device_map) and len(compress_context.device_list) > 1:
                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    block,
                    compress_context.device_map,
                    input_ids,
                    compress_context.low_gpu_mem_usage,
                    self.batch_size,
                    device,
                )
            else:
                block = block.to(device)
                card_0_in_high_risk, loss_device = False, device
        else:
            card_0_in_high_risk, loss_device = False, device

        if len(compress_context.device_list) > 1 and auto_offload:
            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                add_hook_to_module(m, hook, True)

        if q_input is None:
            hook_handles = self._register_act_max_hook(block)

            output = self._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.batch_size * self.infer_bs_coeff,
                device,
                compress_context.cache_device,
            )

            for handle in hook_handles:
                handle.remove()
        else:
            output = self._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.batch_size * self.infer_bs_coeff,
                device,
                compress_context.cache_device,
            )
            hook_handles = self._register_act_max_hook(block)
            if hook_handles:
                self._get_block_outputs(
                    block,
                    q_input if q_input is not None else input_ids,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    device,
                    compress_context.cache_device,
                    save_output=False,
                )

            for handle in hook_handles:
                handle.remove()

        if q_input is not None:
            if input_ids is not q_input:
                clear_memory(input_ids, device_list=compress_context.device_list)
            else:
                clear_memory(device_list=compress_context.device_list)
            input_ids = q_input

        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.enable_torch_compile,
            device=device,
        )
        # Call this before quantization and after applying the block wrapper.
        if is_nv_fp(self.data_type):  # enable qkv and moe structure global_scale fuse.
            from auto_round.data_type.utils import update_fused_layer_global_scales

            modules = block.modules()
            for module in modules:
                update_fused_layer_global_scales(module)
        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                for key in m.params.keys():
                    if "min" in key or "max" in key:
                        minmax_params.append(m.params[key])
                    else:
                        round_params.append(m.params[key])

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)
        is_adam = "adam" in self.__class__.__name__.lower()

        extra_kwargs = {} if is_adam else {"momentum": self.momentum}

        if self.enable_minmax_tuning:
            params = [
                {"params": round_params},
                {"params": minmax_params, "lr": minmax_lr},
            ]
        else:
            params = round_params

        optimizer = self.optimizer(
            params,
            lr=lr,
            weight_decay=0,
            **extra_kwargs,
        )

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            unwrapper_block(block, {})
            mv_module_from_gpu(block)
            return output, output

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        if isinstance(input_ids, dict):  # input_ids of Flux is dict
            nsamples = len(input_ids["hidden_states"])
        else:
            nsamples = len(input_ids)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        num_elm = 1
        mse_reduction = "mean"
        if self.gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_params = {}
        total_loss = 0
        global_batch_size = self.batch_size * self.gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # We assume the block input and output shape is same
        if self.gradient_accumulate_steps != 1 and not self.attention_mask:
            whole_indices = torch.arange(global_batch_size)
            num_elm = self._get_current_num_elm(input_ids, whole_indices)
        setup_ddp_if_needed_(self, block, self.device_list)
        index_sampler = IndexSampler(nsamples, global_batch_size)
        batch_size = self.batch_size
        for i in range(self.iters):
            if self.enable_alg_ext and self.data_type.endswith("dq"):
                for n, m in block.named_modules():
                    m.cur_iter = i
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if self.attention_mask:
                num_elm = self._get_non_zero_cnt(self.attention_mask, global_indices)

            for tmp_step in range(self.gradient_accumulate_steps):
                indices = global_indices[tmp_step * batch_size : (tmp_step + 1) * batch_size]
                current_output = self._get_current_output(output, indices)
                current_output = to_device(current_output, loss_device)
                output_q = self._get_current_q_output(block, input_ids, input_others, indices, device, loss_device)
                loss = self._get_loss(output_q, current_output, indices, mse_loss, device)
                num_elm = 1 if num_elm <= 0 else num_elm
                total_loss += loss.item() / num_elm

                if self.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.5, device_list=compress_context.device_list)

                self._scale_loss_and_backward(scaler, loss)

                if self.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.8, device_list=compress_context.device_list)

            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(block, compress_context.cache_device)
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)

                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(block, compress_context.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if self.iters > 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
            )
        else:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                "layers in the block"
            )

        if self.low_gpu_mem_usage:
            clear_memory(device_list=compress_context.device_list)  # clear cached memory during training
        if len(unquantized_layer_names) != 0:
            logger.info(f"{unquantized_layer_names} have not been quantized")
        with torch.no_grad():
            unwrapper_block(block, best_params)

        if is_nv_fp(self.act_data_type):
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        if self.enable_quanted_input:
            q_outputs = self._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.batch_size * self.infer_bs_coeff,
                device,
                cache_device=compress_context.cache_device,
            )

            if len(compress_context.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)

            clear_memory(input_ids, device_list=compress_context.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return q_outputs, output
        else:
            if len(compress_context.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)
            clear_memory(input_ids, device_list=compress_context.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return None, output

    @torch.no_grad()
    def _get_block_outputs(
        self,
        block: torch.nn.Module,
        input_ids: torch.Tensor | list[torch.Tensor],
        input_others: torch.Tensor | dict,
        bs: int,
        device: Union[str, torch.device],
        cache_device: Union[str, torch.device],
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
        output = []
        nsamples = len(input_ids)
        for i in range(0, nsamples, bs):
            end_index = min(nsamples, i + bs)
            indices = torch.arange(i, end_index).to(torch.long)
            tmp_input_ids, tmp_input_others = self._sampling_inputs(
                input_ids, input_others, indices, self.seqlen, self.batch_dim, share_cache_keys=self.shared_cache_keys
            )
            tmp_output = self.block_forward(
                block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device
            ).to(cache_device)
            if save_output:
                if self.batch_size == 1:
                    output.append(tmp_output)
                else:
                    output.extend(list(torch.split(tmp_output, 1, dim=self.batch_dim)))
        if self.low_gpu_mem_usage:
            clear_memory(device_list=self.device_list)

        return output
