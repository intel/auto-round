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
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Optional, Union

import accelerate
import torch
from torch import autocast

from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig
from auto_round.algorithms.quantization.base import BaseQuantizers
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
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.sign_sgd import SignSGD
from auto_round.utils import (
    clear_memory,
    compile_func,
    convert_module_to_hp_if_necessary,
    get_module,
    htcore,
    is_auto_device_mapping,
    is_hpex_available,
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


class ARQuantizer(BaseQuantizers):

    def __init__(self, config: AutoRoundConfig):
        super().__init__(config)
        self.attention_mask = []

        self.iters = config.iters
        self.lr = config.lr
        self.minmax_lr = config.minmax_lr
        self.lr_scheduler = config.lr_scheduler
        self.seqlen = config.seqlen
        self.nsamples = config.nsamples
        self.batch_size = config.batch_size
        self.batch_dim = config.batch_dim
        self.momentum = config.momentum
        self.infer_bs_coeff = config.infer_bs_coeff
        self.enable_minmax_tuning = config.enable_minmax_tuning
        self.enable_norm_bias_tuning = config.enable_norm_bias_tuning
        self.gradient_accumulate_steps = config.gradient_accumulate_steps
        self.enable_alg_ext = config.enable_alg_ext
        self.not_use_best_mse = config.not_use_best_mse
        self.enable_quanted_input = config.enable_quanted_input
        self.dynamic_max_gap = config.dynamic_max_gap

        self.optimizer = self._get_optimizer(optimizer=config.optimizer)
        self.wrapper_block = wrapper_block

    def post_init(self):
        super().post_init()
        if self.enable_alg_ext:
            try:
                logger.warning_once("using algorithm extension for quantization.")
                from auto_round.alg_ext import wrapper_autoround

                wrapper_autoround(self.quantizer)
            except (ImportError, ModuleNotFoundError):
                logger.error("algorithm extension import error, fallback to default mode")

    @torch.no_grad()
    def _get_block_forward_func(self, name: str) -> Callable:
        """Gets the forward function.

        Args:
            name (str): The name of the function.
        Returns:
            function: The forward function.
        """

        def post_process_cache_data(batch_size, data, data_name):
            """
            Processes store data for batch handling, reshaping if necessary.

            Args:
                batch_size (int): The size of the batch.
                data: The data value to store, potentially for caching.
                data_name (str): Name of the data.

            Returns:
                Processed data or None
            """
            new_data = data
            if batch_size <= 1:
                return new_data
            if data_name in self.model_context.shared_cache_keys:
                return None
            if "alibi" in data_name:
                if isinstance(data, torch.Tensor):
                    alibi = data
                    alibi = alibi.reshape(batch_size, -1, alibi.shape[1], alibi.shape[2])
                    new_data = alibi
            return new_data

        def forward(m, hidden_states=None, *positional_inputs, **kwargs):
            """Rewrite forward function, process and collect input data.

            Args:
                hidden_states (torch.Tensor): The hidden states tensor.
                *positional_inputs: Variable number of positional arguments.
                **kwargs: Variable number of keyword arguments.

            Returns:
                NotImplementedError: Getting the first layer inputs and then raise the error to save runtime.
            """
            if name not in self.inputs:
                self.inputs[name] = {}
                init_cache(positional_inputs, self.inputs[name])

            if self.batch_dim is None:
                self.batch_dim = 0
                if hidden_states is not None and self.batch_size > 1:
                    if hidden_states.shape[0] > self.batch_size:
                        self.batch_dim = 1
                        if len(hidden_states.shape) > 1 and hidden_states.shape[1] > self.batch_size:
                            logger.error(
                                "this model has not been supported, "
                                "please raise an issue in https://github.com/intel/auto-round/issues"
                                " or try to set the `batch_size` to 1 and "
                                "`gradient_accumulate_steps` to your current batch size."
                            )
                            exit(-1)

            if hidden_states is not None:
                kwargs["hidden_states"] = hidden_states

            for key in kwargs.keys():
                if (
                    isinstance(kwargs[key], torch.Tensor)
                    or isinstance(kwargs[key], list)
                    or isinstance(kwargs[key], tuple)
                ):
                    if key not in self.inputs[name].keys():  # initialization
                        data = to_device(kwargs[key], device=torch.device("cpu"))
                        if data is None or (self.batch_size > 1 and key in self.model_context.shared_cache_keys):
                            self.inputs[name][key] = data
                            continue
                        if self.batch_size <= 1:
                            self.inputs[name][key] = [data]
                        else:
                            data = post_process_cache_data(self.batch_size, data, key)
                            self.inputs[name][key] = list(torch.split(data, 1, dim=self.batch_dim))
                    else:  # append cache inputs
                        new_data = post_process_cache_data(self.batch_size, kwargs[key], key)
                        if new_data is None:  # shareable args or NoneType
                            continue
                        new_data = to_device(new_data, device=torch.device("cpu"))
                        if self.batch_size <= 1:
                            self.inputs[name][key].append(new_data)
                        else:
                            self.inputs[name][key].extend(list(torch.split(new_data, 1, dim=self.batch_dim)))
                elif isinstance(kwargs[key], (str, bool, type(None))):
                    if key not in self.inputs[name].keys():
                        self.inputs[name][key] = kwargs[key]
                else:
                    # Parameters not to be cached
                    if check_skippable_keywords(key):
                        logger.warning_once(
                            f"Please note that '{key}' key" " is not currently used in quantization fine-tuning."
                        )
            reset_params(self.inputs[name])

            if self._should_stop_cache_forward(name):
                raise NotImplementedError
            else:
                if hidden_states is not None:
                    kwargs.pop("hidden_states")
                    return m.orig_forward(hidden_states, *positional_inputs, **kwargs)
                else:
                    # Currently only for Llama-3.2-Vision-Instruct Series
                    return m.orig_forward(*positional_inputs, **kwargs)

        return forward

    def normalize_decoding_layer_inputs_(self, decoding_layer_inputs: list[tuple[tuple[Any, dict[str, Any]]]]):
        """
        Processes and stores decoding layer inputs for block quantization.

        This function iterates through a list of captured decoding layer calls,
        replaying them through a fake decoding layer to extract and store the
        inputs required for the decoding block in `self.inputs`. This effectively
        "normalizes" the inputs by making them accessible in a consistent format
        for subsequent quantization steps.

        Args:
            decoding_layer_inputs:
                A list of entries captured by a forward hook on the decoding layer.
                Each element is expected to be a tuple whose first item is
                `(args, kwargs)`, where `args` are the positional arguments and
                `kwargs` are the keyword arguments seen during the original
                forward pass.

                The capture hook look like:

                    def input_capture_hook(module, *args, **kwargs):
                        _all_module_input[module._global_name].append((args, kwargs))
        """
        first_block_name = self.quant_block_list[0][0]

        class _FakeDecodingLayer(torch.nn.Module):

            def forward(self, *args, **kwargs):
                return args, kwargs

        fake_layer = _FakeDecodingLayer()
        fake_layer.orig_forward = fake_layer.forward
        fake_layer.forward = partial(self._get_block_forward_func(first_block_name), fake_layer)

        self.inputs = {}
        self.last_cache_name = None
        for step_input in decoding_layer_inputs:
            args, kwargs = step_input[0]
            fake_layer(*args, **kwargs)

    def _get_current_output(self, output: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
        current_output = [output[x] for x in indices]
        current_output = torch.cat(current_output, dim=self.batch_dim)
        return current_output

    def _get_current_q_output(
        self,
        block: torch.nn.Module,
        input_ids: list[torch.Tensor],
        input_others: dict,
        indices: list[int],
        device: str,
        cache_device: str = "cpu",
    ) -> torch.Tensor:
        current_input_ids, current_input_others = self._sampling_inputs(
            input_ids,
            input_others,
            indices,
            seqlen=self.seqlen,
            batch_dim=self.batch_dim,
            share_cache_keys=self.model_context.shared_cache_keys,
        )
        output_q = self.block_forward(
            block, current_input_ids, current_input_others, self.model_context.amp, self.model_context.amp_dtype, device
        )
        return output_q.to(cache_device)

    def _get_current_num_elm(
        self,
        input_ids: list[torch.Tensor],
        indices: list[int],
    ) -> int:
        current_input_ids = [input_ids[i] for i in indices]
        return sum(id.numel() for id in current_input_ids)

    def _get_non_zero_cnt(self, tensor: list[torch.Tensor], indices: list[int]) -> int:
        current_tensors = [tensor[i] for i in indices]
        non_zero_cnt = 0
        for t in current_tensors:
            non_zero_cnt += torch.count_nonzero(t).item()
        return non_zero_cnt

    def _get_loss(
        self,
        output_q: torch.Tensor,
        current_output: torch.Tensor,
        indices: torch.Tensor,
        mse_loss: Callable,
        device: Union[str, torch.device] = "cpu",
    ):
        autocast_ctx = (
            nullcontext()
            if self.model_context.amp
            else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
        )
        if self.attention_mask:
            tmp_attention_mask = [self.attention_mask[i] for i in indices]
            tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
            tmp_attention_mask.unsqueeze_(-1)

            with autocast_ctx:
                loss = mse_loss(  # pylint: disable=not-callable
                    (output_q * tmp_attention_mask).to(torch.float32),
                    (current_output * tmp_attention_mask).to(torch.float32),
                )
        else:
            with autocast_ctx:
                loss = mse_loss(  # pylint: disable=not-callable
                    output_q.to(torch.float32), current_output.to(torch.float32)
                )

        return loss

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

        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device)

        if auto_offload:
            # card_0_in_high_risk indicates that card_0 memory is already in high usage (90%) w/o any weights
            # loss_device is used to calculate loss on the second device if available and card_0_in_high_risk
            if is_auto_device_mapping(self.compress_context.device_map) and len(self.compress_context.device_list) > 1:
                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    block,
                    self.compress_context.device_map,
                    input_ids,
                    self.compress_context.low_gpu_mem_usage,
                    self.batch_size,
                    device,
                )
            else:
                block = block.to(device)
                card_0_in_high_risk, loss_device = False, device
        else:
            card_0_in_high_risk, loss_device = False, device

        if len(self.compress_context.device_list) > 1 and auto_offload:
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
                self.compress_context.cache_device,
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
                self.compress_context.cache_device,
            )
            hook_handles = self._register_act_max_hook(block)
            if hook_handles:
                self._get_block_outputs(
                    block,
                    q_input if q_input is not None else input_ids,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    device,
                    self.compress_context.cache_device,
                    save_output=False,
                )

            for handle in hook_handles:
                handle.remove()

        if q_input is not None:
            if input_ids is not q_input:
                clear_memory(input_ids, device_list=self.compress_context.device_list)
            else:
                clear_memory(device_list=self.compress_context.device_list)
            input_ids = q_input

        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
        )
        # Call this before quantization and after applying the block wrapper.
        if self.config.is_nv_fp:  # enable qkv and moe structure global_scale fuse.
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
        setup_ddp_if_needed_(self, block, self.compress_context.device_list)
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

                if self.compress_context.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.5, device_list=self.compress_context.device_list)

                self._scale_loss_and_backward(scaler, loss)

                if self.compress_context.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.8, device_list=self.compress_context.device_list)

            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(block, self.compress_context.cache_device)
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)

                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(block, self.compress_context.cache_device)

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

        if self.compress_context.low_gpu_mem_usage:
            clear_memory(device_list=self.compress_context.device_list)  # clear cached memory during training
        if len(unquantized_layer_names) != 0:
            logger.info(f"{unquantized_layer_names} have not been quantized")
        with torch.no_grad():
            unwrapper_block(block, best_params)

        if self.config.is_act_nv_fp:
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        if self.enable_quanted_input:
            q_outputs = self._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.batch_size * self.infer_bs_coeff,
                device,
                cache_device=self.compress_context.cache_device,
            )

            if len(self.compress_context.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)

            clear_memory(input_ids, device_list=self.compress_context.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return q_outputs, output
        else:
            if len(self.compress_context.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)
            clear_memory(input_ids, device_list=self.compress_context.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return None, output

    def quantize_layer(self, layer_name: str, inputs: torch.Tensor, q_inputs: torch.Tensor = None, device: str = "cpu"):
        """Quantize a specific layer of the model using the provided inputs.

        Args:
            layer_name (str): The name of the layer to quantize.
            inputs (torch.Tensor): Input data for quantization.
            q_inputs (torch.Tensor, optional): Quantized input data. Defaults to None.
            device (torch.device, optional): The device to use for quantization. Defaults to torch.device("cpu").

        Returns:
            None
        """
        logger.info(f"quantizing layer {layer_name}")
        layer = get_module(self.model, layer_name)
        if hasattr(layer, "tuning_device"):
            device = layer.tuning_device

        layer = layer.to(device)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(layer.weight.dtype)
            if q_inputs is not None:
                q_inputs[i] = q_inputs[i].to(layer.weight.dtype)

        if self.act_bits <= 8 and check_need_act_calibration(
            self.act_dynamic,
            self.act_data_type,
            self.act_bits,
            self.static_kv_dtype,
            self.static_attention_dtype,
        ):
            tmp_inputs = q_inputs if q_inputs is not None else inputs
            hook_handles = self._register_act_max_hook(layer)
            with torch.no_grad():
                for input in tmp_inputs:
                    layer(input)
            for handle in hook_handles:
                handle.remove()

        wrapper_linear = WrapperLinear(
            layer,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_torch_compile=self.enable_torch_compile,
            device=device,
        ).to(device)
        round_params = []
        minmax_params = []
        for key in wrapper_linear.params.keys():
            if "min" in key or "max" in key:
                minmax_params.append(wrapper_linear.params[key])
            else:
                round_params.append(wrapper_linear.value)
        if len(round_params) + len(minmax_params) <= 0:
            dump_info = f"quantized {layer_name}"
            logger.info(dump_info)
            with torch.no_grad():
                unwrapper_layer(self.model, wrapper_linear, layer_name, {})
            mv_module_from_gpu(layer)

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": minmax_lr}], lr=lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        nsamples = len(inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        gradient_accumulate_steps = self.batch_size  # Force to low gpu

        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        batch_size = 1  # Force to low gpu
        global_batch_size = self.batch_size * gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        if gradient_accumulate_steps != 1 and not self.attention_mask:
            whole_indices = torch.arange(global_batch_size)
            if q_inputs is not None:
                num_elm = self._get_current_num_elm(q_inputs, whole_indices)
            else:
                num_elm = self._get_current_num_elm(inputs, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)

        for i in range(self.iters):
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if self.attention_mask:
                num_elm = self._get_non_zero_cnt(self.attention_mask, global_indices)

            for tmp_step in range(gradient_accumulate_steps):
                indices = global_indices[tmp_step * batch_size : (tmp_step + 1) * batch_size]
                if q_inputs is not None:
                    current_input = [q_inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = [inputs[i] for i in indices]
                    org_input = torch.cat(org_input, dim=0).to(device)
                else:
                    current_input = [inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = current_input
                with torch.no_grad():
                    current_output = layer(org_input)
                autocast_ctx = (
                    nullcontext()
                    if not self.amp
                    else autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype)
                )
                if self.attention_mask:
                    tmp_attention_mask = [self.attention_mask[i] for i in indices]
                    tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
                    tmp_attention_mask.unsqueeze_(-1)

                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            (output_q * tmp_attention_mask).to(torch.float32),
                            (current_output * tmp_attention_mask).to(torch.float32),
                        )

                else:
                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            output_q.to(torch.float32),
                            current_output.to(torch.float32),  # mul 1.0 will copy the output
                        )

                num_elm = 1 if num_elm <= 0 else num_elm
                total_loss += loss.item() / num_elm

                self._scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(wrapper_linear, self.cache_device)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(wrapper_linear, self.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        with torch.no_grad():
            unwrapper_layer(self.model, wrapper_linear, layer_name, best_params)
        mv_module_from_gpu(layer)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        logger.info(dump_info)

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

        self.block_forward = (
            compile_func(block_forward, self.device) if self.compress_context.enable_torch_compile else block_forward
        )

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
                block, tmp_input_ids, tmp_input_others, self.model_context.amp, self.model_context.amp_dtype, device
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
            if (key not in share_cache_keys or len(indices) == 1) and not isinstance(
                input_others[key], (str, bool, type(None))
            ):
                current_input_others[key] = None
                if input_others[key] is not None:
                    current_input_others[key] = [input_others[key][i] for i in indices]
                    if len(indices) == 1:
                        current_input_others[key] = current_input_others[key][0]
                    else:
                        try:
                            current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                        except TypeError as err:
                            logger.warning_once("Please check the model cache inputs or try setting batch_size to 1.")
            else:
                current_input_others[key] = input_others[key]

        return current_input_ids, current_input_others

    def _get_optimizer(self, optimizer: Any):
        """Returns the specified optimizer. In SignRound, we fix the optimizer.

        Args:
        optimizer: The optimizer to be used.

        Returns:
        The specified optimizer.
        """
        if optimizer is not None:
            logger.warning_once(
                "The optimizer setting in config will be ignored in AutoRound, using SignSGD as default."
            )
        return SignSGD

    def _get_scaler(self):
        """Returns scaler, in SignRound, no need to use scaler."""
        return None

    def _scale_loss_and_backward(self, scaler: Any, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss and performs backward pass.

        Args:
        scaler: The scaler to be used.
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        scale_loss.backward()
        if is_hpex_available():
            htcore.mark_step()
        return scale_loss

    def _step(self, scaler: Any, optimizer: Any, lr_schedule: Any):
        """Performs a step in the optimization process.

        Args:
        scaler: The scaler to be used.
        optimizer: The optimizer for the step.
        lr_schedule: The learning rate schedule.

        Returns:
        None
        """
        optimizer.step()
        # for hpu
        if is_hpex_available():
            htcore.mark_step()
        optimizer.zero_grad()
        lr_schedule.step()
