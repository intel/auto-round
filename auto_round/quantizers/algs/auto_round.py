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
import time
import traceback
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from torch import autocast
from tqdm import tqdm

from auto_round.compressors.utils import (
    IndexSampler,
    check_need_act_calibration,
    collect_best_params,
    immediate_saving,
    is_nv_fp,
)
from auto_round.logger import logger
from auto_round.quantizers.algs.base import AlgsBaseQuantizer
from auto_round.quantizers.utils import (
    get_non_zero_cnt,
    get_quantized_layer_names_outside_blocks,
    preprocess_block_inputs,
    quantize_embedding_layer,
    register_act_max_hook,
)
from auto_round.utils import (
    check_to_quantized,
    clear_memory,
    convert_fp8_layer_to_linear,
    get_block_names,
    get_module,
    is_auto_device_mapping,
    is_fp8_linear,
    is_fp8_model,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
)
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    set_auto_device_map_for_block_with_tuning,
)
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


class ARQuantizer(AlgsBaseQuantizer):

    def __init__(self, compressor: "BaseCompressor"):
        super().__init__(compressor)

    def pre_quantize(self, *args, **kwargs):
        return super().pre_quantize(*args, **kwargs)

    def quantize(self, *args, **kwargs):
        if bool(self.compressor.quant_block_list):
            all_blocks = self.compressor.quant_block_list
        else:
            all_blocks = get_block_names(self.compressor.model)

        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.compressor.model, self.compressor.layer_config

        if self.compressor.amp and self.compressor.model.dtype != self.compressor.amp_dtype:
            self.compressor.model = self.compressor.model.to(self.compressor.amp_dtype)

        layer_names = get_quantized_layer_names_outside_blocks(
            model=self.compressor.model,
            layer_config=self.compressor.layer_config,
            supported_types=self.compressor.supported_types,
            quant_block_list=self.compressor.quant_block_list,
        )
        start_time = time.time()
        all_first_block_names = [block[0] for block in all_blocks]
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")

        # TODO: refactor this
        all_inputs = self.compressor.try_cache_inter_data_gpucpu(
            all_first_block_names, self.compressor.nsamples, layer_names=layer_names
        )

        is_quantized_embedding = quantize_embedding_layer(
            model=self.compressor.model,
            layer_config=self.compressor.layer_config,
            scale_dtype=self.compressor.data_type,
            disable_opt_rtn=self.compressor.disable_opt_rtn,
            device=self.compressor.device,
            device_list=self.compressor.device_list,
        )
        clear_memory(device_list=self.compressor.device_list)
        all_q_inputs = None
        if is_quantized_embedding:
            all_inputs = copy.deepcopy(self.compressor.inputs)
            clear_memory(self.compressor.inputs, device_list=self.compressor.device_list)
            # TODO: refactor this
            all_q_inputs = self.compressor.try_cache_inter_data_gpucpu(
                all_first_block_names, self.compressor.nsamples, layer_names=layer_names
            )
        self.compressor.model = mv_module_from_gpu(self.compressor.model)
        clear_memory(device_list=self.compressor.device_list)

        if hasattr(self.compressor.model, "hf_device_map") and len(self.compressor.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(
                self.compressor.model
            )  # self.compressor.model.hf_device_map has not been changed
        logger.info("caching done")
        if len(all_blocks) > 1:
            pbar = tqdm(range(0, sum([len(i) for i in all_blocks]), self.compressor.nblocks))
        else:
            pbar = tqdm(range(0, len(all_blocks[0]), self.compressor.nblocks))  # move the alg warning outside pbar

        for block_names in all_blocks:
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            q_inputs = None
            if all_q_inputs is not None:
                q_inputs = all_q_inputs[block_names[0]]
                all_q_inputs.pop(block_names[0])

            # TODO: refactor this
            inputs, q_inputs = self.compressor._update_inputs(inputs, q_inputs)

            clear_memory(self.compressor.inputs, device_list=self.compressor.device_list)

            if "input_ids" in inputs.keys():
                total_samples = len(inputs["input_ids"])
                if total_samples < self.compressor.batch_size:
                    self.compressor.batch_size = total_samples
                    logger.warning(f"force the train batch size to {total_samples}")

            self._quantize_blocks(
                self.compressor.model,
                inputs,
                block_names,
                q_input=q_inputs if q_inputs is not None else None,
                nblocks=self.compressor.nblocks,
                device=self.compressor.device,
                pbar=pbar,
            )
            if self.compressor.immediate_packing and len(self.compressor.formats) != 1:
                raise ValueError(
                    f"Expected exactly one packing format when 'immediate_packing' is True, "
                    f"but got {len(self.compressor.formats)} formats."
                )
        pbar.set_description("Quantizing done")
        pbar.close()
        self._quantize_layers(layer_names, all_inputs)

        if is_fp8_model(self.compressor.model):
            for n, m in self.compressor.model.named_modules():
                if is_fp8_linear(m):
                    new_layer = convert_fp8_layer_to_linear(m, self.compressor.amp_dtype, self.compressor.device).to(
                        "cpu"
                    )
                    set_module(self.compressor.model, n, new_layer)

        end_time = time.time()
        cost_time = end_time - start_time
        logger.info(f"quantization tuning time {cost_time}")

        # Dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.compressor.model.named_modules():
            if isinstance(m, tuple(self.compressor.supported_types)):
                if check_to_quantized(m):
                    quantized_layers.append(n)
                else:
                    unquantized_layers.append(n)
            elif hasattr(m, "scales") or hasattr(m, "scale"):  ##packing_immediately
                quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            summary_info += f",  {unquantized_layers} have not been quantized"
        logger.info(summary_info)

        self.compressor.quantized = True
        return self.compressor.model, self.compressor.layer_config

    def _quantize_layers(self, layer_names: list, layer_inputs: dict) -> None:
        """Quantizes specified layers based on inputs and configuration.

        Args:
            layer_names (list): list of layer names to quantize.
            layer_inputs (dict): Dictionary mapping layer names to input data.

        Returns:
            None
        """
        # TODO currently we take all the layers outside blocks as post block layers which is not optimal
        # if there is no input for layer, we use rtn

        for layer_name in copy.deepcopy(layer_names):
            if layer_name not in layer_inputs:
                if self.compressor.act_bits < 16 and not self.compressor.act_dynamic:
                    # Activation quantization requires collected inputs
                    msg_prefix = (
                        f"Activation max hook for layer '{layer_name}' is unavailable due to "
                        f"insufficient collected inputs. "
                    )
                    if "fp8_e5m2" in self.compressor.act_data_type:
                        logger.warning(msg_prefix + "Please notes that unit scale is used for this layer.")
                    else:
                        logger.warning(
                            msg_prefix + "Static activation quantization is not supported or ineffective, "
                            "Skipping quantization for this layer."
                        )
                        layer_names.remove(layer_name)
                        continue
                logger.info(f"using rtn to quantize {layer_name}")
                from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

                layer = get_module(self.compressor.model, layer_name)
                layer = layer.to(self.compressor.device)
                if is_fp8_linear(layer):
                    new_layer = convert_fp8_layer_to_linear(
                        layer, self.compressor.amp_dtype, self.compressor.device
                    ).to(self.compressor.device)
                    set_module(self.compressor.model, layer_name, new_layer)
                    layer = new_layer

                wrapper_layer = WrapperLinear(
                    layer,
                    enable_round_tuning=False,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_torch_compile=self.compressor.enable_torch_compile,
                    device=self.compressor.device,
                    disable_opt_rtn=self.compressor.disable_opt_rtn,
                )
                new_layer = wrapper_layer.unwrapper({})
                set_module(self.compressor.model, layer_name, new_layer)
                layer.cpu()
                layer_names.remove(layer_name)
        if len(layer_names) == 0:
            memory_monitor.update()
            memory_monitor.log_summary()
            return
        q_layer_inputs = None
        enable_quanted_input = self.compressor.enable_quanted_input
        has_gguf = False

        if hasattr(self.compressor, "formats"):
            has_gguf = any(format_.is_gguf() for format_ in self.compressor.formats)
        if has_gguf and self.compressor.immediate_packing:
            enable_quanted_input = False

        if (
            hasattr(self.compressor.model, "hf_device_map")
            and len(self.compressor.model.hf_device_map) > 1
            and enable_quanted_input
        ):
            dispatch_model(self.compressor.model, self.compressor.model.hf_device_map)

        if enable_quanted_input:
            logger.info("starting to cache layer inputs for %s, this may be quite slow ", layer_names)
            # TODO: refactor this
            q_layer_inputs = self.compressor.try_cache_inter_data_gpucpu(
                [], self.compressor.nsamples, layer_names=layer_names
            )
            if hasattr(self.compressor.model, "hf_device_map") and len(self.compressor.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.compressor.model
                )  # self.compressor.model.hf_device_map has not been changed
        if not self.compressor.immediate_saving:
            self.compressor.model = mv_module_from_gpu(self.compressor.model)
        clear_memory(device_list=self.compressor.device_list)
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.compressor.cache_device)
            q_layer_input = q_layer_inputs.get(layer_name, None) if q_layer_inputs is not None else None
            q_layer_input = to_device(q_layer_input, self.compressor.cache_device)
            self.quantize_layer(layer_name, layer_input, q_layer_input, device=self.compressor.device)
            if self.compressor.immediate_packing:
                self.compressor._immediate_pack(layer_name)

            if self.compressor.immediate_saving:
                m = get_module(self.compressor.model, layer_name)
                immediate_saving(self.compressor, m, name=layer_name, last_group=True)
            del layer_input
            clear_memory(q_layer_input, device_list=self.compressor.device_list)
            memory_monitor.log_summary()

    def _quantize_blocks(
        self,
        model: torch.nn.Module,
        inputs: dict,
        block_names: list,
        q_input: torch.Tensor = None,
        nblocks: int = 1,
        device: str = "cpu",
        pbar: tqdm = None,
    ):
        """Quantize and dequantize the weights of the specified blocks in the model.

        Args:
        model: The PyTorch model to be quantized.
        inputs: The input data for quantization.
        block_names: The names of the blocks to be quantized and dequantized.
        nblocks: The number of blocks to quantize and dequantize.
        device: The device for quantization and dequantization.

        Returns:
        None
        """
        clear_memory(device_list=self.compressor.device_list)
        for n, m in model.named_parameters():
            m.requires_grad_(False)

        input_ids, input_others = preprocess_block_inputs(
            inputs,
            device_list=self.compressor.device_list,
            first_input_name="input_ids",
            amp=self.compressor.amp,
            amp_dtype=self.compressor.amp_dtype,
            cache_device=self.compressor.cache_device,
            diffusion=self.compressor.diffusion,
        )

        if pbar is None:
            pbar = tqdm(range(0, len(block_names), nblocks))

        for i in range(0, len(block_names), nblocks):
            if i != 0:
                pbar.update(1)
            if nblocks == 1:
                n = block_names[i]
                pbar.set_description(f"Quantizing {n}")
                m = get_module(model, n)
            else:
                names = block_names[i : min(i + nblocks, len(block_names))]
                pbar.set_description(f"Quantizing [{i + 1}-{min(i + nblocks, len(block_names))}]/{len(block_names)}")
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            m.config = model.config if hasattr(model, "config") else None
            q_input, input_ids = self.quantize_block(
                m,
                input_ids,
                input_others,
                q_input=q_input,
                device=device,
            )
            if hasattr(model, "config"):
                del m.config
            if self.compressor.immediate_packing:
                for _, tmp_m in m.named_modules():
                    if not (hasattr(tmp_m, "bits") and check_to_quantized(tmp_m)):
                        continue
                    self.compressor._immediate_pack(tmp_m.tmp_name)

            if self.compressor.immediate_saving:
                last_group = (i + nblocks) >= len(block_names)
                immediate_saving(self.compressor, m, last_group=last_group)
        if pbar is not None:
            pbar.update(1)

        if not self.compressor.immediate_saving:
            self.compressor.model = mv_module_from_gpu(self.compressor.model)
        for n, m in self.compressor.model.named_modules():
            if hasattr(m, "name"):
                delattr(m, "name")

        del q_input
        del input_ids
        del input_others
        del inputs

        clear_memory(device_list=self.compressor.device_list)

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
        layer = get_module(self.compressor.model, layer_name)
        if hasattr(layer, "tuning_device"):
            device = layer.tuning_device

        layer = layer.to(device)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(layer.weight.dtype)
            if q_inputs is not None:
                q_inputs[i] = q_inputs[i].to(layer.weight.dtype)

        if self.compressor.act_bits <= 8 and check_need_act_calibration(
            self.compressor.act_dynamic,
            self.compressor.act_data_type,
            self.compressor.act_bits,
            self.compressor.static_kv_dtype,
            self.compressor.static_attention_dtype,
        ):
            tmp_inputs = q_inputs if q_inputs is not None else inputs
            hook_handles = register_act_max_hook(
                model=self.compressor.model,
                layer_config=self.compressor.layer_config,
                act_group_size=self.compressor.act_group_size,
                act_data_type=self.compressor.act_data_type,
            )
            with torch.no_grad():
                for input in tmp_inputs:
                    layer(input)
            for handle in hook_handles:
                handle.remove()

        wrapper_linear = WrapperLinear(
            layer,
            enable_minmax_tuning=self.compressor.enable_minmax_tuning,
            enable_torch_compile=self.compressor.enable_torch_compile,
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
                unwrapper_layer(self.compressor.model, wrapper_linear, layer_name, {})
            mv_module_from_gpu(layer)

        lr = torch.tensor(self.compressor.lr)
        minmax_lr = torch.tensor(self.compressor.minmax_lr)
        if self.compressor.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": minmax_lr}], lr=lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=lr, weight_decay=0)

        if self.compressor.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.compressor.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.compressor.lr_scheduler)
        nsamples = len(inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        scaler = self.compressor._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        gradient_accumulate_steps = self.compressor.batch_size  # Force to low gpu
        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        batch_size = 1  # Force to low gpu
        global_batch_size = self.compressor.batch_size * gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        if gradient_accumulate_steps != 1 and not self.compressor.attention_mask:
            whole_indices = torch.arange(global_batch_size)
            if q_inputs is not None:
                # Todo: refactor this
                num_elm = self.compressor._get_current_num_elm(q_inputs, whole_indices)
            else:
                num_elm = self.compressor._get_current_num_elm(inputs, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)

        for i in range(self.compressor.iters):
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if self.compressor.attention_mask:
                num_elm = get_non_zero_cnt(self.compressor.attention_mask, global_indices)

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
                    if not self.compressor.amp
                    else autocast(device_type=str(device).split(":")[0], dtype=self.compressor.amp_dtype)
                )
                if self.compressor.attention_mask:
                    tmp_attention_mask = [self.compressor.attention_mask[i] for i in indices]
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

                self.compressor.scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.compressor.not_use_best_mse:
                    best_params = collect_best_params(wrapper_linear, self.compressor.cache_device)
                    last_best_iter = i
            if self.compressor.not_use_best_mse and i == self.compressor.iters - 1:
                best_params = collect_best_params(wrapper_linear, self.compressor.cache_device)

            if not self.compressor.not_use_best_mse:
                if 0 < self.compressor.dynamic_max_gap <= i - last_best_iter:
                    break
            self.compressor._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.compressor.iters
        if not self.compressor.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        with torch.no_grad():
            unwrapper_layer(self.compressor.model, wrapper_linear, layer_name, best_params)
        mv_module_from_gpu(layer)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        logger.info(dump_info)

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
        if is_fp8_model(self.compressor.model):
            for n, m in block.named_modules():
                if is_fp8_linear(m):
                    new_layer = convert_fp8_layer_to_linear(m, self.compressor.amp_dtype, self.compressor.device).to(
                        device
                    )
                    set_module(block, n, new_layer)

        if auto_offload:
            # card_0_in_high_risk indicates that card_0 memory is already in high usage (90%) w/o any weights
            # loss_device is used to calculate loss on the second device if available and card_0_in_high_risk
            if is_auto_device_mapping(self.compressor.device_map) and len(self.compressor.device_list) > 1:
                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    block,
                    self.compressor.device_map,
                    input_ids,
                    self.compressor.low_gpu_mem_usage,
                    self.compressor.batch_size,
                    device,
                )
            else:
                block = block.to(device)
                card_0_in_high_risk, loss_device = False, device
        else:
            card_0_in_high_risk, loss_device = False, device

        if len(self.compressor.device_list) > 1 and auto_offload:
            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                add_hook_to_module(m, hook, True)

        if q_input is None:
            hook_handles = register_act_max_hook(
                model=self.compressor.model,
                layer_config=self.compressor.layer_config,
                act_group_size=self.compressor.act_group_size,
                act_data_type=self.compressor.act_data_type,
            )

            # TODO: refactor this part
            output = self.compressor._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.compressor.batch_size * self.compressor.infer_bs_coeff,
                device,
                self.compressor.cache_device,
            )

            for handle in hook_handles:
                handle.remove()
        else:
            # TODO: refactor this part
            output = self.compressor._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.compressor.batch_size * self.compressor.infer_bs_coeff,
                device,
                self.compressor.cache_device,
            )
            hook_handles = register_act_max_hook(
                model=self.compressor.model,
                layer_config=self.compressor.layer_config,
                act_group_size=self.compressor.act_group_size,
                act_data_type=self.compressor.act_data_type,
            )
            if hook_handles:
                # TODO: refactor this part
                self.compressor._get_block_outputs(
                    block,
                    q_input,
                    input_others,
                    self.compressor.batch_size * self.compressor.infer_bs_coeff,
                    device,
                    self.compressor.cache_device,
                    save_output=False,
                )

            for handle in hook_handles:
                handle.remove()

        if q_input is not None:
            if input_ids is not q_input:
                clear_memory(input_ids, device_list=self.compressor.device_list)
            else:
                clear_memory(device_list=self.compressor.device_list)
            input_ids = q_input

        quantized_layer_names, unquantized_layer_names = self.compressor.wrapper_block(
            block,
            self.compressor.enable_minmax_tuning,
            self.compressor.enable_norm_bias_tuning,
            enable_torch_compile=self.compressor.enable_torch_compile,
            device=device,
        )
        if is_nv_fp(self.compressor.data_type):  # enable qkv and moe structure global_scale fuse
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

        lr = torch.tensor(self.compressor.lr)
        minmax_lr = torch.tensor(self.compressor.minmax_lr)
        is_adam = "adam" in self.compressor.__class__.__name__.lower()

        extra_kwargs = {} if is_adam else {"momentum": self.compressor.momentum}

        if self.compressor.enable_minmax_tuning:
            params = [
                {"params": round_params},
                {"params": minmax_params, "lr": minmax_lr},
            ]
        else:
            params = round_params

        optimizer = self.compressor.optimizer(
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

        if self.compressor.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.compressor.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.compressor.lr_scheduler)

        if isinstance(input_ids, dict):  # input_ids of Flux is dict
            nsamples = len(input_ids["hidden_states"])
        else:
            nsamples = len(input_ids)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        num_elm = 1
        mse_reduction = "mean"
        if self.compressor.gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        scaler = self.compressor._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_params = {}
        total_loss = 0
        global_batch_size = self.compressor.batch_size * self.compressor.gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # We assume the block input and output shape is same
        if self.compressor.gradient_accumulate_steps != 1 and not self.compressor.attention_mask:
            whole_indices = torch.arange(global_batch_size)
            num_elm = self.compressor._get_current_num_elm(input_ids, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)
        batch_size = self.compressor.batch_size
        for i in range(self.compressor.iters):
            if self.compressor.enable_alg_ext and self.compressor.data_type.endswith("dq"):
                for n, m in block.named_modules():
                    m.cur_iter = i
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if self.compressor.attention_mask:
                num_elm = get_non_zero_cnt(self.compressor.attention_mask, global_indices)

            for tmp_step in range(self.compressor.gradient_accumulate_steps):
                indices = global_indices[tmp_step * batch_size : (tmp_step + 1) * batch_size]
                current_output = self._get_current_output(output, indices, self.compressor.batch_dim)
                current_output = to_device(current_output, loss_device)
                # TODO: refactor this
                output_q = self.compressor._get_current_q_output(
                    block, input_ids, input_others, indices, device, loss_device
                )
                # TODO: refactor this
                loss = self.compressor._get_loss(output_q, current_output, indices, mse_loss, device)
                num_elm = 1 if num_elm <= 0 else num_elm
                total_loss += loss.item() / num_elm

                if self.compressor.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.5, device_list=self.compressor.device_list)

                self.compressor._scale_loss_and_backward(scaler, loss)
                if self.compressor.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.8, device_list=self.compressor.device_list)

            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.compressor.not_use_best_mse:
                    best_params = collect_best_params(block, self.compressor.cache_device)
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)

                    last_best_iter = i
            if self.compressor.not_use_best_mse and i == self.compressor.iters - 1:
                best_params = collect_best_params(block, self.compressor.cache_device)

            if not self.compressor.not_use_best_mse:
                if 0 < self.compressor.dynamic_max_gap <= i - last_best_iter:
                    break
            self.compressor._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.compressor.iters
        if not self.compressor.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if self.compressor.iters > 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
            )
        else:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                "layers in the block"
            )

        if self.compressor.low_gpu_mem_usage:
            clear_memory(device_list=self.compressor.device_list)  # clear cached memory during training
        if len(unquantized_layer_names) != 0:
            logger.info(f"{unquantized_layer_names} have not been quantized")
        with torch.no_grad():
            unwrapper_block(block, best_params)

        if is_nv_fp(self.compressor.act_data_type):
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        if self.compressor.enable_quanted_input:
            # TODO: refactor this
            q_outputs = self.compressor._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.compressor.batch_size * self.compressor.infer_bs_coeff,
                device,
                cache_device=self.compressor.cache_device,
            )

            if len(self.compressor.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)

            clear_memory(input_ids, device_list=self.compressor.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return q_outputs, output
        else:
            if len(self.compressor.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)
            clear_memory(input_ids, device_list=self.compressor.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return None, output

    @staticmethod
    def _get_current_output(output: list[torch.Tensor], indices: list[int], batch_dim: int) -> torch.Tensor:
        current_output = [output[x] for x in indices]
        current_output = torch.cat(current_output, dim=batch_dim)
        return current_output
