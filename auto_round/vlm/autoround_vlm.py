# Copyright (c) 2024 Intel Corporation
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

from copy import deepcopy
from typing import Union

import torch
from torch import autocast
from tqdm import tqdm


from auto_round import AutoRound
from auto_round.low_cpu_mem.utils import get_layers_before_block
from auto_round.utils import (
    check_to_quantized,
    clear_memory,
    detect_device,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    logger,
    mllm_load_model,
    to_device,
    to_dtype,
    block_forward,
    is_nv_fp,
    collect_best_params,
    mv_module_from_gpu,
    check_need_act_calibration,
    get_module,
    flatten_list,
)
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block
from .vlm_dataset import get_vlm_dataloader


class AutoRoundVLM(AutoRound):
    """Class for automatic rounding-based quantization with MLLMs.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether sym to be used (default is True).
        layer_config (dict): Configuration for weight quantization (default is None).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset: The path or name of the calib dataset.
        extra_data_dir: The path of extra data such as images, audio and videos.
        enable_quanted_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        low_cpu_mem_usage (bool): Whether to use low CPU memory (default is False).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        nsamples (int): Number of samples (default is 128).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).s
        nblocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                           have different choices.
        act_bits (int): Number of bits for activation quantization. Default is 32.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A string or list whose elements are list of
                            block's layer names to be quantized.
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer
        **kwargs: Additional keyword arguments.


    """

    def __init__(
        self,
        model: torch.nn.Module,
        pipe,
        tokenizer=None,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        layer_config: dict = None,
        batch_size: int = 8,
        amp: bool = True,
        device: str = None,
        lr_scheduler=None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = None,
        extra_data_dir: str = None,
        enable_quanted_input: bool = True,
        enable_minmax_tuning: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        low_gpu_mem_usage: bool = False,
        low_cpu_mem_usage: bool = False,
        iters: int = 200,
        seqlen: int = None,
        nsamples: int = 128,
        sampler: str = "rand",
        seed: int = 42,
        nblocks: int = 1,
        gradient_accumulate_steps: int = 1,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        data_type: str = "int",
        scale_dtype: str = "fp16",
        act_bits: int = 32,
        act_group_size: int = None,
        act_sym: bool = None,
        act_dynamic: bool = True,
        act_data_type: str = None,
        to_quant_block_names: Union[str, list] = None,
        enable_norm_bias_tuning: bool = False,
        truncation: bool = None,
        enable_torch_compile: bool = False,
        model_kwargs: dict = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 1,
        generator: object = None,
        **kwargs,
    ):
        all_blocks = get_block_names(model)
        self.quant_block_list = find_matching_blocks(model, all_blocks, to_quant_block_names)
        if to_quant_block_names is None:
            to_quant_block_names = extract_block_names_to_str(self.quant_block_list)
        self.to_quant_block_names = to_quant_block_names
        self.extra_data_dir = extra_data_dir
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator = generator

        if iters > 0 or check_need_act_calibration(act_dynamic, act_data_type):
            if dataset is None:
                logger.warning("Dataset is not provided, will use coco-2014 captions for calibration")
                dataset = "coco2014"

            if batch_size != 1:
                logger.warning(
                    f"reset batch_size({batch_size}) to 1 and "
                    f"gradient_accumulate_steps({gradient_accumulate_steps}) "
                    f"to {batch_size * gradient_accumulate_steps}, "
                    f"because batch_size={batch_size} cannot be used for calibrating non-text modules."
                )
                gradient_accumulate_steps = batch_size * gradient_accumulate_steps
                batch_size = 1
        seqlen = 2048 if seqlen is None else seqlen
        truncation = True if truncation is None else truncation
        self.truncation = truncation

        if nsamples % batch_size != 0:
            nsamples = (nsamples // batch_size + 1) * batch_size
            logger.warning(f"'nsamples' is not divisible by 'batch_size', will adjusted to {nsamples}")

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            layer_config=layer_config,
            batch_size=batch_size,
            amp=amp,
            device=device,
            lr_scheduler=lr_scheduler,
            dataset=dataset,
            enable_quanted_input=enable_quanted_input,
            enable_minmax_tuning=enable_minmax_tuning,
            lr=lr,
            minmax_lr=minmax_lr,
            low_gpu_mem_usage=low_gpu_mem_usage,
            low_cpu_mem_usage=low_cpu_mem_usage,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            sampler=sampler,
            seed=seed,
            nblocks=nblocks,
            gradient_accumulate_steps=gradient_accumulate_steps,
            not_use_best_mse=not_use_best_mse,
            dynamic_max_gap=dynamic_max_gap,
            data_type=data_type,
            scale_dtype=scale_dtype,
            act_bits=act_bits,
            act_group_size=act_group_size,
            act_sym=act_sym,
            act_dynamic=act_dynamic,
            act_data_type=act_data_type,
            to_quant_block_names=self.to_quant_block_names,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_torch_compile=enable_torch_compile,
            vlm=True,
            **kwargs,
        )

    def calib(self, nsamples, bs):
        """Perform calibration for quantization.

        This method calibrates the model for quantization by processing a specified
        number of samples from the calibration dataset. It ensures that the data is
        properly formatted and feeds it to the model. If the number of samples processed
        is less than the specified number, it logs a warning. If no samples are processed,
        it logs an error and exits.
        Args:
            nsamples (int): The number of samples to use for calibration.
            bs (int): The number of samples to use for calibration
        """
        if isinstance(self.dataset, str):
            dataset = self.dataset.replace(" ", "")
            self.dataloader, self.batch_size, self.gradient_accumulate_steps = get_vlm_dataloader(
                dataset=dataset,
                extra_data_dir=self.extra_data_dir,
                bs=self.batch_size,
                seed=self.seed,
                nsamples=self.nsamples,
                gradient_accumulate_steps=self.gradient_accumulate_steps,
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0

        if self.low_cpu_mem_usage:
            embed_layers = get_layers_before_block(self.model)
            for n, m in embed_layers:
                m = m.to(self.device)

        total = nsamples if not hasattr(self.dataloader, "len") else min(nsamples, len(self.dataloader))
        if self.pipe.dtype != self.model.dtype:
            self.pipe.to(self.model.dtype)
        if self.pipe.device != self.model.device:
            self.pipe.to(self.model.device)
        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                try:
                    self.pipe(
                        prompt=prompts,
                        guidance_scale=self.guidance_scale,
                        #num_inference_steps=self.num_inference_steps,
                        num_inference_steps=1,
                        generator=self.generator
                    )
                except NotImplementedError:
                    pass
                except Exception as error:
                    raise error
                step = len(prompts)
                total_cnt += step
                pbar.update(step)
                if total_cnt >= nsamples:
                    break
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataset or decease the sequence length"
            )
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantization. "
                f"target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )
            if total_cnt < self.batch_size:
                raise ValueError(
                    f"valid samples is less than batch_size({self.batch_size}),"
                    " please adjust self.batch_size or seqlen."
                )
            max_len = (total_cnt // self.batch_size) * self.batch_size
            for k, v in self.inputs.items():
                for key in v:
                    if isinstance(v[key], list) and len(v[key]) == total_cnt:
                        self.inputs[k][key] = v[key][:max_len]

        # clean embed weight to save memory
        if self.low_cpu_mem_usage:
            for n, m in embed_layers:
                m = m.to("meta")
        # torch.cuda.empty_cache()


    @torch.no_grad()
    def _get_block_outputs(self, block, input_ids, input_others, bs, device, cache_device, save_output=True, only_return_hidden_states=True):
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
        new_encoder_hidden_states = deepcopy(input_others.get("encoder_hidden_states", []))
        for i in range(0, nsamples, bs):
            end_index = min(nsamples, i + bs)
            indices = torch.arange(i, end_index).to(torch.long)
            tmp_input_ids, tmp_input_others = AutoRound._sampling_inputs(
                input_ids, input_others, indices, self.seqlen, self.batch_dim, share_cache_keys=self.shared_cache_keys
            )
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device, only_return_hidden_states)
            if "encoder_hidden_states" in tmp_input_others and (isinstance(tmp_output, list) or isinstance(tmp_output, tuple)) and len(tmp_output) == 2:
                encoder_hidden_states, tmp_output = tmp_output
                for i in indices:
                    new_encoder_hidden_states[i] = encoder_hidden_states.to(cache_device)
            tmp_output = tmp_output.to(cache_device)
            if save_output:
                if self.batch_size == 1:
                    output.append(tmp_output)
                else:
                    output.extend(list(torch.split(tmp_output, 1, dim=self.batch_dim)))
        if self.low_gpu_mem_usage:
            clear_memory()
 
        return output, new_encoder_hidden_states


    def _quantize_via_rtn_blockwise(self, all_to_quantized_module_names: list[str]) -> None:
        """Quantize model layers block by block using cached inputs and imatrix.

        Args:
            all_to_quantized_module_names (list[str]): Names of layers to be quantized.
        """
        all_to_quantized_module_names = list(set(all_to_quantized_module_names))

        all_blocks = self.quant_block_list if self.quant_block_list else get_block_names(self.model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        all_first_block_names = [block[0] for block in all_blocks]
        if self.act_bits < 16:
            all_inputs = self.try_cache_inter_data_gpucpu(all_first_block_names, self.nsamples)
        else:
            all_inputs = self.cache_inter_data(all_first_block_names, self.nsamples)

        # Clear hooks for multi-GPU setups
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model)

        pbar = tqdm(range(sum(len(block) for block in all_blocks)))

        for block_names in all_blocks:
            first_block = block_names[0]
            inputs = all_inputs.pop(first_block)
            input_keys = [k for k in inputs if k.startswith("hidden_state")]
            if len(input_keys) != 1:
                raise RuntimeError(
                    "hidden_states arg mismatch. Please file an issue at https://github.com/intel/auto-round/issues"
                )
            inputs["input_ids"] = inputs.pop(input_keys[0])

            clear_memory(self.inputs)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.batch_size:
                self.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")

            input_ids = to_device(inputs.pop("input_ids"), self.cache_device)
            input_others = to_device(inputs, self.cache_device)

            tmp_dtype = self.amp_dtype if self.amp else torch.float32
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]

            for key, val in input_others.items():
                if isinstance(val, torch.Tensor) and val.dtype in (torch.float16, torch.bfloat16):
                    input_others[key] = val.to(tmp_dtype)
                elif isinstance(val, list):
                    input_others[key] = [to_dtype(v, tmp_dtype) for v in val]

            for block_name in block_names:
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model, block_name)
                block = block.to(self.device)
                if hasattr(self.model, "is_fp8"):
                    convert_fp8_model_to_16b_model(block, dtype=self.amp_dtype)
                # Dispatch model if needed
                if self.device_map is not None:
                    from accelerate import dispatch_model
                    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                    for _, m in block.named_modules():
                        if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                            continue
                        hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                        add_hook_to_module(m, hook, True)
                else:
                    block = block.to(self.device)
                input_ids, encoder_hidden_states = self._get_block_outputs(
                    block,
                    input_ids,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    self.device,
                    self.cache_device,
                    only_return_hidden_states=False,
                )
                if "encoder_hidden_states" in input_others:
                    input_others["encoder_hidden_states"] = encoder_hidden_states

                if self.device_map is not None:
                    accelerate.hooks.remove_hook_from_submodules(block)

                if is_nv_fp(self.act_data_type) and any("nv_fp" in format_ for format_ in self.formats):
                    from auto_round.utils import set_amax_for_all_moe_layers

                    # enable moe experts act_max automatic generation for linears
                    set_amax_for_all_moe_layers(block, attr_name="act_max")
                # Normalize imatrix and quantize layers
                for _, m in block.named_modules():
                    if hasattr(m, "imatrix"):
                        m.imatrix /= m.imatrix_cnt
                    if hasattr(m, "tmp_name") and m.tmp_name in all_to_quantized_module_names:
                        self._quantize_layer_via_rtn(m.tmp_name)
                        all_to_quantized_module_names.remove(m.tmp_name)

                mv_module_from_gpu(block, self.low_cpu_mem_usage)
                pbar.update(1)

        pbar.close()
        cnt = 1
        block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
        clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
        if clear_mem_freq == 0:
            clear_mem_freq = 1
        # Process remaining layers not in blocks
        for name in all_to_quantized_module_names:
            self._quantize_layer_via_rtn(name)
            if cnt % clear_mem_freq == 0:
                clear_memory()
                cnt = 1
            cnt += 1


    def _quantize_block(self, block, input_ids, input_others, q_input=None, device=torch.device("cpu")):
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
        if hasattr(self.model, "is_fp8"):
            for n, m in block.named_modules():
                if m.__class__.__name__ == "FP8Linear":
                    new_layer = convert_fp8_layer_to_linear(m, self.amp_dtype).to(device)
                    set_module(block, n, new_layer)

        if self.device_map is not None:
            from accelerate import dispatch_model

            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                add_hook_to_module(m, hook, True)

        if q_input is None:
            hook_handles = self._register_act_max_hook(block)

            output, encoder_hidden_states = self._get_block_outputs(
                block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device, self.cache_device, only_return_hidden_states=False
            )

            for handle in hook_handles:
                handle.remove()
        else:
            output, encoder_hidden_states = self._get_block_outputs(
                block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device, self.cache_device, only_return_hidden_states=False
            )
            hook_handles = self._register_act_max_hook(block)
            if hook_handles:
                self._get_block_outputs(
                    block,
                    q_input,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    device,
                    self.cache_device,
                    save_output=False,
                )

            for handle in hook_handles:
                handle.remove()

        if q_input is not None:
            if input_ids is not q_input:
                clear_memory(input_ids)
            else:
                clear_memory()
            input_ids = q_input

        quantized_layer_names, unquantized_layer_names = wrapper_block(
            block, self.enable_minmax_tuning, self.enable_norm_bias_tuning, device=self.device
        )
        if is_nv_fp(self.data_type):  # enable qkv and moe structure global_scale fuse
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

        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": self.minmax_lr}], lr=self.lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=self.lr, weight_decay=0)

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            unwrapper_block(block, {})  ## TODO Quant layer should change
            mv_module_from_gpu(block, self.low_cpu_mem_usage)
            return output, output

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        nsamples = len(input_ids)
        pick_samples = self.batch_size * self.gradient_accumulate_steps
        pick_samples = min(nsamples, pick_samples)
        if self.sampler != "rand":
            whole_indices = torch.randperm(nsamples)[:pick_samples]
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

        for i in range(self.iters):
            total_loss = 0
            if self.sampler == "rand":
                whole_indices = torch.randperm(nsamples)[:pick_samples]
                ##we assume the block input and output shape is same
                if self.gradient_accumulate_steps != 1:
                    current_input_ids = [input_ids[i] for i in whole_indices]
                    num_elm = sum(id.numel() for id in current_input_ids)
            for tmp_step in range(self.gradient_accumulate_steps):
                indices = whole_indices[tmp_step * self.batch_size : (tmp_step + 1) * self.batch_size]
                current_input_ids, current_input_others = AutoRound._sampling_inputs(
                    input_ids,
                    input_others,
                    indices,
                    seqlen=self.seqlen,
                    batch_dim=self.batch_dim,
                    share_cache_keys=self.shared_cache_keys,
                )

                current_output = [output[x] for x in indices]
                current_output = torch.cat(current_output, dim=self.batch_dim)

                current_output = to_device(current_output, device)

                output_q = block_forward(
                    block, current_input_ids, current_input_others, self.amp, self.amp_dtype, device, only_return_hidden_states=False
                )
                if "encoder_hidden_states" in input_others and (isinstance(output_q, list) or isinstance(output_q, tuple)) and len(output_q) == 2:
                    output_q = output_q[1]
 
                if self.amp:
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                        loss = mse_loss(output_q, current_output)  # pylint: disable=not-callable
                else:
                    loss = mse_loss(  # pylint: disable=not-callable
                        output_q.to(torch.float32), current_output.to(torch.float32)
                    )

                total_loss += loss.item() / num_elm
                self._scale_loss_and_backward(scaler, loss)

            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(block)
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)

                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(block)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        dump_info = (
            f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
            f"layers in the block, loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        )
        logger.info(dump_info)
        if len(unquantized_layer_names) != 0:
            logger.info(f"{unquantized_layer_names} have not been quantized")
        with torch.no_grad():
            unwrapper_block(block, best_params)
        if self.enable_quanted_input:
            if is_nv_fp(self.act_data_type) and any("nv_fp" in format_ for format_ in self.formats):
                from auto_round.utils import set_amax_for_all_moe_layers

                # enable moe experts act_max automatic generation for WrapperWALayer
                set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")
            if self.low_cpu_mem_usage:
                block = block.to(device)
            clear_memory()
            q_outputs, _ = self._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.batch_size * self.infer_bs_coeff,
                device,
                cache_device=self.cache_device,
                only_return_hidden_states=False,
            )
            if self.device_map is not None:
                accelerate.hooks.remove_hook_from_submodules(block)
            mv_module_from_gpu(block, self.low_cpu_mem_usage)
            clear_memory(input_ids)

            if "encoder_hidden_states" in input_others:
                input_others["encoder_hidden_states"] = encoder_hidden_states
            return q_outputs, output

        else:
            if self.device_map is not None:
                accelerate.hooks.remove_hook_from_submodules(block)
            mv_module_from_gpu(block, self.low_cpu_mem_usage)
            clear_memory(input_ids)
            if "encoder_hidden_states" in input_others:
                input_others["encoder_hidden_states"] = encoder_hidden_states
            return None, output


    def save_quantized(self, output_dir=None, format="auto_round", inplace=True, **kwargs):
        """Save the quantized model to the specified output directory in the specified format.

        Args:
            output_dir (str, optional): The directory to save the quantized model. Defaults to None.
            format (str, optional): The format in which to save the model. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place. Defaults to True.
            **kwargs: Additional keyword arguments specific to the export format.

        Returns:
            object: The compressed model object.
        """
        compressed_model = super().save_quantized(
            output_dir=output_dir, format=format, inplace=inplace, **kwargs
        )
        return compressed_model

