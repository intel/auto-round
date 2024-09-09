# Copyright (c) 2023 Intel Corporation
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
from typing import Optional, Union
import logging
import torch
import transformers
from torch import autocast
import numpy as np
from .calib_dataset import get_dataloader
from .quantizer import WrapperMultiblock, wrapper_block, unwrapper_block, WrapperLinear, unwrapper_layer
from .special_model_handler import check_hidden_state_dim, check_share_attention_mask, check_not_share_position_ids
from .utils import (
    # CpuInfo,
    block_forward,
    check_is_cpu,
    check_to_quantized,
    collect_minmax_scale,
    collect_round_v,
    convert_dtype_str2torch,
    detect_device,
    get_block_names,
    get_module,
    htcore,
    is_optimum_habana_available,
    logger,
    sampling_inputs,
    to_device,
    to_dtype,
    get_layer_names_in_block,
    mv_module_from_gpu,
    unsupport_meta_device,
    to_device_amp_dtype,
)

# disable lwq to remove the dependency on accelerator
# from .low_cpu_mem.utils import get_layers_before_block

def get_block_outputs(block, block_inputs):
    block_outputs = []
    for (args, kwargs) in block_inputs:
        outputs = block(*args, **kwargs)
        block_outputs.append(outputs)
    return block_outputs

def _use_first_output(outputs):
    first_outputs = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            first_outputs.append(output)
        elif isinstance(output, (list, tuple)):
            first_outputs.append(output[0])
        else:
            raise ValueError("output should be tuple or list")
    return torch.cat(first_outputs, dim=0)

def freeze_mod_(mod):
    for param in mod.parameters():
        param.requires_grad = False


class AutoRound(object):
    """This is Signround+ which is an advanced version of Signround. For more information,
     please refer to Cheng, Wenhua, et al. "Optimize weight rounding via signed gradient descent
     for the quantization of llms." arXiv preprint arXiv:2309.05516 (2023).

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether symmetric quantization is to be used (default is False).
        layer_config (dict): Configuration for weight quantization (default is an empty dictionary).
        layer_config={
                   'layer1':##layer_name
                   {
                       'data_type': 'int',
                       'bits': 4,
                       'group_size': 128,
                       'sym': False
                       'act_data_type': None,
                       'act_bits': 32,
                       'act_group_size': None,
                       'act_sym': None,

                   }
                   ...
               }
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for tuning (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset (str): The default dataset name (default is "NeelNanda/pile-10k").
        enable_quanted_input (bool): Whether to use the output of the previous quantized block as
                                the input for the current block (default is True).
        enable_minmax_tuning (bool): Whether to enable weight min-max tuning (default is True).
        lr (float): The learning rate (default is None, will be set to 1.0/iters).
        minmax_lr (float): The learning rate for min-max tuning (default is None, it will be set to lr automatically).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is True).
        low_cpu_mem_usage (bool): Whether to use low CPU memory (default is False).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Data length of the sequence for tuning (default is 2048).
        nsamples (int): Number of samples (default is 128).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
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
        quant_block_list (list): A list whose elements are list of block's layer names to be quantized.
    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer,
            bits: int = 4,
            group_size: int = 128,
            sym: bool = False,
            layer_config: dict = {},
            enable_full_range: bool = False,  ##for symmetric, TODO support later
            batch_size: int = 8,
            amp: bool = True,
            device: str = None,
            lr_scheduler=None,
            dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
            enable_quanted_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = None,
            minmax_lr: float = None,
            low_gpu_mem_usage: bool = False,
            low_cpu_mem_usage: bool = False,
            iters: int = 200,
            seqlen: int = 2048,
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
            quant_block_list: list = None,
            model_dtype: torch.dtype = torch.float32,
            **kwargs,
    ):
        self.quantized = False
        if not hasattr(model, "dtype"):
            model.dtype = model_dtype
        self.model_orig_dtype = model.dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        assert not unsupport_meta_device(model),  (
            "autoround does not support for params on meta device by transformers` interfaces," 
            "please do not using device_map='auto' in model loading, "
            "or follow examples/language-modeling/main.py to enable low_cpu_mem_usage")
        self.model = model.eval()
        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        self.amp = amp
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.nsamples = nsamples
        self.nblocks = nblocks
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.data_type = data_type
        self.supported_types = [torch.nn.Linear, transformers.modeling_utils.Conv1D]
        self.layer_config = layer_config
        self.seed = seed
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        self.train_bs = batch_size
        self.nblocks = nblocks
        self.device = detect_device(device)
        self.scale_dtype = convert_dtype_str2torch(scale_dtype)
        self._original_dtype = next(self.model.parameters()).dtype
        self.set_amp_dtype()
        self.cache_device = torch.device("cpu") if self.low_gpu_mem_usage else self.device
        self.dataset = dataset

        self.iters = iters
        self.quant_block_list = quant_block_list
        if self.iters <= 0:
            logger.warning("iters must be positive, reset it to 200")
            self.iters = 200
        self.lr = lr or (1.0 / self.iters)
        self.minmax_lr = minmax_lr or self.lr

        self.sampler = sampler
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.enable_full_range = enable_full_range
        self.lr_scheduler = lr_scheduler
        self.optimizer = self.get_optimizer(None)
        self.share_attention_mask_flag = None
        self.hidden_dim_flag = None
        self.infer_bs_coeff = 1
        self.act_group_size = act_group_size if not (act_group_size is None) else self.group_size
        self.act_bits = act_bits if not (act_bits is None) else self.bits
        self.act_sym = act_sym if not (act_sym is None) else self.sym
        self.act_dynamic = act_dynamic
        self.set_layerwise_config(self.layer_config)
        torch.set_printoptions(precision=3, sci_mode=True)
        self.check_configs()
        logger.info(f"using {self.model.dtype} for quantization tuning")
        if is_optimum_habana_available():
            logger.info("Optimum Habana is available, import htcore explicitly.")
            import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401

    def check_configs(self):
        """Checks if the configurations are valid.

        Raises:
        AssertionError: If any of the configurations are invalid.
        """
        assert isinstance(self.model, torch.nn.Module)
        assert self.bits > 0, "bits must be positive"
        assert self.act_bits > 0, "bits must be positive"
        assert self.group_size == -1 or self.group_size >= 1, "only supports positive group_size or -1(per channel)"
        assert self.act_group_size == -1 or self.act_group_size >= 1,\
            "only supports positive group_size or -1(per channel)"
        assert self.train_bs > 0, "batch size must be positive"
        assert self.iters > 0, "iters must be positive"
        assert self.seqlen > 0, "seqlen must be positive"
        assert self.nblocks > 0, "nblocks must be positive"
        assert self.gradient_accumulate_steps > 0, "gradient accumulate step must be positive"
        assert self.enable_full_range is False, "only support enable_full_range=False currently"
        assert self.act_dynamic is True, "only support dynamic quantization for activation currently"
        # assert self.tokenizer != None or self.dataloader != None

    def quantize(self):
        """Quantize the model and return the quantized model along with layer configurations.
        the entry of AutoRound.

        Returns:
        The quantized model and layer configurations.
        """
        # logger.info("cache block input")
        if bool(self.quant_block_list):
            all_blocks = self.quant_block_list
        else:
            all_blocks = get_block_names(self.model)
                    
        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model, self.layer_config

        if self.amp:
            self.model = self.model.to(self.amp_dtype)

        layer_names = self.get_quantized_layer_names_outside_blocks()
        self.start_time = time.time()
        all_first_block_names = [block[0] for block in all_blocks]
        all_inputs = self.try_cache_inter_data_gpucpu(all_first_block_names, self.nsamples, layer_names=layer_names)
        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        for block_names in all_blocks:
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            self.inputs = None
            del self.inputs
            if "input_ids" in inputs.keys():
                total_samples = len(inputs["input_ids"])
                self.n_samples = total_samples
                if total_samples < self.train_bs:
                    self.train_bs = total_samples
                    logger.warning(f"force the train batch size to {total_samples}")

            torch.cuda.empty_cache()
            self.quant_blocks(
                self.model,
                inputs,
                block_names,
                nblocks=self.nblocks,
                device=self.device,
            )

        self.quant_layers(layer_names, all_inputs)

        self.dump_qinfo_to_layer_config()

        end_time = time.time()
        cost_time = end_time - self.start_time
        logger.info(f"quantization tuning time {cost_time}")

        ## dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
                if m.bits > 8:
                    unquantized_layers.append(n)
                else:
                    quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            summary_info += f",  {unquantized_layers} have not been quantized"
        logger.info(summary_info)

        self.quantized = True
        ##self.model = self.model.to(self.model_orig_dtype)##keep it as amp dtype
        return self.model, self.layer_config

    def dump_qinfo_to_layer_config(self):
        """
        dump quantization scale and zp to layer configuration
        Args:

        Returns:
            None
        """
        # load scale and zp if use low_cpu_memory
        self.model = self.model.to('cpu')

        for n, m in self.model.named_modules():
            if n not in self.layer_config.keys():
                continue
            if hasattr(m, "scale"):
                self.layer_config[n]["scale"] = m.scale
                self.layer_config[n]["zp"] = m.zp
                delattr(m, "scale")
                delattr(m, "zp")
            else:
                self.layer_config[n]["data_type"] = "float"
                if self.amp_dtype == torch.bfloat16:
                    self.layer_config[n]["data_type"] = "bfloat"
                self.layer_config[n]["bits"] = 32
                self.layer_config[n]["group_size"] = None
                self.layer_config[n]["sym"] = None

    def quant_layers(self, layer_names, layer_inputs):
        """Quantizes specified layers based on inputs and configuration.

        Args:
            layer_names (list): List of layer names to quantize.
            layer_inputs (dict): Dictionary mapping layer names to input data.

        Returns:
            None
        """
        ##TODO currently we take all the layers outside blocks as post block layers which is not optimal
        if len(layer_names) == 0:
            return
        q_layer_inputs = None
        if self.enable_quanted_input:
            q_layer_inputs = self.try_cache_inter_data_gpucpu([], self.nsamples, layer_names=layer_names)

        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        torch.cuda.empty_cache()
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.cache_device)
            q_layer_input = q_layer_inputs[layer_name] if self.enable_quanted_input else None
            q_layer_input = to_device(q_layer_input, self.cache_device)
            self.quant_layer(layer_name, layer_input, q_layer_input, device=self.device)
            for i in range(len(layer_input)):
                layer_input[i] = None
                if q_layer_input is not None:
                    q_layer_input[i] = None
            torch.cuda.empty_cache()

    def set_layerwise_config(self, layer_config):
        """Sets the layer-wise configuration based on the provided layer_config.
           By default, only quantize layers in blocks.

        Args:
        layer_config: The layer configuration.

        Returns:
        None
        """
        layers_in_blocks = get_layer_names_in_block(self.model, self.supported_types, self.quant_block_list)
        keys = ["data_type", "bits", "group_size", "sym", "scale_dtype", "act_bits", "act_group_size", "act_sym",
                "act_dynamic"]
        for n, m in self.model.named_modules():
            if not isinstance(m, tuple(self.supported_types)):
                continue
            ##not set in layer config, so use the default values
            if n not in layer_config.keys() and n in layers_in_blocks:
                layer_config[n] = {}
                for key in keys:
                    layer_config[n][key] = getattr(self, key)
            elif n in layer_config.keys():  ## partly set
                for key in keys:
                    if key not in layer_config[n].keys():
                        layer_config[n][key] = getattr(self, key)
            else:  ##not in layer_config and layers in block,
                layer_config[n] = {}
                for key in keys:
                    layer_config[n][key] = getattr(self, key)
                layer_config[n]["bits"] = self.bits
                layer_config[n]["act_bits"] = 32
            
            for key in keys:
                setattr(m, key, layer_config[n][key])

    @torch.no_grad()
    def get_block_outputs(self, block, input_ids, input_others, bs, device, cache_device):
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
            tmp_input_ids, tmp_input_others = sampling_inputs(
                input_ids,
                input_others,
                indices,
                self.seqlen,
                self.share_attention_mask_flag,
                # self.not_share_position_ids_flag,
                # self.input_dim
            )
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device).to(
                cache_device
            )
            output.extend(list(torch.split(tmp_output, 1, dim=self.input_dim)))
        torch.cuda.empty_cache()

        return output

    @torch.no_grad()
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
            dataset = self.dataset.replace(" ", "")  ##remove all whitespaces
            # slow here
            self.dataloader = get_dataloader(
                self.tokenizer,
                self.seqlen,
                dataset,
                self.seed,
                bs,
                self.nsamples,
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0

        # load embed weight if use low_cpu_mem_usage
        if self.low_cpu_mem_usage:
            embed_layers = get_layers_before_block(self.model)
            for n, m in embed_layers:
                m = m.to(self.device)
        
        for data in self.dataloader:
            if data is None:
                continue
            if isinstance(data, torch.Tensor):
                input_ids = data.to(self.device)
                data_new = input_ids
            elif isinstance(data, str):
                if self.tokenizer is None:
                    logger.error("please provide tokenizer for string input")
                    exit()
                data = self.tokenizer(data, truncation=True, max_length=self.seqlen, return_tensors="pt").data
                data_new = {}
                for key in data.keys():
                    data_new[key] = data[key].to(self.device)
                input_ids = data_new["input_ids"]
            elif isinstance(data, tuple) or isinstance(data, list):
                    data_new = data
                    input_ids = data_new[0]
            else:
                data_new = {}
                for key in data.keys():
                    data_new[key] = to_device(data[key], self.model.device)
                    if key == 'images':
                        data_new[key] = to_dtype(data[key], self.model.dtype)
                input_ids = data_new["input_ids"]
            if input_ids.shape[-1] < self.seqlen:
                continue
            
            try:
                if isinstance(data_new, torch.Tensor):
                    self.model(data_new)
                elif isinstance(data_new, tuple) or isinstance(data_new, list):
                    self.model(*data_new)
                else:
                    self.model(**data_new)
            except NotImplementedError:
                pass
            except Exception as error:
                logger.error(error)
            total_cnt += input_ids.shape[0] if len(input_ids.shape) > 1 else 1
            if total_cnt >= nsamples:
                break
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataset or decease the sequence length"
            )
            exit()
        elif total_cnt < nsamples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantification. "
                f"Valid samples size:{total_cnt}, Target sample size:{nsamples}"
            )
        
        # clean embed weight to save memory
        if self.low_cpu_mem_usage:
            for n, m in embed_layers:
                m = m.to("meta")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def try_cache_inter_data_gpucpu(self, block_names, nsamples, layer_names=[], last_cache_name=None):
        """Attempts to cache intermediate data on GPU, if failed, then using CPU.

        Args:
            block_names (list): List of block names to cache data for.
            nsamples (int): Number of samples to use for caching.
            layer_names (list, optional): List of layer names to cache data for. Defaults to [].
            last_cache_name (str, optional): Name of the last cache. Defaults to None.

        Returns:
            all_inputs: Cached intermediate data.

        Raises:
            Exception: If caching on GPU fails, switches to CPU and caches there.
        """
        try:
            if not self.model.device.type == "meta":
                self.model = self.model.to(self.device)
            all_inputs = self.cache_inter_data(
                block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
            )
            self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
            torch.cuda.empty_cache()
        except:
            logger.info("switch to cpu to cache inputs")
            self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
            torch.cuda.empty_cache()
            all_inputs = self.cache_inter_data(
                block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
            )
        return all_inputs

    @torch.no_grad()
    def cache_inter_data(self, block_names, nsamples, layer_names=[], last_cache_name=None):
        """Save the inputs of block_name for calibration. For layers, we cache both of inputs and output.

        This method temporarily replaces the forward method of the model to capture
        the inputs passing through the specified block. It then calibrates the model
        using a specified number of samples. Finally, it restores the original forward
        method and returns the inputs for the specified block.
        Args:
            block_names (list): The names of the blocks for which inputs are to be saved.
            layer_names (list):The names of the layers for which inputs are to be saved.
            nsamples (int): The number of samples to use for calibration.
            last_cache_name (str, optional): The name of the last layer to be cached,
                                       we could break the forward in this layer to save time

        Returns:
            dict: A dictionary containing the inputs for the specified block.
        """
        self.inputs = {}
        self.to_cached_layers = block_names + layer_names
        tmp_dtype = None
        ## have bug if block name is not the first block
        if (len(block_names) > 1 or len(layer_names) > 0) and self.low_gpu_mem_usage:
            tmp_dtype = self.model.dtype
            self.model = self.model.to(torch.bfloat16) if self.amp else self.model.to(torch.float32)

        self.last_cache_name = last_cache_name
        if last_cache_name is None and len(block_names) + len(layer_names) == 1:
            self.last_cache_name = block_names[0] if len(block_names) == 1 else layer_names[0]
        # do not set last_cache_name for multimodal models
        calib_bs = self.train_bs
        self.hook_handles = []
        self._replace_forward()
        self.calib(nsamples, calib_bs)
        self._recover_forward()
        res = self.inputs
        del self.last_cache_name
        del self.to_cached_layers
        if tmp_dtype is not None:
            self.model = self.model.to(tmp_dtype)

        return res

    @torch.no_grad()
    def get_block_forward_func(self, name):
        """Gets the forward function.

        Args:
            name (str): The name of the function.
        Returns:
            function: The forward function.
        """

        def forward(m, hidden_states, *positional_args, **kwargs):
            """Rewrite forward function, process and collect input data.

            Args:
                hidden_states (torch.Tensor): The hidden states tensor.
                *positional_args: Variable number of positional arguments.
                **kwargs: Variable number of keyword arguments.

            Returns:
                NotImplementedError: Getting the first layer inputs and then raise the error to save runtime.
            """
            if self.share_attention_mask_flag is None:
                self.input_dim = check_hidden_state_dim(self.model, positional_args)
                self.share_attention_mask_flag = check_share_attention_mask(self.model, hidden_states, **kwargs)
                self.not_share_position_ids_flag = check_not_share_position_ids(self.model, **kwargs)
            if name in self.inputs:
                self.inputs[name]["input_ids"].extend(list(torch.split(hidden_states.to("cpu"), 1, dim=self.input_dim)))
            else:
                self.inputs[name] = {}
                self.inputs[name]["input_ids"] = list(torch.split(hidden_states.to("cpu"), 1, dim=self.input_dim))

            if "positional_inputs" not in self.inputs[name]:
                self.inputs[name]["positional_inputs"] = []
            for idx, item in enumerate(positional_args):
                self.inputs[name]["positional_inputs"] = to_device(positional_args)

            for key in kwargs.keys():
                if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) \
                        or (key == "alibi") or (key == "attention_mask"):
                    if "attention_mask" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if kwargs[key] is not None:
                            if (not self.share_attention_mask_flag) and self.inputs[name][key] is not None:
                                self.inputs[name][key].extend(list(torch.split(kwargs[key].to("cpu"), 1, dim=0)))
                            else:
                                self.inputs[name][key] = list(torch.split(kwargs[key].to("cpu"), 1, dim=0))
                    elif "alibi" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if isinstance(kwargs[key], torch.Tensor):
                            alibi = kwargs[key]
                            batch = kwargs["attention_mask"].shape[0]
                            alibi = alibi.reshape(batch, -1, alibi.shape[1], alibi.shape[2])
                            if (not self.share_attention_mask_flag) and self.inputs[name][key] is not None:
                                self.inputs[name][key].extend(list(torch.split(alibi.to("cpu"), 1, dim=0)))
                            else:
                                self.inputs[name][key] = list(torch.split(alibi.to("cpu"), 1, dim=0))
                    elif "position_ids" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = list(torch.split(kwargs[key].to("cpu"), 1, dim=0)) \
                                                    if self.not_share_position_ids_flag \
                                                    else to_device(kwargs[key], device=torch.device("cpu"))
                        elif kwargs[key] is not None and self.not_share_position_ids_flag:
                            self.inputs[name][key].extend(list(torch.split(kwargs[key].to("cpu"), 1, dim=0)))
                    elif key not in self.inputs[name].keys():
                        self.inputs[name][key] = to_device(kwargs[key], device=torch.device("cpu"))
            if name == self.last_cache_name:
                raise NotImplementedError
            else:
                return m.orig_forward(hidden_states, *positional_args, **kwargs)

        return forward

    @torch.no_grad()
    def _get_cache_data_hook_for_layer(self, name):
        """A forward hook to save input max of a module
        :param name: the module name
        :return: A hook function."""

        def cache_input_hook(module, inputs, outputs):
            input = inputs
            if isinstance(inputs, tuple) or isinstance(input, list):
                input = inputs[0]
            if name in self.inputs:
                self.inputs[name].extend(list(torch.split(input.to("cpu"), 1, dim=0)))
            else:
                self.inputs[name] = list(torch.split(input.to("cpu"), 1, dim=0))

        return cache_input_hook

    def _recover_forward(self):
        """Recovers the forward function."""
        for n, m in self.model.named_modules():
            if hasattr(m, "orig_forward"):
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        self.hook_handles = []

    def _replace_forward(self):
        """Replaces the forward function."""
        from functools import partial

        for n, m in self.model.named_modules():
            if n in self.to_cached_layers and not isinstance(m, tuple(self.supported_types)):  ##block
                m.orig_forward = m.forward
                m.forward = partial(self.get_block_forward_func(n), m)
            elif n in self.to_cached_layers:  ##linear layer or conv1d layer
                hook_func = self._get_cache_data_hook_for_layer(n)
                hook_handle = m.register_forward_hook(hook_func)
                self.hook_handles.append(hook_handle)

    def quant_layer(self, layer_name, inputs, q_inputs=None, device=torch.device("cpu")):
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
        layer = layer.to(device)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(layer.weight.dtype)
            if q_inputs is not None:
                q_inputs[i] = q_inputs[i].to(layer.weight.dtype)

        wrapper_linear = WrapperLinear(layer, self.enable_minmax_tuning, device).to(device)
        round_params = []
        minmax_params = []
        round_params.append(wrapper_linear.value)
        minmax_params.append(wrapper_linear.min_scale)
        minmax_params.append(wrapper_linear.max_scale)
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": self.minmax_lr}], lr=self.lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=self.lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters, verbose=False
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        nsamples = len(inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        mse_loss = torch.nn.MSELoss().to(device)
        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_v, best_min_scale, best_max_scale = torch.tensor(0), torch.tensor(1.0), torch.tensor(1.0)
        gradient_accumulate_steps = self.train_bs  ##Force to low gpu
        train_bs = 1  ##Force to low gpu
        pick_samples = train_bs * gradient_accumulate_steps

        if self.sampler != "rand":
            whole_indices = torch.randperm(nsamples)[:pick_samples]
        for i in range(self.iters):
            total_loss = 0
            if self.sampler == "rand":
                whole_indices = torch.randperm(nsamples)[:pick_samples]
            for tmp_step in range(gradient_accumulate_steps):
                indices = whole_indices[tmp_step * train_bs: (tmp_step + 1) * train_bs]
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

                if self.amp:
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(output_q, current_output)  # pylint: disable=not-callable
                else:
                    output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                    loss = mse_loss(  # pylint: disable=not-callable
                        output_q.to(torch.float32), current_output.to(torch.float32)
                    )
                total_loss += loss.item() / gradient_accumulate_steps

                self.scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_v = copy.deepcopy(wrapper_linear.value.data)
                    best_min_scale = copy.deepcopy(torch.clamp(wrapper_linear.min_scale.data, 0, 1.0))
                    best_max_scale = copy.deepcopy(torch.clamp(wrapper_linear.max_scale.data, 0, 1.0))

                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_v = copy.deepcopy(wrapper_linear.value.data)
                best_min_scale = copy.deepcopy(torch.clamp(wrapper_linear.min_scale.data, 0, 1.0))
                best_max_scale = copy.deepcopy(torch.clamp(wrapper_linear.max_scale.data, 0, 1.0))

            if not self.not_use_best_mse:
                if self.dynamic_max_gap > 0 and i - last_best_iter >= self.dynamic_max_gap:
                    break
            self.step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        with torch.no_grad():
            unwrapper_layer(self.model, wrapper_linear, layer_name, best_v, best_min_scale, best_max_scale)
        layer = mv_module_from_gpu(layer, self.low_cpu_mem_usage)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        logger.info(dump_info)


    def quant_block_(self, block, block_inputs, block_outputs, device="cuda"):
        """Fine-tune the up and down rounding parameters of a given block.
        
        # More details: Section 3 at https://arxiv.org/pdf/2309.05516
        pseudocode
        wrapper_block = replace_linear_with_qdq_linear(model)
        best_tunable_params = {}
        for i in range(iters):
            qdq_output = wrapper_block(block_inputs)
            loss = mse_loss(qdq_output, block_outputs)
            loss.backward()
            optimizer.step()
            update_best_tunable_params(wrapper_block,loss)
        unwrapper_block = replace_qdq_linear_with_new_linear_using_qdq_weight_and_scale_zp(wrapper_block, best_tunable_params)
        """
        # TODO: unable to support self.use_quant_input by now.
        # TODO: enable amp
        block = block.to(device)
        block_inputs = to_device(block_inputs, device)
        block_outputs = to_device(block_outputs, device)
        freeze_mod_(block)
        quantized_layer_names, unquantized_layer_names = wrapper_block(block, self.enable_minmax_tuning, device)
        block_outputs = torch.cat(block_outputs, dim=0)

        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                round_params.append(m.value)
                minmax_params.append(m.min_scale)
                minmax_params.append(m.max_scale)

        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": self.minmax_lr}], lr=self.lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=self.lr, weight_decay=0)

        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters, verbose=False
        )

        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max

        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_v, best_min_scale, best_max_scale = torch.tensor(0), torch.tensor(1.0), torch.tensor(1.0)
        for i in range(self.iters):
            total_loss = 0
            mse_loss = torch.nn.MSELoss().to(device)

            qdq_block_outputs = get_block_outputs(block, block_inputs)

            qdq_block_outputs = _use_first_output(qdq_block_outputs)
            loss = mse_loss(qdq_block_outputs, block_outputs)

            scale_loss = loss * 1000

            scale_loss.backward()
            self.step(scaler, optimizer, lr_schedule)

            total_loss += loss.detach().item() / self.gradient_accumulate_steps
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_v = collect_round_v(block)
                    best_min_scale, best_max_scale = collect_minmax_scale(block)
                    last_best_iter = i
                    logger.info(f"Update best loss: {best_loss:.6f} at iter {last_best_iter}")
            if self.not_use_best_mse and i == self.iters - 1:
                best_v = collect_round_v(block)
                best_min_scale, best_max_scale = collect_minmax_scale(block)
                logger.info(f"Update best loss: {best_loss:.6f} at iter {last_best_iter}")

            if not self.not_use_best_mse:
                if self.dynamic_max_gap > 0 and i - last_best_iter >= self.dynamic_max_gap:
                    break

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
        if self.amp:
            block = block.to(torch.float32)
        unwrapper_block(block, best_v, best_min_scale, best_max_scale)
        return None, block_outputs
    
    
    def quant_block(self, block, input_ids, input_others, q_input=None, device=torch.device("cpu"), output=None):
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
        block = block.to(device)
        self.device = device
        if output is None:

            output = self.get_block_outputs(block, input_ids, input_others, self.train_bs * self.infer_bs_coeff, device,
                                            self.cache_device)

        if q_input is not None:
            input_ids = q_input

        quantized_layer_names, unquantized_layer_names = wrapper_block(
            block, self.enable_minmax_tuning, device=self.device)

        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                round_params.append(m.value)
                minmax_params.append(m.min_scale)
                minmax_params.append(m.max_scale)

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
            return output, output

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters, verbose=False
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        pick_samples = self.train_bs * self.gradient_accumulate_steps
        pick_batches = self.gradient_accumulate_steps
        nsamples = len(input_ids)
        if self.sampler != "rand":
            whole_indices = torch.randperm(nsamples)[:pick_samples]
            pick_batch_indices = torch.randperm(nsamples)[:pick_batches]
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        mse_loss = torch.nn.MSELoss().to(device)
        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_v, best_min_scale, best_max_scale = torch.tensor(0), torch.tensor(1.0), torch.tensor(1.0)

        for i in range(self.iters):
            total_loss = 0
            if self.sampler == "rand":
                whole_indices = torch.randperm(nsamples)[:pick_samples]
                pick_batch_indices = torch.randperm(nsamples)[:pick_batches]
            for i, tmp_step in enumerate(range(self.gradient_accumulate_steps)):
                indices = whole_indices[tmp_step * self.train_bs: (tmp_step + 1) * self.train_bs]
                current_input_ids, current_input_others = sampling_inputs(
                    input_ids,
                    input_others,
                    indices,
                    seqlen=self.seqlen,
                    share_attention_mask_flag=self.share_attention_mask_flag,
                    # not_share_position_ids_flag=self.not_share_position_ids_flag,
                    # input_dim=self.input_dim,
                )
                
                input_dim = 0

                current_output = [output[i:i+1] for i in indices]
                current_output = torch.cat(current_output, dim=input_dim)

                current_output = to_device(current_output, device)
                current_input_ids = to_device(current_input_ids, device)
                current_input_others = to_device(current_input_others, device)
                # output_q = block_forward(
                #     block, current_input_ids, current_input_others, self.amp, self.amp_dtype, device
                # )
                breakpoint()
                output_q = block(current_input_ids, **current_input_others)
                output_q = _use_first_output(output_q)
                if self.amp:
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                        loss = mse_loss(output_q, current_output)  # pylint: disable=not-callable
                else:
                    loss = mse_loss(  # pylint: disable=not-callable
                        output_q.to(torch.float32), current_output.to(torch.float32)
                    )

                total_loss += loss.item() / self.gradient_accumulate_steps
                self.scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)
                    best_v = collect_round_v(block)
                    best_min_scale, best_max_scale = collect_minmax_scale(block)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_v = collect_round_v(block)
                best_min_scale, best_max_scale = collect_minmax_scale(block)

            if not self.not_use_best_mse:
                if self.dynamic_max_gap > 0 and i - last_best_iter >= self.dynamic_max_gap:
                    break
            self.step(scaler, optimizer, lr_schedule)

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
            unwrapper_block(block, best_v, best_min_scale, best_max_scale)
        block = mv_module_from_gpu(block, self.low_cpu_mem_usage)
        del input_ids
        del input_others

        torch.cuda.empty_cache()
        if 0 and self.enable_quanted_input:
            if self.low_cpu_mem_usage:
                block = block.to(device)
            q_outputs = self.get_block_outputs(
                block, input_ids, input_others, self.train_bs * self.infer_bs_coeff, device,
                cache_device=self.cache_device
            )
            block = mv_module_from_gpu(block, self.low_cpu_mem_usage)
            for i in range(len(input_ids)):
                input_ids[i] = None
            torch.cuda.empty_cache()

            return q_outputs, output

        else:
            block = mv_module_from_gpu(block, self.low_cpu_mem_usage)
            # for i in range(len(input_ids)):
            #     input_ids[i] = None
            # torch.cuda.empty_cache()
            return None, output
        


    def quant_block_v2_(self, block, inputs, outputs, device=torch.device("cpu")):
        """Fine-tune the up and down rounding parameters of a given block.
        
        # More details: Section 3 at https://arxiv.org/pdf/2309.05516
        pseudocode
        wrapper_block = replace_linear_with_qdq_linear(model)
        best_tunable_params = {}
        for i in range(iters):
            qdq_output = wrapper_block(block_inputs)
            loss = mse_loss(qdq_output, block_outputs)
            loss.backward()
            optimizer.step()
            update_best_tunable_params(wrapper_block,loss)
        unwrapper_block = replace_qdq_linear_with_new_linear_using_qdq_weight_and_scale_zp(wrapper_block, best_tunable_params)
        """
        # TODO: unable to support self.use_quant_input by now.
        # TODO: enable amp
        amp_dtype = next(block.parameters()).dtype
        # self.scale_dtype = amp_dtype
        block = block.to(device)
        self.amp_dtype = amp_dtype
        self.device = device

        if isinstance(block, torch.nn.Linear):
            # TODO: WA for Linear, refactor later
            class FakeBlock(torch.nn.Module):
                def __init__(self, block):
                    super(FakeBlock, self).__init__()
                    self.block = block
                def forward(self, x):
                    return self.block(x)
            block = FakeBlock(block)

        for n, p, in block.named_parameters():
            p.requires_grad = False


        quantized_layer_names, unquantized_layer_names = wrapper_block(
            block, self.enable_minmax_tuning, device=self.device)

        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                round_params.append(m.value)
                minmax_params.append(m.min_scale)
                minmax_params.append(m.max_scale)

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
            return outputs, outputs

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters, verbose=False
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        pick_samples = self.train_bs * self.gradient_accumulate_steps
        pick_batches = self.gradient_accumulate_steps
        nsamples = len(inputs)
        if self.sampler != "rand":
            whole_indices = torch.randperm(nsamples)[:pick_samples]
            pick_batch_indices = torch.randperm(nsamples)[:pick_batches]
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        mse_loss = torch.nn.MSELoss().to(device)
        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_v, best_min_scale, best_max_scale = torch.tensor(0), torch.tensor(1.0), torch.tensor(1.0)

        for i in range(self.iters):
            total_loss = 0
            if self.sampler == "rand":
                whole_indices = torch.randperm(nsamples)[:pick_samples]
                pick_batch_indices = torch.randperm(nsamples)[:pick_batches]
            # For each step, we pick a batch of samples to train
            for index, tmp_step in enumerate(range(self.gradient_accumulate_steps)):
                tmp_indices = pick_batch_indices[index]
                cur_inputs = inputs[tmp_indices]
                cur_inputs = to_device(cur_inputs, device)
                cur_inputs = to_device_amp_dtype(cur_inputs, amp_dtype)
                cur_args, cur_kwargs = cur_inputs
                output_q = block.forward(*cur_args, **cur_kwargs)
                
                # Use the hidden_states to calculate the loss
                q_hidden_states = output_q[0] if isinstance(output_q, tuple) else output_q
                # For AO' llama:
                #   outputs[tmp_indices] (tensor, {})
                #   outputs: [(tensor, {}), (tensor, {}), ...]
                # For tansformers's Llama:
                #   outputs[tmp_indices] ((tensor,), {...})
                cur_batch_outputs =  outputs[tmp_indices] # It should be (tensor, {}) or ((tensor, ), {})
                first_outputs = cur_batch_outputs[0]
                cur_outputs = first_outputs[0] if isinstance(first_outputs, tuple) else first_outputs
                orig_hidden_states = cur_outputs[0] if isinstance(cur_outputs, tuple) else cur_outputs
                orig_hidden_states = to_device(orig_hidden_states, device)
                if self.amp:
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                        if np.prod(q_hidden_states.shape) != np.prod(orig_hidden_states.shape):
                            raise ValueError(
                                (f"The shape of the QDQ hidden states {q_hidden_states.shape} "
                                 f"and the original hidden states {orig_hidden_states.shape} are different."
                                 f"Please raise an issue on https://github.com/intel/auto-round/issues."
                                 )
                            )
                        loss = mse_loss(q_hidden_states, orig_hidden_states)  # pylint: disable=not-callable
                else:
                    loss = mse_loss(  # pylint: disable=not-callable
                        q_hidden_states.to(torch.float32), orig_hidden_states.to(torch.float32)
                    )

                total_loss += loss.item() / self.gradient_accumulate_steps
                self.scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)
                    best_v = collect_round_v(block)
                    best_min_scale, best_max_scale = collect_minmax_scale(block)
                    last_best_iter = i
                logging.info(f"Got better result at iter {i}, the loss is {total_loss}")
            if self.not_use_best_mse and i == self.iters - 1:
                best_v = collect_round_v(block)
                best_min_scale, best_max_scale = collect_minmax_scale(block)
                logging.info(f"Got better result at iter {i}, the loss is {total_loss}")

            if not self.not_use_best_mse:
                if self.dynamic_max_gap > 0 and i - last_best_iter >= self.dynamic_max_gap:
                    break
            self.step(scaler, optimizer, lr_schedule)

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
            unwrapper_block(block, best_v, best_min_scale, best_max_scale)
        # block = mv_module_from_gpu(block, self.low_cpu_mem_usage)
        block = block.to(self._original_dtype)
        del inputs
        return block



    def quant_blocks(
            self,
            model: torch.nn.Module,
            inputs,
            block_names,
            nblocks=1,
            device=torch.device("cpu"),
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
        q_input = None
        torch.cuda.empty_cache()
        for n, m in model.named_parameters():
            m.requires_grad_(False)
        input_ids = inputs["input_ids"]
        inputs.pop("input_ids", None)
        input_others = inputs
        torch.cuda.empty_cache()
        input_ids = to_device(input_ids, self.cache_device)
        input_others = to_device(input_others, self.cache_device)
        ## as in calibration phase, we may use bf16 for calibration due to low_gpu_memory usage
        tmp_dtype = self.amp_dtype if self.amp else torch.float32
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i].to(tmp_dtype)

        for key in input_others.keys():
            if isinstance(input_others[key], torch.Tensor) and (
                    input_others[key].dtype == torch.float16 or input_others[key].dtype == torch.bfloat16
            ):
                input_others[key] = input_others[key].to(tmp_dtype)
            elif isinstance(input_others[key], list):
                for i in range(len(input_others[key])):
                    input_others[key][i].to(tmp_dtype)

        for i in range(0, len(block_names), nblocks):
            if nblocks == 1:
                n = block_names[i]
                logger.info(f"quantizing {i + 1}/{len(block_names)}, {n}")
                m = get_module(model, n)
            else:
                names = block_names[i: i + nblocks]
                logger.info(names)
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            if not self.model.device.type == "meta" or self.low_cpu_mem_usage:
                m = m.to(device)

            q_input, input_ids = self.quant_block(
                m,
                input_ids,
                input_others,
                q_input=q_input,
                device=device,
            )

            torch.cuda.empty_cache()
        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)

        del q_input
        del input_ids
        del input_others
        del inputs

        torch.cuda.empty_cache()

    def save_quantized(self, output_dir=None, format="auto_gptq", inplace=True, **kwargs):
        """Save the quantized model to the specified output directory in the specified format.

        Args:
            output_dir (str, optional): The directory to save the quantized model. Defaults to None.
            format (str, optional): The format in which to save the model. Defaults to "auto_gptq".
            inplace (bool, optional): Whether to modify the model in place. Defaults to True.
            **kwargs: Additional keyword arguments specific to the export format.

        Returns:
            object: The compressed model object.
        """
        if self.low_cpu_mem_usage:
            self.model = self.model.to('cpu')

        if not self.quantized:
            logger.warning("please run autoround.quantize first")
            return
        if format == "fake" or format == "qdq" or self.act_bits <= 8:  ##TODO fix act quantizaiton later
            self.model = self.model.to("cpu")
            self.model.save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            return

        from auto_round.export import EXPORT_FORMAT
        backend = format
        format = format.split(":")[0]
        if format not in EXPORT_FORMAT:
            logger.error(f"export format only supports {EXPORT_FORMAT.keys()}")
            exit()
        save_quantized_as_format = EXPORT_FORMAT.get(format)
        serialization_keys = [
            "bits",
            "group_size",
            "sym",
            "data_type",
            "enable_quanted_input",
            "enable_minmax_tuning",
            "data_type",
            "seqlen",
            "train_bs",
            "scale_dtype",
            "lr",
            "minmax_lr",
            "gradient_accumulate_steps",
            "iters",
            "amp",
            "nsamples",
            "low_gpu_mem_usage",
        ]
        if isinstance(self.dataset, str):
            serialization_keys.append("dataset")
        serialization_dict = {}
        for key in serialization_keys:
            serialization_dict[key] = getattr(self, key)
        from .version import __version__

        serialization_dict["autoround_version"] = __version__
        if "scale_dtype" in serialization_dict.keys():
            serialization_dict["scale_dtype"] = str(serialization_dict["scale_dtype"])

        compressed_model = save_quantized_as_format(  ##TODO refine the code
            output_dir,
            model=self.model,
            layer_config=self.layer_config,
            inplace=inplace,
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
            iters=self.iters,
            lr=self.lr,
            minmax_lr=self.minmax_lr,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_quanted_input=self.enable_quanted_input,
            scale_dtype=self.scale_dtype,
            tokenizer=self.tokenizer,
            supported_types=self.supported_types,
            data_type=self.data_type,
            serialization_dict=serialization_dict,
            backend=backend,
            quant_block_list=self.quant_block_list,
            **kwargs
        )
        return compressed_model

    def get_quantized_layer_names_outside_blocks(self):
        """Gets the names of quantized layers outside blocks in the model.

        Returns:
            list: List of layer names outside blocks.
        """
        if self.layer_config is None or len(self.layer_config) == 0:
            return []

        layer_names = []
        all_layers_in_block = get_layer_names_in_block(self.model, self.supported_types, self.quant_block_list)

        for key in self.layer_config.keys():
            if key in all_layers_in_block:
                continue
            layer = get_module(self.model, key)
            if layer is None:
                logger.error(f"could not find layer {key} in the model, exit...")
                exit()
            if isinstance(layer, tuple(self.supported_types)) and check_to_quantized(self.layer_config[key]):
                layer_names.append(key)

        return layer_names

    def set_amp_dtype(self):
        self.amp_dtype = torch.float16
        if self.model.dtype != torch.float32:
            self.amp_dtype = self.model.dtype
        if self.device == "cpu" or "hpu" in self.device:
            self.amp_dtype = torch.bfloat16
        if self.amp:
            # Disable bf16 to remove the dependency on cpuinfo and psutil
            if self.device == "cpu": # and not CpuInfo().bf16:
                self.amp = False
                self.amp_dtype = torch.float32
                self.model = self.model.to(torch.float32)
                logger.warning(
                    f"amp is set to FALSE as the current {self.device} device does not support the 'bf16' data type."
                )
            else:
                self.model = self.model.to(self.amp_dtype)
        else:
            self.amp_dtype = torch.float32
            self.model = self.model.to(torch.float32)

    def get_optimizer(self, optimizer):
        """Returns the specified optimizer. In SignRound, we fix the optimizer.

        Args:
        optimizer: The optimizer to be used.

        Returns:
        The specified optimizer.
        """
        from auto_round.sign_sgd import SignSGD

        return SignSGD

    def get_scaler(self):
        """Returns scaler, in SignRound, no need to use scaler."""
        return None

    def scale_loss_and_backward(self, scaler, loss):
        """Scales the loss and performs backward pass.

        Args:
        scaler: The scaler to be used.
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        scale_loss.backward()
        if is_optimum_habana_available():
            htcore.mark_step()
        return scale_loss

    def step(self, scaler, optimizer, lr_schedule):
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
        if is_optimum_habana_available():
            htcore.mark_step()
        optimizer.zero_grad()
        lr_schedule.step()


class AutoOPTRound(AutoRound):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether sym to be used (default is False).
        layer_config (dict): Configuration for weight quantization (default is an empty dictionary).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset: The default dataset name (default is "NeelNanda/pile-10k").
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
        seed (int): The random seed (default is 42).
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

        **kwargs: Additional keyword arguments.

    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer=None,
            bits: int = 4,
            group_size: int = 128,
            sym: bool = False,
            layer_config: dict = {},
            enable_full_range: bool = False,
            batch_size: int = 8,
            amp: bool = True,
            device=None,
            lr_scheduler=None,
            dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
            enable_quanted_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = None,
            minmax_lr: float = None,
            low_gpu_mem_usage: bool = False,
            low_cpu_mem_usage: bool = False,
            iters: int = 200,
            seqlen: int = 2048,
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
            optimizer="AdamW",
            **kwargs,
    ):
        super(AutoOPTRound, self).__init__(
            model,
            tokenizer,
            bits,
            group_size,
            sym,
            layer_config,
            enable_full_range,
            batch_size,
            amp,
            device,
            lr_scheduler,
            dataset,
            enable_quanted_input,
            enable_minmax_tuning,
            lr,
            minmax_lr,
            low_gpu_mem_usage,
            low_cpu_mem_usage,
            iters,
            seqlen,
            nsamples,
            sampler,
            seed,
            nblocks,
            gradient_accumulate_steps,
            not_use_best_mse,
            dynamic_max_gap,
            data_type,
            scale_dtype,
            act_bits,
            act_group_size,
            act_sym,
            act_dynamic,
            **kwargs,
        )

        self.optimizer = self.get_optimizer(optimizer)

    def get_optimizer(self, optimizer):
        if optimizer is None:
            optimizer = torch.optim.AdamW
        elif isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)
        else:
            optimizer = optimizer
        return optimizer

    def get_scaler(self):
        scaler = None
        if self.amp and not check_is_cpu(self.device):
            from torch.cuda.amp import GradScaler

            scaler = GradScaler(init_scale=1024, growth_interval=100000)
        return scaler

    def scale_loss_and_backward(self, scaler, loss):
        if scaler is not None:
            loss = scaler.scale(loss)

        loss.backward()
        if is_optimum_habana_available():
            htcore.mark_step()
        return loss

    def step(self, scaler, optimizer, lr_schedule):
        if scaler is not None:
            scaler.step(optimizer)
            optimizer.zero_grad()
            lr_schedule.step()
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step()
        if is_optimum_habana_available():
            htcore.mark_step()


class AutoAdamRound(AutoOPTRound):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.
    The default lr has been changed.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (str): Whether symmetric quantization to be used (default is False).
        layer_config (dict): Configuration for weight quantization (default is an empty dictionary).
        enable_full_range (bool): Whether to enable full range quantization (default is False).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset (Union[str, list, tuple, torch.utils.data.DataLoader]):
                The default dataset name (default is "NeelNanda/pile-10k").
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
        seed (int): The random seed (default is 42).
        nblocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        optimizer: string or object
        scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                           have different choices.
        act_bits (int): Number of bits for activation quantization. Default is 32.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.

    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer=None,
            bits: int = 4,
            group_size: int = 128,
            sym: bool = False,
            layer_config: dict = {},
            enable_full_range: bool = False,
            batch_size: int = 8,
            amp: bool = True,
            device=None,
            lr_scheduler=None,
            dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
            enable_quanted_input: bool = True,
            enable_minmax_tuning: bool = True,
            lr: float = None,
            minmax_lr: float = None,
            low_gpu_mem_usage: bool = False,
            low_cpu_mem_usage: bool = False,
            iters: int = 200,
            seqlen: int = 2048,
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
            optimizer="AdamW",
            **kwargs,
    ):
        super(AutoAdamRound, self).__init__(
            model,
            tokenizer,
            bits,
            group_size,
            sym,
            layer_config,
            enable_full_range,
            batch_size,
            amp,
            device,
            lr_scheduler,
            dataset,
            enable_quanted_input,
            enable_minmax_tuning,
            lr,
            minmax_lr,
            low_gpu_mem_usage,
            low_cpu_mem_usage,
            iters,
            seqlen,
            nsamples,
            sampler,
            seed,
            nblocks,
            gradient_accumulate_steps,
            not_use_best_mse,
            dynamic_max_gap,
            data_type,
            scale_dtype,
            act_bits,
            act_group_size,
            act_sym,
            act_dynamic,
            optimizer,
            **kwargs,
        )



