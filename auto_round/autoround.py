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

import os
import torch
import transformers
import copy
import time
from typing import Optional, Union
from transformers import set_seed
from torch import autocast
from tqdm import tqdm
import accelerate
from packaging import version
from .quantizer import WrapperMultiblock, wrapper_block, unwrapper_block, WrapperLinear, unwrapper_layer, \
    WrapperTransformerConv1d
from .special_model_handler import (
    shareable_keywords,
    init_cache_for_special_model,
    reset_params,
    check_skippable_keywords
)
from .utils import (
    CpuInfo,
    block_forward,
    check_is_cpu,
    check_to_quantized,
    collect_best_params,
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
    unsupport_meta_device, clear_memory,
    compile_func,
    find_matching_blocks
)
from .low_cpu_mem.utils import get_layers_before_block

class AutoRound(object):
    """For more information, please refer to Cheng, Wenhua, et al. "Optimize weight rounding via signed gradient descent
     for the quantization of llms." arXiv preprint arXiv:2309.05516 (2023).

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether symmetric quantization is to be used (default is True).
        layer_config (dict): Configuration for weight quantization (default is None).
        layer_config={
                   'layer1':##layer_name
                   {
                       'data_type': 'int',
                       'bits': 4,
                       'group_size': 128,
                       'sym': True
                       'act_data_type': None,
                       'act_bits': 16,
                       'act_group_size': None,
                       'act_sym': None,

                   }
                   ...
               }
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
        act_bits (int): Number of bits for activation quantization. Default is 16.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A string or list whose elements are list of 
                            block's layer names to be quantized.
        enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer, torch>=2.6 True.
    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer,
            bits: int = 4,
            group_size: int = 128,
            sym: bool = True,
            layer_config: dict = None,
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
            act_bits: int = 16,
            act_group_size: int = None,
            act_sym: bool = None,
            act_dynamic: bool = True,
            to_quant_block_names: Union[str, list] = None,
            enable_norm_bias_tuning: bool = False,
            enable_torch_compile: bool = None,
            **kwargs,
    ):
        self.quantized = False
        self.model_orig_dtype = model.dtype
        self.seed = seed
        set_seed(self.seed)
        assert not unsupport_meta_device(model), (
            "AutoRound does not support for params on meta device."
            " Please use more gpus vis set `--device 0,1,2,3` or just use one gpu")

        ## important tuning hype-parameters
        self.amp = amp
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.nsamples = nsamples
        self.nblocks = nblocks
        self.bits = bits
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        self.group_size = group_size
        self.sym = sym
        self.low_gpu_mem_usage = low_gpu_mem_usage
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.layer_config = {} if layer_config is None else layer_config
        self.seqlen = seqlen
        self.batch_size, self.gradient_accumulate_steps = batch_size, gradient_accumulate_steps
        self.nblocks = nblocks
        self.dataset = dataset
        self.iters = iters
        if self.iters <= 0:
            logger.warning("iters must be positive, reset it to 200")
            self.iters = 200
        self.lr = lr or (1.0 / self.iters)  ##must after iter setting
        self.minmax_lr = minmax_lr or self.lr

        ##activation
        self.act_group_size = act_group_size if not (act_group_size is None) else self.group_size
        self.act_bits = act_bits if not (act_bits is None) else self.bits
        self.act_sym = act_sym if not (act_sym is None) else self.sym
        self.act_dynamic = act_dynamic

        self.data_type = data_type
        self.supported_types = [torch.nn.Linear, transformers.modeling_utils.Conv1D]
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = detect_device(device)
        self.scale_dtype = convert_dtype_str2torch(scale_dtype)
        self.set_amp_dtype()
        self.cache_device = torch.device("cpu") if self.low_gpu_mem_usage else self.device
        if not hasattr(self, 'to_quant_block_names'):
            all_blocks = get_block_names(model)
            self.to_quant_block_names = find_matching_blocks(model, all_blocks, to_quant_block_names)
        

        self.sampler = sampler
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.lr_scheduler = lr_scheduler
        self.optimizer = self.get_optimizer(None)
        self.batch_dim = None
        self.infer_bs_coeff = 1

        self.set_layerwise_config(self.layer_config)  ##better place in the end
        torch.set_printoptions(precision=3, sci_mode=True)
        self.check_configs()
        logger.info(f"using {self.model.dtype} for quantization tuning")
        self.enable_torch_compile = enable_torch_compile
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
        assert self.act_group_size == -1 or self.act_group_size >= 1, \
            "only supports positive group_size or -1(per channel)"
        assert self.batch_size > 0, "batch size must be positive"
        assert self.iters > 0, "iters must be positive"
        assert self.seqlen > 0, "seqlen must be positive"
        assert self.nblocks > 0, "nblocks must be positive"
        assert self.gradient_accumulate_steps > 0, "gradient accumulate step must be positive"
        assert self.act_dynamic is True, "only support dynamic quantization for activation currently"
        # assert self.tokenizer != None or self.dataloader != None
        if self.act_bits <= 8:
            logger.warning(
                "please save the quantized model to fake format "
                "as real deployment is not supported for activation quantization currently")

        if "mx_fp" in self.data_type:
            logger.warning(
                "please save the quantized model to fake format "
                "as real deployment is not supported for mx_fp datatype currently")

        if "mx_fp" in self.data_type and self.group_size != 32:
            logger.warning("mx_fp should only support group_size of 32 in real deployment")


        if self.nsamples < self.gradient_accumulate_steps * self.batch_size:
            self.batch_size = min(self.batch_size, self.nsamples)
            self.gradient_accumulate_steps = min(self.nsamples // self.batch_size, self.gradient_accumulate_steps)
            logger.warning(
                f"reset gradient_accumulate_steps to {self.gradient_accumulate_steps} as nsamples must equal or greater"
                " than gradient_accumulate_steps * batch_szie")

    def quantize(self):
        """Quantize the model and return the quantized model along with layer configurations.
        the entry of AutoRound.

        Returns:
        The quantized model and layer configurations.
        """

        if bool(self.to_quant_block_names):
            all_blocks = self.to_quant_block_names
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
        logger.info("start to cache block inputs")
        all_inputs = self.try_cache_inter_data_gpucpu(all_first_block_names, self.nsamples, layer_names=layer_names)
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model)  ##self.model.hf_device_map has not been changed
        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        logger.info("caching done")
        pbar = tqdm(range(0, sum([len(i) for i in all_blocks]), self.nblocks))
        for block_names in all_blocks:
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            keys = inputs.keys()
            input_id_str = [key for key in keys if key.startswith('hidden_state')]
            if len(input_id_str) != 1:
                raise RuntimeError(f"hidden_states arg mismatch error,"
                                   "please raise an issue in https://github.com/intel/auto-round/issues")
            inputs["input_ids"] = inputs.pop(input_id_str[0], None)
            clear_memory(self.inputs)

            if "input_ids" in inputs.keys():
                total_samples = len(inputs["input_ids"])
                self.n_samples = total_samples
                if total_samples < self.batch_size:
                    self.batch_size = total_samples
                    logger.warning(f"force the train batch size to {total_samples}")

            self.quant_blocks(
                self.model,
                inputs,
                block_names,
                nblocks=self.nblocks,
                device=self.device,
                pbar=pbar
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
                if int(m.bits) > 8:
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
                self.layer_config[n]["bits"] = 16
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
        enable_quanted_input = self.enable_quanted_input
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1 and enable_quanted_input:
            from accelerate.big_modeling import dispatch_model

            dispatch_model(self.model, self.model.hf_device_map)

        if enable_quanted_input:
            q_layer_inputs = self.try_cache_inter_data_gpucpu([], self.nsamples, layer_names=layer_names)
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.model)  ##self.model.hf_device_map has not been changed

        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        clear_memory()
        device = next(self.model.parameters()).device
        quant_layer = compile_func(self.quant_layer, device, self.enable_torch_compile)
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.cache_device)
            q_layer_input = q_layer_inputs[layer_name] if enable_quanted_input else None
            q_layer_input = to_device(q_layer_input, self.cache_device)
            quant_layer(layer_name, layer_input, q_layer_input, device=self.device)
            del layer_input
            clear_memory(q_layer_input)

    def set_layerwise_config(self, layer_config):
        """Sets the layer-wise configuration based on the provided layer_config.
           By default, only quantize layers in blocks.

        Args:
        layer_config: The layer configuration.

        Returns:
        None
        """
        layers_in_blocks = get_layer_names_in_block(self.model, self.supported_types, self.to_quant_block_names)
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
                layer_config[n]["bits"] = 16
                layer_config[n]["act_bits"] = 16

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
                self.batch_dim
            )
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device).to(
                cache_device
            )
            if self.batch_size == 1:
                output.append(tmp_output)
            else:
                output.extend(list(torch.split(tmp_output, 1, dim=self.batch_dim)))
        if self.low_gpu_mem_usage:
            clear_memory()

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
        from .calib_dataset import get_dataloader
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
                    exit(-1)
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
                        data_new[key] = to_dtype(data_new[key], self.model.dtype)
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
            except RuntimeError as error:
                logger.warning("When quantization encounters tensor" \
                               " shape mismatch error, you can try to avoid it with batch_size=1")
                logger.error(error)
                pass
            except Exception as error:
                raise error
            total_cnt += input_ids.shape[0] if len(input_ids.shape) > 1 else 1
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
                f"Insufficient number of samples collected may affect the quantification. "
                f"target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )

        # clean embed weight to save memory
        if self.low_cpu_mem_usage:
            for n, m in embed_layers:
                m = m.to("meta")

    @torch.no_grad()
    def try_cache_inter_data_gpucpu(self, block_names, nsamples, layer_names=None, last_cache_name=None):
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
        if layer_names is None:
            layer_names = []
        try:
            if not self.model.device.type == "meta":
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    pass
                else:
                    self.model = self.model.to(self.device)
            all_inputs = self.cache_inter_data(
                block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
            )
            self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
            clear_memory()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.info("switch to cpu to cache block inputs")
                if (("lm_head" in self.layer_config and self.layer_config["lm_head"]["bits"] < 16) or
                        self.__class__.__name__ == "AutoRoundMLLM"):
                    logger.warning(f"we strongly recommend using additional CUDA/HPU devices,e.g. "
                                   f"set `--device '0,1'` in our cmd line usage or "
                                   f"load the model with `device_mapping=auto`,"
                                   f" for optimal performance during calibration "
                                   f"Otherwise, the process may be significantly slower.")
                self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
                clear_memory()
                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
                )
            else:
                raise
        return all_inputs

    @torch.no_grad()
    def cache_inter_data(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Save the inputs of block_name for calibration.

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
        if layer_names is None:
            layer_names = []
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
        calib_bs = self.batch_size
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
            if data_name in shareable_keywords:
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
                init_cache_for_special_model(self.model, positional_inputs, self.inputs[name])

            if self.batch_dim is None:
                self.batch_dim = 0
                if hidden_states is not None and self.batch_size > 1:
                    if hidden_states.shape[0] > self.batch_size:
                        self.batch_dim = 1
                        if len(hidden_states.shape) > 1 and hidden_states.shape[1] > self.batch_size:
                            logger.error(
                                f"this model has not been supported, "
                                f"please raise an issue in https://github.com/intel/auto-round/issues"
                                f" or try to set the `batch_size` to 1 and "
                                f"`gradient_accumulate_steps` to your current batch size.")
                            exit(-1)

            if hidden_states is not None:
                kwargs['hidden_states'] = hidden_states

            for key in kwargs.keys():
                if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) \
                        or isinstance(kwargs[key], tuple):
                    if key not in self.inputs[name].keys():  # initialization
                        data = to_device(kwargs[key], device=torch.device("cpu"))
                        if data is None or (self.batch_size > 1 and key in shareable_keywords):
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
                        logger.warning_once(f"Please note that '{key}' key" \
                                            " is not currently used in quantization fine-tuning.")
            reset_params(self.inputs[name])
            if name == self.last_cache_name:
                raise NotImplementedError
            else:
                if hidden_states is not None:
                    kwargs.pop('hidden_states')
                    return m.orig_forward(hidden_states, *positional_inputs, **kwargs)
                else:
                    # Currently only for Llama-3.2-Vision-Instruct Series
                    return m.orig_forward(*positional_inputs, **kwargs)

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

        if isinstance(layer, torch.nn.Linear):
            wrapper_linear = WrapperLinear(layer, enable_minmax_tuning=self.enable_minmax_tuning, device=device).to(
                device)
        else:
            wrapper_linear = WrapperTransformerConv1d(layer, enable_minmax_tuning=self.enable_minmax_tuning,
                                                      device=device).to(device)
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
        # best_v, best_min_scale, best_max_scale = torch.tensor(0), torch.tensor(1.0), torch.tensor(1.0)
        gradient_accumulate_steps = self.batch_size  ##Force to low gpu
        batch_size = 1  ##Force to low gpu
        pick_samples = batch_size * gradient_accumulate_steps
        pick_samples = min(nsamples, pick_samples)
        if self.sampler != "rand":
            whole_indices = torch.randperm(nsamples)[:pick_samples]
        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)

        for i in range(self.iters):
            total_loss = 0
            if self.sampler == "rand":
                whole_indices = torch.randperm(nsamples)[:pick_samples]
                if gradient_accumulate_steps != 1:
                    if q_inputs is not None:
                        current_input = [q_inputs[i] for i in whole_indices]
                    else:
                        current_input = [inputs[i] for i in whole_indices]
                    num_elm = sum(id.numel() for id in current_input)
            for tmp_step in range(gradient_accumulate_steps):
                indices = whole_indices[tmp_step * batch_size: (tmp_step + 1) * batch_size]
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
                total_loss += loss.item() / num_elm

                self.scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(wrapper_linear)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(wrapper_linear)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self.step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        with torch.no_grad():
            unwrapper_layer(self.model, wrapper_linear, layer_name, best_params)
        mv_module_from_gpu(layer, self.low_cpu_mem_usage)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        logger.debug(dump_info)

    def quant_block(self, block, input_ids, input_others, q_input=None, device=torch.device("cpu")):
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

        output = self.get_block_outputs(block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device,
                                        self.cache_device)

        if q_input is not None:
            input_ids = q_input

        quantized_layer_names, unquantized_layer_names = wrapper_block(
            block, self.enable_minmax_tuning, self.enable_norm_bias_tuning, device=self.device)

        round_params = []
        minmax_params = []
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                if "v" in m.params.keys():
                    round_params.append(m.params['v'])
                if "max_scale" in m.params.keys():
                    minmax_params.append(m.params["min_scale"])
                    minmax_params.append(m.params["max_scale"])
                if "bias_v" in m.params.keys():
                    round_params.append(m.params["bias_v"])

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
        scaler = self.get_scaler()  # pylint: disable=assignment-from-none
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
                indices = whole_indices[tmp_step * self.batch_size: (tmp_step + 1) * self.batch_size]
                current_input_ids, current_input_others = sampling_inputs(
                    input_ids,
                    input_others,
                    indices,
                    seqlen=self.seqlen,
                    batch_dim=self.batch_dim,
                )

                current_output = [output[x] for x in indices]
                current_output = torch.cat(current_output, dim=self.batch_dim)

                current_output = to_device(current_output, device)

                output_q = block_forward(
                    block, current_input_ids, current_input_others, self.amp, self.amp_dtype, device
                )
                if self.amp:
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                        loss = mse_loss(output_q, current_output)  # pylint: disable=not-callable
                else:
                    loss = mse_loss(  # pylint: disable=not-callable
                        output_q.to(torch.float32), current_output.to(torch.float32)
                    )

                total_loss += loss.item() / num_elm
                self.scale_loss_and_backward(scaler, loss)
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
        logger.debug(dump_info)
        if len(unquantized_layer_names) != 0:
            logger.info(f"{unquantized_layer_names} have not been quantized")
        with torch.no_grad():
            unwrapper_block(block, best_params)
        if self.enable_quanted_input:
            if self.low_cpu_mem_usage:
                block = block.to(device)
            clear_memory()
            q_outputs = self.get_block_outputs(
                block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device,
                cache_device=self.cache_device
            )
            mv_module_from_gpu(block, self.low_cpu_mem_usage)
            clear_memory(input_ids)

            return q_outputs, output

        else:
            mv_module_from_gpu(block, self.low_cpu_mem_usage)
            clear_memory(input_ids)
            return None, output

    def quant_blocks(
            self,
            model: torch.nn.Module,
            inputs,
            block_names,
            nblocks=1,
            device=torch.device("cpu"),
            pbar=None
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
        clear_memory()
        for n, m in model.named_parameters():
            m.requires_grad_(False)
        input_ids = inputs["input_ids"]
        inputs.pop("input_ids", None)
        input_others = inputs
        clear_memory()
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
                    to_dtype(input_others[key][i], tmp_dtype)
        quant_block = compile_func(self.quant_block, device, self.enable_torch_compile)

        if pbar is None:
            pbar = tqdm(range(0, len(block_names), nblocks))
        # for i in pbar:
        for i in range(len(block_names)):
            if nblocks == 1:
                n = block_names[i]
                pbar.set_description(f"Quantizing {n}")
                m = get_module(model, n)
            else:
                names = block_names[i: i + nblocks]
                pbar.set_description(f"Quantizing [{i + 1}-{i + nblocks}]/{len(block_names)}")
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            if not self.model.device.type == "meta" or self.low_cpu_mem_usage:
                m = m.to(device)

            q_input, input_ids = quant_block(
                m,
                input_ids,
                input_others,
                q_input=q_input,
                device=device,
            )
            pbar.update(1)

        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)

        del q_input
        del input_ids
        del input_others
        del inputs

        clear_memory()

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
            raise ValueError(f"export format only supports {EXPORT_FORMAT.keys()}, but got {format}")
        save_quantized_as_format = EXPORT_FORMAT.get(format)
        if "gptq" in format and not self.sym:
            logger.warning(
                "The asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                " particularly for 2-bit quantization and smaller models."
                " We recommend exporting to either the AutoAWQ format (4 bits) or "
                "the AutoRound format (2 bits) to enhance performance."
            )
        if "awq" in format and not self.bits == 4:
            raise ValueError("The AWQ format only supports W4 quantization ")

        serialization_keys = [
            "bits",
            "group_size",
            "sym",
            "data_type",
            "enable_quanted_input",
            "enable_minmax_tuning",
            "data_type",
            "seqlen",
            "batch_size",
            "scale_dtype",
            "lr",
            "minmax_lr",
            "gradient_accumulate_steps",
            "iters",
            "amp",
            "nsamples",
            "low_gpu_mem_usage",
            "to_quant_block_names",
            "enable_norm_bias_tuning"
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
            to_quant_block_names=self.to_quant_block_names,
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
        all_layers_in_block = get_layer_names_in_block(self.model, self.supported_types, self.to_quant_block_names)

        for key in self.layer_config.keys():
            if key in all_layers_in_block:
                continue
            layer = get_module(self.model, key)
            if layer is None:
                logger.error(f"could not find layer {key} in the model, exit...")
                exit(-1)
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
            if self.device == "cpu" and not CpuInfo().bf16:
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


class AutoRoundOPT(AutoRound):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.

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
        act_bits (int): Number of bits for activation quantization. Default is 16.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A string or list whose elements are list of 
                            block's layer names to be quantized.
        enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer function
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
            sym: bool = True,
            layer_config=None,
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
            act_bits: int = 16,
            act_group_size: int = None,
            act_sym: bool = None,
            act_dynamic: bool = True,
            to_quant_block_names: Union[str, list] = None,
            enable_norm_bias_tuning: bool = False,
            enable_torch_compile: bool = None,
            optimizer="AdamW",
            **kwargs,
    ):
        super(AutoRoundOPT, self).__init__(
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
            to_quant_block_names=to_quant_block_names,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_torch_compile=enable_torch_compile,
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


class AutoRoundAdam(AutoRoundOPT):
    """Class for automatic rounding-based quantization with optimizers like adamw of a PyTorch model.
    The default lr has been changed.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (str): Whether symmetric quantization to be used (default is True).
        layer_config (dict): Configuration for weight quantization (default is None).
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
        act_bits (int): Number of bits for activation quantization. Default is 16.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A list whose elements are list of block's layer names to be quantized.
        enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer function
    Returns:
        The quantized model.
    """

    def __init__(
            self,
            model,
            tokenizer=None,
            bits: int = 4,
            group_size: int = 128,
            sym: bool = True,
            layer_config=None,
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
            act_bits: int = 16,
            act_group_size: int = None,
            act_sym: bool = None,
            act_dynamic: bool = True,
            to_quant_block_names: Union[str, list] = None,
            enable_norm_bias_tuning: bool = False,
            enable_torch_compile: bool = None,
            optimizer="AdamW",
            **kwargs,
    ):
        super(AutoRoundAdam, self).__init__(
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
            to_quant_block_names=to_quant_block_names,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_torch_compile=enable_torch_compile,
            optimizer=optimizer,
            **kwargs,
        )


