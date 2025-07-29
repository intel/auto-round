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
import os
import re
import sys
import time
from typing import Any, Union

import accelerate
import torch
from torch import autocast
from tqdm import tqdm
from transformers import set_seed

from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.export.export_to_gguf.config import GGUF_CONFIG, GGUF_INNER_CONFIG, ModelType
from auto_round.low_cpu_mem.utils import get_layers_before_block
from auto_round.utils import (
    SUPPORTED_DTYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    _gguf_args_check,
    block_forward,
    check_is_cpu,
    check_need_act_calibration,
    check_seqlen_compatible,
    check_skippable_keywords,
    check_to_quantized,
    clear_memory,
    collect_best_params,
    compile_func,
    convert_dtype_str2torch,
    detect_device,
    find_matching_blocks,
    flatten_list,
    get_block_names,
    get_layer_config_by_gguf_format,
    get_layer_features,
    get_layer_names_in_block,
    get_lm_head_name,
    get_module,
    get_shared_keys,
    htcore,
    infer_bits_by_data_type,
    init_cache,
    is_debug_mode,
    is_optimum_habana_available,
    llm_load_model,
    logger,
    mv_module_from_gpu,
    reset_params,
    set_module,
    to_device,
    to_dtype,
    unsupport_meta_device,
)
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block


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
        act_data_type (str): Specifies the data type for activations.
                             Defaults to None, in which case it inherits the weight data type.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A string or list whose elements are list of
                            block's layer names to be quantized.
        enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer (default it False).
        device_map (str|dict): device map for each block
        disable_opt_rtn (bool): Whether to disable optimization of the RTN mode(iters=0) (default is False).
    Returns:
        The quantized model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
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
        scale_dtype: Union[str, torch.dtype] = "fp16",
        act_bits: int = 16,
        act_group_size: int = None,
        act_sym: bool = None,
        act_data_type: str = None,
        act_dynamic: bool = True,
        to_quant_block_names: Union[str, list] = None,
        enable_norm_bias_tuning: bool = False,
        enable_torch_compile: bool = False,
        device_map: Union[str, dict] = None,
        super_bits: int = None,
        super_group_size: int = None,
        disable_opt_rtn: bool = False,
        model_kwargs: dict = None,
        **kwargs,
    ):
        self.vlm = kwargs.pop("vlm") if "vlm" in kwargs else False
        if kwargs:
            logger.warning(f"unrecognized keys {list(kwargs.keys())} were passed. Please check them.")
        self.quantized = False
        self.model_orig_dtype = model.dtype
        self.seed = seed
        set_seed(self.seed)
        if unsupport_meta_device(model):
            raise RuntimeError(
                "AutoRound does not support parameters on meta device. "
                "Please use more GPUs by setting `--device 0,1,2,3` or just use one GPU."
            )

        ## important tuning hype-parameters
        self.amp = amp
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.nsamples = nsamples
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
        if self.iters < 0:
            logger.warning("`iters` must be non-negative, reset it to 200")
            self.iters = 200
        if self.iters == 0:
            self.lr = 5e-3
        else:
            self.lr = lr or (1.0 / self.iters)  ##must after iter setting
        self.minmax_lr = minmax_lr or self.lr

        self.data_type = data_type
        tmp_bits = infer_bits_by_data_type(self.data_type)
        if tmp_bits < 16 and tmp_bits != bits:
            logger.warning(
                f"bits set in 'data_type' do not match the specified 'bits' setting. Resetting 'bits' to {tmp_bits}."
            )
            self.bits = tmp_bits
        self.supported_types = SUPPORTED_LAYER_TYPES
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = detect_device(device)
        self.scale_dtype = convert_dtype_str2torch(scale_dtype)
        self.set_amp_dtype()
        self.to_quant_block_names = to_quant_block_names
        if not hasattr(self, "quant_block_list"):
            all_blocks = get_block_names(model)
            self.quant_block_list = find_matching_blocks(model, all_blocks, self.to_quant_block_names)
        self.cache_device = torch.device("cpu") if self.low_gpu_mem_usage else self.device

        ##activation
        self.act_group_size = act_group_size if act_group_size is not None else group_size
        self.act_bits = act_bits if act_bits is not None else self.bits
        self.act_sym = act_sym if act_sym is not None else self.sym
        self.act_dynamic = act_dynamic
        self.act_data_type = act_data_type
        if self.act_data_type is None:
            if data_type in SUPPORTED_DTYPES and self.act_bits < 16:
                self.act_data_type = data_type
                logger.info(f"activation adopts {data_type}")
            else:
                self.act_data_type = "float"

        tmp_act_bits = infer_bits_by_data_type(self.act_data_type)
        if tmp_act_bits < 16 and tmp_act_bits != self.act_bits:
            self.act_bits = tmp_act_bits
            logger.warning(
                f"act_bits set in 'act_data_type' do not"
                f" match the specified 'act_bits' setting. Resetting 'act_bits' to {tmp_act_bits}."
            )

        self.sampler = sampler
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.lr_scheduler = lr_scheduler
        self.optimizer = self.get_optimizer(None)
        self.batch_dim = None
        self.infer_bs_coeff = 1

        self.super_bits = super_bits
        self.super_group_size = super_group_size

        self.disable_opt_rtn = disable_opt_rtn

        torch.set_printoptions(precision=3, sci_mode=True)
        self.check_configs()
        if self.act_bits <= 8 and self.amp_dtype == torch.float16:
            logger.warning("force to use bf16 to for quantization tuning when enabling activation quantization")
            self.amp_dtype = torch.bfloat16
            self.model = self.model.to(torch.bfloat16)
        else:
            logger.info(f"using {self.model.dtype} for quantization tuning")

        self.enable_torch_compile = enable_torch_compile
        if (
            not self.enable_torch_compile
            and TORCH_VERSION_AT_LEAST_2_6
            and self.act_bits > 8
            and not is_debug_mode()
            and not self.low_cpu_mem_usage
            and "fp8" not in self.data_type
            and "fp8" not in self.act_data_type
        ):
            logger.info(
                "'enable_torch_compile' is set to `False` by default. "
                "Enabling it can reduce tuning cost by 20%, but it might throw an exception."
            )

        if self.act_bits <= 8 and self.enable_torch_compile:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as activation quantization is enabled")

        if self.low_cpu_mem_usage and self.enable_torch_compile:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as low_cpu_mem_usage is enabled")

        if is_debug_mode() and self.enable_torch_compile:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as debug mode is enabled")

        if ("fp8" in self.data_type or "fp8" in self.act_data_type) and self.enable_torch_compile:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as fp8 is enabled")

        if is_optimum_habana_available():
            logger.info("Optimum Habana is available, import htcore explicitly.")
            import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401]
        self.device_map = device_map

        self.set_device_map_in_blocks(self.device_map)

        self.is_packing_immediate = False  ## whether to pack the layer immediately after tuning

        self.serialization_keys = [
            "bits",
            "group_size",
            "sym",
            "data_type",
            "enable_quanted_input",
            "enable_minmax_tuning",
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
            "enable_norm_bias_tuning",
            "act_bits",
            "act_group_size",
            "act_sym",
            "act_dynamic",
            "act_data_type",
            "super_bits",
            "super_group_size",
        ]

        self.shared_cache_keys = get_shared_keys(self.model)

    def set_device_map_in_blocks(self, device_map):
        """Sets the device map for specific blocks in the model.

        Args:
            device_map (Union[str, dict]): A mapping of module names to devices.
                If provided as a string, it should be in the format
                "module_name:device,module_name:device". Devices can be integers
                (GPU IDs) or strings (e.g., 'cpu', 'cuda:0').
        """
        if self.device_map is None or len(self.device_map) == 0:
            self.device_map = None
        if not device_map:
            return
        if isinstance(device_map, str):
            device_map = device_map.replace(" ", "")
            infos = device_map.split(",")
            device_map_dict = {}
            for info in infos:
                index = info.find(":")
                key = info[:index]
                value = info[index + 1 :]
                device_map_dict[key] = value
            device_map = device_map_dict

        names = [n for n, m in self.model.named_modules() if len(list(m.children())) == 0]

        for key, device in device_map.items():
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            device = detect_device(device)
            try:
                module = get_module(self.model, key)
                module.tuning_device = device
            except:
                matching_names = [name for name in names if re.match(key, name)]
                if len(matching_names) > 0:
                    for name in matching_names:
                        self._set_device_for_matching_module(name, device)
                else:
                    for name in names:
                        if key in name:
                            self._set_device_for_matching_module(name, device)

    def _set_device_for_matching_module(self, name, device):
        module = get_module(self.model, name)
        if hasattr(module, "tuning_device") and module.tuning_device != device:
            logger.warning(
                f"Multiple devices have been set for layer {name}, keeping original device {module.tuning_device}"
            )
        else:
            module.tuning_device = device

    def _dq_check(self):
        """Reset the default value of super_bits and super_group_size"""
        if self.data_type.endswith("_dq"):
            gguf_config = GGUF_INNER_CONFIG[f"gguf:q{self.bits}_k"]
            self.super_bits = gguf_config["super_bits"] if self.super_bits is None else self.super_bits
            self.super_group_size = (
                gguf_config["super_group_size"] if self.super_group_size is None else self.super_group_size
            )

    def check_configs(self):
        """Checks if the configurations are valid.

        Raises:
        ValueError, TypeError: If any of the configurations are invalid.
        """
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.act_bits <= 0:
            raise ValueError("act_bits must be positive")
        if not (self.group_size == -1 or self.group_size >= 0):
            raise ValueError("group_size must be -1 (per channel) or 0 (per-tensor) or a postivie integer")
        if not (self.act_group_size == -1 or self.act_group_size >= 0):
            raise ValueError("act_group_size must be -1 (per channel) or 0 (per-tensor) or a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.iters < 0:
            raise ValueError("iters must be non-negative")
        if self.seqlen <= 0:
            raise ValueError("seqlen must be positive")
        if self.nblocks <= 0:
            raise ValueError("nblocks must be positive")
        if self.gradient_accumulate_steps <= 0:
            raise ValueError("gradient_accumulate_steps must be positive")
        # assert self.tokenizer != None or self.dataloader != None
        if self.act_bits <= 8:
            logger.warning(
                "activation quantization is an experimental feature with limited support and a complex API. "
                "And please save the quantized model to fake format as real deployment is not supported currently"
            )

        if "mx_fp" in self.data_type or "nv_fp" in self.data_type:
            logger.warning(
                "please save the quantized model to fake format "
                "as real deployment is not supported for mx_fp/nv_fp datatype currently"
            )

        if "mx_fp" in self.data_type and self.group_size != 32:
            logger.warning("mx_fp should only support group_size of 32 in real deployment")

        if "nv_fp" in self.data_type and (self.group_size != 16):
            logger.warning("nv_fp should only support group_size of 16 in real deployment")

        if self.nsamples < self.gradient_accumulate_steps * self.batch_size:
            if self.batch_size > self.nsamples:
                logger.warning(
                    f"reset batch_size to {self.nsamples} as nsamples({self.nsamples})"
                    f" is smaller than batch_size({self.batch_size})"
                )
                self.batch_size = self.nsamples
            if self.gradient_accumulate_steps > self.nsamples // self.batch_size:
                self.gradient_accumulate_steps = self.nsamples // self.batch_size
                logger.warning(
                    f"reset gradient_accumulate_steps to {self.gradient_accumulate_steps}"
                    f" as nsamples must equal or greater"
                    f" than gradient_accumulate_steps * batch_size"
                )

        if self.enable_norm_bias_tuning:
            logger.warning("the `enable_norm_bias_tuning` feature is experimental and currently has limited support.")

        self._dq_check()

    def _check_compatibility(self):
        ##check gguf and others
        has_gguf = False
        if hasattr(self, "formats"):
            has_besides_gguf = False
            for format_ in self.formats:
                if "gguf" in format_:
                    has_gguf = True
                elif format_ != "fake":
                    has_besides_gguf = True
            if has_gguf and has_besides_gguf:
                raise ValueError("gguf format is not compatible with other formats, please choose only one of them")
            if has_gguf and self.iters != 0 and self.bits != 3:
                logger.warning(
                    "`iters=0` is recommended when exporting to GGUF format except for bits 3,"
                    " as we have optimized the RTN method for this case."
                    " We are likely to release new algorithm for certain configurations in the future."
                )

        ##check group_size 32 for auto_round
        if (
            self.data_type == "int"
            and hasattr(self, "formats")
            and any(key in fmt for fmt in self.formats for key in ("auto_round", "auto_gptq", "auto_awq"))
        ):
            for n, m in self.model.named_modules():
                if isinstance(m, self.supported_types):
                    if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                        self.layer_config[n] = {"bits": 16}
                        logger.info(
                            f"{n} will not be quantized due to its shape not being divisible by 32,"
                            " resulting in an exporting issue to autogptq"
                        )

        if (
            self.seqlen is not None
            and hasattr(self.model, "config")
            and hasattr(self.model.config, "max_position_embeddings")
        ):
            if self.model.config.max_position_embeddings < self.seqlen:
                logger.warning(
                    f"change sequence length to {self.model.config.max_position_embeddings} "
                    "due to the limitation of max_position_embeddings"
                )
                self.seqlen = min(self.seqlen, self.model.config.max_position_embeddings)

        if self.seqlen is not None and hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length < self.seqlen:
                logger.warning(
                    f"change sequence length to {self.tokenizer.model_max_length} "
                    "due to the limitation of model_max_length. "
                    "You can also try to increase the model_max_length to avoid this issue."
                )
                self.seqlen = min(self.seqlen, self.tokenizer.model_max_length)

        if self.group_size == 0 and "fp8" not in self.data_type:
            logger.warning("group_size of 0 is not supported for data_type other than fp8 ")

    def parse_format_to_list(self, format: str) -> list:
        """Parses the format string into a list of formats.

        This method checks the requested format(s) against the model's
        quantization settings and adjusts them if necessary. It ensures that
        the formats are compatible with the model's data type, bit width,
        and activation quantization settings.

        Args:
            format (str): The requested format(s) for quantization, separated by commas.

        Returns:
            list: A list of validated and updated formats.
        """
        _gguf_args_check(self, format, model_type=ModelType.TEXT)
        if self.vlm:
            _gguf_args_check(self, format, model_type=ModelType.MMPROJ)

        formats = format.replace("q*_", f"q{self.bits}_").replace(" ", "").split(",")
        from auto_round.utils import SUPPORTED_FORMATS

        for format_ in formats:
            if format_ not in SUPPORTED_FORMATS:
                logger.error(f"Unsupported format {format_}, please choose from {SUPPORTED_FORMATS}")
                exit(-1)
        if self.scale_dtype != torch.float32:
            only_gguf = True
            for format_ in formats:
                if not ("gguf" in format_ or "fake" in format_):
                    only_gguf = False
                    break
            if len(formats) == 1 and "fake" == formats[0]:
                only_gguf = False
            if only_gguf:
                self.scale_dtype = torch.float32
                logger.info("change `scale_dtype` to `torch.float32`")

        # Adjust format settings based on compatibility
        for index in range(len(formats)):
            format = formats[index]
            if format == "auto_round":
                if self.sym and "int" in self.data_type:
                    format = format.replace("auto_round", "auto_round:auto_gptq")
                    formats[index] = format
                if self.bits == 4 and not self.sym and "int" in self.data_type:
                    enable_awq = all(
                        config["bits"] == self.bits or config["bits"] >= 16 for config in self.layer_config.values()
                    )
                    if enable_awq:
                        formats[index] = format.replace("auto_round", "auto_round:auto_awq")
                if "fp8" in self.data_type:
                    format = format.replace("auto_round", f"auto_round:{self.data_type}")
                    formats[index] = format

        # Remove duplicates from formats list
        def remove_duplicates(lst):
            seen = set()
            return [x for x in lst if not (x in seen or seen.add(x))]

        formats = remove_duplicates(formats)
        for i in range(len(formats)):
            formats[i] = self._check_supported_format(formats[i])
        formats = remove_duplicates(formats)
        return formats

    def _check_supported_format(self, format: str) -> bool:
        """Checks if the specified format is supported.

        This method validates the requested format against the model's bit width,
        group size, symmetry, and activation quantization settings. It raises an
        error if the format is incompatible with the current model configuration.

        Args:
            format (str): The requested format for quantization.

        Returns:
            bool: True if the format is supported, False otherwise.
        """
        format = format.replace("q*_", f"q{self.bits}_")
        # only support to export afp8
        if self.act_bits <= 8:
            if "fp8" not in self.act_data_type or self.act_dynamic:
                if format == "llmcompressor":
                    bits, group_size, sym, act_bits = 8, -1, True, 8
                    assert (
                        self.bits == bits
                        and self.group_size == group_size
                        and self.sym == sym
                        and self.act_bits == act_bits
                        and self.act_dynamic
                    ), (
                        f"Currently only support to export llmcompressor format for dynamic quantized"
                        f" W{self.bits}A{self.act_bits} model, but got bits={self.bits},"
                        f" group_size={self.group_size}, sym={self.sym}, act_bits={self.act_bits}"
                    )
                elif format != "fake":
                    logger.warning(
                        "Currently only support to export auto_round format quantized model"
                        " with fp8 dtype activation for activation quantization."
                        " Change format to fake and save."
                    )
                    format = "fake"
            else:
                if not (format == "auto_round" or format == "auto_round:fp8"):
                    logger.warning(
                        f"Currently only support to export auto_round format for static W{self.bits}AFP8 model,"
                        " change format to auto_round"
                    )
                    format = "auto_round"
            if self.act_group_size != 0 and not self.act_dynamic and format == "auto_round:fp8":
                logger.warning(
                    f"Please note that quantize activation with act_group_size={self.act_group_size}"
                    " may result in failure to export or import normally."
                )
        if re.search(r"q\d_k", format) and not self.data_type.endswith("_dq"):
            logger.error(
                f"datatype<{self.data_type}> not support to export {format} format."
                " Please change export format or `data_type`."
            )
            sys.exit(-1)

        return format

    def quantize_and_save(self, output_dir: str = "tmp_autoround", format: str = "auto_round", inplace=True, **kwargs):
        """Quantizes the model and saves it in the specified format(s).

        This function checks the validity of the requested format(s), quantizes
        the model accordingly, and saves it to the specified output directory.
        If multiple formats are provided, the model is saved separately for each format.

        Args:
            output_dir (str, optional): The directory where the quantized model
                will be saved. Defaults to "tmp_autoround".
            format (str, optional): The quantization format(s) to use, separated
                by commas if multiple. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place if only
                one format is used. Defaults to True.
            **kwargs: Additional arguments for the quantization and saving process.

        Returns:
            model: A qdq model or packed model based on the configurations
            folders: The folder paths where the quantized models are saved.

        Raises:
            ValueError: If an unsupported format is specified.
        """
        # Validate and process the specified formats
        self.orig_output_dir = output_dir

        # check and update the format based on the current configuration
        format_list = self.parse_format_to_list(format)
        self.formats = format_list

        # If multiple formats are specified, enforce inplace=False
        if len(format_list) > 1:
            inplace = False
        self.inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Perform model quantization
        model, _ = self.quantize()

        # Save the quantized model in the specified format_list
        folders = []
        for format in format_list:
            if "gptq" in format and not self.sym:
                logger.warning(
                    "The asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                    " particularly for 2-bit quantization and smaller models."
                    " We recommend exporting to either the AutoAWQ format ( only 4 bits) or "
                    "the AutoRound format(2/3/4/8 bits)."
                )
            save_folder = self.get_save_folder_name(format)
            self.save_quantized(save_folder, format=format, inplace=inplace, **kwargs)

            folders.append(save_folder)

        return model, folders

    def get_save_folder_name(self, format_str: str) -> str:
        """Generates the save folder name based on the provided format string.

        If there are multiple formats to handle, the function creates a subfolder
        named after the format string with special characters replaced. If there's
        only one format, it returns the original output directory directly.

        Args:
            format_str (str): The format identifier (e.g., 'gguf:q2_k_s').

        Returns:
            str: The path to the folder where results should be saved.
        """
        # Replace special characters to make the folder name filesystem-safe
        sanitized_format = format_str.replace(":", "-").replace("_", "-")

        # Use a subfolder only if there are multiple formats
        if len(self.formats) > 1:
            return os.path.join(self.orig_output_dir, sanitized_format)

        return self.orig_output_dir

    @torch.inference_mode()
    def quantize_embedding_layer(self):
        """Quantizes embedding layers in the model according to the configuration.

        This method iterates through all modules in the model, identifies embedding
        layers specified in `self.layer_config`, and applies the appropriate quantization
        function based on bit precision, grouping strategy, and dtype.

        Returns:
            bool: True if the quantization process completes without critical errors.
        """
        is_quantized = False
        for name, module in self.model.named_modules():
            # Skip non-Embedding modules or layers not in config
            if not isinstance(module, torch.nn.Embedding) or name not in self.layer_config:
                continue

            config = self.layer_config[name]

            # Skip layers that are not marked for quantization
            if not check_to_quantized(config):
                continue
            is_quantized = True
            config["scale_dtype"] = self.scale_dtype
            dtype = config["data_type"]

            # Determine quantization function key with symmetry/asymmetry
            if dtype not in QUANT_FUNC_WITH_DTYPE:
                dtype = f"{dtype}_{'sym' if config['sym'] else 'asym'}"

            # Optionally use optimized rounding (RTN) variant
            if not self.disable_opt_rtn and f"rtn_{dtype}" in QUANT_FUNC_WITH_DTYPE:
                dtype = f"rtn_{dtype}"

            quant_func = QUANT_FUNC_WITH_DTYPE[dtype]

            # Attempt quantization on GPU, fall back to CPU if OOM
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(self.device),
                    **{k: config[k] for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]},
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "MODULE:PT_DEVMEM" in str(e):
                    logger.info("out of VRAM, falling back to CPU.")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config[k]
                            for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                        },
                    )
                else:
                    raise

            # Overwrite the module's weights with the quantized version
            module.weight.data.copy_(weight.cpu())

            # Attach scale and zero point (zp) to the module
            for param_name, value in zip(["scale", "zp"], [scale, zp]):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
                else:
                    setattr(module, param_name, value.cpu())

            # Update config
            self.layer_config.setdefault(name, {}).update(config)

            # Release memory
            clear_memory()

        return is_quantized

    def quant_rtn_with_imatrix(self, all_to_quantized_module_names: list[str]) -> None:
        """Performs RTN quantization using input activation statistics (imatrix).

        This method accumulates per-channel second-moment activation statistics (imatrix)
        via forward hooks and uses them to perform RTN quantization. If CUDA memory runs out,
        it falls back to CPU-based blockwise quantization.

        Args:
            all_to_quantized_module_names (list[str]):
                A list of module names (e.g., 'model.layers.0.self_attn.q_proj') to be quantized.

        Returns:
            None
        """
        logger.info("start to compute imatrix for GGUF quantization.")

        # Load dataset
        from .calib_dataset import get_dataloader

        if isinstance(self.dataset, str):
            dataset_name = self.dataset.replace(" ", "")
            self.dataloader = get_dataloader(
                self.tokenizer, self.seqlen, dataset_name, self.seed, self.batch_size, self.nsamples
            )
        else:
            self.dataloader = self.dataset

        model = self.model

        # Dispatch multi-GPU model if necessary
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            from accelerate.big_modeling import dispatch_model

            dispatch_model(model, model.hf_device_map)

        def register_act_hook(model):
            """Registers hooks to accumulate activation squared norms into `imatrix`."""

            def get_imatrix_hook(module, input, output):
                input = input[0] if isinstance(input, (tuple, list)) else input
                flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
                squared = torch.sum(flattened**2, dim=0).to(torch.float32)

                if not hasattr(module, "imatrix"):
                    module.imatrix = squared
                    module.imatrix_cnt = input.shape[0]
                else:
                    module.imatrix += squared
                    module.imatrix_cnt += input.shape[0]

            hook_handles = []
            for name, module in model.named_modules():
                if isinstance(module, self.supported_types) and check_to_quantized(module):
                    hook = module.register_forward_hook(get_imatrix_hook)
                    hook_handles.append(hook)
            return hook_handles

        hooks = register_act_hook(model)

        try:
            # Move model to target device
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                from accelerate.big_modeling import dispatch_model

                dispatch_model(self.model, self.model.hf_device_map)
            else:
                model = model.to(self.device)
            cnt = 0

            # Run forward pass to accumulate imatrix
            for data in self.dataloader:
                cnt += data["input_ids"].shape[0]
                data = to_device(data, self.device)
                model(**data)
                if cnt >= self.nsamples:
                    break

            # Remove hooks after data collection
            for hook in hooks:
                hook.remove()

            # Normalize imatrix by count
            for _, module in model.named_modules():
                if hasattr(module, "imatrix"):
                    module.imatrix /= module.imatrix_cnt
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                import accelerate

                accelerate.hooks.remove_hook_from_submodules(model)
            # Perform quantization using RTN
            from tqdm import tqdm

            pbar = tqdm(all_to_quantized_module_names)
            block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
            clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
            if clear_mem_freq == 0:
                clear_mem_freq = 1
            cnt = 1
            for name in pbar:
                pbar.set_description(f"Quantizing {name}")
                self.quantize_layer_via_rtn(name)
                if cnt % clear_mem_freq == 0:
                    clear_memory()
                    cnt = 1
                cnt += 1
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "MODULE:PT_DEVMEM" in str(e):
                try:
                    if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                        import accelerate

                        accelerate.hooks.remove_hook_from_submodules(model)
                    # Fallback: out-of-memory → try CPU blockwise quantization
                    logger.warning("Out of VRAM, falling back to blockwise quantization. Accuracy may degrade.")
                    model = model.to("cpu")
                    clear_memory()
                    self.quantize_via_rtn_blockwise(all_to_quantized_module_names)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) or "MODULE:PT_DEVMEM" in str(e):
                        # Final fallback: warn and use CPU-only quantization
                        logger.warning(
                            "Fallback to CPU. "
                            "Consider enabling `low_gpu_mem_usage` or using more GPUs via `--device 0,1,2,3`."
                        )
                        model = model.to("cpu")
                        clear_memory()
                        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                            import accelerate

                            accelerate.hooks.remove_hook_from_submodules(model)

                        orig_device = self.device
                        self.device = "cpu"
                        self.quantize_via_rtn_blockwise(all_to_quantized_module_names)
                        self.device = orig_device
                    else:
                        raise
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

        # Move back to CPU and free memory
        model.to("cpu")
        clear_memory()

    def check_need_to_quantize_lm_head_embedding(self) -> bool:
        """Checks if LM head and embedding layers need quantization for GGUF format.

        This function inspects the current model's formats and determines whether
        it needs to apply quantization settings to the embedding and LM head layers.
        The function modifies `self.layer_config` in-place and updates the model modules.

        Returns:
            bool: True if the LM head needs quantization, otherwise False.

        Raises:
            NotImplementedError: If multiple non-fake GGUF formats are specified.
        """
        if not hasattr(self, "formats"):
            return False

        has_gguf: bool = any("gguf" in fmt for fmt in self.formats)
        if not has_gguf:
            return False

        formats: list[str] = [fmt for fmt in self.formats if "fake" not in fmt]
        if not (len(formats) == 1 and "gguf" in formats[0]):
            raise NotImplementedError("Only one GGUF format can be set at a time.")

        target_format: str = formats[0]
        tie_word_embeddings: bool = getattr(getattr(self.model, "config", None), "tie_word_embeddings", True)

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                key: str = "lm_head" if tie_word_embeddings else "embedding"
                config: dict[str, Any] = GGUF_INNER_CONFIG[GGUF_CONFIG[target_format][key]]
                self._apply_config_to_layer(name, config, True)

        if not tie_word_embeddings:
            lm_head_name: str = get_lm_head_name(self.model)
            config: dict[str, Any] = GGUF_CONFIG[GGUF_CONFIG[target_format]["lm_head"]]
            check_fixed_by_user = (
                self.layer_config[lm_head_name].get("fixed_by_user", False)
                if lm_head_name in self.layer_config
                else None
            )
            self._apply_config_to_layer(lm_head_name, config, check_fixed_by_user=check_fixed_by_user)
            return True

        return False

    def _apply_config_to_layer(
        self,
        layer_name: str,
        config: dict[str, Any],
        check_fixed_by_user: bool = False,
    ) -> None:
        """Applies GGUF quantization configuration to a given layer.

        Args:
            layer_name (str): Name of the layer to configure.
            config (dict[str, Any]): GGUF layer configuration.
            check_fixed_by_user (bool): If True, preserve user-defined settings.
        """
        act_bits: int = 16
        scale_dtype: Any = self.scale_dtype
        keys: list[str] = ["bits", "group_size", "super_bits", "super_group_size", "data_type", "sym"]

        self.layer_config[layer_name] = self.layer_config.get(layer_name, {})

        for key in keys:
            if (
                key in self.layer_config[layer_name]
                and check_fixed_by_user
                # and self.layer_config[layer_name].get("fixed_by_user", False)
            ):
                continue
            self.layer_config[layer_name][key] = config.get(key)
            setattr(get_module(self.model, layer_name), key, config.get(key))

        self.layer_config[layer_name]["act_bits"] = act_bits
        self.layer_config[layer_name]["scale_dtype"] = scale_dtype
        setattr(get_module(self.model, layer_name), "act_bits", act_bits)
        setattr(get_module(self.model, layer_name), "scale_dtype", scale_dtype)

    def quantize_layer_via_rtn(self, name: str) -> None:
        """Quantizes a layer using RTN (Round-To-Nearest) if available.

        This function attempts to quantize a layer by switching its data type to a
        `rtn_*` version if supported, then wraps and unwraps the module to apply
        quantization. If GPU memory is insufficient, it falls back to CPU.

        If packing is enabled (`is_packing_immediate`), the function will also export
        the quantized layer to the appropriate backend format.

        Args:
            name (str): Name of the layer to quantize.

        Raises:
            RuntimeError: If quantization fails for reasons unrelated to memory.
        """
        m = get_module(self.model, name)

        # Step 1: Use optimized RTN data type if available
        if not self.disable_opt_rtn and not m.data_type.startswith("rtn_"):
            from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

            rtn_dtype = "rtn_" + m.data_type
            if rtn_dtype in QUANT_FUNC_WITH_DTYPE:
                m.data_type = rtn_dtype
                self.layer_config[name]["data_type"] = m.data_type

        # Step 2: Try quantization on GPU first, fall back to CPU if OOM
        # if only export gguf, using gguf-packing instead of rtn
        if self.is_packing_immediate and self.iters == 0 and "gguf" in self.formats[0] and not self.disable_opt_rtn:
            m.scale = None
            m.zp = None
        else:
            try:
                m.to(self.device)
                m = WrapperLinear(
                    m,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                )
                m = m.unwrapper({})
                m.to("cpu")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "MODULE:PT_DEVMEM" in str(e):
                    logger.warning("Out of VRAM, falling back to CPU.")
                    m.to("cpu")
                    m = WrapperLinear(
                        m,
                        enable_minmax_tuning=False,
                        enable_norm_bias_tuning=False,
                        enable_round_tuning=False,
                    )
                    m = m.unwrapper({})
                else:
                    raise

        # Step 3: Optional immediate packing/export
        if self.is_packing_immediate:
            from auto_round.export import PACKING_LAYER_WITH_FORMAT

            if check_to_quantized(m):
                target_backend = self.formats[0].split(":")[0] if ":" in self.formats[0] else self.formats[0]
                has_gguf = any("gguf" in fmt for fmt in self.formats)

                if has_gguf:
                    from auto_round.export.export_to_gguf.export import pack_gguf_layer

                    output_dir = self.get_save_folder_name(self.formats[0])
                    model_type = ModelType.MMPROJ if self.vlm else ModelType.TEXT
                    pack_gguf_layer(
                        name,
                        self.model,
                        self.formats[0],
                        output_dir,
                        self.layer_config,
                        self.tokenizer,
                        processor=self.processor if hasattr(self, "processor") else None,
                        image_processor=self.image_processor if hasattr(self, "image_processor") else None,
                        model_type=model_type,
                    )
                else:
                    PACKING_LAYER_WITH_FORMAT[target_backend](name, self.model, self.formats[0])

                # if self.low_gpu_mem_usage:
                #     clear_memory()
        else:
            set_module(self.model, name, m)

    @torch.inference_mode()
    def quantize_rtn(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize all modules in the model using RTN (Round-To-Nearest) strategy.

        If the target format includes GGUF with `k`, and optimized RTN is enabled,
        blockwise quantization with input caching and imatrix is used.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        if self.amp:
            self.model.to(self.amp_dtype)

        all_to_quantized_module_names: list[str] = [n for n, m in self.model.named_modules() if check_to_quantized(m)]

        has_gguf_k = any("gguf" in fmt and "k" in fmt for fmt in getattr(self, "formats", []))

        self.quantize_embedding_layer()

        self.model.to("cpu")
        if has_gguf_k and not self.disable_opt_rtn:
            self.quant_rtn_with_imatrix(all_to_quantized_module_names)
        elif self.act_bits <= 8 and check_need_act_calibration(
            self.act_dynamic, self.act_data_type
        ):  ##TODO, mixed datatype has bug
            hook_handles = self.register_act_max_hook(self.model)
            try:
                self.quantize_via_rtn_blockwise(all_to_quantized_module_names)
            except RuntimeError as e:
                logger.warning("Fallback to CPU. Consider using more GPUs via `--device 0,1,2,3`.")
                self.model = self.model.to("cpu")
                clear_memory()
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(self.model)
                orig_device = self.device
                self.device = "cpu"
                self.quantize_via_rtn_blockwise(all_to_quantized_module_names)
                self.device = orig_device
            for handle in hook_handles:
                handle.remove()
        else:
            block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
            clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
            if clear_mem_freq == 0:
                clear_mem_freq = 1
            pbar = tqdm(all_to_quantized_module_names)
            cnt = 1
            for name in pbar:
                pbar.set_description(f"Quantizing {name}")
                self.quantize_layer_via_rtn(name)
                if cnt % clear_mem_freq == 0:
                    clear_memory()
                    cnt = 1
                cnt += 1

        self.quantized = True
        return self.model, self.layer_config

    def quantize_via_rtn_blockwise(self, all_to_quantized_module_names: list[str]) -> None:
        """Quantize model layers block by block using cached inputs and imatrix.

        Args:
            all_to_quantized_module_names (list[str]): Names of layers to be quantized.
        """
        all_to_quantized_module_names = list(set(all_to_quantized_module_names))

        all_blocks = self.quant_block_list if self.quant_block_list else get_block_names(self.model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        all_first_block_names = [block[0] for block in all_blocks]
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

                input_ids = self.get_block_outputs(
                    block,
                    input_ids,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    self.device,
                    self.cache_device,
                )
                if self.device_map is not None:
                    accelerate.hooks.remove_hook_from_submodules(block)

                # Normalize imatrix and quantize layers
                for _, m in block.named_modules():
                    if hasattr(m, "imatrix"):
                        m.imatrix /= m.imatrix_cnt
                    if hasattr(m, "tmp_name") and m.tmp_name in all_to_quantized_module_names:
                        self.quantize_layer_via_rtn(m.tmp_name)
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
            self.quantize_layer_via_rtn(name)
            if cnt % clear_mem_freq == 0:
                clear_memory()
                cnt = 1
            cnt += 1

    def quantize(self):
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        for n, m in self.model.named_modules():
            m.tmp_name = n
        self._check_compatibility()
        self.has_qlayer_outside_block = self.set_layerwise_config(self.layer_config)
        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            only_gguf = True
            for format_ in self.formats:
                if not ("gguf" in format_ or "fake" in format_):
                    only_gguf = False
                    break
            if len(self.formats) == 1 and self.formats[0] == "fake":
                only_gguf = False
            if only_gguf:
                self.layer_config, gguf_format_config = get_layer_config_by_gguf_format(
                    self.layer_config, self.formats, self.model, model_type=ModelType.TEXT
                )
                if self.vlm:
                    self.layer_config, gguf_format_config = get_layer_config_by_gguf_format(
                        self.layer_config, self.formats, self.model, model_type=ModelType.MMPROJ
                    )
            # Determine if immediate packing is required
            formats = self.formats
            if (
                len(formats) == 1
                and ("awq" in formats[0] or "gptq" in formats[0] or "auto_round" in formats[0] or "gguf" in formats[0])
                and self.inplace
            ):
                self.is_packing_immediate = True
        if self.iters == 0:
            return self.quantize_rtn()

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
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")
        all_inputs = self.try_cache_inter_data_gpucpu(all_first_block_names, self.nsamples, layer_names=layer_names)
        is_quantized_embedding = self.quantize_embedding_layer()
        all_q_inputs = None
        if is_quantized_embedding:
            all_inputs = copy.deepcopy(self.inputs)
            clear_memory(self.inputs)
            all_q_inputs = self.try_cache_inter_data_gpucpu(
                all_first_block_names, self.nsamples, layer_names=layer_names
            )
        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        clear_memory()
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model)  ##self.model.hf_device_map has not been changed
        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        logger.info("caching done")
        pbar = tqdm(range(0, sum([len(i) for i in all_blocks]), self.nblocks))

        for block_names in all_blocks:
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            q_inputs = None
            if all_q_inputs is not None:
                q_inputs = all_q_inputs[block_names[0]]
                all_q_inputs.pop(block_names[0])
            keys = inputs.keys()
            input_id_str = [key for key in keys if key.startswith("hidden_state")]
            if len(input_id_str) != 1:
                raise RuntimeError(
                    "hidden_states arg mismatch error,"
                    "please raise an issue in https://github.com/intel/auto-round/issues"
                )
            inputs["input_ids"] = inputs.pop(input_id_str[0], None)
            if q_inputs is not None:
                q_inputs["input_ids"] = q_inputs.pop(input_id_str[0], None)

            clear_memory(self.inputs)

            if "input_ids" in inputs.keys():
                total_samples = len(inputs["input_ids"])
                if total_samples < self.batch_size:
                    self.batch_size = total_samples
                    logger.warning(f"force the train batch size to {total_samples}")

            self.quant_blocks(
                self.model,
                inputs,
                block_names,
                q_input=q_inputs["input_ids"] if q_inputs is not None else None,
                nblocks=self.nblocks,
                device=self.device,
                pbar=pbar,
            )
            if self.is_packing_immediate and len(self.formats) != 1:
                raise ValueError(
                    f"Expected exactly one packing format when 'is_packing_immediate' is True, "
                    f"but got {len(self.formats)} formats."
                )

        self.quant_layers(layer_names, all_inputs)  ##TODO pack layer immediately

        end_time = time.time()
        cost_time = end_time - self.start_time
        logger.info(f"quantization tuning time {cost_time}")

        ## dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
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

        self.quantized = True
        return self.model, self.layer_config

    def quant_layers(self, layer_names, layer_inputs):
        """Quantizes specified layers based on inputs and configuration.

        Args:
            layer_names (list): list of layer names to quantize.
            layer_inputs (dict): Dictionary mapping layer names to input data.

        Returns:
            None
        """
        ##TODO currently we take all the layers outside blocks as post block layers which is not optimal
        ## if there is no input for layer, we use rtn
        for layer_name in copy.deepcopy(layer_names):
            if layer_name not in layer_inputs:
                logger.info(f"using rtn to quantize {layer_name}")
                from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

                layer = get_module(self.model, layer_name)
                if not self.disable_opt_rtn and "rtn_" + layer.data_type in QUANT_FUNC_WITH_DTYPE:
                    layer.data_type = "rtn_" + layer.data_type
                    logger.info("using optimized rtn method for quantizing %s", layer_name)
                    self.layer_config[layer_name]["data_type"] = layer.data_type
                layer.to(self.device)
                wrapper_layer = WrapperLinear(
                    layer,
                    enable_round_tuning=False,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    device=self.device,
                )
                new_layer = wrapper_layer.unwrapper({})
                set_module(self.model, layer_name, new_layer)
                layer.cpu()
                layer_names.remove(layer_name)
        if len(layer_names) == 0:
            return
        q_layer_inputs = None
        enable_quanted_input = self.enable_quanted_input
        has_gguf = False
        if hasattr(self, "formats"):
            has_gguf = any("gguf" in format_ for format_ in self.formats)
        if has_gguf and self.is_packing_immediate:
            enable_quanted_input = False

        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1 and enable_quanted_input:
            from accelerate.big_modeling import dispatch_model

            dispatch_model(self.model, self.model.hf_device_map)

        if enable_quanted_input:
            logger.info("starting to cache layer inputs for %s, this may be quite slow ", layer_names)
            q_layer_inputs = self.try_cache_inter_data_gpucpu([], self.nsamples, layer_names=layer_names)
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.model
                )  ##self.model.hf_device_map has not been changed

        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        clear_memory()
        if self.enable_torch_compile:
            quant_layer = compile_func(self.quant_layer, self.device)
        else:
            quant_layer = self.quant_layer
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.cache_device)
            q_layer_input = q_layer_inputs[layer_name] if q_layer_inputs is not None else None
            q_layer_input = to_device(q_layer_input, self.cache_device)
            quant_layer(layer_name, layer_input, q_layer_input, device=self.device)
            del layer_input
            clear_memory(q_layer_input)

    def set_layerwise_config(self, layer_config):
        """
        Sets the layer-wise configuration based on the provided `layer_config`.
        By default, only quantize layers in blocks.

        Args:
            layer_config (dict): The configuration dictionary for each layer containing various configuration options.

        Returns:
            bool: Returns True if there are quantized layers outside the blocks (e.g., lm-head),
                  otherwise returns False.
        """
        # Get the names of layers in quantization blocks
        layers_in_blocks = get_layer_names_in_block(self.model, self.supported_types, self.quant_block_list)
        ##process regex in layer_config
        all_supported_layer_names = []
        # List of configuration keys
        keys = [
            "bits",
            "group_size",
            "sym",
            "data_type",
            "scale_dtype",
            "act_bits",
            "act_group_size",
            "act_sym",
            "act_dynamic",
            "act_data_type",
            "super_bits",
            "super_group_size",
        ]

        for n, m in self.model.named_modules():
            # Delete previous configuration to avoid conflicts with prior tuning
            for key in keys:
                if hasattr(m, key):
                    delattr(m, key)

            # Skip unsupported types
            supported_types = self.supported_types

            if not isinstance(m, supported_types):
                continue
            all_supported_layer_names.append(n)

        names_in_layer_config = list(layer_config.keys())
        for name in names_in_layer_config:
            if name in all_supported_layer_names:
                continue
            matched_names = []
            for layer_name in all_supported_layer_names:
                if re.search(re.compile(name), layer_name) is not None:
                    matched_names.append(layer_name)
            if len(matched_names) > 0:
                val = layer_config[name]
                layer_config.pop(name)
                for match_name in matched_names:
                    layer_config[match_name] = val
            else:
                tmp_m = get_module(self.model, name)
                if not isinstance(tmp_m, torch.nn.Embedding):  ##TODO not good code style
                    raise ValueError(f"key {name} in layer_config is invalid, please have a double check")

        has_qlayer_outside_block = False  # Flag to track if there are quantized layers outside blocks (e.g., lm-head)

        # Iterate through all modules in the model
        for n, m in self.model.named_modules():

            # Skip unsupported types
            if not isinstance(m, supported_types):
                continue

            # If the layer is not in the config and is part of a quantization block, use default configuration
            if n not in layer_config.keys() and n in layers_in_blocks:
                layer_config[n] = {}
                for key in keys:
                    layer_config[n][key] = getattr(self, key)
            # If the layer is partially configured, fill in missing values
            elif n in layer_config.keys():
                for key in keys:
                    if key not in layer_config[n].keys():
                        layer_config[n][key] = getattr(self, key)
                layer_config[n]["fixed_by_user"] = True
            # If the layer is not in the config and not part of a quantization block,
            # use default configuration and set specific values
            else:
                layer_config[n] = {}
                for key in keys:
                    layer_config[n][key] = getattr(self, key)
                layer_config[n]["bits"] = 16
                layer_config[n]["act_bits"] = 16

            if n in layers_in_blocks:
                layer_config[n]["in_blocks"] = True
            else:
                layer_config[n]["in_blocks"] = False

            # If the layer is outside a block and requires quantization, mark it as a quantized layer outside the block
            if (
                n not in layers_in_blocks
                and check_to_quantized(layer_config[n])
                and not isinstance(m, torch.nn.Embedding)
            ):
                has_qlayer_outside_block = True

            in_features, out_features = get_layer_features(m)
            if in_features <= layer_config[n]["group_size"]:
                layer_config[n]["group_size"] = -1

            # Apply the configuration to the corresponding layer in the model
            for key in keys:
                setattr(m, key, layer_config[n][key])
        need_to_quantize_lm_head = self.check_need_to_quantize_lm_head_embedding()
        if need_to_quantize_lm_head:
            has_qlayer_outside_block = True

        # Return whether there are quantized layers outside the blocks
        return has_qlayer_outside_block

    @torch.no_grad()
    def get_block_outputs(self, block, input_ids, input_others, bs, device, cache_device, save_output=True):
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
            tmp_input_ids, tmp_input_others = AutoRound.sampling_inputs(
                input_ids, input_others, indices, self.seqlen, self.batch_dim, share_cache_keys=self.shared_cache_keys
            )
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, self.amp, self.amp_dtype, device).to(
                cache_device
            )
            if save_output:
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
                input_ids = data.to(self.model.device)
                data_new = input_ids
            elif isinstance(data, str):
                if self.tokenizer is None:
                    logger.error("please provide tokenizer for string input")
                    exit(-1)
                data = self.tokenizer(data, truncation=True, max_length=self.seqlen, return_tensors="pt").data
                data_new = {}
                for key in data.keys():
                    data_new[key] = data[key].to(self.model.device)
                input_ids = data_new["input_ids"]
            elif isinstance(data, tuple) or isinstance(data, list):
                data_new = to_device(data)
                input_ids = data_new[0]
            else:
                data_new = {}
                for key in data.keys():
                    data_new[key] = to_device(data[key], self.model.device)
                    if key == "images":
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
                error_msg = str(error)
                if "The expanded size of the tensor" in str(error_msg) and "must match the existing size" in error_msg:
                    check_seqlen_compatible(self.seqlen, self.tokenizer, self.model)
                logger.warning(
                    "When quantization encounters tensor shape mismatch error, "
                    "you can try to avoid it with batch_size=1"
                )
                raise error
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
                f"An insufficient number of samples likely reduces the accuracy of the quantized model."
                f"Target samples count is {nsamples}, while valid samples count is {total_cnt}"
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
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "MODULE:PT_DEVMEM" in str(e):
                logger.info("switch to cpu to cache block inputs")
                if self.has_qlayer_outside_block or self.__class__.__name__ == "AutoRoundMLLM":
                    logger.warning(
                        "we strongly recommend using more GPUs in calibration."
                        " Otherwise, some layers may fall back to `rtn` mode, which can affect accuracy."
                    )
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    accelerate.hooks.remove_hook_from_submodules(
                        self.model
                    )  ##self.model.hf_device_map has not been changed
                self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
                clear_memory()
                ## Important change after v0.51, on cpu, we use rtn mode for layers in layer_names
                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=[], last_cache_name=last_cache_name
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
            self.model = self.model.to(torch.bfloat16) if self.amp else self.model.to(torch.float32)  ##model on cpu

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
            if data_name in self.shared_cache_keys:
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
                        if data is None or (self.batch_size > 1 and key in self.shared_cache_keys):
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
            if name == self.last_cache_name:
                raise NotImplementedError
            else:
                if hidden_states is not None:
                    kwargs.pop("hidden_states")
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
        if hasattr(layer, "tuning_device"):
            device = layer.tuning_device

        layer = layer.to(device)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(layer.weight.dtype)
            if q_inputs is not None:
                q_inputs[i] = q_inputs[i].to(layer.weight.dtype)

        wrapper_linear = WrapperLinear(layer, enable_minmax_tuning=self.enable_minmax_tuning, device=device).to(device)
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
            mv_module_from_gpu(layer, self.low_cpu_mem_usage)

        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": self.minmax_lr}], lr=self.lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=self.lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
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
                indices = whole_indices[tmp_step * batch_size : (tmp_step + 1) * batch_size]
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
        logger.info(dump_info)

    def register_act_max_hook(self, model):
        def get_act_max_hook(module, input, output):
            if isinstance(input, (tuple, list)):
                input = input[0]
            input, _, _ = reshape_pad_tensor_by_group_size(input, self.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max"):
                module.act_max = act_max
            else:
                module.act_max = torch.max(act_max.to(module.act_max.device), module.act_max)

        hook_handles = []

        for n, m in model.named_modules():
            if (
                hasattr(m, "act_dynamic")
                and check_need_act_calibration(m.act_dynamic, m.act_data_type)
                and check_to_quantized(m)
            ):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
                continue

            # for whole model, RTN
            if n in self.layer_config:
                config = self.layer_config[n]
                act_dynamic = config.get("act_dynamic", True)
                act_data_type = config.get("act_data_type", None)
                if (
                    config["bits"] <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type)
                    and check_to_quantized(config)
                ):
                    hook = m.register_forward_hook(get_act_max_hook)
                    hook_handles.append(hook)
                    continue
        return hook_handles

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
        if self.device_map is not None:
            from accelerate import dispatch_model

            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                add_hook_to_module(m, hook, True)

        if q_input is None:
            hook_handles = self.register_act_max_hook(block)

            output = self.get_block_outputs(
                block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device, self.cache_device
            )

            for handle in hook_handles:
                handle.remove()
        else:
            output = self.get_block_outputs(
                block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device, self.cache_device
            )
            hook_handles = self.register_act_max_hook(block)
            if hook_handles:
                self.get_block_outputs(
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
                indices = whole_indices[tmp_step * self.batch_size : (tmp_step + 1) * self.batch_size]
                current_input_ids, current_input_others = AutoRound.sampling_inputs(
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
        logger.info(dump_info)
        if len(unquantized_layer_names) != 0:
            logger.info(f"{unquantized_layer_names} have not been quantized")
        with torch.no_grad():
            unwrapper_block(block, best_params)
        if self.enable_quanted_input:
            if self.low_cpu_mem_usage:
                block = block.to(device)
            clear_memory()
            q_outputs = self.get_block_outputs(
                block,
                input_ids,
                input_others,
                self.batch_size * self.infer_bs_coeff,
                device,
                cache_device=self.cache_device,
            )
            if self.device_map is not None:
                accelerate.hooks.remove_hook_from_submodules(block)
            mv_module_from_gpu(block, self.low_cpu_mem_usage)
            clear_memory(input_ids)

            return q_outputs, output

        else:
            if self.device_map is not None:
                accelerate.hooks.remove_hook_from_submodules(block)
            mv_module_from_gpu(block, self.low_cpu_mem_usage)
            clear_memory(input_ids)
            return None, output

    def quant_blocks(
        self, model: torch.nn.Module, inputs, block_names, q_input=None, nblocks=1, device="cpu", pbar=None
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
        if self.enable_torch_compile:
            quant_block = compile_func(self.quant_block, device)
        else:
            quant_block = self.quant_block

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

            if not self.model.device.type == "meta" or self.low_cpu_mem_usage:
                m = m.to(device)

            q_input, input_ids = quant_block(
                m,
                input_ids,
                input_others,
                q_input=q_input,
                device=device,
            )
            if self.is_packing_immediate:
                from auto_round.export import PACKING_LAYER_WITH_FORMAT

                for _, tmp_m in m.named_modules():
                    if hasattr(tmp_m, "bits") and check_to_quantized(tmp_m):
                        target_backend = self.formats[0].split(":")[0] if ":" in self.formats[0] else self.formats[0]
                        has_gguf = any("gguf" in format_ for format_ in self.formats)
                        if has_gguf:
                            from auto_round.export.export_to_gguf.export import pack_gguf_layer

                            output_dir = self.get_save_folder_name(self.formats[0])
                            model_type = ModelType.MMPROJ if self.vlm else ModelType.TEXT
                            pack_gguf_layer(
                                tmp_m.tmp_name,
                                self.model,
                                self.formats[0],
                                output_dir,
                                self.layer_config,
                                self.tokenizer,
                                processor=self.processor if hasattr(self, "processor") else None,
                                image_processor=self.image_processor if hasattr(self, "image_processor") else None,
                                model_type=model_type,
                            )
                        else:
                            PACKING_LAYER_WITH_FORMAT[target_backend](tmp_m.tmp_name, self.model, self.formats[0])
        pbar.set_description("Quantizing done")
        pbar.update(1)
        pbar.close()

        self.model = mv_module_from_gpu(self.model, self.low_cpu_mem_usage)
        for n, m in self.model.named_modules():
            if hasattr(m, "name"):
                delattr(m, "name")

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
        format = self._check_supported_format(format)

        if self.low_cpu_mem_usage:
            self.model = self.model.to("cpu")

        if not self.quantized:
            logger.warning("please run autoround.quantize first")
            return
        if format == "fake" or format == "qdq":  ##TODO fix act quantizaiton later
            self.model = self.model.to("cpu")
            self.model.save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            processor = kwargs.get("processor", None)
            if processor is not None:
                processor.save_pretrained(output_dir)
            return
        if self.act_bits <= 8 and format == "qdq":
            logger.warning(
                "Support for exporting activation quantization is limited. "
                "Please ensure that your configuration is supported."
            )

        from auto_round.export import EXPORT_FORMAT

        backend = format
        format = format.split(":")[0]
        if format not in EXPORT_FORMAT:
            logger.error(f"export format only supports {EXPORT_FORMAT.keys()}")
            raise ValueError(f"export format only supports {EXPORT_FORMAT.keys()}, but got {format}")
        save_quantized_as_format = EXPORT_FORMAT.get(format)
        if "gptq" in format and not self.sym:
            logger.warning(
                "the asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                " particularly for 2-bit quantization and smaller models."
                " We recommend exporting to either the AutoAWQ format ( only 4 bits) or "
                "the AutoRound format(2/3/4/8 bits)."
            )
        if "awq" in format and not self.bits == 4:
            raise ValueError("The AWQ format only supports W4 quantization ")

        if isinstance(self.dataset, str):
            self.serialization_keys.append("dataset")
        serialization_dict = {}
        for key in self.serialization_keys:
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
            quant_block_list=self.quant_block_list,
            **kwargs,
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

    @classmethod
    @torch.no_grad()
    def sampling_inputs(cls, input_ids, input_others, indices, seqlen, batch_dim=0, share_cache_keys=()):
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
        current_input_ids = [input_ids[i] for i in indices]

        current_input_ids = torch.cat(current_input_ids, dim=batch_dim)

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
        act_data_type (str): Specifies the data type for activations.
                             Defaults to None, in which case it inherits the weight data type.
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
        act_data_type: str = None,
        act_dynamic: bool = True,
        to_quant_block_names: Union[str, list] = None,
        enable_norm_bias_tuning: bool = False,
        enable_torch_compile: bool = False,
        device_map: Union[str, dict] = None,
        optimizer="AdamW",
        super_bits: int = None,
        super_group_size: int = None,
        disable_opt_rtn: bool = False,
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
            act_data_type=act_data_type,
            act_dynamic=act_dynamic,
            to_quant_block_names=to_quant_block_names,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_torch_compile=enable_torch_compile,
            device_map=device_map,
            super_bits=super_bits,
            super_group_size=super_group_size,
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
        act_data_type (str): Specifies the data type for activations.
                             Defaults to None, in which case it inherits the weight data type.
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
        act_data_type: str = None,
        act_dynamic: bool = True,
        to_quant_block_names: Union[str, list] = None,
        enable_norm_bias_tuning: bool = False,
        enable_torch_compile: bool = False,
        device_map: Union[str, dict] = None,
        optimizer="AdamW",
        super_bits: int = None,
        super_group_size: int = None,
        disable_opt_rtn: bool = False,
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
            act_data_type=act_data_type,
            act_dynamic=act_dynamic,
            to_quant_block_names=to_quant_block_names,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_torch_compile=enable_torch_compile,
            device_map=device_map,
            optimizer=optimizer,
            super_bits=super_bits,
            super_group_size=super_group_size,
            **kwargs,
        )
