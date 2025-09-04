# Copyright (c) 2025 Intel Corporation
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

from __future__ import annotations

import re
import os
import sys
import copy
import traceback
import functools
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union
from enum import IntEnum

import torch
import accelerate
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from transformers import set_seed

from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.special_model_handler import _handle_moe_model
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_DTYPES,
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    _gguf_args_check,
    _is_fp8_linear,
    _is_fp8_model,
    block_forward,
    check_and_mark_fp8_model,
    check_is_cpu,
    check_need_act_calibration,
    check_seqlen_compatible,
    check_skippable_keywords,
    check_to_quantized,
    clear_memory,
    collect_best_params,
    compile_func,
    convert_dtype_str2torch,
    convert_fp8_layer_to_linear,
    convert_fp8_model_to_16b_model,
    detect_device,
    find_matching_blocks,
    flatten_list,
    get_block_names,
    get_layer_config_by_gguf_format,
    get_layer_features,
    get_layer_names_in_block,
    get_lm_head_name,
    get_max_vram,
    get_module,
    get_quant_keys,
    get_shared_keys,
    htcore,
    infer_bits_by_data_type,
    init_cache,
    is_debug_mode,
    is_mx_fp,
    is_nv_fp,
    is_optimum_habana_available,
    is_standard_fp,
    is_static_wfp8afp8,
    is_wfp8afp8,
    llm_load_model,
    logger,
    mv_module_from_gpu,
    reset_params,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
    to_dtype,
    unsupport_meta_device,
)
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme, preset_name_to_scheme


class QuantizerType(IntEnum):
    MODE = 1
    MODEL_TYPE = 2
    DATA_TYPE = 3


class BaseQuantizer:
    _quantizer_classes: dict[QuantizerType, dict[str, type[BaseQuantizer]]] = {
        QuantizerType.MODE: {},
        QuantizerType.MODEL_TYPE: {},
        QuantizerType.DATA_TYPE: {}
    }
    _HOOKABLE_FUNC = ["quantize", "save_quantized", "quantize_and_save"]
    _pre_functions: Dict[str, List[Callable]] = {}
    _after_functions: Dict[str, List[Callable]] = {}

    bits: int | None
    group_size: int | None
    sym: bool | None
    data_type: str | None
    act_bits: int | None
    act_group_size: int | None
    act_sym: bool | None
    act_data_type: str | None
    act_dynamic: bool | None
    super_bits: int | None
    super_group_size: int | None

    def __init__(
        self,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        **kwargs,
        ):
        """Initialize AutoRound with quantization and tuning configuration.

        Args:
            model (torch.nn.Module | str): Model object or model name to load.
            tokenizer: Tokenizer for text processing. Required if `model` is not a string and `iters > 0`.
            scheme (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations
            bits (int, optional): Weight quantization bits. Defaults to 4.
            group_size (int, optional): Weight quantization group size. Defaults to 128.
            sym (bool, optional): Symmetric weight quantization. Defaults to True.
            layer_config (dict, optional): Layer-wise quantization config. Defaults to None.
            batch_size (int, optional): Calibration batch size. Defaults to 8.
            amp (bool, optional): Use AMP for tuning. Defaults to True.
            device (str | torch.device | int, optional): Compute device. Defaults to 0.
            dataset (str | list | tuple | DataLoader, optional): Calibration data. Defaults to "NeelNanda/pile-10k".
            enable_minmax_tuning (bool, optional): Enable weight min-max tuning. Defaults to True.
            lr (float, optional): Learning rate; if None, set to 1.0 / iters except when iters==0.
            minmax_lr (float, optional): Learning rate for min-max tuning; defaults to `lr`.
            low_gpu_mem_usage (bool, optional): Lower GPU memory mode. Defaults to False.
            iters (int, optional): Optimization iterations. Defaults to 200.
            seqlen (int, optional): Calibration sequence length. Defaults to 2048.
            nsamples (int, optional): Number of calibration samples. Defaults to 128.
            seed (int, optional): Random seed. Defaults to 42.
            gradient_accumulate_steps (int, optional): Gradient accumulation steps. Defaults to 1.
            data_type (str, optional): Weight data type string, e.g., "int". Defaults to "int".
            act_bits (int, optional): Activation quantization bits. Defaults to 16.
            act_group_size (int, optional): Activation group size. Defaults to None.
            act_sym (bool, optional): Symmetric activation quantization. Defaults to None.
            act_data_type (str, optional): Activation data type; inherits weight dtype if None and act_bits < 16.
            act_dynamic (bool, optional): Dynamic activation quantization. Defaults to True.
            enable_torch_compile (bool, optional): Enable torch.compile for quant blocks/layers. Defaults to False.
            device_map (str | dict, optional): Device placement map. Defaults to None.
            disable_opt_rtn (bool, optional): Disable RTN-mode optimization (iters=0). Defaults to False.
            enable_alg_ext (bool, optional): Enable algorithm extension (primarily for INT2). Defaults to False.
            **kwargs: Backward compatible options:
                - enable_alg_ext, lr, lr_scheduler, sampler, not_use_best_mse, dynamic_max_gap,
                  super_group_size, super_bits, scale_dtype ("fp16" etc.),
                  nblocks, low_cpu_mem_usage, to_quant_block_names,
                  enable_norm_bias_tuning, enable_quanted_input,
                  disable_deterministic_algorithms, vlm, static_kv_dtype
        Raises:
            ValueError: If invalid device is provided or tokenizer is missing for non-str model with iters > 0.
            RuntimeError: If model parameters are on meta device.
        Example:
            Layer-wise configuration structure:

            >>> layer_config = {
            ...     "layer1": {
            ...         "data_type": "int",
            ...         "bits": 4,
            ...         "group_size": 128,
            ...         "sym": True,
            ...         "act_data_type": None,
            ...         "act_bits": 16,
            ...         "act_group_size": None,
            ...         "act_sym": None,
            ...     },
            ...     # ...
            ... }
        """
        self.scheme = None
        self._parse_and_set_scheme(scheme, kwargs)

        # Extra/legacy kwargs for backward compatibility
        # Major version releases may pack them with extra configuration options
        amp = kwargs.pop("amp", True)
        lr = kwargs.pop("lr", None)
        enable_alg_ext = kwargs.pop("enable_alg_ext", False)
        enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)
        minmax_lr = kwargs.pop("minmax_lr", None)
        disable_opt_rtn = kwargs.pop("disable_opt_rtn", False)
        lr_scheduler = kwargs.pop("lr_scheduler", None)
        sampler = kwargs.pop("sampler", "rand")
        not_use_best_mse = kwargs.pop("not_use_best_mse", False)
        dynamic_max_gap = kwargs.pop("dynamic_max_gap", -1)
        scale_dtype = kwargs.pop("scale_dtype", "fp16")
        nblocks = kwargs.pop("nblocks", 1)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        to_quant_block_names: Union[str, list, None] = kwargs.pop("to_quant_block_names", None)
        enable_norm_bias_tuning: bool = kwargs.pop("enable_norm_bias_tuning", False)
        enable_quanted_input: bool = kwargs.pop("enable_quanted_input", True)
        disable_deterministic_algorithms = kwargs.pop("disable_deterministic_algorithms", False)
        static_kv_dtype = kwargs.pop("static_kv_dtype", None)
        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("`device` is deprecated, please use `device_map` instead")

        self.vlm = kwargs.pop("vlm") if "vlm" in kwargs else False

        if kwargs:
            logger.warning(f"unrecognized keys {list(kwargs.keys())} were passed. Please check them.")

        if device_map is not None and "," in str(device_map):
            raise ValueError(
                "API does not support explicit set multiple devices," " please set CUDA_VISIBLE_DEVICES=0,1 yourself"
            )
        if device_map is None:
            device_map = 0

        # Set device, must place after model loading
        if isinstance(device_map, (str, torch.device, int)):
            self.device = detect_device(device_map)
        elif isinstance(device_map, dict) and device_map:
            tmp_devices = []
            for val in device_map.values():
                if isinstance(val, (str, torch.device, int)):  # could optimize
                    tmp_device = detect_device(self.device_map)
                    tmp_device = tmp_device.split(":")[0]
                    tmp_devices.append(tmp_device)
            tmp_devices = list(set(tmp_devices))
            if len(tmp_devices) > 1:
                logger.warning(
                    f"there are multiple device types in the device_map, "
                    f"please make sure they are correct,use the first device {tmp_devices[0]} as the core device "
                )

            self.device = tmp_devices[0]

        if isinstance(device_map, dict) and device_map:
            self.device_map = device_map
        else:
            self.device_map = None
        self._set_device_map_in_blocks(self.device_map)

        if not disable_deterministic_algorithms:
            if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=False)

        # Model related
        self.quantized = False
        if isinstance(model, str):
            model, tokenizer, low_cpu_mem_usage = llm_load_model(
                model,
                device="cpu",
                low_cpu_mem_mode=low_cpu_mem_usage,  # always load cpu first
            )
        elif tokenizer is None and iters > 0:
            raise ValueError("A tokenizer must be set for non-str model input")
        self.low_cpu_mem_usage = bool(low_cpu_mem_usage)
        if unsupport_meta_device(model):
            raise RuntimeError(
                "AutoRound does not support parameters on meta device. "
                "Please use more GPUs by setting `--device 0,1,2,3` or just place the model on CPU."
            )
        check_and_mark_fp8_model(model)
        model = _handle_moe_model(model)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.shared_cache_keys = get_shared_keys(self.model)

        # Some other quantization configs
        self.layer_config = {} if layer_config is None else layer_config
        # Check and convert to dict
        for key, item in self.layer_config.items():
            if isinstance(item, str):
                item = asdict(preset_name_to_scheme(item.upper()))
                self.layer_config[key] = item

            if isinstance(item, QuantizationScheme):
                config = asdict(item)
                tmp_keys = copy.deepcopy(list(config.keys()))
                for tmp_key in tmp_keys:  ## Pop None value to be overridden
                    if config[tmp_key] is None:
                        config.pop(tmp_key)
                self.layer_config[key] = config
            elif isinstance(item, dict):
                item_keys = item.keys()
                expected_keys = list(QuantizationScheme.__annotations__.keys())
                if item_keys not in expected_keys:
                    for item_key in item_keys:
                        if item_key not in expected_keys:
                            raise ValueError(
                                f"the key {item_key} in layer_config for layer {key} is invalid,"
                                f" only {expected_keys} are supported"
                            )

        self.to_quant_block_names = to_quant_block_names

        # Tuning hyperparameters
        self.seed = seed
        set_seed(self.seed)
        self.amp = amp
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.nsamples = nsamples
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        self.low_gpu_mem_usage = low_gpu_mem_usage
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
            self.lr = lr or (1.0 / self.iters)  # must place after iter setting
        self.minmax_lr = minmax_lr or self.lr
        self.enable_alg_ext = enable_alg_ext
        self.sampler = sampler
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.lr_scheduler = lr_scheduler
        self.disable_opt_rtn = disable_opt_rtn
        self.is_packing_immediate = False  # whether to pack the layer immediately after tuning

        # KV cache, this one does not affect tuning but will collect some infos during tuning
        self.static_kv_dtype = static_kv_dtype
        if self.static_kv_dtype is not None:
            logger.warning("The static kv is experimental and currently has limited support.")

        # Model related
        self.quantized = False
        if isinstance(model, str):
            model, tokenizer, low_cpu_mem_usage = llm_load_model(
                model, device=device, low_cpu_mem_mode=low_cpu_mem_usage
            )
        elif tokenizer is None and iters > 0:
            raise ValueError("A tokenizer must be set for non-str model input")
        self.low_cpu_mem_usage = bool(low_cpu_mem_usage)
        if unsupport_meta_device(model):
            raise RuntimeError(
                "AutoRound does not support parameters on meta device. "
                "Please use more GPUs by setting `--device_map 0,1,2,3` or just place the model on CPU."
            )
        model = _handle_moe_model(model)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.shared_cache_keys = get_shared_keys(self.model)
        if not hasattr(self, "quant_block_list"):
            all_blocks = get_block_names(model)
            self.quant_block_list = find_matching_blocks(model, all_blocks, self.to_quant_block_names)

        self.scale_dtype = convert_dtype_str2torch(scale_dtype)
        self._set_amp_dtype()
        self.cache_device = torch.device("cpu") if self.low_gpu_mem_usage else self.device
        if self.act_bits <= 8 and self.amp_dtype == torch.float16:
            logger.warning("force to use bf16 to for quantization tuning when enabling activation quantization")
            self.amp_dtype = torch.bfloat16
            if self.model.dtype != torch.bfloat16:  # keep the model's buffer dtype unchanged
                self.model = self.model.to(torch.bfloat16)
        else:
            logger.info(f"using {self.model.dtype} for quantization tuning")

        # Some helpers
        self.supported_types = SUPPORTED_LAYER_TYPES
        self.inner_supported_types = INNER_SUPPORTED_LAYER_TYPES
        if "hpu" in str(self.device):
            self.inner_supported_types = tuple(x for x in INNER_SUPPORTED_LAYER_TYPES if x != "FP8Linear")
        self.batch_dim = None
        self.infer_bs_coeff = 1
        self.enable_torch_compile = enable_torch_compile
        self._adjust_torch_compile(enable_torch_compile)
        self._check_configs()
        torch.set_printoptions(precision=3, sci_mode=True)

        if is_optimum_habana_available():
            logger.info("optimum Habana is available, import htcore explicitly.")
            import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401]

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

        # add pre and after hook
        self._register_hooks()
        for name in self._HOOKABLE_FUNC:
            setattr(self, name, self._wrap_function(getattr(self, name), name))

    def _parse_and_set_scheme(self, scheme: Union[str, dict, QuantizationScheme], kwargs) -> None:
        """Parse and set the quantization scheme."""
        if isinstance(scheme, QuantizationScheme):
            scheme = asdict(scheme)
        elif isinstance(scheme, dict):
            scheme = scheme
        elif isinstance(scheme, str):
            scheme = scheme.upper()
            self.scheme = scheme
            scheme = asdict(preset_name_to_scheme(scheme))

        scheme_keys = (
            "bits",
            "group_size",
            "sym",
            "data_type",
            "act_bits",
            "act_group_size",
            "act_sym",
            "act_data_type",
            "act_dynamic",
            "super_bits",
            "super_group_size",
        )
        for key in scheme_keys:
            if key in kwargs and kwargs[key] is not None:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, scheme.get(key, None))
            kwargs.pop(key, None)
        if self.act_dynamic is None:
            self.act_dynamic = True

        tmp_bits = infer_bits_by_data_type(self.data_type)
        if tmp_bits is not None and tmp_bits < 16 and tmp_bits != self.bits:
            logger.warning(f"'data_type' do not match the specified 'bits' setting. Resetting 'bits' to {tmp_bits}.")
            self.bits = tmp_bits
        if tmp_bits is not None and tmp_bits < 16:
            for supported_dtype in SUPPORTED_DTYPES:  # to easily handle dtype mx_fp4 and layer_config={xxx:{bits:8}}
                if self.data_type.startswith(supported_dtype):
                    if supported_dtype + str(tmp_bits) == self.data_type:  # could not replace FP8_e4m3
                        self.data_type = supported_dtype
                    break

        self.act_group_size = self.act_group_size if self.act_group_size is not None else self.group_size
        self.act_bits = self.act_bits if self.act_bits is not None else 16
        self.act_sym = self.act_sym if self.act_sym is not None else self.sym

        if self.act_data_type is None:
            if self.data_type in SUPPORTED_DTYPES and self.act_bits < 16:
                self.act_data_type = self.data_type
                logger.info(f"activation adopts {self.data_type}")
            else:
                self.act_data_type = "float"
        tmp_act_bits = infer_bits_by_data_type(self.act_data_type)
        if tmp_act_bits is not None and tmp_act_bits < 16 and tmp_act_bits != self.act_bits:
            self.act_bits = tmp_act_bits
            logger.warning(
                f"`act_data_type` do not"
                f" match the specified 'act_bits' setting. Resetting 'act_bits' to {tmp_act_bits}."
            )
        if tmp_act_bits is not None and tmp_act_bits < 16:
            for supported_dtype in SUPPORTED_DTYPES:  # to easily handle dtype mx_fp4 and layer_config={xxx:{bits:8}}
                if self.act_data_type.startswith(supported_dtype):
                    if supported_dtype + str(tmp_act_bits) == self.act_data_type:  # could not replace FP8_e4m3
                        self.act_data_type = supported_dtype
                    break

    def _adjust_torch_compile(self, enable_torch_compile: bool) -> None:
        """Sets the torch compile configuration for the tuning."""
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

        if self.low_cpu_mem_usage and self.enable_torch_compile:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as low_cpu_mem_usage is enabled")

        if is_debug_mode() and self.enable_torch_compile:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as debug mode is enabled")

        if ("fp8" in self.data_type or "fp8" in self.act_data_type) and self.enable_torch_compile:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as fp8 is enabled")

    def _set_device_map_in_blocks(self, device_map: Union[str, dict, None]) -> None:
        """Sets the device map for specific blocks in the model.

        Args:
            device_map (Union[str, dict, None]): A mapping of module names to devices.
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

    def _set_device_for_matching_module(self, name: str, device: str) -> None:
        """Sets the device for a module if it matches the given name."""
        module = get_module(self.model, name)
        if hasattr(module, "tuning_device") and module.tuning_device != device:
            logger.warning(
                f"multiple devices have been set for layer {name}, keeping original device {module.tuning_device}"
            )
        else:
            module.tuning_device = device
    
    def _check_configs(self) -> None:
        """Checks if the configurations are valid.

        Raises:
        ValueError, TypeError: If any of the configurations are invalid.
        """
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if self.bits <= 0:
            raise ValueError("`bits` must be positive")
        if self.act_bits <= 0:
            raise ValueError("`act_bits` must be positive")
        if not (self.group_size == -1 or self.group_size >= 0):
            raise ValueError("`group_size` must be -1 (per channel) or 0 (per-tensor) or a positive integer")
        if not (self.act_group_size == -1 or self.act_group_size >= 0):
            raise ValueError("`act_group_size` must be -1 (per channel) or 0 (per-tensor) or a positive integer")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive")
        if self.iters < 0:
            raise ValueError("`iters` must be non-negative")
        if self.seqlen <= 0:
            raise ValueError("`seqlen` must be positive")
        if self.nblocks <= 0:
            raise ValueError("`nblocks` must be positive")
        if self.gradient_accumulate_steps <= 0:
            raise ValueError("`gradient_accumulate_steps` must be positive")

        if self.act_bits <= 8:
            logger.warning(
                "activation quantization is an experimental feature with limited support and a complex API. "
                "And please save the quantized model to fake format as real deployment is not supported currently"
            )

        if is_mx_fp(self.data_type) and self.group_size != 32:
            logger.warning("dtype mx_fp should only support group_size of 32 in real deployment")

        if is_nv_fp(self.data_type) and (self.group_size != 16):
            logger.warning("dtype nv_fp should only support group_size of 16 in real deployment")

        if self.nsamples < self.gradient_accumulate_steps * self.batch_size:
            if self.batch_size > self.nsamples:
                if self.iters > 0:  # GGUF should log this warning, but we don't know the format here
                    logger.warning(
                        f"reset `batch_size` to {self.nsamples} as `nsamples`({self.nsamples})"
                        f" is smaller than batch_size({self.batch_size})"
                    )
                self.batch_size = self.nsamples
            if self.gradient_accumulate_steps > self.nsamples // self.batch_size:
                self.gradient_accumulate_steps = self.nsamples // self.batch_size
                logger.warning(
                    f"reset `gradient_accumulate_steps` to {self.gradient_accumulate_steps}"
                    f" as nsamples must equal or greater"
                    f" than gradient_accumulate_steps * batch_size"
                )

        if self.enable_norm_bias_tuning:
            logger.warning("the `enable_norm_bias_tuning` feature is experimental and currently has limited support.")

    def _set_amp_dtype(self) -> None:
        """Sets the automatic mixed precision (AMP) data type for the model based on the device and configuration."""
        self.amp_dtype = torch.bfloat16
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
                if self.model.dtype != self.amp_dtype:
                    self.model = self.model.to(self.amp_dtype)
        else:
            self.amp_dtype = torch.float32
            self.model = self.model.to(torch.float32)


    def _get_quantized_layer_names_outside_blocks(self) -> list:
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

    @torch.inference_mode()
    def _quantize_embedding_layer(self):
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
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.info("out of VRAM, falling back to CPU")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config[k]
                            for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                        },
                    )
                except Exception as e:
                    logger.error(cuda_error_msg)
                    raise

            # Overwrite the module's weights with the quantized version
            module.weight.data.copy_(weight.cpu())

            # Attach scale and zero point (zp) to the module
            for param_name, value in zip(["scale", "zp"], [scale, zp]):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
                elif isinstance(value, torch.Tensor):
                    setattr(module, param_name, value.cpu())
                else:
                    setattr(module, param_name, value)

            # Update config
            self.layer_config.setdefault(name, {}).update(config)

            # Release memory
            clear_memory()

        return is_quantized

    def _check_compatibility(self) -> None:
        # Check group_size 32 for auto_round
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
                    f"Change sequence length to {self.model.config.max_position_embeddings} "
                    "due to the limitation of max_position_embeddings"
                )
                self.seqlen = min(self.seqlen, self.model.config.max_position_embeddings)

        if self.seqlen is not None and hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length < self.seqlen:
                logger.warning(
                    f"Change sequence length to {self.tokenizer.model_max_length} "
                    "due to the limitation of model_max_length. "
                    "You can also try to increase the model_max_length to avoid this issue."
                )
                self.seqlen = min(self.seqlen, self.tokenizer.model_max_length)

        if self.group_size == 0 and "fp8" not in self.data_type:
            logger.warning("`group_size==0` is not supported for data_type other than fp8 ")


    def _set_layerwise_config(self, layer_config: dict) -> bool:
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
        supported_types = self.supported_types
        layers_in_blocks = get_layer_names_in_block(
            self.model, supported_types, self.quant_block_list, self.inner_supported_types
        )
        ##process regex in layer_config
        all_supported_layer_names = []
        # List of configuration keys
        keys = get_quant_keys()

        for n, m in self.model.named_modules():
            # Delete previous configuration to avoid conflicts with prior tuning
            for key in keys:
                if hasattr(m, key):
                    delattr(m, key)

            if not isinstance(m, supported_types) and m.__class__.__name__ not in self.inner_supported_types:
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
                if not isinstance(tmp_m, torch.nn.Embedding):  # TODO not good code style
                    raise ValueError(f"key {name} in layer_config is invalid, please have a double check")

        has_qlayer_outside_block = False  # Flag to track if there are quantized layers outside blocks (e.g., lm-head)

        # Iterate through all modules in the model
        is_gguf = hasattr(self, "formats") and any("gguf" in format_ for format_ in self.formats)
        for n, m in self.model.named_modules():
            # Skip unsupported types
            if not isinstance(m, supported_types) and m.__class__.__name__ not in self.inner_supported_types:
                if n in self.layer_config:
                    if not isinstance(m, torch.nn.Embedding):
                        logger.warning(f"{n} is not supported, layer_config {n}: {layer_config[n]} will be ignored.")
                        self.layer_config.pop(n)
                        continue
                    if not is_gguf:
                        if not check_to_quantized(layer_config[n]):
                            self.layer_config.pop(n)
                            continue
                else:
                    continue

            # If the layer is not in the config and is part of a quantization block, use default configuration
            if n not in layer_config.keys() and n in layers_in_blocks:
                layer_config[n] = {}
                for key in keys:
                    layer_config[n][key] = getattr(self, key)

            # If the layer is partially configured, fill in missing values
            elif n in layer_config.keys():
                if "data_type" in layer_config[n] and "bits" not in layer_config[n]:
                    tmp_bits = infer_bits_by_data_type(layer_config[n]["data_type"])
                    if tmp_bits is not None and tmp_bits != self.bits:
                        logger.warning(
                            f"'data_type' do not match the specified 'bits' setting for {n}."
                            f" Resetting 'bits' to {tmp_bits}."
                        )
                        layer_config[n]["bits"] = tmp_bits
                if "act_data_type" in layer_config[n] and "act_bits" not in layer_config[n]:
                    tmp_bits = infer_bits_by_data_type(layer_config[n]["act_data_type"])
                    if tmp_bits is not None and tmp_bits != self.act_bits:
                        logger.warning(
                            f"'act_data_type' do not match the specified 'act_bits' setting for {n}."
                            f" Resetting 'act_bits' to {tmp_bits}."
                        )
                        layer_config[n]["act_bits"] = tmp_bits

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

        # Return whether there are quantized layers outside the blocks
        return has_qlayer_outside_block

    def _parse_format_to_list(self, format: str) -> list:
        formats = format.replace("q*_", f"q{self.bits}_").replace(" ", "").split(",")
        for index in range(len(formats)):
            format_ = formats[index]
            if format_ not in SUPPORTED_FORMATS:
                logger.error(f"Unsupported format {format_}, please choose from {SUPPORTED_FORMATS}")
                exit(-1)
            if format_ == "auto_round":
                if self.sym and "int" in self.data_type:
                    format_ = "auto_round:auto_gptq"
                elif self.bits == 4 and not self.sym and "int" in self.data_type:
                    enable_awq = all(
                        config["bits"] == self.bits or config["bits"] >= 16 for config in self.layer_config.values()
                    )
                    if enable_awq:
                        format_ = "auto_round:auto_awq"
                elif is_nv_fp(self.data_type) or is_mx_fp(self.data_type):
                    format_ = f"auto_round:{self.data_type}"
                elif is_wfp8afp8(self):  # staic wfp8afp8
                    format_ = "auto_round:fp8"
                elif self.data_type == "fp" and self.bits == 8 and self.act_bits >= 16:  # woq fp8
                    format_ = "auto_round:fp8"
                elif self.act_bits < 16:
                    raise ValueError(
                        "AutoRound format does not support exporting "
                        "for the current quantization configuration, "
                        "please change to `fake` format for research purpose"
                    )

                formats[index] = format_
            elif format_ == "llmcompressor":
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                if check_compressed_tensors_supported() and (is_nv_fp(self.data_type) or is_mx_fp(self.data_type)):
                    format_ = format_.replace("llmcompressor", f"llmcompressor:{self.data_type}")
                    formats[index] = format_
                elif not is_wfp8afp8(self):
                    logger.error(
                        "Currently, the llmcompressor format only supports MXFP/NVFP/FP8. "
                        "Please change format to fake or auto_round etc."
                    )
        
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
        if format == "fake":
            return format
        format = format.replace("q*_", f"q{self.bits}_")
        # Only support to export afp8/nv_fp
        if self.act_bits <= 8:
            if not is_standard_fp(self.act_data_type) or self.act_dynamic:
                if format == "llmcompressor":
                    if is_nv_fp(self.act_data_type):
                        return format
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
                elif format != "fake" and not is_nv_fp(format):
                    logger.warning(
                        "Currently only support to export auto_round format quantized model"
                        " with fp8 or nv_fp4 dtype activation for activation quantization."
                        " Change format to fake and save."
                    )
                    format = "fake"
            else:
                if not (format == "auto_round" or format == "auto_round:fp8"):
                    logger.warning(
                        f"Currently only support to export auto_round or fake format for static W{self.bits}AFP8 model,"
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

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        pass

    def save_quantized(
        self, output_dir: str = None, format: str = "auto_round", inplace: bool = True, **kwargs
    ) -> torch.nn.Module:
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
        if format == "fake" or format == "qdq":  # TODO fix act quantization later
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
        if format == "llmcompressor" and (is_nv_fp(self.data_type) or is_mx_fp(self.data_type)):
            format = format.replace("llmcompressor", f"llmcompressor:{self.data_type}")

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
        from auto_round.version import __version__

        serialization_dict["autoround_version"] = __version__
        if "scale_dtype" in serialization_dict.keys():
            serialization_dict["scale_dtype"] = str(serialization_dict["scale_dtype"])

        compressed_model = save_quantized_as_format(  ##TODO refine the code
            output_dir,
            model=self.model,
            layer_config=self.layer_config,
            inplace=inplace,
            bits=self.bits,
            act_bits=self.act_bits,
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
            act_data_type=self.act_data_type,
            serialization_dict=serialization_dict,
            backend=backend,
            to_quant_block_names=self.to_quant_block_names,
            quant_block_list=self.quant_block_list,
            **kwargs,
        )
        return compressed_model

    def _get_save_folder_name(self, format_str: str) -> str:
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

    def quantize_and_save(
        self, output_dir: str = "tmp_autoround", format: str = "auto_round", inplace: bool = True, **kwargs
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
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
        format_list = self._parse_format_to_list(format)
        self.formats = format_list

        # If multiple formats are specified, enforce inplace=False
        if len(format_list) > 1:
            inplace = False
        self.inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Perform model quantization
        if self.static_kv_dtype is not None:
            from auto_round.experimental.kv_cache import kvcache_quant_context

            with kvcache_quant_context(self.model, static_kv_dtype=self.static_kv_dtype):
                model, _ = self.quantize()
        else:
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
            save_folder = self._get_save_folder_name(format)
            self.save_quantized(save_folder, format=format, inplace=inplace, **kwargs)

            folders.append(save_folder)

        return model, folders

    def _register_pre_func(self, func_name: str):
        """Registering a decorator for a pre-function"""
        def decorator(func):
            if func_name not in self._pre_functions:
                self._pre_functions[func_name] = []
            self._pre_functions[func_name].append(func)
            return func
        return decorator
    
    def _register_after_func(self, func_name: str):
        """Registering a decorator for a post-function"""
        def decorator(func):
            if func_name not in self._after_functions:
                self._after_functions[func_name] = []
            self._after_functions[func_name].append(func)
            return func
        return decorator
    
    def _wrap_function(self, func, func_name: str):
        """Wrapping functions to add pre- and after-execution logic"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # run pre-functions
            if func_name in self._pre_functions:
                for pre_func in self._pre_functions[func_name]:
                    pre_func(*args, **kwargs)
            
            # run ori-func
            result = func(*args, **kwargs)
            
            # run after-functions
            if func_name in self._after_functions:
                for after_func in self._after_functions[func_name]:
                    after_func(*args, **kwargs)
            
            return result
        return wrapper
    
    
    def _register_hooks(self):
        """register all hook functions"""
        self._register_pre_func('quantize')(self._check_compatibility)
    
    @classmethod
    def register(cls, *names: str):
        def func(quantizer: BaseQuantizer) -> BaseQuantizer:
            quantizer_type = quantizer.quantizer_type
            for name in names:
                cls._quantizer_classes[quantizer_type][name] = quantizer
            return quantizer

        return func

    @classmethod
    def print_registered_quantizers(cls):
        for quantizer_type, quantizer_classes in cls._quantizer_classes.items():
            logger.info(f"{quantizer_type.name} models:")
            for name in sorted(quantizer_classes.keys()):
                logger.info(f"  - {name}")