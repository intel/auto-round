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
import traceback
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, fields
from functools import partial
from typing import Any, Callable, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory
from torch import autocast
from tqdm import tqdm
from transformers import set_seed

from auto_round import envs
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors.utils import (
    IndexSampler,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    immediate_saving,
    infer_bits_by_data_type,
    init_cache,
    is_mx_fp,
    is_nv_fp,
    is_standard_fp,
    is_static_wfp8afp8,
    is_wfp8afp8,
    reset_params,
    set_layer_config,
)
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.export.export_to_gguf.config import GGUF_INNER_CONFIG, ModelType
from auto_round.formats import OutputFormat, get_formats
from auto_round.logger import logger
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    get_gguf_scheme,
    preset_name_to_scheme,
)
from auto_round.sign_sgd import SignSGD
from auto_round.special_model_handler import update_module
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_DTYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    check_and_mark_fp8_model,
    check_seqlen_compatible,
    check_to_quantized,
    clear_memory,
    compile_func,
    convert_dtype_str2torch,
    convert_fp8_layer_to_linear,
    convert_fp8_model_to_16b_model,
    copy_python_files_from_model_cache,
    detect_device,
    find_matching_blocks,
    flatten_list,
    get_block_names,
    get_layer_names_in_block,
    get_module,
    htcore,
    is_auto_device_mapping,
    is_debug_mode,
    is_fp8_linear,
    is_fp8_model,
    is_hpex_available,
    is_moe_model,
    llm_load_model,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
    to_dtype,
    unsupported_meta_device,
)
from auto_round.utils.device import (
    clear_memory_if_reached_threshold,
    get_major_device,
    parse_available_devices,
    set_auto_device_map_for_block_with_tuning,
    set_non_auto_device_map,
)
from auto_round.wrapper import WrapperLinear, WrapperMultiblock, unwrapper_block, unwrapper_layer, wrapper_block

SERIALIZATION_KEYS = (
    "bits",
    "act_bits",
    "data_type",
    "act_data_type",
    "group_size",
    "act_group_size",
    "sym",
    "act_sym",
    "act_dynamic",
    "amp",
    "batch_size",
    "enable_minmax_tuning",
    "enable_norm_bias_tuning",
    "enable_quanted_input",
    "gradient_accumulate_steps",
    "iters",
    "lr",
    "low_gpu_mem_usage",
    "minmax_lr",
    "nsamples",
    "quant_block_list",
    "regex_config",
    "scale_dtype",
    "seqlen",
    "supported_types",
    "static_attention_dtype",
    "static_kv_dtype",
    "super_bits",
    "super_group_size",
    "to_quant_block_names",
)


class BaseCompressor(object):
    """Base compressor for LLM quantization

    Attributes:
        model (torch.nn.Module): The loaded PyTorch model in eval mode.
        tokenizer: Tokenizer used to prepare input text for calibration/tuning.
        platform (str): The platform to load pretrained moded, options: ["hf", "model_scope"]
        bits (int): Weight quantization bits.
        group_size (int): Per-group size for weight quantization.
        sym (bool): Whether to use symmetric weight quantization.
        layer_config (dict): Per-layer quantization configuration.
        nsamples (int): Number of calibration samples.
        enable_torch_compile (bool): Whether to enable compile_func for quant blocks/layers.
    """

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
        platform="hf",
        scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16",
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
        enable_alg_ext: bool = False,
        disable_opt_rtn: bool | None = None,
        seed: int = 42,
        low_cpu_mem_usage: bool = False,
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
            low_cpu_mem_usage (bool, optional): Lower CPU memory mode. Defaults to False.
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
            disable_opt_rtn (bool, optional): Disable RTN-mode optimization (iters=0). Defaults to None.
            enable_alg_ext (bool, optional): Enable algorithm extension (primarily for INT2). Defaults to False.
            **kwargs: Backward compatible options:
                - enable_alg_ext, quant_lm_head, lr, lr_scheduler, not_use_best_mse, dynamic_max_gap,
                  super_group_size, super_bits, scale_dtype ("fp16" etc.),
                  nblocks, to_quant_block_names,
                  enable_norm_bias_tuning, enable_quanted_input,
                  disable_deterministic_algorithms, mllm, static_kv_dtype,enable_deterministic_algorithms,momentum
        Raises:
            ValueError: If invalid device is provided or tokenizer is missing for non-str model with iters > 0.
            RuntimeError: If model parameters are on meta device.
        Example:
            Layer-wise configuration structure:

            >>> layer_config = {
            ...     "layer1": {
            ...         "bits": 3,
            ...         "group_size": 128,
            ...         "sym": True,
            ...     },
            ...     "layer2": {
            ...         "W8A16"
            ...      }
            ...     # ...
            ... }
        """

        # Model related
        model_dtype = kwargs.pop("model_dtype", None)
        self.mllm = kwargs.pop("mllm") if "mllm" in kwargs else False
        self.diffusion = kwargs.pop("diffusion") if "diffusion" in kwargs else False
        self.quantized = False
        if isinstance(model, str):
            model, tokenizer = llm_load_model(
                model,
                platform=platform,
                device="cpu",  # always load cpu first
                model_dtype=model_dtype,
            )
        elif tokenizer is None and not self.diffusion and iters > 0:
            raise ValueError("A tokenizer must be set for non-str model input")
        if unsupported_meta_device(model):
            raise RuntimeError(
                "AutoRound does not support parameters on meta device. "
                "Please use more GPUs by setting `--device 0,1,2,3` or just place the model on CPU."
            )
        check_and_mark_fp8_model(model)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.shared_cache_keys = get_shared_keys(self.model)

        self.layer_config = layer_config

        # should be set after loading model and set layer_config, cause some special scheme need these.
        self.scheme, self.is_auto_scheme = self._parse_and_set_scheme(scheme, kwargs)

        gguf_scheme_name = get_gguf_scheme(self.scheme)
        # GGUF uses fp32 scale dtype as default
        scale_dtype = kwargs.pop("scale_dtype", None)
        if scale_dtype is None:
            scale_dtype = "fp32" if gguf_scheme_name else "fp16"

        # Extra/legacy kwargs for backward compatibility
        # Major version releases may pack them with extra configuration options
        amp = kwargs.pop("amp", True)
        lr = kwargs.pop("lr", None)
        enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)
        minmax_lr = kwargs.pop("minmax_lr", None)
        lr_scheduler = kwargs.pop("lr_scheduler", None)
        not_use_best_mse = kwargs.pop("not_use_best_mse", False)
        dynamic_max_gap = kwargs.pop("dynamic_max_gap", -1)
        nblocks = kwargs.pop("nblocks", 1)
        to_quant_block_names: Union[str, list, None] = kwargs.pop("to_quant_block_names", None)
        enable_norm_bias_tuning: bool = kwargs.pop("enable_norm_bias_tuning", False)
        enable_quanted_input: bool = kwargs.pop("enable_quanted_input", True)
        disable_deterministic_algorithms = kwargs.pop("disable_deterministic_algorithms", True)
        enable_deterministic_algorithms = kwargs.pop("enable_deterministic_algorithms", False)
        self.momentum = kwargs.pop("momentum", 0.0)
        static_kv_dtype = kwargs.pop("static_kv_dtype", None)
        static_attention_dtype = kwargs.pop("static_attention_dtype", None)
        device = kwargs.pop("device", None)
        if envs.AR_USE_MODELSCOPE:
            platform = "model_scope"
        self.platform = platform
        self.quant_lm_head = kwargs.pop("quant_lm_head", False)

        self.ignore_layers = kwargs.pop("ignore_layers", "")
        self.supported_types = SUPPORTED_LAYER_TYPES
        self.inner_supported_types = INNER_SUPPORTED_LAYER_TYPES
        self.scale_dtype = convert_dtype_str2torch(scale_dtype)
        self.low_cpu_mem_usage = low_cpu_mem_usage

        if kwargs:
            logger.warning(f"unrecognized keys {list(kwargs.keys())} were passed. Please check them.")
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Deprecated, default not to use torch.use_deterministic_algorithms
        if not disable_deterministic_algorithms or enable_deterministic_algorithms:
            if not disable_deterministic_algorithms:
                logger.warning(
                    "default not use deterministic_algorithms. disable_deterministic_algorithms is deprecated,"
                    " please use enable_deterministic_algorithms instead. "
                )

            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.to_quant_block_names = to_quant_block_names
        if not hasattr(self, "quant_block_list"):
            all_blocks = get_block_names(model)
            self.quant_block_list = find_matching_blocks(model, all_blocks, self.to_quant_block_names)

        if device is not None:
            logger.warning("`device` is deprecated, please use `device_map` instead")

        if device_map is None:
            device_map = 0

        self.device_map = device_map
        if isinstance(self.device_map, str):
            self.device_map = self.device_map.replace(" ", "")

        self.device_list = parse_available_devices(device_map)

        # Set device, must place after model loading
        self.device = get_major_device(device_map)
        set_non_auto_device_map(self.model, self.device_map)

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
            if not lr:
                # TODO need to check 4 bits lr setting for auto-round-best, 3bits only validate on small models
                if self.iters >= 1000 and self.bits <= 3:
                    self.lr = 2.0 / self.iters
                    logger.info("set the lr to 2.0/iters for better accuracy")
                else:
                    self.lr = 1.0 / self.iters
            else:
                self.lr = lr

        # Automatically adjust the disable_opt_rtn option if the user does not explicitly set it.
        # To avoid None issue, we keep a copy though it's a little ugly
        self.orig_disable_opt_rtn = disable_opt_rtn
        if self.iters != 0 and self.orig_disable_opt_rtn is not None:
            logger.warning("`disable_opt_rtn` only works when `iters` is set to 0, ignore it now.")
            disable_opt_rtn = True
        if (
            self.bits >= 8
            and self.act_bits >= 16
            and self.iters == 0
            and self.data_type == "int"
            and disable_opt_rtn is None
        ):
            logger.warning("`disable_opt_rtn` is turned on for W8A16 quantization to improve efficiency.")
            disable_opt_rtn = True
        if disable_opt_rtn is None and self.iters == 0:
            logger.info(
                "`enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy."
            )
            disable_opt_rtn = False

        # Important Note! This is not very robust, do NOT rely on it to do high risky thing
        self.is_moe_model = is_moe_model(self.model)

        self.minmax_lr = minmax_lr or self.lr
        self.enable_alg_ext = enable_alg_ext
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.lr_scheduler = lr_scheduler
        self.optimizer = self._get_optimizer(None)
        self.disable_opt_rtn = disable_opt_rtn

        # Whether to pack the layer immediately after tuning
        self.immediate_packing = False
        self.immediate_saving = False

        # KV cache, this one does not affect tuning but will collect some infos during tuning
        self.static_kv_dtype = static_kv_dtype
        if self.static_kv_dtype is not None:
            logger.warning("The static kv is experimental and currently has limited support.")

        # Attention static dtype
        self.static_attention_dtype = static_attention_dtype
        if self.static_attention_dtype is not None:
            logger.warning("The static attention dtype is experimental and currently has limited support.")

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
        if "hpu" in str(self.device):
            self.inner_supported_types = tuple(x for x in INNER_SUPPORTED_LAYER_TYPES if x != "FP8Linear")
        self.batch_dim = None
        self.infer_bs_coeff = 1

        # after setting iters
        self.enable_torch_compile = enable_torch_compile
        self._adjust_torch_compile(enable_torch_compile)

        self.block_forward = compile_func(block_forward, self.device) if self.enable_torch_compile else block_forward
        self._check_configs()
        torch.set_printoptions(precision=3, sci_mode=True)

        if isinstance(scheme, AutoScheme):
            self.layer_config = self._gen_auto_scheme(model, scheme, dataset, self.device_map)

        if is_hpex_available():
            logger.info("habana_frameworks is available, import htcore explicitly.")
            import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401]

        self.attention_mask = []

        self.wrapper_block = wrapper_block
        if self.enable_alg_ext:
            try:
                logger.warning_once("using algorithm extension for quantization.")
                from auto_round.alg_ext import wrapper_autoround

                wrapper_autoround(self)
            except (ImportError, ModuleNotFoundError):
                logger.error("algorithm extension import error, fallback to default mode")

    def _gen_auto_scheme(
        self, model: torch.nn.Module, scheme: AutoScheme, dataset: str, device_map: Union[str, int, dict, torch.device]
    ) -> dict[str, dict]:
        if self.mllm:
            logger.info("AutoScheme is not yet supported for multimodal LLMs.")
            sys.exit(-1)

        if getattr(model, "is_fp8", False):
            logger.info("AutoScheme does not currently support FP8 models.")
            sys.exit(-1)

        all_dtypes = []
        all_gguf = True
        for option in scheme.options:
            # Resolve the quantization scheme or data type
            dtype = "int"
            if isinstance(option, str):
                if not option.lower().startswith("gguf"):
                    all_gguf = False

                option = preset_name_to_scheme(option)

            else:
                all_gguf = False

            if isinstance(option, QuantizationScheme):
                dtype = option.data_type
            elif isinstance(option, dict):
                dtype = option.get("data_type", "int")

            all_dtypes.append(dtype)

        # Check for mixed data types
        unique_dtypes = set(all_dtypes)
        if len(unique_dtypes) > 1 and not all_gguf:
            logger.warning(
                "Models with mixed data_types "
                "cannot yet be exported to real formats except GGUF. "
                "Please save the model using the `fake` format for now."
            )

        layer_config, self.has_qlayer_outside_block, self.regex_config = set_layer_config(
            self.model,
            self.layer_config,
            self.scheme,
            self.scale_dtype,
            self.supported_types,
            self.inner_supported_types,
            self.quant_block_list,
            self.ignore_layers,
            self.quant_lm_head,
            enable_gguf_official_mixed=False,
            is_mllm=self.mllm,
        )
        quant_layer_names = layer_config.keys()
        scheme_keys = {f.name for f in fields(QuantizationScheme)}
        fixed_layer_scheme_new = {
            k: {key: v[key] for key in scheme_keys & v.keys()}
            for k, v in layer_config.items()
            if v.get("fixed_by_user", False)
        }

        # mainly using quant_layers and fixed by users
        from auto_round.auto_scheme.gen_auto_scheme import GenScheme

        if not self.enable_torch_compile and self.super_bits is None and not scheme.low_gpu_mem_usage:
            logger.warning("we strongly recommend to set `enable_torch_compile` to True for AutoScheme to save VRAM")
        self.scheme_generator = GenScheme(
            scheme,
            self.model,
            quant_layer_names,
            fixed_layer_scheme_new,
            dataset,
            device_map=device_map,
            tokenizer=self.tokenizer,
            enable_torch_compile=self.enable_torch_compile,
        )
        layer_config = self.scheme_generator.get_layer_config()
        return layer_config

    def _set_device(self, device_map: Union[str, torch.device, int, dict]) -> None:
        if hasattr(self, "device") and self.device is not None:
            return
        if isinstance(device_map, (str, torch.device, int)):
            self.device = detect_device(device_map)

        elif isinstance(device_map, dict) and device_map:
            tmp_devices = []
            for val in device_map.values():
                if isinstance(val, (str, torch.device, int)):  # could optimize
                    tmp_device = detect_device(val)
                    tmp_device = tmp_device.split(":")[0]
                    tmp_devices.append(tmp_device)
            tmp_devices = list(set(tmp_devices))
            if len(tmp_devices) > 1:
                logger.warning(
                    f"there are multiple device types in the device_map, "
                    f"please make sure they are correct,use the first device {tmp_devices[0]} as the core device."
                )

            self.device = tmp_devices[0]
        else:
            raise TypeError(f"device_map should be [str, torch.device, int, dict], but got {type(device_map)}")

    def _parse_and_set_scheme(
        self, scheme: Union[str, dict, QuantizationScheme], kwargs
    ) -> tuple[QuantizationScheme, bool]:
        """Parse and set the quantization scheme."""

        def _parse_and_set(scheme, kwargs):
            if kwargs.get("data_type", None) and kwargs["data_type"].endswith("_dq") and not scheme.startswith("gguf"):
                if "bits" not in kwargs:
                    data_type = kwargs["data_type"]
                    raise KeyError(
                        f"please set bits when setting data_type={data_type}, or using scheme as an alternative."
                    )
                bits = kwargs["bits"]
                scheme = f"gguf:q{bits}_k" if bits == 6 else f"gguf:q{bits}_k_s"
            res = None
            if isinstance(scheme, QuantizationScheme):
                scheme = asdict(scheme)
            elif isinstance(scheme, dict):
                scheme = scheme
            elif isinstance(scheme, str):
                # We’d better keep the string scheme instead of the dict config,
                # since GGUF uses different mixed-bit strategies for q4_k_s and q4_k_m
                # even though they share the same scheme dict.
                scheme = scheme.strip("'\" ")
                res = scheme
                scheme = scheme.upper()
                self.layer_config = _handle_special_schemes(scheme, self.layer_config, self.model)
                scheme = asdict(preset_name_to_scheme(scheme))
            scheme_keys = [f.name for f in fields(QuantizationScheme)]
            for key in scheme_keys:
                if key in kwargs and kwargs[key] is not None:
                    setattr(self, key, kwargs[key])
                else:
                    setattr(self, key, scheme.get(key, None))
                # kwargs.pop(key, None)
            if self.act_dynamic is None:
                self.act_dynamic = True

            tmp_bits = infer_bits_by_data_type(self.data_type)
            if tmp_bits is not None and tmp_bits < 16 and tmp_bits != self.bits:
                logger.warning(
                    f"'data_type' do not match the specified 'bits' setting. Resetting 'bits' to {tmp_bits}."
                )
                self.bits = tmp_bits
            if tmp_bits is not None and tmp_bits < 16:
                for (
                    supported_dtype
                ) in SUPPORTED_DTYPES:  # to easily handle dtype mx_fp4 and layer_config={xxx:{bits:8}}
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
                for (
                    supported_dtype
                ) in SUPPORTED_DTYPES:  # To easily handle dtype mx_fp4 and layer_config={xxx:{bits:8}}
                    if self.act_data_type.startswith(supported_dtype):
                        if supported_dtype + str(tmp_act_bits) == self.act_data_type:  # Could not replace FP8_e4m3
                            self.act_data_type = supported_dtype
                        break
            for key in scheme_keys:
                scheme[key] = getattr(self, key)
            if res and QuantizationScheme.from_dict(scheme) == preset_name_to_scheme(res):
                return res
            else:
                return QuantizationScheme.from_dict(scheme)

        if isinstance(scheme, AutoScheme):
            if len(scheme.options) <= 0:
                raise ValueError("options of AutoScheme must not be empty")
            options = []
            for option in scheme.options:
                new_option = _parse_and_set(option, kwargs)
                options.append(new_option)
            scheme.options = options
            for opt in options:
                if isinstance(opt, str) and opt == "BF16":
                    continue
                if isinstance(opt, QuantizationScheme):
                    if opt.bits >= 16 and (opt.act_bits is None or opt.act_bits >= 16):
                        continue
                self.scheme = opt  # Choose the first one that not 16 bits
                break
            # apply scheme to set default bits
            scheme = _parse_and_set(self.scheme, kwargs)
            is_auto_scheme = True
        else:
            scheme = _parse_and_set(scheme, kwargs)
            is_auto_scheme = False

        scheme_keys = [f.name for f in fields(QuantizationScheme)]
        for key in scheme_keys:
            kwargs.pop(key, None)

        return scheme, is_auto_scheme

    def _adjust_torch_compile(self, enable_torch_compile: bool) -> None:
        """Sets the torch compile configuration for the tuning."""
        self.enable_torch_compile = enable_torch_compile
        if (
            not self.enable_torch_compile
            and TORCH_VERSION_AT_LEAST_2_6
            and self.act_bits > 8
            and not is_debug_mode()
            and "fp8" not in self.data_type
            and "fp8" not in self.act_data_type
            and self.iters > 0
        ):
            logger.info(
                "%s",
                "'enable_torch_compile' is set to `False` by default. "
                "Enabling it can reduce tuning cost by 20%, but it might throw an exception.",
            )
        # On HPU, we rely on torch.compile to speed up the model execution.
        if self.enable_torch_compile and is_wfp8afp8(self) and not is_hpex_available():
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as fp8 is enabled")
        # TODO: fix https://github.com/intel/auto-round/issues/1109
        if self.enable_torch_compile and is_nv_fp(self.act_data_type):
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as nvfp4 is enabled")

    def _dq_check(self) -> None:
        """Reset the default value of super_bits and super_group_size"""
        if self.data_type.endswith("_dq"):
            gguf_config = GGUF_INNER_CONFIG[f"gguf:q{self.bits}_k"]
            self.super_bits = gguf_config.get("super_bits", None) if self.super_bits is None else self.super_bits
            self.super_group_size = (
                gguf_config.get("super_group_size", None) if self.super_group_size is None else self.super_group_size
            )

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

        if (
            self.act_bits <= 8
            and (not is_nv_fp(self.act_data_type) or "static_gs" not in self.act_data_type)
            and not is_mx_fp(self.act_data_type)
        ):
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

        self._dq_check()

    def _check_compatibility(self) -> None:
        """Checks compatibility of the configurations and model."""
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

        if self.bits <= 2 and (self.iters < 1000 or not self.enable_alg_ext) and self.super_group_size is None:
            logger.warning(
                "for bits <= 2, it is recommended to enable `auto-round-best` " "and turn on `--enable_alg_ext` "
            )

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
        format_list = get_formats(format, self)
        self.formats = format_list

        # If multiple formats are specified, enforce inplace=False
        if len(format_list) > 1:
            inplace = False
        self.inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Perform model quantization
        if self.static_attention_dtype is not None:
            from auto_round.experimental.attention import attention_quant_ctx

            with attention_quant_ctx(self.model, static_attention_dtype=self.static_attention_dtype):
                model, _ = self.quantize()
        elif self.static_kv_dtype is not None:
            from auto_round.experimental.kv_cache import kvcache_quant_context

            with kvcache_quant_context(self.model, static_kv_dtype=self.static_kv_dtype):
                model, _ = self.quantize()
        else:
            model, _ = self.quantize()
        # Save the quantized model in the specified format_list
        model, folders = self.save_quantized(output_dir, format=format, inplace=inplace, return_folders=True, **kwargs)
        memory_monitor.log_summary()

        return model, folders

    def _get_save_folder_name(self, format: OutputFormat) -> str:
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
        sanitized_format = format.get_backend_name().replace(":", "-").replace("_", "-")

        # Use a subfolder only if there are multiple formats
        if len(self.formats) > 1:
            return os.path.join(self.orig_output_dir, sanitized_format)

        return self.orig_output_dir

    def _immediate_pack(self, name: str):
        if not self.immediate_packing:
            return
        self.formats[0].immediate_pack(
            name=name,
            model=self.model,
            device=self.device,
            output_dir=self._get_save_folder_name(self.formats[0]),
            mllm=self.mllm,
            layer_config=self.layer_config,
            tokenizer=self.tokenizer,
            processor=self.processor if hasattr(self, "processor") else None,
            image_processor=self.image_processor if hasattr(self, "image_processor") else None,
        )

    def _update_inputs(self, inputs: dict, q_inputs: dict) -> tuple[dict, torch.Tensor]:
        keys = inputs.keys()
        input_id_str = [key for key in keys if key.startswith("hidden_state")]
        if len(input_id_str) != 1:
            raise RuntimeError(
                "hidden_states arg mismatch error,"
                "please raise an issue in https://github.com/intel/auto-round/issues"
            )
        inputs["input_ids"] = inputs.pop(input_id_str[0], None)
        if q_inputs is not None:
            q_inputs = q_inputs.pop(input_id_str[0], None)
        return inputs, q_inputs

    def configure_layer_config(self, enable_gguf_official_mixed: None | bool = True):
        fill_default_value = True
        if self.is_auto_scheme:
            fill_default_value = False
        self.layer_config, self.has_qlayer_outside_block, self.regex_config = set_layer_config(
            self.model,
            self.layer_config,
            self.scheme,
            self.scale_dtype,
            self.supported_types,
            self.inner_supported_types,
            self.quant_block_list,
            self.ignore_layers,
            self.quant_lm_head,
            enable_gguf_official_mixed=enable_gguf_official_mixed,
            is_mllm=self.mllm,
            fill_default_value=fill_default_value,
        )

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """

        self._check_compatibility()
        formats = self.formats if hasattr(self, "formats") else None
        # It is best to modify the model structure in the quantize function and check the format,
        # because it may cause the gguf format to not be exported normally.
        self.model = update_module(self.model, formats=formats)

        # Temporary names must be assigned after handle_moe_model;
        # placing them earlier would cause them to be removed when the module is replaced.
        for n, m in self.model.named_modules():
            m.tmp_name = n

        if not self.is_auto_scheme:
            enable_gguf_official_mixed = True
        else:
            enable_gguf_official_mixed = False

        self.configure_layer_config(enable_gguf_official_mixed=enable_gguf_official_mixed)
        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            # Determine if immediate packing is required
            formats = self.formats
            if len(formats) == 1 and not formats[0].is_fake() and self.inplace:
                self.immediate_packing = True
                if not formats[0].is_gguf() and self.low_cpu_mem_usage:
                    self.immediate_saving = True

        if self.immediate_saving and "int" not in self.data_type:
            logger.warning("immediate_saving is only supported for int quantization, set to False")
            self.immediate_saving = False

        from auto_round.quantizers import create_quantizers

        quantizers = create_quantizers(self)
        return quantizers.quantize()

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
                        _all_module_input[module._tmp_name].append((args, kwargs))
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
        from auto_round.calib_dataset import get_dataloader

        need_attention_mask = True
        if isinstance(self.dataset, str):
            need_attention_mask = False  # all supported datasets does not use pad
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
        if self.dataloader.__class__.__name__ == "BatchEncoding":
            self.dataloader = [self.dataloader.data]

        for data in self.dataloader:
            if data.__class__.__name__ == "BatchEncoding":
                data = data.data
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
                data_new = to_device(data, self.model.device)
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
            if need_attention_mask:
                if (
                    isinstance(data_new, dict)
                    and "attention_mask" in data_new
                    and data_new["attention_mask"] is not None
                ):
                    new_attention_mask = data_new["attention_mask"]
                elif (
                    self.tokenizer is not None
                    and hasattr(self.tokenizer, "pad_token")
                    and self.tokenizer.pad_token is not None
                ):
                    new_attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
                else:
                    # Default all ones
                    new_attention_mask = torch.ones_like(input_ids, dtype=torch.long)

                    # For each sample, check if there are trailing repeated tokens
                    # If so, set the mask of the last token to 0
                    batch_size, seq_len = input_ids.shape
                    for i in range(batch_size):
                        last_token = input_ids[i, -1]
                        # Check for trailing repeats
                        j = seq_len - 2
                        repeated = False
                        while j >= 0 and input_ids[i, j] == last_token:
                            repeated = True
                            new_attention_mask[i, j] = 0
                            j -= 1
                        # If there was at least one repeat, set last token mask to 0
                        if repeated:
                            new_attention_mask[i, -1] = 0

                # Workaround: some models treat an all-1 attention mask as equivalent to None and
                # will internally replace it with None for block inputs, which can cause tensor
                # concatenation / shape-mismatch issues in downstream code. To avoid providing an
                # all-1 mask, we force the last token in each sequence to be masked out (set to 0)
                # so that the mask is never "all ones". This means the model will not attend to the
                # last position, so the impact on accuracy is minimal as basically equivalent to dropping a single token
                new_attention_mask[:, -1] = 0

                self.attention_mask.extend(list(torch.split(new_attention_mask, 1, dim=0)))
            else:
                new_attention_mask = None
            try:
                kwargs = {"use_cache": False}
                if new_attention_mask is not None and not (isinstance(data_new, dict) and "attention_mask" in data_new):
                    kwargs["attention_mask"] = new_attention_mask

                if isinstance(data_new, torch.Tensor):
                    self.model(data_new, **kwargs)
                elif isinstance(data_new, tuple) or isinstance(data_new, list):
                    self.model(*data_new, **kwargs)
                else:
                    self.model(**data_new, **kwargs)
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
                f"An insufficient number of samples likely reduces the accuracy of the quantized model. "
                f"Target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )

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
        if is_fp8_model(self.model):
            layer_names = []
        if layer_names is None:
            layer_names = []
        if self.low_gpu_mem_usage or (
            len(block_names) == 1
            and len(layer_names) == 0
            and not self.has_qlayer_outside_block
            and (last_cache_name is None or last_cache_name in block_names)
            and getattr(self, "mllm", False) is False
        ):
            # low_gpu_mem_usage or calibrate only the embedding layer, which is also very fast on CPU
            all_inputs = self.cache_inter_data(block_names, nsamples, layer_names=[], last_cache_name=last_cache_name)
        else:
            try:
                if not self.model.device.type == "meta":
                    if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                        self.model = dispatch_model(self.model, device_map=self.model.hf_device_map)
                    else:
                        # Change this if new device is supported
                        if str(self.model.device) == "cpu" and (not self.device.startswith("hpu")):
                            no_split_modules = getattr(self.model, "_no_split_modules", [])
                            devices = parse_available_devices(self.device_map)

                            max_memory = get_max_memory()
                            new_max_memory = {}
                            if "cpu" not in devices:
                                devices.append("cpu")
                            for device in devices:
                                if ":" in device:
                                    device = int(device.split(":")[-1])
                                elif device == "cpu":
                                    device = "cpu"
                                else:
                                    raise ValueError(f"Unsupported device {device} in device_map: {self.device_map}")
                                # Use 90% of the reported max memory to leave headroom for activations,
                                # temporary tensors, other processes, and allocator fragmentation, reducing
                                # the chance of runtime OOM while still utilizing most available memory.
                                new_max_memory[device] = max_memory[device] * 0.9
                            new_max_memory = get_balanced_memory(
                                self.model,
                                max_memory=new_max_memory,
                                no_split_module_classes=no_split_modules,
                            )
                            device_map = infer_auto_device_map(
                                self.model, max_memory=new_max_memory, no_split_module_classes=no_split_modules
                            )
                            if len(devices) > 1 and "cpu" in device_map.values():
                                logger.warning(
                                    "Some layers are offloaded to cpu, which may severely impact calibration speed."
                                    " Please consider using more cards."
                                )

                            try:
                                self.model = dispatch_model(self.model, device_map=device_map)
                            except ValueError as e:
                                if "offload_dir" in e.__str__():
                                    logger.warning(
                                        f"Due to insufficient resources, disk is used to store the model."
                                        f" `offload_dir={envs.AR_WORK_SPACE}`"
                                    )
                                    self.model = dispatch_model(
                                        self.model, device_map=device_map, offload_dir=envs.AR_WORK_SPACE
                                    )
                                else:
                                    raise
                        else:
                            self.model = self.model.to(self.device)

                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
                )
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    accelerate.hooks.remove_hook_from_submodules(self.model)

            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.info("switch to cpu to cache block inputs")
                    if self.has_qlayer_outside_block or self.__class__.__name__ == "AutoRoundMLLM":
                        logger.warning(
                            "we recommend using more GPUs in calibration."
                            " Otherwise, some layers may fall back to `rtn` mode, which can affect accuracy."
                        )
                    if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                        accelerate.hooks.remove_hook_from_submodules(
                            self.model
                        )  # self.model.hf_device_map has not been changed
                    self.model = mv_module_from_gpu(self.model)
                    clear_memory(device_list=self.device_list)
                    # Important change after v0.51, on cpu, we use rtn mode for layers in layer_names
                    all_inputs = self.cache_inter_data(
                        block_names, nsamples, layer_names=[], last_cache_name=last_cache_name
                    )
                except Exception as e:
                    logger.error(cuda_error_msg)
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

        tmp_dtype = None  # TODO delete this as most model is not fp32 now
        ## have bug if block name is not the first block
        if (len(block_names) > 1 or len(layer_names) > 0) and self.low_gpu_mem_usage:
            tmp_dtype = self.model.dtype
            if self.amp:
                if self.model.dtype != self.model.dtype:
                    self.model = self.model.to(torch.bfloat16)
            else:
                self.model = self.model.to(torch.float32)  ##model on cpu

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

        for n, m in self.model.named_modules():
            if n in self.to_cached_layers and type(m) not in self.supported_types:  ##block
                m.orig_forward = m.forward
                m.forward = partial(self._get_block_forward_func(n), m)
            elif n in self.to_cached_layers:  ##linear layer or conv1d layer
                hook_func = self._get_cache_data_hook_for_layer(n)
                hook_handle = m.register_forward_hook(hook_func)
                self.hook_handles.append(hook_handle)

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
            share_cache_keys=self.shared_cache_keys,
        )
        output_q = self.block_forward(block, current_input_ids, current_input_others, self.amp, self.amp_dtype, device)
        return output_q.to(cache_device)

    def _get_current_num_elm(
        self,
        input_ids: list[torch.Tensor],
        indices: list[int],
    ) -> int:
        current_input_ids = [input_ids[i] for i in indices]
        return sum(id.numel() for id in current_input_ids)

    def quantize_block(
        self,
        block: torch.nn.Module,
        inputs: tuple[Union[list[torch.Tensor], dict, Any], Optional[dict]],
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload=True,
    ):
        """
        This function quantizes a specific decoded block of a model.
        It is primarily used by LLM-Compressor. For more details, please refer to the following PR:
        https://github.com/vllm-project/llm-compressor/pull/1994
        """

        # TODO: release below assertion after supporting MLLM and diffusion model quantization with quantize_block
        assert self.__class__.__name__ not in [
            "DiffusionCompressor",
            "MLLMCompressor",
        ], f"Currently, {self.__class__.__name__} does not support support quantize block with this function."
        self.normalize_decoding_layer_inputs_(inputs)
        block_inputs = self.inputs[self.quant_block_list[0][0]]
        decoding_layer_first_input_name = "hidden_states"
        from auto_round.quantizers.algs.auto_round import ARQuantizer
        from auto_round.quantizers.utils import preprocess_block_inputs

        input_ids, input_others = preprocess_block_inputs(
            block_inputs,
            device_list=self.device_list,
            first_input_name=decoding_layer_first_input_name,
            amp=self.amp,
            amp_dtype=self.amp_dtype,
            cache_device=self.cache_device,
            diffusion=self.diffusion,
        )
        ar_quantizer = ARQuantizer(self)
        return ar_quantizer.quantize_block(block, input_ids, input_others, q_input, device, auto_offload)

    def _get_loss(
        self,
        output_q: torch.Tensor,
        current_output: torch.Tensor,
        indices: torch.Tensor,
        mse_loss: Callable,
        device: Union[str, torch.device] = "cpu",
    ):
        autocast_ctx = (
            nullcontext() if self.amp else autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype)
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

    def save_quantized(
        self,
        output_dir: str = None,
        format: Union[str, list[OutputFormat]] = "auto_round",
        inplace: bool = True,
        return_folders=False,
        **kwargs,
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
        self.orig_output_dir = output_dir
        if isinstance(format, str):
            formats = get_formats(format, self)
            if not hasattr(self, "formats"):
                self.formats = formats

        if not self.quantized:
            logger.warning("please run autoround.quantize first")
            return
        folders = []
        for format in formats:
            save_folder = self._get_save_folder_name(format)
            if self.act_bits <= 8 and format.is_fake():
                logger.warning(
                    "Support for exporting activation quantization is limited. "
                    "Please ensure that your configuration is supported."
                )

            serialization_dict = {}
            for key in SERIALIZATION_KEYS:
                serialization_dict[key] = getattr(self, key)
            from auto_round.version import __version__

            serialization_dict["autoround_version"] = __version__
            if "scale_dtype" in serialization_dict.keys():
                serialization_dict["scale_dtype"] = str(serialization_dict["scale_dtype"])
            compressed_model = format.save_quantized(
                save_folder,
                model=self.model,
                layer_config=self.layer_config,
                inplace=inplace,
                tokenizer=self.tokenizer,
                device=self.device,
                serialization_dict=serialization_dict,
                **kwargs,
            )
            folders.append(save_folder)

        if return_folders:
            return compressed_model, folders
        else:
            return compressed_model

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

    def _get_optimizer(self, optimizer: Any):
        """Returns the specified optimizer. In SignRound, we fix the optimizer.

        Args:
        optimizer: The optimizer to be used.

        Returns:
        The specified optimizer.
        """
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


class LLMCompressor(BaseCompressor):
    pass
