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
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory
from tqdm import tqdm
from transformers import AutoConfig, set_seed

from auto_round import envs
from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.base import BaseAlgorithm
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig
from auto_round.algorithms.quantization.auto_round.quantize import ARQuantizer
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.calibration.utils import (
    _infer_last_cache_name,
    _update_inputs,
)
from auto_round.compressors.shard_writer import shard_writer
from auto_round.compressors_new.utils import (
    IndexSampler,
    _get_quantized_layer_names_outside_blocks,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    infer_bits_by_data_type,
    init_cache,
    reset_params,
    set_layer_config,
)
from auto_round.context.compress_context import CompressContext
from auto_round.context.model_context import ModelContext
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.export.export_to_gguf.config import GGUF_INNER_CONFIG
from auto_round.formats import OutputFormat, get_formats
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    _parse_scheme,
    get_gguf_scheme,
    preset_name_to_scheme,
)
from auto_round.special_model_handler import get_predefined_ignore_layers, update_module
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_DTYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    check_and_mark_quantized_module,
    check_seqlen_compatible,
    check_to_quantized,
    clear_memory,
    compile_func,
    convert_dtype_str2torch,
    convert_module_to_hp_if_necessary,
    detect_device,
    find_matching_blocks,
    flatten_list,
    get_block_names,
    get_layer_names_in_block,
    get_lm_head_name,
    get_module,
    global_state,
    htcore,
    is_auto_device_mapping,
    is_debug_mode,
    is_hpex_available,
    is_moe_model,
    is_moe_model_via_config,
    is_quantized_input_module,
    llm_load_model,
    memory_monitor,
    mv_module_from_gpu,
    safe_device_move_with_meta_handling,
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
from auto_round.utils.offload import OffloadManager
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


class BackendDataType(str, Enum):
    STANDARD_FP = "fp"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"


@dataclass
class QuantizationArgs:
    bits: int = None
    group_size: int = None
    sym: bool = None
    data_type: str = None
    act_bits: int = None
    act_group_size: int = None
    act_sym: bool = None
    act_data_type: str = None
    act_dynamic: bool = None
    super_bits: int = None
    super_group_size: int = None

    def is_act_quantize(self):
        return self.act_bits is not None and self.act_bits <= 8

    def is_nv_fp(self, act=False):
        data_type = self.data_type if act is False else self.act_data_type
        return BackendDataType.NV_FP in data_type

    def is_mx_fp(self, act=False):
        data_type = self.data_type if act is False else self.act_data_type
        return BackendDataType.MX_FP in data_type

    def is_dynamic_wint8aint8(self):
        if self.act_dynamic:
            return True
        if ("int8" in self.act_data_type or ("int" in self.act_data_type and self.act_bits == 8)) and (
            "int8" in self.data_type or ("int" in self.data_type and self.bits == 8)
        ):
            return True
        return False

    def is_static_wfp8afp8(self, act=False):
        data_type = self.data_type if act is False else self.act_data_type
        return "fp8_static" in data_type

    def is_standard_fp(self, act=False):
        data_type = self.data_type if act is False else self.act_data_type
        return BackendDataType.STANDARD_FP in data_type and not self.is_mx_fp(act=act) and not self.is_nv_fp(act=act)

    def is_wfp8afp8(self):
        if (
            ("fp8" in self.act_data_type or ("fp" in self.act_data_type and self.act_bits == 8))
            and ("fp8" in self.data_type or ("fp" in self.data_type and self.bits == 8))
            and self.is_standard_fp(act=True)
            and self.is_standard_fp(act=False)
        ):
            return True
        else:
            return False

    @classmethod
    def from_dict(cls, config: dict):
        new_config = {}
        for k, v in config.items():
            if hasattr(cls, k):
                new_config[k] = v
        return cls(**new_config)

    def non_default(self):
        config = {}
        for k, v in asdict(self).items():
            if v:
                config[k] = v
        return config

    def check_config(self):
        if self.bits <= 0:
            raise ValueError("`bits` must be positive")
        if self.act_bits <= 0:
            raise ValueError("`act_bits` must be positive")
        if not (self.group_size == -1 or self.group_size >= 0):
            raise ValueError("`group_size` must be -1 (per channel) or 0 (per-tensor) or a positive integer")
        if not (self.act_group_size == -1 or self.act_group_size >= 0):
            raise ValueError("`act_group_size` must be -1 (per channel) or 0 (per-tensor) or a positive integer")
        """Reset the default value of super_bits and super_group_size"""
        if self.data_type.endswith("_dq"):
            gguf_config = GGUF_INNER_CONFIG[f"gguf:q{self.bits}_k"]
            self.super_bits = gguf_config.get("super_bits", None) if self.super_bits is None else self.super_bits
            self.super_group_size = (
                gguf_config.get("super_group_size", None) if self.super_group_size is None else self.super_group_size
            )


class Compressor(object):
    SKIP_ARGS = ("local_args", "kwargs", "cls", "config")

    def __new__(
        cls,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16",
        platform="hf",
        format=None,
        **kwargs,
    ):
        # using different compressor base on AlgConfigs
        local_args = {k: v for k, v in locals().items() if k not in cls.SKIP_ARGS}

        if isinstance(config, AutoRoundConfig):
            return BaseCompressor(ARQuantizer(config), **local_args, **kwargs)


class BaseCompressor(object):
    need_calib: bool = True

    def __init__(
        self,
        algorithms: Union[BaseAlgorithm, list[BaseAlgorithm]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
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
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ):
        self.quantization_args = QuantizationArgs.from_dict(kwargs)

        self.algorithms = algorithms if isinstance(algorithms, list) else [algorithms]
        # TODO: refactor calibration
        self.calibration = None

        self.scheme = scheme
        self.formats = format
        self.layer_config = layer_config
        self.seqlen = seqlen

        # Extra/legacy kwargs for backward compatibility
        # Major version releases may pack them with extra configuration options
        amp = kwargs.pop("amp", True)
        not_use_best_mse = kwargs.pop("not_use_best_mse", False)
        dynamic_max_gap = kwargs.pop("dynamic_max_gap", -1)
        nblocks = kwargs.pop("nblocks", 1)
        to_quant_block_names: Union[str, list, None] = kwargs.pop("to_quant_block_names", None)
        enable_quanted_input: bool = kwargs.pop("enable_quanted_input", True)
        disable_deterministic_algorithms = kwargs.pop("disable_deterministic_algorithms", True)
        enable_deterministic_algorithms = kwargs.pop("enable_deterministic_algorithms", False)
        self.momentum = kwargs.pop("momentum", 0.0)
        enable_opt_rtn = kwargs.pop("enable_opt_rtn", None)
        self.quant_lm_head = kwargs.pop("quant_lm_head", False)

        self.ignore_layers = kwargs.pop("ignore_layers", "")

        self._offloader = OffloadManager(enabled=low_cpu_mem_usage, mode="offload", offload_dir_prefix="compressor")

        # Model related
        model_dtype = kwargs.pop("model_dtype", None)
        trust_remote_code = kwargs.pop("trust_remote_code") if "trust_remote_code" in kwargs else True

        self.scale_dtype = kwargs.pop("scale_dtype", None)

        self.static_attention_dtype = kwargs.pop("static_attention_dtype", None)
        # Attention static dtype
        if self.static_attention_dtype is not None:
            logger.warning("The static attention dtype is experimental and currently has limited support.")
        # KV cache, this one does not affect tuning but will collect some infos during tuning
        self.static_kv_dtype = kwargs.pop("static_kv_dtype", None)
        if self.static_kv_dtype is not None:
            logger.warning("The static kv is experimental and currently has limited support.")

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

        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("`device` is deprecated, please use `device_map` instead")

        # Tuning hyperparameters
        self.seed = seed
        set_seed(self.seed)
        self.enable_quanted_input = enable_quanted_input

        self.nsamples = nsamples
        self.seqlen = seqlen
        self.batch_size, self.gradient_accumulate_steps = batch_size, gradient_accumulate_steps
        self.nblocks = nblocks
        self.dataset = dataset
        self.iters = iters

        if self.iters == 0:
            self.lr = 5e-3

        # Automatically adjust the disable_opt_rtn option if the user does not explicitly set it.
        # To avoid None issue, we keep a copy though it's a little ugly
        if enable_opt_rtn and disable_opt_rtn:
            raise ValueError("`enable_opt_rtn` and `disable_opt_rtn` are mutually exclusive; " "only one can be set.")
        if enable_opt_rtn:
            disable_opt_rtn = False
        self.orig_disable_opt_rtn = disable_opt_rtn
        self.disable_opt_rtn = disable_opt_rtn
        self.enable_torch_compile = enable_torch_compile

        self.enable_alg_ext = enable_alg_ext
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap

        # Whether to pack the layer immediately after tuning
        self.is_immediate_packing = False
        self.is_immediate_saving = False

        # Some helpers
        self.batch_dim = None

        torch.set_printoptions(precision=3, sci_mode=True)

        if is_hpex_available():
            logger.info("habana_frameworks is available, import htcore explicitly.")
            import habana_frameworks.torch.core as htcore  # pylint: disable=E0401

        self.attention_mask = []

        self.wrapper_block = wrapper_block
        if self.enable_alg_ext:
            try:
                logger.warning_once("using algorithm extension for quantization.")
                from auto_round.alg_ext import wrapper_autoround

                wrapper_autoround(self)
            except (ImportError, ModuleNotFoundError):
                logger.error("algorithm extension import error, fallback to default mode")

        self.model_context = ModelContext.create_context(
            model,
            tokenizer=tokenizer,
            platform=platform,
            model_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            amp=amp,
            need_calib=self.need_calib,
            device=self.compress_context.device,
        )
        self.compress_context = CompressContext.create_context(
            low_cpu_mem_usage,
            low_gpu_mem_usage,
            device_map,
            enable_torch_compile,
        )

    # backward compatible with the legacy API
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

        for obj in ["quantization_args", "model_context"]:
            if obj not in self.__dict__:
                continue
            obj = object.__getattribute__(self, obj)
            try:
                return object.__getattribute__(obj, name)
            except AttributeError:
                continue

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _check_configs(self) -> None:
        """Checks if the configurations are valid.

        Raises:
        ValueError, TypeError: If any of the configurations are invalid.
        """
        self.quantization_args.check_config()
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
            self.quantization_args.is_act_quantize()
            and (
                not self.quantization_args.is_nv_fp(act=True) or "static_gs" not in self.quantization_args.act_data_type
            )
            and not self.quantization_args.is_mx_fp(act=True)
            and not self.quantization_args.is_dynamic_wint8aint8()
            and not self.quantization_args.is_static_wfp8afp8()
        ):
            logger.warning(
                "activation quantization is an experimental feature with limited support and a complex API. "
                "And please save the quantized model to fake format as real deployment is not supported currently"
            )

        if self.quantization_args.is_mx_fp() and self.group_size != 32:
            logger.warning("dtype mx_fp should only support group_size of 32 in real deployment")

        if self.quantization_args.is_nv_fp() and (self.group_size != 16):
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

    def _gen_auto_scheme(self) -> dict[str, dict]:
        if self.mllm:
            logger.info("AutoScheme is not yet supported for multimodal LLMs.")
            sys.exit(-1)

        if is_quantized_input_module(self.model_context.model):
            logger.info("AutoScheme does not currently support quantized input models (e.g., FP8).")
            sys.exit(-1)

        all_dtypes = []
        all_gguf = True
        for option in self.orig_scheme.options:
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
            self.model_context.model,
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

        if not self.enable_torch_compile and self.super_bits is None and not self.orig_scheme.low_gpu_mem_usage:
            logger.warning("we strongly recommend to set `enable_torch_compile` to True for AutoScheme to save VRAM")
        self.scheme_generator = GenScheme(
            self.orig_scheme,
            self.model_context.model,
            quant_layer_names,
            fixed_layer_scheme_new,
            self.dataset,
            device_map=self.compress_context.device_map,
            tokenizer=self.tokenizer,
            enable_torch_compile=self.enable_torch_compile,
        )
        layer_config = self.scheme_generator.get_layer_config()
        return layer_config

    def post_init(self):
        assert self.model_context._model_loaded, "should load model first"
        # should be set after loading model and set layer_config, cause some special scheme need these.
        # Preserve the original, unparsed scheme for later use in auto scheme generation
        # within `configure_layer_config` (which may need the raw value instead of `self.scheme`).
        default_scheme, self.is_auto_scheme, final_attrs = _parse_scheme(
            self.scheme, self.quantization_args.non_default()
        )
        # Bind attributes to self for easy instance-level access
        # for key, value in final_attrs.items():
        #     setattr(self, key, value)
        self.quantization_args = QuantizationArgs.from_dict(final_attrs)
        self._check_configs()

        self.orig_scheme = copy.deepcopy(self.scheme)
        self.scheme = default_scheme

        # check and update the format based on the current configuration
        if self.formats:
            self.formats = get_formats(self.formats, self)
        gguf_scheme_name = get_gguf_scheme(self.scheme)
        # GGUF uses fp32 scale dtype as default
        if self.scale_dtype is None:
            self.scale_dtype = "fp32" if gguf_scheme_name else "fp16"
        self.scale_dtype = convert_dtype_str2torch(self.scale_dtype)

        if not hasattr(self, "quant_block_list"):
            all_blocks = get_block_names(self.model_context.model)
            self.quant_block_list = find_matching_blocks(
                self.model_context.model, all_blocks, self.to_quant_block_names
            )

        # Set device, must place after model loading
        set_non_auto_device_map(self.model_context.model, self.compress_context.device_map)

        if self.iters != 0 and self.orig_disable_opt_rtn is not None:
            logger.warning("`disable_opt_rtn` only works when `iters` is set to 0, ignore it now.")
            self.disable_opt_rtn = True
        if (
            self.quantization_args.bits >= 8
            and self.quantization_args.act_bits >= 8
            and self.iters == 0
            and self.quantization_args.data_type == "int"
            and self.disable_opt_rtn is None
        ):
            logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
            self.disable_opt_rtn = True
        if self.disable_opt_rtn is None and self.iters == 0:
            logger.info(
                "`enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy."
            )
            self.disable_opt_rtn = False

        # after setting iters
        self._adjust_torch_compile(self.enable_torch_compile)

        self.block_forward = (
            compile_func(block_forward, self.compress_context.device) if self.enable_torch_compile else block_forward
        )

        if not self.is_auto_scheme:
            enable_gguf_official_mixed = True
        else:
            enable_gguf_official_mixed = False

        self.configure_layer_config(enable_gguf_official_mixed=enable_gguf_official_mixed)

        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reset()

        def _should_disable_inplace_due_to_layers_outside_block() -> bool:
            return self.has_qlayer_outside_block and self.need_calib

        # Disable inplace mode when there are quantized layers outside blocks
        # under specific iteration/optimization settings.
        if _should_disable_inplace_due_to_layers_outside_block():
            self.inplace = False
        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            # Determine if immediate packing is required
            self._adjust_immediate_packing_and_saving()

    def _adjust_torch_compile(self, enable_torch_compile: bool) -> None:
        """Sets the torch compile configuration for the tuning."""
        self.enable_torch_compile = enable_torch_compile
        if (
            not self.enable_torch_compile
            and TORCH_VERSION_AT_LEAST_2_6
            and self.quantization_args.act_bits > 8
            and not is_debug_mode()
            and "fp8" not in self.quantization_args.data_type
            and "fp8" not in self.quantization_args.act_data_type
            and self.iters > 0
        ):
            logger.info(
                "%s",
                "'enable_torch_compile' is set to `False` by default. "
                "Enabling it can reduce tuning cost by 20%, but it might throw an exception.",
            )
        # On HPU, we rely on torch.compile to speed up the model execution.
        if self.enable_torch_compile and self.quantization_args.is_wfp8afp8 and not is_hpex_available():
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as fp8 is enabled")
        # TODO: fix https://github.com/intel/auto-round/issues/1109
        if self.enable_torch_compile and self.quantization_args.is_nv_fp(act=True):
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as nvfp4 is enabled")

    def configure_layer_config(self, enable_gguf_official_mixed: None | bool = True):
        is_gguf_format = (f := getattr(self, "formats", None)) is not None and len(f) > 0 and f[0].is_gguf()
        if not is_gguf_format:
            predefined_ignore_layers = get_predefined_ignore_layers(self.model_context.model)
            if predefined_ignore_layers:
                logger.info(f"Using predefined ignore_layers: {predefined_ignore_layers}")
                tmp_str = ",".join(predefined_ignore_layers)
                if self.ignore_layers == "":
                    self.ignore_layers = tmp_str
                else:
                    self.ignore_layers += "," + tmp_str

        if self.is_auto_scheme:
            self.layer_config = self._gen_auto_scheme()
        else:
            self.layer_config = _handle_special_schemes(
                self.orig_scheme,
                self.layer_config,
                self.model_context.model,
                supported_types=SUPPORTED_LAYER_TYPES,
                inner_supported_types=INNER_SUPPORTED_LAYER_TYPES,
                quant_lm_head=self.quant_lm_head,
                mllm=self.model_context.is_mllm,
            )

        fill_default_value = True
        if self.is_auto_scheme:
            fill_default_value = False
        self.layer_config, self.has_qlayer_outside_block, self.regex_config = set_layer_config(
            self.model_context.model,
            self.layer_config,
            self.scheme,
            self.scale_dtype,
            SUPPORTED_LAYER_TYPES,
            INNER_SUPPORTED_LAYER_TYPES,
            self.quant_block_list,
            self.ignore_layers,
            self.quant_lm_head,
            enable_gguf_official_mixed=enable_gguf_official_mixed,
            is_mllm=self.model_context.is_mllm,
            fill_default_value=fill_default_value,
        )

    def _adjust_immediate_packing_and_saving(self):
        formats = getattr(self, "formats", [])
        if len(formats) == 1 and not formats[0].is_fake() and self.inplace:
            self.is_immediate_packing = True

        if self.has_qlayer_outside_block and self.iters != 0:
            self.is_immediate_packing = False

        if not ("causallm" in self.model_context.model.__class__.__name__.lower() and not self.model_context.is_mllm):
            # TODO For tied keys, there may some issues, we haven't not verified this
            tied_weight_keys = getattr(self.model_context.model, "_tied_weight_keys", {})
            if len(tied_weight_keys) > 1:
                self.is_immediate_saving = False
                if self.compress_context.low_cpu_mem_usage:
                    logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                return
            if len(tied_weight_keys) == 1:
                key = tied_weight_keys.keys[0]
                if "lm_head" not in key:
                    self.is_immediate_saving = False
                    if self.compress_context.low_cpu_mem_usage:
                        logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                    return

        if self.compress_context.low_cpu_mem_usage and self.is_immediate_packing:
            self.is_immediate_saving = True

        if self.compress_context.low_cpu_mem_usage and not self.is_immediate_packing:
            logger.info(
                "`low_cpu_mem_usage` is only supported when `immediate_packing` is True. "
                "Setting `low_cpu_mem_usage` to False."
            )
            self.compress_context.low_cpu_mem_usage = False
            self.is_immediate_saving = False

        if self.compress_context.low_cpu_mem_usage and self.is_immediate_packing:
            if formats[0].is_gguf():
                logger.warning(
                    "`low_cpu_mem_usage` is not fully supported for gguf format. "
                    "Setting `low_cpu_mem_usage` to False."
                )
                self.compress_context.low_cpu_mem_usage = False
                self.is_immediate_saving = False
            elif self.has_qlayer_outside_block and self.disable_opt_rtn and self.iters == 0:
                logger.info(
                    "Keeping `low_cpu_mem_usage` enabled in RTN mode (iters=0): "
                    "RTN path uses blockwise quantization and supports per-block offloading."
                )
            elif self.has_qlayer_outside_block and self.iters > 0:
                logger.warning(
                    "`low_cpu_mem_usage` is not fully supported "
                    "when there are quantized layers outside blocks and optimized RTN is disabled. "
                    "Setting low_cpu_mem_usage to False."
                )
                self.compress_context.low_cpu_mem_usage = False
                self.is_immediate_saving = False

        if self.is_immediate_saving and "int" not in self.data_type:
            logger.warning("immediate_saving is only supported for int quantization, set to False")
            self.is_immediate_saving = False

        if self.orig_output_dir is None:
            self.is_immediate_saving = False

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
        if is_quantized_input_module(self.model_context.model):
            layer_names = []
        if layer_names is None:
            layer_names = []
        if self.compress_context.low_gpu_mem_usage or (
            len(block_names) == 1
            and len(layer_names) == 0
            and not self.has_qlayer_outside_block
            and (last_cache_name is None or last_cache_name in block_names)
        ):
            # low_gpu_mem_usage or calibrate only the embedding layer, which is also very fast on CPU
            all_inputs = self.cache_inter_data(block_names, nsamples, layer_names=[], last_cache_name=last_cache_name)
        else:
            try:
                if any(p.device.type == "meta" for p in self.model_context.model.parameters()):
                    materialize_model_(self.model_context.model)

                if (
                    hasattr(self.model_context.model, "hf_device_map")
                    and len(self.model_context.model.hf_device_map) > 1
                ):
                    self.model_context.model = dispatch_model(
                        self.model_context.model, device_map=self.model_context.model.hf_device_map
                    )
                else:
                    # Change this if new device is supported
                    if str(self.model_context.model.device) == "cpu" and (
                        not self.compress_context.device.startswith("hpu")
                    ):
                        # type(self.model_context.model._no_split_modules) changes from list to set when transformers > 5.0
                        no_split_modules = list(getattr(self.model_context.model, "_no_split_modules", []))
                        devices = parse_available_devices(self.compress_context.device_map)

                        max_memory = get_max_memory()
                        new_max_memory = {}
                        if "cpu" not in devices:
                            devices.append("cpu")
                        for device in devices:
                            if ":" in device:
                                device = int(device.split(":")[-1])
                            elif device == "cpu":
                                device = "cpu"
                            elif isinstance(device, str):
                                device = 0
                            else:
                                raise ValueError(
                                    f"Unsupported device {device} in device_map: {self.compress_context.device_map}"
                                )
                            if device not in max_memory:
                                # Skip devices that aee not reported by accelerate's max_memory.
                                # This is expected when a device is unavailable or cannot provide memory info.
                                continue
                            # Use 90% of the reported max memory to leave headroom for activations,
                            # temporary tensors, other processes, and allocator fragmentation, reducing
                            # the chance of runtime OOM while still utilizing most available memory.
                            new_max_memory[device] = max_memory[device] * 0.9

                        # If non-CPU devices were requested but none survived, fall back to CPU caching
                        # via the OOM handler below, avoiding unnecessary dispatch overhead.
                        requested_non_cpu = any((d != "cpu") for d in devices)
                        has_non_cpu_memory = any((k != "cpu") for k in new_max_memory)
                        if requested_non_cpu and not has_non_cpu_memory:
                            raise torch.OutOfMemoryError(
                                "No non-CPU device available in accelerate's reported memory. "
                                "Falling back to CPU caching."
                            )

                        new_max_memory = get_balanced_memory(
                            self.model_context.model,
                            max_memory=new_max_memory,
                            no_split_module_classes=no_split_modules,
                        )
                        self.model_context.model.tie_weights()
                        device_map = infer_auto_device_map(
                            self.model_context.model,
                            max_memory=new_max_memory,
                            no_split_module_classes=no_split_modules,
                        )
                        if len(devices) > 1 and "cpu" in device_map.values():
                            logger.warning(
                                "Some layers are offloaded to cpu, which may severely impact calibration speed."
                                " Please consider using more cards."
                            )

                        try:

                            self.model_context.model = dispatch_model(self.model_context.model, device_map=device_map)
                        except ValueError as e:
                            if "offload_dir" in e.__str__():
                                logger.warning(
                                    f"Due to insufficient resources, disk is used to store the model."
                                    f" `offload_dir={envs.AR_WORK_SPACE}`"
                                )
                                self.model_context.model = dispatch_model(
                                    self.model_context.model, device_map=device_map, offload_dir=envs.AR_WORK_SPACE
                                )
                            else:
                                raise
                    else:

                        self.model_context.model = self.model_context.model.to(self.compress_context.device)

                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name
                )
                if (
                    hasattr(self.model_context.model, "hf_device_map")
                    and len(self.model_context.model.hf_device_map) > 1
                ):
                    accelerate.hooks.remove_hook_from_submodules(self.model_context.model)

            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.info("switch to cpu to cache block inputs")
                    self.compress_context.cache_device = torch.device("cpu")
                    if self.has_qlayer_outside_block or self.__class__.__name__ == "AutoRoundMLLM":
                        logger.warning(
                            "we recommend using more GPUs in calibration."
                            " Otherwise, some layers may fall back to `rtn` mode, which can affect accuracy."
                        )
                    accelerate.hooks.remove_hook_from_submodules(self.model_context.model)
                    self.model_context.model = mv_module_from_gpu(self.model_context.model)
                    clear_memory(device_list=self.compress_context.device_list)
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
        if (len(block_names) > 1 or len(layer_names) > 0) and self.compress_context.low_gpu_mem_usage:
            tmp_dtype = self.model_context.model.dtype
            if self.amp:
                if self.model_context.model.dtype != self.model_context.model.dtype:
                    self.model_context.model = self.model_context.model.to(torch.bfloat16)
            else:
                self.model_context.model = self.model_context.model.to(torch.float32)  ##model on cpu

        self.last_cache_name = _infer_last_cache_name(block_names, layer_names, last_cache_name)
        self._cache_target_set = set(self.to_cached_layers)
        self._cache_seen_targets = set()
        calib_bs = self.batch_size
        self.hook_handles = []
        self._replace_forward()
        self.calib(nsamples, calib_bs)
        self.model_context.recover_forward()
        res = self.inputs
        del self.last_cache_name
        del self._cache_target_set
        del self._cache_seen_targets
        del self.to_cached_layers
        if tmp_dtype is not None:
            self.model_context.model = self.model_context.model.to(tmp_dtype)

        return res

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
            logger.warning_once(
                f"An insufficient number of samples likely reduces the accuracy of the quantized model. "
                f"Target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )

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

            if self._should_stop_cache_forward(name):
                raise NotImplementedError

        return cache_input_hook

    def _replace_forward(self):
        """Replaces the forward function."""

        def register_hook(n, m, hook_handles):
            if n in self.to_cached_layers and type(m) not in SUPPORTED_LAYER_TYPES:  ##block
                m.orig_forward = m.forward
                m.forward = partial(self._get_block_forward_func(n), m)
            elif n in self.to_cached_layers:  ##linear layer or conv1d layer
                hook_func = self._get_cache_data_hook_for_layer(n)
                hook_handle = m.register_forward_hook(hook_func)
                hook_handles.append(hook_handle)

        self.model_context.replace_forward(register_hook)

    def _should_stop_cache_forward(self, name: str) -> bool:
        """Determine whether current forward pass can stop after caching `name`."""
        if name == self.last_cache_name:
            return True

        if self.last_cache_name is not None:
            return False

        if not hasattr(self, "_cache_target_set") or not hasattr(self, "_cache_seen_targets"):
            return False

        if name in self._cache_target_set:
            self._cache_seen_targets.add(name)

        if not self._cache_target_set.issubset(self._cache_seen_targets):
            return False

        # Lock the last cache name after the first full forward pass.
        self.last_cache_name = name
        return True

    def _preprocess_block_inputs(self, inputs, first_input_name="input_ids"):
        input_ids, input_others = self._split_inputs(inputs, first_input_name)
        clear_memory(device_list=self.compress_context.device_list)
        input_ids = to_device(input_ids, self.compress_context.cache_device)
        input_others = to_device(input_others, self.compress_context.cache_device)
        # As in calibration phase, we may use bf16 for calibration due to low_gpu_memory usage

        tmp_dtype = self.model_context._amp_dtype if self.model_context._amp else torch.float32
        input_ids = to_dtype(input_ids, tmp_dtype)

        for key in input_others.keys():
            if isinstance(input_others[key], torch.Tensor) and (
                input_others[key].dtype == torch.float16 or input_others[key].dtype == torch.bfloat16
            ):
                input_others[key] = input_others[key].to(tmp_dtype)
            elif isinstance(input_others[key], list):
                for i in range(len(input_others[key])):
                    to_dtype(input_others[key][i], tmp_dtype)
        return input_ids, input_others

    def _split_inputs(self, inputs: dict, first_input_name: str) -> tuple[torch.Tensor, dict]:
        input_ids = inputs[first_input_name]
        inputs.pop(first_input_name, None)
        input_others = inputs
        return input_ids, input_others

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
            dtype = module.weight.dtype
            # As typically float32 are used in RTN to search scale zp,
            # to avoid cache a bf16 copy we'd better use float32
            if config.get("super_group_size", None) is not None:
                dtype = torch.float32

            # Attempt quantization on GPU, fall back to CPU if OOM
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(dtype=dtype, device=self.compress_context.device),
                    **{
                        k: config.get(k, None)
                        for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                    },
                )
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config.get(k, None)
                            for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                        },
                    )
                except Exception as e:
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
            del weight
            del scale
            del zp
            clear_memory(device_list=self.compress_context.device_list)

        return is_quantized

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
        clear_memory(device_list=self.compress_context.device_list)
        for n, m in model.named_parameters():
            m.requires_grad_(False)

        input_ids, input_others = self._preprocess_block_inputs(inputs)

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

            if self.compress_context.low_cpu_mem_usage:
                if nblocks == 1:
                    self._offloader.reload(model, n)
                else:
                    self._offloader.reload(model, names)

            m.config = model.config if hasattr(model, "config") else None
            for alg in self.algorithms:
                q_input, input_ids = alg.quantize_block(
                    m,
                    input_ids,
                    input_others,
                    q_input=q_input,
                    device=device,
                )
            if hasattr(model, "config"):
                del m.config
            if self.is_immediate_packing:
                for n, tmp_m in m.named_modules():
                    if not (hasattr(tmp_m, "bits") and check_to_quantized(tmp_m)):
                        continue
                    self._immediate_pack(tmp_m.global_name)

            if self.is_immediate_saving:
                shard_writer(self, m, is_finalize=False)

            if self.compress_context.low_cpu_mem_usage and not self.is_immediate_saving:
                if nblocks == 1:
                    self._offloader.offload(model, n, overwrite=True)
                else:
                    for name in names:
                        self._offloader.offload(model, name, overwrite=True)
        if pbar is not None:
            pbar.update(1)

        if not self.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        for n, m in self.model.named_modules():
            if hasattr(m, "name"):
                delattr(m, "name")

        del q_input
        del input_ids
        del input_others
        del inputs

        clear_memory(device_list=self.compress_context.device_list)

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        self.model_context._load_model()
        self.post_init()
        self.model_context.initialize(formats=self.formats)

        self._check_compatibility()

        if bool(self.quant_block_list):
            all_blocks = self.quant_block_list
        else:
            all_blocks = get_block_names(self.model_context.model)

        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model_context.model, self.layer_config

        layer_names = _get_quantized_layer_names_outside_blocks(
            model=self.model_context.model,
            layer_config=self.layer_config,
            supported_types=SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quant_block_list,
        )
        start_time = time.time()
        all_first_block_names = [block[0] for block in all_blocks]
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")
        all_inputs = self.try_cache_inter_data_gpucpu(
            all_first_block_names,
            self.nsamples,
            layer_names,
        )
        self.inputs = all_inputs
        is_quantized_embedding = self._quantize_embedding_layer()
        clear_memory(device_list=self.compress_context.device_list)
        all_q_inputs = None
        if is_quantized_embedding:
            all_inputs = copy.deepcopy(self.inputs)
            clear_memory(self.inputs, device_list=self.compress_context.device_list)
            all_q_inputs = self.try_cache_inter_data_gpucpu(all_first_block_names, self.nsamples, layer_names)
            self.inputs = all_q_inputs
        # Remove accelerate dispatch hooks before moving parameters.
        # hf_device_map is kept for reference but hooks are no longer needed.
        if hasattr(self.model_context.model, "hf_device_map") and len(self.model_context.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model_context.model)
        self.model_context.model = mv_module_from_gpu(self.model_context.model)
        clear_memory(device_list=self.compress_context.device_list)
        logger.info("caching done")
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.offload(
                self.model_context.model, all_blocks, clear_memory=True, device_list=self.compress_context.device_list
            )
        if len(all_blocks) > 1:
            pbar = tqdm(range(0, sum([len(i) for i in all_blocks]), self.nblocks))
        else:
            pbar = tqdm(range(0, len(all_blocks[0]), self.nblocks))  # move the alg warning outside pbar

        for block_names in all_blocks:
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            q_inputs = None
            if all_q_inputs is not None:
                q_inputs = all_q_inputs[block_names[0]]
                all_q_inputs.pop(block_names[0])

            inputs, q_inputs = _update_inputs(inputs, q_inputs)

            clear_memory(self.inputs, device_list=self.compress_context.device_list)

            if "input_ids" in inputs.keys():
                total_samples = len(inputs["input_ids"])
                if total_samples < self.batch_size:
                    self.batch_size = total_samples
                    logger.warning(f"force the train batch size to {total_samples}")

            self._quantize_blocks(
                self.model_context.model,
                inputs,
                block_names,
                q_input=q_inputs if q_inputs is not None else None,
                nblocks=self.nblocks,
                device=self.compress_context.device,
                pbar=pbar,
            )
            if self.is_immediate_packing and len(self.formats) != 1:
                raise ValueError(
                    f"Expected exactly one packing format when 'immediate_packing' is True, "
                    f"but got {len(self.formats)} formats."
                )
        pbar.set_description("Quantizing done")
        pbar.close()
        self._quantize_layers(layer_names, all_inputs)

        convert_module_to_hp_if_necessary(
            self.model_context.model, self.amp_dtype, self.compress_context.device, to_cpu=True
        )
        if self.is_immediate_saving:
            shard_writer(self, is_finalize=True)

        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reload(self.model_context.model)

        end_time = time.time()
        cost_time = end_time - start_time
        logger.info(f"quantization tuning time {cost_time}")

        # Dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model_context.model.named_modules():
            if isinstance(m, tuple(SUPPORTED_LAYER_TYPES)):
                if check_to_quantized(m):
                    quantized_layers.append(n)
                else:
                    unquantized_layers.append(n)
            elif hasattr(m, "scales") or hasattr(m, "scale"):  # packing_immediately
                quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            summary_info += f",  {unquantized_layers} have not been quantized"
        logger.info(summary_info)

        self.model_context.quantized = True
        return self.model_context.model, self.layer_config

    def _check_compatibility(self) -> None:
        """Checks compatibility of the configurations and model."""
        if (
            self.seqlen is not None
            and hasattr(self.model_context.model, "config")
            and hasattr(self.model_context.model.config, "max_position_embeddings")
        ):
            if self.model_context.model.config.max_position_embeddings < self.seqlen:
                logger.warning(
                    f"Change sequence length to {self.model_context.model.config.max_position_embeddings} "
                    "due to the limitation of max_position_embeddings"
                )
                self.seqlen = min(self.seqlen, self.model_context.model.config.max_position_embeddings)

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

    def save_quantized(
        self,
        output_dir: str = None,
        format: Union[str, list[OutputFormat]] = None,
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
        if format is not None:
            logger.warning(
                f"save_quantized with format is deprecated and will be deleted in auto_round version 1.0."
                f" Please use Compressor(format='{format}' instead)."
            )
            if isinstance(format, str) and getattr(self, "formats", None) is None:
                formats = get_formats(format, self)
                if not hasattr(self, "formats"):
                    self.formats = formats

        if not self.quantized:
            logger.warning("please run autoround.quantize first")
            return
        folders = []
        for format in self.formats:
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
                model=self.model_context.model,
                layer_config=self.layer_config,
                inplace=inplace,
                tokenizer=self.tokenizer,
                device=self.compress_context.device,
                serialization_dict=serialization_dict,
                **kwargs,
            )
            folders.append(save_folder)

        if return_folders:
            return compressed_model, folders
        else:
            return compressed_model

    def quantize_and_save(
        self, output_dir: str = "tmp_autoround", format: str = None, inplace: bool = True, **kwargs
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
        if format and self.formats is not None:
            logger.warning(
                f"quantize_and_save with format is deprecated and will be deleted in auto_round version 1.0."
                f" Please use Compressor(format='{format}' instead)."
            )
            format_list = get_formats(format, self)
            self.formats = format_list
        if self.formats is None:
            logger.info("format is not set, using default auto_round format.")
            self.formats = get_formats("auto_round", self)

        # If multiple formats are specified, enforce inplace=False
        if len(self.formats) > 1:
            inplace = False
        self.inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Perform model quantization
        if self.static_attention_dtype is not None:
            from auto_round.experimental.attention import attention_quant_ctx

            with attention_quant_ctx(self.model_context.model, static_attention_dtype=self.static_attention_dtype):
                model, _ = self.quantize()
        elif self.static_kv_dtype is not None:
            from auto_round.experimental.kv_cache import kvcache_quant_context

            with kvcache_quant_context(self.model_context.model, static_kv_dtype=self.static_kv_dtype):
                model, _ = self.quantize()
        else:
            model, _ = self.quantize()
        # Save the quantized model in the specified format_list
        model, folders = self.save_quantized(output_dir, inplace=inplace, return_folders=True, **kwargs)
        memory_monitor.log_summary()

        return model, folders


class TuneCompressor(BaseCompressor):
    need_calib: bool = True


class ZeroShotCompressor(BaseCompressor):
    need_calib: bool = False

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """

        pass
