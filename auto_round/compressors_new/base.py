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
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import torch
from transformers import set_seed

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization import BaseQuantizers, QuantizationConfig
from auto_round.compressors_new.utils import block_forward
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.formats import OutputFormat, get_formats
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    compile_func,
    is_debug_mode,
    is_hpex_available,
)
from auto_round.utils.device import set_non_auto_device_map
from auto_round.utils.offload import OffloadManager
from auto_round.wrapper import wrapper_block


@dataclass
class SerializedCompressorConfig:
    bits: Optional[int] = None
    act_bits: Optional[int] = None
    data_type: Optional[str] = None
    act_data_type: Optional[str] = None
    group_size: Optional[int] = None
    act_group_size: Optional[int] = None
    sym: Optional[bool] = None
    act_sym: Optional[bool] = None
    act_dynamic: Optional[bool] = None
    amp: Optional[bool] = None
    batch_size: Optional[int] = None
    enable_minmax_tuning: Optional[bool] = True
    enable_norm_bias_tuning: Optional[bool] = False
    enable_quanted_input: Optional[bool] = True
    gradient_accumulate_steps: Optional[int] = None
    iters: Optional[int] = None
    lr: Optional[float] = None
    low_gpu_mem_usage: Optional[bool] = None
    minmax_lr: Optional[float] = None
    nsamples: Optional[int] = None
    quant_block_list: Optional[list[str]] = None
    regex_config: Optional[dict[str, Any]] = None
    scale_dtype: Optional[str] = None
    seqlen: Optional[int] = None
    supported_types: Optional[list[str]] = SUPPORTED_LAYER_TYPES
    static_attention_dtype: Optional[str] = None
    static_kv_dtype: Optional[str] = None
    super_bits: Optional[int] = None
    super_group_size: Optional[int] = None
    to_quant_block_names: Optional[list[str]] = None


class BaseCompressor(object):
    need_calib: bool = True

    def __init__(
        self,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ):
        self.quantize_config = None
        self.config_list = config if isinstance(config, list) else [config]
        for config in self.config_list:
            if isinstance(config, QuantizationConfig):
                self.quantize_config = config
        assert self.quantize_config is not None, "QuantizationConfig is required for Compressor"
        self.config_list.remove(self.quantize_config)

        # TODO: refactor calibration
        self.calibration = None

        self.formats = format

        # Extra/legacy kwargs for backward compatibility
        # Major version releases may pack them with extra configuration options
        amp = kwargs.pop("amp", True)
        nblocks = kwargs.pop("nblocks", 1)
        disable_deterministic_algorithms = kwargs.pop("disable_deterministic_algorithms", True)
        enable_deterministic_algorithms = kwargs.pop("enable_deterministic_algorithms", False)

        self._offloader = OffloadManager(enabled=low_cpu_mem_usage, mode="offload", offload_dir_prefix="compressor")

        # Model related
        model_dtype = kwargs.pop("model_dtype", None)
        trust_remote_code = kwargs.pop("trust_remote_code") if "trust_remote_code" in kwargs else True

        self.static_attention_dtype = kwargs.pop("static_attention_dtype", None)
        # Attention static dtype
        if self.static_attention_dtype is not None:
            logger.warning("The static attention dtype is experimental and currently has limited support.")
        # KV cache, this one does not affect tuning but will collect some infos during tuning
        self.static_kv_dtype = kwargs.pop("static_kv_dtype", None)
        if self.static_kv_dtype is not None:
            logger.warning("The static kv is experimental and currently has limited support.")

        if kwargs:
            logger.warning(
                f"unrecognized keys {list(kwargs.keys())} were passed. Please check them. If you use old api, just ignore this warning. "
            )
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

        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("`device` is deprecated, please use `device_map` instead")

        # Tuning hyperparameters
        self.seed = seed
        set_seed(self.seed)

        self.nblocks = nblocks

        self.enable_torch_compile = enable_torch_compile

        # Whether to pack the layer immediately after tuning
        self.is_immediate_packing = False
        self.is_immediate_saving = False

        torch.set_printoptions(precision=3, sci_mode=True)

        if is_hpex_available():
            logger.info("habana_frameworks is available, import htcore explicitly.")
            import habana_frameworks.torch.core as htcore  # pylint: disable=E0401

        # Alternatively, you can use CompressContext.create_context
        self.compress_context = CompressContext(
            low_cpu_mem_usage,
            low_gpu_mem_usage,
            device_map,
            enable_torch_compile,
            is_immediate_packing=self.is_immediate_packing,
            is_immediate_saving=self.is_immediate_saving,
            formats=self.formats,
        )
        self.model_context = ModelContext(
            model,
            tokenizer=tokenizer,
            platform=platform,
            model_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            amp=amp,
            need_calib=self.need_calib,
            device=self.compress_context.device,
        )

    def _adjust_torch_compile(self, enable_torch_compile: bool) -> None:
        """Sets the torch compile configuration for the tuning."""
        self.enable_torch_compile = enable_torch_compile
        if (
            not self.enable_torch_compile
            and TORCH_VERSION_AT_LEAST_2_6
            and self.quantize_config.act_bits > 8
            and not is_debug_mode()
            and "fp8" not in self.quantize_config.data_type
            and "fp8" not in self.quantize_config.act_data_type
            and self.need_calib
        ):
            logger.info(
                "%s",
                "'enable_torch_compile' is set to `False` by default. "
                "Enabling it can reduce tuning cost by 20%, but it might throw an exception.",
            )
        # On HPU, we rely on torch.compile to speed up the model execution.
        if self.enable_torch_compile and self.quantize_config.is_wfp8afp8 and not is_hpex_available():
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as fp8 is enabled")
        # TODO: fix https://github.com/intel/auto-round/issues/1109
        if self.enable_torch_compile and self.quantize_config.is_act_nv_fp:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as nvfp4 is enabled")

    def post_init(self):
        self.model_context._load_model()
        assert self.model_context._model_loaded, "should load model first"

        self.quantizer = BaseQuantizers.from_config(self.quantize_config)
        self.quantizer.post_init()
        self.wrapper_block = wrapper_block

        # TODO: add other algs here when they are ready
        # self.other_alg = OtherAlg.from_config(self.other_alg_config) if self.other_alg_config is not None else None
        # self.other_alg.post_init() if self.other_alg is not None else None

        # check and update the format based on the current configuration
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
        self.compress_context.formats = self.formats

        # Set device, must place after model loading
        set_non_auto_device_map(self.model_context.model, self.compress_context.device_map)

        # after setting iters
        self._adjust_torch_compile(self.enable_torch_compile)
        self.compress_context.enable_torch_compile = self.enable_torch_compile

        self.block_forward = (
            compile_func(block_forward, self.compress_context.device) if self.enable_torch_compile else block_forward
        )

        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reset()

        def _should_disable_inplace_due_to_layers_outside_block() -> bool:
            return self.quantizer.has_qlayer_outside_block and self.need_calib

        # Disable inplace mode when there are quantized layers outside blocks
        # under specific iteration/optimization settings.
        if _should_disable_inplace_due_to_layers_outside_block():
            self.inplace = False
        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            # Determine if immediate packing is required
            self._adjust_immediate_packing_and_saving()

    # backward compatible with the legacy API
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

        for obj in ["quantize_config", "model_context", "compress_context", "quantizer"]:
            if obj not in self.__dict__:
                continue
            obj = object.__getattribute__(self, obj)
            try:
                return object.__getattribute__(obj, name)
            except AttributeError:
                continue

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _adjust_immediate_packing_and_saving(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        formats = getattr(self, "formats", [])
        if len(formats) == 1 and not formats[0].is_fake() and self.inplace:
            self.is_immediate_packing = True

        if self.quantizer.has_qlayer_outside_block and self.need_calib:
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
            elif (
                self.has_qlayer_outside_block
                and getattr(self, "disable_opt_rtn", None)
                and isinstance(self.quantize_config, RTNConfig)
            ):
                logger.info(
                    "Keeping `low_cpu_mem_usage` enabled in RTN mode (iters=0): "
                    "RTN path uses blockwise quantization and supports per-block offloading."
                )
            elif self.quantizer.has_qlayer_outside_block and not isinstance(self.quantize_config, RTNConfig):
                logger.warning(
                    "`low_cpu_mem_usage` is not fully supported "
                    "when there are quantized layers outside blocks and optimized RTN is disabled. "
                    "Setting low_cpu_mem_usage to False."
                )
                self.compress_context.low_cpu_mem_usage = False
                self.is_immediate_saving = False

        if self.is_immediate_saving and "int" not in self.quantize_config.data_type:
            logger.warning("immediate_saving is only supported for int quantization, set to False")
            self.is_immediate_saving = False

        if self.output_dir is None:
            self.is_immediate_saving = False

        self.compress_context.is_immediate_packing = self.is_immediate_packing
        self.compress_context.is_immediate_saving = self.is_immediate_saving

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        raise NotImplementedError("quantize method must be implemented in subclass")

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
            return os.path.join(self.output_dir, sanitized_format)

        return self.output_dir

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
        self.output_dir = output_dir
        if format is not None:
            logger.warning(
                f"save_quantized with format is deprecated and will be deleted in auto_round version 1.0."
                f" Please use Compressor(format='{format}' instead)."
            )
            if isinstance(format, str) and getattr(self, "formats", None) is None:
                formats = get_formats(format, self)
                if not hasattr(self, "formats"):
                    self.formats = formats

        if not self.model_context.quantized:
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

            serialization_dict = asdict(SerializedCompressorConfig())
            for key in serialization_dict:
                serialization_dict[key] = getattr(self, key, serialization_dict[key])
            from auto_round.version import __version__

            serialization_dict["autoround_version"] = __version__
            if "scale_dtype" in serialization_dict.keys():
                serialization_dict["scale_dtype"] = str(serialization_dict["scale_dtype"])

            compressed_model = format.save_quantized(
                save_folder,
                model=self.model_context.model,
                layer_config=self.quantizer.layer_config,
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
        self.output_dir = output_dir
        self.compress_context.output_dir = output_dir

        # check and update the format based on the current configuration
        if format and self.formats is not None:
            logger.warning(
                f"quantize_and_save with format is deprecated and will be deleted in auto_round version 1.0."
                f" Please use Compressor(format='{format}' instead)."
            )
            self.formats = format
        if self.formats is None:
            if self.formats is None:
                logger.info("format is not set, using default auto_round format.")
            self.formats = "auto_round"

        # If multiple formats are specified, enforce inplace=False
        if len(self.formats.split(",")) > 1:
            inplace = False
        self.inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Perform model quantization
        if self.static_attention_dtype is not None:
            from auto_round.experimental.attention import attention_quant_ctx

            with attention_quant_ctx(self.model_context.model, static_attention_dtype=self.static_attention_dtype):
                self.quantize()
        elif self.static_kv_dtype is not None:
            from auto_round.experimental.kv_cache import kvcache_quant_context

            with kvcache_quant_context(self.model_context.model, static_kv_dtype=self.static_kv_dtype):
                self.quantize()
        else:
            self.quantize()

        # When immediate_saving is enabled, the model has already been saved during quantization
        # Skip the save_quantized call to avoid attempting to save layers that are on meta device
        if self.is_immediate_saving:
            logger.info("immediate_saving is enabled, model already saved during quantization")
            return self.model, [output_dir]
