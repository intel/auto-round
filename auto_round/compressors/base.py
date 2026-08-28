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
import gc
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Optional, Union

import torch
from transformers import AutoConfig, set_seed

from auto_round.algorithms.quantization import BaseQuantizer, QuantizationConfig
from auto_round.algorithms.transforms import (
    BaseRotationConfig,
    apply_rotation,
)
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors.config_resolution import (
    FormatResolution,
    ResolvedScheme,
    resolve_quantization_config,
    thaw_mapping,
)
from auto_round.compressors.layer_config_resolver import (
    apply_plan_to_model,
    extract_regex_config,
    has_quantized_layer_outside_blocks,
    resolve_layer_config,
)
from auto_round.compressors.shard_writer import ShardWriter
from auto_round.compressors.utils import _get_save_folder_name, is_mx_fp, is_nv_fp
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.export.formats import OutputFormat, resolve_formats
from auto_round.logger import logger
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    get_gguf_scheme,
    parse_scheme,
    preset_name_to_scheme,
    scheme_to_preset_name,
)
from auto_round.special_model_handler import get_predefined_fixed_attr, get_predefined_ignore_layers, update_module
from auto_round.utils import (
    AUDIO_MM_KEYS,
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    VISION_MM_KEYS,
    compress_layer_names,
    convert_dtype_str2torch,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    get_reverse_checkpoint_conversion_mapping,
    is_debug_mode,
    is_hpex_available,
    is_quantized_input_module,
    memory_monitor,
    preserve_original_visual_block_name,
    revert_checkpoint_conversion_mapping,
)
from auto_round.utils.device import (
    _force_trim_malloc,
    patch_xpu_sdpa_drop_causal_mask,
    set_non_auto_device_map,
)
from auto_round.utils.device_manager import default_enable_torch_compile, device_manager
from auto_round.utils.offload import OffloadManager

# ``torch.compile`` only pays for itself when the compiled quant function is
# replayed many times.  Below this many SignRound iterations the one-off
# compilation cost dominates, so compiling is pure overhead.
MIN_ITERS_FOR_TORCH_COMPILE = 10


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
    static_attention_granularity: Optional[str] = "tensor"
    static_kv_granularity: Optional[str] = "tensor"
    super_bits: Optional[int] = None
    super_group_size: Optional[int] = None
    to_quant_block_names: Optional[list[str]] = None
    rotation_configs: Optional[list[dict[str, Any]]] = None


SERIALIZATION_KEYS = tuple(field.name for field in fields(SerializedCompressorConfig))


def collect_user_scheme_overrides(configs: list[Any]) -> dict[str, Any]:
    scheme_fields = {f.name for f in fields(QuantizationScheme)}
    user_scheme_overrides = {}
    user_scheme_sources = {}
    for config in configs:
        for key in getattr(config, "_user_set_scheme_fields", set()):
            if key not in scheme_fields:
                continue
            value = getattr(config, key, None)
            if value is None:
                continue
            if key in user_scheme_overrides and value != user_scheme_overrides[key]:
                prev_config, prev_value = user_scheme_sources[key]
                raise ValueError(
                    f"Conflicting shared scheme field {key!r}: "
                    f"{type(prev_config).__name__}.{key}={prev_value!r}, "
                    f"{type(config).__name__}.{key}={value!r}. "
                    "Use the same value for shared fields or pass scheme arguments through Compressor."
                )
            user_scheme_overrides[key] = value
            user_scheme_sources[key] = (config, value)
    return user_scheme_overrides


def _make_compressor_scheme_property(name):
    def getter(self):
        scheme_context = getattr(self, "scheme_context", None)
        if scheme_context is not None:
            return getattr(scheme_context, name)
        return self.__dict__.get(name, getattr(type(self), name, None))

    def setter(self, value):
        scheme_context = getattr(self, "scheme_context", None)
        if scheme_context is not None:
            setattr(scheme_context, name, value)
        else:
            self.__dict__[name] = value

    return property(getter, setter)


class BaseOrchestrator(object):
    need_calib: bool = True
    compress_context: CompressContext = None
    model_context: ModelContext = None
    shard_writer: ShardWriter = None
    supported_types = SUPPORTED_LAYER_TYPES
    inner_supported_types = INNER_SUPPORTED_LAYER_TYPES

    # ── Scheme state (populated during resolve_scheme / _scheme_post_init) ──
    is_auto_scheme: bool = False
    orig_scheme = None
    scheme = None
    to_quant_block_names = None
    ignore_layers: str = ""
    quant_lm_head: bool = False
    _scheme_resolved: bool = False
    scheme_generator = None
    _scheme_context_fields = set(QuantizationScheme.get_attributes())
    for _scheme_field in QuantizationScheme.get_attributes():
        locals()[_scheme_field] = _make_compressor_scheme_property(_scheme_field)

    @staticmethod
    def _preload_model_config(model: Union[torch.nn.Module, str], trust_remote_code: bool) -> Optional[AutoConfig]:
        if not isinstance(model, str):
            return None

        try:
            return AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        except (OSError, EnvironmentError, ValueError) as e:
            logger.debug(
                "Failed to load config via AutoConfig.from_pretrained for %s: %s. "
                "Proceeding without config-based checks.",
                model,
                e,
            )
            return None

    def __init__(
        self,
        config: Union[object, list[object]],
        model: Union[torch.nn.Module, str],
        tokenizer: Any = None,
        platform: str = "hf",
        format: Union[str, list, None] = None,
        scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16",
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: Optional[bool] = None,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config: Optional[dict] = None,
        nsamples: int = None,
        seqlen: int = None,
        scale_dtype: Optional[Union[str, torch.dtype]] = None,
        ignore_layers: str = "",
        quant_lm_head: bool = False,
        to_quant_block_names: Optional[Union[str, list[str]]] = None,
        dataset: Optional[Union[str, list, tuple, torch.utils.data.DataLoader]] = None,
        **kwargs,
    ) -> None:
        # ``CalibrationContext`` is the single source of truth for calibration
        # runtime state.  Seed every calibration field here in one block so
        # the rest of ``__init__`` only ever interacts with the state object
        # via property forwarders.  ``_resolve_scheme`` later wires this same
        # instance onto the quantizer so the two share state.
        from auto_round.calibration.state import CalibrationContext

        dataset_was_explicitly_set = dataset is not None
        self.dataset = dataset if dataset_was_explicitly_set else "NeelNanda/pile-10k"
        batch_size = min(kwargs.pop("batch_size", 8), nsamples)
        self.calibration_context = CalibrationContext(
            nsamples=nsamples if nsamples is not None else 128,
            seqlen=seqlen if seqlen is not None else 2048,
            batch_size=batch_size,
            orig_batch_size=batch_size,
            dataset=self.dataset,
        )

        self.quantize_config = None
        self.rotation_configs: list[BaseRotationConfig] = []
        _config_list = config if isinstance(config, list) else [config]
        # Keep full list for pipeline construction (includes preprocessor configs).
        self._alg_configs: list = list(_config_list)
        from auto_round.algorithms.config_resolver import split_quantization_configs

        _preprocessor_configs, _block_quantizer_configs = split_quantization_configs(self._alg_configs)
        if len(_block_quantizer_configs) > 1:
            raise ValueError(
                f"Only one block-quantization config is allowed, but got {len(_block_quantizer_configs)}: "
                f"{[type(c).__name__ for c in _block_quantizer_configs]}"
            )
        if _block_quantizer_configs:
            self.quantize_config = _block_quantizer_configs[0]
        elif _preprocessor_configs:
            from auto_round.algorithms.quantization.rtn.config import RTNConfig as _RTNConfig

            self.quantize_config = _RTNConfig()
            self._alg_configs.append(self.quantize_config)
        for _cfg in self._alg_configs:
            if isinstance(_cfg, BaseRotationConfig):
                if hasattr(_cfg, "block_size") and _cfg.block_size is None:
                    if "group_size" in kwargs:
                        block_size = kwargs["group_size"]
                    else:
                        block_size = parse_scheme(scheme, {})[2]["group_size"]
                    _cfg.block_size = block_size  # TODO not robust
                self.rotation_configs.append(_cfg)
        assert self.quantize_config is not None, "QuantizationConfig is required for Compressor"

        # Compressor-level layer params (do not live in QuantizationConfig).
        # Calibration params (nsamples/seqlen/batch_size) are owned by
        # ``self.calibration_context`` (seeded above) and exposed via
        # ``@property`` forwarders.
        self.layer_config = layer_config
        self.scale_dtype = scale_dtype
        self.ignore_layers = ignore_layers
        self.quant_lm_head = quant_lm_head
        self.to_quant_block_names = to_quant_block_names
        # ``post_init()`` may run before ``quantize_and_save()`` in tests and
        # compatibility paths, so seed the same default used by
        # ``quantize_and_save(..., inplace=True)`` here.
        self.inplace = True

        # Scheme is passed directly to the compressor, not stored in QuantizationConfig.
        self.scheme = scheme
        self.scheme_context = None

        # Calibrator strategy (auto_round.calibration.base.Calibrator).  Constructed
        # lazily by ``Compressor.post_init`` based on ``_get_calibrator_kind()``;
        # remains ``None`` when calibration data is not needed (RTN zero-shot path).
        self.calibration = None

        self.formats = format

        # Extra/legacy kwargs for backward compatibility
        # Major version releases may pack them with extra configuration options
        kwargs.pop("iters", None)
        kwargs.pop("enable_alg_ext", None)
        kwargs.pop("vlm", None)
        amp = kwargs.pop("amp", True)
        nblocks = kwargs.pop("nblocks", 1)
        disable_deterministic_algorithms = kwargs.pop("disable_deterministic_algorithms", True)
        enable_deterministic_algorithms = kwargs.pop("enable_deterministic_algorithms", False)

        self._offloader = OffloadManager(enabled=low_cpu_mem_usage, mode="offload", offload_dir_prefix="compressor")

        # Model related
        model_dtype = kwargs.pop("model_dtype", None)
        trust_remote_code = kwargs.pop("trust_remote_code") if "trust_remote_code" in kwargs else True
        quant_nontext_module = kwargs.pop("quant_nontext_module", False)
        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("`device` is deprecated, please use `device_map` instead")

        from auto_round.experimental.utils import normalize_fp8_granularity

        self.static_attention_dtype = kwargs.pop("static_attention_dtype", None)
        self.static_attention_granularity = normalize_fp8_granularity(
            kwargs.pop("static_attention_granularity", "tensor")
        )
        # Attention static dtype
        if self.static_attention_dtype is not None:
            logger.warning("The static attention dtype is experimental and currently has limited support.")
        # KV cache, this one does not affect tuning but will collect some infos during tuning
        self.static_kv_dtype = kwargs.pop("static_kv_dtype", None)
        self.static_kv_granularity = normalize_fp8_granularity(kwargs.pop("static_kv_granularity", "tensor"))
        if self.static_kv_dtype is not None:
            logger.warning("The static kv is experimental and currently has limited support.")

        if kwargs:
            logger.warning_once(
                f"unrecognized keys {list(kwargs.keys())} were passed. "
                "Please check them. If you use old api, just ignore this warning."
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

        # XPU SDPA workaround: drop pure causal masks so FLASH backend is used,
        # and set torch.use_deterministic_algorithms(False)
        # instead of MATH (avoids ~10x peak-VRAM blow-up during block tuning).
        patch_xpu_sdpa_drop_causal_mask()

        # Tuning hyperparameters
        self.seed = seed
        set_seed(self.seed)

        self.nblocks = nblocks

        # ``None`` means "not set by the user", which is the only case where the
        # algorithm-driven auto-disabling below may override the value.
        self._torch_compile_user_specified = enable_torch_compile is not None
        # Fallback explanation used by ``_log_torch_compile_state``.
        self._torch_compile_default_off_reason = None if enable_torch_compile is None else "the user disabled it"
        if enable_torch_compile is None:
            enable_torch_compile = default_enable_torch_compile(self.device, platform_name=sys.platform)
            if not enable_torch_compile:
                self._torch_compile_default_off_reason = "it is off by default on Windows"
                logger.warning_once(
                    "`torch.compile` is disabled by default on Windows because TorchInductor requires the MSVC "
                    "`cl.exe` compiler, which may not be available. Pass `enable_torch_compile=True` or use "
                    "`--enable_torch_compile` to force enable it."
                )
        elif enable_torch_compile and sys.platform == "win32":
            logger.warning_once(
                "Forcing `torch.compile` on Windows. TorchInductor may fail if the MSVC `cl.exe` compiler "
                "is not installed or not available on PATH."
            )
        self.enable_torch_compile = enable_torch_compile

        # Whether to pack the layer immediately after tuning
        # Managed via self.compress_context.is_immediate_packing / is_immediate_saving

        torch.set_printoptions(precision=3, sci_mode=True)

        if is_hpex_available():
            logger.info("habana_frameworks is available, import htcore explicitly.")
            import habana_frameworks.torch.core as htcore  # pylint: disable=E0401

        # Reset both context singletons before creating fresh instances so that
        # consecutive AutoRound creations don't inherit stale config from earlier ones.
        CompressContext.reset_context()
        ModelContext.reset_context()

        # Resolve the device eagerly so ModelContext can be created before
        # CompressContext.  Creating ModelContext first places the large model
        # allocation early in the heap, matching the OLD arch allocation order
        # and reducing C-heap fragmentation (which is amplified on HPU).
        #
        # The process-wide DeviceManager singleton is the single source of truth
        # for the active device / device_list: configure it from ``device_map``
        # up front so both ModelContext and CompressContext (and any OOM fallback)
        # read the same value instead of keeping private copies.
        device_manager.configure(device_map if device_map is not None else 0)
        model_config = self._preload_model_config(model, trust_remote_code)

        self.model_context = ModelContext(
            model,
            tokenizer=tokenizer,
            platform=platform,
            model_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            config=model_config,
            amp=amp,
            need_calib=self.need_calib,
            formats=self.formats,
            is_act_quantize=self.quantize_config.is_act_quantize,
            quant_nontext_module=quant_nontext_module,
        )
        # Reset the singleton so each new orchestrator gets a fresh CompressContext.
        # CompressContext uses AutoSkipInitMeta (singleton), so without a reset the
        # second AutoRound(...) call reuses the previous instance and silently keeps
        # stale values (e.g. low_cpu_mem_usage=True from a prior run).
        CompressContext.reset_context()
        # When the model was built as a meta skeleton (AR_DISK_STREAM_MODEL=1),
        # give the offloader the original checkpoint path so it can materialize
        # each block on first touch directly from disk instead of assuming
        # blocks already hold real weights (see OffloadManager._reload).
        if self.model_context.disk_stream_model_dir is not None:
            self._offloader.model_dir = self.model_context.disk_stream_model_dir
        # Alternatively, you can use CompressContext.create_context
        self.compress_context = CompressContext(
            low_cpu_mem_usage,
            low_gpu_mem_usage,
            enable_torch_compile,
            formats=self.formats,
            static_kv_dtype=self.static_kv_dtype,
            static_attention_dtype=self.static_attention_dtype,
            static_kv_granularity=self.static_kv_granularity,
            static_attention_granularity=self.static_attention_granularity,
        )
        self.shard_writer = None
        # Resumability state deferred from Orchestrator._quantize_data_driven() until
        # quantize_and_save()'s save_quantized() call actually succeeds; see the
        # comment in quantize() near "is_immediate_saving" for why clearing is
        # deferred.
        self._resume_states = None

        # Flag for post_init idempotency.  Set to False here so post_init() can be called
        # either via quantize_and_save() (preferred, outside inference_mode) or directly
        # from quantize() as a fallback for non-AutoScheme cases.
        self._post_init_done = False

        # Apply torch compile adjustments eagerly so that ar.enable_torch_compile
        # reflects the correct value immediately after construction (not only after post_init).
        self._precheck_torch_compile(enable_torch_compile)
        self.compress_context.enable_torch_compile = self.enable_torch_compile

        # ``self.calibration_context`` was created at the top of __init__ so
        # all calibration-related property writes above (nsamples / seqlen /
        # batch_size from kwargs) have already routed through it.

        self.has_variable_block_shape = False
        fixed_attr = get_predefined_fixed_attr(self.model) or {}
        for key, value in fixed_attr.items():
            setattr(self, key, value)

        self.need_calib = self._check_need_calib()
        calibrator_kind = self._get_calibrator_kind()
        # The default pile dataset is model-adaptive for pure-text LLMs. Any
        # non-default dataset remains untouched.
        if self.need_calib and not dataset_was_explicitly_set and calibrator_kind == "llm":
            from auto_round.calib_dataset import get_code_calibration_dataset
            from auto_round.utils.model import is_code_model

            detection_config = model_config or getattr(self.model_context.model, "config", None)
            if is_code_model(model, detection_config):
                self.dataset = get_code_calibration_dataset(self.calibration_context.nsamples)
                logger.info("Automatically selected code calibration dataset: %s", self.dataset)
            else:
                logger.info("Using default calibration dataset %s.", self.dataset)
            self.calibration_context.dataset = self.dataset

    def _check_need_calib(self) -> bool:
        """Whether this compressor instance actually needs calibration data.

        Returns True when imatrix/opt-rtn is enabled, activation calibration is
        needed (e.g. act_dynamic=False with NV FP types), or an AutoScheme is in
        use.  Returns False for pure zero-shot RTN cases.
        """

        # During early __init__ quantize_config may not exist yet — default to True.
        if not hasattr(self, "quantize_config") or self.quantize_config is None:
            return True
        return self._needs_calibration_data()

    def _needs_calibration_data(self) -> bool:
        """Determine whether calibration data is truly required.

        Calibration data IS required when:
        - Static activation quantization is needed (act_dynamic=False with NV FP)
        - AutoScheme is being used (needs delta-loss evaluation)
        - The quantizer uses iterative optimization (iters > 0, i.e., SignRound)

        Otherwise, zero-shot (RTN/opt-RTN) quantization can proceed without data.
        """
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        if any(getattr(config, "need_calib", True) for config in self._alg_configs):
            return True

        # AutoScheme needs data for delta-loss scheme selection
        if isinstance(self.scheme, AutoScheme):
            return True

        # Check if activation calibration is needed
        from auto_round.compressors.utils import check_need_act_calibration

        _, _, final_attrs = parse_scheme(self.scheme, {})
        act_bits = final_attrs["act_bits"]
        act_data_type = final_attrs["act_data_type"]
        act_dynamic = final_attrs["act_dynamic"]
        is_act_quantize = act_bits is not None and act_bits <= 8
        if is_act_quantize and check_need_act_calibration(
            act_dynamic,
            act_data_type,
            act_bits if act_bits is not None else 16,
            static_kv_dtype=self.static_kv_dtype,
            static_attention_dtype=self.static_attention_dtype,
        ):
            return True

        return False

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def tokenizer(self) -> Any:
        """Convenience accessor for the tokenizer stored in ``model_context``."""
        return self.model_context.tokenizer

    def _replace_compression_plan(self, **changes) -> None:
        """Atomically update compatibility state once the immutable plan exists."""
        plan = self.__dict__.get("compression_plan")
        if plan is not None:
            self.__dict__["compression_plan"] = replace(plan, **changes)

    @property
    def scheme_context(self) -> Optional[QuantizationScheme]:
        plan = self.__dict__.get("compression_plan")
        return plan.scheme.value if plan is not None else self.__dict__.get("_scheme_context")

    @scheme_context.setter
    def scheme_context(self, value: Optional[QuantizationScheme]) -> None:
        self.__dict__["_scheme_context"] = value
        plan = self.__dict__.get("compression_plan")
        if plan is not None and value is not None:
            self._replace_compression_plan(
                scheme=ResolvedScheme.from_scheme(value, preset_name=plan.scheme.preset_name)
            )

    @property
    def formats(self):
        raw_value = self.__dict__.get("_formats")
        plan = self.__dict__.get("compression_plan")
        if plan is None or isinstance(raw_value, str) or (not plan.formats and raw_value is None):
            return raw_value
        return list(plan.formats)

    @formats.setter
    def formats(self, value) -> None:
        self.__dict__["_formats"] = value
        if value is not None and not isinstance(value, str):
            self._replace_compression_plan(formats=tuple(value))

    @property
    def layer_config(self) -> Optional[dict]:
        plan = self.__dict__.get("compression_plan")
        if plan is None:
            return self.__dict__.get("_layer_config")
        return {name: dict(config) for name, config in plan.layer_config.items()}

    @layer_config.setter
    def layer_config(self, value) -> None:
        self.__dict__["_layer_config"] = value
        if value is not None:
            self._replace_compression_plan(layer_config=value)

    @property
    def regex_config(self) -> Optional[dict]:
        plan = self.__dict__.get("compression_plan")
        if plan is None:
            return self.__dict__.get("_regex_config")
        return {name: dict(config) for name, config in plan.regex_config.items()}

    @regex_config.setter
    def regex_config(self, value) -> None:
        self.__dict__["_regex_config"] = value
        if value is not None:
            self._replace_compression_plan(regex_config=value)

    @property
    def has_qlayer_outside_block(self) -> bool:
        plan = self.__dict__.get("compression_plan")
        if plan is None:
            return self.__dict__.get("_has_qlayer_outside_block", False)
        return plan.has_qlayer_outside_block

    @has_qlayer_outside_block.setter
    def has_qlayer_outside_block(self, value: bool) -> None:
        self.__dict__["_has_qlayer_outside_block"] = value
        self._replace_compression_plan(has_qlayer_outside_block=value)

    @property
    def scale_dtype(self):
        plan = self.__dict__.get("compression_plan")
        return plan.scale_dtype if plan is not None else self.__dict__.get("_scale_dtype")

    @scale_dtype.setter
    def scale_dtype(self, value) -> None:
        self.__dict__["_scale_dtype"] = value
        self._replace_compression_plan(scale_dtype=value)

    @property
    def quant_block_list(self):
        plan = self.__dict__.get("compression_plan")
        if plan is None:
            return self.__dict__.get("_quant_block_list")
        return [list(group) for group in plan.quant_block_list] if plan.quant_block_list is not None else None

    @quant_block_list.setter
    def quant_block_list(self, value) -> None:
        self.__dict__["_quant_block_list"] = value
        self._replace_compression_plan(quant_block_list=value)

    # ── Scheme resolution ─────────────────────────────────────────────────────

    def resolve_scheme(
        self,
        model_context: Optional[ModelContext] = None,
        compress_context: Optional[CompressContext] = None,
    ) -> None:
        """Phase-1 init: resolve scheme and bind config attrs (no model structure needed).

        Must be called BEFORE ``_resolve_formats()`` and BEFORE ``_scheme_post_init()``.
        Idempotent: safe to call multiple times.
        """
        if self._scheme_resolved:
            return

        if model_context is not None:
            self.model_context = model_context
        if compress_context is not None:
            self.compress_context = compress_context

        user_scheme_overrides = collect_user_scheme_overrides(self._alg_configs)
        default_scheme, self.is_auto_scheme, final_attrs = parse_scheme(self.scheme, user_scheme_overrides)

        self.scheme_context = QuantizationScheme.from_dict(final_attrs)
        for config in self._alg_configs:
            if hasattr(config, "scheme"):
                config.scheme = self.scheme_context
        self.quantize_config.check_config()
        for config in self._alg_configs:
            finalize_scheme = getattr(config, "finalize_scheme", None)
            if callable(finalize_scheme):
                finalize_scheme()

        self.orig_scheme = copy.deepcopy(self.scheme)
        self.scheme = default_scheme

        gguf_scheme_name = get_gguf_scheme(self.scheme)
        if self.scale_dtype is None:
            self.scale_dtype = "fp32" if gguf_scheme_name else "fp16"
        self.scale_dtype = convert_dtype_str2torch(self.scale_dtype)

        self._scheme_resolved = True

    def _scheme_post_init(self) -> None:
        """Phase-4 init: build layer config on the patched model.

        Requires ``resolve_scheme()`` to have been called first.
        Must be called AFTER ``model_context.apply_patches()``.
        """
        assert self._scheme_resolved, (
            "resolve_scheme() must be called before _scheme_post_init(). "
            "BaseCompressor.post_init() does this automatically."
        )

        enable_gguf_official_mixed = not self.is_auto_scheme

        if self.quant_block_list is None:
            quant_nontext_module = getattr(self.model_context, "quant_nontext_module", False)
            all_blocks = get_block_names(self.model_context.model, quant_vision=quant_nontext_module)
            self.quant_block_list = find_matching_blocks(
                self.model_context.model, all_blocks, self.to_quant_block_names
            )
            if self.to_quant_block_names is None and self.quant_block_list:
                self.to_quant_block_names = extract_block_names_to_str(self.quant_block_list)

        self.configure_layer_config(enable_gguf_official_mixed=enable_gguf_official_mixed)

    def _gen_auto_scheme(self) -> dict[str, dict]:
        """Generate per-layer config via AutoScheme delta-loss selection."""
        if self.model_context.is_mllm:
            # AutoScheme on a VLM only scores the language tower (the block
            # walker in delta_loss already skips vision/audio sub-trees) and
            # uses a pure-text calibration dataset by default, falling back to
            # the multimodal dataloader if the VLM rejects text-only forward.
            logger.info(
                "AutoScheme on multimodal LLM: scoring the language tower only "
                "with text-only calibration (multimodal dataloader will be used "
                "as a fallback if needed)."
            )

        if is_quantized_input_module(self.model_context.model):
            raise NotImplementedError("AutoScheme does not currently support quantized input models (e.g., FP8).")

        all_dtypes = []
        all_gguf = True
        for option in self.orig_scheme.options:
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

        unique_dtypes = set(all_dtypes)
        if len(unique_dtypes) > 1 and not all_gguf:
            logger.warning(
                "Models with mixed data_types "
                "cannot yet be exported to real formats except GGUF. "
                "Please save the model using the `fake` format for now."
            )

        preset_name = self.scheme if isinstance(self.scheme, str) else None
        format_resolution = getattr(
            self,
            "_format_resolution",
            FormatResolution(
                formats=tuple(self.formats or ()),
                scheme=ResolvedScheme.from_scheme(self.scheme_context, preset_name=preset_name),
                scale_dtype=self.scale_dtype,
                quant_block_list=self.quant_block_list,
            ),
        )
        resolved_layer_config = resolve_layer_config(
            model=self.model_context.model,
            scheme=format_resolution.scheme,
            layer_config=self.layer_config,
            scale_dtype=self.scale_dtype,
            supported_types=self.supported_types,
            inner_supported_types=self.inner_supported_types,
            quant_block_list=self.quant_block_list,
            ignore_layers=self.ignore_layers,
            quant_lm_head=self.quant_lm_head,
            enable_gguf_official_mixed=False,
            is_mllm=self.model_context.is_mllm,
        )
        regex_config = extract_regex_config(
            model=self.model_context.model,
            scheme=format_resolution.scheme,
            layer_config=self.layer_config,
            scale_dtype=self.scale_dtype,
            supported_types=self.supported_types,
            inner_supported_types=self.inner_supported_types,
            ignore_layers=self.ignore_layers,
        )
        discovery_plan = resolve_quantization_config(
            (
                replace(format_resolution, layer_config_patch={})
                if format_resolution.layer_config_patch
                else format_resolution
            ),
            resolved_layer_config,
            regex_config=regex_config,
            has_qlayer_outside_block=has_quantized_layer_outside_blocks(resolved_layer_config),
        )
        apply_plan_to_model(self.model_context.model, discovery_plan)
        layer_config = {name: dict(config) for name, config in discovery_plan.layer_config.items()}
        self.has_qlayer_outside_block = discovery_plan.has_qlayer_outside_block
        self.regex_config = {name: dict(config) for name, config in discovery_plan.regex_config.items()}
        quant_layer_names = layer_config.keys()

        # ---- VLM: peel non-text sub-trees AutoScheme should not score ---- #
        nontext_skipped_layers: dict[str, dict] = {}
        if self.model_context.is_mllm:
            from auto_round.utils import get_block_names

            quant_nontext = getattr(self, "quant_nontext_module", False)
            scoreable_blocks = get_block_names(self.model_context.model, quant_vision=quant_nontext)
            scoreable_block_prefixes = tuple(blk for group in scoreable_blocks for blk in group)
            if quant_nontext:
                peel_markers = AUDIO_MM_KEYS
                tower_label = "language+vision"
                peel_label = "audio/speech"
            else:
                peel_markers = VISION_MM_KEYS + AUDIO_MM_KEYS
                tower_label = "language"
                peel_label = "vision/audio"

            def _is_scoreable_layer(name: str) -> bool:
                if any(name == p or name.startswith(p + ".") for p in scoreable_block_prefixes):
                    return True
                lname = name.lower()
                return not any(marker in lname for marker in peel_markers)

            scoreable_layer_config = {}
            for name, cfg in layer_config.items():
                if _is_scoreable_layer(name):
                    scoreable_layer_config[name] = cfg
                else:
                    nontext_skipped_layers[name] = cfg

            if nontext_skipped_layers:
                logger.info(
                    "AutoScheme (VLM): scoring %d %s-tower layers; "
                    "%d %s-tower layers kept at their original 16-bit configuration.",
                    len(scoreable_layer_config),
                    tower_label,
                    len(nontext_skipped_layers),
                    peel_label,
                )
                layer_config = scoreable_layer_config
                quant_layer_names = layer_config.keys()

        scheme_keys = {f.name for f in fields(QuantizationScheme)}
        fixed_layer_scheme_new = {
            k: {key: v[key] for key in scheme_keys & v.keys()}
            for k, v in layer_config.items()
            if v.get("fixed_by_user", False)
        }

        from auto_round.auto_scheme.gen_auto_scheme import GenScheme

        if (
            not self.compress_context.enable_torch_compile
            and self.quantize_config.super_bits is None
            and not self.orig_scheme.low_gpu_mem_usage
        ):
            logger.warning("we strongly recommend to set `enable_torch_compile` to True for AutoScheme to save VRAM")
        self.scheme_generator = GenScheme(
            self.orig_scheme,
            self.model_context.model,
            quant_layer_names,
            fixed_layer_scheme_new,
            self.dataset,
            device_map=device_manager.device_map,
            tokenizer=self.model_context.tokenizer,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            processor=self.model_context.processor,
        )
        layer_config = self.scheme_generator.get_layer_config()
        # Re-attach vision/audio-tower layers we peeled off earlier so the
        # downstream quantization pipeline sees the complete layer map.
        if nontext_skipped_layers:
            allowed_keys = {f.name for f in fields(QuantizationScheme)} | {
                "fixed_by_user",
                "scale_dtype",
                "scheme",
            }
            for name, cfg in nontext_skipped_layers.items():
                clean_cfg = {k: v for k, v in cfg.items() if k in allowed_keys} if isinstance(cfg, dict) else cfg
                layer_config.setdefault(name, clean_cfg)
        return layer_config

    def configure_layer_config(self, enable_gguf_official_mixed: bool | None = True) -> None:
        """Build ``self.layer_config`` from the resolved scheme on the patched model."""
        # External callers (e.g. llm-compressor's AutoRoundModifier) may invoke this
        # method directly without going through the normal post_init()/_scheme_post_init()
        # sequence. Make sure the scheme is resolved first so `self.scheme_context` (and
        # everything derived from it below) is never None -- resolve_scheme() is a no-op
        # if it already ran.
        if not self._scheme_resolved and hasattr(self, "_alg_configs"):
            self.resolve_scheme()
        _formats = getattr(self.compress_context, "formats", None)
        is_gguf_format = _formats is not None and any(
            "gguf" in str(getattr(fmt, "output_format", "")) for fmt in _formats
        )
        predefined_ignore_layers = get_predefined_ignore_layers(self.model_context.model) if not is_gguf_format else []
        compressed_predefined_ignore_layers = compress_layer_names(predefined_ignore_layers)

        if not is_gguf_format:
            predefined_ignore_layers = get_predefined_ignore_layers(self.model_context.model)
            if predefined_ignore_layers and self.quant_block_list:
                block_prefixes = [block for group in self.quant_block_list for block in group]
                # Only filter layers that are full paths clearly inside a block.
                predefined_ignore_layers = [
                    name
                    for name in predefined_ignore_layers
                    if any(name.startswith(prefix) for prefix in block_prefixes)
                    or not any(prefix.startswith(name.split(".")[0]) for prefix in block_prefixes)
                ]
            if predefined_ignore_layers:
                logger.info(f"Using predefined ignore_layers: {compress_layer_names(predefined_ignore_layers)}")
                # Join the raw (uncompressed) names so that get_fp_layer_names can do exact
                # substring matching. Compressed forms like "layers.[0-61].gate" are
                # misinterpreted as regex character classes ([0-6] matches only digits 0-6)
                # and fail to cover layers with two-digit indices (7, 8, …).
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

        fill_default_value = not self.is_auto_scheme
        source_layer_config = self.layer_config
        format_resolution = getattr(
            self,
            "_format_resolution",
            FormatResolution(
                formats=tuple(self.formats or ()),
                scheme=ResolvedScheme.from_scheme(
                    self.scheme_context,
                    preset_name=self._resolve_gguf_preset_string(self.formats or []),
                ),
                scale_dtype=self.scale_dtype,
                quant_block_list=self.quant_block_list,
            ),
        )
        resolved_layer_config = resolve_layer_config(
            model=self.model_context.model,
            scheme=format_resolution.scheme,
            layer_config=source_layer_config,
            scale_dtype=self.scale_dtype,
            supported_types=SUPPORTED_LAYER_TYPES,
            inner_supported_types=INNER_SUPPORTED_LAYER_TYPES,
            quant_block_list=self.quant_block_list,
            ignore_layers=self.ignore_layers,
            quant_lm_head=self.quant_lm_head,
            enable_gguf_official_mixed=enable_gguf_official_mixed,
            is_mllm=self.model_context.is_mllm,
            fill_default_value=fill_default_value,
        )
        regex_config = extract_regex_config(
            model=self.model_context.model,
            scheme=format_resolution.scheme,
            layer_config=source_layer_config,
            scale_dtype=self.scale_dtype,
            supported_types=SUPPORTED_LAYER_TYPES,
            inner_supported_types=INNER_SUPPORTED_LAYER_TYPES,
            ignore_layers=self.ignore_layers,
            fill_default_value=fill_default_value,
        )
        # ``resolved_layer_config`` already descends from (and fully subsumes)
        # ``format_resolution.layer_config_patch`` -- ``source_layer_config`` above was
        # seeded from that same patch before ``resolve_layer_config()`` expanded any
        # regex/partial keys (e.g. "self_attn") into concrete layer names. Re-merging the
        # patch here would reintroduce those now-stale, unexpanded keys into the final
        # plan, so drop it and rely solely on the fully-resolved layer configuration.
        if format_resolution.layer_config_patch:
            format_resolution = replace(format_resolution, layer_config_patch={})
        self.compression_plan = resolve_quantization_config(
            format_resolution,
            resolved_layer_config,
            regex_config=regex_config,
            has_qlayer_outside_block=has_quantized_layer_outside_blocks(resolved_layer_config),
        )
        self.layer_config = {name: dict(config) for name, config in self.compression_plan.layer_config.items()}
        self.regex_config = {name: dict(config) for name, config in self.compression_plan.regex_config.items()}
        self.has_qlayer_outside_block = self.compression_plan.has_qlayer_outside_block
        apply_plan_to_model(self.model_context.model, self.compression_plan)
        if self.is_auto_scheme:
            self._log_auto_scheme_avg_bits()

    def _log_auto_scheme_avg_bits(self) -> None:
        """Report AutoScheme bit usage under two denominators.

        ``avg_bits`` targets **only** the layers AutoScheme quantizes -- the set it was
        given as ``quant_layer_names``, which is exactly what the bit-allocation DP
        budgets. Layers outside that set (most notably a VLM's vision/audio tower, which
        is peeled off and kept at 16 bit, or layers pinned via ``layer_config`` /
        ``ignore_layers``) are not part of the target and are never compensated by the DP.

        Two numbers are therefore reported:

        * ``quant layers``: average over the quantized (budgeted) layers. This is the
          metric the target constrains and it must be <= target.
        * ``whole model``: average over every layer carrying quantization metadata, i.e.
          the end-to-end footprint. Informational only -- it can legitimately sit above
          the target when non-quantized towers are kept at high precision.
        """
        from auto_round.auto_scheme.utils import compute_layer_bits

        model = self.model_context.model
        ignore_scale_zp_bits = getattr(self.orig_scheme, "ignore_scale_zp_bits", False)
        target_avg_bits = getattr(self.orig_scheme, "avg_bits", None)

        scheme_generator = getattr(self, "scheme_generator", None)
        quant_layer_names = set(getattr(scheme_generator, "quant_layer_names", None) or [])

        quant_params = quant_bits = quant_count = 0
        model_params = model_bits = model_count = 0
        outside = []
        for name, module in model.named_modules():
            if not hasattr(module, "bits") or not hasattr(module, "weight"):
                continue
            n_param = module.weight.numel()
            if n_param == 0 and hasattr(module, "_cached_weight_numel"):
                n_param = module._cached_weight_numel
            if n_param == 0:
                continue
            layer_bits, _ = compute_layer_bits(module, ignore_scale_zp_bits)

            model_params += n_param
            model_bits += layer_bits
            model_count += 1

            # Without a scheme generator (e.g. a reloaded plan) fall back to
            # "actually quantized" as the definition of the quantized set.
            in_quant_set = name in quant_layer_names if quant_layer_names else getattr(module, "bits", 16) < 16
            if in_quant_set:
                quant_params += n_param
                quant_bits += layer_bits
                quant_count += 1
            else:
                outside.append((name, getattr(module, "bits", 16), n_param, layer_bits))

        quant_avg = quant_bits / quant_params if quant_params else float("nan")
        model_avg = model_bits / model_params if model_params else float("nan")
        has_target = isinstance(target_avg_bits, (int, float))

        logger.info(
            "AutoScheme final avg_bits: quant layers=%.4f (target=%.4f, %d layers, %d params, total_bits=%d); "
            "whole model=%.4f (%d layers, %d params, total_bits=%d, informational only)",
            quant_avg,
            float(target_avg_bits) if has_target else float("nan"),
            quant_count,
            quant_params,
            quant_bits,
            model_avg,
            model_count,
            model_params,
            model_bits,
        )

        # Only the quantized-layer average is bound by the target.
        if has_target and quant_avg > float(target_avg_bits) + 1e-3:
            logger.warning(
                "AutoScheme quantized-layer avg_bits=%.4f exceeds target avg_bits=%.4f. "
                "The bit-allocation budget was not met; please report this together with the "
                "AutoScheme option/range logs above.",
                quant_avg,
                float(target_avg_bits),
            )

        if outside:
            outside_params = sum(item[2] for item in outside)
            outside_bits = sum(item[3] for item in outside)
            logger.info(
                "AutoScheme: %d layer(s) (%d params, %d bits, avg=%.4f) are not AutoScheme quantization targets, "
                "so they are excluded from the avg_bits target and only affect the whole-model number "
                "(typically a VLM vision/audio tower kept at 16 bit, or layers pinned via "
                "`layer_config`/`ignore_layers`): %s",
                len(outside),
                outside_params,
                outside_bits,
                outside_bits / outside_params if outside_params else float("nan"),
                ", ".join(f"{n}(bits={b})" for n, b, _, _ in outside[:8]) + (" ..." if len(outside) > 8 else ""),
            )

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mllm(self) -> bool:
        return self.model_context.is_mllm

    @property
    def diffusion(self) -> bool:
        return self.model_context.is_diffusion

    def _get_torch_compile_guard_state(self) -> tuple[bool, bool]:
        """Return raw dtype state used by torch.compile guard rules."""
        # Determine fp8 / nvfp4 intent from raw config before scheme resolution.
        cfg = self.quantize_config
        raw_scheme = self.scheme if isinstance(self.scheme, str) else ""
        raw_dt = (cfg.data_type or "").lower()
        raw_adt = (cfg.act_data_type or "").lower()
        raw_scheme_upper = raw_scheme.upper()

        is_raw_nv_fp = "nv_fp" in raw_dt or "nv_fp" in raw_adt or "NVFP" in raw_scheme_upper
        has_static_global_scale = "static_gs" in raw_adt or "NVFP4" in raw_scheme_upper
        is_valid_act_static = (cfg.act_dynamic is False or has_static_global_scale) and (
            getattr(cfg, "act_bits", 16) or 16
        ) <= 8

        return is_raw_nv_fp, is_valid_act_static

    def _maybe_log_torch_compile_default_hint(self) -> None:
        """Log the default torch.compile hint once final config state is available."""
        is_raw_nv_fp, is_valid_act_static = self._get_torch_compile_guard_state()

        if (
            not self.enable_torch_compile
            and TORCH_VERSION_AT_LEAST_2_6
            and not is_debug_mode()
            and not is_raw_nv_fp
            and not is_valid_act_static
            and self._torch_compile_disabled_reason(ignore_user_override=True) is None
            and self._torch_compile_unsupported_arch_reason() is None
            and self.need_calib
        ):
            logger.info(
                "%s",
                "'enable_torch_compile' is disabled. Enabling it can reduce tuning cost by about 20%.",
            )

    def _torch_compile_disabled_reason(self, ignore_user_override: bool = False) -> Optional[str]:
        """Return why torch.compile must stay off for the current algorithm, else None.

        RTN and optimized RTN quantize each layer in a single pass, and very short
        SignRound runs (``iters < MIN_ITERS_FOR_TORCH_COMPILE``) finish before the
        compilation cost is amortized, so ``torch.compile`` only adds overhead there.

        This only adjusts the *default*: when the user explicitly passed
        ``enable_torch_compile``, their choice is always honored.  Pass
        ``ignore_user_override=True`` to query the algorithm rules alone (used to
        suppress the "you should enable torch.compile" hint).
        """
        if not ignore_user_override and getattr(self, "_torch_compile_user_specified", False):
            return None

        quantize_config = getattr(self, "quantize_config", None)
        if quantize_config is None:
            return None

        # AutoScheme runs its own delta-loss pass on top of the block quantizer and
        # relies on torch.compile to keep VRAM down, so the rules below don't apply.
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        if getattr(self, "is_auto_scheme", False) or isinstance(getattr(self, "scheme", None), AutoScheme):
            return None

        # OptimizedRTNConfig subclasses RTNConfig, so this covers rtn and opt-rtn.
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        if isinstance(quantize_config, RTNConfig):
            return "RTN/OPT-RTN quantizes each layer in a single pass"

        iters = getattr(quantize_config, "iters", None)
        if iters is not None and iters < MIN_ITERS_FOR_TORCH_COMPILE:
            return f"`iters`={iters} is below {MIN_ITERS_FOR_TORCH_COMPILE}"

        return None

    def _torch_compile_unsupported_arch_reason(self) -> Optional[str]:
        """Return why the model *architecture* forbids ``torch.compile``, else ``None``.

        Rules live in :mod:`auto_round.special_model_handler` so a new architecture can
        be registered in one place.
        """
        from auto_round.special_model_handler import get_torch_compile_off_reason

        model = getattr(getattr(self, "model_context", None), "model", None)
        if model is None:
            model = getattr(self, "model", None)
        return get_torch_compile_off_reason(model)

    def _apply_torch_compile_constraints(self, enable_torch_compile: bool) -> None:
        """Apply torch.compile disabling rules for the current compressor state."""
        self.enable_torch_compile = enable_torch_compile
        # Why compilation ended up off, used by ``_log_torch_compile_state``.  When the
        # incoming value is already False, keep the reason recorded by the earlier
        # precheck pass instead of dropping it.
        self._torch_compile_off_reason = (
            None
            if enable_torch_compile
            else (
                getattr(self, "_torch_compile_off_reason", None)
                or getattr(self, "_torch_compile_default_off_reason", None)
            )
        )
        _, is_valid_act_static = self._get_torch_compile_guard_state()

        # On HPU, we rely on torch.compile to speed up the model execution.
        if self.enable_torch_compile and is_valid_act_static:
            self.enable_torch_compile = False
            self._torch_compile_off_reason = "activation is static"
            logger.warning_once("reset enable_torch_compile to `False` as activation is static")

        # Architecture-level hard block (DeepSeek / GLM-5.3-Flash DSA families). These
        # hit dynamo's recompile_limit and cannot be overridden by an explicit
        # ``enable_torch_compile=True``.
        if self.enable_torch_compile:
            arch_reason = self._torch_compile_unsupported_arch_reason()
            if arch_reason is not None:
                self.enable_torch_compile = False
                self._torch_compile_off_reason = arch_reason
                logger.warning_once("reset enable_torch_compile to `False` as %s", arch_reason)

        if self.enable_torch_compile:
            disabled_reason = self._torch_compile_disabled_reason()
            if disabled_reason is not None:
                self.enable_torch_compile = False
                self._torch_compile_off_reason = disabled_reason
                logger.warning_once(
                    "reset enable_torch_compile to `False` as %s, " "so compilation cost would outweigh its benefit",
                    disabled_reason,
                )

    def _precheck_torch_compile(self, enable_torch_compile: bool) -> None:
        """Apply early torch.compile adjustments before scheme resolution.

        This runs during ``__init__`` so the compressor exposes a sensible
        ``enable_torch_compile`` value immediately after construction, even
        though scheme resolution has not completed yet.
        """
        self._apply_torch_compile_constraints(enable_torch_compile)

    def _finalize_torch_compile(self) -> None:
        """Re-evaluate torch.compile after scheme resolution with final attrs."""
        requested_enable_torch_compile = self.enable_torch_compile
        self._apply_torch_compile_constraints(requested_enable_torch_compile)
        if not requested_enable_torch_compile:
            self._maybe_log_torch_compile_default_hint()
        self._log_torch_compile_state()

    def _log_torch_compile_state(self) -> None:
        """Always report the final torch.compile decision so a run is self-documenting."""
        if self.enable_torch_compile:
            logger.info("`torch.compile` is enabled")
            return

        reason = getattr(self, "_torch_compile_off_reason", None)
        if reason is None:
            logger.info("`torch.compile` is disabled")
        else:
            logger.info("`torch.compile` is disabled, as %s", reason)

    def _get_calibration_dataset(self) -> str:
        """Resolve calibration dataset: self.dataset > AutoScheme.dataset > default."""
        dataset = self.calibration_context.dataset
        if dataset is not None:
            return dataset

        return "NeelNanda/pile-10k"

    def post_init(self) -> None:
        """One-time initialization that requires a loaded model.

        Call this OUTSIDE any ``@torch.inference_mode()`` context when using
        AutoScheme – delta-loss selection needs autograd (backward pass).
        ``quantize_and_save()`` does this automatically before entering the
        inference-mode quantize loop.

        Delegates to ordered pipeline phases; see each ``_resolve_scheme``,
        ``_resolve_formats``, ``_build_quantizer``, ``_patch_model``,
        ``_build_layer_config``, and ``_hardware_setup`` for the precise
        preconditions and postconditions.
        """
        if self._post_init_done:
            return

        self._resolve_scheme()

        # After scheme resolution, is_act_quantize is known.  When activation
        # quantization is enabled and the model is in float16, convert to
        # bfloat16 to match the old arch.  This also detaches any parameter
        # tensors that are still backed by safetensors' mmap, preventing
        # per-block RSS growth (~14 MB/block) when .to(device) page-faults
        # the underlying file pages into physical memory.
        if self.quantize_config.is_act_quantize and self.model_context.amp_dtype == torch.float16:
            logger.warning("force to use bf16 for quantization tuning when enabling activation quantization")
            self.model_context.amp_dtype = torch.bfloat16
            if self.model_context.model.dtype != torch.bfloat16:
                self.model_context.model = self.model_context.model.to(torch.bfloat16)

        self._resolve_formats()
        self._patch_model()
        self._build_layer_config()
        self._apply_rotations()

        # Reclaim temporaries from Phases 1-4 (scheme resolution, format
        # parsing, model patching, layer-config walk) before Phase 5
        # allocates hardware/compile objects.  This compacts the heap so that
        # the fragmentation gap between live and freed blocks is minimised.
        gc.collect()
        _force_trim_malloc()

        self._hardware_setup()

        # BlockForwardRunner is now created inside AlgorithmComposer.__init__,
        # so _build_composer must run first.
        self._build_composer()

        # Set block_forward torch compile for block forward
        # Final trim after all init phases.
        gc.collect()
        _force_trim_malloc()

        self._post_init_done = True

    # ── Pipeline phase methods ────────────────────────────────────────────────

    def _resolve_scheme(self) -> None:
        """Phase 1 – Scheme resolution.

        Preconditions:
          - ``self.quantize_config`` is a valid :class:`QuantizationConfig`.

        Work performed:
          - Calls :meth:`resolve_scheme` to derive ``data_type``, ``bits``,
            ``sym``, ``scale_dtype`` etc. and write them back to both ``self``
            and ``self.quantize_config``.

        Postconditions:
          - ``self.scheme`` and ``self.quantize_config`` carry resolved scheme attrs.
        """
        if self.to_quant_block_names is None:
            self.to_quant_block_names = getattr(self.model_context.model, "_autoround_to_quant_block_names", None)

        # Resolve the scheme (pure config work: sets data_type / bits / sym /
        # scale_dtype etc. on both self and self.quantize_config).
        self.resolve_scheme(
            model_context=self.model_context,
            compress_context=self.compress_context,
        )

    def _build_composer(self) -> None:
        """Phase 1b – Quantizer construction and wiring.

        Preconditions:
                    - :meth:`_resolve_scheme` complete: ``self.quantize_config`` carries
                        resolved scheme attrs.
                    - :meth:`_resolve_formats` complete: format-driven overrides have
                        been synced back to ``self.quantize_config``.

        Work performed:
          - Constructs the block_quantizer from the resolved config.
          - Wraps it in a :class:`~auto_round.algorithms.pipeline.AlgorithmComposer`
            so that the entire compressor operates through the bundle abstraction.
          - Calls ``quantizer.bind(self)`` so the quantizer pulls
            ``model_context`` / ``compress_context`` / ``scale_dtype`` /
            ``CalibrationContext`` from this compressor.  ``quantizer.model``
            is a property that reads ``model_context.model``.
          - Exposes ``self.alg_composer.block_quantizer`` so all quantization
            call-sites can reach the block quantizer directly.

        Postconditions:
          - ``self.alg_composer`` is an ``AlgorithmComposer`` wrapping the block quantizer.
          - ``self.alg_composer.block_quantizer`` is ready and shares ``CalibrationContext``
            with the compressor.
        """
        from auto_round.algorithms.composer import AlgorithmComposer

        self._alg_composer = AlgorithmComposer(self._alg_configs, orchestrator=self)

        # Sync the fully-resolved scheme state (built by _build_layer_config(), which
        # always runs before _build_composer() in post_init()) onto the block quantizer
        # so quantization methods (quantize_block, quantize_layer, etc.) have access to
        # layer_config, quant_block_list, etc. ``scale_dtype``/``scheme`` are NOT set here:
        # they are read-only properties backed by ``BaseAlgorithm.bind()``, which already
        # picked up the final ``self.scale_dtype``/``self.scheme_context`` above.
        plan = getattr(self, "compression_plan", None)
        if plan is not None:
            block_quantizer = self._alg_composer.block_quantizer
            block_quantizer.layer_config = {name: dict(config) for name, config in plan.layer_config.items()}
            block_quantizer.has_qlayer_outside_block = plan.has_qlayer_outside_block
            block_quantizer.regex_config = {name: dict(config) for name, config in plan.regex_config.items()}
            block_quantizer.quant_block_list = (
                [list(group) for group in plan.quant_block_list] if plan.quant_block_list is not None else None
            )
            block_quantizer.to_quant_block_names = self.to_quant_block_names
            block_quantizer.ignore_layers = self.ignore_layers

            from auto_round.algorithms.config_resolver import sync_shared_config_from

            sync_shared_config_from(block_quantizer.config, [pre.config for pre in self._alg_composer.preprocessors])

            # Also sync runtime-only state to all preprocessors so they have access to
            # per-layer quant config during pre-processing (e.g. AWQ grid search uses
            # layer_config to look up bits/group_size for each layer).
            for pre in self._alg_composer.preprocessors:
                pre.layer_config = block_quantizer.layer_config

    @property
    def alg_composer(self) -> Any:
        """The active :class:`~auto_round.algorithms.pipeline.AlgorithmComposer`."""
        return self._alg_composer

    @staticmethod
    def _resolve_gguf_preset_string(formats: list["OutputFormat"]) -> Optional[str]:
        """Return the precise GGUF preset string (e.g. ``"gguf:q4_k_m"``) for the
        single resolved GGUF format, or ``None`` if no GGUF format is present.

        Used by :meth:`_resolve_format_string` so ``self.scheme`` stays a string
        that :func:`auto_round.schemes.get_gguf_scheme` can short-circuit on,
        preserving alias disambiguation (Q4_K_S vs Q4_K_M, Q*_0/Q*_1, ...).
        ``_check_compatibility`` guarantees at most one GGUF format.
        """
        for fmt in formats or []:
            if not fmt.is_gguf():
                continue
            # The outer GGUFFormat reports output_format == "gguf"; the precise,
            # alias-correct, post-rewrite preset lives on backend.output_format
            # (e.g. "gguf:q4_k_m"). A standalone/inner GGUFFormat already carries
            # the precise string on output_format itself.
            backend = getattr(fmt, "backend", None)
            backend_fmt = getattr(backend, "output_format", None) if backend is not None else None
            precise = backend_fmt if backend_fmt and backend_fmt != "gguf" else fmt.output_format
            if precise and precise != "gguf":
                return precise
        return None

    def _resolve_format_string(self, format_str: str) -> list["OutputFormat"]:
        """Resolve one format string via resolve_formats(), then propagate any
        scheme/layer_config/scale_dtype/quant_block_list correction it makes
        (e.g. GGUF's gguf_args_check) back onto self.

        `scheme` may or may not be the object passed in — GGUF's "_mixed -> _s"
        rewrite can rebuild it wholesale — so this always re-pins self.scheme_context
        explicitly rather than relying on in-place mutation.
        """
        preset_name = self.scheme if isinstance(self.scheme, str) else None
        self._format_resolution = resolve_formats(
            ResolvedScheme.from_scheme(self.scheme_context, preset_name=preset_name),
            format=format_str,
            layer_config=self.layer_config or {},
            scale_dtype=self.scale_dtype,
            mllm=self.model_context.is_mllm,
            iters=getattr(self, "iters", 0),
            enable_alg_ext=getattr(self, "enable_alg_ext", False),
            quant_nontext_module=self.quant_nontext_module,
            quant_block_list=self.quant_block_list,
            platform=self.platform,
            is_auto_scheme=self.is_auto_scheme,
            model=self.model_context.model,
        )
        formats = list(self._format_resolution.formats)
        scheme = self._format_resolution.scheme.value
        self.layer_config = thaw_mapping(self._format_resolution.layer_config_patch)
        self.scale_dtype = self._format_resolution.scale_dtype
        self.quant_block_list = (
            [list(group) for group in self._format_resolution.quant_block_list]
            if self._format_resolution.quant_block_list is not None
            else None
        )
        self.scheme_context = scheme
        for config in self._alg_configs:
            if hasattr(config, "scheme"):
                config.scheme = self.scheme_context
        # self.scheme must stay an independent object from self.scheme_context
        # (the existing invariant from resolve_scheme(), Phase 1) — never assign
        # the same reference here even though it would often "work".
        #
        # Special-case GGUF: get_gguf_scheme() can only disambiguate presets that
        # share identical QuantizationScheme fields (e.g. GGUF:Q4_K_S vs GGUF:Q4_K_M,
        # or any GGUF:Q*_0 / GGUF:Q*_1 that its field-matching loop deliberately
        # skips) via its string short-circuit. If we hand it the resolved object
        # form, downstream resolve_layer_config() derives the WRONG (or empty) gguf
        # preset name, corrupting per-tensor qtype/embedding selection. So when the
        # resolved format is GGUF, pin self.scheme to the precise resolved preset
        # string (the outer GGUFFormat keeps it on .backend.output_format, already
        # past any "_mixed"->"_s" rewrite); _validate_format_combination guarantees at most
        # one GGUF format here.
        gguf_preset = self._format_resolution.scheme.preset_name
        if gguf_preset is not None and not gguf_preset.startswith("gguf:"):
            gguf_preset = None
        if gguf_preset is not None:
            self.scheme = gguf_preset
        else:
            self.scheme = copy.deepcopy(scheme)
        return formats

    def _resolve_formats(self) -> None:
        """Phase 2 - Format resolution and scheme/config sync.

        Preconditions:
            - Phase 1 complete: ``self.scheme`` / ``self.scheme_context`` are resolved.

        Work performed:
          - Converts a string ``self.formats`` to a list of
            :class:`~auto_round.export.formats.OutputFormat` objects via
            :meth:`_resolve_format_string`, which also propagates any scheme
            correction (e.g. GGUF's ``gguf_args_check``) onto ``self.scheme``,
            ``self.scheme_context``, the algorithm configs that share it, and
            ``self.quantize_config``.
          - Initialises :class:`~auto_round.compressors.shard_writer.ShardWriter`
            when formats are present.

        Postconditions:
          - ``self.formats`` is a list (or ``None``).
          - ``self.compress_context.formats`` mirrors ``self.formats``.
          - ``self.scheme``, ``self.scheme_context`` and ``self.quantize_config``
            all reflect any format-driven corrections (e.g. GGUF).
        """
        if isinstance(self.formats, str):
            self.formats = self._resolve_format_string(self.formats)
        if self.formats is not None:
            self.compress_context.formats = self.formats
            ShardWriter.reset()
            # Defer ShardWriter construction to _ensure_shard_writer() to avoid
            # heap fragmentation during post_init (parameter iteration).

    def _apply_rotations(self) -> None:
        """Phase 4.5 – Apply Hadamard / rotation transforms to the model.

        Preconditions:
          - Phase 3 complete: model topology is final (``apply_patches`` has
            replaced / merged layers, e.g. MoE experts), so rotation operates
            on the same modules that quantization will later see.
          - Phase 4 complete: ``self.layer_config`` is built; rotation only
            transforms weights and does not change layer names, so this
            ordering matches the old arch where rotation ran after
            ``configure_layer_config``.
          - ``self.quantize_config.data_type`` is final (rotation backend
            dispatch depends on it).

        Work performed:
          - Iterates ``self.rotation_configs`` and calls
            :func:`~auto_round.algorithms.transforms.apply_rotation` on the
            model for each config.

        Postconditions:
          - ``self.model_context.model`` carries the rotated weights and any
            inserted online-Hadamard hooks.
        """
        if not self.rotation_configs:
            return
        logger.info("Applying Hadamard transform to the model.")
        for rotation_cfg in self.rotation_configs:
            self.model_context.model = apply_rotation(
                self.model_context.model,
                rotation_cfg,
                data_type=self.quantize_config.data_type,
            )

    def _patch_model(self) -> None:
        """Phase 3 – Model structure patching.

        Preconditions:
          - Phase 2 complete: ``self.formats`` is resolved so that
            ``apply_patches`` can inspect format-specific requirements.

        Work performed:
          - Delegates to :meth:`~auto_round.context.model.ModelContext.apply_patches`
            which may replace or merge layers (e.g. MoE expert merging, adding
            static-kv wrappers) to produce the final model topology.

        Postconditions:
          - ``self.model_context.model`` reflects the definitive topology that
            :meth:`_build_layer_config` will walk.
        """
        # apply_patches() may replace layers (e.g. MoE expert merging); must
        # happen before configure_layer_config() so it sees the final topology.
        self.model_context.apply_patches(self.formats)

    def _build_layer_config(self) -> None:
        """Phase 4 – Layer-config construction.

        Preconditions:
          - Phase 3 complete: model topology is final.
          - ``self.scheme`` and all scheme-resolved attrs are consistent with
            the (possibly GGUF-adjusted) values set in Phase 2.

        Work performed:
          - Calls :meth:`_scheme_post_init` which walks the patched model to
            build ``self.layer_config``, ``self.quant_block_list``,
            ``self.compression_plan``, etc. On the AutoScheme path this also
            runs delta-loss forward/backward passes to select per-layer schemes.

        Postconditions:
          - ``self.layer_config`` and ``self.compression_plan`` are fully populated.
        """
        # configure_layer_config() walks the patched model; _gen_auto_scheme()
        # (AutoScheme path) runs delta-loss forward+backward passes.
        self._scheme_post_init()

    def _hardware_setup(self) -> None:
        """Phase 5 – Hardware and compile configuration.

        Preconditions:
          - Phase 4 complete: ``layer_config`` is built and
            ``has_qlayer_outside_block`` is known.
          - ``self.quantize_config.data_type`` is the final resolved value
            (needed by :meth:`_finalize_torch_compile`).

        Work performed:
          - Applies the device map via :func:`~auto_round.utils.device.set_non_auto_device_map`.
          - Re-evaluates ``torch.compile`` eligibility now that ``data_type`` is
            resolved and writes the result back to ``compress_context``.
          - Resets the offload manager when ``low_cpu_mem_usage`` is active.
          - Disables ``self.inplace`` when quantized layers live outside
            transformer blocks (incompatible with in-place rewriting).
          - Calls :meth:`_adjust_immediate_packing_and_saving` to decide whether
            layers should be packed / written immediately after each block.

        Postconditions:
          - ``compress_context.enable_torch_compile`` is final.
          - ``self.inplace`` and ``compress_context.is_immediate_packing`` /
            ``compress_context.is_immediate_saving`` are set to their definitive values.
        """
        set_non_auto_device_map(self.model_context.model, device_manager.device_map)
        # Re-evaluate torch.compile eligibility now that data_type is resolved.
        self._finalize_torch_compile()
        self.compress_context.enable_torch_compile = self.enable_torch_compile
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reset()

        # Disable inplace when quantized layers live outside transformer blocks.
        # gguf lm-head used rtn in version>=0.13
        if (
            self.has_qlayer_outside_block
            and self.need_calib
            and (
                self.compress_context.formats is None
                or "gguf" not in self.compress_context.formats[0].__class__.__name__.lower()
            )
        ):
            self.inplace = False

        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            self._adjust_immediate_packing_and_saving()

    # backward compatible with the legacy API
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

        # Never proxy private/dunder attributes — they should be set explicitly
        # in __init__.  Proxying them hides bugs (e.g. missing _post_init_done)
        # and can cause infinite recursion.
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Delegate to block_quantizer: access _alg_composer directly from __dict__
        # to avoid recursion (quantizer attribute access forwards through
        # _alg_composer.block_quantizer; going through a property inside
        # __getattr__ would re-trigger __getattr__ if _alg_composer isn't ready yet).
        _alg_composer = self.__dict__.get("_alg_composer")
        if _alg_composer is not None:
            try:
                return object.__getattribute__(_alg_composer.block_quantizer, name)
            except AttributeError:
                pass

        for attr in ["quantize_config", "model_context", "compress_context"]:
            # These are regular instance attributes; use object.__getattribute__
            # so Python's normal descriptor protocol is used without re-entering
            # __getattr__ on self.
            try:
                obj = object.__getattribute__(self, attr)
            except AttributeError:
                continue
            try:
                return object.__getattribute__(obj, name)
            except AttributeError:
                continue

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ── Device state forwarded to the process-wide DeviceManager singleton ────
    @property
    def device(self) -> str:
        return device_manager.device

    @device.setter
    def device(self, value: Union[str, torch.device]) -> None:
        device_manager.device = value

    @property
    def device_list(self) -> list:
        return device_manager.device_list

    @property
    def device_map(self) -> Any:
        return device_manager.device_map

    @property
    def optimizer(self) -> Any:  # TODO wenhuach delete
        """Return the actual optimizer class, converting string to class for backward compat.

        Old API stored ``self.optimizer = torch.optim.AdamW`` (the class itself).
        New arch stores the optimizer name as a string in ``quantize_config.optimizer``.
        This property converts it so that ``ar.optimizer == torch.optim.AdamW`` works.
        """
        if self.quantize_config is None:
            return None
        opt = getattr(self.quantize_config, "optimizer", None)
        if opt is None:
            # Default to AdamW when enable_adam=True and no explicit optimizer was set
            if getattr(self.quantize_config, "enable_adam", False):
                return torch.optim.AdamW
            return None
        if isinstance(opt, str):
            return getattr(torch.optim, opt, None)
        return opt

    def _adjust_immediate_packing_and_saving(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        if self.formats is None:
            return

        formats = getattr(self, "formats", [])
        if any(not format.is_supported_immediate_packing() for format in formats):
            self.compress_context.is_immediate_packing = False
        if any(not format.is_supported_immediate_saving() for format in formats):
            self.compress_context.is_immediate_saving = False

        has_single_gguf_format = len(formats) == 1 and formats[0].is_gguf()
        # GGUF supports per-block / per-layer immediate packing even when
        # full-model in-place rewriting is disabled by outside-block layers.
        if (
            len(formats) == 1
            and not formats[0].is_fake()
            and formats[0].is_supported_immediate_packing()
            and (self.inplace or has_single_gguf_format)
        ):
            self.compress_context.is_immediate_packing = True

        if self.has_qlayer_outside_block and self.need_calib and not has_single_gguf_format:
            self.compress_context.is_immediate_packing = False
        if not ("causallm" in self.model_context.model.__class__.__name__.lower() and not self.model_context.is_mllm):
            # TODO For tied keys, there may some issues, we haven't not verified this
            tied_weight_keys = getattr(self.model_context.model, "_tied_weight_keys", {})
            if len(tied_weight_keys) > 1:
                self.compress_context.is_immediate_saving = False
                if self.compress_context.low_cpu_mem_usage:
                    logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                return
            if len(tied_weight_keys) == 1:
                key = list(tied_weight_keys.keys())[0]
                if "lm_head" not in key:
                    self.compress_context.is_immediate_saving = False
                    if self.compress_context.low_cpu_mem_usage:
                        logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                    return

        if self.compress_context.low_cpu_mem_usage and self.compress_context.is_immediate_packing:
            self.compress_context.is_immediate_saving = True

        if self.compress_context.low_cpu_mem_usage and self.compress_context.is_immediate_packing:
            if formats[0].is_gguf():
                logger.warning(
                    "`low_cpu_mem_usage` is not fully supported for gguf format. "
                    "Setting `low_cpu_mem_usage` to False."
                )
                self.compress_context.low_cpu_mem_usage = False
                self.compress_context.is_immediate_saving = False
            elif (
                self.has_qlayer_outside_block
                and getattr(self, "disable_opt_rtn", None)
                and isinstance(self.quantize_config, RTNConfig)
            ):
                logger.info(
                    "Keeping `low_cpu_mem_usage` enabled in RTN mode (iters=0): "
                    "RTN path uses blockwise quantization and supports per-block offloading."
                )
            elif self.has_qlayer_outside_block and not isinstance(self.quantize_config, RTNConfig):
                logger.warning(
                    "`low_cpu_mem_usage` is not fully supported "
                    "when there are quantized layers outside blocks and optimized RTN is disabled. "
                    "Setting low_cpu_mem_usage to False."
                )
                self.compress_context.low_cpu_mem_usage = False
                self.compress_context.is_immediate_saving = False

        if self.compress_context.is_immediate_saving and not (
            "int" in self.quantize_config.data_type
            or is_nv_fp(self.quantize_config.data_type)
            or is_mx_fp(self.quantize_config.data_type)
        ):
            logger.warning("immediate_saving is only supported for int/nv_fp/mx_fp quantization, set to False")
            self.compress_context.is_immediate_saving = False

        if self.output_dir is None:
            self.compress_context.is_immediate_saving = False

        # Create ShardWriter eagerly only when immediate saving is active
        # (it interleaves with the quantize loop).  Otherwise keep it deferred
        # until save_quantized() to avoid heap fragmentation during init.
        if self.compress_context.is_immediate_saving:
            self._ensure_shard_writer()

    def _ensure_shard_writer(self):
        """Lazily create ShardWriter if it hasn't been created yet."""
        if self.shard_writer is None and self.formats is not None:
            self.shard_writer = ShardWriter(self.model, bits=8)

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        raise NotImplementedError("quantize method must be implemented in subclass")

    def save_quantized(
        self,
        output_dir: str = None,
        format: Union[str, list[OutputFormat]] = None,
        inplace: bool = True,
        return_folders: bool = False,
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
        if output_dir is not None:
            self.compress_context.output_dir = output_dir
        if format is not None:
            if isinstance(format, str) and getattr(self, "formats", None) is None:
                self.formats = self._resolve_format_string(format)
                self.compress_context.formats = self.formats

        if not self.model_context.quantized:
            logger.warning("please run autoround.quantize first")
            return
        folders = []
        if self.formats is None:
            logger.info("format is not set, using default auto_round format.")
            self.formats = "auto_round"
        if isinstance(self.formats, str):
            self.formats = self._resolve_format_string(self.formats)
            self.compress_context.formats = self.formats
        for format in self.formats:
            save_folder = _get_save_folder_name(format)
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
            if serialization_dict.get("to_quant_block_names") is None and self.quant_block_list:
                serialization_dict["to_quant_block_names"] = extract_block_names_to_str(self.quant_block_list)
            if "scale_dtype" in serialization_dict.keys():
                serialization_dict["scale_dtype"] = str(serialization_dict["scale_dtype"])

            original_to_quant_block_names = serialization_dict.get("to_quant_block_names")
            if isinstance(original_to_quant_block_names, list):
                original_to_quant_block_names = original_to_quant_block_names[:]

            # to match the original name
            reverse_checkpoint_conversion_mapping = get_reverse_checkpoint_conversion_mapping(self.model)

            if isinstance(serialization_dict["to_quant_block_names"], str):
                reverted_block_name = revert_checkpoint_conversion_mapping(
                    serialization_dict["to_quant_block_names"], reverse_checkpoint_conversion_mapping
                )
                serialization_dict["to_quant_block_names"] = preserve_original_visual_block_name(
                    original_to_quant_block_names, reverted_block_name
                )

            elif isinstance(serialization_dict["to_quant_block_names"], list):
                for idx in range(len(serialization_dict["to_quant_block_names"])):
                    reverted_block_name = revert_checkpoint_conversion_mapping(
                        serialization_dict["to_quant_block_names"][idx], reverse_checkpoint_conversion_mapping
                    )
                    original_block_name = None
                    if isinstance(original_to_quant_block_names, list) and idx < len(original_to_quant_block_names):
                        original_block_name = original_to_quant_block_names[idx]
                    serialization_dict["to_quant_block_names"][idx] = preserve_original_visual_block_name(
                        original_block_name, reverted_block_name
                    )

            compressed_model = format.save_quantized(
                save_folder,
                model=self.model_context.model,
                layer_config=self.layer_config,
                inplace=inplace,
                tokenizer=self.model_context.tokenizer,
                device=device_manager.device,
                serialization_dict=serialization_dict,
                **kwargs,
            )
            folders.append(save_folder)

        if return_folders:
            if len(folders) == 1:
                folders = folders[0]
            return compressed_model, folders
        else:
            return compressed_model

    def _get_export_dir(self, output_dir: str, format_str: str) -> str:
        """Derive a descriptive export directory from model name and quantization config.

        Must be called after ``post_init()`` so that scheme-resolved attrs
        (bits, group_size, data_type, etc.) are available on ``self.quantize_config``.

        Mirrors the logic previously in ``__main__.py`` so callers only need to
        pass the base ``output_dir`` and the format string.
        """
        # Diffusion models use save_quantized from DiffusionMixin which manages its own
        # directory layout (model_index.json + per-component subdirs).  Appending a
        # scheme-derived suffix here would place files one level too deep.
        if getattr(self, "diffusion", False):
            return output_dir

        model_name = (getattr(self.model_context.model, "name_or_path", "") or "").rstrip("/")
        cfg = self.quantize_config
        group_size = cfg.group_size
        bits = cfg.bits
        data_type = cfg.data_type or "int"
        act_bits = cfg.act_bits or 16
        act_data_type = cfg.act_data_type or "float"

        is_gguf = "gguf" in (format_str or "")
        last = model_name.split("/")[-1].strip(".")

        if last == "" and not is_gguf:
            # model path is just '.' or './' – put inside output_dir with suffix
            if group_size <= 0:
                suffix = f"afp{act_bits}" if "fp" in act_data_type else f"a{act_bits}"
            else:
                suffix = f"g{group_size}"
            return os.path.join(output_dir, f"w{bits}{suffix}")

        if last == "" and is_gguf:
            return output_dir

        if is_gguf:
            return os.path.join(output_dir, model_name.split("/")[-1] + "-gguf")

        # Normal case: derive suffix from group_size / act config
        if isinstance(group_size, tuple):
            assert len(group_size) == 2, f"Only support 2D group_size, but got {group_size}"
            suffix = f"g{group_size[0]}x{group_size[1]}"
        elif group_size <= 0:
            suffix = f"afp{act_bits}" if "fp" in act_data_type else f"a{act_bits}"
        else:
            suffix = f"g{group_size}"

        prefix = data_type.lower().replace("_", "") if "int" not in data_type or "mx" in data_type else ""
        return os.path.join(
            output_dir,
            model_name.split("/")[-1] + (f"-{prefix}" if prefix else "") + f"-w{bits}{suffix}",
        )

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
        used_default_format = format is None and self.formats is None
        if format and self.formats is None:
            self.formats = format
        if self.formats is None:
            logger.info("format is not set, using default auto_round format.")
            self.formats = "auto_round"

        # If multiple formats are specified, enforce inplace=False
        if len(self.formats.split(",")) > 1:
            inplace = False
        self.inplace = kwargs.get("inplace", inplace)
        kwargs.pop("inplace", None)

        # Perform model quantization
        # IMPORTANT: post_init() must run outside any @torch.inference_mode() context
        # because AutoScheme's delta-loss selection requires gradient tracking.
        self.post_init()
        if used_default_format and scheme_to_preset_name(self.scheme_context) == "FP8_BLOCK":
            logger.warning("--format fp8 is recommended for better compatibility with serving frameworks for now.")
        # If post_init() was called manually before quantize_and_save() (e.g. ar.post_init()
        # in tests), _resolve_formats saw formats=None and was a no-op.  Now that we have set
        # self.formats to a default string above, resolve it into OutputFormat objects so that
        # quantize() and save_quantized() receive proper objects, not a raw string.
        if isinstance(self.formats, str):
            self.formats = self._resolve_format_string(self.formats)
            self.compress_context.formats = self.formats
        # Derive descriptive export dir after post_init so scheme-resolved attrs are available.
        _fmt_str = format or (self.formats if isinstance(self.formats, str) else "")
        output_dir = self._get_export_dir(output_dir, _fmt_str)
        self.output_dir = output_dir
        self.compress_context.output_dir = output_dir
        if self.static_attention_dtype is not None:
            from auto_round.experimental.attention import attention_quant_ctx

            with attention_quant_ctx(
                self.model_context.model,
                static_attention_dtype=self.static_attention_dtype,
                static_attention_granularity=self.static_attention_granularity,
            ):
                self.quantize()
                self.model_context.quantized = True
        elif self.static_kv_dtype is not None:
            from auto_round.experimental.kv_cache import kvcache_quant_context

            with kvcache_quant_context(
                self.model_context.model,
                static_kv_dtype=self.static_kv_dtype,
                static_kv_granularity=self.static_kv_granularity,
            ):
                self.quantize()
                self.model_context.quantized = True
        else:
            self.quantize()
            self.model_context.quantized = True

        # Ensure ShardWriter is ready before saving (deferred from post_init).
        self._ensure_shard_writer()

        # Save the quantized model in the specified format_list
        model, folders = self.save_quantized(output_dir, inplace=inplace, return_folders=True, **kwargs)
        memory_monitor.log_summary()

        # Only now -- after the full export (packing pass, config/tokenizer
        # writes) has actually succeeded -- is it safe to drop the resume
        # manifest. See the deferral comment in Orchestrator._quantize_data_driven().
        if self._resume_states:
            for rs in self._resume_states:
                rs.clear()
            self._resume_states = None

        return model, folders


#: Backward-compatible alias — prefer ``BaseOrchestrator`` in new code.
BaseCompressor = BaseOrchestrator
