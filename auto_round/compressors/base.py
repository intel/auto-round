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
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union

import torch
from transformers import AutoConfig, set_seed

from auto_round.algorithms.quantization import BaseQuantizer, QuantizationConfig
from auto_round.algorithms.transforms import (
    BaseRotationConfig,
    apply_rotation,
)
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors.shard_writer import ShardWriter
from auto_round.compressors.utils import _get_save_folder_name, is_mx_fp, is_nv_fp, set_layer_config
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.formats import OutputFormat, get_formats
from auto_round.logger import logger
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    get_gguf_scheme,
    parse_scheme,
    preset_name_to_scheme,
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
from auto_round.utils.device_manager import device_manager
from auto_round.utils.offload import OffloadManager


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


class BaseCompressor(object):
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
    scale_dtype = None
    layer_config = None
    has_qlayer_outside_block: bool = False
    regex_config: dict = None
    quant_block_list: list = None
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
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config: Optional[dict] = None,
        nsamples: int = None,
        seqlen: int = None,
        scale_dtype: Optional[Union[str, torch.dtype]] = None,
        ignore_layers: str = "",
        quant_lm_head: bool = False,
        to_quant_block_names: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> None:
        # ``CalibrationState`` is the single source of truth for calibration
        # runtime state.  Seed every calibration field here in one block so
        # the rest of ``__init__`` only ever interacts with the state object
        # via property forwarders.  ``_resolve_scheme`` later wires this same
        # instance onto the quantizer so the two share state.
        from auto_round.calibration.state import CalibrationState

        self._calibration_state = CalibrationState(
            nsamples=nsamples if nsamples is not None else 128,
            seqlen=seqlen if seqlen is not None else 2048,
            batch_size=kwargs.pop("batch_size", 8),
            gradient_accumulate_steps=kwargs.pop("gradient_accumulate_steps", 1),
        )

        # ``dataset`` is not a named __init__ parameter – it arrives via
        # **kwargs from the compatibility layer.  Pop it early and route
        # through the property setter so CalibrationState owns it.
        _dataset = kwargs.pop("dataset", None)
        if _dataset is not None:
            self.dataset = _dataset

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
        # ``self._calibration_state`` (seeded above) and exposed via
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
        # lazily by ``DataDrivenCompressor.post_init`` based on ``_get_calibrator_kind()``;
        # remains ``None`` for ``ZeroShotCompressor`` (RTN does not need data).
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

        self.static_attention_dtype = kwargs.pop("static_attention_dtype", None)
        # Attention static dtype
        if self.static_attention_dtype is not None:
            logger.warning("The static attention dtype is experimental and currently has limited support.")
        # KV cache, this one does not affect tuning but will collect some infos during tuning
        self.static_kv_dtype = kwargs.pop("static_kv_dtype", None)
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
        # Alternatively, you can use CompressContext.create_context
        self.compress_context = CompressContext(
            low_cpu_mem_usage,
            low_gpu_mem_usage,
            enable_torch_compile,
            formats=self.formats,
            static_kv_dtype=self.static_kv_dtype,
            static_attention_dtype=self.static_attention_dtype,
        )
        self.shard_writer = None

        # Flag for post_init idempotency.  Set to False here so post_init() can be called
        # either via quantize_and_save() (preferred, outside inference_mode) or directly
        # from quantize() as a fallback for non-AutoScheme cases.
        self._post_init_done = False

        # Apply torch compile adjustments eagerly so that ar.enable_torch_compile
        # reflects the correct value immediately after construction (not only after post_init).
        self._precheck_torch_compile(enable_torch_compile)
        self.compress_context.enable_torch_compile = self.enable_torch_compile

        # ``self._calibration_state`` was created at the top of __init__ so
        # all calibration-related property writes above (nsamples / seqlen /
        # batch_size from kwargs) have already routed through it.

        self.has_variable_block_shape = False
        fixed_attr = get_predefined_fixed_attr(self.model) or {}
        for key, value in fixed_attr.items():
            setattr(self, key, value)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def tokenizer(self) -> Any:
        """Convenience accessor for the tokenizer stored in ``model_context``."""
        return self.model_context.tokenizer

    # ── Scheme resolution ─────────────────────────────────────────────────────

    def resolve_scheme(
        self,
        model_context: Optional[ModelContext] = None,
        compress_context: Optional[CompressContext] = None,
        dataset: Optional[str] = None,
    ) -> None:
        """Phase-1 init: resolve scheme and bind config attrs (no model structure needed).

        Must be called BEFORE ``get_formats()`` and BEFORE ``_scheme_post_init()``.
        Idempotent: safe to call multiple times.
        """
        if self._scheme_resolved:
            return

        if model_context is not None:
            self.model_context = model_context
        if compress_context is not None:
            self.compress_context = compress_context
        if dataset is not None:
            self.dataset = dataset

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
            is_mllm=self.model_context.is_mllm,
        )
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
            _gguf_orig_fmt = getattr(self, "_gguf_original_format_name", None)
            if _gguf_orig_fmt and "_MIXED" in _gguf_orig_fmt.upper():
                self.layer_config = _handle_special_schemes(
                    _gguf_orig_fmt.lower(),
                    self.layer_config,
                    self.model_context.model,
                    supported_types=SUPPORTED_LAYER_TYPES,
                    inner_supported_types=INNER_SUPPORTED_LAYER_TYPES,
                    quant_lm_head=self.quant_lm_head,
                    mllm=self.model_context.is_mllm,
                )

        fill_default_value = not self.is_auto_scheme
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
            gguf_format_name=getattr(self, "_gguf_format_name", None),
        )
        if self.is_auto_scheme:
            from auto_round.auto_scheme.utils import compute_avg_bits_for_model

            ignore_scale_zp_bits = getattr(self.orig_scheme, "ignore_scale_zp_bits", False)
            avg_bits, total_bits = compute_avg_bits_for_model(
                self.model_context.model,
                ignore_scale_zp_bits=ignore_scale_zp_bits,
            )
            logger.info(
                "AutoScheme final effective avg_bits=%.4f, target avg_bits=%.4f, total_bits=%d",
                avg_bits,
                self.orig_scheme.avg_bits,
                total_bits,
            )

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mllm(self) -> bool:
        return self.model_context.is_mllm

    @property
    def diffusion(self) -> bool:
        return self.model_context.is_diffusion

    def _get_torch_compile_guard_state(self) -> tuple[bool, bool, int]:
        """Return raw dtype state used by torch.compile guard rules."""
        # Determine fp8 / nvfp4 intent from raw config before scheme resolution.
        cfg = self.quantize_config
        raw_scheme = self.scheme if isinstance(self.scheme, str) else ""
        raw_dt = (cfg.data_type or "").lower()
        raw_adt = (cfg.act_data_type or "").lower()
        raw_scheme_upper = raw_scheme.upper()

        is_raw_nv_fp = "nv_fp" in raw_dt or "nv_fp" in raw_adt or "NVFP" in raw_scheme_upper
        is_raw_fp8 = (
            "fp8" in raw_dt
            or "fp8" in raw_adt
            or "FP8" in raw_scheme_upper
            or ("fp" in raw_dt and getattr(cfg, "bits", 16) == 8)
            or ("fp" in raw_adt and getattr(cfg, "act_bits", 16) == 8)
        )

        act_bits = getattr(cfg, "act_bits", 16) or 16
        return is_raw_fp8, is_raw_nv_fp, act_bits

    def _maybe_log_torch_compile_default_hint(self) -> None:
        """Log the default torch.compile hint once final config state is available."""
        is_raw_fp8, _, act_bits = self._get_torch_compile_guard_state()
        if (
            not self.enable_torch_compile
            and TORCH_VERSION_AT_LEAST_2_6
            and act_bits > 8
            and not is_debug_mode()
            and not is_raw_fp8
            and self.need_calib
        ):
            logger.info(
                "%s",
                "'enable_torch_compile' is set to `False` by default. "
                "Enabling it can reduce tuning cost by 20%, but it might throw an exception.",
            )

    def _apply_torch_compile_constraints(self, enable_torch_compile: bool) -> None:
        """Apply torch.compile disabling rules for the current compressor state."""
        self.enable_torch_compile = enable_torch_compile
        cfg = self.quantize_config
        is_raw_fp8, is_raw_nv_fp, _ = self._get_torch_compile_guard_state()

        # On HPU, we rely on torch.compile to speed up the model execution.
        if self.enable_torch_compile and is_raw_fp8 and not is_hpex_available():
            self.enable_torch_compile = False
            logger.warning_once("reset enable_torch_compile to `False` as fp8 is enabled")
        # TODO: fix https://github.com/intel/auto-round/issues/1109
        if self.enable_torch_compile and is_raw_nv_fp:
            self.enable_torch_compile = False
            logger.warning_once("reset enable_torch_compile to `False` as nvfp4 is enabled")
        # super_group_size = getattr(cfg, "super_group_size", None)
        # enable_alg_ext = getattr(cfg, "enable_alg_ext", False)
        # if self.enable_torch_compile and super_group_size is not None and enable_alg_ext:
        #     self.enable_torch_compile = False
        #     logger.warning_once(
        #         "reset enable_torch_compile to `False` as super_group_size is set for algorithm extension"
        #     )

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

    def _get_calibration_dataset(self) -> str:
        """Resolve calibration dataset: self.dataset > AutoScheme.dataset > default."""
        dataset = self._calibration_state.dataset
        if dataset is not None:
            return dataset
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        scheme = self.scheme
        if isinstance(scheme, AutoScheme) and scheme.dataset:
            return scheme.dataset
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
        self._build_quantizer()
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
            dataset=self._get_calibration_dataset(),
        )

    def _build_quantizer(self) -> None:
        """Phase 1b – Quantizer construction and wiring.

        Preconditions:
                    - :meth:`_resolve_scheme` complete: ``self.quantize_config`` carries
                        resolved scheme attrs.
                    - :meth:`_resolve_formats` complete: format-driven overrides have
                        been synced back to ``self.quantize_config``.

        Work performed:
          - Constructs the block_quantizer from the resolved config.
          - Wraps it in a :class:`~auto_round.algorithms.pipeline.QuantizationPipeline`
            so that the entire compressor operates through the pipeline abstraction.
          - Calls ``quantizer.bind(self)`` so the quantizer pulls
            ``model_context`` / ``compress_context`` / ``scale_dtype`` /
            ``CalibrationState`` from this compressor.  ``quantizer.model``
            is a property that reads ``model_context.model``.
          - Exposes ``self.quantizer`` as a ``@property`` (see below) that
            transparently delegates to ``self.pipeline.block_quantizer`` so all
            existing call-sites continue to work without modification.

        Postconditions:
          - ``self.pipeline`` is a ``QuantizationPipeline`` wrapping the block quantizer.
          - ``self.quantizer`` (via property) is ready and shares ``CalibrationState``
            with the compressor.
        """
        from auto_round.algorithms.pipeline import QuantizationPipeline

        self._pipeline = QuantizationPipeline.from_configs(self._alg_configs, compressor=self)

    @property
    def quantizer(self) -> BaseQuantizer:
        """Transparent forwarder to ``self.pipeline.block_quantizer``.

        All existing ``self.quantizer.xxx`` call-sites continue to work
        unchanged.  New code should prefer ``self.pipeline`` for pipeline-aware
        operations.
        """
        _pipeline = self.__dict__.get("_pipeline")
        if _pipeline is not None:
            return _pipeline.block_quantizer
        return self.__dict__["_quantizer"]

    @quantizer.setter
    def quantizer(self, value: BaseQuantizer) -> None:
        _pipeline = self.__dict__.get("_pipeline")
        if _pipeline is not None:
            _pipeline.block_quantizer = value
        else:
            self.__dict__["_quantizer"] = value

    @property
    def pipeline(self) -> Any:
        """The active :class:`~auto_round.algorithms.pipeline.QuantizationPipeline`."""
        return self._pipeline

    def _resolve_formats(self) -> None:
        """Phase 2 – Format resolution and config attr sync.

        Preconditions:
                    - Phase 1 complete: the scheme is resolved (``data_type``, ``bits``,
                        ``sym`` etc. are set on both ``self`` and ``self.quantize_config``).

        Work performed:
          - Converts a string ``self.formats`` to a list of
            :class:`~auto_round.formats.OutputFormat` objects via
            :func:`~auto_round.formats.get_formats`.
          - Initialises :class:`~auto_round.compressors.shard_writer.ShardWriter`
            when formats are present.
                    - **(2b)** Detects format-driven attribute mutations (``bits``, ``sym``,
            ``data_type``, ``group_size``, etc.) that ``gguf_args_check`` may
                        have written onto ``self`` inside ``get_formats``, syncs them back
                        to ``self.quantize_config``, and rebuilds ``self.scheme`` accordingly.
                    - Merges any format-injected entries into ``self.layer_config``.

        Postconditions:
          - ``self.formats`` is a list (or ``None``).
          - ``self.compress_context.formats`` mirrors ``self.formats``.
                    - ``self.quantize_config`` and ``self.scheme`` reflect the final attrs.
        """
        # get_formats() inspects data_type / bits etc. that were just resolved.
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
        if self.formats is not None:
            self.compress_context.formats = self.formats
            ShardWriter.reset()
            # Defer ShardWriter construction to _ensure_shard_writer() to avoid
            # heap fragmentation during post_init (parameter iteration).

        # Snapshot the user-specified layer_config before format processing may
        # inject extra per-layer entries (e.g. GGUF embedding / lm_head).
        _pre_gguf_layer_config = copy.copy(self.layer_config) or {}

        # ── 2b: propagate format-adjusted attrs back to quantize_config ─────
        # gguf_args_check (called inside get_formats) may have overridden
        # bits / sym / data_type / super_bits / super_group_size / group_size
        # on *this* BaseCompressor object via setattr(self, ...).  Sync those
        # changes to self.quantize_config before creating the quantizer so it is
        # constructed with the definitive final values.
        _gguf_forwarded_attrs = (
            "bits",
            "sym",
            "data_type",
            "super_bits",
            "super_group_size",
            "group_size",
            "act_bits",
            "scale_dtype",
        )
        # Skip this for AutoScheme — the format is for export only, and the
        # per-layer quantization is already determined by orig_scheme.options.
        # Restore scheme attrs from quantize_config for AutoScheme —
        # gguf_args_check may have set __dict__ entries that shadow the proxy.
        if self.is_auto_scheme:
            for _attr in _gguf_forwarded_attrs:
                if _attr in self.__dict__:
                    del self.__dict__[_attr]

        if not self.is_auto_scheme:
            _any_gguf_attr_changed = False
            for _attr in _gguf_forwarded_attrs:
                if _attr not in self.__dict__:
                    continue
                config_val = getattr(self.quantize_config, _attr, None)
                self_val = self.__dict__[_attr]
                if _attr not in ("scale_dtype", "act_bits") and config_val != self_val:
                    _any_gguf_attr_changed = True
                if config_val != self_val:
                    setattr(self.quantize_config, _attr, self_val)
            # If format resolution changed scheme attrs, rebuild self.scheme so that
            # configure_layer_config() / set_layer_config() see the correct values.
            if _any_gguf_attr_changed:
                from auto_round.schemes import PRESET_SCHEMES
                from auto_round.schemes import QuantizationScheme as _QS

                # Prefer to derive the scheme directly from the gguf format name to
                # avoid ambiguity (e.g. Q4_K_S and Q4_K_M share identical weight attrs).
                _gguf_preset_scheme = None
                _gguf_fmt_name = None
                _gguf_original_fmt_name = None
                for _fmt in self.formats or []:
                    # GGUFFormat (outer) has output_format="gguf" but backend.output_format="gguf:q4_k_m"
                    # GGUFFormat (inner/standalone) has output_format="gguf:q4_k_m"
                    _of = getattr(_fmt, "output_format", "")
                    if "gguf" in str(_of):
                        if str(_of) == "gguf":
                            # outer GGUFFormat: full format in _original_format (e.g. "gguf:q2_k_mixed")
                            # or backend.output_format (e.g. "gguf:q2_k_s" after _mixed → _s conversion)
                            _orig = getattr(_fmt, "_original_format", None)
                            if _orig:
                                _gguf_original_fmt_name = str(_orig).upper()
                            _backend = getattr(_fmt, "backend", None)
                            _of = getattr(_backend, "output_format", _of) if _backend is not None else _of
                        _preset_key = str(_of).upper()
                        if _preset_key in PRESET_SCHEMES:
                            _gguf_preset_scheme = PRESET_SCHEMES[_preset_key]
                            _gguf_fmt_name = _preset_key
                            break
                if _gguf_preset_scheme is not None:
                    # Update scheme on both compressor and quantizer.
                    self.scheme = _gguf_preset_scheme
                    # Store the exact gguf format name so configure_layer_config /
                    # set_layer_config can use it directly, avoiding Q4_K_S / Q4_K_M ambiguity.
                    self._gguf_format_name = _gguf_fmt_name
                    # Store original format name (may include _mixed) for _handle_special_schemes
                    if _gguf_original_fmt_name:
                        self._gguf_original_format_name = _gguf_original_fmt_name
                else:
                    _new_scheme_dict = {f.name: getattr(self, f.name, None) for f in fields(_QS)}
                    _new_scheme = _QS.from_dict({k: v for k, v in _new_scheme_dict.items() if v is not None})
                    self.scheme = _new_scheme

        _gguf_layer_cfg = {
            k: v for k, v in (self.__dict__.get("layer_config") or {}).items() if k not in (_pre_gguf_layer_config)
        }
        if _gguf_layer_cfg:
            if self.layer_config is None:
                self.layer_config = {}
            for _lname, _lval in _gguf_layer_cfg.items():
                self.layer_config.setdefault(_lname, _lval)

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
        """Phase 4 – Layer-config construction and quantizer sync.

        Preconditions:
          - Phase 3 complete: model topology is final.
          - ``self.scheme`` and all scheme-resolved attrs are consistent with
            the (possibly GGUF-adjusted) values set in Phase 2.

        Work performed:
          - Calls :meth:`_scheme_post_init` which walks the patched model to
            build ``self.layer_config``, ``self.quant_block_list``, etc.
            On the AutoScheme path this also runs delta-loss forward/backward
            passes to select per-layer schemes.
          - Syncs the fully-resolved ``layer_config`` and related attrs to
            ``self.quantizer`` so quantization methods have the complete view.

        Postconditions:
          - ``self.layer_config`` is fully populated.
          - ``self.quantizer`` mirrors ``layer_config``, ``has_qlayer_outside_block``,
            ``regex_config``, ``quant_block_list``, ``to_quant_block_names``,
            ``scale_dtype``, and ``ignore_layers``.
        """
        # configure_layer_config() walks the patched model; _gen_auto_scheme()
        # (AutoScheme path) runs delta-loss forward+backward passes.
        self._scheme_post_init()

        # Sync the fully-resolved scheme state to the quantizer so that
        # quantization methods (quantize_block, quantize_layer, etc.) have
        # access to layer_config, scale_dtype, quant_block_list, etc.
        self.quantizer.layer_config = self.layer_config
        self.quantizer.has_qlayer_outside_block = self.has_qlayer_outside_block
        self.quantizer.regex_config = self.regex_config
        self.quantizer.quant_block_list = self.quant_block_list
        self.quantizer.to_quant_block_names = self.to_quant_block_names
        self.quantizer.scale_dtype = self.scale_dtype
        self.quantizer.ignore_layers = self.ignore_layers

        from auto_round.algorithms.config_resolver import sync_shared_config_from

        sync_shared_config_from(self.quantizer.config, [pre.config for pre in self._pipeline.preprocessors])

        # Also sync runtime-only state to all preprocessors in the pipeline so
        # they have access to per-layer quant config during pre-processing (e.g.
        # AWQ grid search uses layer_config to look up bits/group_size for each layer).
        for pre in self._pipeline.preprocessors:
            pre.layer_config = self.layer_config
            pre.scale_dtype = self.scale_dtype

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

        # Delegate to block_quantizer: access _pipeline directly from __dict__ to
        # avoid recursion (quantizer is now a @property backed by _pipeline; going
        # through the property inside __getattr__ would re-trigger __getattr__
        # if _pipeline itself isn't ready yet).
        _pipeline = self.__dict__.get("_pipeline")
        if _pipeline is not None:
            try:
                return object.__getattribute__(_pipeline.block_quantizer, name)
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

    # ── Forwarding properties to ``self._calibration_state`` ──────────────────
    @property
    def calibration_state(self) -> Any:
        return self._calibration_state

    @calibration_state.setter
    def calibration_state(self, value: Any) -> None:
        self._calibration_state = value
        # Re-wire quantizer if it already exists so they keep sharing.
        # quantizer is now a @property forwarding to _pipeline.block_quantizer;
        # use _pipeline directly to avoid triggering __getattr__ loops.
        _pipeline = self.__dict__.get("_pipeline")
        if _pipeline is not None:
            _pipeline.block_quantizer.calibration_state = value

    @property
    def inputs(self) -> dict:
        return self._calibration_state.inputs

    @inputs.setter
    def inputs(self, value: dict) -> None:
        self._calibration_state.inputs = value if value is not None else {}

    @property
    def to_cached_layers(self) -> list:
        return self._calibration_state.to_cached_layers

    @to_cached_layers.setter
    def to_cached_layers(self, value: list) -> None:
        self._calibration_state.to_cached_layers = value if value is not None else []

    @to_cached_layers.deleter
    def to_cached_layers(self) -> None:
        self._calibration_state.to_cached_layers = []

    @property
    def last_cache_name(self) -> Optional[str]:
        return self._calibration_state.last_cache_name

    @last_cache_name.setter
    def last_cache_name(self, value: Optional[str]) -> None:
        self._calibration_state.last_cache_name = value

    @last_cache_name.deleter
    def last_cache_name(self) -> None:
        self._calibration_state.last_cache_name = None

    @property
    def blocks_requiring_input_ids(self) -> list:
        return self._calibration_state.blocks_requiring_input_ids

    @blocks_requiring_input_ids.setter
    def blocks_requiring_input_ids(self, value: list) -> None:
        self._calibration_state.blocks_requiring_input_ids = value if value is not None else []

    @property
    def batch_size(self) -> int:
        return self._calibration_state.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._calibration_state.batch_size = value

    @property
    def gradient_accumulate_steps(self) -> int:
        return self._calibration_state.gradient_accumulate_steps

    @gradient_accumulate_steps.setter
    def gradient_accumulate_steps(self, value: int) -> None:
        if value is not None:
            self._calibration_state.gradient_accumulate_steps = value

    @property
    def nsamples(self) -> int:
        return self._calibration_state.nsamples

    @nsamples.setter
    def nsamples(self, value: int) -> None:
        if value is not None:
            self._calibration_state.nsamples = value

    @property
    def seqlen(self) -> int:
        return self._calibration_state.seqlen

    @seqlen.setter
    def seqlen(self, value: int) -> None:
        if value is not None:
            self._calibration_state.seqlen = value

    @property
    def dataset(self) -> Any:
        return self._calibration_state.dataset

    @dataset.setter
    def dataset(self, value: Any) -> None:
        self._calibration_state.dataset = value

    @property
    def dataloader(self) -> Any:
        return self._calibration_state.dataloader

    @dataloader.setter
    def dataloader(self, value: Any) -> None:
        self._calibration_state.dataloader = value

    @dataloader.deleter
    def dataloader(self) -> None:
        self._calibration_state.dataloader = None

    @property
    def optimizer(self) -> Any:
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
        has_single_gguf_format = len(formats) == 1 and formats[0].is_gguf()
        # GGUF supports per-block / per-layer immediate packing even when
        # full-model in-place rewriting is disabled by outside-block layers.
        if len(formats) == 1 and not formats[0].is_fake() and (self.inplace or has_single_gguf_format):
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
            self.shard_writer = ShardWriter(self.model_context.model, bits=8)

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
                self.formats = get_formats(format, self)
                self.compress_context.formats = self.formats

        if not self.model_context.quantized:
            logger.warning("please run autoround.quantize first")
            return
        folders = []
        if self.formats is None:
            logger.info("format is not set, using default auto_round format.")
            self.formats = "auto_round"
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
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
            if serialization_dict.get("to_quant_block_names") is None and self.quantizer.quant_block_list:
                serialization_dict["to_quant_block_names"] = extract_block_names_to_str(self.quantizer.quant_block_list)
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
                layer_config=self.quantizer.layer_config,
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
        # If post_init() was called manually before quantize_and_save() (e.g. ar.post_init()
        # in tests), _resolve_formats saw formats=None and was a no-op.  Now that we have set
        # self.formats to a default string above, resolve it into OutputFormat objects so that
        # quantize() and save_quantized() receive proper objects, not a raw string.
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
            self.compress_context.formats = self.formats
        # Derive descriptive export dir after post_init so scheme-resolved attrs are available.
        _fmt_str = format or (self.formats if isinstance(self.formats, str) else "")
        output_dir = self._get_export_dir(output_dir, _fmt_str)
        self.output_dir = output_dir
        self.compress_context.output_dir = output_dir
        if self.static_attention_dtype is not None:
            from auto_round.experimental.attention import attention_quant_ctx

            with attention_quant_ctx(self.model_context.model, static_attention_dtype=self.static_attention_dtype):
                self.quantize()
                self.model_context.quantized = True
        elif self.static_kv_dtype is not None:
            from auto_round.experimental.kv_cache import kvcache_quant_context

            with kvcache_quant_context(self.model_context.model, static_kv_dtype=self.static_kv_dtype):
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

        return model, folders
