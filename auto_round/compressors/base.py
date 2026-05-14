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
from transformers import set_seed

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization import BaseQuantizers, QuantizationConfig
from auto_round.algorithms.transforms import (
    BaseRotationConfig,
    apply_rotation,
)
from auto_round.compressors.shard_writer import ShardWriter
from auto_round.compressors.utils import _get_save_folder_name, is_mx_fp, is_nv_fp, set_layer_config
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.formats import OutputFormat, get_formats
from auto_round.logger import logger
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    _parse_scheme,
    get_gguf_scheme,
    preset_name_to_scheme,
)
from auto_round.special_model_handler import get_predefined_fixed_attr, get_predefined_ignore_layers
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    CpuInfo,
    check_and_mark_quantized_module,
    check_seqlen_compatible,
    check_to_quantized,
    clear_memory,
    collapse_ignore_layers,
    compile_func,
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
    revert_checkpoint_conversion_mapping,
)
from auto_round.utils.device import (
    _force_trim_malloc,
    get_major_device,
    patch_xpu_sdpa_drop_causal_mask,
    set_non_auto_device_map,
)
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

    def __init__(
        self,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        scheme="W4A16",
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        layer_config=None,
        nsamples: int = None,
        seqlen: int = None,
        **kwargs,
    ):
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
        for _cfg in _config_list:
            if isinstance(_cfg, QuantizationConfig):
                self.quantize_config = _cfg
            elif isinstance(_cfg, BaseRotationConfig):
                self.rotation_configs.append(_cfg)
        assert self.quantize_config is not None, "QuantizationConfig is required for Compressor"

        # Compressor-level layer params (do not live in QuantizationConfig).
        # Calibration params (nsamples/seqlen/batch_size) are owned by
        # ``self._calibration_state`` (seeded above) and exposed via
        # ``@property`` forwarders.
        self.layer_config = layer_config
        # ``post_init()`` may run before ``quantize_and_save()`` in tests and
        # compatibility paths, so seed the same default used by
        # ``quantize_and_save(..., inplace=True)`` here.
        self.inplace = True

        # Scheme is passed directly to the compressor, not stored in QuantizationConfig.
        self.scheme = scheme

        # Calibrator strategy (auto_round.calibration.base.Calibrator).  Constructed
        # lazily by ``DataDrivenCompressor.post_init`` based on ``_get_calibrator_kind()``;
        # remains ``None`` for ``ZeroShotCompressor`` (RTN does not need data).
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
        quant_nontext_module = kwargs.pop("quant_nontext_module", False)

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

        device = kwargs.pop("device", None)
        if device is not None:
            logger.warning("`device` is deprecated, please use `device_map` instead")

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
        _device = get_major_device(device_map if device_map is not None else 0)

        self.model_context = ModelContext(
            model,
            tokenizer=tokenizer,
            platform=platform,
            model_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            amp=amp,
            need_calib=self.need_calib,
            device=_device,
            formats=self.formats,
            is_act_quantize=self.quantize_config.is_act_quantize,
            quant_nontext_module=quant_nontext_module,
        )
        # Alternatively, you can use CompressContext.create_context
        self.compress_context = CompressContext(
            low_cpu_mem_usage,
            low_gpu_mem_usage,
            device_map,
            enable_torch_compile,
            formats=self.formats,
            static_kv_dtype=self.static_kv_dtype,
            static_attention_dtype=self.static_attention_dtype,
        )
        self.shard_writer = None

        # scale_dtype is resolved in quantizer.resolve_scheme() after scheme resolution,
        # so it is not initialized here to avoid premature evaluation with an unresolved scheme.

        # Flag for post_init idempotency.  Set to False here so post_init() can be called
        # either via quantize_and_save() (preferred, outside inference_mode) or directly
        # from quantize() as a fallback for non-AutoScheme cases.
        self._post_init_done = False

        # Apply torch compile adjustments eagerly so that ar.enable_torch_compile
        # reflects the correct value immediately after construction (not only after post_init).
        self._adjust_torch_compile(enable_torch_compile)
        self.compress_context.enable_torch_compile = self.enable_torch_compile

        # ``self._calibration_state`` was created at the top of __init__ so
        # all calibration-related property writes above (nsamples / seqlen /
        # batch_size from kwargs) have already routed through it.

        self.has_variable_block_shape = False
        fixed_attr = get_predefined_fixed_attr(self.model_context.model) or {}
        for key, value in fixed_attr.items():
            setattr(self, key, value)

    # ── Scheme resolution ─────────────────────────────────────────────────────

    def resolve_scheme(self, model_context=None, compress_context=None, dataset: str = None) -> None:
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

        scheme_fields = {f.name for f in fields(QuantizationScheme)}
        user_scheme_overrides = {
            k: getattr(self.quantize_config, k)
            for k in scheme_fields
            if getattr(self.quantize_config, k, None) is not None
        }
        default_scheme, self.is_auto_scheme, final_attrs = _parse_scheme(self.scheme, user_scheme_overrides)

        for key, value in final_attrs.items():
            setattr(self.quantize_config, key, value)
            if hasattr(self, key):
                setattr(self, key, value)
        self.quantize_config.check_config()

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
                self.quantize_config.to_quant_block_names = self.to_quant_block_names

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
                peel_markers = ("audio", "speech", "wav", "waveform")
                tower_label = "language+vision"
                peel_label = "audio/speech"
            else:
                peel_markers = (
                    "vision",
                    "visual",
                    "image",
                    "img",
                    "audio",
                    "speech",
                    "wav",
                    "waveform",
                )
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
            device_map=self.compress_context.device_map,
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
        is_gguf_format = (f := getattr(self.compress_context, "formats", None)) is not None and "gguf" in f
        predefined_ignore_layers = get_predefined_ignore_layers(self.model_context.model)
        compressed_predefined_ignore_layers = compress_layer_names(predefined_ignore_layers)
        if not is_gguf_format:
            predefined_ignore_layers = get_predefined_ignore_layers(self.model_context.model)
            if predefined_ignore_layers:
                logger.info(f"Using predefined ignore_layers: {compressed_predefined_ignore_layers}")
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

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mllm(self):
        return self.model_context.is_mllm

    @property
    def diffusion(self):
        return self.model_context.is_diffusion

    def _adjust_torch_compile(self, enable_torch_compile: bool) -> None:
        """Sets the torch compile configuration for the tuning."""
        self.enable_torch_compile = enable_torch_compile

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
        # On HPU, we rely on torch.compile to speed up the model execution.
        if self.enable_torch_compile and is_raw_fp8 and not is_hpex_available():
            self.enable_torch_compile = False
            logger.warning_once("reset enable_torch_compile to `False` as fp8 is enabled")
        # TODO: fix https://github.com/intel/auto-round/issues/1109
        if self.enable_torch_compile and is_raw_nv_fp:
            self.enable_torch_compile = False
            logger.warning_once("reset enable_torch_compile to `False` as nvfp4 is enabled")
        super_group_size = getattr(cfg, "super_group_size", None)
        enable_alg_ext = getattr(cfg, "enable_alg_ext", False)
        if self.enable_torch_compile and super_group_size is not None and enable_alg_ext:
            self.enable_torch_compile = False
            logger.warning_once(
                "reset enable_torch_compile to `False` as super_group_size is set for algorithm extension"
            )

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
          - Seeds scheme-related attrs (``scale_dtype``, ``ignore_layers``,
            ``quant_lm_head``, ``to_quant_block_names``) from ``quantize_config``.
          - Calls :meth:`resolve_scheme` to derive ``data_type``, ``bits``,
            ``sym``, ``scale_dtype`` etc. and write them back to both ``self``
            and ``self.quantize_config``.

        Postconditions:
          - ``self.scheme`` and ``self.quantize_config`` carry resolved scheme attrs.
        """
        cfg = self.quantize_config
        self.scale_dtype = cfg.scale_dtype
        # self.layer_config is already set from __init__ (direct compressor param).
        self.ignore_layers = cfg.ignore_layers
        self.quant_lm_head = cfg.quant_lm_head
        self.to_quant_block_names = cfg.to_quant_block_names

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
          - Constructs ``self.quantizer`` from the resolved config.
          - Calls ``quantizer.bind(self)`` so the quantizer pulls
            ``model_context`` / ``compress_context`` / ``scale_dtype`` /
            ``CalibrationState`` from this compressor.  ``quantizer.model``
            is a property that reads ``model_context.model``.

        Postconditions:
          - ``self.quantizer`` is ready and shares ``CalibrationState`` with
            the compressor.
        """
        self.quantizer = BaseQuantizers.from_config(self.quantize_config)
        self.quantizer.bind(self)

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

    def _hardware_setup(self) -> None:
        """Phase 5 – Hardware and compile configuration.

        Preconditions:
          - Phase 4 complete: ``layer_config`` is built and
            ``has_qlayer_outside_block`` is known.
          - ``self.quantize_config.data_type`` is the final resolved value
            (needed by :meth:`_adjust_torch_compile`).

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
        set_non_auto_device_map(self.model_context.model, self.compress_context.device_map)
        # Re-evaluate torch.compile eligibility now that data_type is resolved.
        self._adjust_torch_compile(self.enable_torch_compile)
        self.compress_context.enable_torch_compile = self.enable_torch_compile
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reset()

        # Disable inplace when quantized layers live outside transformer blocks.
        if self.has_qlayer_outside_block and self.need_calib:
            self.inplace = False

        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            self._adjust_immediate_packing_and_saving()

    # backward compatible with the legacy API
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

        for obj in ["quantizer", "quantize_config", "model_context", "compress_context"]:
            if obj not in self.__dict__:
                continue
            obj = object.__getattribute__(self, obj)
            try:
                return object.__getattribute__(obj, name)
            except AttributeError:
                continue

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ── Forwarding properties to ``self._calibration_state`` ──────────────────
    @property
    def calibration_state(self):
        return self._calibration_state

    @calibration_state.setter
    def calibration_state(self, value) -> None:
        self._calibration_state = value
        # Re-wire quantizer if it already exists so they keep sharing.
        q = self.__dict__.get("quantizer")
        if q is not None:
            q.calibration_state = value

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
    def last_cache_name(self):
        return self._calibration_state.last_cache_name

    @last_cache_name.setter
    def last_cache_name(self, value) -> None:
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
    def dataset(self):
        return self._calibration_state.dataset

    @dataset.setter
    def dataset(self, value) -> None:
        self._calibration_state.dataset = value

    @property
    def dataloader(self):
        return self._calibration_state.dataloader

    @dataloader.setter
    def dataloader(self, value) -> None:
        self._calibration_state.dataloader = value

    @dataloader.deleter
    def dataloader(self) -> None:
        self._calibration_state.dataloader = None

    @property
    def optimizer(self):
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
        if len(formats) == 1 and not formats[0].is_fake() and self.inplace:
            self.compress_context.is_immediate_packing = True

        if self.has_qlayer_outside_block and self.need_calib:
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
        if output_dir is not None:
            self.compress_context.output_dir = output_dir
        if format is not None:
            if isinstance(format, str) and getattr(self, "formats", None) is None:
                logger.warning(
                    f"save_quantized with format is deprecated and will be deleted in auto_round version 1.0."
                    f" Please use AutoRound(format='{format}' instead)."
                )
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

            # to match the original name
            reverse_checkpoint_conversion_mapping = get_reverse_checkpoint_conversion_mapping(self.model)

            if isinstance(serialization_dict["to_quant_block_names"], str):
                serialization_dict["to_quant_block_names"] = revert_checkpoint_conversion_mapping(
                    serialization_dict["to_quant_block_names"], reverse_checkpoint_conversion_mapping
                )

            elif isinstance(serialization_dict["to_quant_block_names"], list):
                for idx in range(len(serialization_dict["to_quant_block_names"])):
                    serialization_dict["to_quant_block_names"][idx] = revert_checkpoint_conversion_mapping(
                        serialization_dict["to_quant_block_names"][idx], reverse_checkpoint_conversion_mapping
                    )

            compressed_model = format.save_quantized(
                save_folder,
                model=self.model_context.model,
                layer_config=self.quantizer.layer_config,
                inplace=inplace,
                tokenizer=self.model_context.tokenizer,
                device=self.compress_context.device,
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
            logger.warning(
                f"quantize_and_save with format is deprecated and will be deleted in auto_round version 1.0."
                f" Please use AutoRound(format='{format}' instead)."
            )
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
            if not self.disable_opt_rtn and f"opt_rtn_{dtype}" in QUANT_FUNC_WITH_DTYPE:
                dtype = f"opt_rtn_{dtype}"
            elif f"rtn_{dtype}" in QUANT_FUNC_WITH_DTYPE:
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
                    module.weight.to(dtype=dtype, device=self.device),
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
            clear_memory(device_list=self.device_list)

        return is_quantized

    def _quant_rtn_with_imatrix(self, all_to_quantized_module_names: list[str]) -> None:
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
        logger.info("start to compute imatrix")

        # Load dataset
        from auto_round.calib_dataset import get_dataloader

        if isinstance(self.dataset, str):
            if self.tokenizer is None:
                raise ValueError("A tokenizer must be set for the model when using a dataset string.")
            dataset_name = self.dataset.replace(" ", "")
            self.dataloader = get_dataloader(
                self.tokenizer, self.seqlen, dataset_name, self.seed, self.batch_size, self.nsamples
            )
        else:
            self.dataloader = self.dataset

        model = self.model

        # Dispatch multi-GPU model if necessary
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            dispatch_model(model, model.hf_device_map)

        def register_act_hook(model):
            """Registers hooks to accumulate activation squared norms into `imatrix`."""

            def get_imatrix_hook(module, input, output):
                input = input[0] if isinstance(input, (tuple, list)) else input
                flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
                squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

                if not hasattr(module, "imatrix"):
                    module.imatrix = squared
                    module.imatrix_cnt = input.shape[0]
                else:
                    module.imatrix += squared.to(module.imatrix.device)
                    module.imatrix_cnt += input.shape[0]

            hook_handles = []
            for name, module in model.named_modules():
                if type(module) in self.supported_types and check_to_quantized(module):
                    hook = module.register_forward_hook(get_imatrix_hook)
                    hook_handles.append(hook)
            return hook_handles

        if not is_wint4aint4(self):  # INT4 no imatrix is much better
            hooks = register_act_hook(model)
        else:
            hooks = []

        try:
            if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                import accelerate

                accelerate.hooks.remove_hook_from_submodules(model)
            safe_to_cpu_(model)
            clear_memory(device_list=self.device_list)
            self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            try:
                logger.error(cuda_error_msg)
                # Final fallback: warn and use CPU-only quantization
                logger.warning(
                    "Fallback to CPU. "
                    "Consider enabling `low_gpu_mem_usage` or using more GPUs via `--device 0,1,2,3`."
                )
                safe_to_cpu_(model)
                clear_memory(device_list=self.device_list)
                if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(model)

                orig_device = self.device
                self.device = "cpu"
                self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
                self.device = orig_device
            except Exception as e:
                raise
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

    def _quantize_layer_via_rtn(self, name: str, dtype: torch.dtype = None, to_cpu=True) -> None:
        """Quantizes a layer using RTN (Round-To-Nearest) if available.

        This function attempts to quantize a layer by switching its data type to a
        `rtn_*` version if supported, then wraps and unwraps the module to apply
        quantization. If GPU memory is insufficient, it falls back to CPU.

        If packing is enabled (`immediate_packing`), the function will also export
        the quantized layer to the appropriate backend format.

        Args:
            name (str): Name of the layer to quantize.

        Raises:
            RuntimeError: If quantization fails for reasons unrelated to memory.
        """
        m = get_module(self.model, name)
        if dtype is not None:
            m = m.to(dtype)

        m = convert_module_to_hp_if_necessary(m, self.amp_dtype, self.device)
        set_module(self.model, name, m)
        tuning_device = m.tuning_device if hasattr(m, "tuning_device") else self.device
        # Step 1: let gguf merge layers or rename module first and we will handle the RTN is gguf specific logic
        if self.is_immediate_packing and self.iters == 0 and self.formats[0].is_gguf() and not self.disable_opt_rtn:
            m = m.to(tuning_device)
            m.scale = None
            m.zp = None
        else:
            try:
                disable_opt_rtn = self.disable_opt_rtn
                if (
                    not disable_opt_rtn
                    and self.orig_disable_opt_rtn is None
                    and self.is_moe_model
                    and "expert" in m.global_name
                    and "shared_expert" not in m.global_name
                    and self.super_bits is None  # GGUF still uses the optimized RTN for MoE layers
                ):
                    disable_opt_rtn = True
                    logger.warning_once(
                        "MoE layer detected: optimized RTN is disabled for efficiency. "
                        "Use `--enable_opt_rtn` to force-enable it for MoE layers."
                    )
                m = m.to(tuning_device)
                m = WrapperLinear(
                    m,
                    device=tuning_device,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                    enable_torch_compile=self.enable_torch_compile,
                    disable_opt_rtn=disable_opt_rtn,
                    iters=self.iters,
                )
                m = m.unwrapper({})
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                m = m.orig_layer if hasattr(m, "orig_layer") else m
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU.")
                    m.to("cpu")
                    m = WrapperLinear(
                        m,
                        enable_minmax_tuning=False,
                        enable_norm_bias_tuning=False,
                        enable_round_tuning=False,
                        enable_torch_compile=self.enable_torch_compile,
                        iters=self.iters,
                    )
                    m = m.unwrapper({})
                except Exception as e:
                    raise

        # Step 2: Optional immediate packing/export
        if self.is_immediate_packing:  # For gguf, packing conducts on block level
            self._immediate_pack(name)
            if to_cpu:
                m = m.to("cpu")
                packed_m = get_module(self.model, name)
                set_module(self.model, name, packed_m.to("cpu"))
        else:
            if to_cpu:
                m = m.to("cpu")
            set_module(self.model, name, m)
        if self.is_immediate_saving:
            m = get_module(self.model, name)
            m.to("cpu")
            shard_writer(self, m, name, False)
            # Free RAM immediately: the data is now in the shard-writer buffer
            # (and will be flushed to disk).  Keeping it also in the model tree
            # causes linear RAM growth for large models.
            m.to("meta")

    def _immediate_pack(self, name: str):
        if not self.is_immediate_packing:
            return
        self.formats[0].immediate_pack(
            name=name,
            model=self.model,
            device=self.device,
            output_dir=self._get_save_folder_name(self.formats[0]),
            layer_config=self.layer_config,
            tokenizer=self.tokenizer,
        )

    # Use no_grad instead of inference mode
    # https://github.com/intel/auto-round/issues/1620
    @torch.no_grad()
    def _quantize_rtn(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize all modules in the model using RTN (Round-To-Nearest) strategy.

        If the target format includes GGUF with `k`, and optimized RTN is enabled,
        blockwise quantization with input caching and imatrix is used.

        Returns:
            tuple[nn.Module, Dict[str, Any]]: The quantized model and the layer configuration.
        """
        if self.amp and self.model.dtype != self.amp_dtype:
            self.model.to(self.amp_dtype)

        all_to_quantized_module_names: list[str] = [n for n, m in self.model.named_modules() if check_to_quantized(m)]
        self.all_to_quantized_module_names = all_to_quantized_module_names

        if not (any(fmt.is_gguf() for fmt in getattr(self, "formats", [])) or self.super_bits is not None):
            self._quantize_embedding_layer()  # leave to gguf itself to handle

        # Release memory
        clear_memory(device_list=self.device_list)

        enable_imatrix = False
        if not self.disable_opt_rtn:
            has_gguf_k = (
                any(fmt.is_gguf() and "k" in fmt.output_format for fmt in getattr(self, "formats", []))
                or self.super_bits is not None
            )
            if has_gguf_k:
                enable_imatrix = True
            elif self.data_type == "int" and self.sym:
                enable_imatrix = True
        if enable_imatrix:
            self._quant_rtn_with_imatrix(all_to_quantized_module_names)
        elif self.act_bits <= 8 and check_need_act_calibration(
            self.act_dynamic,
            self.act_data_type,
            self.act_bits,
            self.static_kv_dtype,
            self.static_attention_dtype,
        ):  # TODO, mixed datatype has bug
            hook_handles = self._register_act_max_hook(self.model)
            try:
                self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
            except torch.OutOfMemoryError:
                logger.warning("Fallback to CPU. Consider using more GPUs via `--device 0,1,2,3`.")
                self.model = self.model.to("cpu")
                clear_memory(device_list=self.device_list)
                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    import accelerate

                    accelerate.hooks.remove_hook_from_submodules(self.model)
                orig_device = self.device
                self.device = "cpu"
                self._quantize_via_rtn_blockwise(all_to_quantized_module_names)
                self.device = orig_device
            for handle in hook_handles:
                handle.remove()
        else:
            # By default, we go with layer-wise way if no replacement happened.
            # In RTN mode (iters == 0), force blockwise quantization to avoid
            # full-model materialization and linear CPU RAM growth.
            use_blockwise_quantization = global_state.replaced_module_count > 0
            if self.iters == 0 and not use_blockwise_quantization:
                logger.info(
                    "RTN mode detected (iters=0): force blockwise quantization to avoid "
                    "layer-wise full-model materialization."
                )
                use_blockwise_quantization = True
            tied_weights_keys = getattr(self.model, "_tied_weights_keys", [])
            if tied_weights_keys is None:
                tied_weights_keys = []
            if isinstance(tied_weights_keys, dict):
                tied_weights_values = list(tied_weights_keys.values())
            else:
                tied_weights_values = list(tied_weights_keys)
            tied_weights_layers = [".".join(val.split(".")[:-1]) for val in tied_weights_values]  # rm weight/bias
            # In fact, we should detect whether it is is_separate_lm_head, to simplify, we don't do it
            if hasattr(self, "formats") and self.formats[0].is_gguf():
                lm_head_name = get_lm_head_name(self.model)
                if lm_head_name is not None:
                    tied_weights_layers.append(lm_head_name)

            if use_blockwise_quantization:  # The ram usage is a little higher
                all_to_quantized_module_names = list(dict.fromkeys(all_to_quantized_module_names))

                all_blocks = self.quant_block_list or get_block_names(self.model)
                pbar = tqdm(range(sum(len(block) for block in all_blocks)))
                for block_names in all_blocks:
                    for block_name in block_names:
                        pbar.set_description(f"Quantizing {block_name}")
                        block = get_module(self.model, block_name)

                        materialize_model_(block)

                        for name, m in block.named_modules():
                            if hasattr(m, "global_name") and m.global_name in all_to_quantized_module_names:
                                self._quantize_layer_via_rtn(m.global_name, to_cpu=self.low_gpu_mem_usage)
                                all_to_quantized_module_names.remove(m.global_name)
                            elif (
                                not any(m.children())
                                and len(m.state_dict()) > 0
                                and m.global_name not in tied_weights_layers
                                and self.is_immediate_saving
                            ):
                                set_module(self.model, m.global_name, copy.deepcopy(m))
                                if self.is_immediate_saving:
                                    shard_writer(self, name=m.global_name)
                                    copied_m = get_module(self.model, m.global_name)
                                    copied_m.to("meta")
                                m.to("meta")
                        # Move remaining GPU tensors to CPU; offload to disk if low_cpu_mem_usage.
                        # This mirrors _quantize_via_rtn_blockwise's post-block cleanup.
                        if not self.is_immediate_saving:
                            mv_module_from_gpu(block)
                        else:
                            # Save once at block scope to capture tensors that are not saved
                            # in per-layer branch (e.g., custom module-level params/buffers).
                            shard_writer(self, name=block_name)
                            block.to("meta")
                        if self.low_cpu_mem_usage and not self.is_immediate_saving:
                            self._offloader(self.model, block_name)
                        clear_memory(device_list=self.device_list)
                        memory_monitor.log_summary()
                        pbar.update(1)
                cnt = 1
                for name in all_to_quantized_module_names:
                    logger.info(f"Quantizing remaining layer {name} on CPU.")
                    self._quantize_layer_via_rtn(name, to_cpu=True)
                    cnt += 1
                    if cnt % 10 == 0:
                        clear_memory(device_list=self.device_list)
                        memory_monitor.log_summary()
            else:
                materialize_model_(self.model)
                self.model.to("cpu")
                block_names_cnt = len(flatten_list(get_block_names(self.model, True)))
                clear_mem_freq = len(all_to_quantized_module_names) // block_names_cnt
                cnt = 0
                pbar = tqdm(all_to_quantized_module_names)

                for n, m in self.model.named_modules():
                    if hasattr(m, "global_name") and m.global_name in all_to_quantized_module_names:
                        pbar.set_description(f"Quantizing {m.global_name}")
                        self._quantize_layer_via_rtn(m.global_name)
                        cnt += 1
                        pbar.update()
                        if cnt % clear_mem_freq == 0:
                            clear_memory(device_list=self.device_list)
                            memory_monitor.log_summary()

                    elif (
                        not any(m.children())
                        and len(m.state_dict()) > 0
                        and n not in tied_weights_layers
                        and self.is_immediate_saving
                    ):
                        set_module(self.model, n, copy.deepcopy(m))
                        shard_writer(self, name=n)
                        m.to("meta")

        # Convert remaining fp8
        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device)
        if self.low_cpu_mem_usage:
            self._offloader.reload(self.model)
        if self.is_immediate_saving:
            shard_writer(self, is_finalize=True)

        self.quantized = True
        return self.model, self.layer_config

    def _quantize_via_rtn_blockwise(self, all_to_quantized_module_names: list[str]) -> None:
        """Quantize model layers block by block using cached inputs and imatrix.

        Args:
            all_to_quantized_module_names (list[str]): Names of layers to be quantized.
        """
        all_to_quantized_module_names = list(set(all_to_quantized_module_names))

        all_blocks = self.quant_block_list or get_block_names(self.model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        layer_names = self._get_quantized_layer_names_outside_blocks()
        if self.act_bits < 16 and (not self.act_dynamic or len(layer_names) > 0) or self.has_variable_block_shape:
            if len(layer_names) > 0:
                logger.warning(
                    "quantize layers outside blocks for static activation quantizaiton"
                    " will significantly increase calibration time"
                )
            all_inputs = self.try_cache_inter_data_gpucpu(to_cache_block_names, self.nsamples, layer_names)
        else:
            all_inputs = self.cache_inter_data(to_cache_block_names, self.nsamples)

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

            clear_memory(self.inputs, device_list=self.device_list)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.batch_size:
                self.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")
            tmp_dtype = self.amp_dtype if self.amp else torch.float32

            input_ids = to_device(inputs.pop("input_ids"), self.cache_device)
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]

            def process_input_others(input_others):

                input_others = to_device(input_others, self.cache_device)
                for key, val in input_others.items():
                    if isinstance(val, torch.Tensor) and val.dtype in (torch.float16, torch.bfloat16):
                        input_others[key] = val.to(tmp_dtype)
                    elif isinstance(val, list):
                        input_others[key] = [to_dtype(v, tmp_dtype) for v in val]
                return input_others

            input_others = inputs
            input_others = process_input_others(input_others)
            for block_name in block_names:
                if block_name in all_inputs.keys():
                    input_others = all_inputs[block_name]
                    input_others = process_input_others(input_others)
                    all_inputs.pop(block_name)
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model, block_name)
                materialize_model_(block)
                block.to("cpu")

                block = convert_module_to_hp_if_necessary(block, dtype=self.amp_dtype, device=self.device)
                update_block_global_scale_if_needed(block, self.data_type, self.group_size)
                self._register_act_max_hook(block)
                if (
                    is_auto_device_mapping(self.device_map)
                    and len(self.device_list) > 1
                    and not getattr(self, "is_diffusion", False)
                ):
                    set_auto_device_map_for_block_with_tuning(
                        block, self.device_map, input_ids, self.low_gpu_mem_usage, self.batch_size, self.device
                    )
                # Dispatch model if needed
                if len(self.device_list) > 1:
                    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                    for _, m in block.named_modules():
                        if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                            continue
                        hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                        add_hook_to_module(m, hook, True)
                else:
                    block = block.to(self.device)
                input_ids = self._get_block_outputs(
                    block,
                    input_ids,
                    input_others,
                    self.batch_size * self.infer_bs_coeff,
                    self.device,
                    self.cache_device,
                )

                if len(self.device_list) > 1:
                    accelerate.hooks.remove_hook_from_submodules(block)

                if is_nv_fp(self.act_data_type) or is_static_wfp8afp8(self):
                    # enable moe experts act_max automatic generation for Linear
                    set_amax_for_all_moe_layers(block, attr_name="act_max")
                # Normalize imatrix and quantize layers
                if self.low_gpu_mem_usage:
                    block.to("cpu")
                    clear_memory(device_list=self.device_list)

                for name, m in block.named_modules():
                    # fix issue: Ling-flash-2.0-q2_k_s fail infer on cuda but well on cpu
                    # https://huggingface.co/Intel/Ling-flash-2.0-gguf-q2ks-mixed-AutoRound/discussions/1
                    if hasattr(m, "imatrix"):
                        m.imatrix /= m.imatrix_cnt
                    if hasattr(m, "global_name") and m.global_name in all_to_quantized_module_names:
                        self._quantize_layer_via_rtn(m.global_name, to_cpu=self.low_gpu_mem_usage)
                        all_to_quantized_module_names.remove(m.global_name)

                mv_module_from_gpu(block)
                if self.low_cpu_mem_usage and not self.is_immediate_saving:
                    self._offloader(self.model, block_name)
                if block_name == block_names[-1]:
                    clear_memory(input_ids, device_list=self.device_list)
                else:
                    clear_memory(device_list=self.device_list)

                memory_monitor.log_summary()
                pbar.update(1)
        pbar.close()
        # Process remaining layers not in blocks
        for name in all_to_quantized_module_names:
            dtype = None
            if self.super_group_size is not None:
                dtype = torch.float32
            self._quantize_layer_via_rtn(name, dtype=dtype)
            # clear_memory(device_list=self.device_list)
        # if self.is_immediate_saving:
        #     shard_writer(self, is_finalize=True)

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
        is_gguf_format = (f := getattr(self, "formats", None)) is not None and len(f) > 0 and f[0].is_gguf()
        if not is_gguf_format:
            predefined_ignore_layers = get_predefined_ignore_layers(self.model)

            # Filter out ignore_layers that don't belong to any quantized block.
            if predefined_ignore_layers and self.quant_block_list:
                block_prefixes = [b for group in self.quant_block_list for b in group]
                predefined_ignore_layers = [
                    name
                    for name in predefined_ignore_layers
                    if any(name.startswith(prefix) for prefix in block_prefixes)
                ]

            # Collapse numbered layer names into regex patterns so that
            # extra_config gets one entry instead of N per-layer entries.
            predefined_ignore_layers = collapse_ignore_layers(predefined_ignore_layers)

            compressed_predefined_ignore_layers = compress_layer_names(predefined_ignore_layers)
            if predefined_ignore_layers:
                logger.info(f"Using predefined ignore_layers: {compressed_predefined_ignore_layers}")
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
                self.model,
                supported_types=self.supported_types,
                inner_supported_types=self.inner_supported_types,
                quant_lm_head=self.quant_lm_head,
                mllm=getattr(self, "mllm", False),
            )

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

    def _adjust_immediate_packing_and_saving(self):
        formats = getattr(self, "formats", [])
        if len(formats) == 1 and not formats[0].is_fake() and self.inplace:
            self.is_immediate_packing = True

        if self.has_qlayer_outside_block and self.iters != 0:
            self.is_immediate_packing = False

        if not ("causallm" in self.model.__class__.__name__.lower() and not self.mllm):
            # TODO For tied keys, there may some issues, we haven't not verified this
            tied_weight_keys = getattr(self.model, "_tied_weight_keys", {})
            if len(tied_weight_keys) > 1:
                self.is_immediate_saving = False
                if self.low_cpu_mem_usage:
                    logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                return
            if len(tied_weight_keys) == 1:
                key = tied_weight_keys.keys[0]
                if "lm_head" not in key:
                    self.is_immediate_saving = False
                    if self.low_cpu_mem_usage:
                        logger.warning("reset low_cpu_mem_usage to False due to tied weights")
                    return

        if self.low_cpu_mem_usage and self.is_immediate_packing:
            self.is_immediate_saving = True

        if self.low_cpu_mem_usage and not self.is_immediate_packing:
            logger.info(
                "`low_cpu_mem_usage` is only supported when `immediate_packing` is True. "
                "Setting `low_cpu_mem_usage` to False."
            )
            self.low_cpu_mem_usage = False
            self.is_immediate_saving = False

        if self.low_cpu_mem_usage and self.is_immediate_packing:
            if formats[0].is_gguf():
                logger.warning(
                    "`low_cpu_mem_usage` is not fully supported for gguf format. "
                    "Setting `low_cpu_mem_usage` to False."
                )
                self.low_cpu_mem_usage = False
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
                self.low_cpu_mem_usage = False
                self.is_immediate_saving = False

        if self.is_immediate_saving and not (
            "int" in self.data_type or is_nv_fp(self.data_type) or is_mx_fp(self.data_type)
        ):
            logger.warning("immediate_saving is only supported for int/nv_fp/mx_fp quantization, set to False")
            self.is_immediate_saving = False

        if self.orig_output_dir is None:
            self.is_immediate_saving = False

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """

        self._check_compatibility()
        formats = self.formats if hasattr(self, "formats") else None
        # It is best to modify the model structure in the quantize function and check the format,
        # because it may cause the gguf format to not be exported normally.
        self.model = update_module(
            self.model, formats=formats, trust_remote_code=self.trust_remote_code, cleanup_original=False
        )

        # Temporary names must be assigned after handle_moe_model;
        # placing them earlier would cause them to be removed when the module is replaced.
        for n, m in self.model.named_modules():
            m.global_name = n

        if not self.is_auto_scheme:
            enable_gguf_official_mixed = True
        else:
            enable_gguf_official_mixed = False

        self.configure_layer_config(enable_gguf_official_mixed=enable_gguf_official_mixed)

        if self.low_cpu_mem_usage:
            self._offloader.reset()

        def _should_disable_inplace_due_to_layers_outside_block() -> bool:
            return self.has_qlayer_outside_block and (self.iters != 0 or (self.iters == 0 and not self.disable_opt_rtn))

        # Disable inplace mode when there are quantized layers outside blocks
        # under specific iteration/optimization settings.
        if _should_disable_inplace_due_to_layers_outside_block():
            self.inplace = False
        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            # Determine if immediate packing is required
            self._adjust_immediate_packing_and_saving()

        if self.iters == 0:
            return self._quantize_rtn()

        if bool(self.quant_block_list):
            all_blocks = self.quant_block_list
        else:
            all_blocks = get_block_names(self.model)

        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model, self.layer_config

        if self.amp and self.model.dtype != self.amp_dtype:
            self.model = self.model.to(self.amp_dtype)

        layer_names = self._get_quantized_layer_names_outside_blocks()
        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")
        all_inputs = self.try_cache_inter_data_gpucpu(to_cache_block_names, self.nsamples, layer_names=layer_names)
        is_quantized_embedding = self._quantize_embedding_layer()
        clear_memory(device_list=self.device_list)
        all_q_inputs = None
        if is_quantized_embedding:
            all_inputs = copy.deepcopy(self.inputs)
            clear_memory(self.inputs, device_list=self.device_list)
            all_q_inputs = self.try_cache_inter_data_gpucpu(
                to_cache_block_names, self.nsamples, layer_names=layer_names
            )
        # Remove accelerate dispatch hooks before moving parameters.
        # hf_device_map is kept for reference but hooks are no longer needed.
        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model)
        self.model = mv_module_from_gpu(self.model)
        clear_memory(device_list=self.device_list)
        logger.info("caching done")
        if self.low_cpu_mem_usage:
            if self.is_model_patched and not self.is_immediate_saving:
                self._offloader(self.model, all_blocks, clear_memory=True, device_list=self.device_list)
                if not self._offloader.enabled:
                    self.low_cpu_mem_usage = False
            else:
                self.low_cpu_mem_usage = False
        if len(all_blocks) > 1:
            pbar = tqdm(range(0, sum([len(i) for i in all_blocks]), self.nblocks))
        else:
            pbar = tqdm(range(0, len(all_blocks[0]), self.nblocks))  # move the alg warning outside pbar

        start_time = time.time()
        for block_names in all_blocks:
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            q_inputs = None
            if all_q_inputs is not None:
                q_inputs = all_q_inputs[block_names[0]]
                all_q_inputs.pop(block_names[0])

            inputs, q_inputs = self._update_inputs(inputs, q_inputs)

            clear_memory(self.inputs, device_list=self.device_list)

            if "input_ids" in inputs.keys():
                total_samples = len(inputs["input_ids"])
                if total_samples < self.batch_size:
                    self.batch_size = total_samples
                    logger.warning(f"force the train batch size to {total_samples}")

            self._quantize_blocks(
                self.model,
                inputs,
                block_names,
                q_input=q_inputs if q_inputs is not None else None,
                nblocks=self.nblocks,
                device=self.device,
                pbar=pbar,
                input_others_extra_blocks=all_inputs,
            )
            if self.is_immediate_packing and len(self.formats) != 1:
                raise ValueError(
                    f"Expected exactly one packing format when 'immediate_packing' is True, "
                    f"but got {len(self.formats)} formats."
                )
        pbar.set_description("Quantizing done")
        pbar.close()
        if self.low_cpu_mem_usage:
            self._offloader.reload(self.model)
        self._quantize_layers(layer_names, all_inputs)

        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device, to_cpu=True)
        if self.is_immediate_saving:
            shard_writer(self, is_finalize=True)

        end_time = time.time()
        cost_time = end_time - start_time
        logger.info(f"quantization tuning time {cost_time}")

        # Dump a summary
        quantized_layers = []
        unquantized_layers = []
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(self.supported_types)):
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
            compressed_unquantized_layers = compress_layer_names(unquantized_layers)
            summary_info += f", unquantized layers: {compressed_unquantized_layers}"
        logger.info(summary_info)

        self.quantized = True
        return self.model, self.layer_config

    def _quantize_layers(self, layer_names: list, layer_inputs: dict) -> None:
        """Quantizes specified layers based on inputs and configuration.

        Args:
            layer_names (list): list of layer names to quantize.
            layer_inputs (dict): Dictionary mapping layer names to input data.

        Returns:
            None
        """
        # TODO currently we take all the layers outside blocks as post block layers which is not optimal
        # if there is no input for layer, we use rtn

        for layer_name in copy.deepcopy(layer_names):
            if layer_name not in layer_inputs:
                if self.act_bits < 16 and not self.act_dynamic:
                    if "lm_head" in layer_name:
                        logger.warning_once(
                            "Static activation quantization for lm_head is not fully supported yet. "
                            "If lm_head calibration inputs are missing, activation scale may fall back to unit scale "
                            "or quantization may be skipped."
                        )
                    # Activation quantization requires collected inputs
                    msg_prefix = (
                        f"Activation max hook for layer '{layer_name}' is unavailable due to "
                        f"insufficient collected inputs. "
                    )
                    if "fp8_e5m2" in self.act_data_type:
                        logger.warning(msg_prefix + "Please notes that unit scale is used for this layer.")
                    else:
                        logger.warning(
                            msg_prefix + "Static activation quantization is not supported or ineffective, "
                            "Skipping quantization for this layer."
                        )
                        layer_names.remove(layer_name)
                        continue
                logger.info(f"using rtn to quantize {layer_name}")
                from auto_round.data_type import QUANT_FUNC_WITH_DTYPE

                layer = get_module(self.model, layer_name)
                layer = layer.to(self.device)
                layer = convert_module_to_hp_if_necessary(layer, self.amp_dtype, self.device)
                set_module(self.model, layer_name, layer)

                wrapper_layer = WrapperLinear(
                    layer,
                    enable_round_tuning=False,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_torch_compile=self.enable_torch_compile,
                    device=self.device,
                    disable_opt_rtn=self.disable_opt_rtn,
                    iters=self.iters,
                )
                new_layer = wrapper_layer.unwrapper({})
                set_module(self.model, layer_name, new_layer)
                layer.cpu()
                layer_names.remove(layer_name)
        if len(layer_names) == 0:
            memory_monitor.update()
            memory_monitor.log_summary()
            return
        q_layer_inputs = None
        enable_quanted_input = self.enable_quanted_input
        has_gguf = False

        if hasattr(self, "formats"):
            has_gguf = any(format_.is_gguf() for format_ in self.formats)
        if has_gguf and self.is_immediate_packing:
            enable_quanted_input = False

        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1 and enable_quanted_input:
            dispatch_model(self.model, self.model.hf_device_map)

        if enable_quanted_input:
            logger.info("starting to cache layer inputs for %s, this may be quite slow ", layer_names)
            q_layer_inputs = self.try_cache_inter_data_gpucpu([], self.nsamples, layer_names=layer_names)
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.model
                )  # self.model.hf_device_map has not been changed
        if not self.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        clear_memory(device_list=self.device_list)
        quant_layer = self._quantize_layer
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.cache_device)
            q_layer_input = q_layer_inputs.get(layer_name, None) if q_layer_inputs is not None else None
            q_layer_input = to_device(q_layer_input, self.cache_device)
            quant_layer(layer_name, layer_input, q_layer_input, device=self.device)
            if self.is_immediate_packing:
                self._immediate_pack(layer_name)

            if self.is_immediate_saving:
                m = get_module(self.model, layer_name)
                shard_writer(self, m, name=layer_name, is_finalize=False)
            del layer_input
            clear_memory(q_layer_input, device_list=self.device_list)
            memory_monitor.log_summary()

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
                        _all_module_input[module._global_name].append((args, kwargs))
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
            except NotImplementedError as error:
                error_msg = str(error)
                # Raise NotImplementedError to fallback to CUDA device
                if "flash_attn::" in error_msg and "CPU" in error_msg:
                    raise NotImplementedError(
                        "Could not run 'flash_attn::_flash_attn_varlen_forward' with arguments from the 'CPU' backend."
                    )
                else:
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
        block_names = flatten_list(block_names)
        if is_quantized_input_module(self.model):
            layer_names = []
        if layer_names is None:
            layer_names = []
        self.blocks_requiring_input_ids = [data if isinstance(data, str) else data[0] for data in block_names]

        calibrate_on_cpu = False
        cannot_calibrate_on_cpu = False
        if self.low_gpu_mem_usage or (
            len(block_names) == 1
            and len(layer_names) == 0
            and not self.has_qlayer_outside_block
            and (last_cache_name is None or last_cache_name in block_names)
            and not getattr(self, "mllm", False)
        ):
            # low_gpu_mem_usage or calibrate only the embedding layer, which is also very fast on CPU
            calibrate_on_cpu = True
            try:
                all_inputs = self.cache_inter_data(
                    block_names, nsamples, layer_names=[], last_cache_name=last_cache_name
                )
            except NotImplementedError as error:
                error_msg = str(error)
                if "flash_attn::" in error_msg and "CPU" in error_msg:
                    cannot_calibrate_on_cpu = True  # fallback to GPU when flash attention is not supported on CPU
                else:
                    raise error

        if not calibrate_on_cpu or cannot_calibrate_on_cpu:
            try:
                if any(p.device.type == "meta" for p in self.model.parameters()):
                    materialize_model_(self.model)

                if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                    self.model = dispatch_model(self.model, device_map=self.model.hf_device_map)
                else:
                    # Change this if new device is supported
                    if str(self.model.device) == "cpu" and (not self.device.startswith("hpu")):
                        # type(self.model._no_split_modules) changes from list to set when transformers > 5.0
                        no_split_modules = list(getattr(self.model, "_no_split_modules", []))
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
                            elif isinstance(device, str):
                                device = 0
                            else:
                                raise ValueError(f"Unsupported device {device} in device_map: {self.device_map}")
                            if device not in max_memory:
                                # Skip devices that are not reported by accelerate's max_memory.
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
                        # Keep ngram_embeddings on CPU
                        has_ngram_embeddings, raw_ngram_embeddings = hook_ngram_embeddings_on_cpu(self.model)
                        new_max_memory = get_balanced_memory(
                            self.model,
                            max_memory=new_max_memory,
                            no_split_module_classes=no_split_modules,
                        )
                        if hasattr(self.model, "tie_weights") and callable(self.model.tie_weights):
                            self.model.tie_weights()
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
                            if has_ngram_embeddings:
                                self.model.model.ngram_embeddings = raw_ngram_embeddings
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

            except torch.OutOfMemoryError as e:
                if cannot_calibrate_on_cpu:
                    raise e
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.info("switch to cpu to cache block inputs")
                    self.cache_device = torch.device("cpu")
                    if self.has_qlayer_outside_block or self.__class__.__name__ == "AutoRoundMLLM":
                        logger.warning(
                            "we recommend using more GPUs in calibration."
                            " Otherwise, some layers may fall back to `rtn` mode, which can affect accuracy."
                        )
                    accelerate.hooks.remove_hook_from_submodules(self.model)
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
        block_names = flatten_list(block_names)
        self.to_cached_layers = block_names + layer_names

        tmp_dtype = None  # TODO delete this as most model is not fp32 now
        # There is a bug if the block name is not the first block
        if (len(block_names) > 1 or len(layer_names) > 0) and self.low_gpu_mem_usage:
            tmp_dtype = self.model.dtype
            if self.amp:
                if self.model.dtype != self.model.dtype:
                    self.model = self.model.to(torch.bfloat16)
            else:
                self.model = self.model.to(torch.float32)  # model on cpu

        self.last_cache_name = self._infer_last_cache_name(block_names, layer_names, last_cache_name)
        self._cache_target_set = set(self.to_cached_layers)
        self._cache_seen_targets = set()
        calib_bs = self.batch_size
        self.hook_handles = []
        self._replace_forward()
        try:
            self.calib(nsamples, calib_bs)
        finally:
            # Use finally to recover_forward and delattr in case of that
            # self.calib raises NotImplementedError, such as: flash_attn on CPU.
            self._recover_forward()
            for attr in ("last_cache_name", "_cache_target_set", "_cache_seen_targets", "to_cached_layers"):
                if hasattr(self, attr):
                    delattr(self, attr)
        res = self.inputs
        if tmp_dtype is not None:
            self.model = self.model.to(tmp_dtype)

        return res

    def _infer_last_cache_name(self, block_names, layer_names=None, requested_last_cache_name=None):
        """The latest required cache layer for early-stop forward.

        If there are multiple cache targets, return ``None`` and let runtime
        hooks stop only after all targets are observed in real forward execution.
        """
        if layer_names is None:
            layer_names = []

        if requested_last_cache_name is not None:
            return requested_last_cache_name

        cache_targets = list(block_names) + list(layer_names)
        if len(cache_targets) == 1:
            return cache_targets[0]

        # return None here to enable the logic in _should_stop_cache_forward
        return None

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
            if data_name in self.shared_cache_keys:
                return None
            if batch_size <= 1:
                return new_data
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
                    if (
                        self.has_variable_block_shape
                        and name not in self.blocks_requiring_input_ids
                        and key == "hidden_states"
                    ):
                        continue
                    if key not in self.inputs[name].keys():  # initialization
                        data = to_device(kwargs[key], device=torch.device("cpu"))
                        if data is None or key in self.shared_cache_keys:
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

    def _register_act_max_hook(self, model):

        def get_act_max_hook(module, input, output):
            if isinstance(input, (tuple, list)):
                input = input[0]
            if input.numel() == 0:
                return  # as no needs for act_max update
            input, _, _ = reshape_pad_tensor_by_group_size(input, self.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if is_nv_fp(self.act_data_type):  ## for nvfp per-tensor input_global_scale calculation usage
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                act_max = act_max.to(module.act_max.device)
                if is_nv_fp(self.act_data_type):  ## for nvfp per-tensor input_global_scale calculation usage
                    max_val = torch.max(act_max.max(), module.act_max.max())
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                else:
                    module.act_max = torch.max(act_max, module.act_max)

        hook_handles = []
        # for single layers out of blocks, like lm_head
        if isinstance(model, SUPPORTED_LAYER_TYPES):
            m = model
            if (
                hasattr(m, "act_dynamic")
                and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
                and check_to_quantized(m)
            ):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
            return hook_handles

        for n, m in model.named_modules():
            if (
                hasattr(m, "act_dynamic")
                and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
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
                act_bits = config.get("act_bits", 16)
                if (
                    config["bits"] <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(config)
                ):
                    hook = m.register_forward_hook(get_act_max_hook)
                    hook_handles.append(hook)
                    continue
        return hook_handles

    def _quantize_layer(
        self, layer_name: str, inputs: torch.Tensor, q_inputs: torch.Tensor = None, device: str = "cpu"
    ):
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

        if self.act_bits <= 8 and check_need_act_calibration(
            self.act_dynamic,
            self.act_data_type,
            self.act_bits,
            self.static_kv_dtype,
            self.static_attention_dtype,
        ):
            tmp_inputs = q_inputs if q_inputs is not None else inputs
            hook_handles = self._register_act_max_hook(layer)
            with torch.no_grad():
                for input in tmp_inputs:
                    layer(input)
            for handle in hook_handles:
                handle.remove()

        wrapper_linear = WrapperLinear(
            layer,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_torch_compile=self.enable_torch_compile,
            device=device,
            iters=self.iters,
        ).to(device)
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
            mv_module_from_gpu(layer)

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": minmax_lr}], lr=lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        nsamples = len(inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        gradient_accumulate_steps = self.batch_size  # Force to low gpu

        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        batch_size = 1  # Force to low gpu
        global_batch_size = self.batch_size * gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        if gradient_accumulate_steps != 1 and not self.attention_mask:
            whole_indices = torch.arange(global_batch_size)
            if q_inputs is not None:
                num_elm = self._get_current_num_elm(q_inputs, whole_indices)
            else:
                num_elm = self._get_current_num_elm(inputs, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)

        for i in range(self.iters):
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if self.attention_mask:
                num_elm = self._get_non_zero_cnt(self.attention_mask, global_indices)

            for tmp_step in range(gradient_accumulate_steps):
                indices = global_indices[tmp_step * batch_size : (tmp_step + 1) * batch_size]
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
                autocast_ctx = (
                    nullcontext()
                    if not self.amp
                    else autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype)
                )
                if self.attention_mask:
                    tmp_attention_mask = [self.attention_mask[i] for i in indices]
                    tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
                    tmp_attention_mask.unsqueeze_(-1)

                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            (output_q * tmp_attention_mask).to(torch.float32),
                            (current_output * tmp_attention_mask).to(torch.float32),
                        )

                else:
                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            output_q.to(torch.float32),
                            current_output.to(torch.float32),  # mul 1.0 will copy the output
                        )

                num_elm = 1 if num_elm <= 0 else num_elm
                total_loss += loss.item() / num_elm

                self._scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(wrapper_linear, self.cache_device)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(wrapper_linear, self.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        with torch.no_grad():
            unwrapper_layer(self.model, wrapper_linear, layer_name, best_params)
        mv_module_from_gpu(layer)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        logger.info(dump_info)

    def _get_current_output(self, output: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
        current_output = [output[x] for x in indices]
        current_output = torch.cat(current_output, dim=self.batch_dim)
        return current_output

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

    def _get_non_zero_cnt(self, tensor: list[torch.Tensor], indices: list[int]) -> int:
        current_tensors = [tensor[i] for i in indices]
        non_zero_cnt = 0
        for t in current_tensors:
            non_zero_cnt += torch.count_nonzero(t).item()
        return non_zero_cnt

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
        input_ids, input_others = self._preprocess_block_inputs(block_inputs, decoding_layer_first_input_name)
        return self._quantize_block(block, input_ids, input_others, q_input, device, auto_offload)

    def _get_loss(
        self,
        output_q: torch.Tensor,
        current_output: torch.Tensor,
        indices: torch.Tensor,
        mse_loss: Callable,
        device: Union[str, torch.device] = "cpu",
    ):
        autocast_ctx = (
            nullcontext() if not self.amp else autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype)
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

    def _quantize_block(
        self,
        block: torch.nn.Module,
        input_ids: Union[list[torch.Tensor], dict],
        input_others: dict,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload=True,
    ):
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
        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, self.amp_dtype, device)

        if auto_offload:
            # card_0_in_high_risk indicates that card_0 memory is already in high usage (90%) w/o any weights
            # loss_device is used to calculate loss on the second device if available and card_0_in_high_risk
            if (
                is_auto_device_mapping(self.device_map)
                and len(self.device_list) > 1
                and not getattr(self, "is_diffusion", False)
            ):
                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    block, self.device_map, input_ids, self.low_gpu_mem_usage, self.batch_size, device
                )
            else:
                block = block.to(device)
                card_0_in_high_risk, loss_device = False, device
        else:
            card_0_in_high_risk, loss_device = False, device

        if len(self.device_list) > 1 and auto_offload:
            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                hook = AlignDevicesHook(m.tuning_device, io_same_device=True)
                add_hook_to_module(m, hook, True)

        if q_input is None:
            hook_handles = self._register_act_max_hook(block)

            output = self._get_block_outputs(
                block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device, self.cache_device
            )

            for handle in hook_handles:
                handle.remove()
        else:
            output = self._get_block_outputs(
                block, input_ids, input_others, self.batch_size * self.infer_bs_coeff, device, self.cache_device
            )
            hook_handles = self._register_act_max_hook(block)
            if hook_handles:
                self._get_block_outputs(
                    block,
                    q_input if q_input is not None else input_ids,
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
                clear_memory(input_ids, device_list=self.device_list)
            else:
                clear_memory(device_list=self.device_list)
            input_ids = q_input

        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.enable_torch_compile,
            device=device,
            iters=self.iters,
        )
        # Call this before quantization and after applying the block wrapper.
        if is_nv_fp(self.data_type):  # enable qkv and moe structure global_scale fuse.
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

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)
        is_adam = "adam" in self.__class__.__name__.lower()

        extra_kwargs = {} if is_adam else {"momentum": self.momentum}

        if self.enable_minmax_tuning:
            params = [
                {"params": round_params},
                {"params": minmax_params, "lr": minmax_lr},
            ]
        else:
            params = round_params

        optimizer = self.optimizer(
            params,
            lr=lr,
            weight_decay=0,
            **extra_kwargs,
        )

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            unwrapper_block(block, {})
            mv_module_from_gpu(block)
            return output, output

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        if isinstance(input_ids, dict):  # input_ids of Flux is dict
            nsamples = len(input_ids["hidden_states"])
        else:
            nsamples = len(input_ids)
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
        global_batch_size = self.batch_size * self.gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # We assume the block input and output shape is same
        if self.gradient_accumulate_steps != 1 and not self.attention_mask:
            whole_indices = torch.arange(global_batch_size)
            num_elm = self._get_current_num_elm(input_ids, whole_indices)
        setup_ddp_if_needed_(self, block, self.device_list)
        index_sampler = IndexSampler(nsamples, global_batch_size)
        batch_size = self.batch_size
        for i in range(self.iters):
            if self.enable_alg_ext and self.data_type.endswith("dq"):
                for n, m in block.named_modules():
                    m.cur_iter = i
            total_loss = 0
            global_indices = index_sampler.next_batch()
            if self.attention_mask:
                num_elm = self._get_non_zero_cnt(self.attention_mask, global_indices)

            for tmp_step in range(self.gradient_accumulate_steps):
                indices = global_indices[tmp_step * batch_size : (tmp_step + 1) * batch_size]
                current_output = self._get_current_output(output, indices)
                current_output = to_device(current_output, loss_device)
                output_q = self._get_current_q_output(block, input_ids, input_others, indices, device, loss_device)
                loss = self._get_loss(output_q, current_output, indices, mse_loss, device)
                num_elm = 1 if num_elm <= 0 else num_elm
                total_loss += loss.item() / num_elm

                if self.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.5, device_list=self.device_list)

                self._scale_loss_and_backward(scaler, loss)

                if self.low_gpu_mem_usage and card_0_in_high_risk:
                    # clear memory to avoid OOM due to memory fragmentation
                    clear_memory_if_reached_threshold(threshold=0.8, device_list=self.device_list)

            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(block, self.cache_device)
                    # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)

                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(block, self.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if self.iters > 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
            )
        else:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                "layers in the block"
            )

        if self.low_gpu_mem_usage:
            clear_memory(device_list=self.device_list)  # clear cached memory during training
        if len(unquantized_layer_names) != 0:
            logger.info(f"Unquantized layers: {unquantized_layer_names}")
        with torch.no_grad():
            unwrapper_block(block, best_params)

        if is_nv_fp(self.act_data_type):
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        if self.enable_quanted_input:
            q_outputs = self._get_block_outputs(
                block,
                input_ids,
                input_others,
                self.batch_size * self.infer_bs_coeff,
                device,
                cache_device=self.cache_device,
            )

            if len(self.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)

            clear_memory(input_ids, device_list=self.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return q_outputs, output
        else:
            if len(self.device_list) > 1 and auto_offload:
                accelerate.hooks.remove_hook_from_submodules(block)
            if auto_offload:
                mv_module_from_gpu(block)
            clear_memory(input_ids, device_list=self.device_list)
            memory_info_summary = memory_monitor.get_summary()
            logger.infoclean(dump_info + "," + memory_info_summary)

            return None, output

    def _split_inputs(self, inputs: dict, first_input_name: str) -> tuple[torch.Tensor, dict]:
        input_ids = inputs.get(first_input_name, None)
        inputs.pop(first_input_name, None)
        input_others = inputs
        return input_ids, input_others

    def _preprocess_block_inputs(self, inputs, first_input_name="input_ids"):
        input_ids, input_others = self._split_inputs(inputs, first_input_name)
        clear_memory(device_list=self.device_list)
        tmp_dtype = self.amp_dtype if self.amp else torch.float32
        if input_ids is not None:
            input_ids = to_device(input_ids, self.cache_device)
            input_ids = to_dtype(input_ids, tmp_dtype)
        input_others = to_device(input_others, self.cache_device)
        # As in calibration phase, we may use bf16 for calibration due to low_gpu_memory usage

        for key in input_others.keys():
            if isinstance(input_others[key], torch.Tensor) and (
                input_others[key].dtype == torch.float16 or input_others[key].dtype == torch.bfloat16
            ):
                input_others[key] = input_others[key].to(tmp_dtype)
            elif isinstance(input_others[key], list):
                for i in range(len(input_others[key])):
                    to_dtype(input_others[key][i], tmp_dtype)
        return input_ids, input_others

    def _quantize_blocks(
        self,
        model: torch.nn.Module,
        inputs: dict,
        block_names: list,
        q_input: torch.Tensor = None,
        nblocks: int = 1,
        device: str = "cpu",
        pbar: tqdm = None,
        input_others_extra_blocks: dict = None,
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
        clear_memory(device_list=self.device_list)
        for n, m in model.named_parameters():
            m.requires_grad_(False)

        input_ids, input_others = self._preprocess_block_inputs(inputs)

        if pbar is None:
            pbar = tqdm(range(0, len(block_names), nblocks))

        for i in range(0, len(block_names), nblocks):
            if input_others_extra_blocks and block_names[i] in input_others_extra_blocks:
                input_others = input_others_extra_blocks[block_names[i]]
                _, input_others = self._preprocess_block_inputs(input_others)
                input_others_extra_blocks.pop(block_names[i])
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

            if self.low_cpu_mem_usage:
                if nblocks == 1:
                    self._offloader.reload(model, n)
                else:
                    self._offloader.reload(model, names)

            m.config = model.config if hasattr(model, "config") else None
            q_input, input_ids = self._quantize_block(
                m,
                input_ids,
                input_others,
                q_input=q_input,
                device=device,
            )
            if hasattr(model, "config"):
                del m.config
            if self.enable_torch_compile:
                torch._dynamo.reset()
            if self.is_immediate_packing:
                for n, tmp_m in m.named_modules():
                    if not (hasattr(tmp_m, "bits") and check_to_quantized(tmp_m)):
                        continue
                    self._immediate_pack(tmp_m.global_name)

            if self.is_immediate_saving:
                shard_writer(self, m, is_finalize=False)

            if self.low_cpu_mem_usage and not self.is_immediate_saving:
                if nblocks == 1:
                    self._offloader(model, n, overwrite=True)
                else:
                    for name in names:
                        self._offloader(model, name, overwrite=True)
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

        clear_memory(device_list=self.device_list)

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

            # to match the original name
            reverse_checkpoint_conversion_mapping = get_reverse_checkpoint_conversion_mapping(self.model)

            if isinstance(serialization_dict["to_quant_block_names"], str):
                serialization_dict["to_quant_block_names"] = revert_checkpoint_conversion_mapping(
                    serialization_dict["to_quant_block_names"], reverse_checkpoint_conversion_mapping
                )

            elif isinstance(serialization_dict["to_quant_block_names"], list):
                for idx in range(len(serialization_dict["to_quant_block_names"])):
                    serialization_dict["to_quant_block_names"][idx] = revert_checkpoint_conversion_mapping(
                        serialization_dict["to_quant_block_names"][idx], reverse_checkpoint_conversion_mapping
                    )

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
            if len(folders) == 1:
                folders = folders[0]
            return compressed_model, folders
        else:
            return compressed_model

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
                logger.warning_once(f"could not find layer {key} in the model, skipping")
                continue
            if type(layer) in self.supported_types and check_to_quantized(self.layer_config[key]):
                layer_names.append(key)

        return layer_names

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
            # Shared cache keys (e.g. position_embeddings, position_ids, cache_position) are stored
            # directly as-is (not wrapped in a per-sample list) when batch_size > 1.  Indexing such
            # values by sample index would incorrectly decompose them (e.g. (cos, sin)[0] == cos).
            # Always pass them through unchanged.
            if key in share_cache_keys or isinstance(input_others[key], (str, bool, type(None))):
                current_input_others[key] = input_others[key]
            elif input_others[key] is not None:
                current_input_others[key] = [input_others[key][i] for i in indices]
                if len(indices) == 1:
                    current_input_others[key] = current_input_others[key][0]
                else:
                    try:
                        current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                    except TypeError as err:
                        logger.warning_once("Please check the model cache inputs or try setting batch_size to 1.")
            else:
                current_input_others[key] = None

        return current_input_ids, current_input_others


class LLMCompressor(BaseCompressor):
    pass
