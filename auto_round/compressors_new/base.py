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
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union

import torch
from transformers import set_seed

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization import BaseQuantizers, QuantizationConfig
from auto_round.algorithms.rotation import (
    BaseRotationConfig,
    apply_rotation,
    check_supported_schemes,
)
from auto_round.compressors_new.shard_writer import ShardWriter
from auto_round.compressors_new.utils import _get_save_folder_name, block_forward, set_layer_config
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
from auto_round.special_model_handler import get_predefined_ignore_layers
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    TORCH_VERSION_AT_LEAST_2_6,
    compile_func,
    compress_layer_names,
    convert_dtype_str2torch,
    extract_block_names_to_str,
    find_matching_blocks,
    get_block_names,
    is_debug_mode,
    is_hpex_available,
    is_quantized_input_module,
    memory_monitor,
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
    rotation_configs: Optional[list[dict[str, Any]]] = None


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
        **kwargs,
    ):
        self.quantize_config = None
        self.rotation_configs: list[BaseRotationConfig] = []
        _config_list = config if isinstance(config, list) else [config]
        for _cfg in _config_list:
            if isinstance(_cfg, QuantizationConfig):
                self.quantize_config = _cfg
            elif isinstance(_cfg, BaseRotationConfig):
                self.rotation_configs.append(_cfg)
        assert self.quantize_config is not None, "QuantizationConfig is required for Compressor"

        # Scheme is passed directly to the compressor, not stored in QuantizationConfig.
        self.scheme = scheme

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
            logger.warning(
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

        # Reset both context singletons before creating fresh instances so that
        # consecutive AutoRound creations don't inherit stale config from earlier ones.
        CompressContext.reset_context()
        ModelContext.reset_context()
        # Alternatively, you can use CompressContext.create_context
        self.compress_context = CompressContext(
            low_cpu_mem_usage,
            low_gpu_mem_usage,
            device_map,
            enable_torch_compile,
            is_immediate_packing=self.is_immediate_packing,
            is_immediate_saving=self.is_immediate_saving,
            formats=self.formats,
            static_kv_dtype=self.static_kv_dtype,
            static_attention_dtype=self.static_attention_dtype,
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
            formats=self.formats,
            is_act_quantize=self.quantize_config.is_act_quantize,
            quant_nontext_module=quant_nontext_module,
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
            logger.info("AutoScheme is not yet supported for multimodal LLMs.")
            sys.exit(-1)

        if is_quantized_input_module(self.model_context.model):
            logger.info("AutoScheme does not currently support quantized input models (e.g., FP8).")
            sys.exit(-1)

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
        )
        layer_config = self.scheme_generator.get_layer_config()
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
            logger.warning("reset enable_torch_compile to `False` as fp8 is enabled")
        # TODO: fix https://github.com/intel/auto-round/issues/1109
        if self.enable_torch_compile and is_raw_nv_fp:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as nvfp4 is enabled")
        super_group_size = getattr(cfg, "super_group_size", None)
        enable_alg_ext = getattr(cfg, "enable_alg_ext", False)
        if self.enable_torch_compile and super_group_size is not None and enable_alg_ext:
            self.enable_torch_compile = False
            logger.warning("reset enable_torch_compile to `False` as super_group_size is set for algorithm extension")

    def _get_calibration_dataset(self) -> str:
        """Resolve calibration dataset: self.dataset > AutoScheme.dataset > default."""
        dataset = self.__dict__.get("dataset", None)
        if dataset:
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

        The five phases in order:
          1. Scheme resolution  – pure config, no model structure needed.
          2. Format resolution  – needs data_type/bits from phase 1.
          3. Model patching     – needs formats from phase 2.
          4. Layer-config build – needs patched model from phase 3.
          5. Hardware setup     – device map, torch.compile, offloading.
        """
        if self._post_init_done:
            return

        # ── Phase 1: resolve scheme ───────────────────────────────────────────
        # Initialize scheme state from quantize_config before resolving.
        cfg = self.quantize_config
        self.scale_dtype = cfg.scale_dtype
        self.layer_config = cfg.layer_config
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

        # Create the quantizer now that the config holds resolved values.
        self.quantizer = BaseQuantizers.from_config(self.quantize_config)
        self.quantizer.model_context = self.model_context
        self.quantizer.compress_context = self.compress_context
        self.quantizer.model = self.model_context.model
        self.quantizer.scale_dtype = self.scale_dtype
        self.wrapper_block = wrapper_block

        # ── Phase 2: resolve output format ───────────────────────────────────
        # get_formats() inspects data_type / bits etc. that were just resolved.
        if isinstance(self.formats, str):
            self.formats = get_formats(self.formats, self)
        if self.formats is not None:
            self.compress_context.formats = self.formats
            ShardWriter.reset()
            self.shard_writer = ShardWriter(self.model_context.model, bits=8)

        # ── Phase 2b: propagate GGUF-adjusted attrs back to quantizer ────────
        # gguf_args_check (called inside get_formats) may have overridden
        # bits / sym / data_type / super_bits / super_group_size / group_size
        # on *this* BaseCompressor object.  The quantizer stored its own copies
        # from Phase 1 (resolve_scheme), so we must sync them now, before
        # _scheme_post_init() builds the layer_config in Phase 4.
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
            if _attr in self.__dict__ and hasattr(self.quantizer, _attr):
                if _attr not in ("scale_dtype", "act_bits") and getattr(self.quantizer, _attr) != self.__dict__[_attr]:
                    _any_gguf_attr_changed = True
                setattr(self.quantizer, _attr, self.__dict__[_attr])
        # If gguf_args_check changed scheme attrs, rebuild the scheme on both
        # the compressor (SchemeMixin) and the quantizer so that
        # configure_layer_config() and set_layer_config() use the correct
        # default_dict and gguf_name.
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

        # ── Phase 2c: merge layer_config set by GGUFFormat._mixed handling ───
        # Inner GGUFFormat("q2_k_mixed", ar) calls _handle_special_schemes and
        # stores the result directly in ar.__dict__["layer_config"].
        # self.layer_config is already the authoritative owner of this attr, so
        # just merge any GGUFFormat-supplied per-layer entries that may have
        # been set before Phase 1 (during get_formats → gguf_args_check).
        _gguf_layer_cfg = {
            k: v
            for k, v in (self.__dict__.get("layer_config") or {}).items()
            if k not in (self.quantize_config.layer_config or {})
        }
        if _gguf_layer_cfg:
            if self.layer_config is None:
                self.layer_config = {}
            for _lname, _lval in _gguf_layer_cfg.items():
                self.layer_config.setdefault(_lname, _lval)

        # ── Phase 2d: apply rotation transforms ──────────────────────────────
        if self.rotation_configs:
            check_supported_schemes(self.scheme)
            need_calibration = self.quantize_config.iters > 0
            for rotation_cfg in self.rotation_configs:
                self.model_context.model = apply_rotation(
                    self.model_context.model,
                    rotation_cfg,
                    need_calibration=need_calibration,
                )

        # ── Phase 3: patch model structure ───────────────────────────────────
        # update_module() may replace layers (e.g. MoE expert merging); must
        # happen before configure_layer_config() so it sees the final topology.
        self.model_context.apply_patches(self.formats)

        # ── Phase 4: build layer config ──────────────────────────────────────
        # configure_layer_config() walks the patched model; _gen_auto_scheme()
        # (AutoScheme path) runs delta-loss forward+backward passes.
        # Both methods now live in BaseCompressor and operate on self directly.
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

        # ── Phase 5: hardware / compile setup ────────────────────────────────
        set_non_auto_device_map(self.model_context.model, self.compress_context.device_map)
        # Re-evaluate torch.compile eligibility now that data_type is resolved.
        self._adjust_torch_compile(self.enable_torch_compile)
        self.compress_context.enable_torch_compile = self.enable_torch_compile
        self.block_forward = (
            compile_func(block_forward, self.compress_context.device) if self.enable_torch_compile else block_forward
        )
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reset()

        # Disable inplace when quantized layers live outside transformer blocks.
        if self.has_qlayer_outside_block and self.need_calib:
            self.inplace = False

        if not hasattr(self, "formats"):
            logger.warning("this API is deprecated, please use `quantize_and_save` instead")
        else:
            self._adjust_immediate_packing_and_saving()

        self._post_init_done = True

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
            self.is_immediate_packing = True

        if self.has_qlayer_outside_block and self.need_calib:
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
            elif self.has_qlayer_outside_block and not isinstance(self.quantize_config, RTNConfig):
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
                    f" Please use Compressor(format='{format}' instead)."
                )
                self.formats = get_formats(format, self)
                self.compress_context.formats = self.formats

        if not self.model_context.quantized:
            logger.warning("please run autoround.quantize first")
            return
        folders = []
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
        if format and self.formats is None:
            logger.warning(
                f"quantize_and_save with format is deprecated and will be deleted in auto_round version 1.0."
                f" Please use Compressor(format='{format}' instead)."
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

        # Save the quantized model in the specified format_list
        model, folders = self.save_quantized(output_dir, inplace=inplace, return_folders=True, **kwargs)
        memory_monitor.log_summary()

        return model, folders
