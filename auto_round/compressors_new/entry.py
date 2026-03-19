# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Union

import torch

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization.auto_round.config import AutoRoundConfig
from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.compressors_new.calib import CalibCompressor, CalibratedRTNCompressor
from auto_round.compressors_new.utils import check_need_act_calibration
from auto_round.compressors_new.zero_shot import ZeroShotCompressor
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


def is_weight_scheme(scheme):
    if isinstance(scheme, str):
        return scheme.upper().startswith("W")
    if isinstance(scheme, dict):
        return all(isinstance(s, str) and s.upper().startswith("W") for s in scheme.values())
    if isinstance(scheme, AutoScheme):
        opts = scheme.options
        if isinstance(opts, (list, tuple)):
            return all(isinstance(s, str) and s.upper().startswith("W") for s in opts)
        if isinstance(opts, str):
            return opts.upper().startswith("W")
    return False


def detect_model_type(model):
    """Detect the type of model (LLM, MLLM, or Diffusion).

    Args:
        model: Model instance or model path string

    Returns:
        str: "mllm", "diffusion", or "llm"
    """
    from auto_round.utils import is_diffusion_model, is_mllm_model

    # Check if it's a diffusion model first (more specific)
    if is_diffusion_model(model):
        return "diffusion"

    # Check if it's an MLLM
    if is_mllm_model(model):
        return "mllm"

    # Default to standard LLM
    return "llm"


class Compressor(object):
    SKIP_ARGS = ("local_args", "kwargs", "cls", "config")

    def __new__(
        cls,
        config: Union[AlgConfig, list[AlgConfig]],
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform="hf",
        format=None,
        **kwargs,
    ):
        # using different compressor base on AlgConfigs
        local_args = {k: v for k, v in locals().items() if k not in cls.SKIP_ARGS}

        # Detect model type to determine if we need special compressor
        model_type = detect_model_type(model)

        if isinstance(config, AutoRoundConfig):
            # For AutoRound, we need calibration-based compression
            # Dynamically create combined class using Mixin pattern
            if model_type == "mllm":
                from auto_round.compressors_new.mllm_mixin import MLLMMixin

                # Create dynamic class: MLLMMixin + CalibCompressor
                class MLLMCalibCompressor(MLLMMixin, CalibCompressor):
                    """MLLM model with AutoRound calibration compression"""

                    pass

                return MLLMCalibCompressor(config, **local_args, **kwargs)
            elif model_type == "diffusion":
                from auto_round.compressors_new.diffusion_mixin import DiffusionMixin

                # Create dynamic class: DiffusionMixin + CalibCompressor
                class DiffusionCalibCompressor(DiffusionMixin, CalibCompressor):
                    """Diffusion model with AutoRound calibration compression"""

                    pass

                return DiffusionCalibCompressor(config, **local_args, **kwargs)
            else:
                return CalibCompressor(config, **local_args, **kwargs)

        elif isinstance(config, RTNConfig):
            enable_imatrix = False
            disable_opt_rtn = getattr(config, "disable_opt_rtn", False)
            if not disable_opt_rtn:
                has_gguf_k = "gguf" in format.lower() and "_k" in format.lower() if format else False
                if has_gguf_k:
                    enable_imatrix = True
                else:
                    sym = getattr(config, "sym", True)
                    if sym is not None and sym is False:
                        enable_imatrix = False
                    elif getattr(config, "data_type", "") == "int":
                        enable_imatrix = True
                    elif is_weight_scheme(config.scheme):
                        enable_imatrix = True

            needs_act_calib = getattr(config, "is_act_quantize", False) and check_need_act_calibration(
                getattr(config, "act_dynamic", None),
                getattr(config, "act_data_type", None),
                getattr(config, "act_bits", 16),
                static_kv_dtype=kwargs.get("static_kv_dtype"),
                static_attention_dtype=kwargs.get("static_attention_dtype"),
            )

            if enable_imatrix or needs_act_calib:
                config._alg_cls = "OptimizedRTNQuantizer"
                # For RTN with calibration data, dynamically combine with model-specific Mixin
                if model_type == "mllm":
                    from auto_round.compressors_new.mllm_mixin import MLLMMixin

                    class MLLMCalibratedRTNCompressor(MLLMMixin, CalibratedRTNCompressor):
                        """MLLM model with calibrated RTN compression"""

                        pass

                    return MLLMCalibratedRTNCompressor(config, **local_args, **kwargs)
                elif model_type == "diffusion":
                    from auto_round.compressors_new.diffusion_mixin import DiffusionMixin

                    class DiffusionCalibratedRTNCompressor(DiffusionMixin, CalibratedRTNCompressor):
                        """Diffusion model with calibrated RTN compression"""

                        pass

                    return DiffusionCalibratedRTNCompressor(config, **local_args, **kwargs)
                else:
                    return CalibratedRTNCompressor(config, **local_args, **kwargs)
            else:
                config._alg_cls = "RTNQuantizer"
                # Zero-shot RTN: no calibration data needed
                if model_type == "mllm":
                    from auto_round.compressors_new.mllm_mixin import MLLMMixin

                    class MLLMZeroShotCompressor(MLLMMixin, ZeroShotCompressor):
                        """MLLM model with zero-shot RTN compression"""

                        pass

                    return MLLMZeroShotCompressor(config, **local_args, **kwargs)
                elif model_type == "diffusion":
                    from auto_round.compressors_new.diffusion_mixin import DiffusionMixin

                    class DiffusionZeroShotCompressor(DiffusionMixin, ZeroShotCompressor):
                        """Diffusion model with zero-shot RTN compression"""

                        pass

                    return DiffusionZeroShotCompressor(config, **local_args, **kwargs)
                else:
                    return ZeroShotCompressor(config, **local_args, **kwargs)


class AutoRound:
    """AutoRound wrapper class for backward compatibility.

    This class provides the same API as the old AutoRound class but internally
    uses the new Compressor architecture with Mixin pattern.

    Args:
        model: Model object or model name to load
        tokenizer: Tokenizer for text processing
        platform: Platform to download model ("hf" or "model_scope")
        scheme: Quantization scheme (str, dict, or QuantizationScheme)
        layer_config: Layer-wise quantization config
        dataset: Calibration data
        iters: Optimization iterations
        seqlen: Calibration sequence length
        nsamples: Number of calibration samples
        batch_size: Calibration batch size
        gradient_accumulate_steps: Gradient accumulation steps
        low_gpu_mem_usage: Lower GPU memory mode
        device_map: Device map for each module
        enable_torch_compile: Enable torch.compile
        seed: Random seed
        low_cpu_mem_usage: Lower CPU memory mode
        **kwargs: Additional arguments (bits, group_size, sym, etc.)

    Example:
        >>> # Old API - still works
        >>> from auto_round.compressors_new.entry import AutoRound
        >>> autoround = AutoRound(
        ...     model="/models/opt-125m",
        ...     bits=4,
        ...     group_size=128,
        ...     iters=200,
        ... )
        >>> quantized_model, layer_config = autoround.quantize()
    """

    SKIP_ARGS = ("local_args", "kwargs", "cls", "config")

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

    @staticmethod
    def _pop_config_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract old-API config kwargs and split them by config type."""
        common_keys = (
            "ignore_layers",
            "quant_lm_head",
            "scale_dtype",
            "super_bits",
            "super_group_size",
            "to_quant_block_names",
        )
        auto_round_only_keys = (
            "nblocks",
            "enable_alg_ext",
            "lr_scheduler",
            "not_use_best_mse",
            "dynamic_max_gap",
            "optimizer",
            "enable_adam",
            "momentum",
        )
        common_kwargs = {}
        auto_round_kwargs = {}
        for key in common_keys:
            if key in kwargs:
                common_kwargs[key] = kwargs.pop(key)
        for key in auto_round_only_keys:
            if key in kwargs:
                auto_round_kwargs[key] = kwargs.pop(key)
        return common_kwargs, auto_round_kwargs

    def __new__(
        cls,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        platform: str = "hf",
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
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ):
        """Create AutoRound instance using new Compressor architecture.

        This method translates old AutoRound API to new Compressor API.
        """
        from auto_round.utils import is_diffusion_model, is_mllm_model

        common_config_kwargs, auto_round_config_kwargs = cls._pop_config_kwargs(kwargs)

        # Extract quantization parameters from kwargs or use defaults
        bits = kwargs.pop("bits", None)
        group_size = kwargs.pop("group_size", None)
        sym = kwargs.pop("sym", None)
        data_type = kwargs.pop("data_type", None)
        act_bits = kwargs.pop("act_bits", None)
        act_group_size = kwargs.pop("act_group_size", None)
        act_sym = kwargs.pop("act_sym", None)
        act_data_type = kwargs.pop("act_data_type", None)
        act_dynamic = kwargs.pop("act_dynamic", None)

        # Decide which algorithm to use
        if iters == 0:
            # RTN mode
            disable_opt_rtn = kwargs.pop("disable_opt_rtn", None)
            config = RTNConfig(
                scheme=scheme,
                layer_config=layer_config,
                bits=bits,
                group_size=group_size,
                sym=sym,
                data_type=data_type,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym,
                act_data_type=act_data_type,
                act_dynamic=act_dynamic,
                disable_opt_rtn=disable_opt_rtn,
                # for optRTN
                seqlen=seqlen,
                nsamples=nsamples,
                batch_size=batch_size,
                **common_config_kwargs,
            )
        else:
            # AutoRound mode
            lr = kwargs.pop("lr", None)
            minmax_lr = kwargs.pop("minmax_lr", None)
            enable_minmax_tuning = kwargs.pop("enable_minmax_tuning", True)
            enable_norm_bias_tuning = kwargs.pop("enable_norm_bias_tuning", False)
            enable_quanted_input = kwargs.pop("enable_quanted_input", True)

            config = AutoRoundConfig(
                scheme=scheme,
                layer_config=layer_config,
                iters=iters,
                nsamples=nsamples,
                seqlen=seqlen,
                batch_size=batch_size,
                gradient_accumulate_steps=gradient_accumulate_steps,
                bits=bits,
                group_size=group_size,
                sym=sym,
                data_type=data_type,
                act_bits=act_bits,
                act_group_size=act_group_size,
                act_sym=act_sym,
                act_data_type=act_data_type,
                act_dynamic=act_dynamic,
                lr=lr,
                minmax_lr=minmax_lr,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_norm_bias_tuning=enable_norm_bias_tuning,
                enable_quanted_input=enable_quanted_input,
                **common_config_kwargs,
                **auto_round_config_kwargs,
            )

        # Determine output format if specified
        format = kwargs.pop("format", None)

        # Extract MLLM-specific parameters
        processor = kwargs.pop("processor", None)
        image_processor = kwargs.pop("image_processor", None)
        template = kwargs.pop("template", None)
        extra_data_dir = kwargs.pop("extra_data_dir", None)
        quant_nontext_module = kwargs.pop("quant_nontext_module", False)

        # Extract Diffusion-specific parameters
        guidance_scale = kwargs.pop("guidance_scale", 7.5)
        num_inference_steps = kwargs.pop("num_inference_steps", 50)
        generator_seed = kwargs.pop("generator_seed", None)

        # Check model type for logging
        if is_mllm_model(model, platform=platform):
            logger.info("Using MLLM mode for multimodal model (new architecture).")
        elif is_diffusion_model(model):
            logger.info("Using Diffusion mode for diffusion model (new architecture).")
        else:
            logger.info("Using LLM mode (new architecture).")

        # Create Compressor instance using new architecture
        compressor = Compressor(
            config=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            dataset=dataset,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            # MLLM parameters
            processor=processor,
            image_processor=image_processor,
            template=template,
            extra_data_dir=extra_data_dir,
            quant_nontext_module=quant_nontext_module,
            # Diffusion parameters
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator_seed=generator_seed,
            # Pass remaining kwargs
            **kwargs,
        )

        return compressor
