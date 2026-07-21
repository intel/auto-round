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
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch

from auto_round.compressors.entry_contract import ENTRY_ALLOWED_KWARGS
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme

if TYPE_CHECKING:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.compressors.base import BaseCompressor


_COMPAT_ONLY_KWARGS = {
    # scheme fields (translated into the alg config's scheme overrides)
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
    # tuning hyperparameters / algorithm selection
    "algorithm",
    "lr",
    "minmax_lr",
    "enable_minmax_tuning",
    "enable_norm_bias_tuning",
    "enable_quanted_input",
    "enable_opt_rtn",
    "rotation_config",
    "duo_scaling",
    "n_grid",
    "mappings",
    "optimizer",
    "lr_scheduler",
    "not_use_best_mse",
    "dynamic_max_gap",
    "momentum",
    # deprecated alias
    "device",
}

_COMPAT_KWARGS = ENTRY_ALLOWED_KWARGS | _COMPAT_ONLY_KWARGS


def filter_supported_compat_kwargs(kwargs: dict) -> dict:
    supported = {}
    unknown = []
    for key, value in kwargs.items():
        if key in _COMPAT_KWARGS:
            supported[key] = value
        else:
            unknown.append(key)
    if unknown:
        logger.warning_once(
            "AutoRound compatibility path received unsupported kwargs %s. They will be ignored.",
            ", ".join(sorted(unknown)),
        )
    return supported


# ---------------------------------------------------------------------------
# Backward-compatible (old-API) translation
# ---------------------------------------------------------------------------
# These functions translate the legacy ``bits=/act_bits=/algorithm=/iters=`` API
# into algorithm-config objects and forward to ``PipelineCompressor``. They back
# ``AutoRound``'s compatibility path (the branch taken when the caller does not
# pass ``alg_configs`` directly); there is no separate ``AutoRoundCompatible`` class.


def _compat_pop_config_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Extract old-API config kwargs and split them by config type."""
    common_keys = ("super_bits", "super_group_size")
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


def _compat_pop_compressor_only_kwargs(kwargs: dict) -> dict:
    return {
        "scale_dtype": kwargs.pop("scale_dtype", None),
        "ignore_layers": kwargs.pop("ignore_layers", ""),
        "quant_lm_head": kwargs.pop("quant_lm_head", False),
        "to_quant_block_names": kwargs.pop("to_quant_block_names", None),
    }


def _compat_resolve_algorithm(algorithm, iters) -> str:
    if algorithm and algorithm.lower() == "awq":
        return "awq"
    if (algorithm and algorithm.lower() == "rtn") or iters == 0:
        return "rtn"
    return "signround"


def _compat_pop_shared_quant_kwargs(kwargs: dict) -> dict:
    return {
        "bits": kwargs.pop("bits", None),
        "group_size": kwargs.pop("group_size", None),
        "sym": kwargs.pop("sym", None),
        "data_type": kwargs.pop("data_type", None),
        "act_bits": kwargs.pop("act_bits", None),
        "act_group_size": kwargs.pop("act_group_size", None),
        "act_sym": kwargs.pop("act_sym", None),
        "act_data_type": kwargs.pop("act_data_type", None),
        "act_dynamic": kwargs.pop("act_dynamic", None),
    }


def _compat_build_awq_config(shared_quant_kwargs, *, seqlen, nsamples, batch_size, kwargs, common_config_kwargs):
    from auto_round.algorithms.transforms.awq.config import AWQConfig

    return AWQConfig(
        **shared_quant_kwargs,
        duo_scaling=kwargs.pop("duo_scaling", True),
        n_grid=kwargs.pop("n_grid", 20),
        seqlen=seqlen,
        nsamples=nsamples,
        batch_size=batch_size,
        mappings=kwargs.pop("mappings", None),
        **common_config_kwargs,
    )


def _compat_build_rtn_config(shared_quant_kwargs, *, kwargs, common_config_kwargs):
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.registry import normalize_algorithm_config

    cfg = RTNConfig(
        **shared_quant_kwargs,
        disable_opt_rtn=kwargs.pop("disable_opt_rtn", None),
        enable_opt_rtn=kwargs.pop("enable_opt_rtn", None),
        **common_config_kwargs,
    )
    return normalize_algorithm_config(cfg)


def _compat_build_signround_config(
    shared_quant_kwargs, *, iters, gradient_accumulate_steps, kwargs, common_config_kwargs, auto_round_config_kwargs
):
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.registry import normalize_algorithm_config

    cfg = SignRoundConfig(
        **shared_quant_kwargs,
        iters=iters,
        gradient_accumulate_steps=gradient_accumulate_steps,
        lr=kwargs.pop("lr", None),
        minmax_lr=kwargs.pop("minmax_lr", None),
        enable_minmax_tuning=kwargs.pop("enable_minmax_tuning", True),
        enable_norm_bias_tuning=kwargs.pop("enable_norm_bias_tuning", False),
        enable_quanted_input=kwargs.pop("enable_quanted_input", True),
        **common_config_kwargs,
        **auto_round_config_kwargs,
    )
    return normalize_algorithm_config(cfg)


def _compat_build_alg_config(
    *,
    algorithm,
    iters,
    gradient_accumulate_steps,
    seqlen,
    nsamples,
    batch_size,
    kwargs,
    common_config_kwargs,
    auto_round_config_kwargs,
):
    alg_name = _compat_resolve_algorithm(algorithm, iters)
    shared_quant_kwargs = _compat_pop_shared_quant_kwargs(kwargs)

    if alg_name == "awq":
        return _compat_build_awq_config(
            shared_quant_kwargs,
            seqlen=seqlen,
            nsamples=nsamples,
            batch_size=batch_size,
            kwargs=kwargs,
            common_config_kwargs=common_config_kwargs,
        )
    if alg_name == "rtn":
        return _compat_build_rtn_config(
            shared_quant_kwargs,
            kwargs=kwargs,
            common_config_kwargs=common_config_kwargs,
        )
    return _compat_build_signround_config(
        shared_quant_kwargs,
        iters=iters,
        gradient_accumulate_steps=gradient_accumulate_steps,
        kwargs=kwargs,
        common_config_kwargs=common_config_kwargs,
        auto_round_config_kwargs=auto_round_config_kwargs,
    )


def _compat_build_forward_kwargs(kwargs: dict) -> dict:
    format_name = kwargs.pop("format", None)
    rotation_config = kwargs.pop("rotation_config", None)
    mllm_kwargs = {
        "processor": kwargs.pop("processor", None),
        "image_processor": kwargs.pop("image_processor", None),
        "template": kwargs.pop("template", None),
        "extra_data_dir": kwargs.pop("extra_data_dir", None),
        "quant_nontext_module": kwargs.pop("quant_nontext_module", False),
    }
    diffusion_kwargs = {
        "guidance_scale": kwargs.pop("guidance_scale", 7.5),
        "num_inference_steps": kwargs.pop("num_inference_steps", 50),
        "generator_seed": kwargs.pop("generator_seed", None),
    }
    return {
        "format": format_name,
        "rotation_config": rotation_config,
        **mllm_kwargs,
        **diffusion_kwargs,
        **kwargs,
    }


def build_compatible_compressor(
    model: Union[torch.nn.Module, str],
    tokenizer=None,
    platform: str = "hf",
    scheme: Union[str, dict, QuantizationScheme, "AutoScheme"] = "W4A16",
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
    algorithm: str = None,
    **kwargs,
) -> "BaseCompressor":
    """Translate the legacy (``bits=/algorithm=/iters=``) API into alg-config objects
    and build the compressor via ``PipelineCompressor``.

    Backs :class:`AutoRound`'s compatibility path (the branch taken when the caller
    does not pass ``alg_configs`` directly).
    """
    from auto_round.algorithms.transforms import normalize_rotation_config as _normalize_rotation_alg_config
    from auto_round.algorithms.transforms.hadamard.config import RotationConfig as _NewArchRotationConfig
    from auto_round.compressors.entry import PipelineCompressor, _build_model_free_compressor
    from auto_round.utils import is_diffusion_model, is_mllm_model
    from auto_round.utils.model import is_model_free_route

    device = kwargs.pop("device", None)
    if device is not None:
        logger.warning_once("`device` is deprecated, please use `device_map` instead")
        if device_map in (None, 0):
            device_map = device

    # ---- Model-free fast-path detection --------------------------------
    if is_model_free_route(model, scheme, iters, kwargs.get("disable_opt_rtn"), kwargs):
        announced_via_flag = bool(kwargs.get("model_free", False))
        compressor_only_kwargs = _compat_pop_compressor_only_kwargs(kwargs)
        return _build_model_free_compressor(
            model,
            scheme,
            layer_config,
            tokenizer,
            device_map,
            announced_via_flag=announced_via_flag,
            **compressor_only_kwargs,
            **kwargs,
        )
    # --------------------------------------------------------------------

    compressor_only_kwargs = _compat_pop_compressor_only_kwargs(kwargs)
    common_config_kwargs, auto_round_config_kwargs = _compat_pop_config_kwargs(kwargs)

    config = _compat_build_alg_config(
        algorithm=algorithm,
        iters=iters,
        gradient_accumulate_steps=gradient_accumulate_steps,
        seqlen=seqlen,
        nsamples=nsamples,
        batch_size=batch_size,
        kwargs=kwargs,
        common_config_kwargs=common_config_kwargs,
        auto_round_config_kwargs=auto_round_config_kwargs,
    )

    forward_kwargs = _compat_build_forward_kwargs(kwargs)
    format_name = forward_kwargs.pop("format", None)
    _rotation_config_raw = forward_kwargs.pop("rotation_config", None)
    if _rotation_config_raw is not None:
        _rc = _normalize_rotation_alg_config(_rotation_config_raw)
        if _rc is None:
            _rc = _NewArchRotationConfig()
        config = [config, _rc]

    # Check model type for logging (warning_once avoids repeating per block when
    # called from LLM-Compressor, which instantiates the entry per block).
    if is_mllm_model(model, platform=platform):
        logger.info("Using MLLM mode for multimodal model.")
    elif is_diffusion_model(model):
        logger.info("Using Diffusion mode for diffusion model.")
    else:
        logger.info("Using LLM mode.")

    return PipelineCompressor(
        model,
        scheme,
        config,
        tokenizer=tokenizer,
        platform=platform,
        format=format_name,
        dataset=dataset,
        iters=iters,
        gradient_accumulate_steps=gradient_accumulate_steps,
        low_gpu_mem_usage=low_gpu_mem_usage,
        device_map=device_map,
        enable_torch_compile=enable_torch_compile,
        seed=seed,
        low_cpu_mem_usage=low_cpu_mem_usage,
        layer_config=layer_config,
        nsamples=nsamples,
        seqlen=seqlen,
        batch_size=batch_size,
        **compressor_only_kwargs,
        **forward_kwargs,
    )
