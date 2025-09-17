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

from typing import Any, Callable, Union

import torch

from auto_round.compressors import AdamCompressor, BaseCompressor, LLMCompressor, MLLMCompressor
from auto_round.logger import deprecated, logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import is_mllm_model


def _clean_kwargs(kwargs: dict, model_cls: list[BaseCompressor]) -> dict:
    if MLLMCompressor not in model_cls:
        for key in ["extra_data_dir", "template"]:
            if key in kwargs:
                kwargs.pop(key)
    return kwargs


class AutoRound:
    """Automatic weight rounding (Signed Gradient Descent) for LLM quantization

    Reference:
        Cheng, Wenhua, et al., "Optimize weight rounding via signed gradient descent for
        the quantization of LLMs." arXiv:2309.05516 (2023).

    Attributes:
        model (torch.nn.Module): The loaded PyTorch model in eval mode.
        tokenizer: Tokenizer used to prepare input text for calibration/tuning.
        bits (int): Weight quantization bits.
        group_size (int): Per-group size for weight quantization.
        sym (bool): Whether to use symmetric weight quantization.
        layer_config (dict): Per-layer quantization configuration.
        nsamples (int): Number of calibration samples.
        enable_torch_compile (bool): Whether to enable torch.compile for quant blocks/layers.
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

    def __new__(
        cls,
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
        fp_layers: str = None,
        # for adam
        adam: bool = False,
        # for MLLM
        mllm=False,
        processor=None,
        image_processor=None,
        quant_nontext_module: bool = False,
        **kwargs,
    ) -> BaseCompressor:
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
                - enable_alg_ext, quant_lm_head, lr, lr_scheduler, sampler, not_use_best_mse, dynamic_max_gap,
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
        model_cls = []
        if mllm or is_mllm_model(model):
            logger.info("using MLLM mode for multimodal model.")
            model_cls.append(MLLMCompressor)
            mllm_kwargs = {
                "mllm": mllm,
                "processor": processor,
                "image_processor": image_processor,
                "quant_nontext_module": quant_nontext_module,
            }
            kwargs.update(mllm_kwargs)
        else:
            model_cls.append(LLMCompressor)
        if adam:
            model_cls.append(AdamCompressor)
        dynamic_compressor = type("AutoRound", tuple(model_cls), {})
        kwargs = _clean_kwargs(kwargs, model_cls)
        return dynamic_compressor(
            model=model,
            tokenizer=tokenizer,
            scheme=scheme,
            layer_config=layer_config,
            dataset=dataset,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            batch_size=batch_size,
            gradient_accumulate_steps=gradient_accumulate_steps,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            fp_layers=fp_layers,
            **kwargs,
        )


@deprecated("AutoRound")
class AutoRoundAdam(AutoRound):
    def __init__(self, *args, **kwargs):
        super().__init__()


@deprecated("AutoRound")
class AutoRoundLLM(AutoRound):
    def __init__(self, *args, **kwargs):
        super().__init__()


@deprecated("AutoRound")
class AutoRoundMLLM(AutoRound):
    def __init__(self, *args, **kwargs):
        super().__init__()
