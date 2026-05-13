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

from typing import TYPE_CHECKING, Optional, Union

import torch

from auto_round.logger import deprecated, logger
from auto_round.schemes import QuantizationScheme

if TYPE_CHECKING:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.compressors.base import BaseCompressor
    from auto_round.compressors.config import ExtraConfig


class AutoRound:
    """Automatic weight rounding (Signed Gradient Descent) for LLM quantization

    Reference:
        Cheng, Wenhua, et al., "Optimize weight rounding via signed gradient descent for
        the quantization of LLMs." arXiv:2309.05516 (2023).

    Attributes:
        model (torch.nn.Module | str): The loaded PyTorch model in eval mode.
        tokenizer: Tokenizer used to prepare input text for calibration/tuning.
        platform (str): The platform to load pretrained moded, options: ["hf", "model_scope"]
        bits (int): Weight quantization bits.
        group_size (int or tuple): Per-group size for weight quantization.
        sym (bool): Whether to use symmetric weight quantization.
        layer_config (dict): Per-layer quantization configuration.
        nsamples (int): Number of calibration samples.
        enable_torch_compile (bool): Whether to enable torch.compile for quant blocks/layers.
    """

    SKIP_ARGS = ("local_args", "kwargs", "cls", "model_cls", "dynamic_compressor", "extra_config")

    bits: int | None
    group_size: int | tuple | None
    sym: bool | None
    data_type: str | None
    act_bits: int | None
    act_group_size: int | None
    act_sym: bool | None
    act_data_type: str | None
    act_dynamic: bool | None
    super_bits: int | None
    super_group_size: int | None

    # all args in __new__ need be passed to the dynamic created class __init__
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
        enable_adam: bool = False,
        extra_config: "ExtraConfig" = None,
        enable_alg_ext: bool = False,
        disable_opt_rtn: bool | None = None,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ) -> "BaseCompressor":
        """Initialize AutoRound with quantization and tuning configuration.

        Args:
            model (torch.nn.Module | str): Model object or model name to load.
            tokenizer: Tokenizer for text processing. Required if `model` is not a string and `iters > 0`.
            platform: The platform to download pretrained model, options: ["hf", "model_scope"]
            scheme (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations
            layer_config (dict, optional): Layer-wise quantization config. Defaults to None.
            dataset (str | list | tuple | DataLoader, optional): Calibration data. Defaults to "NeelNanda/pile-10k".
            iters (int, optional): Optimization iterations. Defaults to 200.
            seqlen (int, optional): Calibration sequence length. Defaults to 2048.
            nsamples (int, optional): Number of calibration samples. Defaults to 128.
            batch_size (int, optional): Calibration batch size. Defaults to 8.
            gradient_accumulate_steps (int, optional): Gradient accumulation steps. Defaults to 1.
            low_gpu_mem_usage (bool, optional): Lower GPU memory mode. Defaults to False.
            device_map (str | dict, optional): Device map for each module. Defaults to 0.
            enable_torch_compile (bool, optional): Enable torch.compile for low cost in quantization. Defaults to False.
            seed (int, optional): Random seed. Defaults to 42.
            enable_adam (bool, optional): Enable Adam-based optimizer. Defaults to False.
            extra_config(ExtraConfig, optional): Extra configuration for lots of configurations. Defaults to None.
            enable_alg_ext (bool, optional): Enable algorithm extension (primarily for INT2)
                                             for better accuracy. Defaults to False.
            disable_opt_rtn (bool, optional): Disable RTN-mode optimization (iters=0) for fast quatnziation
                                              with lower accuracy. Defaults to None.
            low_cpu_mem_usage (bool, optional): Lower CPU memory mode. Defaults to False.

            bits (int, optional): Weight quantization bits. Defaults to 4.
            group_size (int or tuple, optional): Weight quantization group size. Defaults to 128.
            sym (bool, optional): Symmetric weight quantization. Defaults to True.
            data_type (str, optional): Weight data type string, e.g., "int". Defaults to "int".
            act_bits (int, optional): Activation quantization bits. Defaults to 16.
            act_group_size (int, optional): Activation group size. Defaults to None.
            act_sym (bool, optional): Symmetric activation quantization. Defaults to None.
            act_data_type (str, optional): Activation data type; inherits weight dtype if None and act_bits < 16.
            act_dynamic (bool, optional): Dynamic activation quantization. Defaults to True.
            model_dtype (str): model dtype used to load pre-trained model.
            amp (bool, optional): Use AMP for tuning. Defaults to True.
            enable_minmax_tuning (bool, optional): Enable weight min-max tuning. Defaults to True.
            lr (float, optional): Learning rate; if None, set to 1.0 / iters except when iters==0.
            minmax_lr (float, optional): Learning rate for min-max tuning; defaults to `lr`.

            **kwargs: Backward compatible options:
                - enable_alg_ext, quant_lm_head, lr, lr_scheduler, sampler, not_use_best_mse, dynamic_max_gap,
                  super_group_size, super_bits, scale_dtype ("fp16" etc.),
                  nblocks, to_quant_block_names,
                  enable_norm_bias_tuning, enable_quanted_input,
                  disable_deterministic_algorithms, vlm, static_kv_dtype
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

        local_args = {k: v for k, v in locals().items() if k not in cls.SKIP_ARGS}
        if extra_config is not None:
            local_args.update({k: v for k, v in extra_config.to_dict().items() if k in local_args and v is not None})

        from auto_round.compressors.entry import AutoRoundCompatible

        return AutoRoundCompatible(**local_args, **kwargs)

    @classmethod
    @torch.no_grad()
    def _sampling_inputs(
        cls,
        input_ids: list[torch.Tensor],
        input_others: dict,
        indices: list[int],
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
        current_input_ids = [input_ids[i] for i in indices]

        current_input_ids = torch.cat(current_input_ids, dim=batch_dim)

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


@deprecated("AutoRound")
class AutoRoundLLM:

    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)


@deprecated("AutoRound")
class AutoRoundAdam:

    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("enable_adam", True)
        return AutoRound(*args, **kwargs)


@deprecated("AutoRound")
class AutoRoundMLLM:

    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)


@deprecated("AutoRound")
class AutoRoundDiffusion:

    def __new__(cls, *args, **kwargs):
        return AutoRound(*args, **kwargs)
