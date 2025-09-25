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

from auto_round.compressors import (
    AdamCompressor,
    BaseCompressor,
    DiffusionCompressor,
    ExtraConfig,
    LLMCompressor,
    MLLMCompressor,
)
from auto_round.logger import deprecated, logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import is_diffusion_model, is_mllm_model


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
        # for adam
        enable_adam: bool = False,
        # for MLLM
        extra_config: ExtraConfig = None,
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

        if (extra_config and not extra_config.mllm_config.is_default()) or is_mllm_model(model):
            logger.info("using MLLM mode for multimodal model.")
            model_cls.append(MLLMCompressor)
            extra_config.diffusion_config = None
        elif (extra_config and not extra_config.diffusion_config.is_default()) or is_diffusion_model(model):
            logger.info("using Diffusion mode for diffusion model.")
            model_cls.append(DiffusionCompressor)
            extra_config.mllm_config = None
        else:
            if extra_config:
                extra_config.mllm_config = None
            model_cls.append(LLMCompressor)

        if enable_adam:
            model_cls.append(AdamCompressor)
        dynamic_compressor = type("AutoRound", tuple(model_cls), {})
        if extra_config:
            kwargs.update(extra_config.to_dict())
        ar = dynamic_compressor(
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
            **kwargs,
        )
        return ar

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
            if (key not in share_cache_keys or len(indices) == 1) and not isinstance(
                input_others[key], (str, bool, type(None))
            ):
                current_input_others[key] = None
                if input_others[key] is not None:
                    current_input_others[key] = [input_others[key][i] for i in indices]
                    if len(indices) == 1:
                        current_input_others[key] = current_input_others[key][0]
                    else:
                        try:
                            current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                        except TypeError as err:
                            logger.warning_once("Please check the model cache inputs or try setting batch_size to 1.")
            else:
                current_input_others[key] = input_others[key]

        return current_input_ids, current_input_others


@deprecated("AutoRound")
class AutoRoundLLM(LLMCompressor):
    """Class for LLM quantization

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
                disable_deterministic_algorithms, mllm, static_kv_dtype
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

    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__(
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
            **kwargs,
        )


@deprecated("AutoRound")
class AutoRoundAdam(AdamCompressor):
    """Class for quantization with optimizers like adamw of a PyTorch model.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        scheme (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether sym to be used (default is True).
        layer_config (dict): Configuration for weight quantization (default is None).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset: The default dataset name (default is "NeelNanda/pile-10k").
        enable_quanted_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        low_cpu_mem_usage (bool): Whether to use low CPU memory (default is False).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        nsamples (int): Number of samples (default is 128).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).
        nblocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                           have different choices.
        act_bits (int): Number of bits for activation quantization. Default is 16.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_data_type (str): Specifies the data type for activations.
                             Defaults to None, in which case it inherits the weight data type.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A string or list whose elements are list of
                            block's layer names to be quantized.
        enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer function
        **kwargs: Additional keyword arguments.

    Returns:
        The quantized model.
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

    def __init__(
        self,
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
        device_map: Union[str, int, torch.device, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        optimizer="AdamW",
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            scheme=scheme,
            layer_config=layer_config,
            batch_size=batch_size,
            dataset=dataset,
            low_gpu_mem_usage=low_gpu_mem_usage,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            seed=seed,
            gradient_accumulate_steps=gradient_accumulate_steps,
            enable_torch_compile=enable_torch_compile,
            device_map=device_map,
            optimizer=optimizer,
            **kwargs,
        )


@deprecated("AutoRound")
class AutoRoundMLLM(MLLMCompressor):
    """Class for automatic rounding-based quantization with MLLMs.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        processor: Any multi-modal model will require an object to encode or
                   decode the data that groups several modalities (among text, vision and audio).
        image_processor: Image processor for special model like llava.
        bits (int): Number of bits for quantization (default is 4).
        group_size (int): Size of the quantization group (default is 128).
        sym (bool): Whether sym to be used (default is True).
        layer_config (dict): Configuration for weight quantization (default is None).
        batch_size (int): Batch size for training (default is 8).
        amp (bool): Whether to use automatic mixed precision (default is True).
        device: The device to be used for training (default is "auto").
        lr_scheduler: The learning rate scheduler to be used.
        dataset: The path or name of the calib dataset.
        extra_data_dir: The path of extra data such as images, audio and videos.
        template: The path or name of template used to specify process for different MLLMs.
        quant_nontext_module: Whether to quantize nontext module.
        enable_quanted_input (bool): Whether to use quantized input data (default is True).
        enable_minmax_tuning (bool): Whether to enable min-max tuning (default is True).
        lr (float): The learning rate (default is 0.005).
        minmax_lr (float): The learning rate for min-max tuning (default is None).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        low_cpu_mem_usage (bool): Whether to use low CPU memory (default is False).
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        nsamples (int): Number of samples (default is 128).
        sampler (str): The sampling method (default is "rand").
        seed (int): The random seed (default is 42).s
        nblocks (int): Number of blocks (default is 1).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        not_use_best_mse (bool): Whether to use mean squared error (default is False).
        dynamic_max_gap (int): The dynamic maximum gap (default is -1).
        data_type (str): The data type to be used (default is "int").
        scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
                           have different choices.
        act_bits (int): Number of bits for activation quantization. Default is 32.
        act_group_size (int): Group size for activation quantization. Default is None.
        act_sym (bool): Whether to use symmetric activation quantization. Default is None.
        act_dynamic (bool): Whether to use dynamic activation quantization. Default is True.
        to_quant_block_names (str|list): A string or list whose elements are list of
                            block's layer names to be quantized.
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer
        **kwargs: Additional keyword arguments.
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

    def __init__(
        self,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        processor=None,
        image_processor=None,
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        quant_nontext_module: bool = False,
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            scheme=scheme,
            layer_config=layer_config,
            dataset=dataset,
            quant_nontext_module=quant_nontext_module,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            batch_size=batch_size,
            gradient_accumulate_steps=gradient_accumulate_steps,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            **kwargs,
        )


@deprecated("AutoRound")
class AutoRoundDiffusion(DiffusionCompressor):
    """Class for automatic rounding-based quantization with Diffusion models.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data, is not used for diffusion models.
        guidance_scale (float): Control how much the image generation process follows the text prompt.
                                The more it is, the more closely it follows the prompt (default is 7.5).
        num_inference_steps (int): The reference number of denoising steps (default is 50).
        generator_seed (int): A sees that controls the initial noise from which an image is generated (default is None).
        scheme: (str| dict | QuantizationScheme ): A preset scheme that defines the quantization configurations.
        layer_config (dict): Configuration for weight quantization (default is None).
        dataset: The path or name of the calib dataset.
        iters (int): Number of iterations (default is 200).
        seqlen (int): Length of the sequence.
        nsamples (int): Number of samples (default is 128).
        batch_size (int): Batch size for training (default is 8).
        gradient_accumulate_steps (int): Number of gradient accumulation steps (default is 1).
        low_gpu_mem_usage (bool): Whether to use low GPU memory (default is False).
        device_map (str | dict | int | torch.device, optional): Device placement map. Defaults to 0.
        enable_torch_compile (bool): Whether to enable torch compile to optimize quant_block/layer
        **kwargs: Additional keyword arguments.
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

    def __init__(
        self,
        model: Union[object, str],
        tokenizer=None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator_seed: int = None,
        scheme: Union[str, dict, QuantizationScheme] = "W8A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "coco2014",
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator_seed=generator_seed,
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
            **kwargs,
        )
