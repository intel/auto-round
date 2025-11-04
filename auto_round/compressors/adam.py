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

from typing import Union

import torch

from auto_round.compressors.base import BaseCompressor
from auto_round.schemes import QuantizationScheme
from auto_round.utils import check_is_cpu, htcore, is_hpex_available


class AdamCompressor(BaseCompressor):
    """Class for quantization with optimizers like adamw of a PyTorch model.

    Args:
        model: The PyTorch model to be quantized.
        tokenizer: An optional tokenizer for processing input data.
        platform (str): The platform to load pretrained moded, options: ["hf", "model_scope"]
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
        platform="hf",
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
        model_dtype: str = None,
        **kwargs,
    ):
        super(AdamCompressor, self).__init__(
            model=model,
            tokenizer=tokenizer,
            platform=platform,
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
            model_dtype=model_dtype,
            **kwargs,
        )

        self.optimizer = self._get_optimizer(optimizer)

    def _get_optimizer(self, optimizer):
        if optimizer is None:
            optimizer = torch.optim.AdamW
        elif isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)
        else:
            optimizer = optimizer
        return optimizer

    def _get_scaler(self):
        scaler = None
        if self.amp and not check_is_cpu(self.device):
            from torch.cuda.amp import GradScaler

            scaler = GradScaler(init_scale=1024, growth_interval=100000)
        return scaler

    def _scale_loss_and_backward(self, scaler, loss):
        if scaler is not None:
            loss = scaler.scale(loss)

        loss.backward()
        if is_hpex_available():
            htcore.mark_step()
        return loss

    def _step(self, scaler, optimizer, lr_schedule):
        if scaler is not None:
            scaler.step(optimizer)
            optimizer.zero_grad()
            lr_schedule.step()
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step()
        if is_hpex_available():
            htcore.mark_step()
