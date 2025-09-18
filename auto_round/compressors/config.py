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
from typing import Any, Callable, Union

import torch


class ExtraConfig:
    """Class for extra or legacy configs."""

    config_type: str = "base"

    def __init__(
        self,
        amp: bool = True,
        device: Union[int, str, torch.dtype] = None,
        dynamic_max_gap: int = -1,
        disable_deterministic_algorithms: bool = True,
        disable_opt_rtn: bool = True,
        enable_alg_ext: bool = False,
        enable_deterministic_algorithms: bool = False,
        enable_minmax_tuning: bool = True,
        enable_norm_bias_tuning: bool = False,
        enable_quanted_input: bool = True,
        fp_layers: str = None,
        lr: float = None,
        lr_scheduler: Callable = None,
        low_cpu_mem_usage: bool = False,
        minmax_lr: float = None,
        mllm: bool = False,
        mem_per_param_scale: int = None,
        not_use_best_mse: bool = False,
        nblocks: int = 1,
        quant_lm_head: bool = False,
        sampler: str = "rand",
        scale_dtype: str = "fp16",
        static_kv_dtype: Union[str, torch.dtype] = None,
        to_quant_block_names: Union[str, list, None] = None,
    ):
        """Initialize

        Args:
            amp (bool, optional): Use AMP for tuning. Defaults to True.
            device (str | torch.device | int, optional): Compute device. Defaults to 0.
            dynamic_max_gap (int): The dynamic maximum gap (default is -1).
            disable_deterministic_algorithms (bool): deprecated, default not use deterministic_algorithms.
            disable_opt_rtn (bool, optional): Disable RTN-mode optimization (iters=0). Defaults to False.
            enable_alg_ext (bool, optional): Enable algorithm extension (primarily for INT2). Defaults to False.
            enable_deterministic_algorithms (bool, optional): whether to use deterministic_algorithms.
            enable_minmax_tuning (bool, optional): Enable weight min-max tuning. Defaults to True.
            enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning.
            enable_quanted_input (bool): Whether to use quantized input data (default is True).
            fp_layers (str): list of Layer names to maintain original data type.
            lr (float, optional): Learning rate; if None, set to 1.0 / iters except when iters==0.
            lr_scheduler: The learning rate scheduler to be used.
            low_cpu_mem_usage (bool): Whether to use low CPU memory (default is False).
            minmax_lr (float): The learning rate for min-max tuning (default is None).
            mllm (bool, optional): Whether to use multi-model mode.
            mem_per_param_scale (int): Scale factor for memory per parameter, used to adjust memory usage estimation for tuning.
            not_use_best_mse (bool): Whether to use mean squared error (default is False).
            nblocks (int): Number of blocks (default is 1).
            quant_lm_head (bool): Whether to quant lm_head.
            sampler (str): The sampling method (default is "rand").
            scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
            static_kv_dtype (str): The data type of kv-cache to be used.
            to_quant_block_names (str|list):  Names of quantitative blocks, please use commas to separate them.
        """
        self.amp = amp
        self.device = device
        self.dynamic_max_gap = dynamic_max_gap
        self.disable_deterministic_algorithms = disable_deterministic_algorithms
        self.disable_opt_rtn = disable_opt_rtn
        self.enable_alg_ext = enable_alg_ext
        self.enable_deterministic_algorithms = enable_deterministic_algorithms
        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        self.enable_quanted_input = enable_quanted_input
        self.fp_layers = fp_layers
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.minmax_lr = minmax_lr
        self.mllm = mllm
        self.mem_per_param_scale = mem_per_param_scale
        self.not_use_best_mse = not_use_best_mse
        self.nblocks = nblocks
        self.quant_lm_head = quant_lm_head
        self.sampler = sampler
        self.scale_dtype = scale_dtype
        self.static_kv_dtype = static_kv_dtype
        self.to_quant_block_names = to_quant_block_names

    @classmethod
    def __init_subclass__(cls):
        if "config_type" not in cls.__dict__:
            raise TypeError(f"Missing property 'config_type' for {cls.__name__!r}")

    def to_dict(self):
        return self.__dict__


class MLLMExtraConfig(ExtraConfig):
    config_type: str = "mllm"

    def __init__(
        self,
        processor=None,
        image_processor=None,
        quant_nontext_module: bool = False,
        extra_data_dir: str = None,
        template: str = None,
    ):
        """Initialize

        Args:
            processor: Any multi-modal model will require an object to encode or
                   decode the data that groups several modalities (among text, vision and audio).
            image_processor: Image processor for special model like llava.
            quant_nontext_module: Whether to quantize nontext module.
            extra_data_dir: The path of extra data such as images, audio and videos.
            template: The path or name of template used to specify process for different MLLMs.
        """
        self.processor = processor
        self.image_processor = image_processor
        self.quant_nontext_module = quant_nontext_module
        self.extra_data_dir = extra_data_dir
        self.template = template
