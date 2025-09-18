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

from dataclasses import dataclass, fields
from typing import Any, Callable, Union

import torch


class ExtraConfig:
    """Class for extra or legacy configs."""

    _model_config = None
    _scheme_config = None
    _tuning_config = None
    _mllm_config = None

    def __init__(
        self,
        # model
        low_cpu_mem_usage: bool = False,
        mllm: bool = False,
        scale_dtype: str = "fp16",
        static_kv_dtype: Union[str, torch.dtype] = None,
        # tuning
        amp: bool = True,
        disable_opt_rtn: bool = True,
        enable_alg_ext: bool = False,
        enable_minmax_tuning: bool = True,
        enable_norm_bias_tuning: bool = False,
        enable_quanted_input: bool = True,
        lr: float = None,
        lr_scheduler: Callable = None,
        minmax_lr: float = None,
        mem_per_param_scale: int = None,
        nblocks: int = 1,
        quant_lm_head: bool = False,
        to_quant_block_names: Union[str, list, None] = None,
        # scheme
        bits: int = None,
        group_size: int = None,
        sym: bool = None,
        data_type: str = None,
        act_bits: int = None,
        act_group_size: int = None,
        act_sym: bool = None,
        act_data_type: str = None,
        act_dynamic: bool = None,
        super_bits: int = None,
        super_group_size: int = None,
        # mllm
        processor: Callable = None,
        image_processor: Callable = None,
        quant_nontext_module: bool = False,
        extra_data_dir: str = None,
        template: str = None,
    ):
        """Initialize

        Args:
            amp (bool, optional): Use AMP for tuning. Defaults to True.
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
            nblocks (int): Number of blocks (default is 1).
            quant_lm_head (bool): Whether to quant lm_head.
            scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
            static_kv_dtype (str): The data type of kv-cache to be used.
            to_quant_block_names (str|list):  Names of quantitative blocks, please use commas to separate them.
        """
        self.model_config = ModelExtraConfig(
            mllm=mllm, low_cpu_mem_usage=low_cpu_mem_usage, scale_dtype=scale_dtype, static_kv_dtype=static_kv_dtype
        )
        self.tuning_config = TuningExtraConfig(
            amp=amp,
            disable_opt_rtn=disable_opt_rtn,
            enable_alg_ext=enable_alg_ext,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_quanted_input=enable_quanted_input,
            lr=lr,
            lr_scheduler=lr_scheduler,
            minmax_lr=minmax_lr,
            mem_per_param_scale=mem_per_param_scale,
            nblocks=nblocks,
            quant_lm_head=quant_lm_head,
            to_quant_block_names=to_quant_block_names,
        )
        self.scheme_config = SchemeExtraConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            data_type=data_type,
            act_bits=act_bits,
            act_group_size=act_group_size,
            act_sym=act_sym,
            act_data_type=act_data_type,
            act_dynamic=act_dynamic,
            super_bits=super_bits,
            super_group_size=super_group_size,
        )
        self.mllm_config = MLLMExtraConfig(
            processor=processor,
            image_processor=image_processor,
            quant_nontext_module=quant_nontext_module,
            extra_data_dir=extra_data_dir,
            template=template,
        )

    @property
    def model_config(self):
        return self._model_config

    @model_config.setter
    def model_config(self, config: ModelExtraConfig):
        assert isinstance(
            config, ModelExtraConfig
        ), f"model_config should be ModelExtraConfig, but got {config.__class__.__name__}"
        self._model_config = config

    @property
    def tuning_config(self):
        return self._tuning_config

    @tuning_config.setter
    def tuning_config(self, config: TuningExtraConfig):
        assert isinstance(
            config, TuningExtraConfig
        ), f"tuning_config should be ModelExtraConfig, but got {config.__class__.__name__}"
        self._tuning_config = config

    @property
    def scheme_config(self):
        return self._scheme_config

    @scheme_config.setter
    def scheme_config(self, config: SchemeExtraConfig):
        assert isinstance(
            config, SchemeExtraConfig
        ), f"scheme_config should be SchemeExtraConfig, but got {config.__class__.__name__}"
        self._scheme_config = config

    @property
    def mllm_config(self):
        return self._mllm_config

    @mllm_config.setter
    def mllm_config(self, config: MLLMExtraConfig):
        assert isinstance(
            config, MLLMExtraConfig
        ), f"mllm_config should be MLLMExtraConfig, but got {config.__class__.__name__}"
        self._mllm_config = config

    def to_dict(self):
        output_dict = {}
        for config in self.__dict__.values():
            output_dict.update(config.to_dict())
        return output_dict


@dataclass
class BaseExtraConfig:

    @classmethod
    def get_attributes(cls: "ModelExtraConfig") -> list[str]:
        return [field.name for field in fields(cls)]

    def __getitem__(self, key: str):
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        return getattr(self, key)

    def __setitem__(self, key: str, value: None | int | str):
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        setattr(self, key, value)

    def __contains__(self, item):
        return item in self.get_attributes()

    def to_dict(self):
        return self.__dict__


@dataclass
class ModelExtraConfig(BaseExtraConfig):
    low_cpu_mem_usage: bool = False
    mllm: bool = False
    scale_dtype: str = "fp16"
    static_kv_dtype: Union[str, torch.dtype] = None


@dataclass
class TuningExtraConfig(BaseExtraConfig):
    amp: bool = True
    disable_opt_rtn: bool = True
    enable_alg_ext: bool = False
    enable_minmax_tuning: bool = True
    enable_norm_bias_tuning: bool = False
    enable_quanted_input: bool = True
    lr: float = None
    lr_scheduler: Callable = None
    minmax_lr: float = None
    mem_per_param_scale: int = None
    nblocks: int = 1
    quant_lm_head: bool = False
    to_quant_block_names: Union[str, list, None] = None


@dataclass
class SchemeExtraConfig(BaseExtraConfig):
    bits: int = None
    group_size: int = None
    sym: bool = None
    data_type: str = None
    act_bits: int = None
    act_group_size: int = None
    act_sym: bool = None
    act_data_type: str = None
    act_dynamic: bool = None
    super_bits: int = None
    super_group_size: int = None


@dataclass
class MLLMExtraConfig(BaseExtraConfig):
    processor: Callable = None
    image_processor: Callable = None
    quant_nontext_module: bool = False
    extra_data_dir: str = None
    template: str = None
