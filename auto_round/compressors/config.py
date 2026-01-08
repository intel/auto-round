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
from typing import Any, Callable, Optional, Union

import torch


class ExtraConfig:
    """Class for extra or legacy configs."""

    _model_config = None
    _scheme_config = None
    _tuning_config = None
    _mllm_config = None
    _diffusion_config = None

    def __init__(
        self,
        # tuning
        amp: bool = True,
        disable_opt_rtn: bool | None = None,
        enable_alg_ext: bool = False,
        enable_minmax_tuning: bool = True,
        enable_norm_bias_tuning: bool = False,
        enable_quanted_input: bool = True,
        enable_deterministic_algorithms: bool = False,
        lr: float = None,
        lr_scheduler: Callable = None,
        minmax_lr: float = None,
        nblocks: int = 1,
        to_quant_block_names: Union[str, list, None] = None,
        scale_dtype: str = "fp16",
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
        static_kv_dtype: Union[str, torch.dtype] = None,
        quant_lm_head: bool = False,
        ignore_layers: str = None,
        # mllm
        processor: Callable = None,
        image_processor: Callable = None,
        quant_nontext_module: bool = False,
        extra_data_dir: str = None,
        template: str = None,
        # diffusion
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        generator_seed: int = None,
    ):
        """Initialize

        Args:
            amp (bool): Whether to use automatic mixed precision (default is True).
            disable_opt_rtn (bool, optional): Disable RTN-mode optimization (iters=0). Defaults to True.
            enable_alg_ext (bool, optional): Enable algorithm extension (primarily for INT2). Defaults to False.
            enable_minmax_tuning (bool, optional): Enable weight min-max tuning. Defaults to True.
            enable_norm_bias_tuning (bool): Whether to enable fast norm/layer_bias tuning.
            enable_quanted_input (bool): Whether to use quantized input data (default is True).
            enable_deterministic_algorithms (bool): Whether to use deterministic_algorithms.
            lr (float): The learning rate (default is 0.005).
            lr_scheduler: The learning rate scheduler to be used.
            minmax_lr (float): The learning rate for min-max tuning (default is None).
            nblocks (int): Number of blocks (default is 1).
            quant_lm_head (bool): Whether to quant lm_head.
            to_quant_block_names (str|list):  Names of quantitative blocks, please use commas to separate them.
            scale_dtype (str): The data type of quantization scale to be used (default is "float16"), different kernels
            bits (int, optional): Weight quantization bits. Defaults to 4.
            group_size (int, optional): Weight quantization group size. Defaults to 128.
            sym (bool, optional): Symmetric weight quantization. Defaults to True.
            data_type (str, optional): Weight data type string, e.g., "int". Defaults to "int".
            act_bits (int, optional): Activation quantization bits. Defaults to 16.
            act_group_size (int, optional): Activation group size. Defaults to None.
            act_sym (bool, optional): Symmetric activation quantization. Defaults to None.
            act_data_type (str, optional): Activation data type; inherits weight dtype if None and act_bits < 16.
            act_dynamic (bool, optional): Dynamic activation quantization. Defaults to True.
            super_bits (int): number of scale and mins quant bits for double quant.
            super_group_size (int): the number of super group size when use double quant.
            static_kv_dtype (str): The data type of kv-cache to be used.
            processor: Any multi-modal model will require an object to encode or
                decode the data that groups several modalities (among text, vision and audio).
            image_processor: Image processor for special model like llava.
            quant_nontext_module: Whether to quantize nontext module.
            extra_data_dir: The path of extra data such as images, audio and videos.
            template: The path or name of template used to specify process for different MLLMs.
            guidance_scale (float): Control how much the image generation process follows the text prompt.
                                    The more it is, the more closely it follows the prompt (default is 7.5).
            num_inference_steps (int): The reference number of denoising steps (default is 50).
            generator_seed (int): A seed that controls the initial noise for image generation (default is None).
        """
        self.tuning_config = TuningExtraConfig(
            amp=amp,
            disable_opt_rtn=disable_opt_rtn,
            enable_alg_ext=enable_alg_ext,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            enable_quanted_input=enable_quanted_input,
            enable_deterministic_algorithms=enable_deterministic_algorithms,
            lr=lr,
            lr_scheduler=lr_scheduler,
            minmax_lr=minmax_lr,
            nblocks=nblocks,
            to_quant_block_names=to_quant_block_names,
            scale_dtype=scale_dtype,
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
            static_kv_dtype=static_kv_dtype,
            quant_lm_head=quant_lm_head,
            ignore_layers=ignore_layers,
        )
        self.mllm_config = MLLMExtraConfig(
            processor=processor,
            image_processor=image_processor,
            quant_nontext_module=quant_nontext_module,
            extra_data_dir=extra_data_dir,
            template=template,
        )
        self.diffusion_config = DiffusionExtraConfig(
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator_seed=generator_seed,
        )

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
        if config is None:
            self._mllm_config = None
        else:
            assert isinstance(
                config, MLLMExtraConfig
            ), f"mllm_config should be MLLMExtraConfig, but got {config.__class__.__name__}"
            self._mllm_config = config

    @property
    def diffusion_config(self):
        return self._diffusion_config

    @diffusion_config.setter
    def diffusion_config(self, config: DiffusionExtraConfig):
        if config is None:
            self._diffusion_config = None
        else:
            assert isinstance(
                config, DiffusionExtraConfig
            ), f"diffusion_config should be DiffusionExtraConfig, but got {config.__class__.__name__}"
            self._diffusion_config = config

    def to_dict(self):
        output_dict = {}
        for config in self.__dict__.values():
            if config:
                output_dict.update(config.to_dict())
        return output_dict


@dataclass
class BaseExtraConfig:

    @classmethod
    def get_attributes(cls: "BaseExtraConfig") -> list[str]:
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

    def is_default(self):
        for field in fields(self):
            default_value = field.default
            current_value = getattr(self, field.name)
            if current_value != default_value:
                return False
        return True


@dataclass
class TuningExtraConfig(BaseExtraConfig):
    amp: bool = True
    disable_opt_rtn: bool | None = True
    enable_alg_ext: bool = False
    enable_minmax_tuning: bool = True
    enable_norm_bias_tuning: bool = False
    enable_quanted_input: bool = True
    enable_deterministic_algorithms: bool = False
    lr: float = None
    lr_scheduler: Callable = None
    minmax_lr: float = None
    nblocks: int = 1
    to_quant_block_names: Union[str, list, None] = None
    scale_dtype: str = "fp16"


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
    static_kv_dtype: Union[str, torch.dtype] = None
    static_attention_dtype: Union[str, torch.dtype] = None
    quant_lm_head: bool = False
    ignore_layers: str = None


@dataclass
class MLLMExtraConfig(BaseExtraConfig):
    processor: Callable = None
    image_processor: Callable = None
    quant_nontext_module: bool = False
    extra_data_dir: str = None
    template: str = None


@dataclass
class DiffusionExtraConfig(BaseExtraConfig):
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    generator_seed: int = None
