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
from enum import Enum
from typing import Union

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.export.export_to_gguf.config import GGUF_INNER_CONFIG
from auto_round.logger import logger
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    _parse_scheme,
    get_gguf_scheme,
    preset_name_to_scheme,
)
from auto_round.utils import convert_dtype_str2torch


class BackendDataType(str, Enum):
    STANDARD_FP = "fp"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"
    FP8_STATIC = "fp8_static"
    FP8 = "fp8"


class QuantizationConfig(AlgConfig):
    _alg_cls: str = None

    def __init__(
        self,
        scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        *,
        # quantization args
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
        scale_dtype: str = None,
        ignore_layers: str = "",
        quant_lm_head: bool = False,
        to_quant_block_names: Union[str, list, None] = None,
    ):

        self.scheme = scheme
        self.layer_config = layer_config

        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.data_type = data_type
        self.act_bits = act_bits
        self.act_group_size = act_group_size
        self.act_sym = act_sym
        self.act_data_type = act_data_type
        self.act_dynamic = act_dynamic
        self.super_bits = super_bits
        self.super_group_size = super_group_size

        self.scale_dtype = scale_dtype
        self.ignore_layers = ignore_layers
        self.quant_lm_head = quant_lm_head
        self.to_quant_block_names = to_quant_block_names

    def check_config(self) -> None:
        """Checks if the configurations are valid.

        Raises:
        ValueError, TypeError: If any of the configurations are invalid.
        """
        if self.bits <= 0:
            raise ValueError("`bits` must be positive")
        if self.act_bits <= 0:
            raise ValueError("`act_bits` must be positive")
        if not (self.group_size == -1 or self.group_size >= 0):
            raise ValueError("`group_size` must be -1 (per channel) or 0 (per-tensor) or a positive integer")
        if not (self.act_group_size == -1 or self.act_group_size >= 0):
            raise ValueError("`act_group_size` must be -1 (per channel) or 0 (per-tensor) or a positive integer")
        """Reset the default value of super_bits and super_group_size"""
        if self.data_type.endswith("_dq"):
            gguf_config = GGUF_INNER_CONFIG[f"gguf:q{self.bits}_k"]
            self.super_bits = gguf_config.get("super_bits", None) if self.super_bits is None else self.super_bits
            self.super_group_size = (
                gguf_config.get("super_group_size", None) if self.super_group_size is None else self.super_group_size
            )

        if (
            self.is_act_quantize
            and (not self.is_act_nv_fp or "static_gs" not in self.act_data_type)
            and not self.is_act_mx_fp
            and not self.is_dynamic_wint8aint8
            and not self.is_static_afp8
        ):
            logger.warning(
                "activation quantization is an experimental feature with limited support and a complex API. "
                "And please save the quantized model to fake format as real deployment is not supported currently"
            )
        if self.is_mx_fp and self.group_size != 32:
            logger.warning("dtype mx_fp should only support group_size of 32 in real deployment")

        if self.is_nv_fp and (self.group_size != 16):
            logger.warning("dtype nv_fp should only support group_size of 16 in real deployment")

    @property
    def is_act_quantize(self):
        return self.act_bits is not None and self.act_bits <= 8

    @property
    def is_nv_fp(self):
        return BackendDataType.NV_FP in self.data_type

    @property
    def is_act_nv_fp(self):
        return BackendDataType.NV_FP in self.act_data_type

    @property
    def is_mx_fp(self):
        return BackendDataType.MX_FP in self.data_type

    @property
    def is_act_mx_fp(self):
        return BackendDataType.MX_FP in self.act_data_type

    @property
    def is_dynamic_wint8aint8(self):
        if self.act_dynamic:
            return True
        if ("int8" in self.act_data_type or ("int" in self.act_data_type and self.act_bits == 8)) and (
            "int8" in self.data_type or ("int" in self.data_type and self.bits == 8)
        ):
            return True
        return False

    @property
    def is_standard_fp(self, act=False):
        return BackendDataType.STANDARD_FP in self.data_type and not self.is_mx_fp and not self.is_nv_fp

    @property
    def is_act_standard_fp(self, act=False):
        return BackendDataType.STANDARD_FP in self.act_data_type and not self.is_act_mx_fp and not self.is_act_nv_fp

    @property
    def is_static_afp8(self):
        return BackendDataType.FP8_STATIC in self.act_data_type

    @property
    def is_static_wfp8afp8(self):
        return BackendDataType.FP8_STATIC in self.data_type and self.is_static_afp8

    @property
    def is_wfp8afp8(self):
        if (
            ("fp8" in self.act_data_type or ("fp" in self.act_data_type and self.act_bits == 8))
            and ("fp8" in self.data_type or ("fp" in self.data_type and self.bits == 8))
            and self.is_standard_fp(act=True)
            and self.is_standard_fp(act=False)
        ):
            return True
        else:
            return False
