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
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Union

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


@dataclass(kw_only=True)
class QuantizationConfig(AlgConfig):
    _alg_cls: ClassVar[str] = None

    # quantization args
    scheme: Union[str, dict, QuantizationScheme, AutoScheme] = "W4A16"
    layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None
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
    scale_dtype: str = None
    ignore_layers: str = ""
    quant_lm_head: bool = False
    to_quant_block_names: Union[str, list, None] = None

    def __post_init__(self):
        # Resolve scheme attributes early so properties (is_act_nv_fp, is_wfp8afp8, etc.)
        # work correctly at construction time without waiting for post_init().
        self._early_resolve_scheme()
        # Run block-wise validation early (at construction time, before model loading).
        # Guard with None checks because _early_resolve_scheme may leave attributes unresolved
        # (e.g. when scheme is an AutoScheme that needs model info).
        if self.group_size is not None and isinstance(self.group_size, (tuple, list)):
            if not (
                self.data_type is not None
                and self.bits is not None
                and self.data_type.startswith("fp")
                and self.bits == 8
            ):
                raise ValueError(
                    "Block-wise quantization (tuple group_size) only supports fp8 weight quantization, "
                    f"but got data_type='{self.data_type}', bits={self.bits}."
                )
            if (
                self.act_dynamic is not None
                and self.act_data_type is not None
                and self.act_bits is not None
                and not (self.act_dynamic and self.act_data_type.startswith("fp") and self.act_bits == 8)
            ):
                raise NotImplementedError(
                    "Block-wise fp8 weight quantization only supports dynamic fp8 activation quantization. "
                    f"Got act_dynamic={self.act_dynamic}, act_data_type='{self.act_data_type}', "
                    f"act_bits={self.act_bits}."
                )
        if self.act_group_size is not None and isinstance(self.act_group_size, (tuple, list)):
            raise ValueError(
                "`act_group_size` must be -1 (per channel), 0 (per-tensor), or a positive integer, not a tuple."
            )

    def _early_resolve_scheme(self) -> None:
        """Resolve scheme attributes early so properties work from init time.

        Both entry.py routing (needs_act_calib) and BaseCompressor._adjust_torch_compile
        need resolved attributes (act_data_type, data_type, is_act_nv_fp, ...) before
        BaseQuantizers.post_init() runs (which is deferred until quantize() / after model
        loading).  This method performs the same _parse_scheme() call eagerly so those
        attributes are available from construction time.

        AutoScheme is left deferred because it requires model information to select its
        concrete option.
        """
        if isinstance(self.scheme, AutoScheme):
            # AutoScheme needs model info for option selection — defer to post_init
            return

        # Collect fields that exist in both QuantizationScheme and QuantizationConfig
        # where the user explicitly provided a value (non-None). These override the
        # scheme's built-in defaults so that e.g. RTNConfig(scheme="NVFP4", bits=8)
        # expands NVFP4 but keeps bits=8 instead of the scheme's default bits=4.
        user_scheme_overrides = {
            k: getattr(self, k) for k in QuantizationScheme.get_attributes() if getattr(self, k, None) is not None
        }

        try:
            _, _, final_attrs = _parse_scheme(self.scheme, user_scheme_overrides)
            vars(self).update(final_attrs)
        except Exception:
            # Silently ignore failures — post_init() will do the authoritative resolution
            pass

    @staticmethod
    def _is_valid_group_size(gs) -> bool:
        """Return True if gs is a valid group_size value.

        Accepts -1 (per-channel), 0 (per-tensor), a positive integer,
        or a tuple/list of such values (e.g. (128, 128) for block-wise FP8).
        """
        if isinstance(gs, (tuple, list)):
            return all(QuantizationConfig._is_valid_group_size(g) for g in gs)
        return gs == -1 or gs >= 0

    def check_config(self) -> None:
        """Checks if the configurations are valid.

        Raises:
        ValueError, TypeError: If any of the configurations are invalid.
        """
        if self.bits <= 0:
            raise ValueError("`bits` must be positive")
        if self.act_bits <= 0:
            raise ValueError("`act_bits` must be positive")
        if not self._is_valid_group_size(self.group_size):
            raise ValueError(
                "`group_size` must be -1 (per channel), 0 (per-tensor), a positive integer, "
                "or a tuple thereof (e.g. (128, 128) for block-wise quantization)"
            )
        if isinstance(self.act_group_size, (tuple, list)):
            raise ValueError(
                "`act_group_size` must be -1 (per channel), 0 (per-tensor), or a positive integer, not a tuple."
            )
        if not self._is_valid_group_size(self.act_group_size):
            raise ValueError(
                "`act_group_size` must be -1 (per channel), 0 (per-tensor), a positive integer, " "or a tuple thereof"
            )
        # Block-wise (tuple group_size) is only valid for fp8 weight quantization
        if isinstance(self.group_size, (tuple, list)):
            if not (self.data_type.startswith("fp") and self.bits == 8):
                raise ValueError(
                    "Block-wise quantization (tuple group_size) only supports fp8 weight quantization, "
                    f"but got data_type='{self.data_type}', bits={self.bits}."
                )
            if not (self.act_dynamic and self.act_data_type.startswith("fp") and self.act_bits == 8):
                raise NotImplementedError(
                    "Block-wise fp8 weight quantization only supports dynamic fp8 activation quantization. "
                    f"Got act_dynamic={self.act_dynamic}, act_data_type='{self.act_data_type}', "
                    f"act_bits={self.act_bits}."
                )
        # Reset the default value of super_bits and super_group_size
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
        # For block-wise group_size (tuple), skip the scalar-only warnings
        scalar_gs = self.group_size if not isinstance(self.group_size, (tuple, list)) else None
        if self.is_mx_fp and scalar_gs != 32:
            logger.warning("dtype mx_fp should only support group_size of 32 in real deployment")
        if self.is_nv_fp and scalar_gs != 16:
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
    def is_standard_fp(self):
        return BackendDataType.STANDARD_FP in self.data_type and not self.is_mx_fp and not self.is_nv_fp

    @property
    def is_act_standard_fp(self):
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
            and self.is_act_standard_fp
            and self.is_standard_fp
        ):
            return True
        else:
            return False
