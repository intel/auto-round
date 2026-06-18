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
from enum import Enum
from typing import ClassVar, Union

from auto_round.export.export_to_gguf.config import GGUF_INNER_CONFIG
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


class BackendDataType(str, Enum):
    STANDARD_FP = "fp"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"
    FP8_STATIC = "fp8_static"
    FP8 = "fp8"


class QuantizationConfig:
    """Common quantization configuration shared by block quantizers.

    Args:
        bits: Weight quantization bit width.
        group_size: Weight quantization group size. Use -1 for per-channel,
            0 for per-tensor, or a positive integer for grouped quantization.
        sym: Whether to use symmetric weight quantization.
        data_type: Weight quantization data type, such as int, mx_fp,
            nv_fp, or fp8 variants.
        act_bits: Activation quantization bit width.
        act_group_size: Activation quantization group size.
        act_sym: Whether to use symmetric activation quantization.
        act_data_type: Activation quantization data type.
        act_dynamic: Whether activation quantization should be dynamic.
        super_bits: Bit width used for double quantization metadata.
        super_group_size: Group size used for double quantization metadata.
    """

    _scheme_fields: ClassVar[set[str]] = set(QuantizationScheme.get_attributes())

    def __init__(self, *, scheme: QuantizationScheme = None, **kwargs) -> None:
        object.__setattr__(self, "scheme", scheme if scheme is not None else QuantizationScheme.empty())
        object.__setattr__(self, "_user_set_scheme_fields", set())

        unknown = []
        for key, value in kwargs.items():
            if key in self._scheme_fields:
                setattr(self.scheme, key, value)
                self._user_set_scheme_fields.add(key)
            else:
                unknown.append(key)
        if unknown:
            unknown_args = ", ".join(repr(arg) for arg in unknown)
            raise TypeError(f"Unexpected quantization config argument(s): {unknown_args}")

        self._check_partial_config()

    def __getattr__(self, name):
        if name in self._scheme_fields:
            return getattr(self.scheme, name, None)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in self._scheme_fields and "scheme" in self.__dict__:
            setattr(self.scheme, name, value)
            self._user_set_scheme_fields.add(name)
            return
        object.__setattr__(self, name, value)

    def _check_partial_config(self):
        # Run block-wise validation early (at construction time, before model loading).
        # Scheme resolution is deferred to BaseCompressor.post_init() via SchemeMixin.
        # Guard with None checks in case the user hasn't explicitly set data_type/bits
        # (they will be resolved from scheme by the compressor before use).
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
    def is_act_quantize(self) -> bool:
        return self.act_bits is not None and self.act_bits <= 8

    @property
    def is_nv_fp(self) -> bool:
        return self.data_type is not None and BackendDataType.NV_FP in self.data_type

    @property
    def is_act_nv_fp(self) -> bool:
        return self.act_data_type is not None and BackendDataType.NV_FP in self.act_data_type

    @property
    def is_mx_fp(self) -> bool:
        return self.data_type is not None and BackendDataType.MX_FP in self.data_type

    @property
    def is_act_mx_fp(self) -> bool:
        return self.act_data_type is not None and BackendDataType.MX_FP in self.act_data_type

    @property
    def is_dynamic_wint8aint8(self) -> bool:
        if self.act_dynamic:
            return True
        if self.act_data_type is not None and self.data_type is not None:
            if ("int8" in self.act_data_type or ("int" in self.act_data_type and self.act_bits == 8)) and (
                "int8" in self.data_type or ("int" in self.data_type and self.bits == 8)
            ):
                return True
        return False

    @property
    def is_standard_fp(self) -> bool:
        return (
            self.data_type is not None
            and BackendDataType.STANDARD_FP in self.data_type
            and not self.is_mx_fp
            and not self.is_nv_fp
        )

    @property
    def is_act_standard_fp(self) -> bool:
        return (
            self.act_data_type is not None
            and BackendDataType.STANDARD_FP in self.act_data_type
            and not self.is_act_mx_fp
            and not self.is_act_nv_fp
        )

    @property
    def is_static_afp8(self) -> bool:
        return self.act_data_type is not None and BackendDataType.FP8_STATIC in self.act_data_type

    @property
    def is_static_wfp8afp8(self) -> bool:
        return self.data_type is not None and BackendDataType.FP8_STATIC in self.data_type and self.is_static_afp8

    @property
    def is_wfp8afp8(self) -> bool:
        if self.act_data_type is None or self.data_type is None:
            return False
        if (
            ("fp8" in self.act_data_type or ("fp" in self.act_data_type and self.act_bits == 8))
            and ("fp8" in self.data_type or ("fp" in self.data_type and self.bits == 8))
            and self.is_act_standard_fp
            and self.is_standard_fp
        ):
            return True
        else:
            return False
