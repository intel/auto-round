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

import copy
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import Enum
from typing import Any, Callable, Optional, Union

import torch
import transformers

from auto_round.export.export_to_gguf.config import ModelType
from auto_round.planning.errors import FormatCompatibilityError
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    compress_layer_names,
    copy_python_files_from_model_cache,
    find_matching_blocks,
    get_block_names,
    get_module,
    logger,
    unsupported_meta_device,
)


class BackendDataType(str, Enum):
    """Quantized data-type/backend-variant identifiers used as the suffix after a format
    name (e.g. ``auto_round:fp8_static``, ``llm_compressor:int8_w8a8``). Shared across the
    ``auto_round`` and ``llm_compressor`` output formats -- not tied to a single format.
    """

    # Weight: FP8, per-channel, may be extended to per-tensor in future
    # Activation: FP8, per-tensor
    FP8_STATIC = "fp8_static"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"
    FP8 = "fp8"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"
    MX_FP_RCEIL = "mx_fp_rceil"
    NV_FP4_WITH_STATIC_GS = "nv_fp4_with_static_gs"
    INT8 = "int8_w8a8"
    FP8_BLOCK = "fp8_block"
    MXINT4 = "mxint4"
    MX_INT = "mx_int"
    WINT_A16 = "wint_a16"


def _check_divisible_by_32(scheme: QuantizationScheme, model, layer_config: dict) -> dict:
    if model is None:
        return layer_config
    default_dict = asdict(scheme)
    skipped_layers = []
    if default_dict["data_type"] == "int" and default_dict["act_bits"] >= 16:
        for n, m in model.named_modules():
            if type(m) in SUPPORTED_LAYER_TYPES or m.__class__.__name__ in INNER_SUPPORTED_LAYER_TYPES:
                if m.weight.shape[0] % 32 or m.weight.shape[1] % 32:
                    if layer_config is None:
                        layer_config = {}
                    if layer_config.get(n) is not None and layer_config[n]["bits"] >= 16:
                        continue
                    layer_config.setdefault(n, copy.deepcopy(default_dict))
                    layer_config[n].update({"bits": 16, "data_type": "fp", "fixed_by_user": True})
                    skipped_layers.append(n)
    compressed_skipped_layers = compress_layer_names(skipped_layers)
    if compressed_skipped_layers:
        logger.warning_once(
            f"some layers are skipped quantization (shape not divisible by 32): {compressed_skipped_layers}"
        )
    return layer_config


class OutputFormat(ABC):
    """ "Base class for different output formats.

    format: determines which method from export module to use for exporting.
            For example, auto_round, gguf, llm_compressor etc.
    backend: determines the specific export process within the format.
            For example, auto_round:fp8_static, auto_round:auto_awq etc.
    """

    support_schemes: list = []
    _format_list: dict[str, OutputFormat] = {}
    format_name = "base"

    def __init__(self, format: str, scheme: QuantizationScheme, ctx: Any):
        """Initialize the OutputFormat class."""
        self.output_format = format
        self.backend = None
        self.mllm = ctx.mllm

        if not self.is_fake() and not self.is_support_scheme(scheme):
            raise FormatCompatibilityError(
                f"Currently, the {self.format_name} format only supports {self.support_schemes}, "
                f"but got scheme {scheme}, please change to fake or auto_round etc."
            )

    @classmethod
    def register(cls, *names: str) -> Callable[[OutputFormat], OutputFormat]:
        assert names

        def func(output_format: OutputFormat) -> OutputFormat:
            for name in names:
                cls._format_list[name] = output_format
            return output_format

        return func

    @classmethod
    def get_support_matrix(cls: OutputFormat) -> str:
        output_str = ""
        for k, v in sorted(cls._format_list.items()):
            if k == "fake":
                support_schemes = "All schemes"
            else:
                if ":" in k and k.split(":")[1] in cls._format_list:
                    support_schemes = cls._format_list[k.split(":")[1]].support_schemes
                else:
                    support_schemes = v.support_schemes
                support_schemes = ", ".join(support_schemes).rstrip(",")
            output_str += f"\x1b[31;1m{k}\x1b[0m support scheme:\n\t{support_schemes}\n"
        return output_str

    def get_backend_name(self) -> str:
        if self.backend is None:
            return self.output_format

        # auto_round:llm_compressor:fp8_static
        if self.backend.backend is not None:
            return f"{self.output_format}:{self.backend.get_backend_name()}"
        # auto_round:fp8_static, llm_compressor:fp8_static, auto_round:auto_awq
        else:
            return self.backend.get_backend_name()

    @classmethod
    def is_support_scheme(cls: OutputFormat, scheme: Union[str, QuantizationScheme]) -> bool:
        if isinstance(scheme, str) and scheme.upper() in cls.support_schemes:
            return True
        if isinstance(scheme, QuantizationScheme):
            return cls.check_scheme_args(scheme)
        return False

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        return True

    def check_and_reset_format(
        self,
        scheme: QuantizationScheme,
        ctx: Any,
    ) -> tuple[Optional[str], QuantizationScheme, dict, list]:
        layer_config, quant_block_list = ctx.layer_config, ctx.quant_block_list
        if self.backend is not None:
            new_format, scheme, layer_config, quant_block_list = self.backend.check_and_reset_format(scheme, ctx)
            ctx.layer_config, ctx.quant_block_list = layer_config, quant_block_list
            self.backend = (
                OutputFormat._format_list[new_format](new_format, scheme, ctx) if new_format else self.backend
            )

        w_fp8 = scheme.data_type.startswith("fp") and scheme.bits == 8
        act_fp8 = scheme.act_data_type.startswith("fp") and scheme.act_bits == 8
        is_block_dynamic_fp8 = (
            self.format_name in ["fp8", "auto_round:fp8"]
            and isinstance(scheme.group_size, tuple)
            and scheme.act_dynamic
        )
        if (w_fp8 or act_fp8) and not is_block_dynamic_fp8:
            error_msg = (
                f"is only supported to export auto_round or llm_compressor format,"
                f" but got {self.format_name}, please check."
            )
            error_msg = ("act_data_type<fp8> " + error_msg) if act_fp8 else error_msg
            error_msg = ("data_type<fp8> " + error_msg) if w_fp8 else error_msg
            raise FormatCompatibilityError(error_msg)

        if (
            scheme.act_bits <= 8
            and (not scheme.is_act_standard_fp() or scheme.act_dynamic)
            and not is_block_dynamic_fp8
        ):
            logger.warning(
                f"{self.format_name} format does not support the current activation quantization configuration,"
                " reset to fake format and save."
            )
            return "fake", scheme, layer_config, quant_block_list

        return None, scheme, layer_config, quant_block_list

    @abstractmethod
    def pack_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_quantized(self, *args, **kwargs):
        pass

    def immediate_pack(self, name: str, model: torch.nn.Module, device: torch.device, **kwargs):
        m = get_module(model, name)
        if not check_to_quantized(m):
            return

        self.pack_layer(name, model, device=device)

    def is_gguf(self) -> bool:
        return "gguf" in self.output_format

    def is_fake(self) -> bool:
        return self.output_format == "fake"

    def is_gptq(self) -> bool:
        return "gptq" in self.output_format or (self.backend is not None and self.backend.is_gptq())

    def is_awq(self) -> bool:
        return "awq" in self.output_format or (self.backend is not None and self.backend.is_awq())

    def is_llm_compressor(self) -> bool:
        return "llm_compressor" in self.output_format or (self.backend is not None and self.backend.is_llm_compressor())
