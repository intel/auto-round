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

import re
import sys
from typing import TYPE_CHECKING, Callable, Union

import torch

from auto_round.compressors.utils import (
    gguf_args_check,
    is_mx_fp,
    is_nv_fp,
    is_standard_fp,
    is_static_wfp8afp8,
    is_wfp8afp8,
)
from auto_round.export.export_to_autoround import AutoRoundExportFormat
from auto_round.export.export_to_gguf.config import ModelType
from auto_round.schemes import (
    PRESET_SCHEMES,
    QuantizationScheme,
    get_gguf_scheme,
)
from auto_round.utils import SUPPORTED_FORMATS, logger

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


def _check_gguf_compatibility(ar: BaseCompressor, formats: list):
    if len([f for f in formats if f.lower() != "fake"]) > 1:
        raise ValueError(
            f"GGUF format is not compatible with other formats, but got {formats}, please choose only one of them"
        )
    gguf_format_name = get_gguf_scheme(ar.scheme)
    if gguf_format_name:
        if gguf_format_name.lower().endswith("mixed"):
            gguf_format_name = gguf_format_name.lower().replace("_mixed", "_s")
        if any([f.lower() not in ["fake", gguf_format_name.lower()] for f in formats]):
            tmp_format_name = gguf_format_name.lower() if "fake" not in formats else f"{gguf_format_name.lower()},fake"
            logger.warning(
                f"reset format {','.join(formats)} to {tmp_format_name} "
                f"since scheme {gguf_format_name} can only be exported to format {gguf_format_name.lower()} or fake"
            )
            formats = tmp_format_name.split(",")
    return formats


def _check_act_compatibility(ar: BaseCompressor, formats: list[str]) -> list[str]:
    for i in range(len(formats)):
        if formats[i] == "fake":
            continue

        # format check for fp8
        w_fp8 = ar.data_type.startswith("fp") and ar.bits == 8
        act_fp8 = ar.act_data_type.startswith("fp") and ar.act_bits == 8
        if (w_fp8 or act_fp8) and re.search("^auto_round|^llm_compressor", formats[i]) is None:
            error_msg = (
                f"is only supported to export auto_round or llm_compressor format,"
                f" but got {formats[i]}, please check."
            )
            error_msg = ("act_data_type<fp8> " + error_msg) if act_fp8 else error_msg
            error_msg = ("data_type<fp8> " + error_msg) if w_fp8 else error_msg
            logger.error(error_msg)
            sys.exit(-1)

        # Only support to export afp8/nv_fp/mx_fp
        if ar.act_bits <= 8:
            if not is_standard_fp(ar.act_data_type) or ar.act_dynamic:
                if "llm_compressor" in formats[i]:
                    if (is_nv_fp(ar.act_data_type) and "static_gs" in ar.act_data_type) or (is_mx_fp(ar.act_data_type)):
                        continue
                    bits, group_size, sym, act_bits = 8, -1, True, 8
                    assert (
                        ar.bits == bits
                        and ar.group_size == group_size
                        and ar.sym == sym
                        and ar.act_bits == act_bits
                        and ar.act_dynamic
                    ), (
                        f"Currently only support to export llm_compressor format for sym dynamic quantized"
                        f" W{ar.bits}A{ar.act_bits} model with group_size={group_size},"
                        f" but got bits={ar.bits}, group_size={ar.group_size}, sym={ar.sym},"
                        f" act_bits={ar.act_bits}"
                    )
                elif "auto_round" in formats[i] and (
                    is_mx_fp(ar.act_data_type) or (is_nv_fp(ar.act_data_type) and "static_gs" in ar.act_data_type)
                ):
                    pass
                elif formats[i] != "fake":
                    logger.warning(
                        "Currently only support to export auto_round format quantized model"
                        " with fp8, mx_fp and nv_fp4 dtype activation for activation quantization."
                        f" Change format <{formats[i]}> to fake and save."
                    )
                    formats[i] = "fake"
            else:
                if (
                    ar.act_group_size != 0
                    and not ar.act_dynamic
                    and formats[i] == f"auto_round:{AutoRoundExportFormat.FP8.value}"
                ):
                    logger.warning(
                        f"Please note that quantize activation with act_group_size={ar.act_group_size}"
                        " may result in failure to export or import normally."
                    )
        if re.search(r"q\d_k", formats[i]) and not ar.data_type.endswith("_dq"):
            logger.error(
                f"datatype<{ar.data_type}> not support to export {formats[i]} format."
                " Please change export format or `data_type`."
            )
            sys.exit(-1)

    return formats


class OutputFormat:
    support_schemes: list = []
    _format_list: dict[str, OutputFormat] = {}
    format_name = "base"

    def __init__(self, format: str, ar: BaseCompressor):
        if not self.is_support_scheme(ar.scheme):
            logger.error(
                f"Currently, the {self.format_name} format only supports {self.support_schemes}, "
                f"but got scheme {ar.scheme}, please change to fake or auto_round etc."
            )
            exit(-1)
        self.output_format = format
        self.backend = None

    @classmethod
    def register(cls, *names: str) -> Callable[[OutputFormat], OutputFormat]:
        assert names

        def func(output_format: OutputFormat) -> OutputFormat:
            for name in names:
                cls._format_list[name] = output_format
            return output_format

        return func

    @classmethod
    def get_formats(
        cls,
        format: str,
        ar: BaseCompressor,
    ) -> list[OutputFormat]:
        """Get the list of OutputFormat instances based on the provided name."""

        def remove_duplicates(lst):
            seen = set()
            return [x for x in lst if not (x in seen or seen.add(x))]

        formats = format.replace("q*_", f"q{ar.bits}_").replace(" ", "").split(",")
        formats = remove_duplicates(formats)  # need the keep origin order

        # check gguf scheme compatibility
        formats = _check_gguf_compatibility(ar, formats)

        # check activation quantization compatibility
        formats = _check_act_compatibility(ar, formats)

        formats = remove_duplicates(formats)

        for i in range(len(formats)):
            if formats[i].startswith("gguf:"):
                formats[i] = GGUFFormat(formats[i], ar)
            elif formats[i] not in cls._format_list:
                raise KeyError(f"Unsupported format {formats[i]}, please choose from {SUPPORTED_FORMATS}")
            else:
                formats[i] = cls._format_list[formats[i]](formats[i], ar)

        if len(formats) == 1 and formats[0].is_gguf and ar.scale_dtype != torch.float32:
            ar.scale_dtype = torch.float32
            logger.info("change `scale_dtype` to `torch.float32` for gguf format")

        return formats

    @classmethod
    def get_support_matrix(cls: OutputFormat) -> str:
        output_str = ""
        for k, v in cls._format_list.items():
            support_scheme = ", ".join(v.support_schemes).rstrip(",")
            output_str += f"\x1b[31;1m{k}\x1b[0m support scheme:\n\t{support_scheme}\n"
        return output_str

    def get_backend_name(self) -> str:
        return self.backend.output_format if self.backend else self.output_format

    @classmethod
    def is_support_scheme(cls: OutputFormat, scheme: Union[str, QuantizationScheme]) -> bool:
        if scheme in cls.support_schemes:
            return True
        if isinstance(scheme, QuantizationScheme):
            for key in cls.support_schemes:
                if scheme == PRESET_SCHEMES[key]:
                    return True
        return False

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


@OutputFormat.register("fake")
class FakeFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W2A16",
        "W3A16",
        "W8A16",
        "MXFP4",
        "MXFP8",
        "NVFP4",
        "FPW8A16",
        "W2A16G64",
        "W2A16G32",
        "FP8_STATIC",
        "BF16",
        "GGUF:Q4_0",
        "GGUF:Q4_1",
        "GGUF:Q5_0",
        "GGUF:Q5_1",
        "GGUF:Q2_K_S",
        "GGUF:Q3_K_S",
        "GGUF:Q3_K_M",
        "GGUF:Q3_K_L",
        "GGUF:Q4_K_S",
        "GGUF:Q4_K_M",
        "GGUF:Q5_K_S",
        "GGUF:Q5_K_M",
        "GGUF:Q6_K",
        "GGUF:Q8_0",
    ]
    format_name = "fake"


@OutputFormat.register("llm_compressor")
class LLMCompressorFormat(OutputFormat):
    support_schemes = ["MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"]
    format_name = "llm_compressor"

    def __init__(self, format, ar):
        if not self.is_support_scheme(ar.scheme):
            logger.error(
                f"Currently, the llm_compressor format only supports {self.support_schemes}, "
                f"but got scheme {ar.scheme}, please change to fake or auto_round etc."
            )
            exit(-1)
        if is_nv_fp(ar.data_type) or is_mx_fp(ar.data_type):
            from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

            check_compressed_tensors_supported()
            format = format.replace("llm_compressor", f"llm_compressor:{ar.data_type}")
        elif is_static_wfp8afp8(ar):
            format = f"llm_compressor:{AutoRoundExportFormat.FP8_STATIC.value}"
            if ar.act_group_size != 0:
                logger.warning(
                    f"scheme FP8_STATIC export to llm_compressor format only support for act_group_size 0,"
                    f" ,but got act_group_size={ar.act_group_size}, reset = 0"
                )
                ar.act_group_size = 0
            if ar.group_size > 0:
                logger.warning(
                    f"please note that group_size={ar.group_size}"
                    " may not be supported for llm_compressor format, and cannot be loaded in llm_compressor"
                )
        elif not is_wfp8afp8(ar):
            logger.error(
                "Currently, the llm_compressor format only supports MXFP/NVFP/FP8. "
                "Please change format to fake or auto_round etc."
            )
        self.output_format = format
        self.backend = None


@OutputFormat.register("auto_gptq")
class AutoGPTQFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
    format_name = "auto_gptq"


@OutputFormat.register("auto_awq")
class AutoAWQFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
    format_name = "auto_awq"

    def __init__(self, format, ar):
        from auto_round.compressors.utils import check_awq_gemm_compatibility

        awq_supported, info = check_awq_gemm_compatibility(ar.model, ar.bits, ar.group_size, ar.sym, ar.layer_config)
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")
        super().__init__(format, ar)


@OutputFormat.register("itrex")
@OutputFormat.register("itrex_xpu")
class ITREXFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
    format_name = "itrex"


@OutputFormat.register("gguf")
class GGUFFormat(OutputFormat):
    support_schemes = [
        "GGUF:Q4_0",
        "GGUF:Q4_1",
        "GGUF:Q5_0",
        "GGUF:Q5_1",
        "GGUF:Q2_K_S",
        "GGUF:Q3_K_S",
        "GGUF:Q3_K_M",
        "GGUF:Q3_K_L",
        "GGUF:Q4_K_S",
        "GGUF:Q4_K_M",
        "GGUF:Q5_K_S",
        "GGUF:Q5_K_M",
        "GGUF:Q6_K",
        "GGUF:Q8_0",
    ]
    format_name = "gguf"

    def __init__(self, format: str, ar: BaseCompressor):
        gguf_args_check(ar, format, model_type=ModelType.TEXT)
        if ar.mllm:
            gguf_args_check(ar, format, model_type=ModelType.MMPROJ)
        ar.scheme = format.upper()

        self.output_format = format
        self.backend_cls = GGUFFormat
        self.backend = None


@OutputFormat.register("auto_round")
@OutputFormat.register("auto_round:auto_awq")
@OutputFormat.register("auto_round:llm_compressor")
@OutputFormat.register("auto_round:gptqmodel", "auto_round:auto_gptq")
class AutoRoundFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W2A16",
        "W3A16",
        "W8A16",
        "MXFP4",
        "MXFP8",
        "NVFP4",
        "FPW8A16",
        "W2A16G64",
        "W2A16G32",
        "FP8_STATIC",
        "BF16",
    ]
    format_name = "auto_round"

    def __init__(self, format: str, ar: BaseCompressor):
        self.output_format = "auto_round"
        self.backend = None

        if format == "auto_round":
            if ar.sym and "int" in ar.data_type:
                self.backend = AutoGPTQFormat("auto_gptq", ar)
            elif ar.bits == 4 and not ar.sym and "int" in ar.data_type:
                enable_awq = all(
                    config["bits"] == ar.bits or config["bits"] >= 16 for config in ar.layer_config.values()
                )
                if enable_awq:
                    self.backend = AutoAWQFormat("auto_awq", ar)
            elif is_nv_fp(ar.data_type) or is_mx_fp(ar.data_type):
                self.backend = AutoRoundFormat(ar.data_type, ar)
            elif is_static_wfp8afp8(ar):  # static wfp8afp8
                self.backend = AutoRoundFormat(AutoRoundExportFormat.FP8_STATIC.value, ar)
            elif ar.data_type.startswith("fp") and ar.bits == 8 and ar.act_bits >= 16:  # woq fp8
                self.backend = AutoRoundFormat(AutoRoundExportFormat.FP8.value, ar)
            elif ar.act_bits < 16:
                raise ValueError(
                    "AutoRound format does not support exporting "
                    "for the current quantization configuration, "
                    "please change to `fake` format for research purpose"
                )
        elif not format.startswith("auto_round"):
            self.output_format = f"auto_round:{format}"
            self.backend = None
        else:
            backend = format.split(":")[1] if ":" in format else None
            self.backend = self._format_list.get(backend)(format, ar) if backend else None

        if self.backend is not None:
            self.support_schemes = self.backend.support_schemes
