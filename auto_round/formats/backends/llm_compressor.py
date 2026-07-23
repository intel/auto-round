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

import re
from typing import Any, Callable, Union

import torch

from auto_round.formats.base import BackendDataType, OutputFormat
from auto_round.logger import logger
from auto_round.planning.errors import FormatCompatibilityError
from auto_round.schemes import QuantizationScheme


@OutputFormat.register("llm_compressor")
class LLMCompressorFormat(OutputFormat):
    support_schemes = [
        "MXFP4",
        "MXFP8",
        "NVFP4",
        "FPW8A16",
        "FP8_STATIC",
        "INT8",
        "INT8_W8A8",
        "FP8_BLOCK",
        "W4A16",
        "W8A16",
    ]
    format_name = "llm_compressor"

    def __init__(self, format: str, scheme: QuantizationScheme, ctx: Any):
        if not self.is_support_scheme(scheme):
            raise FormatCompatibilityError(
                f"Currently, the llm_compressor format only supports {self.support_schemes}, "
                f"but got scheme {scheme}, please change to fake or auto_round etc."
            )
        # if format.startswith("llm_compressor"):
        if re.search("^(auto_round:)?llm_compressor", format):
            self.output_format = format
            self.backend = None
            if scheme.is_nv_fp() or scheme.is_mx_fp():
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported(raise_error=True)
                self.backend = LLMCompressorFormat(scheme.data_type, scheme, ctx)
            elif scheme.is_dynamic_afp8() and scheme.is_block_wfp8():
                self.backend = LLMCompressorFormat(BackendDataType.FP8_BLOCK.value, scheme, ctx)
            elif scheme.is_static_wfp8afp8():
                self.backend = LLMCompressorFormat(BackendDataType.FP8_STATIC.value, scheme, ctx)
                if scheme.act_group_size != 0:
                    logger.warning(
                        f"scheme FP8_STATIC export to llm_compressor format only support for act_group_size 0,"
                        f" ,but got act_group_size={scheme.act_group_size}, reset = 0"
                    )
                    scheme.act_group_size = 0
                if scheme.group_size > 0:
                    logger.warning(
                        f"please note that group_size={scheme.group_size}"
                        " may not be supported for llm_compressor format, and cannot be loaded in llm_compressor"
                    )
            elif scheme.is_dynamic_wint8aint8():
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported()
                self.backend = LLMCompressorFormat(BackendDataType.INT8.name, scheme, ctx)
                self.backend.output_format = f"llm_compressor:{BackendDataType.INT8.value}"
            elif scheme.is_wint_woq():
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported()
                self.backend = LLMCompressorFormat(BackendDataType.WINT_A16.value, scheme, ctx)
        else:
            if format.upper() not in list(BackendDataType.__members__.keys()):
                raise KeyError(f"Unsupported backend format llm_compressor:{format}, please check")
            self.output_format = f"llm_compressor:{format}"
            self.backend = None

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits not in [4, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("mxfp|fp|nvfp|int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.data_type == "fp" and scheme.bits != 8:
            error_logs.append(f"data_type={scheme.data_type}, bits={scheme.bits}")
        if scheme.data_type == "int" and scheme.bits not in [4, 8]:
            error_logs.append(f"data_type={scheme.data_type}, bits={scheme.bits}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if isinstance(scheme.group_size, tuple):
            if scheme.bits != 8:
                error_logs.append(f"bits={scheme.bits}")
            if scheme.data_type != "fp":
                error_logs.append(f"data_type={scheme.data_type}")
            if len(scheme.group_size) != 2:
                error_logs.append(f"group_size={scheme.group_size}")
            if not scheme.act_dynamic:
                error_logs.append(f"act_dynamic={scheme.act_dynamic}")
            if not isinstance(scheme.act_group_size, int):
                error_logs.append(f"act_group_size={scheme.act_group_size}")
            if scheme.act_bits != 8:
                error_logs.append(f"act_bits={scheme.act_bits}")
            if scheme.act_data_type != "fp":
                error_logs.append(f"act_data_type={scheme.act_data_type}")
        if error_logs:
            raise ValueError(
                f"LLMCompressor format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def check_and_reset_format(
        self, scheme: QuantizationScheme, ctx: Any
    ) -> tuple[str, QuantizationScheme, dict, list]:
        layer_config, quant_block_list = ctx.layer_config, ctx.quant_block_list
        if self.backend is not None:
            new_format, scheme, layer_config, quant_block_list = self.backend.check_and_reset_format(scheme, ctx)
            ctx.layer_config, ctx.quant_block_list = layer_config, quant_block_list
            self.backend = (
                OutputFormat._format_list[new_format](new_format, scheme, ctx) if new_format else self.backend
            )

        if scheme.act_bits <= 8 and (not scheme.is_act_standard_fp() or scheme.act_dynamic):
            if (scheme.is_act_nv_fp() and "static_gs" in scheme.act_data_type) or scheme.is_act_mx_fp():
                return None, scheme, layer_config, quant_block_list
            elif scheme.is_dynamic_afp8() and scheme.is_block_wfp8():
                return None, scheme, layer_config, quant_block_list
            else:
                bits, group_size, sym, act_bits = 8, -1, True, 8
                assert (
                    scheme.bits == bits
                    and scheme.group_size == group_size
                    and scheme.sym == sym
                    and scheme.act_bits == act_bits
                    and scheme.act_dynamic
                ), (
                    f"Currently only support to export llm_compressor format for sym dynamic quantized"
                    f" W{scheme.bits}A{scheme.act_bits} model with group_size={group_size},"
                    f" but got bits={scheme.bits}, group_size={scheme.group_size}, sym={scheme.sym},"
                    f" act_bits={scheme.act_bits}"
                )
            return None, scheme, layer_config, quant_block_list
        return None, scheme, layer_config, quant_block_list

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.backend is not None:
            return self.backend.pack_layer(layer_name, model, device=device, **kwargs)
        if re.search(f"{BackendDataType.MX_FP.value}|{BackendDataType.NV_FP.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_fp import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{BackendDataType.FP8_STATIC.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_static_fp import pack_layer

            return pack_layer(layer_name, model, self.get_backend_name(), device=device)
        elif re.search(f"{BackendDataType.INT8.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{BackendDataType.FP8_BLOCK.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{BackendDataType.WINT_A16.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export import pack_layer

            return pack_layer(layer_name, model, device=device)
        ## passed as no other llm_compressor format is supported yet
        logger.warning("No other llm_compressor packing format(except NVFP&MXFP) is supported yet, skip packing")
        return

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        backend = self.get_backend_name()
        if re.search(f"{BackendDataType.MX_FP.value}|{BackendDataType.NV_FP.value}", backend):
            from auto_round.export.export_to_llmcompressor.export_to_fp import save_quantized_as_fp

            export_func = save_quantized_as_fp
        elif re.search(f"{BackendDataType.FP8_STATIC.value}", backend):
            from auto_round.export.export_to_llmcompressor.export_to_static_fp import save_quantized_as_static_fp

            export_func = save_quantized_as_static_fp
        else:
            from auto_round.export.export_to_llmcompressor.export import save_quantized_as_llmcompressor

            export_func = save_quantized_as_llmcompressor
        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            backend=backend,
            serialization_dict=serialization_dict,
            **kwargs,
        )
