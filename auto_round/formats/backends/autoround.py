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

from auto_round.formats.backends.gptq_awq import AutoAWQFormat, AutoGPTQFormat
from auto_round.formats.backends.mlx import MLXFormat
from auto_round.formats.base import BackendDataType, OutputFormat, _check_divisible_by_32
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


@OutputFormat.register("auto_round")
@OutputFormat.register("auto_round:auto_awq")
@OutputFormat.register("auto_round:llm_compressor")
@OutputFormat.register("auto_round:gptqmodel", "auto_round:auto_gptq")
@OutputFormat.register("auto_round:fp8")
@OutputFormat.register("auto_round:mlx")
class AutoRoundFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W4A16_MIXED",
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
        "FP8_BLOCK",
        "MXINT4",
    ]
    format_name = "auto_round"

    def __init__(self, format: str, scheme: QuantizationScheme, ctx: Any):
        self.output_format = "auto_round"
        self.backend = None

        if format == "auto_round":
            if scheme.sym and "int" in scheme.data_type and "mx" not in scheme.data_type:
                self.backend = AutoGPTQFormat("auto_round:auto_gptq", scheme, ctx)
            elif scheme.bits == 4 and not scheme.sym and "int" in scheme.data_type:
                if ctx.layer_config is None:
                    enable_awq = True
                else:
                    enable_awq = all(
                        config["bits"] == scheme.bits or config["bits"] >= 16 for config in ctx.layer_config.values()
                    )
                if enable_awq:
                    self.backend = AutoAWQFormat("auto_round:auto_awq", scheme, ctx)
            elif scheme.is_nv_fp() or scheme.is_mx_fp():
                self.backend = AutoRoundFormat(scheme.data_type, scheme, ctx)
            elif scheme.is_mx_int() and scheme.bits == 4:  # only add mx_int4 now
                self.backend = AutoRoundFormat(scheme.data_type, scheme, ctx)
            elif scheme.is_static_wfp8afp8():  # static wfp8afp8
                self.backend = AutoRoundFormat(BackendDataType.FP8_STATIC.value, scheme, ctx)
            elif scheme.data_type.startswith("fp") and scheme.bits == 8 and scheme.act_bits >= 16:  # woq fp8
                self.backend = AutoRoundFormat(BackendDataType.FP8.value, scheme, ctx)
            elif scheme.data_type.startswith("fp") and scheme.bits == 8 and isinstance(scheme.group_size, tuple):
                self.backend = AutoRoundFormat("auto_round:fp8", scheme, ctx)
            elif scheme.act_bits < 16:
                raise ValueError(
                    "AutoRound format does not support exporting "
                    "for the current quantization configuration, "
                    "please change to `fake` format for research purpose"
                )
        # for auto_round:fp8_static, auto_round:nv_fp etc.
        elif not format.startswith("auto_round"):
            if format == "mlx":
                self.backend = MLXFormat("mlx", scheme, ctx)
            elif format.upper() not in list(BackendDataType.__members__.keys()):
                raise KeyError(f"Unsupported backend format auto_round:{format}, please check")
            else:
                self.output_format = f"auto_round:{format}"
                self.backend = None
        elif format == "auto_round:mlx":
            self.backend = MLXFormat("mlx", scheme, ctx)
        else:
            backend = format.split(":")[1] if ":" in format else None
            self.backend = self._format_list.get(backend)(format, scheme, ctx) if backend else None

        if self.backend is not None:
            self.support_schemes = self.backend.support_schemes

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx: Any):
        layer_config, quant_block_list = ctx.layer_config, ctx.quant_block_list
        if self.backend is not None:
            new_format, scheme, layer_config, quant_block_list = self.backend.check_and_reset_format(scheme, ctx)
            ctx.layer_config, ctx.quant_block_list = layer_config, quant_block_list
            self.backend = (
                OutputFormat._format_list[new_format](new_format, scheme, ctx) if new_format else self.backend
            )

        if scheme.act_bits <= 8:
            if scheme.is_act_standard_fp() and not scheme.act_dynamic:
                if (
                    scheme.act_group_size != 0
                    and not scheme.act_dynamic
                    and self.get_backend_name() == f"auto_round:{BackendDataType.FP8.value}"
                ):
                    logger.warning(
                        f"Please note that quantize activation with act_group_size={scheme.act_group_size}"
                        " may result in failure to export or import normally."
                    )
        if self.backend is None:
            layer_config = _check_divisible_by_32(scheme, ctx.model, layer_config)
        return None, scheme, layer_config, quant_block_list

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.backend is not None:
            return self.backend.pack_layer(layer_name, model, device=device, **kwargs)

        backend = self.get_backend_name()

        if self.output_format in [
            f"auto_round:{BackendDataType.NV_FP.value}",
            f"auto_round:{BackendDataType.MX_FP.value}",
            f"auto_round:{BackendDataType.MX_FP_RCEIL.value}",
            f"auto_round:{BackendDataType.NV_FP4_WITH_STATIC_GS.value}",
        ]:
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import pack_layer

            pack_func = pack_layer
        elif self.output_format in [f"auto_round:{BackendDataType.MX_INT.value}"]:
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import pack_layer

            pack_func = pack_layer
        elif self.output_format in [
            f"auto_round:{BackendDataType.FP8.value}",
            f"auto_round:{BackendDataType.FP8_STATIC.value}",
            f"auto_round:{BackendDataType.FP8_STATIC.value}",
        ]:
            from auto_round.export.export_to_autoround.export_to_fp8 import pack_layer

            pack_func = pack_layer
        else:
            from auto_round.export.export_to_autoround.export import pack_layer

            pack_func = pack_layer
        return pack_func(layer_name, model, backend, device)

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
        if self.backend is not None:
            extra_kwargs = {}
            if isinstance(self.backend, MLXFormat):
                extra_kwargs["autoround_format"] = True
            return self.backend.save_quantized(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                layer_config=layer_config,
                inplace=inplace,
                device=device,
                serialization_dict=serialization_dict,
                **extra_kwargs,
                **kwargs,
            )
        backend = self.get_backend_name()
        if re.search(f"{BackendDataType.MX_FP.value}|{BackendDataType.NV_FP.value}", backend):
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import save_quantized_as_fp

            backend = "auto_round:llm_compressor"
            export_func = save_quantized_as_fp
        elif serialization_dict.get("data_type", "int") == "fp" and serialization_dict.get("bits", 16) == 8:
            from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

            backend = "auto_round:fp8_static" if serialization_dict.get("act_bits", 16) == 8 else None
            export_func = save_quantized_as_autoround
        elif re.search(f"{BackendDataType.MX_INT.value}", backend):
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import save_quantized_as_fp

            backend = "auto_round"
            export_func = save_quantized_as_fp
        else:
            from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

            export_func = save_quantized_as_autoround
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
