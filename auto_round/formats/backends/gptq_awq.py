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
import transformers

from auto_round.formats.base import OutputFormat, _check_divisible_by_32
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


@OutputFormat.register("auto_gptq", "gptqmodel")
class AutoGPTQFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32", "W4A16_MIXED"]
    format_name = "auto_gptq"

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx: Any):
        if not scheme.sym:
            logger.warning(
                "the asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                " particularly for 2-bit quantization and smaller models."
                " We recommend exporting to either the AutoAWQ format ( only 4 bits) or "
                "the AutoRound format(2/3/4/8 bits)."
            )
        if self.backend is None:
            ctx.layer_config = _check_divisible_by_32(scheme, ctx.model, ctx.layer_config)
        return super().check_and_reset_format(scheme, ctx)

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits not in [2, 3, 4, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.output_format.startswith("auto_round"):
            from auto_round.export.export_to_autoround.export import pack_layer

            pack_layer(layer_name, model, backend=self.output_format, device=device)
        else:
            from auto_round.export.export_to_autogptq.export import pack_layer

            pack_layer(layer_name, model, backend=self.output_format, device=device)

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
        if backend == "auto_round:auto_gptq" or backend == "auto_round:gptqmodel":
            from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

            export_func = save_quantized_as_autoround
        else:
            from auto_round.export.export_to_autogptq.export import save_quantized_as_autogptq

            export_func = save_quantized_as_autogptq
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


@OutputFormat.register("auto_awq")
class AutoAWQFormat(OutputFormat):
    support_schemes = ["W4A16"]
    format_name = "auto_awq"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits != 4:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    @staticmethod
    def check_awq_gemm_compatibility(model, bits, group_size, sym, layer_configs=None):
        """Checks if a model is compatible with the AutoAWQ GEMM kernel.

        Args:
            model: The model object to evaluate, typically a PyTorch model.
            bits (int): The number of bits for quantization (must be 4 for compatibility).
            group_size (int): The group size for quantization.
            sym (bool): Whether symmetric quantization is used (not utilized in the current function logic).
            layer_configs (dict, optional): A dictionary mapping layer names to configurations, where each
                configuration can specify a custom number of bits for the layer.

        Returns:
            tuple: A tuple containing:
                - bool: `True` if the model is compatible, `False` otherwise.
                - str: An error message describing why the model is incompatible, or an empty string if compatible.
        """
        from auto_round.utils.model import get_layer_names_in_block, get_module

        if bits != 4:
            return False, "AutoAWQ GEMM kernel only supports 4 bits"
        for n, m in model.named_modules():
            if type(m) == transformers.pytorch_utils.Conv1D:
                return False, "AutoAWQ GEMM kernel does not support conv1d"

        layer_names = get_layer_names_in_block(model)
        for layer_name in layer_names:
            if (
                layer_configs is not None
                and layer_name in layer_configs.keys()
                and layer_configs[layer_name].get("bits", bits) > 8
            ):
                continue

            layer = get_module(model, layer_name)
            if layer.in_features % group_size != 0:
                return False, f"Layer {layer_name} in_features is not multiple of group_size {group_size}"
            if layer.out_features % (32 // bits) != 0:
                return False, f"Layer {layer_name} out_features is not multiple of 32 // bits"

        return True, ""

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx: Any):
        awq_supported, info = self.check_awq_gemm_compatibility(
            ctx.model, scheme.bits, scheme.group_size, scheme.sym, ctx.layer_config
        )
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")
        if scheme.bits != 4:
            raise ValueError(f"auto_awq format support quantization scheme with W4A16 but got bits={scheme.bits}")

        if self.backend is None:
            ctx.layer_config = _check_divisible_by_32(scheme, ctx.model, ctx.layer_config)

        return super().check_and_reset_format(scheme, ctx)

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.export_to_awq.export import pack_layer

        pack_layer(layer_name, model, backend=self.output_format, device=device)

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
        if backend == "auto_round:auto_awq":
            from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

            export_func = save_quantized_as_autoround
        else:
            from auto_round.export.export_to_awq.export import save_quantized_as_autoawq

            export_func = save_quantized_as_autoawq

        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            backend=backend,
            device=device,
            serialization_dict=serialization_dict,
            **kwargs,
        )
