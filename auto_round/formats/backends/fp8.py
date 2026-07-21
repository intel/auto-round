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

from typing import Callable, Union

import torch

from auto_round.formats.base import OutputFormat
from auto_round.schemes import QuantizationScheme


@OutputFormat.register("fp8")
class FP8Format(OutputFormat):
    support_schemes = ["FP8_BLOCK"]
    format_name = "fp8"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits != 8:
            error_logs.append(f"bits={scheme.bits}")
        if scheme.data_type != "fp":
            error_logs.append(f"data_type={scheme.data_type}")
        if not isinstance(scheme.group_size, tuple):
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
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.export_to_autoround.export_to_fp8 import pack_layer

        pack_layer(layer_name, model, self.output_format, device=device)

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
        from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

        backend = self.get_backend_name()

        if isinstance(serialization_dict["group_size"], tuple):
            serialization_dict["weight_block_size"] = serialization_dict["group_size"]

            ignored_layers = []
            for layer_name, cfg in layer_config.items():
                if cfg["bits"] >= 16 and cfg["act_bits"] >= 16:
                    ignored_layers.append(layer_name)
            if len(ignored_layers) > 0:
                serialization_dict["ignored_layers"] = ignored_layers

        return save_quantized_as_autoround(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            serialization_dict=serialization_dict,
            quant_method=backend,
            **kwargs,
        )
