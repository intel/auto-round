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
from typing import Callable, Union

import torch

from auto_round.formats.base import OutputFormat
from auto_round.schemes import QuantizationScheme


@OutputFormat.register("mlx")
class MLXFormat(OutputFormat):
    support_schemes = ["W2A16", "W2A16G32", "W2A16G64", "W3A16", "W4A16", "W5A16", "W6A16", "W8A16", "BF16"]
    format_name = "mlx"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits not in [2, 3, 4, 5, 6, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if scheme.act_bits != 16:
            error_logs.append(f"act_bits={scheme.act_bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"MLX format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.export_to_mlx.export import pack_layer

        pack_layer(layer_name, model, device=device, **kwargs)

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
        from auto_round.export.export_to_mlx.export import save_quantized_as_mlx

        return save_quantized_as_mlx(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            serialization_dict=serialization_dict,
            **kwargs,
        )
