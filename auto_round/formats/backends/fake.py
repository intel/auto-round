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

from typing import Any, Callable, Union

import torch

from auto_round.formats.base import OutputFormat
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import copy_python_files_from_model_cache, unsupported_meta_device


@OutputFormat.register("fake")
class FakeFormat(OutputFormat):
    support_schemes = None
    format_name = "fake"

    def check_and_reset_format(
        self, scheme: QuantizationScheme, ctx: Any
    ) -> tuple[None, QuantizationScheme, dict, list]:
        return None, scheme, ctx.layer_config, ctx.quant_block_list

    # fake format will not execute pack_layer.
    def pack_layer(self, *args, **kwargs):
        pass

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
    ):
        if not unsupported_meta_device(model):
            model = model.to("cpu")
            model.save_pretrained(output_dir)
        elif hasattr(model, "config") and model.config is not None:
            model.config.save_pretrained(output_dir)

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        processor = kwargs.get("processor", None)
        if processor is not None:
            processor.save_pretrained(output_dir)
        try:
            copy_python_files_from_model_cache(model, output_dir)
        except Exception as e:
            logger.warning("Skipping source model Python file copy due to error: %s", e)
        return model
