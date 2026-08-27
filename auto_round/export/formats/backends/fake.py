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
from typing import Any, Callable, Union

import torch

from auto_round.export.formats.base import OutputFormat
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import copy_python_files_from_model_cache, unsupported_meta_device


def _serialize_quantization_config_value(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_quantization_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_quantization_config_value(item) for item in value]
    return value


def _build_fake_act_quant_config(serialization_dict: dict | None, layer: torch.nn.Module) -> QuantizationScheme:
    """Build a QuantizationScheme for FakeActQuantLinear from export metadata."""
    config_dict = dict(serialization_dict or {})
    for key in QuantizationScheme.get_attributes():
        if hasattr(layer, key):
            value = getattr(layer, key)
            if value is not None:
                config_dict.setdefault(key, value)
    return QuantizationScheme.from_dict(config_dict)


def _unwrap_wrapper_module(module: torch.nn.Module) -> torch.nn.Module:
    """Strip nested WrapperLinear/WrapperWALayer containers and return the base layer."""
    from auto_round.wrapper import WrapperLinear, WrapperWALayer

    layer = module
    while isinstance(layer, (WrapperLinear, WrapperWALayer)):
        layer = layer.orig_layer
    return layer


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
        has_meta_device = unsupported_meta_device(model)
        if not inplace and not has_meta_device:
            model = copy.deepcopy(model.to("cpu"))

        from auto_round.utils.model import set_module
        from auto_round.wrapper import WrapperLinear, WrapperWALayer

        if not has_meta_device:
            wrapped_modules = [
                (name, module)
                for name, module in model.named_modules()
                if name and isinstance(module, (WrapperLinear, WrapperWALayer))
            ]
            wrapped_names = {name for name, _ in wrapped_modules}
            has_fake_act_quant = False
            for name, module in wrapped_modules:
                if any(name.startswith(f"{parent}.") for parent in wrapped_names if parent != name):
                    continue
                orig_layer = _unwrap_wrapper_module(module)
                act_bits = getattr(orig_layer, "act_bits", None)
                if act_bits is None:
                    act_bits = (serialization_dict or {}).get("act_bits")
                if act_bits is not None and act_bits <= 8:
                    from auto_round.experimental.qmodules.fake import FakeActQuantLinear

                    fake_config = _build_fake_act_quant_config(serialization_dict, orig_layer)
                    replacement = FakeActQuantLinear.from_original(fake_config, orig_layer).to("cpu")
                    has_fake_act_quant = True
                else:
                    replacement = orig_layer.to("cpu")
                set_module(model, name, replacement)

        quantization_config = _serialize_quantization_config_value(dict(serialization_dict or {}))
        quantization_config["quant_method"] = "auto-round"
        if locals().get("has_fake_act_quant", False):
            quantization_config["packing_format"] = "auto_round:fake"
        quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
        from auto_round.export.utils import filter_quantization_config

        filter_quantization_config(quantization_config)
        if hasattr(model, "config") and model.config is not None:
            model.config.quantization_config = quantization_config

        if not has_meta_device:
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
