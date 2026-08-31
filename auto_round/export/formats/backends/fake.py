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
import glob
import os
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


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
    """Collapse wrapped-layer keys like ``*.orig_layer.weight`` to ``*.weight``."""
    normalized = {}
    changed = False
    for key, value in state_dict.items():
        new_key = key.replace(".orig_layer.", ".")
        if new_key != key:
            changed = True
        # Prefer already-normalized keys if both happen to exist.
        if new_key not in normalized:
            normalized[new_key] = value
    return normalized, changed


def _rewrite_saved_weights_without_orig_layer(output_dir: str) -> None:
    """Rewrite saved checkpoint shards in-place so no ``.orig_layer.`` keys remain."""
    if not os.path.isdir(output_dir):
        return

    safetensor_files = sorted(glob.glob(os.path.join(output_dir, "*.safetensors")))
    for file_path in safetensor_files:
        try:
            from safetensors.torch import load_file as safe_load_file
            from safetensors.torch import save_file as safe_save_file

            state = safe_load_file(file_path)
            normalized, changed = _normalize_state_dict_keys(state)
            if changed:
                safe_save_file(normalized, file_path)
        except Exception as exc:
            logger.warning("Failed to normalize safetensors keys for %s: %s", file_path, exc)

    bin_files = sorted(glob.glob(os.path.join(output_dir, "*.bin")))
    for file_path in bin_files:
        try:
            state = torch.load(file_path, weights_only=True)
            if not isinstance(state, dict):
                continue
            normalized, changed = _normalize_state_dict_keys(state)
            if changed:
                torch.save(normalized, file_path)
        except Exception as exc:
            logger.warning("Failed to normalize pytorch checkpoint keys for %s: %s", file_path, exc)


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
        has_fake_act_quant = False
        logger.warning(
            "Saving fake-quantized model to disk. "
            "Linear replacement is deferred to load-time (via auto_round:fake backend); "
            "save-time now keeps the in-memory quantized model structure unchanged."
        )
        has_meta_device = unsupported_meta_device(model)
        if not inplace and not has_meta_device:
            model = copy.deepcopy(model.to("cpu"))

        config_act_bits = (serialization_dict or {}).get("act_bits")
        if config_act_bits is not None and config_act_bits <= 8:
            has_fake_act_quant = True

        quantization_config = _serialize_quantization_config_value(dict(serialization_dict or {}))
        quantization_config["quant_method"] = "auto-round"
        if has_fake_act_quant:
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

        # Some save flows write wrapper keys first; normalize to plain Linear keys
        # so HF loading does not report UNEXPECTED/MISSING pairs.
        if has_fake_act_quant:
            _rewrite_saved_weights_without_orig_layer(output_dir)

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
