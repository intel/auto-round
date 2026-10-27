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

import json
import os
import re
from collections.abc import Callable

import torch
import torch.nn as nn

from auto_round.export.base import OutputFormat
from auto_round.export.common.artifacts import save_artifacts
from auto_round.export.common.packing import pack_layers, prepare_model
from auto_round.export.formats.mlx.config import (
    _TEXT_SUB_CONFIG_KEYS,
    _build_mlx_quantization_config,
    _build_text_subconfig_quantization,
    _detect_text_module_prefix,
    _ensure_rope_theta_from_config_obj,
    _flatten_rope_parameters_recursive,
    _preserve_original_model_types,
    _snapshot_original_model_types,
)
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import copy_python_files_from_model_cache, unsupported_meta_device


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
        from auto_round.export.formats.mlx.packing import pack_layer

        pack_layer(layer_name, model, device=device, **kwargs)

    def save_quantized_as_format(
        self,
        output_dir: str,
        model: nn.Module = None,
        tokenizer: Callable | None = None,
        layer_config: dict | None = None,
        inplace: bool = True,
        device: str | torch.device = "cpu",
        serialization_dict: dict | None = None,
        **kwargs,
    ) -> nn.Module:
        """Save quantized model in MLX-compatible format.

        The output can be loaded by mlx-lm::

            from mlx_lm import load, generate
            model, tokenizer = load("output_dir")
            response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
        """
        model = prepare_model(model, inplace, device="cpu")

        os.makedirs(output_dir, exist_ok=True)

        bits = serialization_dict.get("bits", 4) if serialization_dict else 4
        group_size = serialization_dict.get("group_size", 128) if serialization_dict else 128

        # Pack all quantized layers (skip if already packed by immediate_pack)
        if layer_config:
            pack_layers(model, layer_config, self.pack_layer, device)

        # Snapshot ``model_type`` (top-level + sub-configs) from the live
        # ``model.config`` BEFORE HF's ``save_pretrained`` runs. The in-memory
        # config still holds the original strings — newer ``transformers`` only
        # corrupts them on serialize.
        orig_model_types = _snapshot_original_model_types(model)

        # Save model weights (uint32 packed weights are saved directly by safetensors)
        if not unsupported_meta_device(model):
            model.save_pretrained(output_dir, safe_serialization=True)

        # Write config.json with MLX-compatible quantization info
        config_path = os.path.join(output_dir, "config.json")
        if hasattr(model, "config") and not os.path.exists(config_path):
            model.config.save_pretrained(output_dir)

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            # Build mixed-bit aware quantization dict (mlx-community format).
            quant_cfg = _build_mlx_quantization_config(model, default_bits=bits, default_group_size=group_size)
            config["quantization"] = quant_cfg
            autoround_format = kwargs.get("autoround_format", False)
            quant_cfg_full = None
            if autoround_format:
                quant_cfg_full = {
                    "quant_method": "auto-round",
                    "packing_format": "mlx",
                    "bits": bits,
                    "group_size": group_size,
                    "sym": serialization_dict.get("sym", True) if serialization_dict else True,
                    "data_type": serialization_dict.get("data_type", "int") if serialization_dict else "int",
                }
                config["quantization_config"] = quant_cfg_full
            else:
                config["quantization_config"] = quant_cfg

            # ---- VLM support ------------------------------------------------ #
            # If the model has a language-model sub-module (e.g. Qwen-VL family),
            # mlx-vlm's language loader queries the quantization dict under
            # ``text_config`` (or its alias) with keys relative to that sub-module.
            # Mirror our config there so both code paths work.
            text_prefix = _detect_text_module_prefix(model)
            if text_prefix:
                sub_quant = _build_text_subconfig_quantization(quant_cfg, text_prefix)
                for sub_key in _TEXT_SUB_CONFIG_KEYS:
                    sub_cfg = config.get(sub_key)
                    if isinstance(sub_cfg, dict):
                        sub_cfg["quantization"] = sub_quant
                        if autoround_format and quant_cfg_full is not None:
                            sub_cfg["quantization_config"] = quant_cfg_full

            # Flatten rope_parameters recursively so mlx-lm/mlx-vlm find rope_theta
            # at the level it expects (top-level for LLMs, text_config for VLMs).
            _flatten_rope_parameters_recursive(config)

            # Backfill rope_theta from the live model.config object — required by
            # HF's model __init__ for the ``auto_round:mlx`` packing format (the
            # checkpoint is loaded through transformers.AutoModelForCausalLM, which
            # raises KeyError for missing rope_theta on Qwen-family models).
            _ensure_rope_theta_from_config_obj(config, getattr(model, "config", None))

            # Restore ``model_type`` (top-level + sub-configs) from the in-memory
            # snapshot so that newer ``transformers`` versions don't silently
            # rename them on save (e.g. vision_config: qwen3_5 -> qwen3_5_vision).
            _preserve_original_model_types(config, orig_model_types)

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        save_artifacts(output_dir, tokenizer, kwargs.get("processor"), kwargs.get("image_processor"))

        # Try to copy Python files from model cache
        try:
            copy_python_files_from_model_cache(model, output_dir)
        except Exception as e:
            logger.warning(f"Failed to copy Python files from model cache: {e}")

        logger.info(f"Model saved to {output_dir} in MLX format (bits={bits}, group_size={group_size})")
        logger.info(f"Load with: from mlx_lm import load, generate; " f"model, tokenizer = load('{output_dir}')")

        return model
