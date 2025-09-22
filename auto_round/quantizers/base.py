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

import traceback
import types
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Union

import torch

from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import check_to_quantized, clear_memory

if TYPE_CHECKING:
    from auto_round.compressors import BaseCompressor


class QuantizerType(IntEnum):
    MODE = 1
    MODEL_TYPE = 2
    DATA_TYPE = 3


class BaseQuantizer:
    _quantizer_classes: dict[QuantizerType, dict[str, type[BaseQuantizer]]] = {
        QuantizerType.MODE: {},
        QuantizerType.MODEL_TYPE: {},
        QuantizerType.DATA_TYPE: {},
    }
    compressor: "BaseCompressor" = None

    def __init__(self, compressor: "BaseCompressor"):
        self.compressor = compressor

    @staticmethod
    def quanize(self):
        """Quantize the model and return the quantized model along with layer configurations.The entry of Quantizer.
        Returns:
        The quantized model and layer configurations.
        """
        pass

    def __getattr__(self, name):
        if hasattr(self.compressor, name) and not isinstance(getattr(self.compressor, name), types.MethodType):
            return getattr(self.compressor, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if hasattr(self.compressor, name):
            setattr(self.compressor, name, value)
        else:
            super().__setattr__(name, value)

    @torch.inference_mode()
    def _quantize_embedding_layer(self):
        """Quantizes embedding layers in the model according to the configuration.

        This method iterates through all modules in the model, identifies embedding
        layers specified in `self.layer_config`, and applies the appropriate quantization
        function based on bit precision, grouping strategy, and dtype.

        Returns:
            bool: True if the quantization process completes without critical errors.
        """
        is_quantized = False
        for name, module in self.model.named_modules():
            # Skip non-Embedding modules or layers not in config
            if not isinstance(module, torch.nn.Embedding) or name not in self.layer_config:
                continue

            config = self.layer_config[name]

            # Skip layers that are not marked for quantization
            if not check_to_quantized(config):
                continue
            is_quantized = True
            config["scale_dtype"] = self.scale_dtype
            dtype = config["data_type"]

            # Determine quantization function key with symmetry/asymmetry
            if dtype not in QUANT_FUNC_WITH_DTYPE:
                dtype = f"{dtype}_{'sym' if config['sym'] else 'asym'}"

            # Optionally use optimized rounding (RTN) variant
            if not self.disable_opt_rtn and f"rtn_{dtype}" in QUANT_FUNC_WITH_DTYPE:
                dtype = f"rtn_{dtype}"

            quant_func = QUANT_FUNC_WITH_DTYPE[dtype]

            # Attempt quantization on GPU, fall back to CPU if OOM
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(self.device),
                    **{k: config[k] for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]},
                )
            except RuntimeError as e:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config[k]
                            for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                        },
                    )
                except Exception as e:
                    raise

            # Overwrite the module's weights with the quantized version
            module.weight.data.copy_(weight.cpu())

            # Attach scale and zero point (zp) to the module
            for param_name, value in zip(["scale", "zp"], [scale, zp]):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
                elif isinstance(value, torch.Tensor):
                    setattr(module, param_name, value.cpu())
                else:
                    setattr(module, param_name, value)

            # Update config
            self.layer_config.setdefault(name, {}).update(config)

            # Release memory
            clear_memory()

        return is_quantized
