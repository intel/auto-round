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
import importlib
import traceback

import torch

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.compressors_new.utils import (
    check_need_act_calibration,
)
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
)


class BaseQuantizers:
    # Class-level attribute declarations for convenient access in quantization methods.
    # Scheme-related attrs (layer_config, scale_dtype, has_qlayer_outside_block, etc.)
    # are resolved by SchemeMixin in BaseCompressor and synced here after post_init().
    model_context = None
    compress_context = None
    dataset = None
    supported_types = SUPPORTED_LAYER_TYPES
    inner_supported_types = INNER_SUPPORTED_LAYER_TYPES

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.layer_config = config.layer_config
        self.bits = config.bits
        self.group_size = config.group_size
        self.sym = config.sym
        self.data_type = config.data_type
        self.act_bits = config.act_bits
        self.act_group_size = config.act_group_size
        self.act_sym = config.act_sym
        self.act_data_type = config.act_data_type
        self.act_dynamic = config.act_dynamic
        self.super_bits = config.super_bits
        self.super_group_size = config.super_group_size
        self.scale_dtype = config.scale_dtype
        self.ignore_layers = config.ignore_layers
        self.quant_lm_head = config.quant_lm_head
        self.to_quant_block_names = config.to_quant_block_names

    @classmethod
    def from_config(cls, config: QuantizationConfig):
        if cls.__name__ == config._alg_cls:
            return cls(config)
        else:
            module = importlib.import_module("auto_round.algorithms.quantization")
            alg_cls = getattr(module, config._alg_cls)
            return alg_cls(config)

    @property
    def formats(self):
        return getattr(self.compress_context, "formats", None)

    @property
    def amp(self):
        return getattr(self.model_context, "amp", False)

    @property
    def amp_dtype(self):
        import torch

        return getattr(self.model_context, "amp_dtype", torch.float32)

    def resolve_scheme(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "resolve_scheme() has been moved to BaseCompressor in compressors_new/base.py. "
            "Call BaseCompressor.post_init() instead."
        )

    def post_init(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "post_init() has been moved to BaseCompressor/_scheme_post_init() in "
            "compressors_new/base.py. Call BaseCompressor.post_init() instead."
        )

    def _register_act_max_hook(self, model):

        def get_act_max_hook(module, input, output):
            if isinstance(input, (tuple, list)):
                input = input[0]
            if input.numel() == 0:
                return  # as no needs for act_max update
            input, _, _ = reshape_pad_tensor_by_group_size(input, self.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if self.config.is_act_nv_fp:  ## for nvfp per-tensor input_global_scale calculation usage
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                act_max = act_max.to(module.act_max.device)
                if self.config.is_act_nv_fp:  ## for nvfp per-tensor input_global_scale calculation usage
                    max_val = torch.max(act_max.max(), module.act_max.max())
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                else:
                    module.act_max = torch.max(act_max, module.act_max)

        hook_handles = []
        # for single layers out of blocks, like lm_head
        if isinstance(model, SUPPORTED_LAYER_TYPES):
            m = model
            if (
                hasattr(m, "act_dynamic")
                and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
                and check_to_quantized(m)
            ):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
            return hook_handles

        for n, m in model.named_modules():
            if (
                hasattr(m, "act_dynamic")
                and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
                and check_to_quantized(m)
            ):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
                continue

            # for whole model, RTN
            if n in self.layer_config:
                config = self.layer_config[n]
                act_dynamic = config.get("act_dynamic", True)
                act_data_type = config.get("act_data_type", None)
                act_bits = config.get("act_bits", 16)
                if (
                    config["bits"] <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(config)
                ):
                    hook = m.register_forward_hook(get_act_max_hook)
                    hook_handles.append(hook)
                    continue
        return hook_handles

    @torch.inference_mode()
    def _quantize_embedding_layer(self):
        """Quantizes embedding layers in the model according to the configuration.

        This method iterates through all modules in the model, identifies embedding
        layers specified in `self.quantizer.layer_config`, and applies the appropriate quantization
        function based on bit precision, grouping strategy, and dtype.

        Returns:
            bool: True if the quantization process completes without critical errors.
        """
        is_quantized = False
        for name, module in self.model_context.model.named_modules():
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

            quant_func = QUANT_FUNC_WITH_DTYPE[dtype]
            dtype = module.weight.dtype
            # As typically float32 are used in RTN to search scale zp,
            # to avoid cache a bf16 copy we'd better use float32
            if config.get("super_group_size", None) is not None:
                dtype = torch.float32

            # Attempt quantization on GPU, fall back to CPU if OOM
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(dtype=dtype, device=self.compress_context.device),
                    **{
                        k: config.get(k, None)
                        for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                    },
                )
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config.get(k, None)
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
            del weight
            del scale
            del zp
            clear_memory(device_list=self.compress_context.device_list)

        return is_quantized

    def quantize_block(
        self, block: torch.nn.Module, input_ids=None, input_others=None, reference_output=None, **kwargs
    ) -> dict:
        """Apply the quantization algorithm to a prepared block.

        This is the **pure-algorithm** entry point called by the Compressor after
        all infrastructure work (device placement, data collection, act-max hook
        registration, DDP setup) has been completed.

        Implementations should:
        - Perform the algorithm-specific weight/activation quantization on ``block``.
        - Return a dict of best parameters (may be empty for zero-shot algorithms).

        Args:
            block: Module already placed on the correct device(s).
            input_ids: Calibration inputs on cache_device (None for zero-shot RTN).
            input_others: Additional inputs (None for zero-shot RTN).
            reference_output: FP reference outputs collected by Compressor
                (None for algorithms that don't need a reconstruction loss).
            **kwargs: Algorithm-specific keyword arguments (e.g. ``loss_device``,
                ``card_0_in_high_risk`` for ARQuantizer).

        Returns:
            dict: Best quantization parameters found, or ``{}`` if not applicable.
        """
        raise NotImplementedError("quantize_block must be implemented in subclasses of BaseQuantizers")

    def quantize_layer(self, layer_name: str, **kwargs):
        """Quantizes a single layer of the model.

        Args:
            layer_name (str): The name of the layer to quantize. The layer module is
                retrieved internally via get_module(model, layer_name).
        """
        raise NotImplementedError("quantize_layer must be implemented in subclasses of BaseQuantizers")
