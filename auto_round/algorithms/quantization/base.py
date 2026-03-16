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
import importlib
import sys
from dataclasses import fields

import torch

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.compressors_new.utils import (
    IndexSampler,
    _get_quantized_layer_names_outside_blocks,
    block_forward,
    check_need_act_calibration,
    check_skippable_keywords,
    collect_best_params,
    get_shared_keys,
    infer_bits_by_data_type,
    init_cache,
    reset_params,
    set_layer_config,
)
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.schemes import (
    QuantizationScheme,
    _handle_special_schemes,
    _parse_scheme,
    get_gguf_scheme,
    preset_name_to_scheme,
)
from auto_round.special_model_handler import get_predefined_ignore_layers, update_module
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    convert_dtype_str2torch,
    find_matching_blocks,
    get_block_names,
    is_quantized_input_module,
)


class BaseQuantizers:

    def __init__(self, config: QuantizationConfig):
        self.scheme = config.scheme
        self.layer_config = config.layer_config
        self.quant_lm_head = config.quant_lm_head
        self.scale_dtype = config.scale_dtype
        self.to_quant_block_names = config.to_quant_block_names
        self.ignore_layers = config.ignore_layers
        self.config = config

    @classmethod
    def from_config(cls, config: QuantizationConfig):
        if cls.__name__ == config._alg_cls:
            return cls(config)
        else:
            module = importlib.import_module("auto_round.algorithms.quantization")
            alg_cls = getattr(module, config._alg_cls)
            return alg_cls(config)

    def post_init(self):
        # should be set after loading model and set layer_config, cause some special scheme need these.
        # Preserve the original, unparsed scheme for later use in auto scheme generation
        # within `configure_layer_config` (which may need the raw value instead of `self.scheme`).

        # # Alternatively, you can use ModelContext.get_context
        self.model_context = ModelContext()
        self.compress_context = CompressContext()

        scheme_fields = {f.name for f in fields(QuantizationScheme)}
        user_scheme_overrides = {}
        for k in scheme_fields:
            v = getattr(self.config, k, None)
            if v is not None:
                user_scheme_overrides[k] = v
        default_scheme, self.is_auto_scheme, final_attrs = _parse_scheme(self.scheme, user_scheme_overrides)

        # Bind attributes to self.config for easy instance-level access
        for key, value in final_attrs.items():
            setattr(self.config, key, value)
        self.config.check_config()

        self.orig_scheme = copy.deepcopy(self.scheme)
        self.scheme = default_scheme

        gguf_scheme_name = get_gguf_scheme(self.scheme)
        # GGUF uses fp32 scale dtype as default
        if self.scale_dtype is None:
            self.scale_dtype = "fp32" if gguf_scheme_name else "fp16"
        self.scale_dtype = convert_dtype_str2torch(self.scale_dtype)

        if not self.is_auto_scheme:
            enable_gguf_official_mixed = True
        else:
            enable_gguf_official_mixed = False

        if not hasattr(self, "quant_block_list"):
            all_blocks = get_block_names(self.model_context.model)
            self.quant_block_list = find_matching_blocks(
                self.model_context.model, all_blocks, self.to_quant_block_names
            )

        self.configure_layer_config(enable_gguf_official_mixed=enable_gguf_official_mixed)

    def _gen_auto_scheme(self) -> dict[str, dict]:
        if self.mllm:
            logger.info("AutoScheme is not yet supported for multimodal LLMs.")
            sys.exit(-1)

        if is_quantized_input_module(self.model_context.model):
            logger.info("AutoScheme does not currently support quantized input models (e.g., FP8).")
            sys.exit(-1)

        all_dtypes = []
        all_gguf = True
        for option in self.orig_scheme.options:
            # Resolve the quantization scheme or data type
            dtype = "int"
            if isinstance(option, str):
                if not option.lower().startswith("gguf"):
                    all_gguf = False

                option = preset_name_to_scheme(option)

            else:
                all_gguf = False

            if isinstance(option, QuantizationScheme):
                dtype = option.data_type
            elif isinstance(option, dict):
                dtype = option.get("data_type", "int")

            all_dtypes.append(dtype)

        # Check for mixed data types
        unique_dtypes = set(all_dtypes)
        if len(unique_dtypes) > 1 and not all_gguf:
            logger.warning(
                "Models with mixed data_types "
                "cannot yet be exported to real formats except GGUF. "
                "Please save the model using the `fake` format for now."
            )

        layer_config, self.has_qlayer_outside_block, self.regex_config = set_layer_config(
            self.model_context.model,
            self.layer_config,
            self.scheme,
            self.scale_dtype,
            self.supported_types,
            self.inner_supported_types,
            self.quant_block_list,
            self.ignore_layers,
            self.quant_lm_head,
            enable_gguf_official_mixed=False,
            is_mllm=self.mllm,
        )
        quant_layer_names = layer_config.keys()
        scheme_keys = {f.name for f in fields(QuantizationScheme)}
        fixed_layer_scheme_new = {
            k: {key: v[key] for key in scheme_keys & v.keys()}
            for k, v in layer_config.items()
            if v.get("fixed_by_user", False)
        }

        # mainly using quant_layers and fixed by users
        from auto_round.auto_scheme.gen_auto_scheme import GenScheme

        if not self.enable_torch_compile and self.super_bits is None and not self.orig_scheme.low_gpu_mem_usage:
            logger.warning("we strongly recommend to set `enable_torch_compile` to True for AutoScheme to save VRAM")
        self.scheme_generator = GenScheme(
            self.orig_scheme,
            self.model_context.model,
            quant_layer_names,
            fixed_layer_scheme_new,
            self.dataset,
            device_map=self.compress_context.device_map,
            tokenizer=self.tokenizer,
            enable_torch_compile=self.enable_torch_compile,
        )
        layer_config = self.scheme_generator.get_layer_config()
        return layer_config

    def configure_layer_config(self, enable_gguf_official_mixed: None | bool = True):

        is_gguf_format = (f := getattr(self, "formats", None)) is not None and len(f) > 0 and f[0].is_gguf()
        if not is_gguf_format:
            predefined_ignore_layers = get_predefined_ignore_layers(self.model_context.model)
            if predefined_ignore_layers:
                logger.info(f"Using predefined ignore_layers: {predefined_ignore_layers}")
                tmp_str = ",".join(predefined_ignore_layers)
                if self.ignore_layers == "":
                    self.ignore_layers = tmp_str
                else:
                    self.ignore_layers += "," + tmp_str

        if self.is_auto_scheme:
            self.layer_config = self._gen_auto_scheme()
        else:
            self.layer_config = _handle_special_schemes(
                self.orig_scheme,
                self.layer_config,
                self.model_context.model,
                supported_types=SUPPORTED_LAYER_TYPES,
                inner_supported_types=INNER_SUPPORTED_LAYER_TYPES,
                quant_lm_head=self.quant_lm_head,
                mllm=self.model_context.is_mllm,
            )

        fill_default_value = True
        if self.is_auto_scheme:
            fill_default_value = False
        self.layer_config, self.has_qlayer_outside_block, self.regex_config = set_layer_config(
            self.model_context.model,
            self.layer_config,
            self.scheme,
            self.scale_dtype,
            SUPPORTED_LAYER_TYPES,
            INNER_SUPPORTED_LAYER_TYPES,
            self.quant_block_list,
            self.ignore_layers,
            self.quant_lm_head,
            enable_gguf_official_mixed=enable_gguf_official_mixed,
            is_mllm=self.model_context.is_mllm,
            fill_default_value=fill_default_value,
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
