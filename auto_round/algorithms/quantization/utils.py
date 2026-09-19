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

import torch

from auto_round.data_type.base import cache_activation_quantizer
from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized


def register_act_max_hooks(quantizer, model):

    def get_act_max_hook(module, input, output):
        if isinstance(input, (tuple, list)):
            input = input[0]
        if input.numel() == 0:
            return
        activation_quantizer = cache_activation_quantizer(module)
        if activation_quantizer is not None:
            module.act_max = activation_quantizer.observe(input, getattr(module, "act_max", None))

    hook_handles = []
    if isinstance(model, SUPPORTED_LAYER_TYPES):
        if (
            hasattr(model, "act_dynamic")
            and getattr(cache_activation_quantizer(model), "requires_calibration", False)
            and check_to_quantized(model)
        ):
            hook_handles.append(model.register_forward_hook(get_act_max_hook))
        return hook_handles

    for name, module in model.named_modules():
        if (
            hasattr(module, "act_dynamic")
            and getattr(cache_activation_quantizer(module), "requires_calibration", False)
            and check_to_quantized(module)
        ):
            hook_handles.append(module.register_forward_hook(get_act_max_hook))
            continue

        if name in quantizer.layer_config:
            config = quantizer.layer_config[name]
            module.act_dynamic = config.get("act_dynamic", True)
            module.act_data_type = config.get("act_data_type", "int_sym")
            module.act_bits = config.get("act_bits", 16)
            module.act_group_size = config.get("act_group_size", quantizer.act_group_size)
            module.act_sym = config.get("act_sym", True)
            if (
                config["bits"] <= 8
                and getattr(cache_activation_quantizer(module), "requires_calibration", False)
                and check_to_quantized(config)
            ):
                hook_handles.append(module.register_forward_hook(get_act_max_hook))
                continue
    return hook_handles


def register_imatrix_hooks(quantizer, model, *, with_count: bool = False):

    def get_imatrix_hook(module, input, output):
        input = input[0] if isinstance(input, (tuple, list)) else input
        flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
        squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

        if not hasattr(module, "imatrix"):
            module.imatrix = squared
            if with_count:
                module.imatrix_cnt = input.shape[0]
        else:
            module.imatrix += squared.to(module.imatrix.device)
            if with_count:
                module.imatrix_cnt += input.shape[0]

    hook_handles = []
    for _, module in model.named_modules():
        if isinstance(module, quantizer.supported_types) and check_to_quantized(module):
            hook_handles.append(module.register_forward_hook(get_imatrix_hook))
    return hook_handles
