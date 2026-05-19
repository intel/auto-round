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

from auto_round.compressors.utils import check_need_act_calibration
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized


def register_act_max_hooks(quantizer, model):

    def get_act_max_hook(module, input, output):
        if isinstance(input, (tuple, list)):
            input = input[0]
        if input.numel() == 0:
            return
        input, _, _ = reshape_pad_tensor_by_group_size(input, quantizer.act_group_size)
        act_max = torch.max(torch.abs(input), dim=-1).values
        if not hasattr(module, "act_max") or module.act_max.numel() == 0:
            module.act_max = act_max
            if quantizer.config.is_act_nv_fp:
                max_val = act_max.max()
                module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
        else:
            act_max = act_max.to(module.act_max.device)
            if quantizer.config.is_act_nv_fp:
                max_val = torch.max(act_max.max(), module.act_max.max())
                module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                module.act_max = torch.max(act_max, module.act_max)

    hook_handles = []
    if isinstance(model, SUPPORTED_LAYER_TYPES):
        if (
            hasattr(model, "act_dynamic")
            and check_need_act_calibration(model.act_dynamic, model.act_data_type, model.act_bits)
            and check_to_quantized(model)
        ):
            hook_handles.append(model.register_forward_hook(get_act_max_hook))
        return hook_handles

    for name, module in model.named_modules():
        if (
            hasattr(module, "act_dynamic")
            and check_need_act_calibration(module.act_dynamic, module.act_data_type, module.act_bits)
            and check_to_quantized(module)
        ):
            hook_handles.append(module.register_forward_hook(get_act_max_hook))
            continue

        if name in quantizer.layer_config:
            config = quantizer.layer_config[name]
            act_dynamic = config.get("act_dynamic", True)
            act_data_type = config.get("act_data_type", None)
            act_bits = config.get("act_bits", 16)
            if (
                config["bits"] <= 8
                and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
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
