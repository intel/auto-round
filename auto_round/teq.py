# Copyright (c) 2024 Intel Corporation
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

from auto_round import utils


def get_scale_param_from_block(block: torch.nn.Module):

    scale_params = []
    for name, mod in block.named_modules():
        if hasattr(mod, "weight_scale_calculator"):
            scale_params.extend(mod.weight_scale_calculator.parameters())
    return scale_params


def _transform_weight(weight, weight_scale):
    # Apply weight scale to weight
    updated_weight = weight * weight_scale.reshape(1, -1)
    return updated_weight


def _transform_input(x, weight_scale):
    """
    Format:
        weight: [OC, IC]
        weight_scale: [IC]
        x: [..., IC]
    """
    input_scale_target_shape = (1,) * (len(x.shape) - 1) + (-1,)
    input_scale_for_x = weight_scale.reshape(input_scale_target_shape)
    updated_x = torch.div(x, input_scale_for_x)
    return updated_x


def equalization_transform(weight, x, weight_scale):
    """
    Format:
        weight: [OC, IC]
        weight_scale: [IC]
        x: [..., IC]

    Equivalent transformation:
        y = x @ weight.T + bias
        y = (x * weight_scale^-1) @ (weight_scale * weight.T) + bias
    """
    updated_x = _transform_input(x, weight_scale)
    updated_weight = _transform_weight(weight, weight_scale)
    return updated_weight, updated_x


class MulLinear(torch.nn.Module):
    def __init__(self, module, weight_scale=None):
        super().__init__()
        if weight_scale is None:
            weight_scale = torch.ones(module.in_features)
        self.register_buffer("weight_scale", weight_scale)
        utils.logger.debug(f"Original module weight shape: {module.weight.shape}.")
        module.weight *= weight_scale.reshape(1, -1)
        self.add_module("linear", module)
        utils.logger.debug(f"MulLinear: {module} has been wrapped as `MulLinear`.")

    def forward(self, X):
        updated_x = _transform_input(X, self.weight_scale)
        y = self.linear(updated_x)
        return y

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, weight):
        self.linear.weight = weight

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, bias):
        self.linear.bias = bias


def replace_linear_with_smoothed_linear(module, weight_scale):
    utils.logger.debug(
        (
            f"Replace {module} with `MulLinear`, weight_scale shape: {weight_scale.shape}, "
            f"weight scale min: {weight_scale.min()}, weight scale max: {weight_scale.max()}"
        )
    )
    return MulLinear(module, weight_scale)


def get_weight_scale(weight_data):
    assert len(weight_data.shape) == 2, f"weight_data shape len should be 2, got {weight_data.shape}"
    alpha = 0.5
    weight_amax = torch.max(torch.abs(weight_data), dim=0).values
    norm_weight_amax = torch.pow(weight_amax, alpha)
    norm_weight_amax = norm_weight_amax.reshape(1, -1)
    norm_weight_amax_clip = norm_weight_amax + 1
    return norm_weight_amax_clip.to(weight_data.device)


# Vanilla ScaleCalculator
class ScaleCalculatorV(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, device):
        """
        Format:
            weight: [OC, IC]
            scale1: [IC]

        """
        super().__init__()
        shape = weight.shape[1]
        device = device
        self.scale1 = torch.nn.Parameter(torch.ones(shape, device=device), requires_grad=True)

    def forward(self, x):
        update_scale = self.scale1
        return update_scale

    def get_final_scale(self):
        return self.scale1


ScaleCalculator = ScaleCalculatorV
