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


class SVDQuantLinear(torch.nn.Module):
    """Linear decomposed into a quantizable residual branch and fp low-rank branch."""

    def __init__(
        self,
        residual_linear: torch.nn.Linear,
        lora_down: torch.nn.Linear,
        lora_up: torch.nn.Linear,
        smooth: torch.Tensor,
    ):
        super().__init__()
        self.residual_linear = residual_linear
        self.lora_down = lora_down
        self.lora_up = lora_up
        self.register_buffer("smooth", smooth.detach().clone())

    def forward(self, x):
        smooth = self.smooth.to(device=x.device, dtype=x.dtype)
        x_hat = x * smooth
        return self.residual_linear(x_hat) + self.lora_up(self.lora_down(x_hat))

