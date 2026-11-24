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

"""Small dependency-free Lion optimizer for experimental TEQ refinement."""

import torch


class Lion(torch.optim.Optimizer):
    """Lion update rule from Chen et al., restricted to dense TEQ parameters."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                update = exp_avg.lerp(grad, 1 - beta1).sign_()
                parameter.add_(update, alpha=-group["lr"])
                exp_avg.lerp_(grad, 1 - beta2)
        return loss
