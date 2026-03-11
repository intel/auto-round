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

import inspect
import math
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from auto_round.experimental.transform.utils.hadamard import deterministic_hadamard_matrix
from auto_round.experimental.transform.utils.matrix import apply_transform_weight


def filter_kwarg_dict(fn_or_method: Callable, kwarg_dict: Dict[str, Any]) -> Dict[str, Any]:
    fn_or_method_keys = inspect.signature(fn_or_method).parameters.keys()
    return {k: v for k, v in kwarg_dict.items() if k in fn_or_method_keys}


class IdentityTransform(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

    def remove_parametrizations(self) -> None:
        pass


class HadamardTransform(nn.Module):

    def __init__(
        self,
        transform_block_size: int = 32,
        device: torch.device = None,
        precision: torch.dtype = None,
        location: str = "weight",
        module_type: type[torch.nn.Module] = torch.nn.Linear,
    ):
        super().__init__()
        self.size = transform_block_size
        self.scale = 1 / math.sqrt(self.size)
        self.location = location
        self.module_type = module_type
        self.weight = self._create_weight(self.size, device, precision)

    def _create_weight(
        self,
        size: int,
        device: torch.device = None,
        precision: torch.dtype = None,
    ) -> torch.nn.Parameter:
        data = deterministic_hadamard_matrix(size, precision, device) * self.scale
        # TODO: implement SpinQuant, which rotation matrix is learnable
        return nn.Parameter(data, requires_grad=False)

    def forward(self, x: torch.Tensor):
        # Hadamard transform is it own inverse
        ori_shape = x.shape
        x = x.view(-1, self.size)
        return (
            (
                apply_transform_weight(
                    self.weight.to(device=x.device),
                    x.to(dtype=self.weight.dtype),
                    self.location,
                    self.module_type,
                )
            )
            .to(x.dtype)
            .view(ori_shape)
        )


TRANSFORMS = {
    "identity": IdentityTransform,
    "hadamard": HadamardTransform,
}


def build_transform(transform_type: str, **transform_kwargs):
    transform = TRANSFORMS[transform_type]
    return transform(**filter_kwarg_dict(transform.__init__, transform_kwargs))
