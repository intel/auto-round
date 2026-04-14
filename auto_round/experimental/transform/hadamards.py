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

from auto_round.experimental.transform.utils.hadamard import deterministic_hadamard_matrix, random_hadamard_matrix
from auto_round.experimental.transform.utils.matrix import apply_transform_weight


def filter_kwarg_dict(fn_or_method: Callable, kwarg_dict: Dict[str, Any]) -> Dict[str, Any]:
    fn_or_method_keys = inspect.signature(fn_or_method).parameters.keys()
    return {k: v for k, v in kwarg_dict.items() if k in fn_or_method_keys}


class HadamardTransform(nn.Module):

    def __init__(
        self,
        block_size: int = 32,
        device: torch.device = None,
        precision: torch.dtype = torch.float32,
        location: str = "weight",
        module_type: type[torch.nn.Module] = torch.nn.Linear,
        inverse: bool = False,
    ):
        """Initialize a Hadamard transform module.

        Args:
            block_size: Size of each Hadamard block. The input tensor is reshaped
                to ``(-1, block_size)`` before applying the transform.
            device: Device on which to create the Hadamard matrix.
            precision: Data type used for the Hadamard matrix weights, using float32 as default.
            location: Target location used by ``apply_transform_weight`` when
                applying the transform.
            module_type: Module type associated with the transform application,
                typically ``torch.nn.Linear``.
            inverse: Whether to build the inverse form of the transform.
        """

        super().__init__()
        self.size = block_size
        self.scale = 1 / math.sqrt(self.size)
        self.location = location
        self.module_type = module_type
        self.inverse = inverse
        self.weight = self._create_weight(self.size, device, precision)

    def _create_weight(
        self,
        size: int,
        device: torch.device = None,
        precision: torch.dtype = torch.float32,
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
                    self.weight.to(x.device),
                    x.to(dtype=self.weight.dtype),
                    self.location,
                    self.module_type,
                )
            )
            .to(x.dtype)
            .view(ori_shape)
        )


class RandomHadamardTransform(HadamardTransform):
    def __init__(
        self,
        block_size: int = 32,
        device: torch.device = None,
        precision: torch.dtype = None,
        location: str = "weight",
        module_type: type[torch.nn.Module] = torch.nn.Linear,
        inverse: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

        super().__init__(
            block_size=block_size,
            device=device,
            precision=precision,
            location=location,
            module_type=module_type,
            inverse=inverse,
        )

    def _create_weight(
        self,
        size: int,
        device: torch.device = None,
        precision: torch.dtype = None,
    ) -> torch.nn.Parameter:
        data = random_hadamard_matrix(size, precision, device, self.generator) * self.scale
        # activation needs transpose
        if self.inverse:
            data = data.T
        # data = deterministic_hadamard_matrix(size, precision, device) * self.scale
        # TODO: implement SpinQuant, which rotation matrix is learnable
        return nn.Parameter(data, requires_grad=False)


HADAMARDS = {
    "hadamard": HadamardTransform,
    "random_hadamard": RandomHadamardTransform,
}


def build_hadamard_transform(hadamard_type: str, **hadamard_kwargs):
    hadamard = HADAMARDS[hadamard_type]
    return hadamard(**filter_kwarg_dict(hadamard.__init__, hadamard_kwargs))
