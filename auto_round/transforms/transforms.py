import math
import torch
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform

import inspect
from typing import Any, Callable, Dict

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

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.group_size = group_size
        self.scale = 1 / math.sqrt(self.group_size)

    def forward(self, x: torch.Tensor):
        # Hadamard transform is it own inverse
        x_shape = x.shape
        return hadamard_transform(x.view(-1, self.group_size), scale=self.scale).view(x_shape)

    def get_transform_matrix(self, device: torch.device = None, dtype: torch.dtype = None):
        return hadamard_transform(torch.eye(self.group_size, device=device, dtype=dtype), scale=1 / math.sqrt(self.group_size))


TRANSFORMS = {
    "identity": IdentityTransform,
    "hadamard": HadamardTransform,
}

def build_transform(transform_class: str, **transform_kwargs):
    transform = TRANSFORMS[transform_class]
    return transform(**filter_kwarg_dict(transform.__init__, transform_kwargs))
