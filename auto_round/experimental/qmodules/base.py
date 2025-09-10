# Copyright (c) 2025 Intel Corporation
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

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch

__all__ = ["QModuleBase"]


class QModuleBase(torch.nn.Module):
    """
    Base class used to describe the weight creation and forward pass
    of different quantization schemes supported by Auto-Round.
    The design is inspired by vLLM's CompressedTensorsScheme:
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_scheme.py

    """

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def from_original(cls, config, original_layer: torch.nn.Module):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.
        """
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        """
        raise NotImplementedError
