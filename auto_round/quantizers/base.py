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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


class BaseQuantizer(ABC):
    def __init__(self, compressor: "BaseCompressor"):
        self.compressor = compressor

    def __mro_call(self, method_name: str, *args, **kwargs):
        for cls in type(self).mro():
            method = cls.__dict__.get(method_name, None)
            if method:
                method(self, *args, **kwargs)

    def pre_quantize(self, *args, **kwargs):
        self.__mro_call("_pre_quantize_impl", *args, **kwargs)

    @abstractmethod
    def _pre_quantize_impl(self, *args, **kwargs):
        pass

    def quantize(self, *args, **kwargs):
        self.pre_quantize(*args, **kwargs)
        self._quantize_impl(*args, **kwargs)
        self.post_quantize(*args, **kwargs)
        return self.compressor.model, self.compressor.layer_config

    @abstractmethod
    def _quantize_impl(self, *args, **kwargs):
        pass

    def post_quantize(self, *args, **kwargs):
        self.__mro_call("_post_quantize_impl", *args, **kwargs)

    @abstractmethod
    def _post_quantize_impl(self, *args, **kwargs):
        pass

    def pre_quantize_layer(self, *args, **kwargs):
        self.__mro_call("_pre_quantize_layer_impl", *args, **kwargs)

    @abstractmethod
    def _pre_quantize_layer_impl(self, *args, **kwargs):
        pass

    def quantize_layer(self, *args, **kwargs):
        self.pre_quantize_layer(*args, **kwargs)
        result = self._quantize_layer_impl(*args, **kwargs)
        self.post_quantize_layer(*args, **kwargs)
        return result

    @abstractmethod
    def _quantize_layer_impl(self, *args, **kwargs):
        pass

    def post_quantize_layer(self, *args, **kwargs):
        pass
        self.__mro_call("_post_quantize_layer_impl", *args, **kwargs)

    @abstractmethod
    def _post_quantize_layer_impl(self, *args, **kwargs):
        pass

    def pre_quantize_block(self, *args, **kwargs):
        self.__mro_call("_pre_quantize_block_impl", *args, **kwargs)

    @abstractmethod
    def _pre_quantize_block_impl(self, *args, **kwargs):
        pass

    def quantize_block(self, *args, **kwargs):
        self.pre_quantize_block(*args, **kwargs)
        result = self._quantize_block_impl(*args, **kwargs)
        self.post_quantize_block(*args, **kwargs)
        return result

    @abstractmethod
    def _quantize_block_impl(self, *args, **kwargs):
        pass

    def post_quantize_block(self, *args, **kwargs):
        self.__mro_call("_post_quantize_block_impl", *args, **kwargs)

    @abstractmethod
    def _post_quantize_block_impl(self, *args, **kwargs):
        pass
