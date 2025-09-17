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


class ExtraConfig:
    config_type: str = "base"

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def __init_subclass__(cls):
        if "config_type" not in cls.__dict__:
            raise TypeError(f"Missing property 'config_type' for {cls.__name__!r}")

    def to_dict(self):
        return self.__dict__


class MLLMExtraConfig(ExtraConfig):
    config_type: str = "mllm"

    def __init__(
        self,
        processor=None,
        image_processor=None,
        quant_nontext_module: bool = False,
        extra_data_dir: str = None,
        template: str = None,
    ):
        self.processor = processor
        self.image_processor = image_processor
        self.quant_nontext_module = quant_nontext_module
        self.extra_data_dir = extra_data_dir
        self.template = template
