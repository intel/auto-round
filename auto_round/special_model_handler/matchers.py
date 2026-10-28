# Copyright (c) 2023 Intel Corporation
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
import re

class ModelNameMatcher:
    """model.config.name_or_path"""

    def __init__(self, pattern: str, mode="in"):
        self.pattern = pattern
        self.mode = mode

    def __call__(self, model) -> bool:
        name = getattr(model.config, "name_or_path", "")
        if self.mode == "full":
            return name == self.pattern
        elif self.mode == "in":
            return self.pattern in name
        elif self.mode == "regex":
            return re.search(self.pattern, name) is not None
        else:
            raise ValueError("unsupported mode {self.mode}")

    """Matches config.architectures."""


class ArchitectureMatcher:
    """match config.architectures"""

    def __init__(self, arch: str, mode="in"):
        self.arch = arch
        self.mode = mode

    def __call__(self, model) -> bool:
        archs = getattr(model.config, "architectures", [])
        archs_str = ",".join(archs) if isinstance(archs, list) else str(archs)

        if self.mode == "full":
            return archs_str == self.arch
        elif self.mode == "in":
            return self.arch in archs_str
        elif self.mode == "regex":
            return re.search(self.arch, archs_str) is not None
        else:
            raise ValueError(f"unsupported mode {self.mode}")


class ModelTypeMatcher:
    """match config.architectures"""

    def __init__(self, model_type: str, mode="in"):
        self.model_type = model_type
        self.mode = mode

    def __call__(self, model) -> bool:
        model_type = getattr(model.config, "model_type", None)
        if model_type is None:
            return False

        if self.mode == "full":
            return model_type == self.model_type
        elif self.mode == "in":
            return self.model_type in model_type
        elif self.mode == "regex":
            return re.search(self.model_type, model_type) is not None
        else:
            raise ValueError(f"unsupported mode {self.mode}")
