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

from typing import Any, Protocol


class FormatExecutor(Protocol):
    """Execution boundary for packing and saving a resolved output format."""

    def pack_layer(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def save_quantized(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class FormatPolicy(Protocol):
    """Pure compatibility policy associated with an output format."""

    name: str

    def supports(self, scheme: Any) -> bool:
        raise NotImplementedError
