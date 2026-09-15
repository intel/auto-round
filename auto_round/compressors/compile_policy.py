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

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchCompilePolicy:
    enabled: bool
    reason: str | None = None


def resolve_torch_compile_policy(
    requested: bool,
    *,
    user_specified: bool,
    activation_is_static: bool,
    architecture_off_reason: str | None,
    is_auto_scheme: bool,
    is_rtn: bool,
    iters: int | None,
    min_iters: int,
    components_compatible: bool,
    datatypes_compatible: bool,
) -> TorchCompilePolicy:
    if not requested:
        return TorchCompilePolicy(False)
    if activation_is_static:
        return TorchCompilePolicy(False, "activation is static")
    if architecture_off_reason is not None:
        return TorchCompilePolicy(False, architecture_off_reason)
    if not user_specified and not is_auto_scheme:
        if is_rtn:
            return TorchCompilePolicy(False, "RTN/OPT-RTN quantizes each layer in a single pass")
        if iters is not None and iters < min_iters:
            return TorchCompilePolicy(False, f"`iters`={iters} is below {min_iters}")
    if not components_compatible:
        return TorchCompilePolicy(False, "an algorithm component is incompatible")
    if not datatypes_compatible:
        return TorchCompilePolicy(False, "a datatype preparation callback is incompatible")
    return TorchCompilePolicy(True)
