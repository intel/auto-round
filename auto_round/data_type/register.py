# Copyright (c) 2024 Intel Corporation
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


QUANT_FUNC_WITH_DTYPE = {}


def register_dtype(names):
    """Class decorator to register a EXPORT subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        names: A string. Define the export type.

    Returns:
        cls: The class of register.
    """

    def register(dtype):
        if isinstance(names, (tuple, list)):
            for name in names:
                QUANT_FUNC_WITH_DTYPE[name] = dtype
        else:
            QUANT_FUNC_WITH_DTYPE[names] = dtype

        return dtype

    return register
