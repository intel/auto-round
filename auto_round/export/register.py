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


EXPORT_FORMAT = {}


def register_format(name):
    """Class decorator to register a EXPORT subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the export type.

    Returns:
        cls: The class of register.
    """

    def register(format):
        EXPORT_FORMAT[name] = format
        return format

    return register


PACKING_LAYER_WITH_FORMAT = {}


def register_layer_packing(name):
    """Class decorator to register a EXPORT subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the export type.

    Returns:
        cls: The class of register.
    """

    def register(format):
        PACKING_LAYER_WITH_FORMAT[name] = format
        return format

    return register
