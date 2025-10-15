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

AUTO_SCHEME_METHODS = {}


def register_scheme_methods(names):
    """Class decorator to register a mixed precision algorithm to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        names: A string. Define the export type.

    Returns:
        cls: The class of register.
    """

    def register(alg):
        if isinstance(names, (tuple, list)):
            for name in names:
                AUTO_SCHEME_METHODS[name] = alg
        else:
            AUTO_SCHEME_METHODS[names] = alg

        return alg

    return register


import auto_round.auto_scheme.haha  # pylint: disable=E0611,E0401
