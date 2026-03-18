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


"""Registry for quantization functions keyed by data-type name.

This module provides the ``QUANT_FUNC_WITH_DTYPE`` dictionary that maps
data-type strings (e.g. ``"int_sym"``, ``"fp8"``) to their corresponding
quantization callables, and the :func:`register_dtype` decorator used by
each data-type module to populate that registry.
"""

QUANT_FUNC_WITH_DTYPE = {}


def register_dtype(names):
    """Decorator that registers a quantization function under one or more data-type names.

    Each registered callable is stored in :data:`QUANT_FUNC_WITH_DTYPE` and can
    be looked up at runtime via :func:`~auto_round.data_type.utils.get_quant_func`.

    Args:
        names (str or list or tuple): One or more data-type name strings under
            which the decorated function will be registered (e.g. ``"int_sym"``
            or ``("fp8_sym", "fp8", "fp8_e4m3")``).

    Returns:
        Callable: A decorator that registers the wrapped function and returns it
            unchanged.
    """

    def register(dtype):
        """Register the quantization function and return it unchanged.

        Args:
            dtype (Callable): The quantization function to register.

        Returns:
            Callable: The same function, unmodified.
        """
        if isinstance(names, (tuple, list)):
            for name in names:
                QUANT_FUNC_WITH_DTYPE[name] = dtype
        else:
            QUANT_FUNC_WITH_DTYPE[names] = dtype

        return dtype

    return register
