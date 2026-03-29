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
# Note: the design of this module is inspired by vLLM's envs.py
# For detailed usage and configuration guide, see: docs/environments.md
"""AutoRound runtime environment variable configuration.

This module exposes AutoRound runtime settings as module-level attributes backed
by environment variables.  Attribute access is lazy: each read evaluates the
corresponding lambda at call time so that ``os.environ`` changes are reflected
immediately.

Available settings (with their environment variable names):

    ``AR_LOG_LEVEL`` (str):
        Default logging level. Reads ``$AR_LOG_LEVEL``. Default: ``"INFO"``.
    ``AR_ENABLE_COMPILE_PACKING`` (bool):
        Enable ``torch.compile`` during weight packing. Reads
        ``$AR_ENABLE_COMPILE_PACKING``. Default: ``False``.
    ``AR_USE_MODELSCOPE`` (bool):
        Use ModelScope as the model hub. Reads ``$AR_USE_MODELSCOPE``.
        Default: ``False``.
    ``AR_WORK_SPACE`` (str):
        Working directory for temporary files. Reads ``$AR_WORK_SPACE``.
        Default: ``"ar_work_space"``.
    ``AR_ENABLE_UNIFY_MOE_INPUT_SCALE`` (bool):
        Unify MoE input scale across experts. Reads
        ``$AR_ENABLE_UNIFY_MOE_INPUT_SCALE``. Default: ``False``.
    ``AR_OMP_NUM_THREADS`` (str | None):
        OpenMP thread count. Reads ``$AR_OMP_NUM_THREADS``. Default: ``None``.

Usage::

    import auto_round.envs as envs

    print(envs.AR_LOG_LEVEL)   # "INFO" by default
    envs.set_config(AR_LOG_LEVEL="DEBUG")
    print(envs.AR_LOG_LEVEL)   # "DEBUG"
"""
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    AR_LOG_LEVEL: str = "INFO"
    AR_USE_MODELSCOPE: bool = "False"

environment_variables: dict[str, Callable[[], Any]] = {
    # this is used for configuring the default logging level
    "AR_LOG_LEVEL": lambda: os.getenv("AR_LOG_LEVEL", "INFO").upper(),
    "AR_ENABLE_COMPILE_PACKING": lambda: os.getenv("AR_ENABLE_COMPILE_PACKING", "0").lower() in ("1", "true", "yes"),
    "AR_USE_MODELSCOPE": lambda: os.getenv("AR_USE_MODELSCOPE", "False").lower() in ["1", "true"],
    "AR_WORK_SPACE": lambda: os.getenv("AR_WORK_SPACE", "ar_work_space").lower(),
    "AR_ENABLE_UNIFY_MOE_INPUT_SCALE": lambda: os.getenv("AR_ENABLE_UNIFY_MOE_INPUT_SCALE", "False").lower()
    in ["1", "true"],
    "AR_OMP_NUM_THREADS": lambda: os.getenv("AR_OMP_NUM_THREADS", None),
    "AR_DISABLE_OFFLOAD": lambda: os.getenv("AR_DISABLE_OFFLOAD", "0").lower() in ("1", "true", "yes"),
    "AR_DISABLE_COPY_MTP_WEIGHTS": lambda: os.getenv("AR_DISABLE_COPY_MTP_WEIGHTS", "0").lower()
    in ("1", "true", "yes"),
}


def __getattr__(name: str):
    """Lazily evaluates the requested environment variable.

    Args:
        name (str): Name of the environment variable / module attribute.

    Returns:
        Any: The evaluated value of the corresponding lambda in
        ``environment_variables``.

    Raises:
        AttributeError: If ``name`` is not a known environment variable.
    """
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Returns the list of configurable environment variable names.

    Returns:
        list[str]: All keys in ``environment_variables``.
    """
    return list(environment_variables.keys())


def is_set(name: str):
    """Checks whether an environment variable is explicitly set in the OS environment.

    Args:
        name (str): Environment variable name to check.

    Returns:
        bool: ``True`` if the variable is present in ``os.environ``.

    Raises:
        AttributeError: If ``name`` is not a known environment variable.
    """
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_config(**kwargs):
    """
    Set configuration values for environment variables.

    Args:
        **kwargs: Keyword arguments where keys are environment variable names
                 and values are the desired values to set.

    Example:
        set_config(AR_LOG_LEVEL="DEBUG", AR_USE_MODELSCOPE=True)
    """
    for key, value in kwargs.items():
        if key in environment_variables:
            # Convert value to appropriate string format
            if key == "AR_USE_MODELSCOPE":
                # Handle boolean values for AR_USE_MODELSCOPE
                str_value = "true" if value in [True, "True", "true", "1", 1] else "false"
            else:
                # For other variables, convert to string
                str_value = str(value)

            # Set the environment variable
            os.environ[key] = str_value
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {key!r}")
