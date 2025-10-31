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

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    AR_LOG_LEVEL: str = "INFO"
    AR_USE_MODELSCOPE: bool = "False"

environment_variables: dict[str, Callable[[], Any]] = {
    # this is used for configuring the default logging level
    "AR_LOG_LEVEL": lambda: os.getenv("AR_LOG_LEVEL", "INFO").upper(),
    "AR_USE_MODELSCOPE": lambda: os.getenv("AR_USE_MODELSCOPE", "False").lower() in ["1", "true"],
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
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
