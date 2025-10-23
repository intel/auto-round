# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

# Define extra environment variables
extra_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_AR_MXFP8_DISABLE_INPUT_QDQ": lambda: os.getenv("VLLM_AR_MXFP8_DISABLE_INPUT_QDQ", "0")
    in ("1", "true", "True"),
    "VLLM_AR_MXFP4_DISABLE_INPUT_QDQ": lambda: os.getenv("VLLM_AR_MXFP4_DISABLE_INPUT_QDQ", "0")
    in ("1", "true", "True"),
    "VLLM_MXFP4_PRE_UNPACK_WEIGHTS": lambda: os.getenv("VLLM_MXFP4_PRE_UNPACK_WEIGHTS", "0") in ("1", "true", "True"),
    "VLLM_ENABLE_STATIC_MOE": lambda: os.getenv("VLLM_ENABLE_STATIC_MOE", "0") in ("1", "true", "True"),
    "VLLM_AR_MXFP4_MODULAR_MOE": lambda: os.getenv("VLLM_AR_MXFP4_MODULAR_MOE", "1") in ("1", "true", "True"),
}
# Add the extra environment variables to vllm.envs
import vllm.envs as envs
from vllm.envs import environment_variables

# Merge the environment variables
all_environment_variables = {**environment_variables, **extra_environment_variables}


for name, value_fn in extra_environment_variables.items():
    setattr(envs, name, value_fn())

logger.warning_once(f"Added extra environment variables: {list(extra_environment_variables.keys())}")
