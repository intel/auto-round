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

import os
from typing import Any, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

# Define extra environment variables
extra_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_MXFP4_PRE_UNPACK_WEIGHTS": lambda: os.getenv("VLLM_MXFP4_PRE_UNPACK_WEIGHTS", "1") in ("1", "true", "True"),
    "VLLM_ENABLE_STATIC_MOE": lambda: os.getenv("VLLM_ENABLE_STATIC_MOE", "1") in ("1", "true", "True"),
    "VLLM_AR_MXFP4_MODULAR_MOE": lambda: os.getenv("VLLM_AR_MXFP4_MODULAR_MOE", "0") in ("1", "true", "True"),
    "VLLM_AR_POST_PROCESS_GPTOSS": lambda: os.getenv("VLLM_AR_POST_PROCESS_GPTOSS", "0") in ("1", "true", "True"),
}
# Add the extra environment variables to vllm.envs
import vllm.envs as envs
from vllm.envs import environment_variables

# Merge the environment variables
all_environment_variables = {**environment_variables, **extra_environment_variables}


for name, value_fn in extra_environment_variables.items():
    setattr(envs, name, value_fn())

logger.warning_once(f"Added extra environment variables: {list(extra_environment_variables.keys())}")
