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

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    AR_LOG_LEVEL: str = "INFO"
    AR_USE_MODELSCOPE: bool = "False"
    AR_MODEL_FREE_SHARD_PARALLELISM: Optional[int] = None
    AUTO_ROUND_CACHE: Optional[str] = None
    AUTO_ROUND_GGUF_AUTO_UPDATE: bool = False
    LLAMA_CPP_ROOT: Optional[str] = None
    AR_AUTO_SCHEME_NSAMPLES: Optional[int] = None
    AR_AUTO_SCHEME_BATCH_SIZE: Optional[int] = None


def _get_optional_positive_int_env(name: str) -> Optional[int]:
    """Read an optional env var that must be a positive integer when set."""
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


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
    "AR_DISABLE_DATASET_SUBPROCESS": lambda: os.getenv("AR_DISABLE_DATASET_SUBPROCESS", "0").lower() in ("1", "true"),
    "AR_DISABLE_COPY_MTP_WEIGHTS": lambda: os.getenv("AR_DISABLE_COPY_MTP_WEIGHTS", "0").lower()
    in ("1", "true", "yes"),
    "AR_ACT_SCALE": lambda: float(os.getenv("AR_ACT_SCALE", "1.0")),
    "AR_ENABLE_ACT_MINMAX_TUNING": lambda: os.getenv("AR_ENABLE_ACT_MINMAX_TUNING", "0").lower()
    in ("1", "true", "yes"),
    "AR_FUSE_ONLINE_ROTATION": lambda: os.getenv("AR_FUSE_ONLINE_ROTATION", "0").lower() in ("1", "true", "yes"),
    # Controls the search range ratio for symmetric int scale search in
    # `auto_round.data_type.int.search_scales`. The search bound is
    # `nmax * AR_SEARCH_SCALE_RATIO` (default None).
    "AR_SEARCH_SCALE_RATIO": lambda: (
        float(os.getenv("AR_SEARCH_SCALE_RATIO")) if os.getenv("AR_SEARCH_SCALE_RATIO") is not None else None
    ),
    # Minimum value to which torch._dynamo cache_size_limit /
    # accumulated_cache_size_limit / recompile_limit are bumped when
    # ``enable_torch_compile`` is used. The default of 16 is enough to cover
    # all distinct linear-weight shapes inside one transformer block (q/k/v/
    # o_proj, gate/up/down_proj, ...) so that per-layer static recompiles do
    # not exceed dynamo's default limit (8) and fall back to eager.
    "AR_DYNAMO_CACHE_SIZE_LIMIT": lambda: int(os.getenv("AR_DYNAMO_CACHE_SIZE_LIMIT", "16")),
    "AR_MODEL_FREE_SHARD_PARALLELISM": lambda: _get_optional_positive_int_env("AR_MODEL_FREE_SHARD_PARALLELISM"),
    "AUTO_ROUND_CACHE": lambda: os.getenv("AUTO_ROUND_CACHE", None),
    "AUTO_ROUND_GGUF_AUTO_UPDATE": lambda: os.getenv("AUTO_ROUND_GGUF_AUTO_UPDATE", "0").lower()
    in ("1", "true", "yes", "on"),
    "LLAMA_CPP_ROOT": lambda: os.getenv("LLAMA_CPP_ROOT", None),
    # Controls the default number of calibration samples used by AutoScheme scoring
    # when ``AutoScheme.nsamples`` is not explicitly set.
    # When unset, AutoScheme uses 16.
    "AR_AUTO_SCHEME_NSAMPLES": lambda: _get_optional_positive_int_env("AR_AUTO_SCHEME_NSAMPLES"),
    # Controls the default batch size used by AutoScheme scoring
    # when ``AutoScheme.batch_size`` is not explicitly set.
    # When unset, AutoScheme uses its built-in heuristic (8 for low GPU memory mode, 1 for normal mode).
    "AR_AUTO_SCHEME_BATCH_SIZE": lambda: _get_optional_positive_int_env("AR_AUTO_SCHEME_BATCH_SIZE"),
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
                # Handle boolean values for boolean env flags
                str_value = "true" if value in [True, "True", "true", "1", 1] else "false"
            else:
                # For other variables, convert to string
                str_value = str(value)

            # Set the environment variable
            os.environ[key] = str_value
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {key!r}")
