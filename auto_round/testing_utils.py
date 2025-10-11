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

import importlib.util
import unittest
from functools import wraps
from typing import Callable, Literal

import torch
from transformers.utils.versions import require_version

from auto_round.logger import logger


def is_gguf_available():
    return importlib.util.find_spec("gguf") is not None


def is_autogptq_available():
    return importlib.util.find_spec("auto_gptq") is not None


def is_awq_available():
    return importlib.util.find_spec("awq") is not None


def is_optimum_available():
    return importlib.util.find_spec("optimum") is not None


def is_ipex_available():
    try:
        require_version("intel-extension-for-pytorch>=2.5")
        return True
    except ImportError:
        return False


def is_itrex_available():
    return importlib.util.find_spec("intel_extension_for_transformers") is not None


def is_flash_attn_available():
    return importlib.util.find_spec("flash_attn") is not None


def is_gptqmodel_available():
    try:
        require_version("gptqmodel>=2.0")
        return True
    except ImportError:
        return False


def greater_than_050():
    try:
        require_version("auto-round>=0.5.0")
        return True
    except ImportError:
        return False


def greater_than_051():
    try:
        require_version("auto-round>0.5.1")
        return True
    except ImportError:
        return False


def require_gguf(test_case):
    """
    Decorator marking a test that requires gguf.

    These tests are skipped when gguf isn't installed.

    """
    return unittest.skipUnless(is_gguf_available(), "test requires gguf")(test_case)


def require_autogptq(test_case):
    """
    Decorator marking a test that requires auto-gptq.

    These tests are skipped when auto-gptq isn't installed.

    """
    return unittest.skipUnless(is_autogptq_available(), "test requires auto-gptq")(test_case)


def require_gptqmodel(test_case):
    """
    Decorator marking a test that requires gptqmodel.

    These tests are skipped when gptqmodel isn't installed.

    """
    return unittest.skipUnless(is_gptqmodel_available(), "test requires gptqmodel>=2.0")(test_case)


def require_awq(test_case):
    """
    Decorator marking a test that requires autoawq.

    These tests are skipped when autoawq isn't installed.

    """
    return unittest.skipUnless(is_awq_available(), "test requires autoawq")(test_case)


def require_ipex(test_case):
    """
    Decorator marking a test that requires intel-extension-for-pytorch.

    These tests are skipped when intel-extension-for-pytorch isn't installed.

    """
    return unittest.skipUnless(is_ipex_available(), "test requires intel-extension-for-pytorch>=2.5")(test_case)


def require_itrex(test_case):
    """
    Decorator marking a test that requires intel-extension-for-transformers.

    These tests are skipped when intel-extension-for-transformers isn't installed.

    """
    return unittest.skipUnless(is_itrex_available(), "test requires intel-extension-for-transformers")(test_case)


def require_optimum(test_case):
    """
    Decorator marking a test that optimum.

    These tests are skipped when optimum isn't installed.

    """
    return unittest.skipUnless(is_optimum_available(), "test requires optimum")(test_case)


def require_greater_than_050(test_case):
    """
    Decorator marking a test that requires auto-round>=0.5.0.

    These tests are skipped when auto-round<0.5.0.

    """
    return unittest.skipUnless(greater_than_050(), "test requires auto-round>=0.5.0")(test_case)


def require_greater_than_051(test_case):
    """
    Decorator marking a test that requires auto-round>0.5.1

    These tests are skipped when auto-round<=0.5.1.

    """
    return unittest.skipUnless(greater_than_051(), "test requires auto-round>0.5.1")(test_case)


def multi_card(test_case):
    """
    Decorator marking a test that requires multi cards.

    These tests are skipped when use only one card or cpu.

    """
    return unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() > 1, "test requires multiple cards."
    )(test_case)


def require_old_version(test_case):
    """
    Decorator marking a test that requires old version of transformers and torch.

    These tests are skipped when not use special version.

    """
    env_check = True
    try:
        require_version("torch<2.7.0")
        env_check &= True
    except ImportError:
        env_check &= False
    return unittest.skipUnless(env_check, "Environment is not satisfactory")(test_case)


def require_vlm_env(test_case):
    """
    Decorator marking a test that requires some special env to load vlm model.

    These tests are skipped when not meet the environment requirements.

    """

    env_check = True
    # pip install flash-attn --no-build-isolation
    env_check &= is_flash_attn_available()

    # pip install git+https://github.com/haotian-liu/LLaVA.git@v1.2.2
    env_check &= importlib.util.find_spec("llava") is not None

    env_check &= importlib.util.find_spec("xformers") is not None

    return unittest.skipUnless(env_check, "Environment is not satisfactory")(test_case)


def require_package_version(
    package: str, version_spec: str, on_fail: Literal["skip", "warn", "error"] = "skip"
) -> bool:
    """
    Check if a package satisfies a version requirement.

    Args:
        package (str): Name of the Python package (e.g., "transformers").
        version_spec (str): Version specifier string (e.g., ">=0.5.0", "<4.57.0").
        on_fail (str, optional): Behavior when requirement is not met:
            - "skip": return False (used for skipping tests)
            - "warn": issue a warning but return True
            - "error": raise ImportError
        Defaults to "skip".

    Returns:
        bool: True if requirement satisfied or on_fail=="warn", False if skipped
    """
    try:
        require_version(f"{package}{version_spec}")
        return True
    except ImportError as e:
        msg = f"Requirement not satisfied: {package}{version_spec}"
        if on_fail == "skip":
            return False
        elif on_fail == "warn":
            logger.warning_once(msg)
            return True
        else:  # on_fail == "error"
            raise ImportError(msg) from e


def require_package_version_ut(package: str, version_spec: str) -> Callable:
    """
    Decorator for unittest.TestCase methods to enforce a package version requirement.

    The decorated test will be skipped if the package version does not satisfy the requirement.

    Args:
        package (str): Name of the package (e.g., "transformers", "auto-round").
        version_spec (str): Version specifier string (e.g., ">=0.5.0", "<4.57.0").

    Returns:
        Callable: A decorator to wrap unittest test methods.
    """

    def decorator(test_func: Callable) -> Callable:
        reason = f"Test requires {package}{version_spec}"
        return unittest.skipUnless(require_package_version(package, version_spec, on_fail="skip"), reason)(test_func)

    return decorator
