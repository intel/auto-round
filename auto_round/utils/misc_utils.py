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
from __future__ import annotations

import importlib
import os
import re
import sys
from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from auto_round.logger import logger


class LazyImport(object):
    """Lazy import python module till use."""

    def __init__(self, module_name):
        """Init LazyImport object.

        Args:
            module_name (string): The name of module imported later
        """
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        """Get the attributes of the module by name."""
        try:
            self.module = importlib.import_module(self.module_name)
            mod = getattr(self.module, name)
        except:
            spec = importlib.util.find_spec(str(self.module_name + "." + name))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def __call__(self, *args, **kwargs):
        """Call the function in that module."""
        function_name = self.module_name.split(".")[-1]
        module_name = self.module_name.split(f".{function_name}")[0]
        self.module = importlib.import_module(module_name)
        function = getattr(self.module, function_name)
        return function(*args, **kwargs)


auto_gptq = LazyImport("auto_gptq")
htcore = LazyImport("habana_frameworks.torch.core")


def is_debug_mode():
    """Checks if the Python interpreter is running in debug mode.

    Returns:
        bool: True if debugging is enabled, False otherwise.
    """
    return sys.gettrace() is not None or sys.flags.debug == 1


def is_local_path(path):
    """Checks if a given path exists locally.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path exists locally, False otherwise.
    """
    format_list = (
        "json",
        "txt",
    )
    flag = None
    for x in format_list:
        flag = True if x in path else flag
    return flag and os.path.exists(path)


def get_library_version(library_name):
    from packaging.version import Version

    python_version = Version(sys.version.split()[0])
    if python_version < Version("3.8"):
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import pkg_resources  # pylint: disable=E0401

        try:
            version = pkg_resources.get_distribution(library_name).version
            return version
        except pkg_resources.DistributionNotFound:
            return f"{library_name} is not installed"
    else:
        import importlib.metadata  # pylint: disable=E0401

        try:
            version = importlib.metadata.version(library_name)
            return version
        except importlib.metadata.PackageNotFoundError:
            return f"{library_name} is not installed"


def str2bool(v):
    import argparse

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def to_standard_regex(pattern: str) -> str:
    """
    Convert a user-specified string into a standardized regex for layer matching.

    Rules:
    - If the pattern already contains regex tokens ('.*', '^', '$', etc.),
      keep them as-is.
    - Otherwise, wrap the pattern with `.*` on both sides to allow substring matching.
    - Always ensure the returned regex is valid (compilable by re).

    Examples:
    >>> to_standard_regex("model.embed_tokens")
    '.*model\\.embed_tokens.*'
    >>> to_standard_regex("mlp.gate")
    '.*mlp\\.gate.*'
    >>> to_standard_regex("mlp.gate$")
    '.*mlp\\.gate$'
    >>> to_standard_regex("mlp.*gate")
    '.*mlp.*gate.*'
    """
    # Heuristic: if pattern contains regex meta characters, assume partial regex
    meta_chars = {".*", "^", "$", "|", "(", ")", "[", "]", "?", "+"}
    has_regex = any(tok in pattern for tok in meta_chars)
    if not has_regex:
        # Escape literal dots, etc., and wrap with .* for substring matching
        pattern = re.escape(pattern)
        regex = f".*{pattern}.*"
    else:
        # Only escape bare dots that are not already part of regex constructs
        # Avoid double escaping .* sequences
        tmp = []
        i = 0
        while i < len(pattern):
            if pattern[i] == ".":
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    tmp.append(".*")  # keep regex token
                    i += 2
                    continue
                else:
                    tmp.append("\\.")  # escape bare dot
            else:
                tmp.append(pattern[i])
            i += 1
        regex = "".join(tmp)
        # If no anchors are provided, allow substring matching
        if not regex.startswith("^") and not regex.startswith(".*"):
            regex = ".*" + regex
        if not regex.endswith("$") and not regex.endswith(".*"):
            regex = regex + ".*"
    # Validate regex
    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError(f"Invalid regex generated from pattern '{pattern}': {e}")
    return regex


def matches_any_regex(layer_name: str, regex_config: Dict[str, dict]) -> bool:
    """
    Check whether `layer_name` matches any regex pattern key in `regex_config`.
    Args:
        layer_name (str): The layer name to test.
        regex_config (Dict[str, dict]): A mapping of regex patterns to configs.
    Returns:
        bool: True if any pattern matches `layer_name`, otherwise False.
    """
    if not regex_config:
        return False

    for pattern in regex_config:
        # Strip dynamic prefixes (e.g., "+:" or "-:")
        raw_pattern = pattern[2:] if pattern.startswith(("+:", "-:")) else pattern

        try:
            if re.search(raw_pattern, layer_name):
                return True
        except re.error as e:
            logger.warning("Skipping invalid regex pattern %r: %s", pattern, e)
            continue

    return False


def json_serialize(obj: Any):
    """Convert non-JSON-serializable objects into JSON-friendly formats."""
    if isinstance(obj, torch.dtype):
        return str(obj).split(".")[-1]  # e.g., torch.float16 -> "float16"
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
