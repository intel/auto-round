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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import transformers
from packaging import version

from auto_round.export.export_to_gguf.config import GGUF_CONFIG
from auto_round.logger import logger


def compare_versions(v1, v2):
    return version.parse(v1) >= version.parse(v2)


def torch_version_at_least(version_string):
    return compare_versions(torch.__version__, version_string)


TORCH_VERSION_AT_LEAST_2_6_PRE_RELEASE = torch_version_at_least("2.5.99")
TORCH_VERSION_AT_LEAST_2_6 = torch_version_at_least("2.6.0")
TORCH_VERSION_AT_LEAST_2_5 = torch_version_at_least("2.5.0")
TORCH_VERSION_AT_LEAST_2_4 = torch_version_at_least("2.4.0")


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


class SupportedFormats:

    def __init__(self):
        self._support_format = (
            "auto_round",
            "auto_gptq",
            "auto_awq",
            "auto_round:auto_gptq",
            "auto_round:gptqmodel",
            "auto_round:auto_awq",
            "auto_round:llm_compressor",
            "fake",
            "llm_compressor",
        )
        self._gguf_format = tuple(sorted(GGUF_CONFIG.keys()))
        self._support_list = self._support_format + self._gguf_format

    def __contains__(self, key):
        return True if key in self._support_list else False

    def __str__(self):
        # Return "(%s)" % ', '.join(self._support_format + ("gguf:q*_0", "gguf:q*_1", "gguf:q*_k_s"))
        return "(%s)" % ", ".join(self._support_list)

    def __getitem__(self, key):
        return self._support_list[key]


SHARED_CACHE_KEYS = ("position_ids", "cache_position", "position_embeddings")

deepspeed_exists = False
if importlib.util.find_spec("deepspeed"):  # check if deepspeed is installed
    deepspeed_exists = True

SUPPORTED_DTYPES = ("int", "mx_fp", "fp", "nv_fp")
SUPPORTED_FORMATS = SupportedFormats()
SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D)
# Changed to str as it relies on triton or others lib to load this
INNER_SUPPORTED_LAYER_TYPES = ("FP8Linear",)
# transformers.integrations.finegrained_fp8.FP8Linear
if deepspeed_exists:
    from deepspeed.module_inject import LinearAllreduce, LinearLayer

    SUPPORTED_LAYER_TYPES = SUPPORTED_LAYER_TYPES + (LinearLayer, LinearAllreduce)

MM_KEYS = [
    "multi_modal_projector",
    "vision_tower",
    "multimodal_projector",
    "thinker",
    "visual",
    "audio",
    "talker",
    "token2wav",
    "vision_model",
    "audio_tower",
    "vision_encoder",
    "vision_language_adapter",
    "patch_merger",
    "pre_mm_projector_norm",
    "vision",
]


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


def get_reciprocal(tensor):
    """
    Memory-frugal reciprocal:
    - Inplace operations on original tensor
    - Only allocates small boolean mask
    """
    eps = 1e-5 if tensor.dtype == torch.float16 else 1e-30

    # Create mask for very small elements (small overhead)
    mask = tensor.abs() < eps

    # Prepare output in place: reuse tensor if allowed, otherwise create once
    recip = torch.empty_like(tensor)

    # Safe reciprocal: for nonzero elements
    nonzero_mask = ~mask
    recip[nonzero_mask] = 1.0 / tensor[nonzero_mask]

    # Zero out elements below threshold
    recip[mask] = 0.0

    return recip


def normalize_input(
    decoding_layer_inputs: tuple[Union[list[torch.Tensor], dict, Any], Optional[dict]],
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Normalize the decoding layer inputs into input_ids and other inputs."""
    input_ids = []
    input_others = {"positional_inputs": []}
    key_items = ["attention_mask"]
    for cur_inp in decoding_layer_inputs:
        input_ids.append(cur_inp[0][0][0])
        for key, val in cur_inp[0][1].items():
            if key in key_items:
                if key not in input_others:
                    input_others[key] = []
                input_others[key].append(val)
            else:
                input_others[key] = val
    # Force 'use_cache' to be False
    if "use_cache" in input_others and input_others["use_cache"] is True:
        logger.warning_once("Forcing 'use_cache' to be False during calibration.")
        input_others["use_cache"] = False
    return input_ids, input_others
