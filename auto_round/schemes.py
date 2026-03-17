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
import copy
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from auto_round.logger import logger
from auto_round.utils import SUPPORTED_DTYPES, infer_bits_by_data_type

__all__ = ["QuantizationScheme", "get_gguf_scheme", "preset_name_to_scheme"]

if TYPE_CHECKING:
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme


@dataclass
class QuantizationScheme:
    bits: int = 4
    group_size: int = 128
    sym: bool = True
    data_type: str = "int"
    act_bits: Optional[int] = None
    act_group_size: Optional[int] = None
    act_sym: Optional[bool] = None
    act_data_type: Optional[str] = None
    act_dynamic: Optional[bool] = None
    super_bits: Optional[int] = None
    super_group_size: Optional[int] = None

    @classmethod
    def from_dict(cls, config: dict):
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config.items() if k in field_names}
        return cls(**filtered_config)

    @classmethod
    def get_attributes(cls: "QuantizationScheme") -> list[str]:
        return [field.name for field in fields(cls)]

    def __getitem__(self, key: str):
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        return getattr(self, key)

    def __setitem__(self, key: str, value: None | int | str):
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        setattr(self, key, value)

    def items(self):
        return ((field, getattr(self, field)) for field in self.get_attributes())

    def keys(self):
        return self.get_attributes()

    def values(self):
        return (getattr(self, field) for field in self.get_attributes())

    def get(self, key: str, default=None):
        if key not in self.get_attributes():
            return default
        res = getattr(self, key)
        # In case the attribute is explicitly set to None, return default
        if res is None:
            return default
        return getattr(self, key)

    def __eq__(self, other: "QuantizationScheme") -> bool:
        if not isinstance(other, QuantizationScheme):
            return False
        skip_act_check = False
        self_act_bits = 16 if self.act_bits is None else self.act_bits
        other_act_bits = 16 if other.act_bits is None else other.act_bits
        if self_act_bits == other_act_bits and other_act_bits >= 16:
            skip_act_check = True

        for field in self.get_attributes():
            if skip_act_check and field.startswith("act_"):
                continue
            if getattr(self, field) != getattr(other, field):
                return False
        return True


def preset_name_to_scheme(name: str) -> QuantizationScheme:
    """Get a QuantizationScheme instance from a preset scheme name."""
    name = name.upper()

    if name not in PRESET_SCHEMES:
        raise KeyError(f"Unknown preset scheme name {name}, " f"available names: {list(PRESET_SCHEMES.keys())}")

    scheme_args = deepcopy(PRESET_SCHEMES[name])
    return scheme_args


def is_preset_scheme(name: str) -> bool:
    """Check if the given name is a preset scheme name."""
    return name.upper() in PRESET_SCHEMES


def _reconcile_bits_and_dtype(config: dict, prefix: str = ""):
    """
    Harmonizes 'bits' and 'data_type' for weights or activations.
    Ensures internal consistency by prioritizing data_type inference.
    """

    dt_key = f"{prefix}data_type"
    bits_key = f"{prefix}bits"

    if config.get(dt_key) is None:
        return

    # Infer the correct bit-width based on the data_type string
    inferred_bits = infer_bits_by_data_type(config[dt_key])

    if inferred_bits is not None and inferred_bits < 16:
        # Check for conflict between user-specified bits and inferred bits
        if inferred_bits != config.get(bits_key):
            logger.warning(f"'{dt_key}' does not match '{bits_key}'. " f"Resetting '{bits_key}' to {inferred_bits}.")
            config[bits_key] = inferred_bits

        # Normalize data_type (e.g., 'mx_fp4' -> 'mx')
        for supported in SUPPORTED_DTYPES:
            if config[dt_key] == f"{supported}{inferred_bits}":
                config[dt_key] = supported
                break


def _override_scheme_with_user_specify(
    scheme: Union[str, dict, QuantizationScheme], user_scheme_overrides: dict[str, Any], return_str=True
) -> Union[str, QuantizationScheme]:
    """
    Updates a base quantization scheme with user-provided overrides.
    Handles GGUF formatting and synchronizes weight/activation parameters.
    """
    # 1. GGUF special handling: map data_type suffix to GGUF scheme names
    dt_override = user_scheme_overrides.get("data_type", "")
    if (
        isinstance(scheme, QuantizationScheme) or (isinstance(scheme, str) and not scheme.startswith("gguf"))
    ) and dt_override.endswith("_dq"):
        if "bits" not in user_scheme_overrides:
            raise KeyError(f"Must specify 'bits' when using data_type={dt_override}")

        bits = user_scheme_overrides["bits"]
        suffix = "k" if bits == 6 else "k_s"
        scheme = f"gguf:q{bits}_{suffix}"

    # 2. Convert input scheme to a dictionary for processing
    if isinstance(scheme, QuantizationScheme):
        scheme_dict = asdict(scheme)
    elif isinstance(scheme, str):
        normalized_name = scheme.strip("'\" ").upper()
        if normalized_name.startswith("GGUF") and len(user_scheme_overrides) > 0:
            logger.warning_once(
                "When using GGUF scheme, user-specified overrides will be ignored to ensure format compatibility."
            )
            user_scheme_overrides = {}
        # If no overrides exist, return the normalized string immediately
        if not user_scheme_overrides and return_str:
            return normalized_name
        scheme_dict = asdict(preset_name_to_scheme(normalized_name))
    else:
        scheme_dict = scheme.copy()

    # 3. Apply overrides and define default behaviors
    scheme_dict.update(user_scheme_overrides)

    if scheme_dict.get("act_dynamic") is None:
        scheme_dict["act_dynamic"] = True

    # 4. Reconcile weight settings (bits vs data_type)
    _reconcile_bits_and_dtype(scheme_dict)

    # 5. Fallback logic: Inherit activation settings from weight settings
    scheme_dict["act_group_size"] = (
        scheme_dict.get("act_group_size")
        if scheme_dict.get("act_group_size") is not None
        else scheme_dict.get("group_size")
    )
    scheme_dict["act_bits"] = scheme_dict.get("act_bits") or 16
    scheme_dict["act_sym"] = (
        scheme_dict.get("act_sym") if scheme_dict.get("act_sym") is not None else scheme_dict.get("sym")
    )

    # 6. Activation data_type logic
    if scheme_dict.get("act_data_type") is None:
        is_supported = scheme_dict["data_type"] in SUPPORTED_DTYPES
        if is_supported and scheme_dict["act_bits"] < 16:
            scheme_dict["act_data_type"] = scheme_dict["data_type"]
            logger.info(f"Activation adopting weight data_type: {scheme_dict['data_type']}")
        else:
            scheme_dict["act_data_type"] = "float"

    # 7. Reconcile activation settings
    _reconcile_bits_and_dtype(scheme_dict, prefix="act_")

    return QuantizationScheme.from_dict(scheme_dict)


def _parse_scheme(
    scheme: Union[str, dict, QuantizationScheme, "AutoScheme"], user_scheme_overrides: dict[str, Any]
) -> tuple[Union[str, QuantizationScheme], bool]:
    """
    Parses the final scheme.
    """
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    is_auto_scheme = isinstance(scheme, AutoScheme)
    if is_auto_scheme:
        if not scheme.options:
            raise ValueError("AutoScheme options cannot be empty")
        else:
            for option in scheme.options:
                if isinstance(option, str):
                    if "mixed" in option:
                        raise ValueError(f"Mixed option {option} is not supported")

        # Map user overrides across all auto-scheme options
        scheme.options = [_override_scheme_with_user_specify(opt, user_scheme_overrides) for opt in scheme.options]

        # Select the primary scheme for attribute binding (skipping BF16)
        default_scheme = scheme.options[0]
        for opt in scheme.options:
            if opt == "BF16":
                continue
            if isinstance(opt, QuantizationScheme):
                if opt.bits < 16 or (opt.act_bits and opt.act_bits < 16):
                    default_scheme = opt
                    break
    else:
        default_scheme = _override_scheme_with_user_specify(scheme, user_scheme_overrides)

    # Extract attributes from the chosen default_scheme
    if isinstance(default_scheme, str):
        final_attrs = _override_scheme_with_user_specify(default_scheme, user_scheme_overrides, return_str=False)
        final_attrs = asdict(final_attrs)
    else:
        final_attrs = asdict(default_scheme)
    return default_scheme, is_auto_scheme, final_attrs


W4A16 = QuantizationScheme.from_dict(
    {
        "bits": 4,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W2A16 = QuantizationScheme.from_dict(
    {
        "bits": 2,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W2A16G64 = QuantizationScheme.from_dict(
    {
        "bits": 2,
        "sym": True,
        "group_size": 64,
        "data_type": "int",
        "act_bits": 16,
    }
)

W2A16G32 = QuantizationScheme.from_dict(
    {
        "bits": 2,
        "sym": True,
        "group_size": 32,
        "data_type": "int",
        "act_bits": 16,
    }
)

W3A16 = QuantizationScheme.from_dict(
    {
        "bits": 3,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

W8A16 = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "sym": True,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
    }
)

MXFP4 = QuantizationScheme.from_dict(
    {
        "bits": 4,
        "group_size": 32,
        "data_type": "mx_fp",
        "act_bits": 4,
        "act_data_type": "mx_fp",
        "act_group_size": 32,
        "act_sym": True,
        "act_dynamic": True,
    }
)

MXFP4_RCEIL = QuantizationScheme.from_dict(
    {
        "bits": 4,
        "group_size": 32,
        "data_type": "mx_fp",
        "act_bits": 4,
        "act_data_type": "mx_fp_rceil",
        "act_group_size": 32,
        "act_sym": True,
        "act_dynamic": True,
    }
)

MXFP8 = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "group_size": 32,
        "data_type": "mx_fp",
        "act_bits": 8,
        "act_data_type": "mx_fp",
        "act_group_size": 32,
        "act_sym": True,
        "act_dynamic": True,
    }
)

MXFP8_RCEIL = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "group_size": 32,
        "data_type": "mx_fp",
        "act_bits": 8,
        "act_data_type": "mx_fp_rceil",
        "act_group_size": 32,
        "act_sym": True,
        "act_dynamic": True,
    }
)

NVFP4 = QuantizationScheme.from_dict(
    {
        "bits": 4,
        "group_size": 16,
        "data_type": "nv_fp",
        "act_bits": 4,
        "act_data_type": "nv_fp4_with_static_gs",
        "act_group_size": 16,
        "act_sym": True,
        "act_dynamic": True,
    }
)

FPW8A16 = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "group_size": 0,
        "data_type": "fp",
        "act_bits": 16,
        "act_data_type": "fp",
    }
)

FP8_BLOCK = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "group_size": (128, 128),
        "data_type": "fp",
        "act_bits": 8,
        "act_group_size": 128,
        "act_data_type": "fp",
        "act_dynamic": True,
        "act_sym": True,
    }
)

# FP8 = asdict(QuantArgs.from_dict({
#     "bits": 8,
#     "group_size": 128,
#     "data_type": "fp",
#     "act_bits": 8,
#     "act_data_type": "fp",
# }))

FP8_STATIC = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "group_size": -1,
        "data_type": "fp",
        "act_bits": 8,
        "act_group_size": 0,
        "act_data_type": "fp",
        "act_dynamic": False,
        "act_sym": True,
    }
)

INT8_W8A8 = QuantizationScheme.from_dict(
    {
        "bits": 8,
        "group_size": -1,
        "data_type": "int",
        "sym": True,
        "act_bits": 8,
        "act_group_size": -1,
        "act_data_type": "int",
        "act_dynamic": True,
        "act_sym": True,
    }
)

# For AutoScheme 16 bits options
BF16 = QuantizationScheme.from_dict(
    {
        "bits": 16,
        "group_size": 128,
        "data_type": "fp",
        "act_bits": 16,
        "act_data_type": "fp",
    }
)

PRESET_SCHEMES = {
    "W4A16": W4A16,
    "W2A16": W2A16,
    "W3A16": W3A16,
    "W8A16": W8A16,
    "MXFP4": MXFP4,
    "MXFP4_RCEIL": MXFP4_RCEIL,
    "MXFP8": MXFP8,
    "MXFP8_RCEIL": MXFP8_RCEIL,
    "NVFP4": NVFP4,
    "FPW8A16": FPW8A16,
    "W2A16G64": W2A16G64,
    "W2A16G32": W2A16G32,
    "FP8_STATIC": FP8_STATIC,
    "BF16": BF16,
    "W4A16_MIXED": W4A16,
    "INT8_W8A8": INT8_W8A8,
    "FP8_BLOCK": FP8_BLOCK,
}
from auto_round.export.export_to_gguf.config import GGUF_CONFIG

for key, val in GGUF_CONFIG.items():
    value = copy.deepcopy(val)
    value.pop("mostly", None)
    value.pop("embedding", None)
    value.pop("lm_head", None)
    PRESET_SCHEMES[key.upper()] = QuantizationScheme.from_dict(value)


def _handle_special_schemes(
    scheme_name: str,
    layer_config: dict,
    model: torch.nn.Module,
    supported_types=None,
    inner_supported_types=None,
    quant_lm_head=False,
    mllm=False,
) -> dict:
    """handle special schemes, like GGUF:Q2_K_MIXED.
    Provide some special auto_round recipes.

    """
    if not isinstance(scheme_name, str):
        return layer_config
    if layer_config is None:
        layer_config = {}
    if scheme_name.lower() == "gguf:q2_k_mixed":
        for n, m in model.named_modules():
            if n in layer_config:
                continue
            if n == "lm_head" or isinstance(m, torch.nn.Embedding):
                layer_config[n] = "GGUF:Q8_0"
            elif isinstance(m, torch.nn.Linear) and ("expert" not in n or "shared_experts" in n) and n != "lm_head":
                layer_config[n] = "GGUF:Q4_K_S"
    if scheme_name.lower() == "w4a16_mixed":
        logger.warning("W4A16_MIXED is experimental and the recipe may change in the future.")
        from auto_round.utils import get_lm_head_name

        lm_head_name = get_lm_head_name(model)
        if supported_types is None:
            from auto_round.utils import SUPPORTED_DTYPES

            supported_types = SUPPORTED_DTYPES
        if inner_supported_types is None:
            from auto_round.utils import INNER_SUPPORTED_LAYER_TYPES

            inner_supported_types = INNER_SUPPORTED_LAYER_TYPES
        for n, m in model.named_modules():
            if n in layer_config:
                continue
            if type(m) in supported_types or type(m) in inner_supported_types:
                if "expert" in n and "shared" not in n:
                    layer_config[n] = {"bits": 4, "data_type": "int"}
                elif n != lm_head_name and mllm:
                    layer_config[n] = {"bits": 16}
                elif n != lm_head_name:
                    layer_config[n] = {"bits": 8, "data_type": "int"}
                elif n == lm_head_name and quant_lm_head:
                    layer_config[n] = {"bits": 8, "data_type": "int"}
    return layer_config


def get_gguf_scheme(scheme: Union[str, QuantizationScheme]) -> str:
    if isinstance(scheme, str) and scheme.upper().startswith("GGUF"):
        return scheme
    if isinstance(scheme, str):
        return ""
    for key, val in PRESET_SCHEMES.items():
        # For q40 or q4_1 we only support it with str scheme， otherwise it will be matched incorrectly with W4G32
        if not key.upper().startswith("GGUF") or ("0" in key or "1" in key):
            continue
        equal = True
        for scheme_key in val.keys():
            if val[scheme_key] is not None and val[scheme_key] != scheme.get(scheme_key, None):
                equal = False
                break
        if equal:
            return key
    return ""
