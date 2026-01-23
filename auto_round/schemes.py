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
from dataclasses import dataclass, fields
from typing import Optional, Union

import torch

__all__ = ["QuantizationScheme", "get_gguf_scheme", "preset_name_to_scheme"]


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
        return cls(**config)

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
                    layer_config[n] = {"bits": 4}
                elif n != lm_head_name and mllm:
                    layer_config[n] = {"bits": 16}
                elif n != lm_head_name:
                    layer_config[n] = {"bits": 8}
                elif n == lm_head_name and quant_lm_head:
                    layer_config[n] = {"bits": 8}
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
