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
"""Quantization scheme definitions and preset configurations for AutoRound.

This module provides the :class:`QuantizationScheme` dataclass, which encapsulates
all parameters required for weight and activation quantization, as well as a registry
of named preset schemes (e.g. ``"W4A16"``, ``"MXFP4"``, ``"FP8_STATIC"``) and
helper functions for resolving them.

Example::

    from auto_round.schemes import preset_name_to_scheme
    scheme = preset_name_to_scheme("W4A16")
    print(scheme.bits, scheme.group_size)  # 4  128
"""
import copy
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Optional, Union

import torch

from auto_round.logger import logger

__all__ = ["QuantizationScheme", "get_gguf_scheme", "preset_name_to_scheme"]


@dataclass
class QuantizationScheme:
    """A dataclass representing the full quantization configuration for a layer.

    This dataclass is used to parameterize weight quantization (``bits``,
    ``group_size``, ``sym``, ``data_type``) and optional activation quantization
    (``act_*`` fields), as well as double-quantization of scales (``super_*``
    fields).

    Attributes:
        bits (int): Number of bits for weight quantization. Defaults to 4.
        group_size (int): Group size for per-group weight quantization.
            Defaults to 128.
        sym (bool): Whether to use symmetric weight quantization. Defaults to
            ``True``.
        data_type (str): Weight data type identifier (e.g. ``"int"``, ``"fp"``,
            ``"mx_fp"``). Defaults to ``"int"``.
        act_bits (int | None): Activation quantization bits; ``None`` or ``>= 16``
            means no activation quantization.
        act_group_size (int | None): Group size for activation quantization.
        act_sym (bool | None): Symmetric activation quantization flag.
        act_data_type (str | None): Activation data type identifier.
        act_dynamic (bool | None): Whether to use dynamic activation
            quantization (calibrate at runtime).
        super_bits (int | None): Bits used for the secondary (super) scale
            quantization in double-quantization schemes.
        super_group_size (int | None): Group size for the secondary scale
            quantization.
    """

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
    transform_config: Optional[dict] = None

    @classmethod
    def from_dict(cls, config: dict):
        """Creates a QuantizationScheme instance from a configuration dictionary.

        Args:
            config (dict): Dictionary whose keys correspond to field names of
                :class:`QuantizationScheme`.

        Returns:
            QuantizationScheme: New instance populated from the dictionary.
        """
        return cls(**config)

    @classmethod
    def get_attributes(cls: "QuantizationScheme") -> list[str]:
        """Returns the list of all field names of QuantizationScheme.

        Returns:
            list[str]: Ordered list of dataclass field names.
        """
        return [field.name for field in fields(cls)]

    def __getitem__(self, key: str):
        """Gets the value of the named attribute via bracket notation.

        Args:
            key (str): Name of the attribute to retrieve.

        Returns:
            Any: Value of the attribute.

        Raises:
            KeyError: If ``key`` is not a valid attribute name.
        """
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        return getattr(self, key)

    def __setitem__(self, key: str, value: None | int | str):
        """Sets the value of the named attribute via bracket notation.

        Args:
            key (str): Name of the attribute to set.
            value (None | int | str): New value for the attribute.

        Raises:
            KeyError: If ``key`` is not a valid attribute name.
        """
        if key not in self.get_attributes():
            raise KeyError(f"{key} is not a valid attribute")
        setattr(self, key, value)

    def items(self):
        """Returns an iterator of (field_name, value) pairs.

        Returns:
            generator: Yields ``(str, Any)`` tuples for each field.
        """
        return ((field, getattr(self, field)) for field in self.get_attributes())

    def keys(self):
        """Returns the list of all field names.

        Returns:
            list[str]: All field names of this dataclass.
        """
        return self.get_attributes()

    def values(self):
        """Returns an iterator over all field values.

        Returns:
            generator: Yields the value of each field in declaration order.
        """
        return (getattr(self, field) for field in self.get_attributes())

    def get(self, key: str, default=None):
        """Returns the value of an attribute by name, or a default.

        Unlike ``__getitem__``, this method returns ``default`` both when the
        key is absent and when the field value is ``None``.

        Args:
            key (str): Attribute name to look up.
            default: Value to return when the key is absent or the field is
                ``None``. Defaults to ``None``.

        Returns:
            Any: Field value, or ``default`` if not found / ``None``.
        """
        if key not in self.get_attributes():
            return default
        res = getattr(self, key)
        # In case the attribute is explicitly set to None, return default
        if res is None:
            return default
        return getattr(self, key)

    def __eq__(self, other: "QuantizationScheme") -> bool:
        """Checks equality with another QuantizationScheme.

        Activation fields are skipped when both schemes have ``act_bits >= 16``
        (i.e., no activation quantization in either).

        Args:
            other (QuantizationScheme): The other scheme to compare against.

        Returns:
            bool: ``True`` if all compared fields are equal.
        """
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
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            # Treat None and empty dict as equivalent for dict fields like transform_config
            if self_val != other_val:
                if isinstance(self_val, dict) and not self_val and other_val is None:
                    continue
                if isinstance(other_val, dict) and not other_val and self_val is None:
                    continue
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
    """Checks whether the given name corresponds to a known preset scheme.

    Args:
        name (str): Scheme name to look up (case-insensitive).

    Returns:
        bool: ``True`` if ``name.upper()`` is a key in ``PRESET_SCHEMES``.
    """
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
    """Handles special mixed-precision scheme recipes and populates ``layer_config``.

    For schemes such as ``"GGUF:Q2_K_MIXED"`` and ``"W4A16_MIXED"`` this
    function inspects the model structure and assigns per-layer scheme overrides
    to ``layer_config`` (e.g. routing expert layers to lower precision while
    keeping shared or lm-head layers at higher precision).

    Args:
        scheme_name (str): Scheme identifier string (case-insensitive).
        layer_config (dict): Existing per-layer configuration dictionary; may be
            ``None`` (will be initialised to an empty dict).
        model (torch.nn.Module): The model whose named modules are inspected.
        supported_types (tuple | None, optional): Supported linear layer types.
            Defaults to ``SUPPORTED_DTYPES``.
        inner_supported_types (tuple | None, optional): Supported inner layer
            types. Defaults to ``INNER_SUPPORTED_LAYER_TYPES``.
        quant_lm_head (bool, optional): Whether to quantize the LM head layer.
            Defaults to ``False``.
        mllm (bool, optional): Whether the model is a multimodal LLM. Defaults
            to ``False``.

    Returns:
        dict: Updated ``layer_config`` with any new per-layer overrides applied.
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
    """Resolves a QuantizationScheme object to its GGUF preset name, if any.

    Iterates over all ``GGUF:*`` entries in ``PRESET_SCHEMES`` and returns the
    first name whose parameters match the given scheme.

    Args:
        scheme (str | QuantizationScheme): Scheme to look up.  If already a
            string starting with ``"GGUF"`` it is returned as-is; other strings
            return an empty string.

    Returns:
        str: Matching GGUF preset name (e.g. ``"GGUF:Q4_K_M"``) or an empty
        string if no match is found.
    """
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
