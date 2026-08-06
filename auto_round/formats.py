# Copyright (c) 2026 Intel Corporation
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

"""Backward-compatible imports for the relocated output-format API."""

from enum import Enum

from auto_round.compressors.config_resolution import ResolvedScheme
from auto_round.export.formats import (
    AutoAWQFormat,
    AutoGPTQFormat,
    AutoRoundFormat,
    BackendDataType,
    FakeFormat,
    FP8Format,
    GGUFFormat,
    LLMCompressorFormat,
    MLXFormat,
    OutputFormat,
    resolve_formats,
)


class AutoRoundExportFormat(str, Enum):
    """Legacy names retained for callers importing the old module."""

    FP8_STATIC = "fp8_static"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"
    FP8 = "fp8"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"
    MX_FP_RCEIL = "mx_fp_rceil"
    NV_FP4_WITH_STATIC_GS = "nv_fp4_with_static_gs"
    INT8 = "int8_w8a8"
    FP8_BLOCK = "fp8_block"
    MXINT4 = "mxint4"
    MX_INT = "mx_int"
    WINT_A16 = "wint_a16"


def get_formats(format: str, ar):
    """Resolve formats using the relocated resolver for the legacy API."""
    from auto_round.schemes import QuantizationScheme, parse_scheme

    scheme = ar.scheme
    if not isinstance(scheme, QuantizationScheme):
        _, _, attrs = parse_scheme(scheme, {})
        scheme = QuantizationScheme.from_dict(attrs)
    resolution = resolve_formats(
        ResolvedScheme.from_scheme(scheme),
        format=format,
        layer_config=getattr(ar, "layer_config", None),
        scale_dtype=getattr(ar, "scale_dtype", None),
        iters=getattr(ar, "iters", 0) or 0,
        model=getattr(ar, "model", None),
    )
    return list(resolution.formats)


__all__ = [
    "AutoAWQFormat",
    "AutoRoundExportFormat",
    "AutoGPTQFormat",
    "AutoRoundFormat",
    "BackendDataType",
    "FakeFormat",
    "FP8Format",
    "GGUFFormat",
    "LLMCompressorFormat",
    "MLXFormat",
    "OutputFormat",
    "resolve_formats",
    "get_formats",
]
