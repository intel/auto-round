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

from __future__ import annotations

import copy
from typing import Any, Mapping

from auto_round.planning.contracts import ResolvedScheme
from auto_round.planning.errors import SchemeResolutionError
from auto_round.schemes import QuantizationScheme, parse_scheme


def resolve_scheme_value(scheme: Any, overrides: Mapping[str, Any]) -> ResolvedScheme:
    """Resolve a model-independent scheme without modifying caller-owned inputs."""
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

    if isinstance(scheme, AutoScheme):
        raise SchemeResolutionError("AutoScheme requires model-aware resolution")

    try:
        _, _, final_attrs = parse_scheme(copy.deepcopy(scheme), copy.deepcopy(dict(overrides)))
    except (KeyError, NotImplementedError, TypeError, ValueError) as error:
        raise SchemeResolutionError(str(error)) from error

    preset_name = None
    if isinstance(scheme, str):
        normalized = scheme.strip("'\" ")
        preset_name = normalized.lower() if normalized.lower().startswith("gguf:") else normalized.upper()
    return ResolvedScheme.from_scheme(QuantizationScheme.from_dict(final_attrs), preset_name=preset_name)
