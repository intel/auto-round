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

import pytest

from auto_round.formats import resolve_formats
from auto_round.planning import CompressionIntent, FormatCompatibilityError, resolve_scheme_value


def test_resolve_formats_does_not_mutate_inputs():
    layer_config = {"layer": {"bits": 4}}
    intent = CompressionIntent(format="auto_round", layer_config=layer_config)
    scheme = resolve_scheme_value("W4A16", {})

    result = resolve_formats(intent, scheme, model=None)

    layer_config["layer"]["bits"] = 8
    assert scheme.value.bits == 4
    assert result.scheme.value.bits == 4
    assert result.layer_config_patch["layer"]["bits"] == 4


def test_resolve_formats_rejects_gguf_with_real_companion():
    scheme = resolve_scheme_value("W4A16", {})

    with pytest.raises(FormatCompatibilityError):
        resolve_formats(CompressionIntent(format="gguf:q4_k_m,auto_round"), scheme, model=None)
