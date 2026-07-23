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

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
from auto_round.cli import main as cli_main
from auto_round.planning import (
    CompressionIntent,
    FormatCompatibilityError,
    ResolvedScheme,
    SchemeResolutionError,
    resolve_scheme_value,
)
from auto_round.schemes import QuantizationScheme


def test_resolve_scheme_value_preserves_precise_gguf_identity():
    resolved = resolve_scheme_value("GGUF:Q4_K_M", {})

    assert resolved.preset_name == "gguf:q4_k_m"
    assert resolved.value.bits == 4
    assert resolved.value.data_type == "int_asym_dq"
    assert not isinstance(resolved.value, str)


def test_resolve_scheme_value_applies_overrides_without_mutating_input():
    overrides = {"bits": 8}

    resolved = resolve_scheme_value("W4A16", overrides)

    assert resolved.preset_name == "W4A16"
    assert resolved.value.bits == 8
    assert overrides == {"bits": 8}


def test_resolve_scheme_value_defers_model_aware_auto_scheme():
    scheme = AutoScheme(avg_bits=4.0, options=["W4A16", "W8A16"])

    with pytest.raises(SchemeResolutionError, match="model-aware"):
        resolve_scheme_value(scheme, {})

    assert scheme.options == ["W4A16", "W8A16"]


def test_resolved_scheme_is_deeply_isolated_and_frozen():
    source = QuantizationScheme(bits=4, data_type="int", rotation_config={"mode": "hadamard"})
    resolved = ResolvedScheme.from_scheme(source, preset_name="W4A16")

    source.bits = 8
    source.rotation_config["mode"] = "random"
    leaked_copy = resolved.value
    leaked_copy.bits = 2
    leaked_copy.rotation_config["mode"] = "identity"

    assert resolved.value.bits == 4
    assert resolved.value.rotation_config == {"mode": "hadamard"}
    with pytest.raises(FrozenInstanceError):
        resolved.preset_name = "W8A16"


def test_compression_intent_freezes_nested_layer_config_copy():
    source = {"model.layers.0.q_proj": {"bits": 4, "metadata": {"source": "user"}}}
    intent = CompressionIntent(format="auto_round", layer_config=source)

    source["model.layers.0.q_proj"]["bits"] = 8
    source["model.layers.0.q_proj"]["metadata"]["source"] = "default"

    assert intent.layer_config["model.layers.0.q_proj"]["bits"] == 4
    assert intent.layer_config["model.layers.0.q_proj"]["metadata"] == {"source": "user"}
    with pytest.raises(TypeError):
        intent.layer_config["model.layers.0.q_proj"]["bits"] = 2


def test_cli_converts_domain_error_to_nonzero_exit(monkeypatch):
    messages = []
    monkeypatch.setattr(cli_main.sys, "argv", ["auto-round"])
    monkeypatch.setattr(cli_main, "start", lambda argv=None: (_ for _ in ()).throw(FormatCompatibilityError("bad")))
    monkeypatch.setattr(cli_main, "logger", SimpleNamespace(error=messages.append))

    with pytest.raises(SystemExit) as error:
        cli_main.run()

    assert error.value.code != 0
    assert messages == ["bad"]
