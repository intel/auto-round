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

import torch.nn as nn

from auto_round.compressors.config_resolution import ResolvedScheme
from auto_round.compressors.layer_config_resolver import (
    extract_regex_config,
    has_quantized_layer_outside_blocks,
    resolve_layer_config,
)
from auto_round.schemes import QuantizationScheme


def test_resolver_does_not_write_quantization_attributes_to_modules():
    model = nn.Sequential(nn.Linear(32, 32))
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(act_bits=16, act_data_type="float"))

    resolved = resolve_layer_config(
        model=model,
        scheme=scheme,
        layer_config={"0": {}},
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    assert "0" in resolved
    assert not hasattr(model[0], "bits")


def test_extract_regex_config_keeps_pattern_separate_from_expanded_layers():
    model = nn.Sequential(nn.Linear(32, 32))
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(act_bits=16, act_data_type="float"))

    regex_config = extract_regex_config(
        model=model,
        scheme=scheme,
        layer_config={"0": {"bits": 4}, "missing.*": {"bits": 16}},
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    assert set(regex_config) == {"missing.*"}
    assert regex_config["missing.*"]["bits"] == 16


def test_has_quantized_layer_outside_blocks_is_derived_from_final_mapping():
    assert has_quantized_layer_outside_blocks({"layer": {"bits": 4, "in_blocks": False}})
    assert not has_quantized_layer_outside_blocks({"layer": {"bits": 16, "in_blocks": False}})


def test_explicit_w8_asym_layer_config_entry_raises():
    """An explicit 8-bit asymmetric entry can never be exported or served;
    refuse it during layer-config resolution, before any quantization work."""
    import pytest

    model = nn.Sequential(nn.Linear(32, 32))
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(bits=4, sym=False, act_bits=16, act_data_type="float"))

    with pytest.raises(ValueError, match="8-bit asymmetric"):
        resolve_layer_config(
            model=model,
            scheme=scheme,
            layer_config={"0": {"bits": 8, "sym": False, "data_type": "int"}},
            supported_types=(nn.Linear,),
            inner_supported_types=(),
        )


def test_inherited_w8_asym_layer_config_entry_pinned_symmetric():
    """An 8-bit entry without an explicit sym only inherits the global asym;
    pin it symmetric instead of failing the whole run (conservative default;
    the llm_compressor format and allow_w8_asym are the escape hatches)."""
    model = nn.Sequential(nn.Linear(32, 32))
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(bits=4, sym=False, act_bits=16, act_data_type="float"))

    resolved = resolve_layer_config(
        model=model,
        scheme=scheme,
        layer_config={"0": {"bits": 8, "data_type": "int"}},
        supported_types=(nn.Linear,),
        inner_supported_types=(),
    )

    assert resolved["0"]["sym"] is True


def test_extract_regex_config_honors_format_allowance():
    """AutoScheme output can carry DP-selected W8-asym entries; extract_regex_config
    (the post-scoring export-metadata pass) must keep them under llm_compressor and
    still refuse them for native formats."""
    import torch.nn as nn

    from auto_round.compressors.config_resolution import ResolvedScheme
    from auto_round.compressors.layer_config_resolver import extract_regex_config
    from auto_round.schemes import QuantizationScheme

    model = nn.Sequential(nn.Linear(32, 32))
    scheme = ResolvedScheme.from_scheme(QuantizationScheme(bits=4, sym=False, act_bits=16, act_data_type="float"))
    layer_config = {"0": {"bits": 8, "sym": False, "data_type": "int", "group_size": 32}}

    # Regression: the post-scoring export-metadata pass must not refuse a
    # DP-selected W8-asym entry when the export format serves it (this exact
    # shape crashed AutoScheme + llm_compressor + --asym after scoring).
    extract_regex_config(
        model=model,
        scheme=scheme,
        layer_config=dict(layer_config),
        supported_types=(nn.Linear,),
        inner_supported_types=(),
        format="auto_round:llm_compressor",
    )

    import pytest as _pytest

    with _pytest.raises(ValueError, match="8-bit asymmetric"):
        extract_regex_config(
            model=model,
            scheme=scheme,
            layer_config=dict(layer_config),
            supported_types=(nn.Linear,),
            inner_supported_types=(),
            format="auto_round",
        )

    # flag exempts native formats too
    extract_regex_config(
        model=model,
        scheme=scheme,
        layer_config=dict(layer_config),
        supported_types=(nn.Linear,),
        inner_supported_types=(),
        format="auto_round",
        allow_w8_asym=True,
    )
