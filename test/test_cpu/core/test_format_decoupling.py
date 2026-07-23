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
"""Decoupling regression for the `export-format` capability.

Two behaviors that must survive the scheme/format/compressor decoupling:

1. Format-selection: ``get_formats(scheme, ...)`` (now scheme-centric, no ``ar``)
   resolves the same backend as pre-refactor for representative combos.
2. GGUF scheme correction propagates back onto ``self.scheme`` /
   ``self.scheme_context`` / ``self.quantize_config`` (the staleness fix), with
   ``self.scheme`` pinned to the precise preset *string* for alias
   disambiguation.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.compressors.base import BaseCompressor
from auto_round.formats import _check_compatibility, get_formats
from auto_round.planning import FormatResolution, ResolvedScheme
from auto_round.schemes import QuantizationScheme, parse_scheme


def _resolved_scheme(scheme_name):
    # Real call sites always fully resolve the scheme (parse_scheme's third return,
    # final_attrs) before calling get_formats — e.g. act_data_type defaults to "float"
    # rather than staying None. Mirror that precondition here.
    _, _, final_attrs = parse_scheme(scheme_name, {})
    return QuantizationScheme.from_dict(final_attrs)


# Harvested verbatim from a live run against pre-refactor code (HEAD=33b7df0e).
EXPECTED_FORMAT_SELECTION_BASELINE = {
    ("W4A16", "auto_round"): ["auto_round:auto_gptq"],
    ("FP8_STATIC", "llm_compressor"): ["llm_compressor:fp8_static"],
    ("MXFP4", "auto_round"): ["auto_round:mx_fp"],
    ("NVFP4", "auto_round"): ["auto_round:nv_fp"],
}


def test_format_selection_baseline():
    results = {}
    for scheme_name, fmt in EXPECTED_FORMAT_SELECTION_BASELINE:
        scheme = _resolved_scheme(scheme_name)
        # _check_divisible_by_32 iterates model.named_modules() for int/act_bits>=16,
        # so a real nn.Module is required.
        model = nn.Sequential(nn.Linear(32, 32))
        formats, _, _, _, _ = get_formats(fmt, scheme, model=model, layer_config=None, scale_dtype=torch.float16)
        results[(scheme_name, fmt)] = [f.get_backend_name() for f in formats]
    assert results == EXPECTED_FORMAT_SELECTION_BASELINE


@pytest.mark.parametrize("other_format", ["auto_round", "llm_compressor"])
def test_gguf_rejects_non_fake_companion_formats(other_format):
    scheme = _resolved_scheme("W4A16")

    with pytest.raises(ValueError, match="GGUF format is not compatible"):
        _check_compatibility(["gguf:q4_k_m", other_format], scheme)


def test_gguf_allows_fake_companion_format():
    scheme = _resolved_scheme("W4A16")

    assert _check_compatibility(["gguf:q4_k_m", "fake"], scheme) == ["gguf:q4_k_m", "fake"]


def test_gguf_correction_propagates_to_scheme_and_quantize_config(monkeypatch):
    # W4A16's plain int-woq fields don't match every field gguf_args_check enforces
    # for a GGUF target; after resolving "gguf:q4_k_m" the scheme-bearing objects
    # must all agree, and self.scheme must be pinned to the precise preset string so
    # get_gguf_scheme()'s string short-circuit can disambiguate Q4_K_S vs Q4_K_M.
    from auto_round.schemes import get_gguf_scheme

    original_scheme = _resolved_scheme("W4A16")
    corrected_scheme = _resolved_scheme("GGUF:Q4_K_M")
    quantize_config = QuantizationConfig(scheme=original_scheme.copy())
    alg_config = QuantizationConfig(scheme=original_scheme.copy())
    resolved_format = SimpleNamespace(
        is_gguf=lambda: True,
        output_format="gguf",
        backend=SimpleNamespace(output_format="gguf:q4_k_m"),
    )

    def fake_resolve_formats(*args, **kwargs):
        return FormatResolution(
            formats=(resolved_format,),
            scheme=ResolvedScheme.from_scheme(corrected_scheme, preset_name="gguf:q4_k_m"),
            layer_config_patch={"layer": {"bits": 4}},
            scale_dtype=torch.float32,
            quant_block_list=(("model.layers.0",),),
        )

    monkeypatch.setattr("auto_round.compressors.base.resolve_formats", fake_resolve_formats)

    class FormatResolutionHost:
        _resolve_gguf_preset_string = staticmethod(BaseCompressor._resolve_gguf_preset_string)
        _resolve_format_string = BaseCompressor._resolve_format_string

    ar = FormatResolutionHost()
    ar.scheme_context = original_scheme
    ar.scheme = original_scheme.copy()
    ar.quantize_config = quantize_config
    ar._alg_configs = [alg_config, quantize_config]
    ar.model_context = SimpleNamespace(model=nn.Linear(1, 1), is_mllm=False)
    ar.layer_config = None
    ar.scale_dtype = torch.float16
    ar.quant_nontext_module = False
    ar.quant_block_list = None
    ar.platform = "hf"
    ar.is_auto_scheme = False

    formats = ar._resolve_format_string("gguf:q4_k_m")

    assert formats == [resolved_format]
    assert isinstance(ar.scheme, str)
    assert ar.scheme.lower() == "gguf:q4_k_m"
    assert get_gguf_scheme(ar.scheme) == ar.scheme  # short-circuit preserved
    assert ar.scheme_context == corrected_scheme
    assert alg_config.scheme is ar.scheme_context
    assert quantize_config.scheme is ar.scheme_context
    assert ar.scheme_context.bits == ar.quantize_config.bits
    assert ar.scheme_context.data_type == ar.quantize_config.data_type
    assert ar.scheme_context.sym == ar.quantize_config.sym
    assert ar.layer_config == {"layer": {"bits": 4}}
    assert ar.scale_dtype is torch.float32
    assert ar.quant_block_list == [["model.layers.0"]]
