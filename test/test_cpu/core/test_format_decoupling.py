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

import torch
import torch.nn as nn

from auto_round.formats import get_formats
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


def test_gguf_correction_propagates_to_scheme_and_quantize_config(tiny_qwen_model_path):
    # W4A16's plain int-woq fields don't match every field gguf_args_check enforces
    # for a GGUF target; after resolving "gguf:q4_k_m" the scheme-bearing objects
    # must all agree, and self.scheme must be pinned to the precise preset string so
    # get_gguf_scheme()'s string short-circuit can disambiguate Q4_K_S vs Q4_K_M.
    from auto_round import AutoRound
    from auto_round.schemes import get_gguf_scheme

    ar = AutoRound(model=tiny_qwen_model_path, scheme="W4A16", iters=0)
    ar.quantize()
    ar.formats = "gguf:q4_k_m"
    ar._resolve_formats()

    assert isinstance(ar.scheme, str)
    assert ar.scheme.lower() == "gguf:q4_k_m"
    assert get_gguf_scheme(ar.scheme) == ar.scheme  # short-circuit preserved
    assert ar.scheme_context.bits == ar.quantize_config.bits
    assert ar.scheme_context.data_type == ar.quantize_config.data_type
    assert ar.scheme_context.sym == ar.quantize_config.sym
