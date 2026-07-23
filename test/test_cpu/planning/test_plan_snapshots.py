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
import torch
import torch.nn as nn

from auto_round.formats import resolve_formats
from auto_round.planning import build_compression_plan, resolve_scheme_value

EXPECTED_PLAN_SNAPSHOTS = {
    "w4_autoround": {
        "scheme_name": "W4A16",
        "format_name": "auto_round",
        "backend": "auto_round:auto_gptq",
        "bits": 4,
        "group_size": 128,
        "data_type": "int",
        "act_bits": 16,
        "act_data_type": "float",
        "scale_dtype": "torch.float16",
    },
    "fp8_static_llmc": {
        "scheme_name": "FP8_STATIC",
        "format_name": "llm_compressor",
        "backend": "llm_compressor:fp8_static",
        "bits": 8,
        "group_size": -1,
        "data_type": "fp",
        "act_bits": 8,
        "act_data_type": "fp",
        "scale_dtype": "torch.float16",
    },
    "gguf_q4_k_m": {
        "scheme_name": "GGUF:Q4_K_M",
        "format_name": "gguf:q4_k_m",
        "backend": "gguf:q4_k_m",
        "bits": 4,
        "group_size": 32,
        "data_type": "int_asym_dq",
        "act_bits": 16,
        "act_data_type": "float",
        "scale_dtype": "torch.float32",
    },
    "mxfp4_autoround": {
        "scheme_name": "MXFP4",
        "format_name": "auto_round",
        "backend": "auto_round:mx_fp",
        "bits": 4,
        "group_size": 32,
        "data_type": "mx_fp",
        "act_bits": 4,
        "act_data_type": "mx_fp",
        "scale_dtype": "torch.float16",
    },
    "nvfp4_autoround": {
        "scheme_name": "NVFP4",
        "format_name": "auto_round",
        "backend": "auto_round:nv_fp",
        "bits": 4,
        "group_size": 16,
        "data_type": "nv_fp",
        "act_bits": 4,
        "act_data_type": "nv_fp4_with_static_gs",
        "scale_dtype": "torch.float16",
    },
}


@pytest.mark.parametrize("case_name", EXPECTED_PLAN_SNAPSHOTS)
def test_scheme_format_plan_snapshot(case_name):
    expected = EXPECTED_PLAN_SNAPSHOTS[case_name]
    scheme = resolve_scheme_value(expected["scheme_name"], {})
    resolution = resolve_formats(
        scheme,
        format=expected["format_name"],
        layer_config={"layer": {"bits": scheme.value.bits}},
        scale_dtype=torch.float16,
        model=nn.Sequential(nn.Linear(32, 32)),
    )
    plan = build_compression_plan(resolution, {"layer": {"bits": resolution.scheme.value.bits}})
    scheme_value = plan.scheme.value
    snapshot = {
        "scheme_name": expected["scheme_name"],
        "format_name": expected["format_name"],
        "backend": plan.formats[0].get_backend_name(),
        "bits": scheme_value.bits,
        "group_size": scheme_value.group_size,
        "data_type": scheme_value.data_type,
        "act_bits": scheme_value.act_bits,
        "act_data_type": scheme_value.act_data_type,
        "scale_dtype": str(plan.scale_dtype),
    }

    assert snapshot == expected
    assert {name: dict(config) for name, config in plan.layer_config.items()} == {"layer": {"bits": expected["bits"]}}
    assert dict(plan.regex_config) == {}
    assert plan.has_qlayer_outside_block is False
    assert plan.quant_block_list is None
