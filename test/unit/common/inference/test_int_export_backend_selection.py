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
"""Regression tests for CUDA INT export backend policy without a CUDA runtime.

The export matrix uses the generic Torch backend to cover serialization and
forward compatibility. Optimized backend functionality is covered by the
Marlin, ExLlamaV2, and Triton CUDA suites. These tests keep the reference path
and the representative ``backend=auto`` selection policy separate.
"""

from pathlib import Path

import pytest

from auto_round.inference.backend import BackendInfos, get_layer_backend


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("sym", [True, False])
def test_torch_backend_supports_int_export_matrix(bits, group_size, sym):
    backend = get_layer_backend(
        device="cuda",
        backend="torch",
        packing_format="auto_round",
        config={"bits": bits, "group_size": group_size, "sym": sym, "data_type": "int"},
        in_features=128,
        out_features=128,
    )

    assert backend == "auto_round:torch"


@pytest.mark.parametrize(
    "bits,group_size,sym,expected_backend",
    [
        (2, 32, True, "auto_round:tritonv2"),
        (4, 32, True, "gptqmodel:marlin"),
        (4, 32, False, "gptqmodel:exllamav2"),
    ],
)
def test_auto_backend_selection_for_int_exports(monkeypatch, bits, group_size, sym, expected_backend):
    """Check capability/priority selection independently of optional packages."""
    for backend_info in BackendInfos.values():
        monkeypatch.setattr(backend_info, "requirements", None)

    backend = get_layer_backend(
        device="cuda",
        backend="auto",
        packing_format="auto_round",
        config={"bits": bits, "group_size": group_size, "sym": sym, "data_type": "int"},
        in_features=128,
        out_features=128,
    )

    assert backend == expected_backend


def test_generic_int_export_uses_torch_reload_backend():
    source = (Path(__file__).parents[2] / "test_cuda/export/test_autoround_int_export.py").read_text()

    assert 'AutoRoundConfig(backend="torch")' in source
