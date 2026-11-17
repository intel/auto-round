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
"""Self-check of the independent minifloat oracle in ``test/fuzz_helpers.py`` against torch casts.

The quantization fuzz tests trust ``FloatGrid``; this pins it to PyTorch's own
float8/bfloat16/float16 conversions so an oracle bug cannot hide a library bug.
"""

from test.fuzz_helpers import FP8_E4M3, FP8_E5M2, FloatGrid, assert_where, to_f64

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

TORCH_GRIDS = [
    (torch.bfloat16, FloatGrid(mbits=7, min_exp=-126, max_norm=torch.finfo(torch.bfloat16).max)),
    (torch.float16, FloatGrid(mbits=10, min_exp=-14, max_norm=torch.finfo(torch.float16).max)),
]
if hasattr(torch, "float8_e4m3fn"):
    TORCH_GRIDS.append((torch.float8_e4m3fn, FP8_E4M3))
if hasattr(torch, "float8_e5m2"):
    TORCH_GRIDS.append((torch.float8_e5m2, FP8_E5M2))


@pytest.mark.parametrize("dtype, grid", TORCH_GRIDS, ids=lambda p: str(p).replace("torch.", ""))
@given(
    mantissas=st.lists(st.floats(-2.0, 2.0, allow_nan=False, width=32), min_size=1, max_size=64),
    exponent=st.integers(-30, 20),
)
def test_float_grid_snap_matches_torch_cast(dtype, grid, mantissas, exponent):
    values = torch.tensor(mantissas, dtype=torch.float32) * (2.0**exponent)
    values = values.clamp(-grid.max_norm, grid.max_norm)
    expected = to_f64(values.to(dtype))
    snapped = grid.snap(to_f64(values))
    assert_where(snapped == expected, "FloatGrid.snap disagrees with torch cast", x=to_f64(values), snapped=snapped)
    assert bool(grid.on_grid(expected).all())
    err = (to_f64(values) - expected).abs()
    assert_where(err <= grid.max_error(to_f64(values).abs()), "FloatGrid.max_error is not an upper bound", err=err)
