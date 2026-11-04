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
"""Fuzz the group reshape/pad helpers every quantization data type is built on."""

import math
from test.fuzz_helpers import and_shapes, group_view, tensors, weight_shapes

import torch
from hypothesis import given, note
from hypothesis import strategies as st

from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

GROUP_SIZES = st.one_of(st.sampled_from([-1, 0, 16, 32, 64, 128]), st.integers(1, 300))
PAD_VALUES = st.sampled_from([0.0, 1.0, -3.5, 1e-5])


@given(x=tensors(shapes=st.one_of(weight_shapes(), and_shapes())), group_size=GROUP_SIZES, val=PAD_VALUES)
def test_reshape_pad_matches_reference_and_round_trips(x, group_size, val):
    grouped, orig_shape, pad_len = reshape_pad_tensor_by_group_size(x, group_size, val=val)

    assert orig_shape == x.shape
    expected = group_view(x, group_size)
    assert grouped.shape == expected.shape, f"{tuple(grouped.shape)} != {tuple(expected.shape)}"
    assert grouped.dtype == x.dtype and grouped.device == x.device

    cols = x.shape[-1]
    if group_size > 0 and cols >= group_size:
        assert pad_len == (-cols) % group_size
        assert grouped.shape[-1] == group_size
    else:
        assert pad_len == 0

    # Real elements land exactly where the reference puts them; only the tail padding holds ``val``.
    rows = x.reshape(-1, cols)
    padded = grouped.reshape(rows.shape[0], -1) if pad_len else None
    if padded is not None:
        assert torch.equal(padded[:, :cols], rows)
        assert bool((padded[:, cols:] == val).all()), "padding must be filled with the requested value"
    else:
        assert torch.equal(grouped, expected)

    restored = revert_tensor_by_pad(grouped, orig_shape, pad_len)
    assert restored.shape == x.shape
    assert torch.equal(restored, x), "revert_tensor_by_pad must be the exact inverse of the reshape"


@given(
    x=tensors(shapes=st.tuples(st.integers(1, 40), st.integers(1, 40))),
    block=st.tuples(st.integers(1, 24), st.integers(1, 24)),
)
def test_block_reshape_pad_matches_reference_and_round_trips(x, block):
    m, n = block
    rows, cols = x.shape
    grouped, orig_shape, pad_len = reshape_pad_tensor_by_group_size(x, block)
    note(f"block={block} grouped={tuple(grouped.shape)} pad={pad_len}")

    mb, nb = math.ceil(rows / m), math.ceil(cols / n)
    assert grouped.shape == (mb, nb, m, n)
    assert pad_len == (mb * m - rows, nb * n - cols)

    reference = torch.nn.functional.pad(x, (0, nb * n - cols, 0, mb * m - rows))
    for i in range(mb):
        for j in range(nb):
            assert torch.equal(grouped[i, j], reference[i * m : (i + 1) * m, j * n : (j + 1) * n])

    restored = revert_tensor_by_pad(grouped, orig_shape, pad_len)
    assert torch.equal(restored, x)
