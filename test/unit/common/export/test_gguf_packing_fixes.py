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
"""Regression tests for bugs fixed in ``auto_round/export/export_to_gguf/packing.py``."""

import numpy as np
import pytest
import torch

gguf_quants = pytest.importorskip("gguf.quants")

from gguf import GGMLQuantizationType  # noqa: E402

from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES  # noqa: E402
from auto_round.export.export_to_gguf.packing import (  # noqa: E402
    ggml_quant,
    make_qx_quants,
    make_qx_quants_chunk,
)


def _rel_mse(a, b):
    return ((a - b).pow(2).mean() / b.pow(2).mean()).item()


@pytest.fixture
def blocks():
    torch.manual_seed(0)
    return torch.randn(32, 16, 16)


class TestMakeQxQuantsSearch:
    @pytest.mark.parametrize("rmse_type", [1, 2, 3, 4])
    def test_matches_chunked_implementation(self, blocks, rmse_type):
        s_ref, l_ref = make_qx_quants_chunk(blocks.clone(), bits=6, rmse_type=rmse_type)
        s, l = make_qx_quants(blocks.clone(), bits=6, rmse_type=rmse_type)
        assert torch.equal(l, l_ref)
        torch.testing.assert_close(s, s_ref)

    def test_search_refines_initial_levels(self, blocks):
        _, l_initial = make_qx_quants(blocks.clone(), bits=6, rmse_type=-1)
        _, l_searched = make_qx_quants(blocks.clone(), bits=6, rmse_type=1)
        assert not torch.equal(l_initial, l_searched)


class TestMakeQxQuantsChunkNegativeRmse:
    @pytest.mark.parametrize("rmse_type", [-1, -2, -3, -4])
    def test_every_chunk_returns_early(self, blocks, rmse_type):
        s1, l1 = make_qx_quants_chunk(blocks.clone(), bits=6, rmse_type=rmse_type, split_num=1)
        s4, l4 = make_qx_quants_chunk(blocks.clone(), bits=6, rmse_type=rmse_type, split_num=4)
        assert torch.equal(l1, l4)
        assert torch.equal(s1, s4)


class TestQ6KOriginal:
    def test_roundtrip_through_reference_dequantizer(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 512)
        packed = ggml_quant(weight.clone(), "q6_k", device="cpu", original=True)
        _, type_size = GGML_QUANT_SIZES["q6_k"]
        assert packed.dtype == np.uint8
        assert packed.shape == (weight.shape[0], weight.shape[1] // 256 * type_size)

        deq = gguf_quants.dequantize(packed, GGMLQuantizationType.Q6_K).reshape(weight.shape)
        assert _rel_mse(torch.from_numpy(deq).float(), weight) < 1e-3

    def test_all_zero_block(self):
        weight = torch.zeros(2, 256)
        packed = ggml_quant(weight.clone(), "q6_k", device="cpu", original=True)
        deq = gguf_quants.dequantize(packed, GGMLQuantizationType.Q6_K)
        assert np.all(deq == 0)


class TestBF16QuantBlock:
    def test_matches_gguf_reference(self):
        torch.manual_seed(0)
        special = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.00390625, 1.01171875, 3.4e38] * 8
        )
        data = torch.cat([torch.randn(4, 64), special.reshape(1, 64)])
        packed = ggml_quant(data.clone(), "bf16", device="cpu")
        expected = gguf_quants.quantize(data.numpy(), GGMLQuantizationType.BF16).view(np.uint8)
        assert packed.dtype == np.uint8
        np.testing.assert_array_equal(packed, expected.reshape(packed.shape))
