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
"""CPU-only tests for ``auto_round/export/export_to_gguf/packing.py``.

Covers the llama.cpp-style ``original=True`` packing path, the legacy block
types without precomputed scales, the standalone search helpers and the
chunking / OOM fallback logic of ``ggml_quant``. All tensors are tiny.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from auto_round.export.export_to_gguf import packing
from auto_round.export.export_to_gguf.packing import (
    ggml_quant,
    ggml_quant_core,
    make_qkx2_quants,
    make_qp_quants,
    make_qx_quants,
    make_qx_quants_chunk,
    torch_roundf,
)

gguf_quants = pytest.importorskip("gguf.quants")
from gguf import GGMLQuantizationType  # noqa: E402


def _dequantize(packed, qtype, shape):
    return torch.from_numpy(gguf_quants.dequantize(packed, GGMLQuantizationType[qtype.upper()])).reshape(shape)


def _rel_mse(a, b):
    return (((a - b) ** 2).mean() / (b**2).mean()).item()


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(4, 512)


class TestOriginalKQuantPacking:
    """``original=True`` mirrors llama.cpp's reference K-quant search."""

    @pytest.mark.parametrize(
        "qtype, max_rel_mse",
        [("q2_k", 0.25), ("q3_k", 0.08), ("q4_k", 0.02), ("q5_k", 0.006)],
    )
    def test_roundtrip_through_reference_dequantizer(self, weight, qtype, max_rel_mse):
        packed = ggml_quant(weight.clone(), qtype, device="cpu", original=True)
        _, type_size = packing.GGML_QUANT_SIZES[qtype]
        assert packed.dtype == np.uint8
        assert packed.shape == (weight.shape[0], weight.shape[1] // 256 * type_size)

        deq = _dequantize(packed, qtype, weight.shape)
        assert torch.isfinite(deq).all()
        assert _rel_mse(deq, weight) < max_rel_mse

    def test_error_decreases_with_bits(self, weight):
        errors = []
        for qtype in ("q2_k", "q3_k", "q4_k", "q5_k"):
            packed = ggml_quant(weight.clone(), qtype, device="cpu", original=True)
            errors.append(_rel_mse(_dequantize(packed, qtype, weight.shape), weight))
        assert errors == sorted(errors, reverse=True), errors


class TestLegacyBlockPacking:
    """Legacy (32-element block) types compute their own scales when none are given."""

    @pytest.mark.parametrize(
        "qtype, max_rel_mse",
        [("q4_0", 0.02), ("q4_1", 0.02), ("q5_0", 0.006), ("q5_1", 0.006), ("q8_0", 1e-4)],
    )
    def test_roundtrip(self, weight, qtype, max_rel_mse):
        packed = ggml_quant(weight.clone(), qtype, device="cpu")
        deq = _dequantize(packed, qtype, weight.shape)
        assert _rel_mse(deq, weight) < max_rel_mse

    def test_q5_1_with_precomputed_scale_and_zp(self):
        torch.manual_seed(1)
        blocks = torch.rand(8, 32) * 2 - 0.5
        bmin = blocks.min(dim=-1, keepdim=True)[0]
        bmax = blocks.max(dim=-1, keepdim=True)[0]
        scale = (bmax - bmin) / 31
        zp = -bmin / scale
        packed = ggml_quant(blocks.clone(), "q5_1", scale=scale, zp=zp, device="cpu")
        deq = _dequantize(packed, "q5_1", blocks.shape)
        torch.testing.assert_close(deq, blocks, atol=scale.max().item(), rtol=0)

    def test_mismatched_scale_shape_is_ignored(self, weight):
        # A 256-group scale cannot be reshaped onto 32-element q4_0 blocks; it must be
        # dropped so q4_0 computes its own scales instead of crashing.
        bad_scale = torch.ones(3)
        bad_scale_2d = torch.ones(3, 1)
        ref = ggml_quant(weight.clone(), "q4_0", device="cpu")
        out = ggml_quant(weight.clone(), "q4_0", scale=bad_scale_2d, d_scale=bad_scale, device="cpu")
        np.testing.assert_array_equal(out, ref)


class TestGgmlQuantChunking:
    def test_split_path_matches_single_shot(self, weight):
        ref = ggml_quant(weight.clone(), "q8_0", device="cpu")
        with (
            patch.object(packing, "_LARGE_TENSOR_BYTES_THRESHOLD", 1),
            patch.object(packing, "clear_memory") as clear_mock,
        ):
            out = ggml_quant(weight.clone(), "q8_0", device="cpu")
        np.testing.assert_array_equal(out, ref)
        assert clear_mock.call_count == 16

    def test_core_falls_back_to_cpu_on_oom(self):
        calls = []

        def flaky_quant(blocks, scale, **kwargs):
            calls.append(blocks.device.type)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("fake oom")
            return blocks.numpy()

        blocks = torch.randn(2, 32)
        scale = torch.ones(2, 1)
        with patch.object(packing, "clear_memory") as clear_mock:
            out = ggml_quant_core(
                flaky_quant,
                blocks,
                scale,
                zp=torch.zeros(2, 1),
                wmin=torch.zeros(2, 1),
                d_scale=torch.ones(2, 1),
                d_wmin=torch.ones(2, 1),
                imatrix=torch.ones(32),
                original=False,
            )
        assert calls == ["cpu", "cpu"]
        clear_mock.assert_called_once()
        np.testing.assert_array_equal(out, blocks.numpy())


class TestSearchHelpers:
    def test_torch_roundf_rounds_half_away_from_zero(self):
        x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1.2, -1.7])
        torch.testing.assert_close(torch_roundf(x), torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 1.0, -2.0]))

    @pytest.mark.parametrize("rmse_type", [0, -1, 1, 2, 3, 4])
    def test_make_qx_quants_levels_and_reconstruction(self, rmse_type):
        torch.manual_seed(0)
        data = torch.randn(2, 16, 16)
        bits = 6
        nmax = 2 ** (bits - 1)
        scales, L = make_qx_quants(data.clone(), bits=bits, rmse_type=rmse_type)
        assert scales.shape == (2, 16)
        assert L.dtype == torch.uint8
        assert int(L.max()) <= 2 * nmax - 1
        recon = scales.unsqueeze(-1) * (L.float() - nmax)
        assert _rel_mse(recon, data) < 0.01

    def test_make_qx_quants_with_importance_weights(self):
        torch.manual_seed(0)
        data = torch.randn(2, 16, 16)
        qw = torch.rand_like(data)
        scales, L = make_qx_quants(data.clone(), bits=4, rmse_type=1, qw=qw)
        assert scales.shape == (2, 16) and L.shape == data.shape

    @pytest.mark.parametrize("rmse_type", [0, -1])
    def test_make_qx_quants_chunk_matches_reference(self, rmse_type):
        torch.manual_seed(0)
        data = torch.randn(2, 16, 16)
        s_ref, l_ref = make_qx_quants(data.clone(), bits=6, rmse_type=rmse_type)
        s, l = make_qx_quants_chunk(data.clone(), bits=6, rmse_type=rmse_type)
        torch.testing.assert_close(s, s_ref)
        assert torch.equal(l, l_ref)

    @pytest.mark.parametrize("rmse_type", [0, 1, 2, 3, 4])
    def test_make_qx_quants_chunk_split_matches_unsplit(self, rmse_type):
        torch.manual_seed(0)
        data = torch.randn(4, 16, 16)
        s1, l1 = make_qx_quants_chunk(data.clone(), bits=6, rmse_type=rmse_type, split_num=1)
        s2, l2 = make_qx_quants_chunk(data.clone(), bits=6, rmse_type=rmse_type, split_num=2)
        torch.testing.assert_close(s1, s2)
        assert torch.equal(l1, l2)

    def test_make_qx_quants_chunk_row_pattern_matches_full_weights(self):
        torch.manual_seed(0)
        blocks_per_row = 2
        data = torch.randn(6, 16, 16)
        pattern = torch.rand(blocks_per_row, 16, 16)
        full_qw = pattern.repeat(3, 1, 1)
        s_full, l_full = make_qx_quants_chunk(data.clone(), bits=6, rmse_type=1, qw=full_qw, split_num=3)
        s_pat, l_pat = make_qx_quants_chunk(data.clone(), bits=6, rmse_type=1, qw_row_pattern=pattern, split_num=3)
        torch.testing.assert_close(s_full, s_pat)
        assert torch.equal(l_full, l_pat)

    @pytest.mark.parametrize("bits, shape", [(2, (16, 16)), (4, (8, 32)), (5, (8, 32))])
    def test_make_qkx2_quants_reshapes_2d_input(self, bits, shape):
        torch.manual_seed(0)
        data = torch.randn(3, 256)
        scale, L, mins = make_qkx2_quants(data.clone(), bits=bits, nstep=2)
        assert scale.shape == (3, shape[0])
        assert mins.shape == (3, shape[0])
        assert L.shape == (3, *shape)
        assert int(L.max()) <= 2**bits - 1

    def test_make_qkx2_quants_unsupported_bits(self):
        with pytest.raises(NotImplementedError):
            make_qkx2_quants(torch.randn(2, 256), bits=3)

    def test_make_qkx2_quants_search_not_worse_than_initial(self):
        torch.manual_seed(0)
        data = torch.randn(2, 8, 32)
        weights = torch.rand_like(data)

        def err(scale, L, mins):
            recon = scale.unsqueeze(-1) * L.float() - mins.unsqueeze(-1)
            return (weights * (recon - data) ** 2).sum().item()

        no_search = err(*make_qkx2_quants(data.clone(), bits=4, weights=weights, nstep=0))
        searched = err(*make_qkx2_quants(data.clone(), bits=4, weights=weights, nstep=20))
        assert searched <= no_search + 1e-6

    def test_make_qp_quants(self):
        torch.manual_seed(0)
        nmax = 15
        data = torch.rand(4, 16) + 0.1
        quant_weights = torch.rand(4, 16) + 0.5
        scale, L = make_qp_quants(nmax, data.clone(), quant_weights)
        assert scale.shape == (4,)
        assert (L >= 0).all() and (L <= nmax).all()
        recon = scale.unsqueeze(-1) * L
        assert _rel_mse(recon, data) < 0.01
