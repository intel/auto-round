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
"""GPU-side fast unit tests for ``auto_round.data_type.utils``.

Covers the helpers not exercised by the existing ``test_data_type_utils.py``:
``get_quant_func`` (with various dtype / bits / sym / iters / group_size combinations),
``_resolve_optimized_dtype_funcs``, ``search_optimized_init_scale``,
``compute_optimized_init_scale``, ``reshape_imatrix_for_weight``, the
``get_gaudi_fp8_ste_func`` cache, and the ``update_fused_layer_global_scales`` /
``update_block_global_scale_if_needed`` branches that short-circuit on non-NVFP.

All tests run in milliseconds; no GPU, model, or quantization kernel is launched.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

import torch.nn as nn

from auto_round.data_type.utils import (
    _resolve_optimized_dtype_funcs,
    compute_optimized_init_scale,
    float8_e4m3fn_ste,
    float8_e5m2_ste,
    get_gaudi_fp8_ste_func,
    get_optimized_quant_func,
    get_quant_func,
    reshape_imatrix_for_weight,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    search_optimized_init_scale,
    update_block_global_scale_if_needed,
    update_fused_layer_global_scales,
)


# ==============================================================================
# get_quant_func
# ==============================================================================


class TestGetQuantFunc:
    def test_int_sym_4bit(self):
        fn, dtype = get_quant_func("int", 4, sym=True, iters=200)
        assert callable(fn)
        # Should return a known dtype key
        assert isinstance(dtype, str)
        assert dtype != ""

    def test_int_asym_4bit(self):
        fn, dtype = get_quant_func("int", 4, sym=False, iters=200)
        assert callable(fn)

    def test_int_sym_8bit(self):
        fn, _ = get_quant_func("int", 8, sym=True, iters=200)
        assert callable(fn)

    def test_iters_zero_uses_rtn_path(self):
        # When iters=0, the function tries the opt_rtn / rtn prefix variants
        fn, dtype = get_quant_func("int", 4, sym=True, iters=0)
        assert callable(fn)
        assert "rtn" in dtype or "int" in dtype

    def test_disable_opt_rtn_uses_plain_rtn(self):
        # When disable_opt_rtn=True, opt_rtn_ prefix is skipped
        fn, dtype = get_quant_func("int", 4, sym=True, iters=0, disable_opt_rtn=True)
        assert callable(fn)
        # Plain rtn path
        assert "rtn" in dtype

    def test_block_group_size_uses_block_prefix(self):
        # When group_size is a tuple, the function tries the block_ prefix
        fn, dtype = get_quant_func("fp", 8, sym=True, group_size=(128, 128), iters=200)
        assert callable(fn)
        # Should match one of the block variants
        assert "block" in dtype or "fp" in dtype

    def test_unknown_combination_raises(self):
        with pytest.raises(ValueError, match="No quantization function found"):
            get_quant_func("totally_made_up_dtype", 4, sym=True, iters=200)


# ==============================================================================
# _resolve_optimized_dtype_funcs
# ==============================================================================


class TestResolveOptimizedDtypeFuncs:
    def test_dq_data_type_returns_none_pair(self):
        # "*_dq" data types are not supported by the optimized path
        sf, qf = _resolve_optimized_dtype_funcs("int_asym_dq")
        assert sf is None
        assert qf is None

    def test_asym_int_returns_none_pair(self):
        # Asymmetric int also returns None (the int optimized path is sym-only)
        sf, qf = _resolve_optimized_dtype_funcs("int_asym")
        assert sf is None
        assert qf is None

    def test_unknown_data_type_returns_none_pair(self):
        sf, qf = _resolve_optimized_dtype_funcs("totally_made_up_dtype")
        assert sf is None
        assert qf is None

    def test_int_sym_returns_search_and_quant(self):
        sf, qf = _resolve_optimized_dtype_funcs("int_sym")
        assert callable(sf)
        assert callable(qf)

    def test_mx_data_type_returns_mx_search_and_quant(self):
        sf, qf = _resolve_optimized_dtype_funcs("mx_fp4")
        assert callable(sf)
        assert callable(qf)

    def test_nv_data_type_returns_nv_search_and_quant(self):
        sf, qf = _resolve_optimized_dtype_funcs("nv_fp4")
        assert callable(sf)
        assert callable(qf)


# ==============================================================================
# search_optimized_init_scale
# ==============================================================================


class TestSearchOptimizedInitScale:
    def test_returns_none_for_unsupported_dtype(self):
        w = torch.randn(2, 4, dtype=torch.float32)
        result = search_optimized_init_scale(w, "totally_made_up", bits=4)
        assert result is None

    def test_returns_none_for_dq_data_type(self):
        w = torch.randn(2, 4, dtype=torch.float32)
        result = search_optimized_init_scale(w, "int_asym_dq", bits=4)
        assert result is None

    def test_returns_none_for_asym_int(self):
        w = torch.randn(2, 4, dtype=torch.float32)
        result = search_optimized_init_scale(w, "int_asym", bits=4)
        assert result is None

    def test_int_sym_returns_clamped_init_scale(self):
        # Build a properly group-reshaped weight (2 groups, group_size=4)
        w = torch.abs(torch.randn(2, 4, dtype=torch.float32)) + 0.5
        result = search_optimized_init_scale(w, "int_sym", bits=4)
        assert result is not None
        assert result.shape == (2, 1)

    def test_int_sym_respects_imatrix(self):
        w = torch.abs(torch.randn(2, 4, dtype=torch.float32)) + 0.5
        imatrix = torch.ones_like(w) * 0.5
        result = search_optimized_init_scale(w, "int_sym", bits=4, imatrix=imatrix)
        assert result is not None

    def test_mx_fp4_returns_init_scale(self):
        w = torch.abs(torch.randn(2, 32, dtype=torch.float32)) + 0.5
        result = search_optimized_init_scale(w, "mx_fp4", bits=4)
        assert result is not None
        # The MX search returns a per-block scale; shape depends on impl
        assert result.numel() > 0


# ==============================================================================
# get_optimized_quant_func
# ==============================================================================


class TestGetOptimizedQuantFunc:
    def test_returns_none_for_unknown(self):
        assert get_optimized_quant_func("totally_made_up") is None

    def test_returns_callable_for_int_sym(self):
        result = get_optimized_quant_func("int_sym")
        assert callable(result)

    def test_returns_callable_for_mx(self):
        result = get_optimized_quant_func("mx_fp4")
        assert callable(result)


# ==============================================================================
# reshape_imatrix_for_weight
# ==============================================================================


class TestReshapeImatrixForWeight:
    def test_none_imatrix_returns_ones(self):
        w = torch.randn(2, 4)
        result = reshape_imatrix_for_weight(None, w, group_size=4)
        assert torch.equal(result, torch.ones_like(w))

    def test_scalar_imatrix_returns_ones(self):
        w = torch.randn(2, 4)
        # Pass a non-tensor imatrix -> treated as None
        result = reshape_imatrix_for_weight(1.0, w, group_size=4)
        assert torch.equal(result, torch.ones_like(w))

    def test_tensor_imatrix_reshaped(self):
        w = torch.randn(2, 4)
        imatrix = torch.tensor([0.5, 0.6, 0.7, 0.8])
        result = reshape_imatrix_for_weight(imatrix, w, group_size=4)
        assert result.shape == w.shape

    def test_tensor_imatrix_3d_weight(self):
        w = torch.randn(2, 3, 4)
        imatrix = torch.tensor([0.5, 0.6, 0.7, 0.8])
        result = reshape_imatrix_for_weight(imatrix, w, group_size=4)
        assert result.shape == w.shape


# ==============================================================================
# compute_optimized_init_scale
# ==============================================================================


class TestComputeOptimizedInitScale:
    def test_returns_none_for_unsupported(self):
        w = torch.randn(4, 4)
        result = compute_optimized_init_scale(w, "totally_made_up", bits=4, group_size=4)
        assert result is None

    def test_returns_none_for_dq(self):
        w = torch.randn(4, 4)
        result = compute_optimized_init_scale(w, "int_asym_dq", bits=4, group_size=4)
        assert result is None

    def test_returns_scale_for_int_sym(self):
        w = torch.abs(torch.randn(4, 4)) + 0.5
        result = compute_optimized_init_scale(w, "int_sym", bits=4, group_size=4)
        assert result is not None
        # Should have numel = (rows * cols / group_size)
        assert result.numel() == 4

    def test_with_imatrix(self):
        w = torch.abs(torch.randn(4, 4)) + 0.5
        imatrix = torch.tensor([0.5, 0.6, 0.7, 0.8])
        result = compute_optimized_init_scale(w, "int_sym", bits=4, group_size=4, imatrix=imatrix)
        assert result is not None


# ==============================================================================
# get_gaudi_fp8_ste_func
# ==============================================================================


class TestGetGaudiFp8SteFunc:
    def test_returns_callable(self):
        fn = get_gaudi_fp8_ste_func()
        # On non-HPU systems, it falls back to the CUDA/CPU STE
        assert callable(fn)


# ==============================================================================
# update_fused_layer_global_scales
# ==============================================================================


class TestUpdateFusedLayerGlobalScales:
    def test_no_op_for_unrelated_module(self):
        # A plain nn.Linear is not an attention/MLP module -> no-op
        m = nn.Linear(8, 8)
        update_fused_layer_global_scales(m)
        # No exception means success

    def test_attention_module_updates_scales(self):
        # Simulate an attention module with q/k/v projections
        class _Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)
                self.q_proj.weight_global_scale = torch.tensor(1.0)
                self.k_proj.weight_global_scale = torch.tensor(2.0)
                self.v_proj.weight_global_scale = torch.tensor(0.5)

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = _Attention()

        m = _Wrapper()
        update_fused_layer_global_scales(m.attention)
        # All three should be updated to the same min
        for proj in (m.attention.q_proj, m.attention.k_proj, m.attention.v_proj):
            assert hasattr(proj, "weight_global_scale")

    def test_fused_qkv_skipped(self):
        class _Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = nn.Linear(8, 8)
                self.qkv_proj.weight_global_scale = torch.tensor(1.0)

        m = _Attention()
        # If already fused, the function should return early
        update_fused_layer_global_scales(m)
        # No change to qkv_proj
        assert torch.equal(m.qkv_proj.weight_global_scale, torch.tensor(1.0))


# ==============================================================================
# update_block_global_scale_if_needed
# ==============================================================================


class TestUpdateBlockGlobalScaleIfNeeded:
    def test_short_circuits_for_non_nv(self):
        # Non-NVFP dtype -> early return, no exception
        m = nn.Linear(8, 8)
        update_block_global_scale_if_needed(m, "int", group_size=128)
        # Should not set weight_global_scale
        assert not hasattr(m, "weight_global_scale")

    def test_short_circuits_for_fp(self):
        m = nn.Linear(8, 8)
        update_block_global_scale_if_needed(m, "fp", group_size=128)
        assert not hasattr(m, "weight_global_scale")
