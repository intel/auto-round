# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.data_type.fp8``."""

import torch

from auto_round.data_type.fp8 import (
    quant_block_fp_sym,
    quant_fp8_sym,
    quant_fp8_e5m2,
    quant_fp8_unit_scale,
    quant_fp8_e5m2_unit_scale,
    quant_fp8_sym_gaudi3,
)


class TestQuantBlockFpSym:
    """Test quant_block_fp_sym function."""

    def test_basic(self):
        t = torch.randn(128, 128, dtype=torch.bfloat16)
        q, s, z = quant_block_fp_sym(t, group_size=(128, 128))
        assert q.shape == t.shape
        assert q.dtype == t.dtype
        assert s is not None
        assert z is None

    def test_with_max_scale(self):
        t = torch.randn(64, 64, dtype=torch.bfloat16)
        ms = torch.tensor(1.0)
        q, s, z = quant_block_fp_sym(t, max_scale=ms, group_size=(64, 64))
        assert q.shape == t.shape
        assert z is None

    def test_with_tensor_max(self):
        t = torch.randn(64, 64, dtype=torch.bfloat16)
        tm = torch.tensor([[1.0]])
        q, s, z = quant_block_fp_sym(t, tensor_max=tm, group_size=(64, 64))
        assert q.shape == t.shape

    def test_with_tensor_max_and_min(self):
        t = torch.randn(64, 64, dtype=torch.bfloat16)
        q, s, z = quant_block_fp_sym(
            t, tensor_max=torch.tensor([[1.0]]), tensor_min=torch.tensor([[-0.5]]), group_size=(64, 64)
        )
        assert q.shape == t.shape

    def test_float16_preserved(self):
        t = torch.randn(64, 64, dtype=torch.float16)
        q, s, z = quant_block_fp_sym(t, group_size=(64, 64))
        assert q.dtype == torch.float16


class TestQuantFp8Sym:
    """Test quant_fp8_sym function."""

    def test_basic(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_sym(t)
        assert q.shape == t.shape
        assert z is None

    def test_dynamic_scale(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_sym(t, max_scale=1.0)
        assert q.shape == t.shape

    def test_with_tensor_max(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_sym(t, tensor_max=torch.tensor(1.0))
        assert q.shape == t.shape

    def test_with_max_and_min(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_sym(t, tensor_max=torch.tensor(0.5), tensor_min=torch.tensor(-0.5))
        assert q.shape == t.shape

    def test_with_v(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_sym(t, v=torch.zeros(128, dtype=torch.bfloat16))
        assert q.shape == t.shape

    def test_float16(self):
        t = torch.randn(4, 128, dtype=torch.float16)
        q, s, z = quant_fp8_sym(t)
        assert q.shape == t.shape


class TestQuantFp8E5m2:
    """Test quant_fp8_e5m2 function."""

    def test_basic(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_e5m2(t)
        assert q.shape == t.shape
        assert z is None

    def test_with_max_and_min(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_e5m2(t, tensor_max=torch.tensor(0.5), tensor_min=torch.tensor(-0.5))
        assert q.shape == t.shape


class TestQuantFp8UnitScale:
    """Test quant_fp8_unit_scale function."""

    def test_basic(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_unit_scale(t)
        assert q.shape == t.shape
        assert z is None

    def test_with_v(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_unit_scale(t, v=torch.zeros(128, dtype=torch.bfloat16))
        assert q.shape == t.shape


class TestQuantFp8E5m2UnitScale:
    """Test quant_fp8_e5m2_unit_scale function."""

    def test_basic(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_e5m2_unit_scale(t)
        assert q.shape == t.shape
        assert z is None


class TestQuantFp8SymGaudi3:
    """Test quant_fp8_sym_gaudi3 function."""

    def test_basic(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_sym_gaudi3(t)
        assert q.shape == t.shape
        assert z is None

    def test_with_max_and_min(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        q, s, z = quant_fp8_sym_gaudi3(
            t, tensor_max=torch.tensor(0.5), tensor_min=torch.tensor(-0.5)
        )
        assert q.shape == t.shape
