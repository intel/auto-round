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

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.export.export_to_autoround.export_to_fp8 import (
    FP8BlockQLinear,
    FP8QLinear,
    pack_layer as pack_layer_fp8,
)
from auto_round.export.export_to_autoround.export_to_nvfp_mx import pack_layer as pack_layer_nvfp
from auto_round.export.export_to_autoround.qlinear_fp import (
    QuantLinear as FpQuantLinear,
    _pack_fp4_to_uint8,
    pack_fp4_to_uint8,
)
from auto_round.export.export_to_autoround.qlinear_int import (
    QuantLinear as IntQuantLinear,
    _pack_int4_to_uint8,
    pack_int4_to_uint8,
)
from auto_round.export.export_to_autoround.qlinear_triton_act import QuantLinear as TritonActQuantLinear

# ==============================================================================
# 4-bit packing helpers
# ==============================================================================


class TestPack4ToUint8:
    def test_pack_fp4_shape_and_dtype(self):
        x = torch.tensor([[1.0, -2.0, 3.0, 0.5]], dtype=torch.float32)
        packed = _pack_fp4_to_uint8(x)
        assert packed.shape == (1, 2)
        assert packed.dtype == torch.uint8

    def test_pack_fp4_cpu_dispatch(self):
        x = torch.tensor([[1.0, -2.0, 3.0, 0.5]], dtype=torch.float32)
        packed = pack_fp4_to_uint8(x)
        assert packed.shape == (1, 2)

    def test_pack_int4_shape_and_dtype(self):
        x = torch.tensor([[1.0, -2.0, 3.0, 0.5]], dtype=torch.float32)
        packed = _pack_int4_to_uint8(x)
        assert packed.shape == (1, 2)
        assert packed.dtype == torch.uint8

    def test_pack_int4_cpu_dispatch(self):
        x = torch.tensor([[1.0, -2.0, 3.0, 0.5]], dtype=torch.float32)
        packed = pack_int4_to_uint8(x)
        assert packed.shape == (1, 2)


# ==============================================================================
# qlinear_fp.QuantLinear
# ==============================================================================


class TestFpQuantLinear:
    def _make_linear(self):
        return nn.Linear(32, 8)

    def test_invalid_bits_raise(self):
        with pytest.raises(NotImplementedError, match="Only 4,8 bits"):
            FpQuantLinear(3, 32, 32, 8, False)

    def test_mx_group_size_validation(self):
        with pytest.raises(NotImplementedError, match="group_size 32"):
            FpQuantLinear(4, 16, 32, 8, False, data_type="mx_fp4")

    def test_nv_group_size_validation(self):
        with pytest.raises(NotImplementedError, match="group_size 16"):
            FpQuantLinear(4, 8, 32, 8, False, data_type="nv_fp4")

    def test_pack_mxfp8(self):
        ql = FpQuantLinear(8, 32, 32, 8, True, data_type="mx_fp8e4m3")
        scales = torch.ones(8, 1)
        with patch("auto_round.export.export_to_autoround.qlinear_fp.get_packing_device", return_value="cpu"):
            ql.pack(self._make_linear(), scales)
        assert ql.weight.dtype == torch.float8_e4m3fn

    def test_pack_mxfp4(self):
        ql = FpQuantLinear(4, 32, 32, 8, False, data_type="mx_fp4")
        scales = torch.ones(8, 1)
        with patch("auto_round.export.export_to_autoround.qlinear_fp.get_packing_device", return_value="cpu"):
            ql.pack(self._make_linear(), scales)
        assert ql.weight_packed.dtype == torch.uint8

    def test_post_init_noop(self):
        ql = FpQuantLinear(4, 32, 32, 8, False, data_type="mx_fp4")
        ql.post_init()  # must not raise


class TestIntQuantLinear:
    def test_pack_int4(self):
        ql = IntQuantLinear(4, 32, 32, 8, False, data_type="mx_int4")
        lin = nn.Linear(32, 8)
        scales = torch.ones(8, 1)
        with patch("auto_round.export.export_to_autoround.qlinear_int.get_packing_device", return_value="cpu"):
            ql.pack(lin, scales)
        assert ql.weight_packed.dtype == torch.uint8


# ==============================================================================
# qlinear_triton_act.QuantLinear
# ==============================================================================


class TestTritonActQuantLinear:
    def test_init_registers_buffers(self):
        q = TritonActQuantLinear(4, 32, 64, 64, True)
        assert q.qweight.shape == (8, 64)
        assert q.scales.shape == (2, 64)
        assert q.qzeros.shape == (2, 8)

    def test_invalid_bits_raise(self):
        with pytest.raises(NotImplementedError, match="Only 2,4,8 bits"):
            TritonActQuantLinear(3, 32, 64, 64, False)

    def test_optional_bias(self):
        q = TritonActQuantLinear(4, 32, 64, 64, False)
        assert q.bias is None

    def test_repr(self):
        q = TritonActQuantLinear(4, 32, 64, 64, False)
        assert "QuantLinear" in repr(q)

    def test_pack(self):
        q = TritonActQuantLinear(4, 32, 64, 64, True)
        lin = nn.Linear(64, 64)
        scales = torch.ones(1, 64)
        zeros = torch.zeros(1, 64)
        with patch("auto_round.export.export_to_autoround.qlinear_triton_act.get_packing_device", return_value="cpu"):
            q.pack(lin, scales, zeros, torch.ones(1), torch.ones(1))
        assert q.qweight.shape == (8, 64)

    def test_warmup_returns_none(self):
        assert TritonActQuantLinear.warmup(nn.Linear(64, 64)) is None


# ==============================================================================
# FP8 export_to_fp8
# ==============================================================================


class TestFp8LinearModules:
    def test_fp8qlinear_registers_buffers(self):
        l = FP8QLinear(
            4,
            8,
            torch.randn(8, 4),
            torch.ones(8, 1),
            bias=torch.zeros(8),
            weight_zp=torch.tensor(1.0),
            input_scale=torch.ones(1),
        )
        assert l.weight.shape == (8, 4)
        assert l.weight_scale.shape == (8, 1)
        assert hasattr(l, "weight_zp")
        assert hasattr(l, "input_scale")

    def test_fp8qlinear_optional_args(self):
        l = FP8QLinear(4, 8, torch.randn(8, 4), torch.ones(1), dtype=torch.float32)
        assert l.bias is None
        assert not hasattr(l, "weight_zp")
        assert not hasattr(l, "input_scale")

    def test_fp8blockqlinear(self):
        l = FP8BlockQLinear(4, 8, torch.randn(8, 4), torch.ones(2, 1, 1), dtype=torch.bfloat16)
        assert l.weight_scale_inv.shape == (2, 1, 1)


class TestFp8PackLayer:
    def _quantized_linear(self):
        lin = nn.Linear(4, 8)
        lin.scale = torch.ones(8, 1)
        lin.zp = None
        lin.act_scale = torch.ones(4)
        lin.group_size = 4
        lin.bits = 4
        lin.data_type = "fp8"
        lin.sym = False
        lin.act_bits = 16
        return lin

    def test_pack_replaces_with_fp8qlinear(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float32
                self.layers = nn.ModuleList([nn.Linear(4, 8)])

        m = Model()
        m.layers[0] = self._quantized_linear()
        with patch("auto_round.export.export_to_autoround.export_to_fp8.get_packing_device", return_value="cpu"):
            pack_layer_fp8("layers.0", m, "fp8", device="cpu")
        assert isinstance(m.layers[0], FP8QLinear)
        assert m.layers[0].weight.dtype == torch.float8_e4m3fn

    def test_pack_skips_unquantized(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float32
                self.layers = nn.ModuleList([nn.Linear(4, 8)])

        m = Model()  # plain Linear without quant attrs
        with patch("auto_round.export.export_to_autoround.export_to_fp8.get_packing_device", return_value="cpu"):
            pack_layer_fp8("layers.0", m, "fp8", device="cpu")
        assert isinstance(m.layers[0], nn.Linear)


# ==============================================================================
# NVFP/MX export_to_nvfp_mx
# ==============================================================================


class TestNvfpPackLayer:
    def _quantized_linear(self):
        lin = nn.Linear(32, 8)
        lin.data_type = "nv_fp4"
        lin.act_bits = 16
        lin.act_data_type = "fp"
        lin.bits = 4
        lin.group_size = 32
        lin.sym = True
        lin.scale = torch.ones(8, 1)
        lin.weight_global_scale = torch.tensor([1.0])
        return lin

    def test_pack_skips_non_supported_type(self):
        # A non-supported layer type is treated as already-packed -> skip.
        class _Custom(nn.Module):
            pass

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Custom()])

        m = Model()
        with patch("auto_round.export.export_to_autoround.export_to_nvfp_mx.get_module") as gm:
            gm.return_value = m.layers[0]
            with patch("auto_round.export.export_to_autoround.export_to_nvfp_mx.set_module") as sm:
                pack_layer_nvfp("layers.0", m, "fp", device="cpu")
                sm.assert_not_called()

    def test_pack_quantized_linear(self):
        from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear as FpQLinear

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(32, 8)])

        m = Model()
        m.layers[0] = self._quantized_linear()
        with patch("auto_round.export.export_to_autoround.export_to_nvfp_mx.get_module") as gm:
            gm.return_value = m.layers[0]
            with patch("auto_round.export.export_to_autoround.export_to_nvfp_mx.set_module") as sm:
                pack_layer_nvfp("layers.0", m, "fp", device="cpu")
            sm.assert_called_once()
            # set_module(model, name, qlayer) -> layer is the 3rd arg
            new_layer = sm.call_args[0][2]
            assert isinstance(new_layer, FpQLinear)
            assert new_layer.data_type == "nv_fp4"
