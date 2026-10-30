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
"""Tests for ``WeightTypeHandler.convert_layer`` of the MXFP4/MXFP8/NVFP4/WOQ handlers.

Layers are tiny hand-built modules that mimic compressed-tensors ``CompressedLinear``
buffers, so dequantization is checked numerically on CPU without loading a checkpoint.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.utils.weight_handler import ModuleWeightType, get_handler

OUT, IN = 4, 32


class _CompressedLinear(nn.Module):
    def __init__(self, bias=True, **buffers):
        super().__init__()
        self.in_features, self.out_features = IN, OUT
        self.bias = nn.Parameter(torch.arange(OUT, dtype=torch.float32)) if bias else None
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)
        self.bits, self.data_type, self.global_name = 4, "mx_fp", "model.layer"


def _row_scales():
    # e8m0 exponent bias is 127, so these rows scale by 1, 2, 4, 0.5
    return torch.tensor([[127], [128], [129], [126]], dtype=torch.uint8)


EXPECTED_ROW_SCALE = torch.tensor([1.0, 2.0, 4.0, 0.5]).reshape(OUT, 1)


def _check_common(new_layer, layer, dtype):
    assert type(new_layer) is nn.Linear
    assert new_layer.weight.dtype == dtype
    assert torch.equal(new_layer.bias, torch.arange(OUT, dtype=dtype))
    assert new_layer.bits == 4 and new_layer.data_type == "mx_fp" and new_layer.global_name == "model.layer"
    assert new_layer.group_size is None
    assert all(b.device.type == "meta" for b in layer.buffers())


class TestMXFP4Convert:
    def test_dequantizes_packed_weight(self):
        # 0x22 packs two e2m1 nibbles with value 1.0
        layer = _CompressedLinear(
            weight_packed=torch.full((OUT, IN // 2), 0x22, dtype=torch.uint8), weight_scale=_row_scales()
        )
        new_layer = get_handler(ModuleWeightType.MXFP4).convert_layer(layer, dtype=torch.float32, to_cpu=True)
        _check_common(new_layer, layer, torch.float32)
        assert torch.equal(new_layer.weight, EXPECTED_ROW_SCALE.expand(OUT, IN))

    def test_falls_back_to_weight_attr_on_gaudi2(self):
        layer = _CompressedLinear(
            bias=False, weight=torch.full((OUT, IN // 2), 0x55, dtype=torch.uint8), weight_scale=_row_scales()
        )
        with patch("auto_round.utils.device.is_gaudi2", return_value=True):
            new_layer = get_handler(ModuleWeightType.MXFP4).convert_layer(layer, device="hpu")
        assert new_layer.bias is None and new_layer.weight.dtype == torch.bfloat16
        # nibble 0x5 is 3.0 in e2m1
        assert torch.equal(new_layer.weight.float(), 3.0 * EXPECTED_ROW_SCALE.expand(OUT, IN))

    def test_missing_packed_weight(self):
        layer = _CompressedLinear(weight_scale=_row_scales())
        with pytest.raises(ValueError, match="weight_packed and weight_scale"):
            get_handler(ModuleWeightType.MXFP4).convert_layer(layer)


class TestMXFP8Convert:
    def test_dequantizes_fp8_weight(self):
        base = torch.linspace(-2, 2, OUT * IN).reshape(OUT, IN)
        weight_fp8 = base.to(torch.float8_e4m3fn)
        layer = _CompressedLinear(weight=weight_fp8, weight_scale=_row_scales())
        new_layer = get_handler(ModuleWeightType.MXFP8).convert_layer(layer, dtype=torch.float32, to_cpu=True)
        _check_common(new_layer, layer, torch.float32)
        assert torch.equal(new_layer.weight, weight_fp8.float() * EXPECTED_ROW_SCALE)

    def test_gaudi2_forces_cpu(self):
        layer = _CompressedLinear(weight=torch.ones(OUT, IN, dtype=torch.float8_e4m3fn), weight_scale=_row_scales())
        with patch("auto_round.utils.device.is_gaudi2", return_value=True):
            new_layer = get_handler(ModuleWeightType.MXFP8).convert_layer(layer, device="hpu")
        assert torch.equal(new_layer.weight.float(), EXPECTED_ROW_SCALE.expand(OUT, IN))


def _compressor_layer(bias=True):
    layer = _CompressedLinear(bias=bias, weight_packed=torch.zeros(OUT, IN // 2, dtype=torch.uint8))
    dq = torch.randn(OUT, IN)
    layer.compressor = MagicMock()
    layer.compressor.decompress_module.return_value = dq
    return layer, dq


@pytest.mark.parametrize("weight_type", [ModuleWeightType.NVFP4, ModuleWeightType.WOQ])
class TestCompressorBackedConvert:
    def test_uses_compressor_decompress(self, weight_type):
        layer, dq = _compressor_layer()
        new_layer = get_handler(weight_type).convert_layer(layer, dtype=torch.float32, to_cpu=True)
        layer.compressor.decompress_module.assert_called_once_with(layer)
        assert torch.equal(new_layer.weight, dq)
        assert torch.equal(new_layer.bias, torch.arange(OUT, dtype=torch.float32))
        assert layer.weight_packed.device.type == "meta"

    def test_without_bias(self, weight_type):
        layer, dq = _compressor_layer(bias=False)
        new_layer = get_handler(weight_type).convert_layer(layer)
        assert new_layer.bias is None
        assert torch.equal(new_layer.weight, dq.to(torch.bfloat16))

    def test_linear_with_quantization_scheme_decompresses_in_place(self, weight_type):
        linear = nn.Linear(IN, OUT)
        linear.quantization_scheme = object()
        decompress = MagicMock()
        base_mod = types.ModuleType("compressed_tensors.compressors.base")
        base_mod.decompress_module = decompress
        with patch.dict(sys.modules, {"compressed_tensors.compressors.base": base_mod}):
            assert get_handler(weight_type).convert_layer(linear) is linear
        decompress.assert_called_once_with(linear)


def test_nvfp4_gaudi2_forces_cpu():
    layer, dq = _compressor_layer()
    with patch("auto_round.utils.device.is_gaudi2", return_value=True):
        new_layer = get_handler(ModuleWeightType.NVFP4).convert_layer(layer, dtype=torch.float32, device="hpu")
    assert torch.equal(new_layer.weight, dq)
