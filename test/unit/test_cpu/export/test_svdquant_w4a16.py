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

import torch

from auto_round.export.svdquant_w4a16 import (
    dequantize_adanorm_w4a16,
    pack_adanorm_w4a16,
    quantize_adanorm_w4a16_rtn,
    unpack_adanorm_w4a16,
)


def _representable_fixture(dtype=torch.float16):
    rows = torch.arange(12).reshape(12, 1)
    columns = torch.arange(1024).reshape(1, 1024)
    signed = ((rows * 5 + columns * 3) % 15 - 7).to(torch.float32)
    scales = (torch.arange(12 * 16).reshape(12, 16) % 13 + 1).to(dtype) / 8
    weight = (signed.reshape(12, 16, 64) * scales.unsqueeze(-1)).reshape(12, 1024).to(dtype)
    return weight, scales, signed.to(torch.int8)


def test_adanorm_w4a16_pack_uses_runtime_layout_and_roundtrips_codes():
    weight, scales, signed = _representable_fixture()

    packed = pack_adanorm_w4a16(weight, scales, splits=3)

    assert packed.qweight.shape == (3, 512)
    assert packed.wscales.shape == (16, 12)
    assert packed.wzeros.shape == (16, 12)
    expected_codes = signed.reshape(3, 4, 1024).permute(1, 0, 2).reshape(12, 1024)
    expected_weight = weight.reshape(3, 4, 1024).permute(1, 0, 2).reshape(12, 1024)
    assert torch.equal(unpack_adanorm_w4a16(packed), expected_codes)
    torch.testing.assert_close(dequantize_adanorm_w4a16(packed), expected_weight)


def test_adanorm_w4a16_rtn_emits_finite_runtime_payload():
    weight = torch.randn(12, 1024, generator=torch.Generator().manual_seed(7), dtype=torch.bfloat16)

    packed = quantize_adanorm_w4a16_rtn(weight, splits=3)

    assert packed.qweight.dtype == torch.int32
    assert torch.isfinite(packed.wscales).all()
    assert torch.isfinite(packed.wzeros).all()
    assert torch.isfinite(dequantize_adanorm_w4a16(packed)).all()
