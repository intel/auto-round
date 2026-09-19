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

import hashlib

import pytest
import torch

from auto_round.data_type.mxfp import quant_mx_rceil
from auto_round.export.svdquant_mxfp4 import (
    NunchakuMXFP4Packer,
    pack_lowrank_weight,
    unpack_lowrank_weight,
)


@pytest.mark.parametrize("down,shape", [(True, (3, 17)), (False, (17, 3))])
def test_lowrank_pack_roundtrips_and_matches_fixed_layout(down, shape):
    logical = torch.arange(shape[0] * shape[1], dtype=torch.float16).reshape(shape)
    expected_hash = {
        True: "f62c895e44a7139fc942941b1244857d65143dfe52ad852e8847339aa6119029",
        False: "0690f5c24d1ca25ad4f7714d9ce6b626b792ac89e46986d7a73dd0f327b163b4",
    }[down]

    packed = pack_lowrank_weight(logical, down=down)
    unpacked = unpack_lowrank_weight(packed, down=down)

    assert packed.shape == (128, 16)
    torch.testing.assert_close(unpacked[: shape[0], : shape[1]], logical)
    assert hashlib.sha256(bytes(packed.view(torch.uint8).flatten().tolist())).hexdigest() == expected_hash


@pytest.mark.parametrize("shape", [(128, 128), (7, 65)])
def test_nunchaku_mxfp4_pack_roundtrip_matches_autoround_rceil_qdq(shape):
    weight = torch.randn(shape, generator=torch.Generator().manual_seed(20260713)) * 3
    expected, _, _ = quant_mx_rceil(weight, bits=4, group_size=32, data_type="mx_fp4e2m1")
    packer = NunchakuMXFP4Packer()

    packed = packer.pack_residual(weight)
    actual = packer.unpack_residual(packed.qweight, packed.wscales, packed.logical_shape)

    assert packed.padded_shape[0] % 128 == 0
    assert packed.padded_shape[1] % 128 == 0
    torch.testing.assert_close(actual, expected)
