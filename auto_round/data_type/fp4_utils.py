# Copyright (c) 2025 Intel Corporation
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


# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

kE2M1ToFloat = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def unpack_fp4_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into FP4. Each uint8 contains two FP4 values
    (low nibble first). The 4-bit indices are mapped to FP4 values using kE2M1ToFloat.
    """
    if a.device.type == "cuda":
        return _unpack_fp4_from_uint8_cuda(a, m, n, dtype)
    else:
        return _unpack_fp4_from_uint8_cpu(a, m, n, dtype)


@torch.compiler.disable()
def _unpack_fp4_from_uint8_cpu(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    return _unpack_fp4_from_uint8(a, m, n, dtype)


@torch.compile(fullgraph=True, dynamic=True)
def _unpack_fp4_from_uint8_cuda(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    return _unpack_fp4_from_uint8(a, m, n, dtype)


# reference: : https://github.com/vllm-project/vllm/pull/16362
# @torch.compile(fullgraph=True, dynamic=True)
def _unpack_fp4_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into fp4. Each uint8 consists of two fp4 values
    (i.e. first four bits correspond to one fp4 value, last four correspond to a
    consecutive fp4 value). The bits represent an index, which are mapped to an fp4
    value.

    :param a: tensor to unpack
    :param m: original dim 0 size of the unpacked tensor
    :param n: original dim 1 size of the unpacked tensor
    :param dtype: dense dtype to cast the unpacked tensor to
    """
    assert a.dtype == torch.uint8, f"expected uint8, got {a.dtype}"

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)
