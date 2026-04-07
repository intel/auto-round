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

_DEVICE_E0M4_TENSORS = {}

# Constants for INT4 values
_E0M4_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


def get_e0m4_tensor(device):
    """Get device-specific E0M4 lookup tensor, creating it if needed."""
    device_str = str(device)
    if device_str not in _DEVICE_E0M4_TENSORS:
        _DEVICE_E0M4_TENSORS[device_str] = torch.tensor(_E0M4_VALUES, dtype=torch.float32, device=device)
    return _DEVICE_E0M4_TENSORS[device_str]


def unpack_int4_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into int4. Each uint8 contains two int4 values
    (low nibble first). The 4-bit indices are mapped to int4 values using kE0M4ToFloat.
    """
    if a.device.type == "cuda":
        return _unpack_int4_from_uint8_cuda(a, m, n, dtype)
    else:
        return _unpack_int4_from_uint8_cpu(a, m, n, dtype)


@torch.compiler.disable()
def _unpack_int4_from_uint8_cpu(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    return _unpack_int4_from_uint8(a, m, n, dtype)


# @torch.compile(fullgraph=True, dynamic=True)
def _unpack_int4_from_uint8_cuda(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    return _unpack_int4_from_uint8(a, m, n, dtype)


# reference: : https://github.com/vllm-project/vllm/pull/16362
def _unpack_int4_from_uint8(
    a: torch.Tensor, m: int, n: int, dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    """
    Unpacks uint8 values into int4. Each uint8 consists of two int4 values
    (i.e. first four bits correspond to one int4 value, last four correspond to a
    consecutive int4 value). The bits represent an index, which are mapped to an int4
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
    kE0M4 = get_e0m4_tensor(device=a.device)
    values = kE0M4[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)
