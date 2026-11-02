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

"""Shared capability checks for native TP entry points and individual layers."""


def validate_tp_quantization(scheme):
    """Reject unsupported schemes without changing their quantization semantics."""
    data_type = scheme.data_type
    blocks = {"mx_fp": 32, "mx_fp4": 32, "nv_fp": 16, "nv_fp4": 16}
    if data_type != "int" and data_type not in blocks:
        raise NotImplementedError("Native TP supports INT, MXFP4 and NVFP4 weight-only quantization.")
    if scheme.act_bits is None or scheme.act_bits <= 8:
        raise NotImplementedError(
            'Native TP supports weight-only quantization; explicitly set act_bits=16 and act_data_type="float" for FP4.'
        )
    group_size = scheme.group_size
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size == 0 or group_size < -1:
        raise NotImplementedError("Native TP requires a positive integer group_size or -1.")
    if data_type in blocks and (scheme.bits != 4 or not scheme.sym or group_size != blocks[data_type]):
        raise NotImplementedError(
            f"Native TP {data_type} requires bits=4, sym=True and group_size={blocks[data_type]}."
        )
