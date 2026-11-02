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
        raise NotImplementedError("Native TP supports INT, MXFP4 and NVFP4 weight quantization.")
    if scheme.act_bits is None or scheme.act_bits <= 8:
        act_type = getattr(scheme, "act_data_type", None)
        act_blocks = {"mx_fp": 32, "mx_fp4": 32, "mx_fp_rceil": 32, "nv_fp4": 16, "nv_fp4_with_static_gs": 16}
        if not isinstance(act_type, str) or act_type not in act_blocks:
            raise NotImplementedError("Native TP activation quantization supports dynamic MXFP and NVFP4 only.")
        valid_bits = (4, 6, 8) if act_type.startswith("mx_fp") else (4,)
        if (
            scheme.act_bits not in valid_bits
            or not scheme.act_sym
            or scheme.act_group_size != act_blocks[act_type]
            or not scheme.act_dynamic
            or (act_type == "mx_fp4" and scheme.act_bits != 4)
        ):
            raise NotImplementedError(
                f"Native TP {act_type} requires act_bits in {valid_bits}, act_sym=True, "
                f"act_group_size={act_blocks[act_type]} and act_dynamic=True."
            )
    group_size = scheme.group_size
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size == 0 or group_size < -1:
        raise NotImplementedError("Native TP requires a positive integer group_size or -1.")
    if data_type in blocks and (scheme.bits != 4 or not scheme.sym or group_size != blocks[data_type]):
        raise NotImplementedError(
            f"Native TP {data_type} requires bits=4, sym=True and group_size={blocks[data_type]}."
        )
