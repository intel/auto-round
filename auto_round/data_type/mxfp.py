# Copyright (c) 2024 Intel Corporation
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
from .utils import floor_ste, round_ste
from auto_round.data_type.register import register_dtype, QUANT_FUNC_WITH_DTYPE

MXFP_FORMAT_CACHE = {
    # data type: ebits, mbits, emax, max_norm, min_norm
    "mx_int8": (0, 8, 0, 1.984375, 0),
    "mx_int4": (0, 4, 0, 1.75, 0),
    "mx_int2": (0, 2, 0, 1.0, 0),
    "mx_fp8e5m2": (5, 4, 15, 57344.0, 6.103515625e-05),
    "mx_fp8e4m3": (4, 5, 8, 448.0, 0.015625),
    "mx_fp6e3m2": (3, 4, 4, 28.0, 0.25),
    "mx_fp6e2m3": (2, 5, 2, 7.5, 1.0),
    "mx_fp4": (2, 3, 2, 6.0, 1.0),
    "mx_fp4e2m1": (2, 3, 2, 6.0, 1.0),
    "mx_float16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_fp16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_bfloat16": (8, 9, 127, 3.3895313892515355e+38, 1.1754943508222875e-38),
    "mx_bf16": (8, 9, 127, 3.3895313892515355e+38, 1.1754943508222875e-38),
}

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def quant_mx(weight, data_type, v, max_scale, **kwargs):
    ebits, mbits, emax, max_norm, min_norm = MXFP_FORMAT_CACHE[data_type]
    shared_exp, _ = torch.max(torch.abs(weight), dim=-1, keepdim=True)
    if isinstance(max_scale, torch.Tensor):
        shared_exp *= (max_scale.unsqueeze(dim=-1))
    else:
        shared_exp *= max_scale

    shared_exp = torch.log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype))
    shared_exp = (shared_exp - emax)
    shared_exp = floor_ste(shared_exp)
    scale_emax = 2 ** (8 - 1) - 1
    shared_exp[shared_exp > scale_emax] = scale_emax  ##changed Nan
    shared_exp[shared_exp < -scale_emax] = -scale_emax
    weight = weight / (2 ** shared_exp)
    multiply = 2 if data_type == "mx_fp4" else 1  ## 2 is a tricky setting
    weight = weight + v * multiply
    if ebits != 0:
        private_exp = floor_ste(torch.log2(torch.abs(weight) + (weight == 0).type(weight.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2 ** (ebits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of mbits are in the integer portion of the number
    weight = weight * (2 ** (mbits - 2)) if private_exp is None else weight / (2 ** private_exp) * (2 ** (mbits - 2))

    weight = torch.sign(weight) * round_ste(torch.abs(weight))  ##adopt round-to-floor which we found is much better
    max_mantissa = 2 ** (mbits - 1) - 1
    weight = torch.clamp(weight, -max_mantissa, max_mantissa)

    # Undo scaling
    weight = weight / (2 ** (mbits - 2)) if private_exp is None else weight / (2 ** (mbits - 2)) * (2 ** private_exp)

    weight = torch.clamp(weight, min=-max_norm, max=max_norm)
    weight = weight * (2 ** shared_exp)
    return weight, shared_exp, None


for key in MXFP_FORMAT_CACHE.keys():
    QUANT_FUNC_WITH_DTYPE[key + "_asym"] = quant_mx
    QUANT_FUNC_WITH_DTYPE[key + "_sym"] = quant_mx
