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

import torch

from auto_round.data_type.fp8 import float8_e4m3fn_ste
from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad


# taken from
# https://github.com/vllm-project/vllm/blob/ebb554cdb7cd9cc54b2feec20c45ab9cd9067d52/tests/kernels/test_nvfp4_quant.py
def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def cast_to_fp4_ste(x):
    fp4 = (cast_to_fp4(x).to(x.dtype) - x).detach() + x

    return fp4


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT8_E4M3_MIN = torch.finfo(torch.float8_e4m3fn).min

def ref_nvfp4_quant(x, global_scale, block_size=16, v=0):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = float8_e4m3fn_ste(scale).to(torch.float32)
    scale = torch.clamp(scale, min=FLOAT8_E4M3_MIN, max=FLOAT8_E4M3_MAX)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale + v
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0)
    return (cast_to_fp4_ste(clipped_x) * get_reciprocal(output_scale)).reshape(m, n), output_scale


@register_dtype("nv_fp4")
def full_quant(tensor, bits=4, group_size=16, v=0, **kwargs):
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    tensor_amax = tensor.abs().max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    qdq_res, output_scale = ref_nvfp4_quant(tensor, global_scale, group_size, v)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), output_scale, None


FLOAT8_UE5M3_MAX = 114688


def float_to_e5m3_frexp(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    e5m3 = torch.zeros_like(x, dtype=torch.uint8)

    mask = x > 0
    x_masked = x[mask]

    # 正常数：x >= 2^-14
    normal_mask = x_masked >= 2 ** -14
    x_normal = x_masked[normal_mask]
    mantissa, exponent = torch.frexp(x_normal)

    m3 = torch.clamp(torch.round((mantissa - 0.5) * 16), 0, 7).to(torch.uint8)
    e5 = torch.clamp(exponent + 14, 0, 31).to(torch.uint8)  # 0 reserved for subnormal, 31 reserved for NaN

    e5m3_vals = ((e5 << 3) | m3).to(torch.uint8)

    # sumnorm：0 < x < 2^-14
    subnormal_mask = ~normal_mask
    x_subnormal = x_masked[subnormal_mask]
    m_sub = torch.clamp(torch.round(x_subnormal / (2 ** -14) * 8), 1, 7).to(torch.uint8)  # exponent = 0
    e5m3_sub = m_sub  # top 5 bits = 0

    out_vals = torch.zeros_like(x_masked, dtype=torch.uint8)
    out_vals[normal_mask] = e5m3_vals
    out_vals[subnormal_mask] = e5m3_sub

    e5m3[mask] = out_vals
    return e5m3


def e5m3_to_float_tensor(e5m3: torch.Tensor) -> torch.Tensor:
    assert e5m3.dtype == torch.uint8

    x = torch.zeros_like(e5m3, dtype=torch.float32)
    mask_nonzero = e5m3 != 0
    e = ((e5m3[mask_nonzero] >> 3) & 0x1F).to(torch.int32)
    m = (e5m3[mask_nonzero] & 0x07).to(torch.int32)

    is_nan = (e == 31) & (m == 7)
    is_subnormal = (e == 0)
    is_normal = (e > 0) & (~is_nan)

    out = torch.zeros_like(e, dtype=torch.float32)

    # subnormal: exponent = -14, no implicit leading 1
    out[is_subnormal] = (m[is_subnormal].float() / 8.0) * (2 ** -14)

    # normal: exponent = e - 15, implicit leading 1
    mant = 1.0 + m[is_normal].float() / 8.0
    exp = e[is_normal] - 15
    out[is_normal] = torch.ldexp(mant, exp)

    out[is_nan] = float('nan')
    x[mask_nonzero] = out
    return x


def cast_to_ue5m3(tensor):
    orig_dtype = tensor.dtype
    encoded = float_to_e5m3_frexp(tensor)
    res = e5m3_to_float_tensor(encoded)
    res = res.to(orig_dtype)
    return res


def cast_to_ue5m3_ste(x):
    fp4 = (cast_to_ue5m3(x).to(x.dtype) - x).detach() + x

    return fp4


if __name__ == "__main__":
    test = torch.tensor(
        [0.0, 2 ** (-17), (2 ** -14) * 0.875, 2 ** -14, 2 ** -13, 2 ** -6,
         1e-6, 2.7657e-05, 0.1, 1.0, 3.14, 1000.0,
         114688,
         1e10],
        dtype=torch.float32)
    encoded = float_to_e5m3_frexp(test)
    decoded = e5m3_to_float_tensor(encoded)
    decoded_bf16 = decoded.to(torch.bfloat16)
    print(decoded_bf16)

    for i in range(len(test)):
        print(
            f"{test[i].item():.6g} -> {encoded[i].item():3d} -> {decoded[i].item():.6g} "
            f"(error={abs(test[i] - decoded[i]).item():.3g})")
