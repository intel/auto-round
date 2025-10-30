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
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger


# taken from
# https://github.com/vllm-project/vllm/blob/ebb554cdb7cd9cc54b2feec20c45ab9cd9067d52/tests/kernels/test_nvfp4_quant.py
def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)

    step1 = round_ste(2.0 * x) / 2.0
    step2 = round_ste(x)
    step3 = 2.0 * round_ste(x / 2.0)

    mask1 = x < 2.0
    mask2 = x < 4.0
    x = step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)
    x = x.clamp(-6, 6)

    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.zeros_like(x, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max if hasattr(torch, "float8_e4m3fn") else 448
FLOAT8_E4M3_MIN = torch.finfo(torch.float8_e4m3fn).min if hasattr(torch, "float8_e4m3fn") else -448


def calculate_gparam(tensor, group_size=16, device="cpu"):
    assert group_size == 16
    if isinstance(tensor, (float, int)):
        tensor_amax = torch.ones((1), device=device) * tensor
    elif isinstance(tensor, torch.Tensor):
        tensor_amax = tensor.abs().max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_amax)
    return global_scale


def ref_nvfp4_quant(x, global_scale, block_size=16, v=0, scale_coeff=1.0):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    if isinstance(scale_coeff, torch.Tensor):
        scale_coeff = scale_coeff.view(-1, 1).to(x.device)
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32) * scale_coeff
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = torch.clamp(scale, min=FLOAT8_E4M3_MIN, max=FLOAT8_E4M3_MAX)
    scale = float8_e4m3fn_ste(scale).to(torch.float32)  ##e4m3 does not support torch compile
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))
    scaled_x = x.to(torch.float32) * output_scale + v
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0)
    return (cast_to_fp4(clipped_x) * get_reciprocal(output_scale)).reshape(m, n), scale


@register_dtype("nv_fp4")
def nv_fp4(tensor, bits=4, group_size=16, v=0, global_scale=None, max_scale=1.0, **kwargs):
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if global_scale is None:
        tensor_max = tensor.abs().max().to(torch.float32)
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    global_scale = global_scale.to(device=tensor.device, dtype=torch.float32)
    qdq_res, scale = ref_nvfp4_quant(tensor, global_scale, group_size, v, scale_coeff=max_scale)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


@register_dtype("nv_fp4_with_static_gs")
def nv_fp4_with_static_gs(tensor, bits=4, group_size=16, v=0, tensor_max=None, **kwargs):
    if tensor is None or tensor.numel() == 0:
        return tensor, None, None
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor_max is None:
        tensor_max = tensor.abs().max().to(torch.float32)
    else:
        if not isinstance(tensor_max, torch.Tensor):
            tensor_max = torch.tensor(tensor_max, device=tensor.device, dtype=torch.float32)
        else:
            tensor_max = tensor_max.to(device=tensor.device, dtype=torch.float32)
        if tensor_max.numel() != 1:
            tensor_max = tensor_max.abs().max()

    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    global_scale = global_scale.to(tensor.device)
    qdq_res, scale = ref_nvfp4_quant(tensor, global_scale, group_size, v)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


FLOAT8_UE5M3_MAX = 114688


def float_to_e5m3_frexp(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    e5m3 = torch.zeros_like(x, dtype=torch.uint8)

    mask = x > 0
    x_masked = x[mask]

    # normal number: x >= 2^-14
    normal_mask = x_masked >= 2**-14
    x_normal = x_masked[normal_mask]
    mantissa, exponent = torch.frexp(x_normal)

    m3 = torch.clamp(torch.round((mantissa - 0.5) * 16), 0, 7).to(torch.uint8)
    e5 = torch.clamp(exponent + 14, 0, 31).to(torch.uint8)  # 0 reserved for subnormal, 31 reserved for NaN

    e5m3_vals = ((e5 << 3) | m3).to(torch.uint8)

    # sumnorm：0 < x < 2^-14
    subnormal_mask = ~normal_mask
    x_subnormal = x_masked[subnormal_mask]
    m_sub = torch.clamp(torch.round(x_subnormal / (2**-14) * 8), 1, 7).to(torch.uint8)  # exponent = 0
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
    is_subnormal = e == 0
    is_normal = (e > 0) & (~is_nan)

    out = torch.zeros_like(e, dtype=torch.float32)

    # subnormal: exponent = -14, no implicit leading 1
    out[is_subnormal] = (m[is_subnormal].float() / 8.0) * (2**-14)

    # normal: exponent = e - 15, implicit leading 1
    mant = 1.0 + m[is_normal].float() / 8.0
    exp = e[is_normal] - 15
    out[is_normal] = torch.ldexp(mant, exp)

    out[is_nan] = float("nan")
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


def ref_fp4_quant(x, global_scale, block_size=16, v=0, max_scale=1.0):
    assert (not isinstance(global_scale, torch.Tensor)) or global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.unsqueeze(dim=-1).to(x.device)
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32) * max_scale
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = torch.clip(scale, 0, FLOAT8_UE5M3_MAX)
    scale = cast_to_ue5m3_ste(scale).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))
    scaled_x = x.to(torch.float32) * output_scale + v
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0)
    return (cast_to_fp4(clipped_x) * get_reciprocal(output_scale)).reshape(m, n), scale


@register_dtype("fp4_v2_with_global_scale")
def fp4_v2_with_global_scale(tensor, bits=4, group_size=16, v=0, tensor_max=None, max_scale=1.0, **kwargs):
    assert group_size == 32 or group_size == 16
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor_max is None:
        tensor_max = tensor.abs().max().to(torch.float32)
    elif tensor_max is not None:
        if not isinstance(tensor_max, torch.Tensor):
            tensor_max = torch.tensor(tensor_max, device=tensor.device, dtype=torch.float32)
        if tensor_max.numel() != 1:
            tensor_max = tensor.abs().max().to(torch.float32)
    global_scale = FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    qdq_res, scale = ref_fp4_quant(tensor, global_scale, group_size, v)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


@register_dtype("fp4_v2")
def fp4_v2(tensor, bits=4, group_size=32, v=0, max_scale=1.0, **kwargs):
    assert group_size == 32 or group_size == 16
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    global_scale = 1.0
    qdq_res, scale = ref_fp4_quant(tensor, global_scale, group_size, v, max_scale)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


if __name__ == "__main__":
    data = torch.tensor([0.0, 0.25, 0.4, 0.75, 1.25, 1.4, 1.75, 2.5, 2.9, 3.5, 5.0, 5.1, 6.0, 6.2, 8.9])
    data1 = cast_to_fp4(data)
    gt = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0])
    assert torch.sum(torch.abs(data1 - gt)) < 1e-6

    data_neg = data * -1
    data2 = cast_to_fp4(data_neg)
    assert torch.sum(torch.abs(data2 - gt * -1)) < 1e-6

    test = torch.tensor(
        [
            0.0,
            1e-38,
            2 ** (-17),
            (2**-14) * 0.875,
            2**-14,
            2**-13,
            2**-6,
            1e-6,
            2.7657e-05,
            0.1,
            1.0,
            3.14,
            1000.0,
            114688,
            1e10,
        ],
        dtype=torch.float32,
    )
    encoded = float_to_e5m3_frexp(test)
    decoded = e5m3_to_float_tensor(encoded)
    decoded_bf16 = decoded.to(torch.bfloat16)
    print(decoded_bf16)

    for i in range(len(test)):
        print(
            f"{test[i].item():.6g} -> {encoded[i].item():3d} -> {decoded[i].item():.6g} "
            f"(error={abs(test[i] - decoded[i]).item():.3g})"
        )
