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

import os

import torch

from auto_round.data_type.fp8 import float8_e4m3fn_ste
from auto_round.data_type.gguf import _imatrix_handle_zero
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
    """Calculate global scaling factor for NVFP quantization."""
    assert group_size == 16, f"Only group_size=16 is supported, got {group_size}"
    if isinstance(tensor, (float, int)):
        tensor_amax = torch.tensor(tensor, device=device, dtype=torch.float32).abs()
    elif isinstance(tensor, torch.Tensor):
        tensor_amax = tensor.to(torch.float32).abs().max()
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
def nv_fp4(tensor, bits=4, group_size=16, v=0, global_scale=None, max_scale=1.0, init_scale=1.0, **kwargs):
    orig_dtype = tensor.dtype
    init_scale = 1.0 if init_scale is None else init_scale
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if global_scale is None:
        tensor_max = tensor.to(torch.float32).abs().max()
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    global_scale = global_scale.to(device=tensor.device, dtype=torch.float32)
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.view(-1).to(tensor.device)
    if isinstance(init_scale, torch.Tensor):
        init_scale = init_scale.view(-1).to(tensor.device)
    qdq_res, scale = ref_nvfp4_quant(tensor, global_scale, group_size, v, scale_coeff=max_scale * init_scale)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


@register_dtype("nv_fp4_with_static_gs")
def nv_fp4_with_static_gs(tensor, bits=4, group_size=16, v=0, tensor_max=None, global_scale=None, **kwargs):
    if tensor is None or tensor.numel() == 0:
        return tensor, None, None
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if global_scale is None:
        if tensor_max is None:
            tensor_max = tensor.to(torch.float32).abs().max()
        elif not isinstance(tensor_max, torch.Tensor):
            tensor_max = torch.tensor(tensor_max, device=tensor.device, dtype=torch.float32)
        else:
            tensor_max = tensor_max.to(device=tensor.device, dtype=torch.float32)
            if tensor_max.numel() != 1:
                tensor_max = tensor_max.abs().max()
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    elif not isinstance(global_scale, torch.Tensor):
        global_scale = torch.tensor(global_scale, device=tensor.device, dtype=torch.float32)
    global_scale = global_scale.to(device=tensor.device, dtype=torch.float32)
    qdq_res, scale = ref_nvfp4_quant(tensor, global_scale, group_size, v)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


FLOAT8_UE5M3_MAX = 114688


def float_to_e5m3_frexp(x: torch.Tensor) -> torch.Tensor:
    input_fp32 = x.to(torch.float32)
    finite = torch.nan_to_num(input_fp32, nan=0.0, posinf=FLOAT8_UE5M3_MAX, neginf=0.0).clamp_(0.0, FLOAT8_UE5M3_MAX)

    mantissa, exponent = torch.frexp(finite.clamp_min(2**-14))
    m3 = torch.round((mantissa - 0.5) * 16).to(torch.int32)
    carry = m3 == 8
    m3 = torch.where(carry, 0, m3)
    e5 = exponent + 14 + carry.to(exponent.dtype)
    normal = (e5 << 3) | m3

    # RNE may underflow to zero or carry from the largest subnormal to 0x08.
    subnormal = torch.round(finite * 2**17).to(torch.int32)
    encoded = torch.where(finite < 2**-14, subnormal, normal).clamp_(0, 0xFE).to(torch.uint8)
    return torch.where(torch.isnan(input_fp32), 0xFF, encoded)


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
    dequant_scale = scale * get_reciprocal(global_scale)
    scaled_x = torch.where(
        dequant_scale == 0,
        torch.zeros_like(x, dtype=torch.float32),
        x.to(torch.float32) / dequant_scale,
    )
    scaled_x = scaled_x + v
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0)
    return (cast_to_fp4(clipped_x) * dequant_scale).reshape(m, n), scale


@register_dtype("nvfp4_v2_with_global_scale")
def nvfp4_v2_with_global_scale(tensor, bits=4, group_size=16, v=0, tensor_max=None, max_scale=1.0, **kwargs):
    assert group_size == 32 or group_size == 16
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor_max is None:
        tensor_max = tensor.to(torch.float32).abs().max()
    elif tensor_max is not None:
        if not isinstance(tensor_max, torch.Tensor):
            tensor_max = torch.tensor(tensor_max, device=tensor.device, dtype=torch.float32)
        if tensor_max.numel() != 1:
            tensor_max = tensor.to(torch.float32).abs().max()
    global_scale = FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    qdq_res, scale = ref_fp4_quant(tensor, global_scale, group_size, v)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


@register_dtype("nvfp4_v2")
def nvfp4_v2(tensor, bits=4, group_size=32, v=0, max_scale=1.0, **kwargs):
    assert group_size == 32 or group_size == 16
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    global_scale = 1.0
    qdq_res, scale = ref_fp4_quant(tensor, global_scale, group_size, v, max_scale)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


@register_dtype("nv_fp4_rtn")
def nv_fp4_rtn(tensor, bits=4, group_size=16, v=0, global_scale=None, max_scale=1.0, init_scale=1.0, **kwargs):
    orig_dtype = tensor.dtype
    init_scale = 1.0 if init_scale is None else init_scale
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if global_scale is None:
        tensor_max = tensor.abs().max().to(torch.float32)
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    global_scale = global_scale.to(device=tensor.device, dtype=torch.float32)
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.view(-1).to(tensor.device)
    if isinstance(init_scale, torch.Tensor):
        init_scale = init_scale.view(-1).to(tensor.device)
    qdq_res, scale = ref_nvfp4_quant_inplace(tensor, global_scale, group_size, v, scale_coeff=max_scale * init_scale)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


@torch._dynamo.disable()
def to_float8_e4m3fn(tensor):
    return tensor.to(torch.float8_e4m3fn).to(torch.float32)


def ref_nvfp4_quant_inplace(
    x,
    global_scale,
    block_size=16,
    v=0,
    scale_coeff=1.0,
    out=None,
):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2

    m, n = x.shape

    if isinstance(scale_coeff, torch.Tensor):
        scale_coeff = scale_coeff.view(-1, 1).to(x.device)

    # vec_max: [m, 1]
    vec_max = torch.amax(torch.abs(x), dim=-1, keepdim=True).float()

    if isinstance(scale_coeff, torch.Tensor):
        vec_max.mul_(scale_coeff)
    else:
        vec_max.mul_(scale_coeff)

    # scale: [m,1]
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))

    scale.clamp_(
        min=FLOAT8_E4M3_MIN,
        max=FLOAT8_E4M3_MAX,
    )

    scale = to_float8_e4m3fn(scale)

    # output_scale reuse scale buffer
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    # allocate once
    if out is None:
        out = torch.empty_like(x, dtype=torch.float32)

    # out = x * output_scale
    out.copy_(x)
    out.mul_(output_scale)

    if v != 0:
        out.add_(v)

    # clamp inplace
    out.clamp_(-6.0, 6.0)

    # fp4 cast
    out = cast_to_fp4(out) * get_reciprocal(output_scale)

    return out.reshape(m, n), scale


def _resolve_neighbor_search_steps() -> int:
    """Resolve neighboring-search radius (steps around baseline) from env.

    A value of N searches up to N previous and N next representable scales
    around the baseline scale for each group.
    """
    raw_value = os.getenv("AR_NVFP4_NEIGHBOR_SEARCH_STEPS", "8")
    try:
        steps = int(raw_value)
    except ValueError:
        logger.warning_once(
            "Invalid AR_NVFP4_NEIGHBOR_SEARCH_STEPS=%s; falling back to 8",
            raw_value,
        )
        return 8
    if steps < 1:
        logger.warning_once(
            "AR_NVFP4_NEIGHBOR_SEARCH_STEPS=%s is < 1; clamping to 1",
            raw_value,
        )
        return 1
    return steps


def _rowwise_weighted_mse(qdq_t: torch.Tensor, tensor_fp32: torch.Tensor, qw: torch.Tensor) -> torch.Tensor:
    diff = qdq_t - tensor_fp32
    return (diff * diff * qw).sum(dim=-1)


def _enumerate_neighbor_scale_coeffs(base_scale: torch.Tensor, *, signed: bool, steps: int) -> list[torch.Tensor]:
    """Enumerate scale coefficients for +/- discrete neighbors around baseline."""
    if steps < 1:
        return []

    prev_cursor = base_scale
    next_cursor = base_scale
    coeffs: list[torch.Tensor] = []
    for _ in range(steps):
        prev_cursor, _ = _neighboring_discrete_scales(prev_cursor, signed=signed)
        _, next_cursor = _neighboring_discrete_scales(next_cursor, signed=signed)
        coeffs.append(_scale_coeffs_from_neighbor_scales(prev_cursor, base_scale))
        coeffs.append(_scale_coeffs_from_neighbor_scales(next_cursor, base_scale))
    return coeffs


def _neighboring_discrete_scales(scale: torch.Tensor, *, signed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the previous and next representable scale values.

    For signed E4M3 (standard NVFP4) we step one raw float8-e4m3fn code in each
    direction. For unsigned E5M3 (NVFP4_E5M3) we step one encoded UE5M3 value in
    each direction while clamping to the valid finite range.
    """
    scale_fp32 = scale.detach().to(torch.float32).contiguous()
    if signed:
        encoded = scale_fp32.to(torch.float8_e4m3fn).view(torch.uint8)
        # NVFP4 scales are non-negative; clamp to the largest finite positive
        # E4M3 code to avoid stepping into NaN/Inf encodings.
        prev_encoded = (encoded.to(torch.int16) - 1).clamp(0, 0x7E).to(torch.uint8)
        next_encoded = (encoded.to(torch.int16) + 1).clamp(0, 0x7E).to(torch.uint8)
        prev_scale = prev_encoded.view(torch.float8_e4m3fn).to(torch.float32)
        next_scale = next_encoded.view(torch.float8_e4m3fn).to(torch.float32)
        return prev_scale.reshape_as(scale_fp32), next_scale.reshape_as(scale_fp32)

    encoded = float_to_e5m3_frexp(scale_fp32)
    prev_encoded = torch.where(encoded > 0, encoded - 1, encoded)
    next_encoded = torch.where(encoded < 0xFE, encoded + 1, encoded)
    return e5m3_to_float_tensor(prev_encoded).reshape_as(scale_fp32), e5m3_to_float_tensor(next_encoded).reshape_as(
        scale_fp32
    )


def _scale_coeffs_from_neighbor_scales(target_scale: torch.Tensor, base_scale: torch.Tensor) -> torch.Tensor:
    """Convert neighboring discrete scales back into multiplicative scale coefficients."""
    target_scale = target_scale.to(torch.float32)
    base_scale = base_scale.to(torch.float32)
    return torch.where(base_scale != 0, target_scale * get_reciprocal(base_scale), torch.ones_like(target_scale))


def search_nvfp4_scale(tensor, bits=4, qw=None, global_scale=None):
    tensor_fp32 = tensor.float()

    qdq_t, scale, _ = nv_fp4(
        tensor_fp32,
        bits=bits,
        group_size=16,
        v=0,
        global_scale=global_scale,
        max_scale=1.0,
    )

    best_loss = _rowwise_weighted_mse(qdq_t, tensor_fp32, qw)
    base_scale = scale.detach().to(torch.float32)
    best_scale = torch.ones_like(scale)
    max_steps = _resolve_neighbor_search_steps()
    candidate_coeffs = _enumerate_neighbor_scale_coeffs(base_scale, signed=True, steps=max_steps)

    for scale_coeff in candidate_coeffs:
        valid = torch.ones_like(scale_coeff, dtype=torch.bool).view(-1)
        if not torch.any(valid):
            continue

        bounded_coeff = torch.where(valid.view(-1, 1), scale_coeff, torch.ones_like(scale_coeff))
        tmp_qdq, _, _ = nv_fp4_rtn(
            tensor_fp32,
            bits=bits,
            group_size=16,
            v=0,
            global_scale=global_scale,
            max_scale=bounded_coeff,
        )
        loss = _rowwise_weighted_mse(tmp_qdq, tensor_fp32, qw)
        mask = valid & (loss < best_loss)
        if torch.any(mask):
            best_loss[mask] = loss[mask]
            best_scale[mask] = bounded_coeff[mask]

    return best_scale


def search_nvfp4_v2_scale(tensor, bits=4, group_size=16, qw=None):
    tensor_fp32 = tensor.float()

    qdq_t, scale, _ = nvfp4_v2(
        tensor_fp32,
        bits=bits,
        group_size=group_size,
        v=0,
        max_scale=1.0,
    )

    best_loss = _rowwise_weighted_mse(qdq_t, tensor_fp32, qw)
    base_scale = scale.detach().to(torch.float32)
    best_scale = torch.ones_like(scale)
    max_steps = _resolve_neighbor_search_steps()
    candidate_coeffs = _enumerate_neighbor_scale_coeffs(base_scale, signed=False, steps=max_steps)

    for scale_coeff in candidate_coeffs:
        valid = torch.ones_like(scale_coeff, dtype=torch.bool).view(-1)
        if not torch.any(valid):
            continue

        bounded_coeff = torch.where(valid.view(-1, 1), scale_coeff, torch.ones_like(scale_coeff))
        tmp_qdq, _, _ = nvfp4_v2(
            tensor_fp32,
            bits=bits,
            group_size=group_size,
            v=0,
            max_scale=bounded_coeff.view(-1),
        )
        loss = _rowwise_weighted_mse(tmp_qdq, tensor_fp32, qw)
        mask = valid & (loss < best_loss)
        if torch.any(mask):
            best_loss[mask] = loss[mask]
            best_scale[mask] = bounded_coeff[mask]

    return best_scale


@register_dtype("opt_rtn_nvfp4_v2")
def opt_rtn_nvfp4_v2(
    tensor,
    bits=4,
    group_size=16,
    v=0,
    max_scale=1.0,
    imatrix=1.0,
    **kwargs,
):
    assert group_size == 32 or group_size == 16
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if not isinstance(imatrix, torch.Tensor):
        qw = 1.0
    else:
        imatrix = imatrix.reshape(1, -1)
        imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        imatrix = imatrix.expand(tensor.numel() // imatrix.numel(), -1)
        imatrix = imatrix.reshape(tensor.shape)
        imatrix = _imatrix_handle_zero(imatrix, tensor, bits, group_size)
        qw = imatrix

    init_scale = search_nvfp4_v2_scale(tensor, bits, group_size, qw)
    tensor = revert_tensor_by_pad(tensor, orig_shape, pad_len)
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.view(-1).to(tensor.device)
    return nvfp4_v2(tensor, bits, group_size, v, max_scale=max_scale * init_scale.view(-1))


@register_dtype("opt_rtn_nv_fp4")
def opt_rtn_fast_nvfp4(
    tensor,
    bits=4,
    group_size=16,
    v=0,
    global_scale=None,
    max_scale=1.0,
    imatrix=1.0,
    **kwargs,
):
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if not isinstance(imatrix, torch.Tensor):
        qw = 1.0
    else:
        imatrix = imatrix.reshape(1, -1)
        imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        imatrix = imatrix.expand(tensor.numel() // imatrix.numel(), -1)
        imatrix = imatrix.reshape(tensor.shape)
        imatrix = _imatrix_handle_zero(imatrix, tensor, bits, group_size)
        qw = imatrix

    init_scale = search_nvfp4_scale(
        tensor,
        4,
        qw,
        global_scale=global_scale,
    )
    tensor = revert_tensor_by_pad(tensor, orig_shape, pad_len)
    return nv_fp4_rtn(tensor, bits, group_size, v, global_scale, max_scale, init_scale=init_scale)


@register_dtype("rtn_nv_fp4_with_static_gs")
def rtn_nv_fp4_with_static_gs(tensor, bits=4, group_size=16, v=0, tensor_max=None, **kwargs):
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
    qdq_res, scale = ref_nvfp4_quant_inplace(tensor, global_scale, group_size, v)
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
