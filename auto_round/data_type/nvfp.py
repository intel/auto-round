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
        is_zero = x == 0
        denominator = torch.where(is_zero, torch.ones_like(x), x)
        return torch.where(is_zero, torch.zeros_like(x, dtype=x.dtype), 1.0 / denominator)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max if hasattr(torch, "float8_e4m3fn") else 448
FLOAT8_E4M3_MIN = torch.finfo(torch.float8_e4m3fn).min if hasattr(torch, "float8_e4m3fn") else -448


def calculate_gparam(tensor, group_size=16, device="cpu", scale_max=None):
    """Calculate global scaling factor for NVFP quantization."""
    assert group_size == 16, f"Only group_size=16 is supported, got {group_size}"
    scale_max = FLOAT8_E4M3_MAX if scale_max is None else scale_max
    if isinstance(tensor, (float, int)):
        tensor_amax = torch.tensor(tensor, device=device, dtype=torch.float32).abs()
    elif isinstance(tensor, torch.Tensor):
        tensor_amax = tensor.to(torch.float32).abs().max()
    global_scale = scale_max * FLOAT4_E2M1_MAX * get_reciprocal(tensor_amax)
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

    bits = e5m3.to(torch.int32)
    e = (bits >> 3) & 0x1F
    m = (bits & 0x07).to(torch.float32)

    is_nan = (e == 31) & (m == 7)
    # subnormal (e == 0): exponent -14, no implicit leading 1; zero encodes as m == 0
    subnormal = (m / 8.0) * (2**-14)
    # normal: exponent e - 15, implicit leading 1
    normal = torch.ldexp(1.0 + m / 8.0, e - 15)
    out = torch.where(e == 0, subnormal, normal)
    return torch.where(is_nan, torch.full_like(out, float("nan")), out)


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
    else:
        if not isinstance(tensor_max, torch.Tensor):
            tensor_max = torch.tensor(tensor_max, device=tensor.device, dtype=torch.float32)
        else:
            tensor_max = tensor_max.to(device=tensor.device, dtype=torch.float32)
        if tensor_max.numel() != 1:
            tensor_max = tensor_max.abs().max()
    global_scale = FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.reshape(-1).to(tensor.device)
    qdq_res, scale = ref_fp4_quant(tensor, global_scale, group_size, v, max_scale)
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


def calculate_gparam_e5m3(tensor, group_size=16, device="cpu"):
    """Calculate the per-tensor global scale for NVFP4 E5M3 (UE5M3 block scales)."""
    return calculate_gparam(tensor, group_size, device, scale_max=FLOAT8_UE5M3_MAX)


def calculate_gparam_by_dtype(tensor, data_type, group_size=16, device="cpu"):
    if "nvfp4_v2" in str(data_type).lower():
        return calculate_gparam_e5m3(tensor, group_size, device)
    return calculate_gparam(tensor, group_size, device)


def _to_global_scale_tensor(global_scale, device):
    if not isinstance(global_scale, torch.Tensor):
        global_scale = torch.tensor(global_scale, dtype=torch.float32)
    return global_scale.to(device=device, dtype=torch.float32).reshape(-1)[:1]


@register_dtype("nvfp4_v2_gs")
def nvfp4_v2_gs(tensor, bits=4, group_size=16, v=0, global_scale=None, max_scale=1.0, **kwargs):
    """FP4 E2M1 with UE5M3 block scales and a per-tensor FP32 global scale."""
    assert group_size == 16, f"NVFP4_E5M3_GS only supports group_size=16, got {group_size}"
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if global_scale is None:
        global_scale = calculate_gparam_e5m3(tensor, group_size)
    global_scale = _to_global_scale_tensor(global_scale, tensor.device)
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.reshape(-1).to(tensor.device)
    qdq_res, scale = ref_fp4_quant(tensor, global_scale, group_size, v, max_scale)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


@register_dtype("nvfp4_v2_with_static_gs")
def nvfp4_v2_with_static_gs(tensor, bits=4, group_size=16, v=0, tensor_max=None, global_scale=None, **kwargs):
    """Activation QDQ for NVFP4_E5M3_GS; the global scale comes from calibrated ``tensor_max``."""
    if tensor is None or tensor.numel() == 0:
        return tensor, None, None
    assert group_size == 16, f"NVFP4_E5M3_GS only supports group_size=16, got {group_size}"
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if global_scale is None:
        if tensor_max is None:
            tensor_max = tensor.to(torch.float32).abs().max()
        elif isinstance(tensor_max, torch.Tensor):
            tensor_max = tensor_max.to(device=tensor.device, dtype=torch.float32).abs().max()
        global_scale = calculate_gparam_e5m3(tensor_max, group_size, tensor.device)
    global_scale = _to_global_scale_tensor(global_scale, tensor.device)
    qdq_res, scale = ref_fp4_quant(tensor, global_scale, group_size, v)
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


def _nvfp4_e4m3_block_scale(vec_amax, multiplier, global_scale):
    """FP8 E4M3 block scale that ``nv_fp4_rtn`` produces for per-group amax and multiplier."""
    scale = global_scale * ((vec_amax * multiplier) * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.clamp(min=FLOAT8_E4M3_MIN, max=FLOAT8_E4M3_MAX)
    return to_float8_e4m3fn(scale)


def _nvfp4_e4m3_qdq(x, block_scale, global_scale):
    """FP4 QDQ of grouped ``x`` [G, 16] with FP8 E4M3 block scales [G]; mirrors ``ref_nvfp4_quant_inplace``."""
    output_scale = get_reciprocal(block_scale * get_reciprocal(global_scale)).unsqueeze(-1)
    scaled = (x * output_scale).clamp(-6.0, 6.0)
    return cast_to_fp4(scaled) * get_reciprocal(output_scale)


_NVFP4_E4M3_MULTIPLIERS = [scale_value / 100.0 for scale_value in range(50, 152) if scale_value != 100]


def _search_nvfp4_e4m3_multiplier(x, weight):
    """Search per-group amax multipliers in [0.5, 1.51] for the NVFP4 E4M3 block scale.

    Equivalent to re-running ``nv_fp4_rtn`` on the full tensor for every candidate with a strict ``<`` update in
    ascending order, but only groups whose FP8 block scale changed since the previous candidate are re-quantized.
    An unchanged scale yields the same loss as the previous candidate, which is never strictly better than the
    current best, so skipping it does not change the result.

    Args:
        x: float32 tensor of shape [G, 16].
        weight: float32 importance weights broadcastable to ``x``.

    Returns:
        float32 tensor of shape [G] holding the best multiplier per group.
    """
    amax = x.abs().amax(dim=-1)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(x.abs().max())

    base_scale = _nvfp4_e4m3_block_scale(amax, torch.ones((), device=x.device), global_scale)
    diff = _nvfp4_e4m3_qdq(x, base_scale, global_scale) - x
    best_loss = (diff * diff * weight).sum(dim=-1)
    best_mult = torch.ones_like(best_loss)

    multipliers = torch.tensor(_NVFP4_E4M3_MULTIPLIERS, dtype=torch.float32, device=x.device)
    prev_scale = base_scale
    for mult in multipliers.unbind(0):
        scale = _nvfp4_e4m3_block_scale(amax, mult, global_scale)
        idx = torch.nonzero(scale != prev_scale).squeeze(-1)
        prev_scale = scale
        if idx.numel() == 0:
            continue
        xs = x.index_select(0, idx)
        d = _nvfp4_e4m3_qdq(xs, scale.index_select(0, idx), global_scale) - xs
        loss = (d * d * weight.index_select(0, idx)).sum(dim=-1)
        old_loss = best_loss.index_select(0, idx)
        better = loss < old_loss
        best_loss.index_copy_(0, idx, torch.where(better, loss, old_loss))
        best_mult.index_copy_(0, idx, torch.where(better, mult, best_mult.index_select(0, idx)))
    return best_mult


def search_nvfp4_scale(tensor, bits=4, qw=None, quant_func=None, group_size=16):
    """Search the per-group block-scale multiplier in [0.5, 1.51] for NVFP4 (E4M3 block scale).

    With ``quant_func=None`` the changed-only search is used. Passing ``quant_func`` selects the exhaustive
    reference implementation, which re-quantizes the full tensor for every candidate.
    """
    if quant_func is None:
        if qw is None:
            raise TypeError("search_nvfp4_scale requires `qw` (importance weights)")
        x = tensor.float().reshape(-1, group_size)
        weight = torch.as_tensor(qw, dtype=torch.float32, device=x.device)
        weight = weight.reshape_as(x) if weight.numel() == x.numel() else weight.expand_as(x)
        return _search_nvfp4_e4m3_multiplier(x, weight).unsqueeze(-1)

    tensor_fp32 = tensor.float()
    baseline_func = quant_func
    candidate_func = quant_func

    qdq_t, scale, _ = baseline_func(
        tensor_fp32,
        bits=bits,
        group_size=group_size,
        v=0,
        max_scale=1.0,
    )

    # loss buffer
    diff = torch.empty_like(tensor_fp32)
    loss = torch.empty_like(tensor_fp32[..., 0])

    diff.copy_(qdq_t)
    diff.sub_(tensor_fp32)
    diff.mul_(diff)
    diff.mul_(qw)
    loss.copy_(diff.sum(dim=-1))

    best_loss = loss.clone()

    # scale buffer reuse
    best_scale = torch.ones_like(scale)

    # inplace modify
    test_scale = torch.empty_like(scale)

    for scale_value in range(50, 152):
        tmp_scale = scale_value / 100.0

        if tmp_scale == 1.0:
            continue

        test_scale.fill_(tmp_scale)
        candidate_scale = test_scale.squeeze(-1) if quant_func is nvfp4_v2 else test_scale

        tmp_qdq, _, _ = candidate_func(
            tensor_fp32,
            bits=bits,
            group_size=group_size,
            v=0,
            max_scale=candidate_scale,
        )

        diff.copy_(tmp_qdq)
        diff.sub_(tensor_fp32)
        diff.mul_(diff)
        diff.mul_(qw)

        loss.copy_(diff.sum(dim=-1))

        mask = loss < best_loss

        best_loss = torch.where(mask, loss, best_loss)
        best_scale = torch.where(mask.unsqueeze(-1), test_scale, best_scale)

    return best_scale


def _nvfp4_e5m3_block_scale(vec_amax, multiplier, global_scale=1.0):
    """UE5M3 block scale that ``ref_fp4_quant`` produces for per-group amax and multiplier."""
    scale = global_scale * ((vec_amax * multiplier) * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = torch.clip(scale, 0, FLOAT8_UE5M3_MAX)
    return cast_to_ue5m3_ste(scale).to(torch.float32)


def _nvfp4_e5m3_qdq(x, block_scale, global_scale=1.0):
    """FP4 QDQ of grouped ``x`` (shape [G, 16]) with per-group UE5M3 block scales (shape [G])."""
    dequant_scale = block_scale.unsqueeze(-1) * get_reciprocal(global_scale)
    scaled_x = torch.where(dequant_scale == 0, torch.zeros_like(x), x / dequant_scale)
    clipped = torch.clamp(scaled_x, -6.0, 6.0)
    return cast_to_fp4(clipped) * dequant_scale


_NVFP4_SEARCH_CHUNK_MAX_ELEMS = 2**24  # candidates x elements per compiled call; caps eager-fallback temporaries


def _nvfp4_e5m3_search_chunk(x, amax, weight, global_scale, mults, best_loss, best_mult):
    """Evaluate a chunk of amax multipliers and keep the first best one per group if it strictly improves."""
    block_scale = _nvfp4_e5m3_block_scale(amax, mults.unsqueeze(-1), global_scale)
    diff = _nvfp4_e5m3_qdq(x, block_scale, global_scale) - x
    loss = (diff * diff * weight).sum(dim=-1)
    chunk_loss, idx = loss.min(dim=0)
    chunk_mult = mults[idx]
    better = chunk_loss < best_loss
    return torch.where(better, chunk_loss, best_loss), torch.where(better, chunk_mult, best_mult)


_NVFP4_E5M3_SEARCH_CHUNK_CACHE = {}


def _get_nvfp4_e5m3_search_chunk(device):
    key = str(device)
    if key not in _NVFP4_E5M3_SEARCH_CHUNK_CACHE:
        from auto_round.utils.device import compile_func

        _NVFP4_E5M3_SEARCH_CHUNK_CACHE[key] = compile_func(_nvfp4_e5m3_search_chunk, device)
    return _NVFP4_E5M3_SEARCH_CHUNK_CACHE[key]


def _search_nvfp4_e5m3_multiplier(tensor, qw=None, global_scale=None, enable_torch_compile=False):
    """Search per-group amax multipliers in [0.5, 1.51] for the NVFP4 E5M3 block scale.

    With ``enable_torch_compile`` the candidates run as fixed-shape compiled chunks over the full tensor
    (chunk size bounded by ``_NVFP4_SEARCH_CHUNK_MAX_ELEMS``; the last chunk repeats the final candidate,
    which never wins under the strict ``<`` update). Eagerly, only groups whose UE5M3 scale changed from
    the previous candidate are re-evaluated, since an unchanged scale cannot strictly improve the loss.
    """
    x = tensor.float()
    gs = 1.0 if global_scale is None else _to_global_scale_tensor(global_scale, x.device)
    amax = x.abs().amax(dim=-1)
    weight = (
        torch.ones_like(x) if qw is None else torch.as_tensor(qw, dtype=torch.float32, device=x.device).expand_as(x)
    )

    base_scale = _nvfp4_e5m3_block_scale(amax, torch.tensor(1.0, device=x.device), gs)
    diff = _nvfp4_e5m3_qdq(x, base_scale, gs) - x
    best_loss = (diff * diff * weight).sum(dim=-1)
    best_mult = torch.ones_like(best_loss)

    multipliers = torch.tensor(
        [scale_value / 100.0 for scale_value in range(50, 152) if scale_value != 100],
        dtype=torch.float32,
        device=x.device,
    )

    if not enable_torch_compile:
        prev_scale = base_scale
        for mult in multipliers.unbind(0):
            scale = _nvfp4_e5m3_block_scale(amax, mult, gs)
            idx = torch.nonzero(scale != prev_scale).squeeze(-1)
            prev_scale = scale
            if idx.numel() == 0:
                continue
            xs = x.index_select(0, idx)
            d = _nvfp4_e5m3_qdq(xs, scale.index_select(0, idx), gs) - xs
            loss = (d * d * weight.index_select(0, idx)).sum(dim=-1)
            old_loss = best_loss.index_select(0, idx)
            better = loss < old_loss
            best_loss.index_copy_(0, idx, torch.where(better, loss, old_loss))
            best_mult.index_copy_(0, idx, torch.where(better, mult, best_mult.index_select(0, idx)))
        return best_mult

    chunk = min(max(1, _NVFP4_SEARCH_CHUNK_MAX_ELEMS // x.numel()), multipliers.numel())
    pad = -multipliers.numel() % chunk
    multipliers = torch.cat([multipliers, multipliers[-1:].expand(pad)]).view(-1, chunk)

    step = _get_nvfp4_e5m3_search_chunk(x.device)
    for mults in multipliers.unbind(0):
        best_loss, best_mult = step(x, amax, weight, gs, mults, best_loss, best_mult)

    return best_mult


def search_nvfp4_v2_scale(tensor, bits=4, qw=None, enable_torch_compile=False):
    """Search per-group scales for the legacy-registry NVFP4 E5M3 quantizer."""
    return _search_nvfp4_e5m3_multiplier(tensor, qw, enable_torch_compile=enable_torch_compile)


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

    init_scale = search_nvfp4_scale(tensor, 4, qw)
    tensor = revert_tensor_by_pad(tensor, orig_shape, pad_len)
    return nv_fp4_rtn(tensor, bits, group_size, v, global_scale, max_scale, init_scale=init_scale)


@register_dtype("opt_rtn_nvfp4_v2_gs")
def opt_rtn_nvfp4_v2_gs(
    tensor,
    bits=4,
    group_size=16,
    v=0,
    global_scale=None,
    max_scale=1.0,
    imatrix=1.0,
    enable_torch_compile=False,
    **kwargs,
):
    grouped, _, _ = reshape_pad_tensor_by_group_size(tensor, group_size)
    if not isinstance(imatrix, torch.Tensor):
        qw = 1.0
    else:
        imatrix = imatrix.reshape(1, -1)
        imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        imatrix = imatrix.expand(grouped.numel() // imatrix.numel(), -1)
        imatrix = imatrix.reshape(grouped.shape)
        qw = _imatrix_handle_zero(imatrix, grouped, bits, group_size)
    if global_scale is None:
        global_scale = calculate_gparam_e5m3(grouped, group_size)

    init_scale = _search_nvfp4_e5m3_multiplier(grouped, qw, global_scale, enable_torch_compile)
    if isinstance(max_scale, torch.Tensor):
        init_scale = init_scale * max_scale.reshape(-1).to(init_scale.device)
    else:
        init_scale = init_scale * max_scale
    return nvfp4_v2_gs(tensor, bits, group_size, v, global_scale=global_scale, max_scale=init_scale)


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
