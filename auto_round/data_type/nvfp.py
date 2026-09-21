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

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from auto_round.data_type.base import register_dtype, register_quantizer
from auto_round.data_type.fp8 import float8_e4m3fn_ste
from auto_round.data_type.gguf import _imatrix_handle_zero
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


def search_nvfp4_scale(tensor, bits=4, qw=None, quant_func=None, group_size=16):
    tensor_fp32 = tensor.float()
    baseline_func = nv_fp4 if quant_func is None else quant_func
    candidate_func = nv_fp4_rtn if quant_func is None else quant_func

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

        best_loss[mask] = loss[mask]
        best_scale[mask] = test_scale[mask]

    return best_scale


def search_nvfp4_v2_scale(tensor, bits=4, qw=None):
    return search_nvfp4_scale(tensor, bits, qw, quant_func=nvfp4_v2, group_size=tensor.shape[-1]).squeeze(-1)


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


@dataclass(frozen=True)
class _NVFPState:
    """NVFP tuning values plus optional global and optimized scales."""

    tunables: Mapping[str, torch.Tensor]
    global_scale: torch.Tensor | None
    optimized_init: torch.Tensor | None


class _NVFPWeightQuantizer:
    """Own NVFP4 weight QDQ, including its global-scale preparation."""

    def __init__(self, spec, family="plain"):
        self.spec = spec
        self.family = family

    @classmethod
    def from_spec(cls, spec, canonical=None):
        """Create the NVFP4 weight quantizer for a resolved layer."""
        return cls(spec)

    @staticmethod
    def create_activation(spec):
        """Create dynamic NVFP4 activation quantization."""
        return _NVFPActivationQuantizer(spec, "nv_fp4")

    def create_state(self, weight, *, imatrix=None, mode, tune_rounding, tune_minmax):
        self.family = "optimized" if mode == "optimized_rtn" else "plain"
        grouped, _, _ = reshape_pad_tensor_by_group_size(weight, self.spec.group_size)
        tunables = {}
        if self.family == "plain" and tune_rounding:
            tunables["value"] = torch.nn.Parameter(torch.zeros_like(grouped, dtype=torch.float32))
        if self.family != "optimized" and tune_minmax:
            tunables["max_scale"] = torch.nn.Parameter(
                torch.ones(grouped.shape[:-1], device=weight.device, dtype=torch.float32)
            )

        global_scale = self.spec.global_scale
        if global_scale is None:
            global_scale = calculate_gparam(weight, self.spec.group_size, weight.device)
        else:
            global_scale = global_scale.to(weight.device)
        optimized_init = None
        if self.family == "optimized":
            if isinstance(imatrix, torch.Tensor):
                imatrix = imatrix.reshape(1, -1)
                imatrix = reshape_pad_tensor_by_group_size(imatrix, self.spec.group_size, val=1e-5)[0].view(1, -1)
                imatrix = imatrix.expand(grouped.numel() // imatrix.numel(), -1).reshape(grouped.shape)
                qw = _imatrix_handle_zero(imatrix, grouped, self.spec.bits, self.spec.group_size)
            else:
                qw = 1.0
            optimized_init = search_nvfp4_scale(grouped, self.spec.bits, qw)
        return _NVFPState(tunables, global_scale, optimized_init)

    def qdq(self, weight, state, *, tunables, materialize=False):
        kwargs = {
            "bits": self.spec.bits,
            "group_size": self.spec.group_size,
            "v": tunables.get("value", 0),
            "max_scale": tunables.get("max_scale", 1.0),
        }
        kwargs["global_scale"] = state.global_scale
        if state.optimized_init is not None:
            kwargs["init_scale"] = state.optimized_init
        quantized, scale, zero_point = nv_fp4(weight, **kwargs)
        from auto_round.data_type.base import WeightQuantizationResult

        return WeightQuantizationResult(
            quantized,
            scale if materialize else None,
            zero_point if materialize else None,
            state.global_scale if materialize else None,
        )

    @staticmethod
    def apply_result(module, result):
        if result.scale is None:
            raise ValueError("NVFP weight result was not materialized")
        module.weight.data.copy_(result.weight)
        rows = result.logical_rows or result.weight.shape[0]
        module.scale = result.scale.reshape(rows, -1).cpu()
        module.zp = None
        if result.metadata is not None:
            module.weight_global_scale = result.metadata.cpu()


class _NVFPV2WeightQuantizer:
    """Own the independent NVFP4-v2 weight QDQ implementation."""

    def __init__(self, spec):
        self.spec = spec

    @classmethod
    def from_spec(cls, spec, canonical=None):
        """Create the NVFP4-v2 weight quantizer for a resolved layer."""
        return cls(spec)

    @staticmethod
    def create_activation(spec):
        """Create dynamic NVFP4-v2 activation quantization."""
        return _NVFPActivationQuantizer(spec, "nvfp4_v2")

    def create_state(self, weight, *, imatrix=None, mode, tune_rounding, tune_minmax):
        grouped, _, _ = reshape_pad_tensor_by_group_size(weight, self.spec.group_size)
        tunables = {}
        optimized_rtn = mode == "optimized_rtn"
        if tune_rounding and not optimized_rtn:
            tunables["value"] = torch.nn.Parameter(torch.zeros_like(grouped, dtype=torch.float32))
        if tune_minmax and not optimized_rtn:
            tunables["max_scale"] = torch.nn.Parameter(
                torch.ones(grouped.shape[:-1], device=weight.device, dtype=torch.float32)
            )
        optimized_scale = None
        if optimized_rtn:
            if isinstance(imatrix, torch.Tensor):
                imatrix = imatrix.reshape(1, -1)
                imatrix = reshape_pad_tensor_by_group_size(imatrix, self.spec.group_size, val=1e-5)[0].view(1, -1)
                imatrix = imatrix.expand(grouped.numel() // imatrix.numel(), -1).reshape(grouped.shape)
                qw = _imatrix_handle_zero(imatrix, grouped, self.spec.bits, self.spec.group_size)
            else:
                qw = 1.0
            optimized_scale = search_nvfp4_v2_scale(grouped, self.spec.bits, qw)
        return _NVFPState(tunables, None, optimized_scale)

    def qdq(self, weight, state, *, tunables, materialize=False):
        quantized, scale, zero_point = nvfp4_v2(
            weight,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            v=tunables.get("value", 0),
            max_scale=tunables.get("max_scale", state.optimized_init if state.optimized_init is not None else 1.0),
        )
        from auto_round.data_type.base import WeightQuantizationResult

        return WeightQuantizationResult(quantized, scale if materialize else None, zero_point if materialize else None)

    apply_result = staticmethod(_NVFPWeightQuantizer.apply_result)


class _NVFPActivationQuantizer:
    """Quantize dynamic or calibrated NVFP activations for one NVFP format."""

    def __init__(self, spec, data_type):
        is_static = data_type in ("nv_fp4_with_static_gs", "nvfp4_v2_with_global_scale")
        if not is_static and not spec.dynamic:
            raise ValueError(f"NVFP datatype {data_type!r} supports only dynamic activation quantization")
        self.spec = spec
        self.data_type = data_type
        self.requires_calibration = is_static

    def observe(self, activation, current):
        if not self.requires_calibration:
            raise RuntimeError("Dynamic NVFP activation quantization does not require calibration")
        observed = activation.detach().float().abs().max().unsqueeze(0)
        return observed if current is None else torch.maximum(current.to(observed.device), observed)

    def qdq_with_scale(self, activation, *, observed_max=None, min_scale=1.0, max_scale=1.0, global_scale=None):
        if self.requires_calibration and observed_max is None:
            raise ValueError(f"{self.data_type} activation requires observed_max")
        primitive = {
            "nv_fp4": nv_fp4,
            "nv_fp4_with_static_gs": nv_fp4_with_static_gs,
            "nvfp4_v2": nvfp4_v2,
            "nvfp4_v2_with_global_scale": nvfp4_v2_with_global_scale,
        }[self.data_type]
        kwargs = {"bits": self.spec.bits, "group_size": self.spec.group_size, "max_scale": max_scale}
        if self.requires_calibration:
            kwargs["tensor_max"] = observed_max
        if global_scale is not None:
            kwargs["global_scale"] = global_scale
        return primitive(activation, **kwargs)

    def qdq(self, activation, *, observed_max=None, min_scale=1.0, max_scale=1.0, global_scale=None):
        quantized, _, _ = self.qdq_with_scale(
            activation,
            observed_max=observed_max,
            min_scale=min_scale,
            max_scale=max_scale,
            global_scale=global_scale,
        )
        return quantized


def _prepare_nvfp_block(block, layer_runtimes):
    """Materialize shared NVFP scales once after a block's observations."""
    from auto_round.data_type.utils import update_block_global_scale_if_needed

    runtime = next(iter(layer_runtimes.values()), None)
    if runtime is None:
        return
    update_block_global_scale_if_needed(block, runtime["data_type"], runtime["group_size"])


_NVFPWeightQuantizer.prepare_block = staticmethod(_prepare_nvfp_block)


class _NVFPStaticActivation:
    """Datatype entry that provides static NVFP activation quantization only."""

    @staticmethod
    def create_activation(spec):
        return _NVFPActivationQuantizer(spec, "nv_fp4_with_static_gs")


class _NVFPV2StaticActivation:
    """Datatype entry that provides global-scale NVFP-v2 activation only."""

    @staticmethod
    def create_activation(spec):
        return _NVFPActivationQuantizer(spec, "nvfp4_v2_with_global_scale")


register_quantizer(
    "nv_fp4",
    aliases=(
        "nv_fp",
        "nv_fp_sym",
        "nv_fp4_sym",
        "rtn_nv_fp",
        "rtn_nv_fp4",
        "rtn_nv_fp_sym",
        "rtn_nv_fp4_sym",
        "opt_rtn_nv_fp",
        "opt_rtn_nv_fp4",
        "opt_rtn_nv_fp_sym",
        "opt_rtn_nv_fp4_sym",
    ),
)(_NVFPWeightQuantizer)
register_quantizer("nvfp4_v2")(_NVFPV2WeightQuantizer)
register_quantizer("nv_fp4_with_static_gs", aliases=("rtn_nv_fp4_with_static_gs",))(_NVFPStaticActivation)
register_quantizer("nvfp4_v2_with_global_scale")(_NVFPV2StaticActivation)


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
