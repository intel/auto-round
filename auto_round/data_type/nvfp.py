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

"""NVIDIA FP4 (NVFP4) and FP4-v2 quantization for auto-round.

This module implements NVFP4 weight quantization as used by NVIDIA hardware
(e.g., Blackwell GPUs) and exposed through vLLM.  Two flavours are provided:

- **nv_fp4** — standard NVFP4 with a dynamically computed global scale.
- **nv_fp4_with_static_gs** — NVFP4 where the global scale is derived from a
  caller-supplied ``tensor_max`` value.
- **fp4_v2 / fp4_v2_with_global_scale** — variant using the UE5M3 scale
  format instead of E4M3.

Helper functions for FP4 casting, UE5M3 encoding/decoding, and global-scale
computation are also provided here.
"""

import torch

from auto_round.data_type.fp8 import float8_e4m3fn_ste
from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger


# taken from
# https://github.com/vllm-project/vllm/blob/ebb554cdb7cd9cc54b2feec20c45ab9cd9067d52/tests/kernels/test_nvfp4_quant.py
def cast_to_fp4(x):
    """Cast a tensor to the E2M1 FP4 representable grid.

    Values are mapped to the nearest representable FP4 value using the
    piecewise rounding scheme from the vLLM NVFP4 reference implementation.
    The representable magnitudes are ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` and the
    result is clamped to ``[-6, 6]``.

    Args:
        x (torch.Tensor): Input tensor of any shape and floating-point dtype.

    Returns:
        torch.Tensor: Tensor with the same shape and dtype as *x* whose values
            lie on the FP4 grid.
    """
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
    """Compute the reciprocal of *x*, returning 0 where *x* is zero.

    Args:
        x (torch.Tensor or float or int): Value or tensor to invert.

    Returns:
        torch.Tensor or float: Element-wise reciprocal with zeros preserved
            as zeros.

    Raises:
        TypeError: If *x* is neither a ``torch.Tensor`` nor a numeric scalar.
    """
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
    """Calculate the global scaling factor for NVFP4 quantization.

    The global scale is defined as ``FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX /
    abs_max(tensor)`` and converts the full-precision absolute maximum of the
    weight tensor into the FP8 × FP4 representable range.

    Args:
        tensor (torch.Tensor or float or int): Weight tensor (or scalar) whose
            absolute maximum is used as the reference.
        group_size (int, optional): Block size for quantization.  Only
            ``group_size=16`` is currently supported.  Defaults to ``16``.
        device (str, optional): Target device for scalar inputs.  Defaults to
            ``"cpu"``.

    Returns:
        torch.Tensor: Scalar float32 tensor containing the global scale.

    Raises:
        AssertionError: If *group_size* is not 16.
    """
    assert group_size == 16, f"Only group_size=16 is supported, got {group_size}"
    if isinstance(tensor, (float, int)):
        tensor_amax = torch.tensor(tensor, device=device, dtype=torch.float32).abs()
    elif isinstance(tensor, torch.Tensor):
        tensor_amax = tensor.abs().max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_amax)
    return global_scale


def ref_nvfp4_quant(x, global_scale, block_size=16, v=0, scale_coeff=1.0):
    """Reference NVFP4 quantization for a 2-D weight tensor.

    Per-group (row-wise) local scales are computed in float8 E4M3 format and
    then used to map each element to the FP4 E2M1 grid.

    Args:
        x (torch.Tensor): 2-D input tensor with shape ``(m, n)``.
        global_scale (torch.Tensor): Scalar float32 global scale (see
            :func:`calculate_gparam`).
        block_size (int, optional): Number of elements per quantization group
            (must match the column dimension).  Defaults to ``16``.
        v (float or torch.Tensor, optional): Optional additive offset applied
            after scaling (used for rounding perturbation).  Defaults to
            ``0``.
        scale_coeff (float or torch.Tensor, optional): Per-row scale
            coefficient multiplied into the group maximum before computing the
            local scale.  Defaults to ``1.0``.

    Returns:
        tuple:
            - qdq_tensor (torch.Tensor): Quantized-dequantized tensor with
              shape ``(m, n)``.
            - scale (torch.Tensor): Per-group float8 scale tensor.
    """
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
    """Quantize a weight tensor to NVFP4 format using a dynamic global scale.

    The global scale is computed from the tensor's absolute maximum when not
    provided.  A per-group local scale in FP8 E4M3 format is then derived
    and used to map each group to the FP4 E2M1 grid.

    Args:
        tensor (torch.Tensor): Input weight tensor.
        bits (int, optional): Quantization bit width (informational; FP4 is
            always 4-bit).  Defaults to ``4``.
        group_size (int, optional): Number of elements per quantization group.
            Defaults to ``16``.
        v (float or torch.Tensor, optional): Additive rounding perturbation.
            Defaults to ``0``.
        global_scale (torch.Tensor or None, optional): Pre-computed float32
            global scale.  When ``None`` it is derived from the tensor.
            Defaults to ``None``.
        max_scale (float or torch.Tensor, optional): Per-row scale coefficient
            for the local scale computation.  Defaults to ``1.0``.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        tuple:
            - qdq_res (torch.Tensor): Quantized-dequantized tensor.
            - scale (torch.Tensor): Per-group FP8 local scale tensor.
            - None: Placeholder for zero-point (not used).
    """
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
    """Quantize a weight tensor to NVFP4 using a static (caller-supplied) global scale.

    Unlike :func:`nv_fp4`, the global scale is derived from *tensor_max*
    rather than the current tensor's maximum, enabling consistent scaling
    across calibration and inference.

    Args:
        tensor (torch.Tensor or None): Input weight tensor.  When ``None`` or
            empty the function returns ``(tensor, None, None)`` immediately.
        bits (int, optional): Quantization bit width (informational).
            Defaults to ``4``.
        group_size (int, optional): Number of elements per quantization group.
            Defaults to ``16``.
        v (float or torch.Tensor, optional): Additive rounding perturbation.
            Defaults to ``0``.
        tensor_max (float or torch.Tensor or None, optional): Reference
            absolute maximum used to compute the global scale.  When ``None``
            the current tensor's maximum is used.  Defaults to ``None``.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        tuple:
            - qdq_res (torch.Tensor): Quantized-dequantized tensor.
            - scale (torch.Tensor): Per-group FP8 local scale tensor.
            - None: Placeholder for zero-point (not used).
    """
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
    """Encode non-negative floats into the UE5M3 (unsigned 5-exponent, 3-mantissa) byte format.

    The encoding uses the E5M3 layout where exponent 0 is reserved for
    subnormal values and exponent 31 / mantissa 7 indicates NaN.

    Args:
        x (torch.Tensor): Non-negative float tensor of any shape.  Negative
            values are clamped to zero before encoding.

    Returns:
        torch.Tensor: Byte tensor (``torch.uint8``) with the same shape as
            *x* containing the UE5M3-encoded values.
    """
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
    """Decode a UE5M3 byte tensor back to float32.

    Args:
        e5m3 (torch.Tensor): Byte tensor (``torch.uint8``) containing
            UE5M3-encoded values, as produced by :func:`float_to_e5m3_frexp`.

    Returns:
        torch.Tensor: Float32 tensor with the same shape as *e5m3*.
            NaN-encoded entries are decoded as ``float("nan")``.

    Raises:
        AssertionError: If *e5m3* dtype is not ``torch.uint8``.
    """
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
    """Cast a tensor to the UE5M3 representable grid via encode-then-decode.

    Args:
        tensor (torch.Tensor): Input float tensor.

    Returns:
        torch.Tensor: Tensor with the same shape and original dtype whose
            values lie on the UE5M3 grid.
    """
    orig_dtype = tensor.dtype
    encoded = float_to_e5m3_frexp(tensor)
    res = e5m3_to_float_tensor(encoded)
    res = res.to(orig_dtype)
    return res


def cast_to_ue5m3_ste(x):
    """Straight-Through Estimator (STE) wrapper for :func:`cast_to_ue5m3`.

    Applies UE5M3 quantization in the forward pass while passing gradients
    through unchanged during backpropagation.

    Args:
        x (torch.Tensor): Input float tensor.

    Returns:
        torch.Tensor: UE5M3-quantized tensor with gradients flowing through
            the identity function.
    """
    fp4 = (cast_to_ue5m3(x).to(x.dtype) - x).detach() + x

    return fp4


def ref_fp4_quant(x, global_scale, block_size=16, v=0, max_scale=1.0):
    """Reference FP4-v2 quantization using UE5M3 local scales.

    Similar to :func:`ref_nvfp4_quant` but uses the UE5M3 format instead of
    FP8 E4M3 for the per-group local scale.

    Args:
        x (torch.Tensor): 2-D input tensor with shape ``(m, n)``.
        global_scale (float or torch.Tensor): Scalar global scale; when a
            tensor its dtype must be ``torch.float32``.
        block_size (int, optional): Elements per quantization group.
            Defaults to ``16``.
        v (float or torch.Tensor, optional): Additive rounding perturbation.
            Defaults to ``0``.
        max_scale (float or torch.Tensor, optional): Per-row scale coefficient.
            Defaults to ``1.0``.

    Returns:
        tuple:
            - qdq_tensor (torch.Tensor): Quantized-dequantized tensor with
              shape ``(m, n)``.
            - scale (torch.Tensor): Per-group UE5M3 local scale tensor.
    """
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
    """Quantize a weight tensor to FP4-v2 using a static-derived global UE5M3 scale.

    The global scale is computed from *tensor_max* (or the tensor's current
    maximum when *tensor_max* is ``None``) and uses the UE5M3 per-group scale
    format.

    Args:
        tensor (torch.Tensor): Input weight tensor.
        bits (int, optional): Quantization bit width (informational).
            Defaults to ``4``.
        group_size (int, optional): Number of elements per quantization group.
            Must be ``16`` or ``32``.  Defaults to ``16``.
        v (float or torch.Tensor, optional): Additive rounding perturbation.
            Defaults to ``0``.
        tensor_max (float or torch.Tensor or None, optional): Reference
            absolute maximum for global-scale computation.  When ``None`` the
            current tensor's maximum is used.  Defaults to ``None``.
        max_scale (float or torch.Tensor, optional): Per-row scale coefficient.
            Defaults to ``1.0``.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        tuple:
            - qdq_res (torch.Tensor): Quantized-dequantized tensor.
            - scale (torch.Tensor): Per-group UE5M3 local scale tensor.
            - None: Placeholder for zero-point (not used).

    Raises:
        AssertionError: If *group_size* is not ``16`` or ``32``.
    """
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
    """Quantize a weight tensor to FP4-v2 with a unit global scale.

    Uses ``global_scale=1.0`` so the per-group UE5M3 local scale absorbs the
    full dynamic range of the tensor.

    Args:
        tensor (torch.Tensor): Input weight tensor.
        bits (int, optional): Quantization bit width (informational).
            Defaults to ``4``.
        group_size (int, optional): Number of elements per quantization group.
            Must be ``16`` or ``32``.  Defaults to ``32``.
        v (float or torch.Tensor, optional): Additive rounding perturbation.
            Defaults to ``0``.
        max_scale (float or torch.Tensor, optional): Per-row scale coefficient.
            Defaults to ``1.0``.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        tuple:
            - qdq_res (torch.Tensor): Quantized-dequantized tensor.
            - scale (torch.Tensor): Per-group UE5M3 local scale tensor.
            - None: Placeholder for zero-point (not used).

    Raises:
        AssertionError: If *group_size* is not ``16`` or ``32``.
    """
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
