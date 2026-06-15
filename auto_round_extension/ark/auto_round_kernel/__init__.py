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

from typing import Any, Optional
import torch
import sys
import warnings
from dataclasses import dataclass

class ARK_DT:
    float64 = 64
    float32 = 32
    float16 = 16
    bfloat16 = 65552
    int2 = 258
    int3 = 259
    int4 = 260
    int5 = 261
    int6 = 262
    int7 = 263
    int8 = 264
    int32 = 288
    float8_e4m3 = 8
    float8_e5m2 = 65544
    float8_e8m0 = 196616
    undef = 0


def cvt_dtype(dtype):
    if dtype == torch.float32:
        return ARK_DT.float32
    if dtype == torch.float16:
        return ARK_DT.float16
    if dtype == torch.bfloat16:
        return ARK_DT.bfloat16
    if dtype == torch.float8_e4m3fn:
        return ARK_DT.float8_e4m3
    if dtype == torch.float8_e5m2:
        return ARK_DT.float8_e5m2
    if dtype == torch.int8:
        return ARK_DT.int8
    if dtype == torch.int32:
        return ARK_DT.int32
    return ARK_DT.undef


def cvtstr_dtype(dtype):
    if dtype == "fp32":
        return ARK_DT.float32
    if dtype == "fp16":
        return ARK_DT.float16
    if dtype == "bf16":
        return ARK_DT.bfloat16
    if dtype == "fp8_e4m3":
        return ARK_DT.float8_e4m3
    if dtype == "fp8_e5m2":
        return ARK_DT.float8_e5m2
    if dtype == "fp8_e8m0":
        return ARK_DT.float8_e8m0
    if dtype == "int8":
        return ARK_DT.int8
    if dtype == "int4":
        return ARK_DT.int4
    if dtype == "int2":
        return ARK_DT.int2
    if dtype == "int3":
        return ARK_DT.int3
    if dtype == "int5":
        return ARK_DT.int5
    if dtype == "int6":
        return ARK_DT.int6
    if dtype == "int7":
        return ARK_DT.int7
    if dtype == "int32":
        return ARK_DT.int32
    return ARK_DT.undef


def get_stream(A: torch.Tensor) -> int:
    if A.device.type == "cpu":
        return 0
    if A.device.type == "xpu":
        return torch.xpu.current_stream().sycl_queue


def _normalize_tensor_layout(tensor_layout: str) -> str:
    layout = tensor_layout.upper()
    if layout not in ("HND", "NHD"):
        raise ValueError(f"tensor_layout must be either 'HND' or 'NHD', got {tensor_layout!r}")
    return layout


def _attention_shape(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, int, int, int]:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        batch, num_heads, seq_len, head_dim = tensor.shape
    else:
        batch, seq_len, num_heads, head_dim = tensor.shape
    return batch, num_heads, seq_len, head_dim


def _attention_strides_qko(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, int, int, int]:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        batch_stride, head_stride, seq_stride, dim_stride = tensor.stride()
    else:
        batch_stride, seq_stride, head_stride, dim_stride = tensor.stride()
    return seq_stride, dim_stride, head_stride, batch_stride


def _attention_strides_v(tensor: torch.Tensor, tensor_layout: str) -> tuple[int, int, int, int]:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        batch_stride, head_stride, seq_stride, dim_stride = tensor.stride()
    else:
        batch_stride, seq_stride, head_stride, dim_stride = tensor.stride()
    return dim_stride, seq_stride, head_stride, batch_stride


def _validate_attention_tensor(
    tensor: torch.Tensor,
    name: str,
    tensor_layout: str,
    *,
    expected_dtype: torch.dtype | None = None,
) -> tuple[int, int, int, int]:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be a 4D tensor")
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}")

    qko_strides = _attention_strides_qko(tensor, tensor_layout)
    if qko_strides[1] != 1:
        raise ValueError(f"{name} must be contiguous along the head-dim axis; got stride {qko_strides[1]}")
    if any(stride <= 0 for stride in qko_strides):
        raise ValueError(f"{name} must have positive non-zero strides, got {tensor.stride()}")

    return _attention_shape(tensor, tensor_layout)


def _empty_attention_output(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    tensor_layout: str,
) -> torch.Tensor:
    layout = _normalize_tensor_layout(tensor_layout)
    shape = (batch, num_heads, seq_len, head_dim) if layout == "HND" else (batch, seq_len, num_heads, head_dim)
    return torch.empty(shape, device=device, dtype=dtype)


# -----------------------------------------------------------------------------
# Module-level lib loading (replaces the previous singleton ``ARK`` class).
# -----------------------------------------------------------------------------

cpu_lib = None
xpu_lib = None

try:
    from . import auto_round_kernel_cpu as _cpu_lib_mod

    cpu_lib = _cpu_lib_mod
except ImportError as _e:
    print(f"ARK is unable to load CPU lib: {_e}")
    cpu_lib = None

if torch.xpu.is_available():
    try:
        from . import auto_round_kernel_xpu as _xpu_lib_mod

        xpu_lib = _xpu_lib_mod
    except ImportError as _e:
        print(f"ARK is unable to load XPU lib: {_e}")
        xpu_lib = None


def get_lib(A: torch.Tensor):
    lib = None
    if A.device.type == "xpu":
        lib = xpu_lib
    if A.device.type == "cpu":
        lib = cpu_lib
    if lib is None:
        raise NotImplementedError(f"Current device {A.device} is not supported")
    return lib


# A: mxk,  B: nxk, bias: n
def matmul(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    m = A.shape[0]
    n = B.shape[0]
    k = B.shape[1]
    lib = get_lib(A)
    ctype = A.dtype
    if A.device.type == "cpu":
        ctype = torch.float32
    C = torch.zeros(m, n, dtype=ctype, device=A.device)
    stream = get_stream(A)
    lib.matmul(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        cvt_dtype(B.dtype),
        C.contiguous().data_ptr(),
        cvt_dtype(C.dtype),
        bias.to(C.dtype).contiguous().data_ptr(),
        True,
    )
    return C


# A: mxk:s8,  B: nxk:s8, return: mxn:s32
def igemm_s8s8s32(A: torch.Tensor, B: torch.Tensor):
    m = A.shape[0]
    n = B.shape[0]
    k = B.shape[1]
    lib = get_lib(A)
    if lib is None:
        raise NotImplementedError(f"Current device {A.device} is not supported")
    C = torch.zeros(m, n, dtype=torch.int32, device=A.device)
    stream = get_stream(A)
    lib.matmul(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        cvt_dtype(B.dtype),
        C.contiguous().data_ptr(),
        cvt_dtype(C.dtype),
        0,
        True,
    )
    return C


# A: mxk:DT,  B: nxk:s8, scaleB: n:DT
# return: mxn:DT
def woqgemm_s8(A: torch.Tensor, B: torch.Tensor, scaleB: torch.Tensor, bias: torch.Tensor):
    m = A.shape[0]
    n = B.shape[0]
    k = B.shape[1]
    lib = get_lib(A)

    C = torch.zeros(m, n, dtype=A.dtype, device=A.device)
    stream = get_stream(A)
    lib.woqgemm_s8(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        C.contiguous().data_ptr(),
        bias.contiguous().data_ptr(),
        True,
        scaleB.contiguous().data_ptr(),
    )
    return C


# A: mxk:DT,  B: BS:s8, bias: n:DT
# return: C: mxn:DT
def woqgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    n,
    k,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    m = A.shape[0]
    lib = get_lib(A)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)
    st = cvtstr_dtype(scale_type)
    C = torch.zeros(m, n, dtype=A.dtype, device=A.device)
    stream = get_stream(A)
    lib.woqgemm(
        stream,
        m,
        n,
        k,
        A.contiguous().data_ptr(),
        cvt_dtype(A.dtype),
        B.contiguous().data_ptr(),
        C.contiguous().data_ptr(),
        bias.contiguous().data_ptr(),
        groupsize,
        ct,
        wt,
        st,
        asym,
    )
    return C


# QB: k*n:int8,  scaleB: k/blocksize*n:DT
# return: blob:BS:int8
def _repack_quantized_weight_core(
    QB: torch.Tensor,
    scaleB: torch.Tensor,
    zp: torch.Tensor,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    k = QB.shape[0]
    n = QB.shape[1]
    lib = get_lib(QB)
    stream = get_stream(QB)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)
    st = cvtstr_dtype(scale_type)
    BS = lib.packed_weight_size(stream, n, k, groupsize, ct, wt, st, asym)
    blob = torch.zeros(BS, dtype=torch.int8, device=QB.device)
    lib.repack_quantized_weight(
        stream,
        QB.contiguous().data_ptr(),
        zp.contiguous().data_ptr(),
        scaleB.contiguous().data_ptr(),
        blob.data_ptr(),
        n,
        k,
        groupsize,
        ct,
        wt,
        st,
        asym,
    )
    return blob


# QB: blob:BS:int8
# return: out:nxk:out_dtype
def _unpack_weight_core(
    blob: torch.Tensor,
    out_dtype: torch.dtype,
    n,
    k,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    lib = get_lib(blob)
    stream = get_stream(blob)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)

    st = cvtstr_dtype(scale_type)
    oshape = (n, k) if blob.device.type == "xpu" else (k, n)
    out = torch.zeros(oshape, dtype=out_dtype, device=blob.device)
    lib.unpack_weight(
        stream,
        blob.data_ptr(),
        out.data_ptr(),
        cvt_dtype(out_dtype),
        n,
        k,
        groupsize,
        ct,
        wt,
        st,
        asym,
    )
    if blob.device.type == "cpu":
        return out.T
    return out


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Scaled dot-product attention (SDPA) prefill+decode.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - scale: Softmax scale. Uses 1 / sqrt(D) when None.
    - tensor_layout: Layout of Q/K/V/O tensors.

    Returns:
    - O: same layout as the input tensors.
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sdpa is only supported on XPU")

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=query.dtype)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout, expected_dtype=query.dtype)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128, 96, 192):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128, 96, 192")

    if dropout_p != 0.0:
        raise NotImplementedError(f"dropout_p must be 0.0 (got {dropout_p}); dropout is not supported")

    if attn_mask is not None:
        if attn_mask.device.type != "xpu":
            raise ValueError("attn_mask must be on XPU")
        if not attn_mask.is_contiguous():
            raise ValueError("attn_mask must be contiguous")
        if attn_mask.dtype != torch.float32:
            raise ValueError(f"attn_mask must be float32 (additive bias), got {attn_mask.dtype}")
        expected_mask_shape = (B, 1, Sq, Skv)
        if attn_mask.shape != expected_mask_shape:
            raise ValueError(f"attn_mask shape must be {expected_mask_shape}, got {tuple(attn_mask.shape)}")

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    q_strides = _attention_strides_qko(query, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    o_strides = _attention_strides_qko(O, tensor_layout)
    lib.sdpa(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return O


def sage(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    qscale: torch.Tensor = None,
    kscale: torch.Tensor = None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """SAGE attention prefill+decode.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - scale: Attention scale. Uses 1 / sqrt(D) when None.
    - quant_block_size: Block size for qscale and kscale.
    - tensor_layout: Layout of Q/K/V/O tensors.

    Returns:
    - O: same layout as the input tensors.
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sdpa is only supported on XPU")

    # if query.dtype not in (torch.float16, torch.bfloat16):
    #     raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    # if key.dtype != query.dtype or value.dtype != query.dtype:
    #     raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    q_strides = _attention_strides_qko(query, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    o_strides = _attention_strides_qko(O, tensor_layout)
    lib.sage(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        qscale.data_ptr() if qscale is not None else 0,
        kscale.data_ptr() if kscale is not None else 0,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return O


def sage_pvi8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    qscale: torch.Tensor = None,
    kscale: torch.Tensor = None,
    vscale: torch.Tensor = None,
    out_dtype: torch.dtype = torch.float16,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Low-level SAGE attention with pre-quantized INT8 Q/K/V and PV int8.

    Expects contiguous layouts:
    - query: [B, Hq, Sq, D] int8
    - key: [B, Hkv, Skv, D] int8
    - value: [B, Hkv, Skv, D] int8
    - qscale: [B, Hq, ceil(Sq / quant_block_size), 1] float32
    - kscale: [B, Hkv, ceil(Skv / quant_block_size), 1] float32
    - vscale: [B, Hkv, ceil(Skv / quant_block_size), D] float32

    Returns:
    - O: [B, Hq, Sq, D] float16
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sage_pvi8 is only supported on XPU")
    if query.dtype != torch.int8 or key.dtype != torch.int8 or value.dtype != torch.int8:
        raise ValueError(f"Q/K/V must be int8, got Q={query.dtype}, K={key.dtype}, V={value.dtype}")
    if out_dtype != torch.float16:
        raise ValueError(f"sage_pvi8 output must be float16, got {out_dtype}")
    if qscale is None or kscale is None or vscale is None:
        raise ValueError("qscale, kscale and vscale must be provided for sage_pvi8")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    q_blocks = (Sq + quant_block_size - 1) // quant_block_size
    kv_blocks = (Skv + quant_block_size - 1) // quant_block_size
    if qscale.numel() != B * Hq * q_blocks:
        raise ValueError(
            f"qscale must have {B * Hq * q_blocks} elements for shape [B, Hq, ceil(Sq/block), 1], got {qscale.numel()}"
        )
    if kscale.numel() != B * Hkv * kv_blocks:
        raise ValueError(
            f"kscale must have {B * Hkv * kv_blocks} elements for shape [B, Hkv, ceil(Skv/block), 1], got {kscale.numel()}"
        )
    if vscale.numel() != B * Hkv * kv_blocks * D:
        raise ValueError(
            f"vscale must have {B * Hkv * kv_blocks * D} elements for shape [B, Hkv, ceil(Skv/block), D], got {vscale.numel()}"
        )

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=out_dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    q_strides = _attention_strides_qko(query, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    o_strides = _attention_strides_qko(O, tensor_layout)
    lib.sage_pvi8(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        qscale.data_ptr(),
        kscale.data_ptr(),
        vscale.data_ptr(),
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return O


def sage_sparse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    qscale: torch.Tensor = None,
    kscale: torch.Tensor = None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Low-level sparse SAGE attention with pre-quantized INT8 Q/K and LUT-driven block selection."""
    if query.device.type != "xpu":
        raise NotImplementedError("sage_sparse is only supported on XPU")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError(f"Q/K must be int8, got Q={query.dtype}, K={key.dtype}")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"V must be float16 or bfloat16, got {value.dtype}")
    if qscale is None or kscale is None:
        raise ValueError("qscale and kscale must be provided for sage_sparse")
    if lut.dtype != torch.int32 or valid_block_num.dtype != torch.int32:
        raise ValueError("lut and valid_block_num must be int32 tensors")
    if lut.device != query.device or valid_block_num.device != query.device:
        raise ValueError("lut and valid_block_num must be on the same XPU device as Q/K/V")
    if qscale.device != query.device or kscale.device != query.device:
        raise ValueError("qscale and kscale must be on the same XPU device as Q/K/V")
    if quant_block_size <= 0:
        raise ValueError(f"quant_block_size must be positive, got {quant_block_size}")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    q_blocks = (Sq + quant_block_size - 1) // quant_block_size
    kv_blocks = (Skv + quant_block_size - 1) // quant_block_size
    if tuple(lut.shape) != (B, Hq, q_blocks, kv_blocks):
        raise ValueError(f"lut must have shape {(B, Hq, q_blocks, kv_blocks)}, got {tuple(lut.shape)}")
    if tuple(valid_block_num.shape) != (B, Hq, q_blocks):
        raise ValueError(f"valid_block_num must have shape {(B, Hq, q_blocks)}, got {tuple(valid_block_num.shape)}")
    if qscale.numel() != B * Hq * q_blocks:
        raise ValueError(
            f"qscale must have {B * Hq * q_blocks} elements for shape [B, Hq, ceil(Sq/block), 1], got {qscale.numel()}"
        )
    if kscale.numel() != B * Hkv * kv_blocks:
        raise ValueError(
            f"kscale must have {B * Hkv * kv_blocks} elements for shape [B, Hkv, ceil(Skv/block), 1], got {kscale.numel()}"
        )
    if torch.any(valid_block_num < 0).item():
        raise ValueError("valid_block_num entries must be non-negative")
    if torch.any(valid_block_num > kv_blocks).item():
        raise ValueError(f"valid_block_num entries must be <= {kv_blocks}")

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    q_strides = _attention_strides_qko(query, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    o_strides = _attention_strides_qko(O, tensor_layout)
    lib.sage_sparse(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        qscale.data_ptr(),
        kscale.data_ptr(),
        lut.data_ptr(),
        valid_block_num.data_ptr(),
        q_blocks,
        kv_blocks,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return O


def sage_sparse_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    qscale: torch.Tensor | None = None,
    kscale: torch.Tensor | None = None,
    kscale_cache: torch.Tensor | None = None,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Low-level sparse SAGE decode with pre-quantized INT8 Q/K and explicit KV cache."""
    if query.device.type != "xpu":
        raise NotImplementedError("sage_sparse_decode is only supported on XPU")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p must be 0.0 for sage_sparse_decode")
    if enable_gqa:
        raise NotImplementedError("enable_gqa is not used by sage_sparse_decode; provide explicit Hq/Hkv tensors")
    if query.dtype != torch.int8 or key.dtype != torch.int8 or key_cache.dtype != torch.int8:
        raise ValueError(f"Q/K/K_cache must be int8, got Q={query.dtype}, K={key.dtype}, K_cache={key_cache.dtype}")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"V must be float16 or bfloat16, got {value.dtype}")
    if value_cache.dtype != value.dtype:
        raise ValueError(f"V_cache dtype must match V dtype, got {value_cache.dtype} vs {value.dtype}")
    if qscale is None or kscale is None or kscale_cache is None:
        raise ValueError("qscale, kscale, and kscale_cache must be provided for sage_sparse_decode")
    if lut.dtype != torch.int32 or valid_block_num.dtype != torch.int32:
        raise ValueError("lut and valid_block_num must be int32 tensors")
    if quant_block_size <= 0:
        raise ValueError(f"quant_block_size must be positive, got {quant_block_size}")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout)
    Bkc, Hkvc, Skvc, Dkc = _validate_attention_tensor(key_cache, "K_cache", tensor_layout)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout)
    Bvc, Hkvc2, Skvc2, Dvc = _validate_attention_tensor(value_cache, "V_cache", tensor_layout)
    if Bk != B or Bkc != B or Bv != B or Bvc != B:
        raise ValueError("Batch size mismatch between Q/K/V/cache tensors")
    if Hkv2 != Hkv or Hkvc != Hkv or Hkvc2 != Hkv:
        raise ValueError("KV head-count mismatch")
    if Skv2 != Skv or Skvc2 != Skvc:
        raise ValueError("K/V sequence length mismatch")
    if Dk != D or Dkc != D or Dv != D or Dvc != D:
        raise ValueError("Head dim mismatch between Q/K/V/cache tensors")
    if Sq != 1:
        raise ValueError(f"sage_sparse_decode currently supports only seq_len_q == 1, got {Sq}")
    if Skv <= 0 or Skvc <= 0:
        raise ValueError("sage_sparse_decode requires both current-step KV and KV cache to be non-empty")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    q_blocks = (Sq + quant_block_size - 1) // quant_block_size
    cur_blocks = (Skv + quant_block_size - 1) // quant_block_size
    cache_blocks = (Skvc + quant_block_size - 1) // quant_block_size
    total_blocks = cur_blocks + cache_blocks
    if tuple(lut.shape) != (B, Hq, q_blocks, total_blocks):
        raise ValueError(f"lut must have shape {(B, Hq, q_blocks, total_blocks)}, got {tuple(lut.shape)}")
    if tuple(valid_block_num.shape) != (B, Hq, q_blocks):
        raise ValueError(f"valid_block_num must have shape {(B, Hq, q_blocks)}, got {tuple(valid_block_num.shape)}")
    if qscale.numel() != B * Hq * q_blocks:
        raise ValueError(f"qscale must have {B * Hq * q_blocks} elements, got {qscale.numel()}")
    if kscale.numel() != B * Hkv * cur_blocks:
        raise ValueError(f"kscale must have {B * Hkv * cur_blocks} elements, got {kscale.numel()}")
    if kscale_cache.numel() != B * Hkv * cache_blocks:
        raise ValueError(f"kscale_cache must have {B * Hkv * cache_blocks} elements, got {kscale_cache.numel()}")
    if torch.any(valid_block_num < 0).item():
        raise ValueError("valid_block_num entries must be non-negative")
    if torch.any(valid_block_num > total_blocks).item():
        raise ValueError(f"valid_block_num entries must be <= {total_blocks}")

    q_hnd = _to_hnd(query, tensor_layout)
    k_hnd = _to_hnd(key, tensor_layout)
    v_hnd = _to_hnd(value, tensor_layout)
    k_cache_hnd = _to_hnd(key_cache, tensor_layout)
    v_cache_hnd = _to_hnd(value_cache, tensor_layout)
    O_hnd = torch.empty((B, Hq, Sq, D), dtype=value.dtype, device=query.device)
    q_strides = _attention_strides_qko(q_hnd, "HND")
    k_strides = _attention_strides_qko(k_hnd, "HND")
    v_strides = _attention_strides_v(v_hnd, "HND")
    o_strides = _attention_strides_qko(O_hnd, "HND")
    kscale_total = torch.cat([kscale_cache.reshape(B, Hkv, cache_blocks, 1), kscale.reshape(B, Hkv, cur_blocks, 1)], dim=2)
    lib = get_lib(query)
    stream = get_stream(query)
    lib.sage_sparse_decode(
        stream,
        q_hnd.data_ptr(),
        k_hnd.data_ptr(),
        v_hnd.data_ptr(),
        k_cache_hnd.data_ptr(),
        v_cache_hnd.data_ptr(),
        O_hnd.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        qscale.data_ptr(),
        kscale_total.data_ptr(),
        lut.data_ptr(),
        valid_block_num.data_ptr(),
        q_blocks,
        total_blocks,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(q_hnd.dtype),
        cvt_dtype(k_hnd.dtype),
        cvt_dtype(O_hnd.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        Skvc,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return _from_hnd(O_hnd, tensor_layout)


def sagev1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """SAGE v1 attention prefill+decode.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - scale: Attention scale. Uses 1 / sqrt(D) when None.
    - quant_block_size: Quantization block size used by the kernel.
    - tensor_layout: Layout of Q/K/V/O tensors.

    Returns:
    - O: same layout as the input tensors.
    """
    if quant_block_size <= 0:
        return sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            tensor_layout=tensor_layout,
        )
    if query.device.type != "xpu":
        raise NotImplementedError("sdpa is only supported on XPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=query.dtype)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout, expected_dtype=query.dtype)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    q_strides = _attention_strides_qko(query, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    o_strides = _attention_strides_qko(O, tensor_layout)
    lib.sagev1(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(value.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return O


def sagev1_pvi8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """SAGE v1 attention with PV int8 path.

    Expects FP16 Q/K/V input and quantizes Q/K/V internally before calling
    the PV int8 kernel.
    """
    if quant_block_size <= 0:
        return sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            tensor_layout=tensor_layout,
        )
    if query.device.type != "xpu":
        raise NotImplementedError("sagev1_pvi8 is only supported on XPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=query.dtype)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout, expected_dtype=query.dtype)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    lib = get_lib(query)
    stream = get_stream(query)
    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    q_strides = _attention_strides_qko(query, tensor_layout)
    k_strides = _attention_strides_qko(key, tensor_layout)
    v_strides = _attention_strides_v(value, tensor_layout)
    o_strides = _attention_strides_qko(O, tensor_layout)
    lib.sagev1_pvi8(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(value.dtype),
        cvt_dtype(O.dtype),
        B,
        Hq,
        Hkv,
        Sq,
        Skv,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
    )
    return O


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    kernel: str = "v1_pvhalf",
    **kwargs,
) -> torch.Tensor:
    """SAGE attention dispatcher.

    Signature mirrors ``sageattention.sageattn``.

    Args:
    - q, k, v: Query/Key/Value tensors. Layout selected by ``tensor_layout``.
    - tensor_layout: "HND" or "NHD".
    - is_causal: Whether to apply causal mask.
    - sm_scale: Softmax scale. Uses ``1 / sqrt(head_dim)`` when None.
    - return_lse: Not supported; must be False.
    - kernel: Which SAGE variant to dispatch to.
        - "v1_pvhalf" (default): PV in half precision (calls ``sagev1``).
        - "v1_pvi8": PV in INT8 precision (calls ``sagev1_pvi8``).
    - kwargs: Forwarded to the underlying kernel (e.g. ``attn_mask``,
      ``dropout_p``, ``enable_gqa``, ``quant_block_size``).

    Returns:
    - O: same layout as the input tensors.
    """
    if return_lse:
        raise NotImplementedError("return_lse is not supported in ARK sageattn")

    if kernel == "v1_pvhalf":
        impl = sagev1
    elif kernel == "v1_pvi8":
        impl = sagev1_pvi8
    else:
        raise ValueError(f"Unsupported sageattn kernel={kernel!r}; supported: 'v1_pvhalf', 'v1_pvi8'")

    return impl(
        query=q,
        key=k,
        value=v,
        is_causal=is_causal,
        scale=sm_scale,
        tensor_layout=tensor_layout,
        **kwargs,
    )


def _normalize_sparse_mask(
    attn_mask: torch.Tensor | None,
    batch: int,
    seq_q: int,
    seq_kv: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        raise ValueError("Boolean attention masks are not supported")

    mask = attn_mask
    if mask.ndim == 2:
        if mask.shape != (seq_q, seq_kv):
            raise ValueError(f"Unsupported 2D attention mask shape {tuple(mask.shape)}")
        mask = mask.view(1, 1, seq_q, seq_kv)
    elif mask.ndim == 3:
        if mask.shape != (batch, seq_q, seq_kv):
            raise ValueError(f"Unsupported 3D attention mask shape {tuple(mask.shape)}")
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[-2:] != (seq_q, seq_kv):
            raise ValueError(f"Unsupported 4D attention mask shape {tuple(mask.shape)}")
        if mask.shape[1] != 1:
            raise ValueError("Only attention masks with head dimension 1 are supported")
        if mask.shape[0] == 1 and batch != 1:
            mask = mask.expand(batch, -1, -1, -1)
        elif mask.shape[0] != batch:
            raise ValueError(f"Unsupported attention mask batch dimension {mask.shape[0]} != {batch}")
    else:
        raise ValueError(f"Unsupported attention mask rank {mask.ndim}")

    return mask.contiguous().to(device=device, dtype=torch.float32)


def _normalize_per_head_hparam(
    value: float | int | torch.Tensor,
    num_heads: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.float32).flatten()
        if out.numel() == 1:
            out = out.expand(num_heads)
        elif out.numel() != num_heads:
            raise ValueError(f"{name} must have 1 or {num_heads} elements, got {out.numel()}")
        return out.contiguous()
    return torch.full((num_heads,), float(value), device=device, dtype=torch.float32)


def _query_tile_tokens_for_head_dim(head_dim: int) -> int:
    if head_dim == 64:
        return 128
    if head_dim == 128:
        return 256
    raise ValueError(f"Unsupported head_dim={head_dim}; supported: 64, 128")


def _sequence_mean_native_layout(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    layout = _normalize_tensor_layout(tensor_layout)
    seq_dim = 2 if layout == "HND" else 1
    return tensor.mean(dim=seq_dim).contiguous()


def _slice_sequence_native_layout(tensor: torch.Tensor, tensor_layout: str, start: int, end: int) -> torch.Tensor:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        return tensor[:, :, start:end, :]
    return tensor[:, start:end, :, :]


def _to_hnd(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        return tensor.contiguous()
    return tensor.permute(0, 2, 1, 3).contiguous()


def _from_hnd(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        return tensor.contiguous()
    return tensor.permute(0, 2, 1, 3).contiguous()


def _safe_softmax(scores: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.full_like(scores, -1.0e9))
    probs = torch.softmax(safe_scores, dim=-1)
    probs = torch.where(finite, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, probs / denom, torch.zeros_like(probs))


def _build_block_causal_mask(num_q_tiles: int, num_k_blocks: int, blocks_per_qtile: int, device: torch.device) -> torch.Tensor:
    q_idx = torch.arange(num_q_tiles, device=device, dtype=torch.int64).view(-1, 1)
    k_idx = torch.arange(num_k_blocks, device=device, dtype=torch.int64).view(1, -1)
    return k_idx < ((q_idx + 1) * blocks_per_qtile)


def _fill_block_map_torch(final_map: torch.Tensor, num_to_select: torch.Tensor, sorted_indices: torch.Tensor) -> torch.Tensor:
    k_blocks = final_map.shape[-1]
    filled = final_map.clone()
    column_ids = torch.arange(k_blocks, device=final_map.device).view(1, 1, 1, k_blocks)
    for rank in range(k_blocks):
        active = (num_to_select > rank).unsqueeze(-1)
        filled |= active & (column_ids == sorted_indices[..., rank : rank + 1])
    return filled


def _block_map_lut_torch(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid_block_num = block_map.to(torch.int32).sum(dim=-1)
    _, _, _, num_k_blocks = block_map.shape
    one_matrix = torch.ones(block_map.shape, dtype=torch.int32, device=block_map.device)
    cum_matrix = torch.cumsum(one_matrix, dim=-1)
    masked_cum_matrix = cum_matrix * block_map.to(torch.int32)
    filled_matrix = masked_cum_matrix.clone()
    filled_matrix[~block_map] = 10_000_000
    lut = torch.sort(filled_matrix, dim=-1)[0] - 1
    lut[..., 1:] = lut[..., 1:] - lut[..., :-1]
    invalid_mask = torch.arange(num_k_blocks, device=block_map.device).view(1, 1, 1, num_k_blocks) >= valid_block_num.unsqueeze(-1)
    lut = torch.where(invalid_mask, torch.zeros_like(lut), lut)
    return lut.to(torch.int32).contiguous(), valid_block_num.to(torch.int32).contiguous()


def _pool_sim_and_quant_torch(
    x: torch.Tensor,
    block_size: int,
    sim_threshold: torch.Tensor,
    tensor_layout: str,
    mean_subtract: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        bsz, num_heads, seq_len, head_dim = x.shape
    else:
        bsz, seq_len, num_heads, head_dim = x.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    pad_tokens = num_blocks * block_size - seq_len
    if layout == "HND":
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_tokens))
        x_blocks = x_pad.view(bsz, num_heads, num_blocks, block_size, head_dim).to(torch.float32)
    else:
        if pad_tokens:
            pad = torch.zeros((bsz, pad_tokens, num_heads, head_dim), dtype=x.dtype, device=x.device)
            x_pad = torch.cat([x, pad], dim=1)
        else:
            x_pad = x
        x_blocks = x_pad.view(bsz, num_blocks, block_size, num_heads, head_dim).permute(0, 3, 1, 2, 4).to(torch.float32)

    valid_tokens = torch.ones((seq_len,), device=x.device, dtype=torch.float32)
    if pad_tokens:
        valid_tokens = torch.nn.functional.pad(valid_tokens, (0, pad_tokens))
    valid_mask = valid_tokens.view(1, 1, num_blocks, block_size, 1)
    counts = valid_mask.sum(dim=-2).clamp_min_(1.0)

    if mean_subtract is not None:
        mean_values = mean_subtract.squeeze(-2) if mean_subtract.ndim == 4 else mean_subtract
        x_blocks = x_blocks - mean_values.to(torch.float32).unsqueeze(2).unsqueeze(2)
    x_blocks = x_blocks * valid_mask

    pooled = x_blocks.sum(dim=-2) / counts

    norms = torch.linalg.vector_norm(x_blocks, dim=-1, keepdim=True)
    normalized = torch.where(norms > 0, x_blocks / norms, torch.zeros_like(x_blocks))
    grams = torch.matmul(normalized, normalized.transpose(-1, -2))
    mean_sim = grams.sum(dim=(-1, -2)) / counts.squeeze(-1).squeeze(-1).pow(2)
    sim_blocks = mean_sim > sim_threshold.view(1, num_heads, 1)

    max_abs = x_blocks.abs().amax(dim=(-1, -2)).clamp_min_(0.0)
    scales = (max_abs / 127.0) + 1.0e-7
    q = x_blocks / scales.unsqueeze(-1).unsqueeze(-1)
    q = torch.where(q >= 0, torch.floor(q + 0.5), torch.ceil(q - 0.5))
    q = q.clamp_(-127, 127).to(torch.int8)
    if layout == "HND":
        q_native = q.view(bsz, num_heads, num_blocks * block_size, head_dim)[:, :, :seq_len, :].contiguous()
    else:
        q_native = (
            q.permute(0, 2, 3, 1, 4)
            .reshape(bsz, num_blocks * block_size, num_heads, head_dim)[:, :seq_len, :, :]
            .contiguous()
        )

    return pooled.to(x.dtype), sim_blocks.contiguous(), q_native, scales.unsqueeze(-1).to(torch.float32).contiguous()


def _get_sparse_block_sparsity_stats(
    valid_block_num: torch.Tensor,
    lut: torch.Tensor,
    *,
    is_causal: bool = False,
) -> tuple[int, int, float, float, float]:
    total_selected = int(valid_block_num.sum().item())
    del is_causal
    total_candidates = lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1)
    selected_ratio = float(total_selected / total_candidates) if total_candidates > 0 else 0.0
    sparsity_ratio = 1.0 - selected_ratio
    total_rows = valid_block_num.numel()
    selected_blocks_per_row = float(total_selected / total_rows) if total_rows > 0 else 0.0
    return total_selected, total_candidates, selected_ratio, sparsity_ratio, selected_blocks_per_row


@dataclass(frozen=True)
class _SpargePreprocessContext:
    query: torch.Tensor
    key: torch.Tensor
    batch: int
    num_heads_q: int
    num_heads_kv: int
    seq_len_q: int
    seq_len_kv: int
    head_dim: int
    is_causal: bool
    smooth_k: bool
    simthreshd1: torch.Tensor
    topk: torch.Tensor
    attention_sink: bool
    quant_block_size: int
    tensor_layout: str
    query_tile_tokens: int
    blocks_per_qtile: int
    num_q_tiles: int


def _build_sparge_preprocess_context(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    is_causal: bool,
    smooth_k: bool,
    simthreshd1: float | torch.Tensor,
    topk: float | torch.Tensor,
    attention_sink: bool,
    quant_block_size: int,
    tensor_layout: str,
) -> _SpargePreprocessContext:
    if query.device.type != "xpu":
        raise NotImplementedError("sparge_preprocess_topk is only supported on XPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype:
        raise ValueError(f"K dtype must match Q dtype, got K={key.dtype}, Q={query.dtype}")
    if quant_block_size != 64:
        raise ValueError(f"quant_block_size={quant_block_size} is not supported in sparge_preprocess_topk; only 64 is supported")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout, expected_dtype=query.dtype)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=query.dtype)
    if Bk != B:
        raise ValueError("Batch size mismatch between Q and K")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K")
    if Sq != Skv:
        raise ValueError("sparge_preprocess_topk currently supports prefill only: seq_len_q must equal seq_len_kv")
    if Hq % Hkv != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_kv")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    query_tile_tokens = _query_tile_tokens_for_head_dim(D)
    blocks_per_qtile = query_tile_tokens // quant_block_size
    num_q_tiles = (Sq + query_tile_tokens - 1) // query_tile_tokens

    return _SpargePreprocessContext(
        query=query,
        key=key,
        batch=B,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        seq_len_q=Sq,
        seq_len_kv=Skv,
        head_dim=D,
        is_causal=is_causal,
        smooth_k=smooth_k,
        simthreshd1=_normalize_per_head_hparam(simthreshd1, Hq, query.device, "simthreshd1"),
        topk=_normalize_per_head_hparam(topk, Hq, query.device, "topk").clamp_(0.0, 1.0),
        attention_sink=attention_sink,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
        query_tile_tokens=query_tile_tokens,
        blocks_per_qtile=blocks_per_qtile,
        num_q_tiles=num_q_tiles,
    )


def _sparge_preprocess_topk_torch_impl(ctx: _SpargePreprocessContext) -> dict[str, Any]:
    key_mean = _sequence_mean_native_layout(ctx.key, ctx.tensor_layout) if ctx.smooth_k else None
    pooled_q, sim_qblocks, q_int8_hnd, q_scale = _pool_sim_and_quant_torch(
        ctx.query,
        ctx.quant_block_size,
        ctx.simthreshd1,
        ctx.tensor_layout,
    )
    pooled_k, sim_kblocks, k_int8_hnd, k_scale = _pool_sim_and_quant_torch(
        ctx.key,
        ctx.quant_block_size,
        ctx.simthreshd1[: ctx.num_heads_kv],
        ctx.tensor_layout,
        key_mean,
    )

    if ctx.blocks_per_qtile > 1:
        tile_pooled_q = []
        tile_sim_q = []
        for qtile in range(ctx.num_q_tiles):
            qblk_start = qtile * ctx.blocks_per_qtile
            qblk_end = min(qblk_start + ctx.blocks_per_qtile, pooled_q.size(2))
            tile_tokens = _slice_sequence_native_layout(
                ctx.query,
                ctx.tensor_layout,
                qblk_start * ctx.quant_block_size,
                min((qblk_end * ctx.quant_block_size), ctx.seq_len_q),
            )
            pooled_tile, sim_tile, _, _ = _pool_sim_and_quant_torch(
                tile_tokens,
                ctx.query_tile_tokens,
                ctx.simthreshd1,
                ctx.tensor_layout,
            )
            tile_pooled_q.append(pooled_tile[:, :, 0, :])
            tile_sim_q.append(sim_tile[:, :, 0])
        pooled_q_for_routing = torch.stack(tile_pooled_q, dim=2)
        sim_q_for_routing = torch.stack(tile_sim_q, dim=2)
    else:
        pooled_q_for_routing = pooled_q
        sim_q_for_routing = sim_qblocks

    kv_head_index = torch.arange(ctx.num_heads_q, device=ctx.query.device, dtype=torch.int64) // (
        ctx.num_heads_q // ctx.num_heads_kv
    )
    pooled_k_for_q = pooled_k[:, kv_head_index]
    sim_k_for_q = sim_kblocks[:, kv_head_index]
    sim_k_expand = sim_k_for_q.unsqueeze(-2).expand(-1, -1, ctx.num_q_tiles, -1)
    sim_q_expand = sim_q_for_routing.unsqueeze(-1).expand(-1, -1, -1, pooled_k.size(2))

    pooled_score = torch.matmul(pooled_q_for_routing.to(torch.float32), pooled_k_for_q.transpose(-1, -2).to(torch.float32))
    pooled_score *= ctx.head_dim ** -0.5
    pooled_score = pooled_score.masked_fill(~sim_k_expand, -torch.inf)
    if ctx.is_causal:
        causal_mask = _build_block_causal_mask(ctx.num_q_tiles, pooled_k.size(2), ctx.blocks_per_qtile, ctx.query.device)
        pooled_score = pooled_score.masked_fill(~causal_mask.view(1, 1, ctx.num_q_tiles, pooled_k.size(2)), -torch.inf)
    else:
        causal_mask = None

    pooled_prob = _safe_softmax(pooled_score)
    sorted_prob = torch.sort(pooled_prob, dim=-1, descending=True)
    _, _, _, num_k_blocks = pooled_prob.shape
    num_to_select = (
        (ctx.topk.view(1, ctx.num_heads_q, 1) * num_k_blocks)
        .to(torch.int64)
        .expand(ctx.batch, -1, ctx.num_q_tiles)
        .contiguous()
    )
    final_tile_map = torch.zeros_like(pooled_prob, dtype=torch.bool)
    final_tile_map[~sim_k_expand] = True
    final_tile_map[~sim_q_expand] = True
    final_tile_map = _fill_block_map_torch(final_tile_map, num_to_select, sorted_prob.indices)
    if causal_mask is not None:
        final_tile_map &= causal_mask.view(1, 1, ctx.num_q_tiles, num_k_blocks)
    if ctx.attention_sink:
        final_tile_map[..., 0] = True

    q_block_to_tile = (
        torch.arange((ctx.seq_len_q + ctx.quant_block_size - 1) // ctx.quant_block_size, device=ctx.query.device)
        * ctx.quant_block_size
    ) // ctx.query_tile_tokens
    q_block_to_tile = q_block_to_tile.clamp_max(ctx.num_q_tiles - 1)
    raw_block_map = final_tile_map.index_select(2, q_block_to_tile).contiguous()

    return {
        "query_i8": q_int8_hnd,
        "key_i8": k_int8_hnd,
        "qscale": q_scale,
        "kscale": k_scale,
        "raw_block_map": raw_block_map,
        "tile_block_map": final_tile_map.contiguous(),
        "sim_qblocks": sim_q_for_routing.contiguous(),
        "sim_kblocks": sim_kblocks.contiguous(),
        "backend": "torch",
    }


def _finalize_sparge_preprocess_outputs(
    ctx: _SpargePreprocessContext,
    backend_result: dict[str, Any],
) -> dict[str, Any]:
    raw_block_map = backend_result["raw_block_map"].contiguous()
    block_map = raw_block_map
    lut = backend_result.get("lut")
    valid_block_num = backend_result.get("valid_block_num")
    if lut is None or valid_block_num is None:
        lut, valid_block_num = _block_map_lut_torch(block_map)
    total_selected, total_candidates, selected_ratio, sparsity_ratio, selected_blocks_per_row = _get_sparse_block_sparsity_stats(
        valid_block_num,
        lut,
        is_causal=ctx.is_causal,
    )

    return {
        "query_i8": backend_result["query_i8"].contiguous(),
        "key_i8": backend_result["key_i8"].contiguous(),
        "qscale": backend_result["qscale"].contiguous(),
        "kscale": backend_result["kscale"].contiguous(),
        "lut": lut,
        "valid_block_num": valid_block_num,
        "block_map": block_map,
        "raw_block_map": raw_block_map,
        "tile_block_map": backend_result["tile_block_map"].contiguous(),
        "sim_qblocks": backend_result["sim_qblocks"].contiguous(),
        "sim_kblocks": backend_result["sim_kblocks"].contiguous(),
        "query_tile_tokens": ctx.query_tile_tokens,
        "quant_block_size": ctx.quant_block_size,
        "backend": backend_result["backend"],
        "kernel_compatibility_added_blocks": 0,
        "stats": {
            "total_selected": total_selected,
            "total_candidates": total_candidates,
            "selected_ratio": selected_ratio,
            "sparsity_ratio": sparsity_ratio,
            "selected_blocks_per_row": selected_blocks_per_row,
        },
    }


def _sparge_preprocess_topk_dispatch(
    ctx: _SpargePreprocessContext,
    *,
    backend_preference: str = "auto",
) -> dict[str, Any]:
    from .sparge_preprocess_triton import dispatch_sparge_preprocess_backend

    backend_result = dispatch_sparge_preprocess_backend(
        ctx=ctx,
        torch_backend=lambda: _sparge_preprocess_topk_torch_impl(ctx),
        backend_preference=backend_preference,
    )
    return _finalize_sparge_preprocess_outputs(ctx, backend_result)


def _sparge_preprocess_topk_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    is_causal: bool = False,
    smooth_k: bool = True,
    simthreshd1: float | torch.Tensor = -0.1,
    topk: float | torch.Tensor = 0.5,
    attention_sink: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
) -> dict[str, Any]:
    ctx = _build_sparge_preprocess_context(
        query,
        key,
        is_causal=is_causal,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        topk=topk,
        attention_sink=attention_sink,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
    )
    return _finalize_sparge_preprocess_outputs(ctx, _sparge_preprocess_topk_torch_impl(ctx))


def sparge_block_map_to_mask(
    block_map: torch.Tensor,
    *,
    quant_block_size: int = 64,
    seq_len_q: int | None = None,
    seq_len_kv: int | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    if block_map.dtype != torch.bool or block_map.ndim != 4:
        raise ValueError("block_map must be a 4D bool tensor")
    if quant_block_size <= 0:
        raise ValueError(f"quant_block_size must be positive, got {quant_block_size}")
    batch, heads, q_blocks, kv_blocks = block_map.shape
    full_q = q_blocks * quant_block_size
    full_k = kv_blocks * quant_block_size
    seq_q = full_q if seq_len_q is None else seq_len_q
    seq_kv = full_k if seq_len_kv is None else seq_len_kv
    expanded = block_map.repeat_interleave(quant_block_size, dim=-2).repeat_interleave(quant_block_size, dim=-1)
    expanded = expanded[:, :, :seq_q, :seq_kv]
    mask = torch.full(expanded.shape, -1.0e9, dtype=torch.float32, device=block_map.device)
    mask = torch.where(expanded, torch.zeros_like(mask), mask)
    if is_causal:
        causal = torch.triu(torch.full((seq_q, seq_kv), -1.0e9, dtype=torch.float32, device=block_map.device), diagonal=1)
        mask = torch.minimum(mask, causal.view(1, 1, seq_q, seq_kv))
    return mask.contiguous()


def sparge_preprocess_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    is_causal: bool = False,
    smooth_k: bool = True,
    simthreshd1: float | torch.Tensor = -0.1,
    topk: float | torch.Tensor = 0.5,
    attention_sink: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
) -> dict[str, Any]:
    ctx = _build_sparge_preprocess_context(
        query,
        key,
        is_causal=is_causal,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        topk=topk,
        attention_sink=attention_sink,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
    )
    return _sparge_preprocess_topk_dispatch(ctx)


def sparge_preprocess_topk_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    key_cache: torch.Tensor,
    *,
    smooth_k: bool = True,
    simthreshd1: float | torch.Tensor = -0.1,
    topk: float | torch.Tensor = 0.5,
    attention_sink: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
) -> dict[str, Any]:
    if query.device.type != "xpu":
        raise NotImplementedError("sparge_preprocess_topk_decode is only supported on XPU")
    if quant_block_size != 64:
        raise ValueError("sparge_preprocess_topk_decode currently supports only quant_block_size=64")
    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout, expected_dtype=query.dtype)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=key.dtype)
    Bkc, Hkvc, Skvc, Dkc = _validate_attention_tensor(key_cache, "K_cache", tensor_layout, expected_dtype=key_cache.dtype)
    if Bk != B or Bkc != B:
        raise ValueError("Batch size mismatch between Q/K/K_cache")
    if Hkvc != Hkv:
        raise ValueError("K and K_cache head-count mismatch")
    if Dk != D or Dkc != D:
        raise ValueError("Head dim mismatch between Q/K/K_cache")
    if query.dtype != key.dtype or key_cache.dtype != query.dtype:
        raise ValueError("Q/K/K_cache dtype must match")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q/K/K_cache must be float16 or bfloat16, got {query.dtype}")
    if Sq != 1:
        raise ValueError(f"sparge_preprocess_topk_decode currently supports only seq_len_q == 1, got {Sq}")
    if Skv <= 0 or Skvc <= 0:
        raise ValueError("sparge_preprocess_topk_decode requires non-empty current K and K_cache")
    if Hq % Hkv != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_kv")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    query_hnd = _to_hnd(query, tensor_layout)
    key_hnd = _to_hnd(key, tensor_layout)
    key_cache_hnd = _to_hnd(key_cache, tensor_layout)
    key_total_hnd = torch.cat([key_cache_hnd, key_hnd], dim=2).contiguous()
    total_seq = key_total_hnd.size(2)
    num_k_blocks = (total_seq + quant_block_size - 1) // quant_block_size
    simthreshd1_tensor = _normalize_per_head_hparam(simthreshd1, Hq, query.device, "simthreshd1")
    topk_tensor = _normalize_per_head_hparam(topk, Hq, query.device, "topk").clamp_(0.0, 1.0)

    _, _, q_int8_hnd, q_scale = _pool_sim_and_quant_torch(query_hnd, quant_block_size, simthreshd1_tensor, "HND")
    pooled_q_route, sim_q_route, _, _ = _pool_sim_and_quant_torch(
        query_hnd,
        _query_tile_tokens_for_head_dim(D),
        simthreshd1_tensor,
        "HND",
    )
    key_mean = key_total_hnd.mean(dim=-2, keepdim=True) if smooth_k else None
    pooled_k, sim_kblocks, key_i8_total_hnd, k_scale_total = _pool_sim_and_quant_torch(
        key_total_hnd,
        quant_block_size,
        simthreshd1_tensor[:Hkv],
        "HND",
        key_mean,
    )

    kv_head_index = torch.arange(Hq, device=query.device, dtype=torch.int64) // (Hq // Hkv)
    pooled_k_for_q = pooled_k[:, kv_head_index]
    sim_k_for_q = sim_kblocks[:, kv_head_index]
    pooled_score = torch.matmul(
        pooled_q_route[:, :, :1, :].to(torch.float32),
        pooled_k_for_q.transpose(-1, -2).to(torch.float32),
    )
    pooled_score *= D ** -0.5
    pooled_score = pooled_score.masked_fill(~sim_k_for_q.unsqueeze(-2), -torch.inf)
    pooled_prob = _safe_softmax(pooled_score)
    sorted_prob = torch.sort(pooled_prob, dim=-1, descending=True)
    num_to_select = (
        (topk_tensor.view(1, Hq, 1) * num_k_blocks)
        .to(torch.int64)
        .expand(B, -1, 1)
        .contiguous()
    )
    final_tile_map = torch.zeros_like(pooled_prob, dtype=torch.bool)
    final_tile_map[~sim_k_for_q.unsqueeze(-2)] = True
    final_tile_map = _fill_block_map_torch(final_tile_map, num_to_select, sorted_prob.indices)
    if attention_sink:
        final_tile_map[..., 0] = True

    raw_block_map = final_tile_map.contiguous()
    block_map = raw_block_map
    lut, valid_block_num = _block_map_lut_torch(block_map)
    total_selected, total_candidates, selected_ratio, sparsity_ratio, selected_blocks_per_row = _get_sparse_block_sparsity_stats(
        valid_block_num,
        lut,
        is_causal=False,
    )
    cache_blocks = (Skvc + quant_block_size - 1) // quant_block_size
    cur_blocks = (Skv + quant_block_size - 1) // quant_block_size

    return {
        "query_i8": _from_hnd(q_int8_hnd, tensor_layout),
        "key_i8": _from_hnd(key_i8_total_hnd[:, :, Skvc:, :].contiguous(), tensor_layout),
        "key_cache_i8": _from_hnd(key_i8_total_hnd[:, :, :Skvc, :].contiguous(), tensor_layout),
        "qscale": q_scale.contiguous(),
        "kscale": k_scale_total[:, :, cache_blocks:, :].contiguous(),
        "kscale_cache": k_scale_total[:, :, :cache_blocks, :].contiguous(),
        "lut": lut,
        "valid_block_num": valid_block_num,
        "block_map": block_map,
        "raw_block_map": raw_block_map,
        "tile_block_map": final_tile_map.contiguous(),
        "sim_qblocks": sim_q_route[:, :, :1].contiguous(),
        "sim_kblocks": sim_kblocks.contiguous(),
        "query_tile_tokens": _query_tile_tokens_for_head_dim(D),
        "quant_block_size": quant_block_size,
        "backend": "torch",
        "kernel_compatibility_added_blocks": 0,
        "stats": {
            "total_selected": total_selected,
            "total_candidates": total_candidates,
            "selected_ratio": selected_ratio,
            "sparsity_ratio": sparsity_ratio,
            "selected_blocks_per_row": selected_blocks_per_row,
        },
    }


def sparge_sage2_decode_meansim_topk_xpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: float | None = None,
    smooth_k: bool = True,
    simthreshd1: float | torch.Tensor = -0.1,
    cdfthreshd: float | torch.Tensor | None = None,
    topk: float | torch.Tensor = 0.5,
    pvthreshd: float | torch.Tensor = 50,
    attention_sink: bool = False,
    tensor_layout: str = "HND",
    output_dtype: torch.dtype | None = None,
    return_sparsity: bool = False,
    return_metadata: bool = False,
) -> torch.Tensor | tuple[Any, ...]:
    if query.device.type != "xpu":
        raise NotImplementedError("sparge_sage2_decode_meansim_topk_xpu is only supported on XPU")
    if cdfthreshd is not None:
        raise NotImplementedError("cdfthreshd routing is not implemented yet; use topk for the first slice")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p must be 0.0 for sparge_sage2_decode_meansim_topk_xpu")
    if attn_mask is not None and is_causal:
        raise ValueError("attn_mask and is_causal cannot both be set")
    if output_dtype is not None and output_dtype != value.dtype:
        raise ValueError(f"output_dtype must match value.dtype in the current implementation, got {output_dtype} vs {value.dtype}")
    if pvthreshd not in (None, 50):
        warnings.warn("pvthreshd is not supported by the current ARK sparse kernel and is ignored", stacklevel=2)

    metadata = sparge_preprocess_topk_decode(
        query,
        key,
        key_cache,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        topk=topk,
        attention_sink=attention_sink,
        quant_block_size=64,
        tensor_layout=tensor_layout,
    )
    out = sage_sparse_decode(
        metadata["query_i8"],
        metadata["key_i8"],
        value,
        metadata["key_cache_i8"],
        value_cache,
        metadata["lut"],
        metadata["valid_block_num"],
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        quant_block_size=metadata["quant_block_size"],
        qscale=metadata["qscale"],
        kscale=metadata["kscale"],
        kscale_cache=metadata["kscale_cache"],
        tensor_layout=tensor_layout,
    )
    sparsity_ratio = metadata["stats"]["sparsity_ratio"]
    if return_metadata and return_sparsity:
        return out, sparsity_ratio, metadata
    if return_metadata:
        return out, metadata
    if return_sparsity:
        return out, sparsity_ratio
    return out


def sparge_sage2_attn_meansim_topk_xpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    smooth_k: bool = True,
    simthreshd1: float | torch.Tensor = -0.1,
    cdfthreshd: float | torch.Tensor | None = None,
    topk: float | torch.Tensor = 0.5,
    pvthreshd: float | torch.Tensor = 50,
    attention_sink: bool = False,
    tensor_layout: str = "HND",
    output_dtype: torch.dtype | None = None,
    return_sparsity: bool = False,
    return_metadata: bool = False,
) -> torch.Tensor | tuple[Any, ...]:
    if query.device.type != "xpu":
        raise NotImplementedError("sparge_sage2_attn_meansim_topk_xpu is only supported on XPU")
    if cdfthreshd is not None:
        raise NotImplementedError("cdfthreshd routing is not implemented yet; use topk for the first slice")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p must be 0.0 for sparge_sage2_attn_meansim_topk_xpu")
    if attn_mask is not None and is_causal:
        raise ValueError("attn_mask and is_causal cannot both be set")
    if output_dtype is not None and output_dtype != value.dtype:
        raise ValueError(f"output_dtype must match value.dtype in the current implementation, got {output_dtype} vs {value.dtype}")
    if pvthreshd not in (None, 50):
        warnings.warn("pvthreshd is not supported by the current ARK sparse kernel and is ignored", stacklevel=2)

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout, expected_dtype=query.dtype)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=key.dtype)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout, expected_dtype=value.dtype)
    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if query.dtype != key.dtype:
        raise ValueError("Q and K dtype must match")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q/K must be float16 or bfloat16, got {query.dtype}")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"V must be float16 or bfloat16, got {value.dtype}")

    normalized_mask = _normalize_sparse_mask(attn_mask, B, Sq, Skv, query.device)
    metadata = sparge_preprocess_topk(
        query,
        key,
        is_causal=is_causal,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        topk=topk,
        attention_sink=attention_sink,
        quant_block_size=64,
        tensor_layout=tensor_layout,
    )
    out = sage_sparse(
        metadata["query_i8"],
        metadata["key_i8"],
        value,
        metadata["lut"],
        metadata["valid_block_num"],
        attn_mask=normalized_mask,
        is_causal=is_causal,
        scale=scale,
        quant_block_size=metadata["quant_block_size"],
        qscale=metadata["qscale"],
        kscale=metadata["kscale"],
        tensor_layout=tensor_layout,
    )
    sparsity_ratio = metadata["stats"]["sparsity_ratio"]
    if return_metadata and return_sparsity:
        return out, sparsity_ratio, metadata
    if return_metadata:
        return out, metadata
    if return_sparsity:
        return out, sparsity_ratio
    return out


def sage_dynquant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    quant_block_size: int = 64,
) -> torch.Tensor:
    """SAGE Attention with dynamic INT8 block-wise quantization of Q/K.

    Takes FP16 Q, K, V inputs. Quantizes Q and K to INT8 per-block
    using a fused SYCL kernel, then calls SAGE V1 with INT8 data.
    API is like SDPA but with an extra quant_block_size parameter.

    Args:
        query: [B, Hq, Sq, D] float16
        key:   [B, Hkv, Skv, D] float16
        value: [B, Hkv, Skv, D] float16
        quant_block_size: Number of tokens sharing one INT8 scale.
            E.g. 64 means 64 consecutive tokens share one absmax.
            0 means per-token (block_size=1).

    Returns:
        O: [B, Hq, Sq, D] float16
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sage_dynquant is only supported on XPU")

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape

    # block_size=0 means per-token
    block_size = quant_block_size if quant_block_size > 0 else 1

    # SAGE V1 kernel uses K-tile size=32; quant_block_size must be 1 or >=32
    if block_size != 1 and block_size < 32:
        raise ValueError(
            f"quant_block_size={block_size} is not supported. "
            f"Must be 1 (per-token) or >= 32 (e.g. 32, 64, 128, 256)."
        )

    lib = get_lib(query)
    stream = get_stream(query)

    # Auto-pad Q and K/V seq lengths to be divisible by block_size
    # so sage_dynquant works as a drop-in replacement for SDPA
    def _ceil_div(a, b):
        return (a + b - 1) // b

    Sq_pad = _ceil_div(Sq, block_size) * block_size
    Skv_pad = _ceil_div(Skv, block_size) * block_size
    need_pad_q = Sq_pad != Sq
    need_pad_kv = Skv_pad != Skv

    if need_pad_q:
        pad_q = Sq_pad - Sq
        query = torch.nn.functional.pad(query, (0, 0, 0, pad_q))  # pad S dim with zeros
    if need_pad_kv:
        pad_kv = Skv_pad - Skv
        key = torch.nn.functional.pad(key, (0, 0, 0, pad_kv))
        value = torch.nn.functional.pad(value, (0, 0, 0, pad_kv))

    # Fused block-wise quantization via SYCL kernel
    # Tensor layout: [B, H, S, D] is contiguous → [B*H*S, D] flattened
    # block_size tokens share one scale → num_blocks = B*H*S / block_size
    # For Q: num_rows = B*Hq*Sq_pad, scale shape = [B, Hq, Sq_pad/block_size, 1]
    q_num_rows = B * Hq * Sq_pad
    q_num_blocks = q_num_rows // block_size
    q_i8 = torch.empty_like(query, dtype=torch.int8)
    q_scale = torch.empty(q_num_blocks, dtype=torch.float32, device=query.device)
    lib.sage_dynamic_quant(
        stream,
        query.data_ptr(),
        0,
        q_i8.data_ptr(),
        q_scale.data_ptr(),
        q_num_rows,
        D,
        block_size,
    )
    q_scale = q_scale.reshape(B, Hq, Sq_pad // block_size, 1)

    k_num_rows = B * Hkv * Skv_pad
    k_num_blocks = k_num_rows // block_size
    k_i8 = torch.empty_like(key, dtype=torch.int8)
    k_scale = torch.empty(k_num_blocks, dtype=torch.float32, device=key.device)
    lib.sage_dynamic_quant(
        stream,
        key.data_ptr(),
        0,
        k_i8.data_ptr(),
        k_scale.data_ptr(),
        k_num_rows,
        D,
        block_size,
    )
    k_scale = k_scale.reshape(B, Hkv, Skv_pad // block_size, 1)

    # Call SAGE v1 with matching quant_block_size
    out = sage(
        q_i8,
        k_i8,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        quant_block_size=block_size,
        qscale=q_scale,
        kscale=k_scale,
    )

    # Slice back to original seq length if padded
    if need_pad_q:
        out = out[:, :, :Sq, :]
    return out


def moe_gemm(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MOE GEMM (Mixture of Experts Grouped GEMM).

    Computes grouped GEMM for MOE layers where different experts process
    different numbers of tokens.

    Expects contiguous layouts:
    - activations: [total_tokens, K] (BF16/FP16)
    - weights: [num_experts, K, N] (BF16/FP16, Row major)
    - num_tokens_per_expert: [num_experts] (int32)
    - scales (optional): [num_experts, N] or None

    Returns:
    - outputs: [total_tokens, N] (same dtype as activations)
    """
    if activations.device.type != "xpu":
        raise NotImplementedError("moe_gemm is only supported on XPU")

    if activations.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"activations must be fp16/bf16, got {activations.dtype}")
    if weights.dtype != activations.dtype:
        raise ValueError("weights dtype must match activations dtype")

    if activations.ndim != 2 or weights.ndim != 3:
        raise ValueError("activations must be 2D [total_tokens, K], weights must be 3D [num_experts, K, N]")

    if not activations.is_contiguous() or not weights.is_contiguous():
        raise ValueError("activations and weights must be contiguous")

    if num_tokens_per_expert.dtype != torch.int32:
        num_tokens_per_expert = num_tokens_per_expert.to(torch.int32)

    if not num_tokens_per_expert.is_contiguous():
        num_tokens_per_expert = num_tokens_per_expert.contiguous()

    total_tokens, K = activations.shape
    num_experts, K_w, N = weights.shape  # weights are [num_experts, K, N]

    if K != K_w:
        raise ValueError(f"K dimension mismatch: activations K={K}, weights K={K_w}")

    if num_tokens_per_expert.shape[0] != num_experts:
        raise ValueError(f"num_tokens_per_expert length {num_tokens_per_expert.shape[0]} != num_experts {num_experts}")

    # Validate total tokens
    expected_total = int(num_tokens_per_expert.sum().item())
    if expected_total != total_tokens:
        raise ValueError(f"Sum of num_tokens_per_expert ({expected_total}) != total_tokens ({total_tokens})")

    lib = get_lib(activations)
    stream = get_stream(activations)
    outputs = torch.empty((total_tokens, N), device=activations.device, dtype=activations.dtype)

    scales_ptr = scales.data_ptr() if scales is not None else 0

    lib.moe_gemm(
        stream,
        activations.data_ptr(),
        weights.data_ptr(),
        scales_ptr,
        outputs.data_ptr(),
        cvt_dtype(activations.dtype),
        N,
        K,
        num_tokens_per_expert.data_ptr(),
        num_experts,
    )
    return outputs


def patch_torch_sdpa(*args, **kwargs):
    from .torch_sdpa_patch import patch_torch_sdpa_with_ark

    return patch_torch_sdpa_with_ark(*args, **kwargs)


def unpatch_torch_sdpa():
    from .torch_sdpa_patch import unpatch_torch_sdpa_with_ark

    return unpatch_torch_sdpa_with_ark()


__all__ = ["patch_torch_sdpa", "unpatch_torch_sdpa"]


# -----------------------------------------------------------------------------
# Compatibility layer
#
# Some callers (e.g. auto_round_extension/ark/qlinear.py) historically imported
# this package as a module and expected certain functions to exist at the module
# level (e.g. ark.repack_quantized_weight, ark.woq_linear).
#
# The wrappers below keep backward compatibility for legacy positional call
# styles without changing the compiled extension.
# -----------------------------------------------------------------------------


def check_isa_supported(_isa: str) -> bool:
    # Best-effort: some builds expose ISA checks via native libs; keep safe.
    # Returning False is conservative and avoids misconfiguration.
    return False


def repack_quantized_weight(*args, **kwargs):
    """Repack quantized weights into ARK/BestLA packed format.

    Supports two call styles:

    1) New style (recommended):
       repack_quantized_weight(QB, scaleB, zp, groupsize, compute_type, weight_type, scale_type, asym)

    2) Legacy style used by qlinear.py:
       repack_quantized_weight(QB, scaleB, zp, g_idx, compute_type, weight_type, scale_type, asym, groupsize)
       repack_quantized_weight(QB, scaleB, zp, g_idx, weight_type, compute_type, scale_type, asym, groupsize)
       (g_idx is ignored)
    """

    if kwargs:
        return _repack_quantized_weight_core(**kwargs)

    if len(args) == 8:
        QB, scaleB, zp, groupsize, compute_type, weight_type, scale_type, asym = args
    elif len(args) == 9:
        QB, scaleB, zp, _g_idx, a4, a5, scale_type, asym, groupsize = args
        # Legacy call sites sometimes swap compute_type/weight_type.
        compute_types = {"fp16", "bf16", "fp32", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0"}
        if isinstance(a4, str) and a4 in compute_types:
            compute_type, weight_type = a4, a5
        else:
            weight_type, compute_type = a4, a5
    else:
        raise TypeError("repack_quantized_weight() expects 8 or 9 positional arguments; " f"got {len(args)}")

    # Some native paths may still expect a valid zp pointer even when asym=False.
    if (zp is None) or (isinstance(zp, torch.Tensor) and zp.numel() == 0):
        if not bool(asym):
            k = QB.shape[0]
            n = QB.shape[1]
            zp = torch.zeros((k // int(groupsize), n), dtype=torch.int8, device=QB.device)
        else:
            zp = torch.empty(0, dtype=torch.int8, device=QB.device)

    return _repack_quantized_weight_core(
        QB,
        scaleB,
        zp,
        groupsize,
        compute_type,
        weight_type,
        scale_type,
        asym,
    )


def unpack_weight(
    blob: torch.Tensor,
    out_dtype: torch.dtype,
    n,
    k,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    return _unpack_weight_core(blob, out_dtype, n, k, groupsize, compute_type, weight_type, scale_type, asym)


def packed_weight_size(A: torch.Tensor, n, k, groupsize, compute_type, weight_type, scale_type, asym):
    # Keep signature convenient for Python callers; native library needs a stream.
    lib = get_lib(A)
    stream = get_stream(A)
    ct = cvtstr_dtype(compute_type)
    wt = cvtstr_dtype(weight_type)
    st = cvtstr_dtype(scale_type)
    return lib.packed_weight_size(stream, n, k, groupsize, ct, wt, st, asym)


def woq_linear(
    A: torch.Tensor,
    packed_B: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    compute_type,
    weight_type,
    scale_type,
    asym,
    groupsize=None,
):
    """Linear helper that writes into a preallocated output tensor."""

    if groupsize is None:
        groupsize = A.shape[-1]

    result = woqgemm(
        A,
        packed_B,
        bias,
        out.shape[-1],
        A.shape[-1],
        int(groupsize),
        compute_type,
        weight_type,
        scale_type,
        bool(asym),
    )
    out.copy_(result)
    return out


if __name__ == "__main__":
    print(cpu_lib is None, xpu_lib is None)

    def matmul_test():
        m = n = k = 128
        dt = torch.int8
        device = "cpu"
        has_bias = False
        if dt == torch.int8:
            A = torch.randint(-128, 127, (m, k), dtype=dt, device=device)
            B = torch.randint(-128, 127, (n, k), dtype=dt, device=device)
            C = igemm_s8s8s32(A, B)
            print(C)
        else:
            A = torch.rand(m, k, dtype=dt, device=device) - 0.5
            B = torch.rand(k, n, dtype=dt, device=device) - 0.5
            bias = torch.rand(1, n, dtype=dt, device=device) if has_bias else torch.empty(0)
            C = matmul(A, B, bias)
        ref = torch.matmul(A, B.T)
        if has_bias:
            ref = ref + bias
        dff = abs(C - ref)
        if dt != torch.int8:
            print(dff.max(), dff.mean())
            print(torch.allclose(ref, C, 0.01, 0.1))

    def woq():
        m = n = k = 128
        dt = torch.float32
        device = "cpu"
        A = torch.rand(m, k, dtype=dt, device=device) - 0.5
        bias = torch.rand(1, n, dtype=dt, device=device) + 1000
        B = torch.randint(-128, 127, (n, k), dtype=torch.int8, device=device)
        scaleB = torch.rand(n, 1, dtype=dt, device=device)
        C = woqgemm_s8(A, B, scaleB, bias)
        print(C)
        DB = B.to(dt) * scaleB
        ref = torch.matmul(A, DB.T) + bias
        print(ref)
        dff = abs(C - ref)
        print(dff.max(), dff.mean())

    def pack_unpack():
        m = n = k = 128
        groupsize = 32
        dt = torch.float32
        device = "xpu"
        B = torch.randint(-8, 7, (k, n), dtype=torch.int8, device=device)
        zp = torch.randint(-8, 7, (k // groupsize, n), dtype=torch.int8, device=device)
        scaleB = torch.rand(k // groupsize, n, dtype=dt, device=device) / 100
        blob = repack_quantized_weight(B, scaleB, zp, groupsize, "fp32", "int4", "fp32", False)
        dq = unpack_weight(blob, dt, n, k, groupsize, "fp32", "int4", "fp32", False)
        print(blob, dq)
        scale_re = scaleB.repeat_interleave(repeats=groupsize, dim=0).to(dt)

        DB = B.to(dt) * scale_re
        dff = abs(DB.T - dq)
        print(dff.max(), dff.mean())

    pack_unpack()
