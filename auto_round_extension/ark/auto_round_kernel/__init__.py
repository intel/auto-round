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

from typing import Optional
import torch
import sys


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


def moe_gemm_decode(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    scales: Optional[torch.Tensor] = None,
    zeros: Optional[torch.Tensor] = None,
    weight_bits: int = 4,
    group_size: int = 128,
    asym: bool = False,
) -> torch.Tensor:
    """MoE GEMV optimized for the decode phase.

    Each expert typically processes only 1-2 tokens (top-k routing with
    small batch). Activations must already be gathered/sorted by expert
    (same convention as ``moe_gemm``).

    Args:
        activations: ``[total_tokens, K]`` in fp16 or bf16.
        weights: 3-D tensor ``[E, N, K_packed]``. The accepted layouts are:

            * Unquantized (``weight_bits=16``): ``torch.float16`` / ``torch.bfloat16``
              matching the activations dtype, ``K_packed == K``.
            * Int8 (``weight_bits=8``): ``torch.uint8``, ``K_packed == K``.
              Sym (``asym=False``) reinterprets each byte as signed int8;
              asym (``asym=True``) treats each byte as ``uint8`` with a
              per-group zero-point.
            * Int4 (``weight_bits=4``): ``torch.uint8`` packed,
              ``K_packed == K // 2`` (two 4-bit values per byte; low nibble
              at the lower K index).
            * Int2 (``weight_bits=2``): ``torch.uint8`` packed,
              ``K_packed == K // 4`` (four 2-bit values per byte; field j at
              K index ``4*i + j`` occupies bits 2j and 2j+1 of byte i).
            * FP8 (``torch.float8_e4m3fn`` / ``torch.float8_e5m2``):
              ``K_packed == K``. ``weight_bits`` is ignored; ``asym`` must
              be ``False`` (no zero-points for FP8).
        num_tokens_per_expert: ``[E]`` int32. Sum must equal
            ``activations.shape[0]``.
        scales: ``[E, N, K // group_size]`` in activations dtype. Required
            for all quantized paths (int8/int4/int2/fp8); must be ``None``
            for unquantized weights.
        zeros: ``[E, N, K // group_size]`` in activations dtype. Required
            when ``asym=True`` (int8/int4/int2 only); otherwise ``None``.
        weight_bits: 2, 4, 8, or 16. Ignored when ``weights`` is an FP8
            tensor (the FP8 sub-format is taken from ``weights.dtype``).
        group_size: group along K for quantized weights (default 128).
        asym: if ``True``, weights use unsigned encoding and ``zeros`` must
            be provided. Not supported for FP8.

    Returns:
        outputs: ``[total_tokens, N]`` in the same dtype as activations.
    """
    activations, weights, scales, zeros, num_tokens_per_expert, weight_dtype, total_tokens, N, K, num_experts = (
        _validate_moe_quant_args(
            activations,
            weights,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=weight_bits,
            group_size=group_size,
            asym=asym,
            api_name="moe_gemm_decode",
        )
    )

    lib = get_lib(activations)
    stream = get_stream(activations)
    outputs = torch.empty((total_tokens, N), device=activations.device, dtype=activations.dtype)
    # Scratch buffer mapping each token to its expert id; filled on-device
    # inside the kernel wrapper so we avoid host-device sync.
    expert_id_per_token = torch.empty((total_tokens,), device=activations.device, dtype=torch.int32)

    scales_ptr = scales.data_ptr() if scales is not None else 0
    zeros_ptr = zeros.data_ptr() if zeros is not None else 0

    lib.moe_gemm_decode(
        stream,
        activations.data_ptr(),
        weights.data_ptr(),
        scales_ptr,
        zeros_ptr,
        outputs.data_ptr(),
        expert_id_per_token.data_ptr(),
        cvt_dtype(activations.dtype),
        weight_dtype,
        N,
        K,
        group_size,
        num_tokens_per_expert.data_ptr(),
        num_experts,
        total_tokens,
        bool(asym),
    )
    return outputs


def _validate_moe_quant_args(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    scales: Optional[torch.Tensor],
    zeros: Optional[torch.Tensor],
    weight_bits: int,
    group_size: int,
    asym: bool,
    api_name: str,
):
    """Shared validation/normalisation for quantized MoE entry points.

    Returns a tuple of normalised tensors and dtype/shape metadata used by the
    kernel-call site:
        ``(activations, weights, scales, zeros, num_tokens_per_expert,
           weight_dtype, total_tokens, N, K, num_experts)``.
    """
    if activations.device.type != "xpu":
        raise NotImplementedError(f"{api_name} is only supported on XPU")

    if activations.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"activations must be fp16/bf16, got {activations.dtype}")

    if activations.ndim != 2:
        raise ValueError("activations must be 2D [total_tokens, K]")
    if weights.ndim != 3:
        raise ValueError("weights must be 3D [E, N, K_packed]")

    if not activations.is_contiguous():
        activations = activations.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()

    if num_tokens_per_expert.dtype != torch.int32:
        num_tokens_per_expert = num_tokens_per_expert.to(torch.int32)
    if not num_tokens_per_expert.is_contiguous():
        num_tokens_per_expert = num_tokens_per_expert.contiguous()

    total_tokens, K = activations.shape
    num_experts = weights.shape[0]
    N = weights.shape[1]

    if num_tokens_per_expert.shape[0] != num_experts:
        raise ValueError(f"num_tokens_per_expert length {num_tokens_per_expert.shape[0]} != num_experts {num_experts}")

    # Detect FP8 weight dtype first (overrides weight_bits).
    is_fp8 = weights.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if is_fp8:
        if asym:
            raise ValueError("FP8 weights do not support asym=True")
        if weights.shape[2] != K:
            raise ValueError(f"FP8 weights K dim {weights.shape[2]} != activations K {K}")
        if scales is None:
            raise ValueError("scales is required for FP8 weights")
        if scales.dtype != activations.dtype:
            raise ValueError("scales dtype must match activations dtype")
        if K % group_size != 0:
            raise ValueError("K must be a multiple of group_size")
        expected_scale_shape = (num_experts, N, K // group_size)
        if tuple(scales.shape) != expected_scale_shape:
            raise ValueError(f"scales shape {tuple(scales.shape)} != expected {expected_scale_shape}")
        if zeros is not None:
            raise ValueError("zeros must be None for FP8 weights")
        weight_dtype = ARK_DT.float8_e4m3 if weights.dtype == torch.float8_e4m3fn else ARK_DT.float8_e5m2
        if not scales.is_contiguous():
            scales = scales.contiguous()
    elif weight_bits == 16:
        if weights.dtype != activations.dtype:
            raise ValueError("Unquantized weights must match activations dtype")
        if weights.shape[2] != K:
            raise ValueError(f"Unquantized weights K dim {weights.shape[2]} != activations K {K}")
        weight_dtype = cvt_dtype(activations.dtype)
        if scales is not None or zeros is not None:
            raise ValueError("scales/zeros must be None when weight_bits=16")
    elif weight_bits in (8, 4, 2):
        if weights.dtype != torch.uint8:
            raise ValueError(f"Int{weight_bits} packed weights must be torch.uint8")
        if weight_bits == 8:
            k_packed_expected = K
            k_div = 1
        elif weight_bits == 4:
            k_packed_expected = K // 2
            k_div = 2
        else:  # weight_bits == 2
            k_packed_expected = K // 4
            k_div = 4
        if K % k_div != 0:
            raise ValueError(f"K must be a multiple of {k_div} for weight_bits={weight_bits}")
        if weights.shape[2] != k_packed_expected:
            raise ValueError(
                f"Int{weight_bits} packed weights last dim {weights.shape[2]} must equal K/{k_div} "
                f"({k_packed_expected})"
            )
        if scales is None:
            raise ValueError(f"scales is required for int{weight_bits} weights")
        if scales.dtype != activations.dtype:
            raise ValueError("scales dtype must match activations dtype")
        if K % group_size != 0:
            raise ValueError("K must be a multiple of group_size")
        # Group_size constraints per dtype.
        if weight_bits == 4 and (group_size & 1) != 0:
            raise ValueError("group_size must be even for int4 weights")
        if weight_bits == 2 and (group_size & 3) != 0:
            raise ValueError("group_size must be a multiple of 4 for int2 weights")
        expected_scale_shape = (num_experts, N, K // group_size)
        if tuple(scales.shape) != expected_scale_shape:
            raise ValueError(f"scales shape {tuple(scales.shape)} != expected {expected_scale_shape}")
        if asym:
            if zeros is None:
                raise ValueError("zeros is required when asym=True")
            if zeros.dtype != activations.dtype:
                raise ValueError("zeros dtype must match activations dtype")
            if tuple(zeros.shape) != expected_scale_shape:
                raise ValueError(f"zeros shape {tuple(zeros.shape)} != expected {expected_scale_shape}")
        else:
            if zeros is not None:
                raise ValueError("zeros must be None when asym=False")
        weight_dtype = {8: ARK_DT.int8, 4: ARK_DT.int4, 2: ARK_DT.int2}[weight_bits]
        if not scales.is_contiguous():
            scales = scales.contiguous()
        if asym and not zeros.is_contiguous():
            zeros = zeros.contiguous()
    else:
        raise ValueError(f"Unsupported weight_bits={weight_bits} (supported: 2, 4, 8, 16)")

    if N % 16 != 0:
        raise ValueError(f"N must be a multiple of 16 (got {N})")

    expected_total = int(num_tokens_per_expert.sum().item())
    if expected_total != total_tokens:
        raise ValueError(f"Sum of num_tokens_per_expert ({expected_total}) != total_tokens ({total_tokens})")

    return (activations, weights, scales, zeros, num_tokens_per_expert, weight_dtype, total_tokens, N, K, num_experts)


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


def moe_gemm_prefill(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    scales: Optional[torch.Tensor] = None,
    zeros: Optional[torch.Tensor] = None,
    weight_bits: int = 4,
    group_size: int = 128,
    asym: bool = False,
) -> torch.Tensor:
    """MoE Grouped GEMM optimized for the prefill phase, supporting all weight
    encodings of ``moe_gemm_decode`` (FP16/BF16, INT8 sym/asym, INT4 sym/asym,
    INT2 sym/asym, FP8 E4M3/E5M2).

    The argument shapes/dtypes match :func:`moe_gemm_decode` exactly so the same
    quantized weights/scales/zeros tensors can be re-used between prefill and
    decode without reshaping. Internally, for the quantized paths the kernel
    materialises a ``[E, K, N]`` ``act_dtype`` temporary via an on-device
    dequantization kernel and then dispatches to the existing CUTLASS-SYCL
    Grouped GEMM (``moe_gemm``). Numerical results are bit-identical to
    ``moe_gemm`` applied to the same dequantized weights.

    Args:
        activations: ``[total_tokens, K]`` in fp16 or bf16.
        weights: 3-D tensor; same layout/dtype contract as
            :func:`moe_gemm_decode`. Quantized layouts are ``[E, N, K_packed]``;
            the unquantized fast path (``weight_bits=16``) accepts
            ``[E, N, K]`` -- callers providing already-``[E, K, N]`` weights
            (as ``moe_gemm`` requires) should call ``moe_gemm`` directly.
        num_tokens_per_expert: ``[E]`` int32. Sum must equal
            ``activations.shape[0]``.
        scales: ``[E, N, K // group_size]`` in activations dtype. Required for
            quantized paths; ignored (must be ``None``) for unquantized.
        zeros: ``[E, N, K // group_size]`` in activations dtype, required when
            ``asym=True`` (int8/int4/int2 only).
        weight_bits: 2, 4, 8, or 16. Ignored for FP8 weights.
        group_size: group along K for quantized weights (default 128).
        asym: if ``True``, weights use unsigned encoding; ``zeros`` required.

    Returns:
        outputs: ``[total_tokens, N]`` in the same dtype as activations.
    """
    activations, weights, scales, zeros, num_tokens_per_expert, weight_dtype, total_tokens, N, K, num_experts = (
        _validate_moe_quant_args(
            activations,
            weights,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=weight_bits,
            group_size=group_size,
            asym=asym,
            api_name="moe_gemm_prefill",
        )
    )

    lib = get_lib(activations)
    stream = get_stream(activations)
    outputs = torch.empty((total_tokens, N), device=activations.device, dtype=activations.dtype)

    # Quantized paths need an [E, K, N] act-dtype scratch buffer that the
    # on-device dequant kernel fills before the inner Grouped GEMM consumes
    # it. The unquantized fast path forwards directly through `moe_gemm` and
    # doesn't need scratch (passing 0 is safe -- the C++ wrapper short-circuits
    # before touching the workspace pointer in that case). We allocate the
    # workspace from PyTorch's caching allocator so repeated calls reuse the
    # same memory.
    is_unquantized = (weight_bits == 16) and (weights.dtype == activations.dtype)
    if is_unquantized:
        # `moe_gemm` requires `[E, K, N]` row-major weights; the decode-style
        # `[E, N, K]` weight shape coming through this validator can be
        # transposed into a temporary contiguous `[E, K, N]` view. The
        # workspace serves the same role as the dequant scratch so the
        # on-device path stays uniform.
        dequant_workspace = weights.transpose(1, 2).contiguous()
        weights_ptr = dequant_workspace.data_ptr()
    else:
        # Reuse a persistent `[E, K, N]` workspace across calls with the same
        # (device, dtype, E, K, N). For real MoE prefill workloads the same
        # shape is dispatched on every iteration; allocating a fresh
        # `E*K*N*sizeof(act)` tensor each call adds non-trivial caching-
        # allocator overhead (and, on the small shapes, dominates the
        # quantized GEMM cost). The workspace is kept alive by the cache so
        # we hand the data_ptr() to the kernel without taking a new ref.
        dequant_workspace = _get_moe_prefill_workspace(activations.device, activations.dtype, num_experts, K, N)
        weights_ptr = weights.data_ptr()

    scales_ptr = scales.data_ptr() if scales is not None else 0
    zeros_ptr = zeros.data_ptr() if zeros is not None else 0

    lib.moe_gemm_prefill(
        stream,
        activations.data_ptr(),
        weights_ptr,
        scales_ptr,
        zeros_ptr,
        outputs.data_ptr(),
        dequant_workspace.data_ptr(),
        cvt_dtype(activations.dtype),
        weight_dtype,
        N,
        K,
        group_size,
        num_tokens_per_expert.data_ptr(),
        num_experts,
        total_tokens,
        bool(asym),
    )
    # The inner CUTLASS-SYCL `moe_gemm` calls `event.wait()` before returning
    # (see `moe_detail::moe_gemm_launcher` in `sycl_tla_moe.hpp`), so by the
    # time `lib.moe_gemm_prefill` returns the device has already consumed the
    # workspace. For the unquantized fast path the workspace is a per-call
    # transposed copy of `weights` -- drop it now. For the quantized paths
    # the workspace lives in the module-level cache (`_get_moe_prefill_workspace`)
    # and is intentionally retained for reuse on the next call.
    if is_unquantized:
        del dequant_workspace
    return outputs


# ---------------------------------------------------------------------------
# `moe_gemm_prefill` dequant-workspace cache.
#
# The Stage-1 quantized prefill kernel dequantises weights into an
# `[E, K, N]` act-dtype scratch buffer before dispatching to the existing
# CUTLASS-SYCL grouped GEMM. In real model usage the same `(E, K, N, dtype)`
# tuple is hit on every prefill step, so allocating a fresh
# `E * K * N * sizeof(act_dtype)` tensor per call adds caching-allocator
# overhead that is significant on the small/medium shapes.
#
# We cache one tensor per `(device, dtype, E, K, N)` key. The cache holds
# references that keep the tensors alive across calls; callers can clear it
# explicitly via `clear_moe_prefill_workspace_cache()` if they need to
# release the memory (e.g., before allocating large buffers for a different
# subsystem).
# ---------------------------------------------------------------------------

_MOE_PREFILL_WORKSPACE_CACHE: "dict[tuple, torch.Tensor]" = {}


def _get_moe_prefill_workspace(device: torch.device, dtype: torch.dtype, E: int, K: int, N: int) -> torch.Tensor:
    """Return a persistent `[E, K, N]` workspace tensor for the prefill kernel.

    The tensor is allocated lazily on first use and retained in a module-level
    cache so subsequent calls with the same `(device, dtype, E, K, N)` reuse
    the same memory. Returned tensors are contiguous and uninitialised; the
    kernel writes every element before reading.
    """
    # `device` may be a `torch.device` or a string; normalise so the cache key
    # is hashable and identifies the exact device (including ordinal).
    if not isinstance(device, torch.device):
        device = torch.device(device)
    key = (device.type, device.index, dtype, int(E), int(K), int(N))
    ws = _MOE_PREFILL_WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = torch.empty((E, K, N), device=device, dtype=dtype)
        _MOE_PREFILL_WORKSPACE_CACHE[key] = ws
    return ws


def clear_moe_prefill_workspace_cache() -> None:
    """Release all cached `moe_gemm_prefill` dequant-workspace tensors."""
    _MOE_PREFILL_WORKSPACE_CACHE.clear()


# ---------------------------------------------------------------------------
# Unified MoE entry point
#
# `moe_gemm_decode` and `moe_gemm_prefill` accept identical argument shapes
# and dtypes -- the only difference is which underlying SYCL kernel is
# launched (a GEMV variant tuned for 1-2 tokens/expert vs. a Grouped GEMM
# variant tuned for many tokens/expert). Model code that runs through both
# regimes (prefill of a prompt, then autoregressive decode) traditionally
# has to keep two call sites and branch on phase. `moe(...)` collapses that
# into a single API and auto-selects the right kernel from the token
# distribution.
#
# Callers that already know the phase (e.g., a model's generation loop knows
# whether it's in prefill or decode) should pass it via the `phase` argument
# to avoid the small host-device sync that `phase="auto"` needs to inspect
# `num_tokens_per_expert.max()`.
# ---------------------------------------------------------------------------

# Default tokens-per-expert threshold used by `phase="auto"`. The decode
# GEMV kernel is faster when every expert sees only a handful of tokens
# (TopK >= 1 with batch size 1-4); above that the GEMM-tuned prefill kernel
# wins. The crossover is hardware-dependent but `4` is a conservative default
# that matches the regime `moe_gemm_decode`'s docstring describes
# ("typically only 1-2 tokens", up to top-k * small batch).
_MOE_AUTO_DECODE_MAX_TOKENS_PER_EXPERT = 4

_MOE_VALID_PHASES = ("auto", "decode", "prefill")


def moe(
    activations: torch.Tensor,
    weights: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    scales: Optional[torch.Tensor] = None,
    zeros: Optional[torch.Tensor] = None,
    weight_bits: int = 4,
    group_size: int = 128,
    asym: bool = False,
    phase: str = "auto",
    decode_threshold: int = _MOE_AUTO_DECODE_MAX_TOKENS_PER_EXPERT,
) -> torch.Tensor:
    """Unified MoE GEMM entry point that dispatches to decode or prefill.

    This is a thin Python-side dispatcher over :func:`moe_gemm_decode` and
    :func:`moe_gemm_prefill`. The two underlying kernels accept the same
    argument shapes/dtypes (see :func:`moe_gemm_decode` for the full layout
    contract); ``moe`` simply picks the one that is faster for the current
    token distribution so model code can have a single call site for both
    prefill and decode phases.

    Args:
        activations: ``[total_tokens, K]`` in fp16 or bf16.
        weights: ``[E, N, K_packed]`` -- see :func:`moe_gemm_decode` for the
            quant-specific layout/dtype contract.
        num_tokens_per_expert: ``[E]`` int32. Sum must equal
            ``activations.shape[0]``.
        scales, zeros, weight_bits, group_size, asym: forwarded to the
            underlying kernel; see :func:`moe_gemm_decode`.
        phase: dispatch mode.

            * ``"auto"`` (default): inspect ``num_tokens_per_expert.max()``
              and pick decode if every expert sees ``<= decode_threshold``
              tokens, otherwise prefill. This incurs one small host-device
              sync per call.
            * ``"decode"``: always dispatch to :func:`moe_gemm_decode`. Use
              when the model's generation loop already knows it is in the
              decode phase; avoids the sync.
            * ``"prefill"``: always dispatch to :func:`moe_gemm_prefill`.
              Use when the model knows it is in the prefill phase.
        decode_threshold: ``"auto"`` mode dispatches to decode when
            ``num_tokens_per_expert.max() <= decode_threshold``. Defaults to
            4 (the regime the decode GEMV kernel is tuned for).

    Returns:
        ``[total_tokens, N]`` in the activations dtype. Bit-identical to the
        underlying kernel that was dispatched.
    """
    if phase not in _MOE_VALID_PHASES:
        raise ValueError(f"phase must be one of {_MOE_VALID_PHASES}, got {phase!r}")

    if phase == "auto":
        # `.max().item()` triggers a host-device sync; callers in tight
        # decode loops should pass `phase="decode"` explicitly to skip this.
        # We tolerate a non-int32 / non-contiguous tensor here because the
        # downstream kernel wrappers will normalise it anyway.
        if num_tokens_per_expert.numel() == 0:
            raise ValueError("num_tokens_per_expert must be non-empty")
        max_tpe = int(num_tokens_per_expert.max().item())
        phase = "decode" if max_tpe <= int(decode_threshold) else "prefill"

    if phase == "decode":
        return moe_gemm_decode(
            activations,
            weights,
            num_tokens_per_expert,
            scales=scales,
            zeros=zeros,
            weight_bits=weight_bits,
            group_size=group_size,
            asym=asym,
        )
    # phase == "prefill"
    return moe_gemm_prefill(
        activations,
        weights,
        num_tokens_per_expert,
        scales=scales,
        zeros=zeros,
        weight_bits=weight_bits,
        group_size=group_size,
        asym=asym,
    )


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
