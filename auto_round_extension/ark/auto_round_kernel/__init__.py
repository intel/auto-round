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
from typing import Optional
import torch

# Intel GPU compiler (IGC) environment variable: ensure implicit local IDs are
# not removed by the compiler even when unused. This guarantees consistent
# behavior across driver upgrades and prevents performance regressions.
# Reference: https://github.com/intel/intel-graphics-compiler/issues/412
os.environ.setdefault("IGC_RemoveUnusedIdImplicitLocalIDs", "0")

# Tensor layout constants passed to native C++ backends.
# 0 = HND ([B, H, S, D]), 1 = NHD ([B, S, H, D]).
LAYOUT_HND = 0
LAYOUT_NHD = 1


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


def _validate_canonical_strides(tensor: torch.Tensor, name: str, tensor_layout: str) -> None:
    """Verify that *batched* 4-D tensor strides match the canonical packed layout.

    C++ backends compute strides from layout + shape alone (no stride params),
    so the input tensors must have the exact canonical strides.

    ``tensor.stride()`` returns strides in dimension order:

        HND [B, H, S, D]:  stride=(H*S*D, S*D, D, 1)
        NHD [B, S, H, D]:  stride=(S*H*D, H*D, D, 1)
    """
    layout = _normalize_tensor_layout(tensor_layout)
    if layout == "HND":
        B, H, S, D = tensor.shape
        expected = (H * S * D, S * D, D, 1)
    else:
        B, S, H, D = tensor.shape
        expected = (S * H * D, H * D, D, 1)
    actual = tensor.stride()
    if actual != expected:
        raise ValueError(
            f"{name} strides {actual} do not match canonical {layout} layout {expected}. "
            f"Non-contiguous inputs are not supported; call .contiguous() first."
        )


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


# A: mxk,  B: nxk, bias: n or [1, n]
def matmul_sycl_tla(A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None):
    if A.device.type != "xpu" or B.device.type != "xpu":
        raise NotImplementedError("matmul_sycl_tla is only supported on XPU")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D tensors")
    if A.device != B.device:
        raise ValueError("A and B must be on the same device")
    if A.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("matmul_sycl_tla only supports torch.float16 and torch.bfloat16")
    if B.dtype != A.dtype:
        raise ValueError("A and B must have the same dtype")

    m, k = A.shape
    n, kb = B.shape
    if k != kb:
        raise ValueError(f"Shape mismatch: A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}")

    lib = get_lib(A)
    if lib is None or not hasattr(lib, "matmul_sycl_tla"):
        raise NotImplementedError("Current XPU build does not expose matmul_sycl_tla")

    A_arg = A.contiguous()
    B_arg = B.contiguous()
    C = torch.empty(m, n, dtype=A.dtype, device=A.device)

    bias_ptr = 0
    bias_arg = None
    if bias is not None and bias.numel() > 0:
        bias_arg = bias.to(dtype=C.dtype, device=A.device).contiguous().view(-1)
        if bias_arg.numel() != n:
            raise ValueError(f"bias must have {n} elements, got {bias_arg.numel()}")
        bias_ptr = bias_arg.data_ptr()

    stream = get_stream(A_arg)
    lib.matmul_sycl_tla(
        stream,
        m,
        n,
        k,
        A_arg.data_ptr(),
        cvt_dtype(A_arg.dtype),
        B_arg.data_ptr(),
        cvt_dtype(B_arg.dtype),
        C.data_ptr(),
        cvt_dtype(C.dtype),
        bias_ptr,
        True,
    )
    return C


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
    _validate_packed_blob(B, n, k, groupsize, compute_type, weight_type, scale_type, asym)
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
        B.numel(),
    )
    return C


def _validate_packed_blob(
    blob: torch.Tensor,
    n: int,
    k: int,
    groupsize: int,
    compute_type: str,
    weight_type: str,
    scale_type: str,
    asym: bool,
) -> None:
    """Validate a packed-weight blob before passing it to native code.

    Raises ``TypeError`` or ``ValueError`` if any check fails, preventing
    out-of-bounds memory access in the native ``unpack_weight`` /
    ``woqgemm`` path when the blob is malformed or the parameters are
    inconsistent with the blob contents.
    """
    if not isinstance(blob, torch.Tensor):
        raise TypeError(f"blob must be a torch.Tensor, got {type(blob).__name__}")

    if blob.device.type not in ("cpu", "xpu"):
        raise ValueError(f"blob must reside on cpu or xpu, got {blob.device.type}")

    if blob.dtype != torch.int8:
        raise ValueError(f"blob must have dtype torch.int8, got {blob.dtype}")

    if blob.dim() != 1:
        raise ValueError(f"blob must be a 1-D tensor, got {blob.dim()}-D")

    if not isinstance(n, (int,)) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(k, (int,)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if not isinstance(groupsize, (int,)) or groupsize <= 0:
        raise ValueError(f"groupsize must be a positive integer, got {groupsize!r}")
    if not isinstance(asym, bool):
        raise TypeError(f"asym must be a bool, got {type(asym).__name__}")

    valid_types = {
        "fp32",
        "fp16",
        "bf16",
        "int8",
        "int4",
        "int2",
        "int3",
        "int5",
        "int6",
        "int7",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e8m0",
    }
    valid_compute_types = valid_types | {"auto"}
    if compute_type not in valid_compute_types:
        raise ValueError(f"compute_type must be one of {valid_compute_types}, got {compute_type!r}")
    if weight_type not in valid_types:
        raise ValueError(f"weight_type must be one of {valid_types}, got {weight_type!r}")
    if scale_type not in valid_types:
        raise ValueError(f"scale_type must be one of {valid_types}, got {scale_type!r}")


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
    if not isinstance(QB, torch.Tensor) or QB.dim() != 2:
        raise ValueError(f"QB must be a 2-D tensor, got shape {QB.shape!r}")
    if not isinstance(scaleB, torch.Tensor) or scaleB.dim() != 2:
        raise ValueError(f"scaleB must be a 2-D tensor, got shape {scaleB.shape!r}")
    if not isinstance(zp, torch.Tensor):
        raise TypeError(f"zp must be a torch.Tensor, got {type(zp).__name__}")
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
    _validate_packed_blob(blob, n, k, groupsize, compute_type, weight_type, scale_type, asym)
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
        blob.numel(),
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
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention (SDPA) prefill+decode.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - scale: Softmax scale. Uses 1 / sqrt(D) when None.
    - tensor_layout: Layout of Q/K/V/O tensors.
    - return_lse: If True, returns (O, LSE) where LSE[b, h, q] = log(sum_j exp(score_{b,h,q,j})).

    Returns:
    - O: same layout as the input tensors.
    - (O, LSE): if return_lse is True.
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

    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)

    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )

    LSE = torch.empty(B, Hq, Sq, dtype=torch.float32, device=query.device) if return_lse else None

    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sdpa(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
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
        layout_code,
        LSE.data_ptr() if LSE is not None else 0,
    )

    if return_lse:
        return O, LSE
    return O


def sdpa_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention (SDPA) prefill+decode with variable-length sequences.

    Q, K, V use a flat 3-D layout where tokens from all sequences in the batch
    are concatenated along dim-0 (the *total* dimension).  Per-sequence
    boundaries are given by ``cu_seqlens_q`` / ``cu_seqlens_k``.

    Shapes:
        Q:   [total_q, num_heads_q, head_dim]
        K:   [total_kv, num_heads_kv, head_dim]
        V:   [total_kv, num_heads_kv, head_dim]
        O:   [total_q, num_heads_q, head_dim]

    where ``total_q = cu_seqlens_q[-1]`` and ``total_kv = cu_seqlens_k[-1]``.

    Args:
        cu_seqlens_q: Cumulative sequence lengths for Q  [batch_size + 1] int32.
        cu_seqlens_k: Cumulative sequence lengths for K/V  [batch_size + 1] int32.
        max_seqlen_q: Maximum Q sequence length in the batch.
        max_seqlen_k: Maximum K/V sequence length in the batch.
        scale: Softmax scale.  Uses ``1 / sqrt(head_dim)`` when ``None``.

    Returns:
        O: Flat output with shape [total_q, num_heads_q, head_dim].
    """
    if query.device.type != "xpu":
        raise NotImplementedError("sdpa_varlen is only supported on XPU")

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"K/V dtype must match Q dtype, got K={key.dtype}, V={value.dtype}, Q={query.dtype}")

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(f"Q/K/V must be 3-D for varlen; got Q={query.ndim}D, K={key.ndim}D, V={value.ndim}D")

    if cu_seqlens_q.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens_q must be int32 or int64, got {cu_seqlens_q.dtype}")
    if cu_seqlens_k.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens_k must be int32 or int64, got {cu_seqlens_k.dtype}")
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1-D")
    if cu_seqlens_q.device.type != "xpu" or cu_seqlens_k.device.type != "xpu":
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be on XPU")

    batch = cu_seqlens_q.shape[0] - 1
    if cu_seqlens_k.shape[0] - 1 != batch:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")

    total_q, Hq, D = query.shape
    total_kv, Hkv, Dk = key.shape
    total_kv_v, Hkv2, Dv = value.shape

    if total_q != cu_seqlens_q[-1].item():
        raise ValueError(f"Q dim-0 ({total_q}) != cu_seqlens_q[-1] ({cu_seqlens_q[-1].item()})")
    if total_kv != cu_seqlens_k[-1].item():
        raise ValueError(f"K dim-0 ({total_kv}) != cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()})")
    if total_kv_v != cu_seqlens_k[-1].item():
        raise ValueError(f"V dim-0 ({total_kv_v}) != cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()})")
    if Hkv != Hkv2 or Dk != Dv:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128, 96, 192):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128, 96, 192")

    if dropout_p != 0.0:
        raise NotImplementedError(f"dropout_p must be 0.0 (got {dropout_p}); dropout is not supported")

    if attn_mask is not None:
        raise NotImplementedError("attn_mask is not yet supported in sdpa_varlen")

    if Hq % Hkv != 0:
        raise ValueError(f"num_heads_q ({Hq}) must be divisible by num_heads_kv ({Hkv})")

    # Validate strides: the dim axis must be contiguous.
    # Flat layout [total, H, D]: dim-stride (stride(2)) must be 1.
    if query.stride(2) != 1:
        raise ValueError(f"Q must be contiguous along head-dim axis; got stride {query.stride()}")
    if key.stride(2) != 1:
        raise ValueError(f"K must be contiguous along head-dim axis; got stride {key.stride()}")
    if value.stride(2) != 1:
        raise ValueError(f"V must be contiguous along head-dim axis; got stride {value.stride()}")

    lib = get_lib(query)
    stream = get_stream(query)
    cu_seqlens_q_i32 = cu_seqlens_q.contiguous().to(torch.int32)
    cu_seqlens_k_i32 = cu_seqlens_k.contiguous().to(torch.int32)

    O = torch.empty(total_q, Hq, D, dtype=value.dtype, device=query.device)

    if return_lse:
        max_q = int((cu_seqlens_q_i32[1:] - cu_seqlens_q_i32[:-1]).max().item())
        if max_seqlen_q < max_q:
            raise ValueError(f"max_seqlen_q ({max_seqlen_q}) < max sequence length in cu_seqlens_q ({max_q})")
        LSE = torch.full(
            (batch, Hq, max_seqlen_q),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
    else:
        LSE = None
    lib.sdpa_varlen(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        cvt_dtype(query.dtype),
        cvt_dtype(key.dtype),
        cvt_dtype(O.dtype),
        batch,
        Hq,
        Hkv,
        total_q,
        total_kv,
        max_seqlen_q,
        max_seqlen_k,
        D,
        float(scale) if scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        cu_seqlens_q_i32.data_ptr(),
        cu_seqlens_k_i32.data_ptr(),
        LAYOUT_HND,  # unused by the native varlen path (always flat 3-D)
        LSE.data_ptr() if LSE is not None else 0,
    )

    if return_lse:
        return O, LSE
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

    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)

    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
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
        layout_code,
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

    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)

    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=out_dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )
    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
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
        layout_code,
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
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SAGE v1 attention prefill+decode.

    Supported tensor layouts:
    - HND: [B, H, N, D]
    - NHD: [B, N, H, D]

    Args:
    - scale: Attention scale. Uses 1 / sqrt(D) when None.
    - quant_block_size: Quantization block size used by the kernel.
    - tensor_layout: Layout of Q/K/V/O tensors.
    - return_lse: If True, returns (O, LSE) where LSE[b, h, q] = log(sum_j exp(score_{b,h,q,j})).

    Returns:
    - O: same layout as the input tensors.
    - (O, LSE): if return_lse is True.
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
            return_lse=return_lse,
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

    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)

    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )

    LSE = torch.empty(B, Hq, Sq, dtype=torch.float32, device=query.device) if return_lse else None

    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sagev1(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
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
        layout_code,
        LSE.data_ptr() if LSE is not None else 0,
    )

    if return_lse:
        return O, LSE
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
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            return_lse=return_lse,
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

    _validate_canonical_strides(query, "Q", tensor_layout)
    _validate_canonical_strides(key, "K", tensor_layout)
    _validate_canonical_strides(value, "V", tensor_layout)

    O = _empty_attention_output(
        B,
        Hq,
        Sq,
        D,
        dtype=value.dtype,
        device=query.device,
        tensor_layout=tensor_layout,
    )

    LSE = torch.empty(B, Hq, Sq, dtype=torch.float32, device=query.device) if return_lse else None

    layout_code = LAYOUT_HND if _normalize_tensor_layout(tensor_layout) == "HND" else LAYOUT_NHD
    lib.sagev1_pvi8(
        stream,
        query.data_ptr(),
        key.data_ptr(),
        value.data_ptr(),
        O.data_ptr(),
        attn_mask.data_ptr() if attn_mask is not None else 0,
        quant_block_size,
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
        layout_code,
        LSE.data_ptr() if LSE is not None else 0,
    )

    if return_lse:
        return O, LSE
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SAGE attention dispatcher.

    Signature mirrors ``sageattention.sageattn``.

    Args:
    - q, k, v: Query/Key/Value tensors. Layout selected by ``tensor_layout``.
    - tensor_layout: "HND" or "NHD".
    - is_causal: Whether to apply causal mask.
    - sm_scale: Softmax scale. Uses ``1 / sqrt(head_dim)`` when None.
    - return_lse: If True, returns (O, LSE) tuple.
    - kernel: Which SAGE variant to dispatch to.
        - "v1_pvhalf" (default): PV in half precision (calls ``sagev1``).
        - "v1_pvi8": PV in INT8 precision (calls ``sagev1_pvi8``).
    - kwargs: Forwarded to the underlying kernel (e.g. ``attn_mask``,
      ``dropout_p``, ``enable_gqa``, ``quant_block_size``).

    Returns:
    - O: same layout as the input tensors.
    - (O, LSE): if return_lse is True.
    """
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
        return_lse=return_lse,
        **kwargs,
    )


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: float | None = None,
    kernel: str = "v1_pvhalf",
    quant_block_size: int = 64,
    return_lse: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """SAGE attention with variable-length sequences (no padding).

    Q/K/V are flat 3-D tensors: [total_tokens, num_heads, head_dim].
    Per-sequence boundaries given by ``cu_seqlens_q`` / ``cu_seqlens_k``.

    Internally quantizes Q/K (and V for pvi8) to INT8, then dispatches to
    the SAGE varlen kernel.

    Args:
        q:   [total_q, num_heads_q, head_dim] float16/bfloat16
        k:   [total_kv, num_heads_kv, head_dim] float16/bfloat16
        v:   [total_kv, num_heads_kv, head_dim] float16/bfloat16
        cu_seqlens_q: Cumulative sequence lengths for Q [batch+1] int32.
        cu_seqlens_k: Cumulative sequence lengths for K/V [batch+1] int32.
        max_seqlen_q: Maximum Q sequence length.
        max_seqlen_k: Maximum K/V sequence length.
        is_causal: Apply causal mask.
        sm_scale: Softmax scale. Uses 1/sqrt(D) when None.
        kernel: "v1_pvhalf" (PV half) or "v1_pvi8" (PV int8).
        quant_block_size: Block size for INT8 quantization (default 64).
        **kwargs: Forwarded (attn_mask, dropout_p etc. not yet supported).

    Returns:
        O: [total_q, num_heads_q, head_dim] float16/bfloat16
    """
    if q.device.type != "xpu":
        raise NotImplementedError("sageattn_varlen is only supported on XPU")

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {q.dtype}")

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"Q/K/V must be 3-D for varlen; got Q={q.ndim}D, K={k.ndim}D, V={v.ndim}D")

    if cu_seqlens_q.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens_q must be int32 or int64, got {cu_seqlens_q.dtype}")
    if cu_seqlens_k.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens_k must be int32 or int64, got {cu_seqlens_k.dtype}")
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1-D")
    if cu_seqlens_q.device.type != "xpu" or cu_seqlens_k.device.type != "xpu":
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be on XPU")

    batch = cu_seqlens_q.shape[0] - 1
    if cu_seqlens_k.shape[0] - 1 != batch:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")

    total_q, Hq, D = q.shape
    total_kv, Hkv, Dk = k.shape
    total_kv_v, Hkv2, Dv = v.shape

    if total_q != cu_seqlens_q[-1].item():
        raise ValueError(f"Q dim-0 ({total_q}) != cu_seqlens_q[-1] ({cu_seqlens_q[-1].item()})")
    if total_kv != cu_seqlens_k[-1].item():
        raise ValueError(f"K dim-0 ({total_kv}) != cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()})")
    if total_kv_v != cu_seqlens_k[-1].item():
        raise ValueError(f"V dim-0 ({total_kv_v}) != cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()})")
    if Hkv != Hkv2 or Dk != Dv:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; only 64, 128 supported for SAGE")

    if kernel not in ("v1_pvhalf", "v1_pvi8"):
        raise ValueError(f"Unsupported kernel={kernel!r}; supported: 'v1_pvhalf', 'v1_pvi8'")

    if quant_block_size <= 0:
        quant_block_size = 1

    use_int8_pv = 1 if kernel == "v1_pvi8" else 0

    # Validate contiguous along head-dim
    if q.stride(2) != 1:
        raise ValueError(f"Q must be contiguous along head-dim axis; got stride {q.stride()}")
    if k.stride(2) != 1:
        raise ValueError(f"K must be contiguous along head-dim axis; got stride {k.stride()}")
    if v.stride(2) != 1:
        raise ValueError(f"V must be contiguous along head-dim axis; got stride {v.stride()}")

    lib = get_lib(q)
    stream = get_stream(q)
    cu_seqlens_q_i32 = cu_seqlens_q.contiguous().to(torch.int32)
    cu_seqlens_k_i32 = cu_seqlens_k.contiguous().to(torch.int32)

    O = torch.empty(total_q, Hq, D, dtype=v.dtype, device=q.device)

    if return_lse:
        max_q = int((cu_seqlens_q_i32[1:] - cu_seqlens_q_i32[:-1]).max().item())
        if max_seqlen_q < max_q:
            raise ValueError(f"max_seqlen_q ({max_seqlen_q}) < max sequence length in cu_seqlens_q ({max_q})")
        LSE = torch.full(
            (batch, Hq, max_seqlen_q),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        LSE = None
    lib.sagev1_varlen(
        stream,
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        O.data_ptr(),
        0,  # mask
        quant_block_size,
        cvt_dtype(q.dtype),
        cvt_dtype(k.dtype),
        cvt_dtype(v.dtype),
        cvt_dtype(O.dtype),
        batch,
        Hq,
        Hkv,
        total_q,
        total_kv,
        max_seqlen_q,
        max_seqlen_k,
        D,
        sm_scale if sm_scale is not None else 1.0 / (D**0.5),
        bool(is_causal),
        cu_seqlens_q_i32.data_ptr(),
        cu_seqlens_k_i32.data_ptr(),
        use_int8_pv,
        LSE.data_ptr() if LSE is not None else 0,
    )

    if return_lse:
        return O, LSE
    return O


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


def woqgemm_linear(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None,
    n,
    k,
    groupsize,
    compute_type,
    weight_type,
    scale_type,
    asym,
):
    n = int(n)
    k = int(k)
    groupsize = int(groupsize)
    asym = bool(asym)

    if A.shape[-1] != k:
        raise ValueError(f"k must match A.shape[-1] ({A.shape[-1]}), got {k}")

    raw_input_dtype = A.dtype
    target_dtype = torch.float16 if A.device.type == "xpu" else torch.float32
    out_shape = A.shape[:-1] + (n,)
    A_2d = A.to(target_dtype).reshape(-1, A.shape[-1])
    if bias is None or bias.numel() == 0:
        bias = torch.empty(0, dtype=target_dtype, device=A.device)
    else:
        bias = bias.to(device=A.device, dtype=target_dtype)

    out = woqgemm(
        A_2d,
        B,
        bias,
        n,
        k,
        groupsize,
        compute_type,
        weight_type,
        scale_type,
        asym,
    )

    return out.to(raw_input_dtype).reshape(out_shape)


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
