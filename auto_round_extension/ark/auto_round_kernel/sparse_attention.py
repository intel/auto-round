# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch

from . import (
    _attention_strides_qko,
    _attention_strides_v,
    _empty_attention_output,
    _normalize_tensor_layout,
    _validate_attention_tensor,
    cvt_dtype,
    get_lib,
    get_stream,
)
from . import sagev1 as _dense_sagev1
from . import sagev1_pvi8 as _dense_sagev1_pvi8
from . import (
    sdpa,
)


def _get_xpu_sparse_kernel_backend() -> str:
    backend = os.getenv("SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND", "sycl").strip().lower()
    if backend != "sycl":
        raise ValueError("Unsupported SAGE_ATTN_XPU_SPARSE_KERNEL_BACKEND=" f"{backend!r}; expected: sycl")
    return backend


def _get_sparse_preprocess_backend_preference() -> str:
    if _prefix_protection_requested():
        return "torch"
    return os.getenv("SAGE_ATTN_SPARSE_PREPROCESS_BACKEND", "auto").strip().lower()


def _validate_gqa_head_config(num_heads_q: int, num_heads_kv: int, *, op_name: str) -> None:
    if num_heads_q <= 0 or num_heads_kv <= 0:
        raise ValueError(f"{op_name} requires positive num_heads_q/num_heads_kv, got {num_heads_q} and {num_heads_kv}")
    if num_heads_q < num_heads_kv:
        raise ValueError(
            f"{op_name} requires num_heads_q >= num_heads_kv for MHA/GQA, got {num_heads_q} and {num_heads_kv}"
        )
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"{op_name} requires num_heads_q to be divisible by num_heads_kv, got {num_heads_q} and {num_heads_kv}"
        )


def _kv_head_index_for_q_heads(num_heads_q: int, num_heads_kv: int, device: torch.device) -> torch.Tensor:
    _validate_gqa_head_config(num_heads_q, num_heads_kv, op_name="sparse attention")
    return torch.arange(num_heads_q, device=device, dtype=torch.int64) // (num_heads_q // num_heads_kv)


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
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
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
    if q_tile_override not in (0, 64, 128, 256):
        raise ValueError(f"Unsupported q_tile_override={q_tile_override}; supported values: 0, 64, 128, 256")
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
    _validate_gqa_head_config(Hq, Hkv, op_name="sage_sparse")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")
    _validate_sparse_q_tile_override_for_head_dim(D, q_tile_override)

    effective_sparse_q_block_tokens = quant_block_size if sparse_q_block_tokens is None else int(sparse_q_block_tokens)
    effective_sparse_k_block_tokens = quant_block_size if sparse_k_block_tokens is None else int(sparse_k_block_tokens)
    if effective_sparse_q_block_tokens <= 0 or effective_sparse_k_block_tokens <= 0:
        raise ValueError(
            "sparse_q_block_tokens and sparse_k_block_tokens must be positive when provided; "
            f"got {effective_sparse_q_block_tokens} and {effective_sparse_k_block_tokens}"
        )

    q_scale_blocks = (Sq + quant_block_size - 1) // quant_block_size
    kv_scale_blocks = (Skv + quant_block_size - 1) // quant_block_size
    q_sparse_blocks = (Sq + effective_sparse_q_block_tokens - 1) // effective_sparse_q_block_tokens
    kv_sparse_blocks = (Skv + effective_sparse_k_block_tokens - 1) // effective_sparse_k_block_tokens
    decoupled_sparse_rows = (
        effective_sparse_q_block_tokens != quant_block_size or effective_sparse_k_block_tokens != quant_block_size
    )
    if tuple(lut.shape) != (B, Hq, q_sparse_blocks, kv_sparse_blocks):
        raise ValueError(f"lut must have shape {(B, Hq, q_sparse_blocks, kv_sparse_blocks)}, got {tuple(lut.shape)}")
    if tuple(valid_block_num.shape) != (B, Hq, q_sparse_blocks):
        raise ValueError(
            f"valid_block_num must have shape {(B, Hq, q_sparse_blocks)}, got {tuple(valid_block_num.shape)}"
        )
    if qscale.numel() != B * Hq * q_scale_blocks:
        raise ValueError(
            f"qscale must have {B * Hq * q_scale_blocks} elements for shape [B, Hq, ceil(Sq/block), 1], got {qscale.numel()}"
        )
    if kscale.numel() != B * Hkv * kv_scale_blocks:
        raise ValueError(
            f"kscale must have {B * Hkv * kv_scale_blocks} elements for shape [B, Hkv, ceil(Skv/block), 1], got {kscale.numel()}"
        )
    if torch.any(valid_block_num < 0).item():
        raise ValueError("valid_block_num entries must be non-negative")
    if torch.any(valid_block_num > kv_sparse_blocks).item():
        raise ValueError(f"valid_block_num entries must be <= {kv_sparse_blocks}")
    if decoupled_sparse_rows and not _is_sparse_qtile256_row64k_config(
        head_dim=D,
        quant_block_size=quant_block_size,
        q_tile_override=q_tile_override,
        sparse_q_block_tokens=effective_sparse_q_block_tokens,
        sparse_k_block_tokens=effective_sparse_k_block_tokens,
        tensor_layout=tensor_layout,
    ):
        raise ValueError(
            "Only the decoupled sparse config "
            "(head_dim=128, quant_block_size=64, q_tile_override=256, "
            "sparse_q_block_tokens=256, sparse_k_block_tokens=64, tensor_layout in {'HND', 'NHD'}) is supported"
        )

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
    sparse_fn_name = "sage_sparse_qtile256_row64k" if decoupled_sparse_rows else "sage_sparse"
    if not hasattr(lib, sparse_fn_name):
        raise RuntimeError(f"Loaded XPU extension does not expose {sparse_fn_name}")
    sparse_fn = getattr(lib, sparse_fn_name)
    sparse_fn(
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
        q_sparse_blocks,
        kv_sparse_blocks,
        q_tile_override,
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


def sage_sparse_row_linear(
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
    q_tile_override: int = 64,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Sparse SAGE prefill with one sparse row per workgroup via the q_tile=64 launcher."""
    if quant_block_size != 64:
        raise ValueError(f"sage_sparse_row_linear currently requires quant_block_size == 64, got {quant_block_size}")
    if q_tile_override not in (0, 64):
        raise ValueError(f"Unsupported q_tile_override={q_tile_override}; row-linear backend supports only 0 or 64")
    if query.device.type != "xpu":
        raise NotImplementedError("sage_sparse_row_linear is only supported on XPU")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError(f"Q/K must be int8, got Q={query.dtype}, K={key.dtype}")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"V must be float16 or bfloat16, got {value.dtype}")
    if qscale is None or kscale is None:
        raise ValueError("qscale and kscale must be provided for sage_sparse_row_linear")
    if lut.dtype != torch.int32 or valid_block_num.dtype != torch.int32:
        raise ValueError("lut and valid_block_num must be int32 tensors")
    if lut.device != query.device or valid_block_num.device != query.device:
        raise ValueError("lut and valid_block_num must be on the same XPU device as Q/K/V")
    if qscale.device != query.device or kscale.device != query.device:
        raise ValueError("qscale and kscale must be on the same XPU device as Q/K/V")

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout)
    Bv, Hkv2, Skv2, Dv = _validate_attention_tensor(value, "V", tensor_layout)

    if Bk != B or Bv != B:
        raise ValueError("Batch size mismatch between Q/K/V")
    if Hkv2 != Hkv or Skv2 != Skv or Dv != Dk:
        raise ValueError("K/V shape mismatch")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K/V")
    _validate_gqa_head_config(Hq, Hkv, op_name="sage_sparse_row_linear")
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
    lib.sage_sparse_row_linear(
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
        q_tile_override,
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
    kscale_total = torch.cat(
        [kscale_cache.reshape(B, Hkv, cache_blocks, 1), kscale.reshape(B, Hkv, cur_blocks, 1)], dim=2
    )
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
    return _dense_sagev1(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
    )


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
    return _dense_sagev1_pvi8(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
    )


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
        return 64
    raise ValueError(f"Unsupported head_dim={head_dim}; supported: 64, 128")


def _normalize_query_tile_tokens(
    query_tile_tokens: int | None,
    *,
    head_dim: int,
    quant_block_size: int,
) -> int:
    if query_tile_tokens is None:
        return _query_tile_tokens_for_head_dim(head_dim)

    tokens = int(query_tile_tokens)
    if tokens not in (64, 128, 256):
        raise ValueError(f"query_tile_tokens={tokens} is not supported; supported values: 64, 128, 256")
    if tokens <= 0 or tokens % quant_block_size != 0:
        raise ValueError(
            f"query_tile_tokens={tokens} must be a positive multiple of quant_block_size={quant_block_size}"
        )
    return tokens


def _validate_sparse_q_tile_override_for_head_dim(head_dim: int, q_tile_override: int) -> None:
    if head_dim == 128:
        supported = (0, 64, 128, 256)
    elif head_dim == 64:
        supported = (0, 128)
    else:
        raise ValueError(f"Unsupported head_dim={head_dim}; supported: 64, 128")

    if q_tile_override not in supported:
        raise ValueError(
            f"q_tile_override={q_tile_override} is not supported for head_dim={head_dim}; "
            f"supported values: {', '.join(str(v) for v in supported)}"
        )


def _routing_k_block_tokens_for_head_dim(head_dim: int, quant_block_size: int) -> int:
    if head_dim == 64:
        return quant_block_size
    if head_dim == 128:
        return 128
    raise ValueError(f"Unsupported head_dim={head_dim}; supported: 64, 128")


def _normalize_sparse_q_block_tokens(
    sparse_q_block_tokens: int | None,
    *,
    quant_block_size: int,
    q_route_block_tokens: int,
) -> int:
    tokens = quant_block_size if sparse_q_block_tokens is None else int(sparse_q_block_tokens)
    if tokens not in (64, 128, 256):
        raise ValueError(f"sparse_q_block_tokens={tokens} is not supported; supported values: 64, 128, 256")
    if tokens <= 0 or tokens % quant_block_size != 0:
        raise ValueError(
            f"sparse_q_block_tokens={tokens} must be a positive multiple of quant_block_size={quant_block_size}"
        )
    if q_route_block_tokens % tokens != 0:
        raise ValueError(
            f"q_route_block_tokens={q_route_block_tokens} must be divisible by sparse_q_block_tokens={tokens}"
        )
    return tokens


def _normalize_sparse_k_block_tokens(
    sparse_k_block_tokens: int | None,
    *,
    quant_block_size: int,
    k_route_block_tokens: int,
) -> int:
    tokens = quant_block_size if sparse_k_block_tokens is None else int(sparse_k_block_tokens)
    if tokens not in (64, 128):
        raise ValueError(f"sparse_k_block_tokens={tokens} is not supported; supported values: 64, 128")
    if tokens <= 0 or tokens % quant_block_size != 0:
        raise ValueError(
            f"sparse_k_block_tokens={tokens} must be a positive multiple of quant_block_size={quant_block_size}"
        )
    if k_route_block_tokens % tokens != 0:
        raise ValueError(
            f"k_route_block_tokens={k_route_block_tokens} must be divisible by sparse_k_block_tokens={tokens}"
        )
    return tokens


def _is_sparse_qtile256_row64k_config(
    *,
    head_dim: int,
    quant_block_size: int,
    q_tile_override: int,
    sparse_q_block_tokens: int,
    sparse_k_block_tokens: int,
    tensor_layout: str,
) -> bool:
    return (
        head_dim == 128
        and quant_block_size == 64
        and q_tile_override == 256
        and sparse_q_block_tokens == 256
        and sparse_k_block_tokens == 64
        and _normalize_tensor_layout(tensor_layout) in {"HND", "NHD"}
    )


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


def _select_blocks_for_cdf(
    sorted_prob: torch.Tensor,
    sorted_indices: torch.Tensor,
    target_mass: torch.Tensor,
) -> torch.Tensor:
    """Select the smallest sorted prefix whose probability reaches target_mass."""
    cumulative_before = sorted_prob.cumsum(dim=-1) - sorted_prob
    selected_sorted = cumulative_before < target_mass
    selected = torch.zeros_like(selected_sorted, dtype=torch.bool)
    return selected.scatter(-1, sorted_indices, selected_sorted)


def _build_block_causal_mask(
    num_q_tiles: int,
    num_k_blocks: int,
    q_route_block_tokens: int,
    k_route_block_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    q_idx = torch.arange(num_q_tiles, device=device, dtype=torch.int64).view(-1, 1)
    k_idx = torch.arange(num_k_blocks, device=device, dtype=torch.int64).view(1, -1)
    valid_k_per_q = ((q_idx + 1) * q_route_block_tokens + k_route_block_tokens - 1) // k_route_block_tokens
    return k_idx < valid_k_per_q


def _fill_block_map_torch(
    final_map: torch.Tensor, num_to_select: torch.Tensor, sorted_indices: torch.Tensor
) -> torch.Tensor:
    k_blocks = final_map.shape[-1]
    filled = final_map.clone()
    column_ids = torch.arange(k_blocks, device=final_map.device).view(1, 1, 1, k_blocks)
    target_new = torch.maximum(num_to_select, torch.ones_like(num_to_select))
    added = torch.zeros_like(num_to_select)
    for rank in range(k_blocks):
        idx_match = column_ids == sorted_indices[..., rank : rank + 1]
        is_new = idx_match & ~filled
        should_add = (added < target_new).unsqueeze(-1)
        newly_selected = should_add & is_new
        filled |= newly_selected
        added = added + newly_selected.any(dim=-1).to(added.dtype)
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
    invalid_mask = torch.arange(num_k_blocks, device=block_map.device).view(
        1, 1, 1, num_k_blocks
    ) >= valid_block_num.unsqueeze(-1)
    lut = torch.where(invalid_mask, torch.zeros_like(lut), lut)
    return lut.to(torch.int32).contiguous(), valid_block_num.to(torch.int32).contiguous()


def _prefix_keep_cross_attn_enabled() -> bool:
    return os.getenv("SPARGE_KEEP_PREFIX_ON_CROSS_ATTN", "0").strip().lower() in {"1", "true", "yes"}


def _get_protected_kv_tokens() -> int:
    value = os.getenv("SPARGE_PROTECTED_KV_TOKENS", "").strip()
    if not value:
        return 0
    tokens = int(value)
    if tokens < 0:
        raise ValueError(f"SPARGE_PROTECTED_KV_TOKENS must be >= 0, got {tokens}")
    return tokens


def _get_protected_kv_blocks() -> int:
    value = os.getenv("SPARGE_PROTECTED_KV_BLOCKS", "").strip()
    if not value:
        return 0
    blocks = int(value)
    if blocks < 0:
        raise ValueError(f"SPARGE_PROTECTED_KV_BLOCKS must be >= 0, got {blocks}")
    return blocks


def _prefix_protection_requested() -> bool:
    return (
        _prefix_keep_cross_attn_enabled()
        or _get_protected_kv_tokens() > 0
        or _get_protected_kv_blocks() > 0
    )


def _get_explicit_protected_prefix(
    ctx: "_SpargePreprocessContext",
) -> tuple[int, int]:
    protected_tokens = _get_protected_kv_tokens()
    protected_sparse_blocks = _get_protected_kv_blocks()
    protected_tile_blocks = 0
    if protected_tokens > 0:
        protected_sparse_blocks = max(
            protected_sparse_blocks,
            (protected_tokens + ctx.sparse_k_block_tokens - 1) // ctx.sparse_k_block_tokens,
        )
        protected_tile_blocks = max(
            protected_tile_blocks,
            (protected_tokens + ctx.k_route_block_tokens - 1) // ctx.k_route_block_tokens,
        )
    if protected_sparse_blocks > 0 and protected_tile_blocks == 0:
        protected_tile_blocks = (
            protected_sparse_blocks * ctx.sparse_k_block_tokens + ctx.k_route_block_tokens - 1
        ) // ctx.k_route_block_tokens
    return protected_sparse_blocks, protected_tile_blocks


def _get_prefix_protection_blocks(
    ctx: "_SpargePreprocessContext",
    *,
    raw_block_count: int,
    tile_block_count: int,
) -> tuple[int, int]:
    explicit_sparse_blocks, explicit_tile_blocks = _get_explicit_protected_prefix(ctx)
    if explicit_sparse_blocks > 0 or explicit_tile_blocks > 0:
        return (
            min(explicit_sparse_blocks, raw_block_count),
            min(explicit_tile_blocks, tile_block_count),
        )
    if not _prefix_keep_cross_attn_enabled():
        return 0, 0
    if ctx.is_causal or ctx.seq_len_kv <= ctx.seq_len_q:
        return 0, 0

    prefix_tokens = ctx.seq_len_kv - ctx.seq_len_q
    return (
        min((prefix_tokens + ctx.sparse_k_block_tokens - 1) // ctx.sparse_k_block_tokens, raw_block_count),
        min((prefix_tokens + ctx.k_route_block_tokens - 1) // ctx.k_route_block_tokens, tile_block_count),
    )


def _maybe_force_keep_prefix_blocks(
    ctx: _SpargePreprocessContext,
    raw_block_map: torch.Tensor,
    tile_block_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_sparse_blocks, prefix_tile_blocks = _get_prefix_protection_blocks(
        ctx,
        raw_block_count=raw_block_map.shape[-1],
        tile_block_count=tile_block_map.shape[-1],
    )
    if prefix_sparse_blocks <= 0 and prefix_tile_blocks <= 0:
        return raw_block_map, tile_block_map

    raw_block_map = raw_block_map.clone()
    tile_block_map = tile_block_map.clone()
    if prefix_sparse_blocks > 0:
        raw_block_map[..., :prefix_sparse_blocks] = True
    if prefix_tile_blocks > 0:
        tile_block_map[..., :prefix_tile_blocks] = True
    return raw_block_map, tile_block_map


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
    # Match the CUDA reference (SpargeAttn spas_sage_attn/utils.py): a partial
    # tail block (pad_tokens > 0 only ever affects the last block) divides padded
    # rows by a zero norm there (0/0 -> NaN -> NaN > thr == False), forcing it
    # dense. Our guarded norm would mark it prunable and sparsify the sequence
    # tail (= last video frames), so force the partial last block to False.
    if pad_tokens:
        sim_blocks[:, :, -1] = False

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
    cdfthreshd: torch.Tensor | None
    attention_sink: bool
    quant_block_size: int
    tensor_layout: str
    q_route_block_tokens: int
    k_route_block_tokens: int
    sparse_q_block_tokens: int
    sparse_k_block_tokens: int
    query_tile_tokens: int
    q_blocks_per_tile: int
    k_blocks_per_tile: int
    q_sparse_blocks_per_tile: int
    k_sparse_blocks_per_tile: int
    num_q_blocks: int
    num_k_blocks: int
    num_sparse_q_blocks: int
    num_sparse_k_blocks: int
    num_q_tiles: int
    num_k_tiles: int
    k_quant_granularity: int


def _build_sparge_preprocess_context(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    is_causal: bool,
    smooth_k: bool,
    simthreshd1: float | torch.Tensor,
    topk: float | torch.Tensor,
    cdfthreshd: float | torch.Tensor | None = None,
    attention_sink: bool,
    quant_block_size: int,
    tensor_layout: str,
    k_quant_granularity: int = 64,
    query_tile_tokens: int | None = None,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
) -> _SpargePreprocessContext:
    if query.device.type != "xpu":
        raise NotImplementedError("sparge_preprocess_topk is only supported on XPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {query.dtype}")
    if key.dtype != query.dtype:
        raise ValueError(f"K dtype must match Q dtype, got K={key.dtype}, Q={query.dtype}")
    if quant_block_size != 64:
        raise ValueError(
            f"quant_block_size={quant_block_size} is not supported in sparge_preprocess_topk; only 64 is supported"
        )
    if k_quant_granularity not in (64, 128):
        raise ValueError(f"k_quant_granularity={k_quant_granularity} is not supported; only 64 or 128")
    if k_quant_granularity % quant_block_size != 0:
        raise ValueError(
            f"k_quant_granularity={k_quant_granularity} must be a multiple of quant_block_size={quant_block_size}"
        )

    B, Hq, Sq, D = _validate_attention_tensor(query, "Q", tensor_layout, expected_dtype=query.dtype)
    Bk, Hkv, Skv, Dk = _validate_attention_tensor(key, "K", tensor_layout, expected_dtype=query.dtype)
    if Bk != B:
        raise ValueError("Batch size mismatch between Q and K")
    if Dk != D:
        raise ValueError("Head dim mismatch between Q and K")
    _validate_gqa_head_config(Hq, Hkv, op_name="sparge_preprocess_topk")
    if D not in (64, 128):
        raise ValueError(f"Unsupported head_dim={D}; supported: 64, 128")

    q_route_block_tokens = _normalize_query_tile_tokens(
        query_tile_tokens,
        head_dim=D,
        quant_block_size=quant_block_size,
    )
    requested_sparse_q_block_tokens = None if sparse_q_block_tokens is None else int(sparse_q_block_tokens)
    requested_sparse_k_block_tokens = None if sparse_k_block_tokens is None else int(sparse_k_block_tokens)
    k_route_block_tokens = _routing_k_block_tokens_for_head_dim(D, quant_block_size)
    if (
        D == 128
        and q_route_block_tokens == 256
        and requested_sparse_q_block_tokens == 256
        and requested_sparse_k_block_tokens in (None, 64)
    ):
        k_route_block_tokens = 64
    sparse_q_block_tokens = _normalize_sparse_q_block_tokens(
        sparse_q_block_tokens,
        quant_block_size=quant_block_size,
        q_route_block_tokens=q_route_block_tokens,
    )
    sparse_k_block_tokens = _normalize_sparse_k_block_tokens(
        sparse_k_block_tokens,
        quant_block_size=quant_block_size,
        k_route_block_tokens=k_route_block_tokens,
    )
    q_blocks_per_tile = q_route_block_tokens // quant_block_size
    k_blocks_per_tile = k_route_block_tokens // quant_block_size
    q_sparse_blocks_per_tile = q_route_block_tokens // sparse_q_block_tokens
    k_sparse_blocks_per_tile = k_route_block_tokens // sparse_k_block_tokens
    num_q_blocks = (Sq + quant_block_size - 1) // quant_block_size
    num_k_blocks = (Skv + quant_block_size - 1) // quant_block_size
    num_sparse_q_blocks = (Sq + sparse_q_block_tokens - 1) // sparse_q_block_tokens
    num_sparse_k_blocks = (Skv + sparse_k_block_tokens - 1) // sparse_k_block_tokens
    num_q_tiles = (Sq + q_route_block_tokens - 1) // q_route_block_tokens
    num_k_tiles = (Skv + k_route_block_tokens - 1) // k_route_block_tokens

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
        cdfthreshd=(
            None
            if cdfthreshd is None
            else _normalize_per_head_hparam(cdfthreshd, Hq, query.device, "cdfthreshd").clamp_(0.0, 1.0)
        ),
        attention_sink=attention_sink,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
        q_route_block_tokens=q_route_block_tokens,
        k_route_block_tokens=k_route_block_tokens,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
        query_tile_tokens=q_route_block_tokens,
        q_blocks_per_tile=q_blocks_per_tile,
        k_blocks_per_tile=k_blocks_per_tile,
        q_sparse_blocks_per_tile=q_sparse_blocks_per_tile,
        k_sparse_blocks_per_tile=k_sparse_blocks_per_tile,
        num_q_blocks=num_q_blocks,
        num_k_blocks=num_k_blocks,
        num_sparse_q_blocks=num_sparse_q_blocks,
        num_sparse_k_blocks=num_sparse_k_blocks,
        num_q_tiles=num_q_tiles,
        num_k_tiles=num_k_tiles,
        k_quant_granularity=k_quant_granularity,
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

    if ctx.q_blocks_per_tile > 1:
        tile_pooled_q = []
        tile_sim_q = []
        for qtile in range(ctx.num_q_tiles):
            qblk_start = qtile * ctx.q_blocks_per_tile
            qblk_end = min(qblk_start + ctx.q_blocks_per_tile, pooled_q.size(2))
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

    if ctx.k_blocks_per_tile > 1:
        pooled_k_for_routing, sim_k_for_routing, _, _ = _pool_sim_and_quant_torch(
            ctx.key,
            ctx.k_route_block_tokens,
            ctx.simthreshd1[: ctx.num_heads_kv],
            ctx.tensor_layout,
            key_mean,
        )
    else:
        pooled_k_for_routing = pooled_k
        sim_k_for_routing = sim_kblocks

    kv_head_index = _kv_head_index_for_q_heads(ctx.num_heads_q, ctx.num_heads_kv, ctx.query.device)
    pooled_k_for_q = pooled_k_for_routing[:, kv_head_index]
    sim_k_for_q = sim_k_for_routing[:, kv_head_index]
    sim_k_expand = sim_k_for_q.unsqueeze(-2).expand(-1, -1, ctx.num_q_tiles, -1)
    sim_q_expand = sim_q_for_routing.unsqueeze(-1).expand(-1, -1, -1, pooled_k_for_routing.size(2))

    pooled_score = torch.matmul(
        pooled_q_for_routing.to(torch.float32), pooled_k_for_q.transpose(-1, -2).to(torch.float32)
    )
    pooled_score *= ctx.head_dim**-0.5
    pooled_score = pooled_score.masked_fill(~sim_k_expand, -torch.inf)
    if ctx.is_causal:
        causal_mask = _build_block_causal_mask(
            ctx.num_q_tiles,
            pooled_k_for_routing.size(2),
            ctx.q_route_block_tokens,
            ctx.k_route_block_tokens,
            ctx.query.device,
        )
        pooled_score = pooled_score.masked_fill(
            ~causal_mask.view(1, 1, ctx.num_q_tiles, pooled_k_for_routing.size(2)), -torch.inf
        )
    else:
        causal_mask = None

    pooled_prob = _safe_softmax(pooled_score)
    _, _, _, num_k_route_blocks = pooled_prob.shape
    prefix_sparse_blocks, prefix_route_blocks = _get_prefix_protection_blocks(
        ctx,
        raw_block_count=ctx.num_sparse_k_blocks,
        tile_block_count=num_k_route_blocks,
    )
    del prefix_sparse_blocks
    if prefix_route_blocks > 0:
        final_tile_map = torch.zeros_like(pooled_prob, dtype=torch.bool)
        final_tile_map[..., :prefix_route_blocks] = True

        tail_prob = pooled_prob[..., prefix_route_blocks:]
        tail_sim_q_expand = sim_q_expand[..., prefix_route_blocks:]
        tail_sim_k_expand = sim_k_expand[..., prefix_route_blocks:]
        if tail_prob.size(-1) > 0:
            tail_sorted_prob = torch.sort(tail_prob, dim=-1, descending=True)
            num_tail_route_blocks = tail_prob.size(-1)
            num_to_select_tail = (
                (ctx.topk.view(1, ctx.num_heads_q, 1) * num_tail_route_blocks)
                .to(torch.int64)
                .expand(ctx.batch, -1, ctx.num_q_tiles)
                .contiguous()
            )
            tail_map = torch.zeros_like(tail_prob, dtype=torch.bool)
            tail_map[~tail_sim_k_expand] = True
            tail_map[~tail_sim_q_expand] = True
            tail_map = _fill_block_map_torch(tail_map, num_to_select_tail, tail_sorted_prob.indices)
            if ctx.cdfthreshd is not None:
                prefix_mass = pooled_prob[..., :prefix_route_blocks].sum(dim=-1, keepdim=True)
                target_tail_mass = (
                    ctx.cdfthreshd.view(1, ctx.num_heads_q, 1, 1) - prefix_mass
                ).clamp_min_(0.0)
                tail_map |= _select_blocks_for_cdf(
                    tail_sorted_prob.values,
                    tail_sorted_prob.indices,
                    target_tail_mass,
                )
            final_tile_map[..., prefix_route_blocks:] = tail_map
        if ctx.attention_sink:
            final_tile_map[..., 0] = True
    else:
        sorted_prob = torch.sort(pooled_prob, dim=-1, descending=True)
        num_to_select = (
            (ctx.topk.view(1, ctx.num_heads_q, 1) * num_k_route_blocks)
            .to(torch.int64)
            .expand(ctx.batch, -1, ctx.num_q_tiles)
            .contiguous()
        )
        final_tile_map = torch.zeros_like(pooled_prob, dtype=torch.bool)
        final_tile_map[~sim_k_expand] = True
        final_tile_map[~sim_q_expand] = True
        final_tile_map = _fill_block_map_torch(final_tile_map, num_to_select, sorted_prob.indices)
        if ctx.cdfthreshd is not None:
            final_tile_map |= _select_blocks_for_cdf(
                sorted_prob.values,
                sorted_prob.indices,
                ctx.cdfthreshd.view(1, ctx.num_heads_q, 1, 1),
            )
        if ctx.attention_sink:
            final_tile_map[..., 0] = True
    if causal_mask is not None:
        final_tile_map &= causal_mask.view(1, 1, ctx.num_q_tiles, num_k_route_blocks)

    q_block_to_tile = (
        torch.arange(ctx.num_sparse_q_blocks, device=ctx.query.device, dtype=torch.int64)
        // ctx.q_sparse_blocks_per_tile
    )
    q_block_to_tile = q_block_to_tile.clamp_max(ctx.num_q_tiles - 1)
    k_block_to_tile = (
        torch.arange(ctx.num_sparse_k_blocks, device=ctx.query.device, dtype=torch.int64)
        // ctx.k_sparse_blocks_per_tile
    )
    k_block_to_tile = k_block_to_tile.clamp_max(ctx.num_k_tiles - 1)
    raw_block_map = final_tile_map.index_select(2, q_block_to_tile).index_select(3, k_block_to_tile).contiguous()

    return {
        "query_i8": q_int8_hnd,
        "key_i8": k_int8_hnd,
        "qscale": q_scale,
        "kscale": k_scale,
        "raw_block_map": raw_block_map,
        "tile_block_map": final_tile_map.contiguous(),
        "sim_qblocks": sim_q_for_routing.contiguous(),
        "sim_kblocks": sim_k_for_routing.contiguous(),
        "backend": "torch",
    }


def _finalize_sparge_preprocess_outputs(
    ctx: _SpargePreprocessContext,
    backend_result: dict[str, Any],
) -> dict[str, Any]:
    raw_block_map = backend_result["raw_block_map"].contiguous()
    tile_block_map = backend_result["tile_block_map"].contiguous()
    raw_block_map, tile_block_map = _maybe_force_keep_prefix_blocks(
        ctx,
        raw_block_map,
        tile_block_map,
    )
    block_map = raw_block_map
    lut = backend_result.get("lut")
    valid_block_num = backend_result.get("valid_block_num")
    if (
        lut is None
        or valid_block_num is None
        or _prefix_protection_requested()
    ):
        lut, valid_block_num = _block_map_lut_torch(block_map)
    total_selected, total_candidates, selected_ratio, sparsity_ratio, selected_blocks_per_row = (
        _get_sparse_block_sparsity_stats(
            valid_block_num,
            lut,
            is_causal=ctx.is_causal,
        )
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
        "tile_block_map": tile_block_map,
        "sim_qblocks": backend_result["sim_qblocks"].contiguous(),
        "sim_kblocks": backend_result["sim_kblocks"].contiguous(),
        "query_tile_tokens": ctx.query_tile_tokens,
        "quant_block_size": ctx.quant_block_size,
        "sparse_q_block_tokens": ctx.sparse_q_block_tokens,
        "sparse_k_block_tokens": ctx.sparse_k_block_tokens,
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
    cdfthreshd: float | torch.Tensor | None = None,
    attention_sink: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
    query_tile_tokens: int | None = None,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
) -> dict[str, Any]:
    ctx = _build_sparge_preprocess_context(
        query,
        key,
        is_causal=is_causal,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        topk=topk,
        cdfthreshd=cdfthreshd,
        attention_sink=attention_sink,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
        query_tile_tokens=query_tile_tokens,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
    )
    return _finalize_sparge_preprocess_outputs(ctx, _sparge_preprocess_topk_torch_impl(ctx))


def sparge_block_map_to_mask(
    block_map: torch.Tensor,
    *,
    quant_block_size: int = 64,
    q_block_tokens: int | None = None,
    k_block_tokens: int | None = None,
    seq_len_q: int | None = None,
    seq_len_kv: int | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    if block_map.dtype != torch.bool or block_map.ndim != 4:
        raise ValueError("block_map must be a 4D bool tensor")
    if quant_block_size <= 0:
        raise ValueError(f"quant_block_size must be positive, got {quant_block_size}")
    q_tokens = quant_block_size if q_block_tokens is None else int(q_block_tokens)
    k_tokens = quant_block_size if k_block_tokens is None else int(k_block_tokens)
    if q_tokens <= 0 or k_tokens <= 0:
        raise ValueError(f"q_block_tokens and k_block_tokens must be positive, got {q_tokens} and {k_tokens}")
    batch, heads, q_blocks, kv_blocks = block_map.shape
    full_q = q_blocks * q_tokens
    full_k = kv_blocks * k_tokens
    seq_q = full_q if seq_len_q is None else seq_len_q
    seq_kv = full_k if seq_len_kv is None else seq_len_kv
    expanded = block_map.repeat_interleave(q_tokens, dim=-2).repeat_interleave(k_tokens, dim=-1)
    expanded = expanded[:, :, :seq_q, :seq_kv]
    mask = torch.full(expanded.shape, -1.0e9, dtype=torch.float32, device=block_map.device)
    mask = torch.where(expanded, torch.zeros_like(mask), mask)
    if is_causal:
        causal = torch.triu(
            torch.full((seq_q, seq_kv), -1.0e9, dtype=torch.float32, device=block_map.device), diagonal=1
        )
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
    cdfthreshd: float | torch.Tensor | None = None,
    attention_sink: bool = False,
    quant_block_size: int = 64,
    tensor_layout: str = "HND",
    k_quant_granularity: int = 64,
    query_tile_tokens: int | None = None,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
    backend_preference: str | None = None,
) -> dict[str, Any]:
    ctx = _build_sparge_preprocess_context(
        query,
        key,
        is_causal=is_causal,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        topk=topk,
        cdfthreshd=cdfthreshd,
        attention_sink=attention_sink,
        quant_block_size=quant_block_size,
        tensor_layout=tensor_layout,
        k_quant_granularity=k_quant_granularity,
        query_tile_tokens=query_tile_tokens,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
    )
    return _sparge_preprocess_topk_dispatch(
        ctx,
        backend_preference=(
            "torch"
            if cdfthreshd is not None
            else backend_preference or _get_sparse_preprocess_backend_preference()
        ),
    )


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
    Bkc, Hkvc, Skvc, Dkc = _validate_attention_tensor(
        key_cache, "K_cache", tensor_layout, expected_dtype=key_cache.dtype
    )
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
    _validate_gqa_head_config(Hq, Hkv, op_name="sparge_preprocess_topk_decode")
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

    kv_head_index = _kv_head_index_for_q_heads(Hq, Hkv, query.device)
    pooled_k_for_q = pooled_k[:, kv_head_index]
    sim_k_for_q = sim_kblocks[:, kv_head_index]
    pooled_score = torch.matmul(
        pooled_q_route[:, :, :1, :].to(torch.float32),
        pooled_k_for_q.transpose(-1, -2).to(torch.float32),
    )
    pooled_score *= D**-0.5
    pooled_score = pooled_score.masked_fill(~sim_k_for_q.unsqueeze(-2), -torch.inf)
    pooled_prob = _safe_softmax(pooled_score)
    sorted_prob = torch.sort(pooled_prob, dim=-1, descending=True)
    num_to_select = (topk_tensor.view(1, Hq, 1) * num_k_blocks).to(torch.int64).expand(B, -1, 1).contiguous()
    final_tile_map = torch.zeros_like(pooled_prob, dtype=torch.bool)
    final_tile_map[~sim_k_for_q.unsqueeze(-2)] = True
    final_tile_map = _fill_block_map_torch(final_tile_map, num_to_select, sorted_prob.indices)
    if attention_sink:
        final_tile_map[..., 0] = True

    raw_block_map = final_tile_map.contiguous()
    block_map = raw_block_map
    lut, valid_block_num = _block_map_lut_torch(block_map)
    total_selected, total_candidates, selected_ratio, sparsity_ratio, selected_blocks_per_row = (
        _get_sparse_block_sparsity_stats(
            valid_block_num,
            lut,
            is_causal=False,
        )
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
        raise ValueError(
            f"output_dtype must match value.dtype in the current implementation, got {output_dtype} vs {value.dtype}"
        )
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
    k_quant_granularity: int = 64,
    query_tile_tokens: int | None = None,
    q_tile_override: int = 0,
    sparse_q_block_tokens: int | None = None,
    sparse_k_block_tokens: int | None = None,
) -> torch.Tensor | tuple[Any, ...]:
    if query.device.type != "xpu":
        raise NotImplementedError("sparge_sage2_attn_meansim_topk_xpu is only supported on XPU")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p must be 0.0 for sparge_sage2_attn_meansim_topk_xpu")
    if attn_mask is not None and is_causal:
        raise ValueError("attn_mask and is_causal cannot both be set")
    if output_dtype is not None and output_dtype != value.dtype:
        raise ValueError(
            f"output_dtype must match value.dtype in the current implementation, got {output_dtype} vs {value.dtype}"
        )
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

    effective_query_tile_tokens = query_tile_tokens
    effective_q_tile_override = q_tile_override
    if effective_query_tile_tokens is None and q_tile_override in (64, 128, 256):
        effective_query_tile_tokens = q_tile_override
    elif effective_query_tile_tokens is not None:
        if q_tile_override == 0:
            effective_q_tile_override = int(effective_query_tile_tokens)
        elif q_tile_override != int(effective_query_tile_tokens):
            raise ValueError(
                "query_tile_tokens and q_tile_override must match when both are set; "
                f"got query_tile_tokens={effective_query_tile_tokens}, q_tile_override={q_tile_override}"
            )
    _validate_sparse_q_tile_override_for_head_dim(D, effective_q_tile_override)

    normalized_mask = _normalize_sparse_mask(attn_mask, B, Sq, Skv, query.device)
    metadata = sparge_preprocess_topk(
        query,
        key,
        is_causal=is_causal,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        topk=topk,
        cdfthreshd=cdfthreshd,
        attention_sink=attention_sink,
        quant_block_size=64,
        tensor_layout=tensor_layout,
        k_quant_granularity=k_quant_granularity,
        query_tile_tokens=effective_query_tile_tokens,
        sparse_q_block_tokens=sparse_q_block_tokens,
        sparse_k_block_tokens=sparse_k_block_tokens,
    )
    _get_xpu_sparse_kernel_backend()
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
        q_tile_override=effective_q_tile_override,
        sparse_q_block_tokens=metadata["sparse_q_block_tokens"],
        sparse_k_block_tokens=metadata["sparse_k_block_tokens"],
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
