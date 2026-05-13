# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch

from . import ARK

logger = logging.getLogger(__name__)

_ARK_SDPA_PATCHED = False
_ORIG_SDPA = None
_PATCH_CONFIG = None
_FALLBACK_WARNING_LOGGED = False


def _log_fallback_warning_once(backend: str, error: Exception) -> None:
    global _FALLBACK_WARNING_LOGGED

    if _FALLBACK_WARNING_LOGGED:
        return

    logger.warning(
        "ARK %s patch failed and fell back to torch SDPA. Subsequent failures will be suppressed. Error: %s",
        backend,
        error,
    )
    _FALLBACK_WARNING_LOGGED = True


def _validate_backend(backend: str, quant_block_size: int) -> str:
    backend = backend.lower()
    if backend not in {"sdpa", "sagev1"}:
        raise ValueError(f"Unsupported ARK attention backend: {backend}")
    if backend == "sagev1" and quant_block_size <= 0:
        raise ValueError("quant_block_size must be > 0 when backend='sagev1'")
    return backend


def _normalize_attn_mask(attn_mask: torch.Tensor, batch: int, seq_q: int, seq_kv: int) -> torch.Tensor:
    if attn_mask.dtype == torch.bool:
        raise ValueError("Boolean attention masks are not supported by the ARK SDPA patch")

    if attn_mask.ndim == 2:
        if attn_mask.shape != (seq_q, seq_kv):
            raise ValueError("Unsupported 2D attention mask shape")
        attn_mask = attn_mask.view(1, 1, seq_q, seq_kv)
    elif attn_mask.ndim == 3:
        if attn_mask.shape != (batch, seq_q, seq_kv):
            raise ValueError("Unsupported 3D attention mask shape")
        attn_mask = attn_mask.unsqueeze(1)
    elif attn_mask.ndim == 4:
        if attn_mask.shape[-2:] != (seq_q, seq_kv):
            raise ValueError("Unsupported 4D attention mask shape")
        if attn_mask.shape[1] != 1:
            raise ValueError("Only attention masks with head dimension 1 are supported")
        if attn_mask.shape[0] == 1 and batch != 1:
            attn_mask = attn_mask.expand(batch, -1, -1, -1)
        elif attn_mask.shape[0] != batch:
            raise ValueError("Unsupported batch dimension in attention mask")
    else:
        raise ValueError("Unsupported attention mask rank")

    return attn_mask.contiguous().to(torch.float32)


def _is_pure_causal_mask(attn_mask: torch.Tensor) -> bool:
    seq_q = attn_mask.shape[-2]
    seq_kv = attn_mask.shape[-1]
    if seq_q != seq_kv:
        return False

    mask_2d = attn_mask.reshape(-1, seq_q, seq_kv)[0]
    tri_up = torch.triu(torch.ones(seq_q, seq_kv, dtype=torch.bool, device=attn_mask.device), 1)
    return bool(torch.isinf(mask_2d[tri_up]).all().item()) and bool((mask_2d[~tri_up] == 0).all().item())


def _can_use_ark_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    backend: str,
) -> bool:
    if query.device.type != "xpu":
        return False
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        return False
    if key.shape != value.shape:
        return False
    if query.shape[-1] != key.shape[-1]:
        return False
    if dropout_p != 0.0:
        return False
    if attn_mask is not None and attn_mask.device.type != "xpu":
        return False
    if backend == "sdpa":
        return query.dtype in (torch.float16, torch.bfloat16) and query.shape[-1] in (
            64,
            96,
            128,
            192,
        )
    if backend == "sagev1":
        return query.dtype == torch.float16 and query.shape[-1] in (64, 128)
    return True


def patch_torch_sdpa_with_ark(*, strict: bool = False, backend: str = "sdpa", quant_block_size: int = 64) -> bool:
    global _ARK_SDPA_PATCHED, _ORIG_SDPA, _PATCH_CONFIG

    backend = _validate_backend(backend, quant_block_size)
    new_config = {"backend": backend, "quant_block_size": int(quant_block_size)}

    if _ARK_SDPA_PATCHED:
        if _PATCH_CONFIG == new_config:
            return True
        unpatch_torch_sdpa_with_ark()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        if strict:
            raise RuntimeError("XPU is not available")
        return False

    ark = ARK()
    if ark.xpu_lib is None or not hasattr(ark.xpu_lib, backend):
        if strict:
            raise RuntimeError(f"ARK XPU {backend} kernel is not available")
        return False

    _ORIG_SDPA = torch.nn.functional.scaled_dot_product_attention
    orig_sdpa = _ORIG_SDPA

    def _patched_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        **kwargs,
    ):
        if not _can_use_ark_attention(query, key, value, attn_mask, dropout_p, backend):
            return orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

        normalized_mask = None
        if attn_mask is not None:
            try:
                normalized_mask = _normalize_attn_mask(attn_mask, query.shape[0], query.shape[-2], key.shape[-2])
            except ValueError:
                return orig_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs,
                )
            if not is_causal and _is_pure_causal_mask(normalized_mask):
                normalized_mask = None
                is_causal = True

        try:
            if backend == "sdpa":
                return ark.sdpa(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    attn_mask=normalized_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            return ark.sagev1(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                attn_mask=normalized_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                quant_block_size=quant_block_size,
            )
        except (NotImplementedError, RuntimeError, ValueError) as error:
            _log_fallback_warning_once(backend, error)
            return orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    _ARK_SDPA_PATCHED = True
    _PATCH_CONFIG = new_config
    return True


def unpatch_torch_sdpa_with_ark() -> bool:
    global _ARK_SDPA_PATCHED, _ORIG_SDPA, _PATCH_CONFIG, _FALLBACK_WARNING_LOGGED

    if not _ARK_SDPA_PATCHED or _ORIG_SDPA is None:
        return False

    torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA
    _ARK_SDPA_PATCHED = False
    _ORIG_SDPA = None
    _PATCH_CONFIG = None
    _FALLBACK_WARNING_LOGGED = False
    return True
