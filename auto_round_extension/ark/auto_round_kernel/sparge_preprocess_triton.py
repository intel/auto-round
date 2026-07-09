# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_TRITON_FALLBACK_WARNING_LOGGED = False


def _log_fallback_warning_once(error: Exception) -> None:
    global _TRITON_FALLBACK_WARNING_LOGGED

    if _TRITON_FALLBACK_WARNING_LOGGED:
        return

    logger.warning(
        "ARK SPARGE preprocess Triton-XPU backend failed and fell back to torch. Subsequent failures will be suppressed. Error: %s",
        error,
    )
    _TRITON_FALLBACK_WARNING_LOGGED = True


def _normalize_backend_preference(backend_preference: str) -> str:
    backend = backend_preference.lower()
    if backend not in {"auto", "torch", "triton_xpu"}:
        raise ValueError(f"Unsupported SPARGE preprocess backend: {backend_preference}")
    return backend


def _ensure_triton_xpu_available(query: torch.Tensor, head_dim: int) -> None:
    if query.device.type != "xpu":
        raise RuntimeError("Triton-XPU preprocess requires XPU tensors")
    if head_dim not in (64, 128):
        raise ValueError(f"Unsupported head_dim={head_dim} for Triton-XPU preprocess")


def _sequence_mean_native_layout(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    layout = tensor_layout.upper()
    if layout == "HND":
        return tensor.mean(dim=2).contiguous()
    return tensor.mean(dim=1).contiguous()


def _slice_sequence_native_layout(tensor: torch.Tensor, tensor_layout: str, start: int, end: int) -> torch.Tensor:
    layout = tensor_layout.upper()
    if layout == "HND":
        return tensor[:, :, start:end, :]
    return tensor[:, start:end, :, :]


@triton.jit
def _triton_bmm_pool_sim_simmean_fuse_quant_xpu(
    x_ptr,
    xm_ptr,
    pool_ptr,
    sim_ptr,
    x_quant_ptr,
    scale_ptr,
    simthreshd1_ptr,
    N,
    x_stride_b,
    x_stride_s,
    x_stride_h,
    x_stride_d,
    xq_stride_b,
    xq_stride_s,
    xq_stride_h,
    xq_stride_d,
    D: tl.constexpr,
    BS: tl.constexpr,
    FUSE_MEAN: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nb = tl.program_id(2)
    H = tl.num_programs(1)

    row_ids = nb * BS + tl.arange(0, BS)
    xmask = row_ids[:, None] < N
    x_ptrs = (
        x_ptr + b * x_stride_b + row_ids[:, None] * x_stride_s + h * x_stride_h + tl.arange(0, D)[None, :] * x_stride_d
    )
    x = tl.load(x_ptrs, mask=xmask, other=0.0)
    valid_rows = N - nb * BS
    bs_eff = tl.minimum(valid_rows, BS)
    x_fp32 = x.to(tl.float32)

    if FUSE_MEAN:
        xm_ptrs = xm_ptr + (b * H * D) + (h * D) + tl.arange(0, D)
        x_mean = tl.load(xm_ptrs).to(tl.float32)
        x_fp32 = x_fp32 - x_mean[None, :]
        x_fp32 = tl.where(xmask, x_fp32, 0.0)

    cur_h1 = tl.load(simthreshd1_ptr + h)
    pool = tl.sum(x_fp32, axis=0) / bs_eff
    x_sq = x_fp32 * x_fp32
    x_norm = tl.sqrt(tl.sum(x_sq, axis=1, keep_dims=True))
    x_norm = tl.where(x_norm > 0, x_norm, 1.0)
    x_normed = (x_fp32 / x_norm).to(tl.float16)
    grams = tl.dot(x_normed, tl.trans(x_normed))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (bs_eff * bs_eff)) > cur_h1
    # Match the CUDA reference (SpargeAttn spas_sage_attn/utils.py): there the
    # self-similarity divides zero-padded rows by a zero norm (0/0 -> NaN), so a
    # partial tail block compares NaN > thr == False and is forced dense. Our
    # guarded norm would instead mark the tail block similar/prunable, sparsifying
    # the sequence tail (= last video frames). Force partial blocks to False.
    cur_sim = cur_sim & (bs_eff >= BS)

    num_blocks = tl.cdiv(N, BS)
    pool_block_offset = (b * H * num_blocks * D) + (h * num_blocks * D) + (nb * D)
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = (b * H * num_blocks) + (h * num_blocks) + nb
    tl.store(sim_ptr + sim_offset, cur_sim)

    scale = tl.max(tl.abs(x_fp32)) / 127.0
    scale = scale + 1.0e-7
    x_int8 = x_fp32 / scale
    x_int8 = x_int8 + 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    x_quant_ptrs = (
        x_quant_ptr
        + b * xq_stride_b
        + row_ids[:, None] * xq_stride_s
        + h * xq_stride_h
        + tl.arange(0, D)[None, :] * xq_stride_d
    )
    scale_ptrs = scale_ptr + (b * H * num_blocks) + (h * num_blocks) + nb
    tl.store(x_quant_ptrs, x_int8, mask=xmask)
    tl.store(scale_ptrs, scale)


def _safe_softmax(scores: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.full_like(scores, -1.0e9))
    probs = torch.softmax(safe_scores, dim=-1)
    probs = torch.where(finite, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, probs / denom, torch.zeros_like(probs))


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


@triton.jit
def _triton_fill_block_map_kernel(final_map_ptr, num_to_select_ptr, sorted_indices_ptr, NK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q = tl.program_id(2)
    H = tl.num_programs(1)
    Q = tl.num_programs(2)

    cur_num_to_select = tl.load(num_to_select_ptr + b * H * Q + h * Q + q)
    sorted_row_ptr = sorted_indices_ptr + b * H * Q * NK + h * Q * NK + q * NK
    final_row_ptr = final_map_ptr + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = tl.where(cur_num_to_select == 0, 1, cur_num_to_select)
    added = 0
    for i in range(NK):
        if added < cur_num_to_select:
            cur_idx = tl.load(sorted_row_ptr + i)
            cur_val = tl.load(final_row_ptr + cur_idx)
            if cur_val == 0:
                tl.store(final_row_ptr + cur_idx, 1)
                added += 1


def _fill_block_map_triton(
    final_map: torch.Tensor, num_to_select: torch.Tensor, sorted_indices: torch.Tensor
) -> torch.Tensor:
    final_map_u8 = final_map.contiguous().to(torch.uint8)
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    bsz, num_heads, num_q, num_k = final_map.shape
    grid = (bsz, num_heads, num_q)
    _triton_fill_block_map_kernel[grid](final_map_u8, num_to_select, sorted_indices, NK=num_k)
    return final_map_u8.to(torch.bool)


@triton.jit
def _triton_block_map_to_lut_kernel(map_ptr, lut_ptr, valid_block_num_ptr, NK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q = tl.program_id(2)
    H = tl.num_programs(1)
    Q = tl.num_programs(2)

    row_map_ptr = map_ptr + b * H * Q * NK + h * Q * NK + q * NK
    row_lut_ptr = lut_ptr + b * H * Q * NK + h * Q * NK + q * NK
    row_valid_ptr = valid_block_num_ptr + b * H * Q + h * Q + q

    valid_block_num = 0
    prev_block = 0
    for i in range(NK):
        cur_block = tl.load(row_map_ptr + i)
        if cur_block:
            tl.store(row_lut_ptr + valid_block_num, i - prev_block)
            valid_block_num += 1
            prev_block = i
    tl.store(row_valid_ptr, valid_block_num)


def _block_map_lut_triton(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block_map_u8 = block_map.contiguous().to(torch.uint8)
    bsz, num_heads, num_q, num_k = block_map.shape
    lut = torch.zeros((bsz, num_heads, num_q, num_k), dtype=torch.int32, device=block_map.device)
    valid_block_num = torch.zeros((bsz, num_heads, num_q), dtype=torch.int32, device=block_map.device)
    grid = (bsz, num_heads, num_q)
    _triton_block_map_to_lut_kernel[grid](block_map_u8, lut, valid_block_num, NK=num_k)
    return lut, valid_block_num


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


def _get_pool_sim_triton_simmean_fuse_quant(
    x: torch.Tensor,
    x_mean: torch.Tensor | None,
    block_size: int,
    simthreshd1: torch.Tensor,
    tensor_layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    layout = tensor_layout.upper()
    if layout == "HND":
        bsz, num_heads, seq_len, head_dim = x.shape
        x_stride_b, x_stride_h, x_stride_s, x_stride_d = x.stride()
    else:
        bsz, seq_len, num_heads, head_dim = x.shape
        x_stride_b, x_stride_s, x_stride_h, x_stride_d = x.stride()
    num_blocks = (seq_len + block_size - 1) // block_size
    pool = torch.empty((bsz, num_heads, num_blocks, head_dim), device=x.device, dtype=x.dtype)
    sim_u8 = torch.empty((bsz, num_heads, num_blocks), device=x.device, dtype=torch.uint8)
    x_quant = torch.empty_like(x, dtype=torch.int8)
    x_scale = torch.empty((bsz, num_heads, num_blocks), device=x.device, dtype=torch.float32)
    if layout == "HND":
        xq_stride_b, xq_stride_h, xq_stride_s, xq_stride_d = x_quant.stride()
    else:
        xq_stride_b, xq_stride_s, xq_stride_h, xq_stride_d = x_quant.stride()
    mean = None if x_mean is None else x_mean.contiguous().squeeze(-2)
    grid = (bsz, num_heads, num_blocks)
    num_warps = 4 if block_size <= 64 else 8
    _triton_bmm_pool_sim_simmean_fuse_quant_xpu[grid](
        x,
        mean,
        pool,
        sim_u8,
        x_quant,
        x_scale,
        simthreshd1.contiguous(),
        seq_len,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        x_stride_d,
        xq_stride_b,
        xq_stride_s,
        xq_stride_h,
        xq_stride_d,
        D=head_dim,
        BS=block_size,
        FUSE_MEAN=mean is not None,
        num_warps=num_warps,
    )
    return pool, sim_u8.to(torch.bool), x_quant, x_scale.unsqueeze(-1).contiguous()


def _run_triton_xpu_preprocess(ctx: Any) -> dict[str, Any]:
    key_mean = _sequence_mean_native_layout(ctx.key, ctx.tensor_layout) if ctx.smooth_k else None
    pooled_q, sim_qblocks, q_int8_hnd, q_scale = _get_pool_sim_triton_simmean_fuse_quant(
        ctx.query,
        None,
        ctx.quant_block_size,
        ctx.simthreshd1,
        ctx.tensor_layout,
    )
    pooled_k, sim_kblocks, k_int8_hnd, k_scale = _get_pool_sim_triton_simmean_fuse_quant(
        ctx.key,
        key_mean,
        ctx.quant_block_size,
        ctx.simthreshd1[: ctx.num_heads_kv],
        ctx.tensor_layout,
    )
    k_quant_gran = getattr(ctx, "k_quant_granularity", ctx.quant_block_size)
    if k_quant_gran > ctx.quant_block_size:
        _, _, k_int8_hnd, k_scale = _get_pool_sim_triton_simmean_fuse_quant(
            ctx.key,
            key_mean,
            k_quant_gran,
            ctx.simthreshd1[: ctx.num_heads_kv],
            ctx.tensor_layout,
        )
        ratio = k_quant_gran // ctx.quant_block_size
        num_k_blocks_fine = (ctx.seq_len_kv + ctx.quant_block_size - 1) // ctx.quant_block_size
        k_scale = k_scale.repeat_interleave(ratio, dim=2)[:, :, :num_k_blocks_fine, :]
    if ctx.q_blocks_per_tile > 1:
        pooled_q_for_routing, sim_q_for_routing, _, _ = _get_pool_sim_triton_simmean_fuse_quant(
            ctx.query,
            None,
            ctx.query_tile_tokens,
            ctx.simthreshd1,
            ctx.tensor_layout,
        )
    else:
        pooled_q_for_routing = pooled_q
        sim_q_for_routing = sim_qblocks

    if ctx.k_blocks_per_tile > 1:
        pooled_k_for_routing, sim_k_for_routing, _, _ = _get_pool_sim_triton_simmean_fuse_quant(
            ctx.key,
            key_mean,
            ctx.k_route_block_tokens,
            ctx.simthreshd1[: ctx.num_heads_kv],
            ctx.tensor_layout,
        )
    else:
        pooled_k_for_routing = pooled_k
        sim_k_for_routing = sim_kblocks

    kv_head_index = torch.arange(ctx.num_heads_q, device=ctx.query.device, dtype=torch.int64) // (
        ctx.num_heads_q // ctx.num_heads_kv
    )
    pooled_k_for_q = pooled_k_for_routing[:, kv_head_index]
    sim_k_for_q = sim_k_for_routing[:, kv_head_index]
    sim_k_expand = sim_k_for_q.unsqueeze(-2).expand(-1, -1, ctx.num_q_tiles, -1)
    sim_q_expand = sim_q_for_routing.unsqueeze(-1).expand(-1, -1, -1, pooled_k_for_routing.size(2))

    pooled_score = torch.matmul(
        pooled_q_for_routing.to(torch.float32),
        pooled_k_for_q.transpose(-1, -2).to(torch.float32),
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
    sorted_prob = torch.sort(pooled_prob, dim=-1, descending=True)
    _, _, _, num_k_route_blocks = pooled_prob.shape
    num_to_select = (
        (ctx.topk.view(1, ctx.num_heads_q, 1) * num_k_route_blocks)
        .to(torch.int64)
        .expand(ctx.batch, -1, ctx.num_q_tiles)
        .contiguous()
    )
    final_tile_map = torch.zeros_like(pooled_prob, dtype=torch.bool)
    final_tile_map[~sim_k_expand] = True
    final_tile_map[~sim_q_expand] = True
    final_tile_map = _fill_block_map_triton(final_tile_map, num_to_select, sorted_prob.indices)
    if causal_mask is not None:
        final_tile_map &= causal_mask.view(1, 1, ctx.num_q_tiles, num_k_route_blocks)
    if ctx.attention_sink:
        final_tile_map[..., 0] = True

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
    lut, valid_block_num = _block_map_lut_triton(raw_block_map)

    return {
        "query_i8": q_int8_hnd,
        "key_i8": k_int8_hnd,
        "qscale": q_scale,
        "kscale": k_scale,
        "lut": lut,
        "valid_block_num": valid_block_num,
        "raw_block_map": raw_block_map,
        "tile_block_map": final_tile_map.contiguous(),
        "sim_qblocks": sim_q_for_routing.contiguous(),
        "sim_kblocks": sim_k_for_routing.contiguous(),
        "backend": "triton_xpu",
    }


def dispatch_sparge_preprocess_backend(
    *,
    ctx: Any,
    torch_backend: Callable[[], dict[str, Any]],
    backend_preference: str = "auto",
) -> dict[str, Any]:
    backend = _normalize_backend_preference(backend_preference)
    if backend == "torch":
        result = torch_backend()
        result["backend"] = "torch"
        return result

    try:
        _ensure_triton_xpu_available(ctx.query, ctx.head_dim)
        result = _run_triton_xpu_preprocess(ctx)
        result["backend"] = "triton_xpu"
        return result
    except (NotImplementedError, RuntimeError, ValueError, triton.runtime.errors.TritonError) as error:
        if backend == "triton_xpu":
            raise
        _log_fallback_warning_once(error)
        result = torch_backend()
        result["backend"] = "torch"
        return result
