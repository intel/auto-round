from __future__ import annotations

from typing import Any

import torch

from .sparge_preprocess_triton import (
    _block_map_lut_triton,
    _build_block_causal_mask,
    _fill_block_map_triton,
    _get_pool_sim_triton_simmean_fuse_quant,
    _sequence_mean_native_layout,
)


def run_triton_xpu_cuda_port_preprocess(ctx: Any) -> dict[str, Any]:
    """CUDA-style Triton preprocess port kept as an independent backend path.

    This starts from the same Triton-XPU kernel building blocks as the current
    path, but the entrypoint is intentionally separated so future work can keep
    the CUDA-port semantics isolated from the existing backend.
    """
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
    pooled_score *= ctx.head_dim ** -0.5
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
            ~causal_mask.view(1, 1, ctx.num_q_tiles, pooled_k_for_routing.size(2)),
            -torch.inf,
        )
    else:
        causal_mask = None

    pooled_prob = torch.softmax(torch.where(torch.isfinite(pooled_score), pooled_score, torch.full_like(pooled_score, -1.0e9)), dim=-1)
    pooled_prob = torch.where(torch.isfinite(pooled_score), pooled_prob, torch.zeros_like(pooled_prob))
    denom = pooled_prob.sum(dim=-1, keepdim=True)
    pooled_prob = torch.where(denom > 0, pooled_prob / denom, torch.zeros_like(pooled_prob))
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

    q_block_to_tile = torch.arange(ctx.num_q_blocks, device=ctx.query.device, dtype=torch.int64) // ctx.q_blocks_per_tile
    q_block_to_tile = q_block_to_tile.clamp_max(ctx.num_q_tiles - 1)
    k_block_to_tile = torch.arange(ctx.num_k_blocks, device=ctx.query.device, dtype=torch.int64) // ctx.k_blocks_per_tile
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
        "backend": "triton_xpu_cuda_port",
    }
