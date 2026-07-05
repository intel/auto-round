from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sage_sparse_prefill_hnd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    qscale_ptr,
    kscale_ptr,
    lut_ptr,
    valid_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_qsc_b,
    stride_qsc_h,
    stride_qsc_blk,
    stride_ksc_b,
    stride_ksc_h,
    stride_ksc_blk,
    stride_lut_b,
    stride_lut_h,
    stride_lut_qb,
    stride_lut_k,
    stride_valid_b,
    stride_valid_h,
    stride_valid_qb,
    seq_len_q,
    seq_len_kv,
    scale,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_K_BLOCKS: tl.constexpr,
    USE_BF16: tl.constexpr,
):
    q_block = tl.program_id(0)
    hq = tl.program_id(1)
    batch = tl.program_id(2)
    hk = hq // NUM_KV_GROUPS

    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_ptr + batch * stride_qb + hq * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len_q, other=0).to(tl.float32)
    q_scale = tl.load(qscale_ptr + batch * stride_qsc_b + hq * stride_qsc_h + q_block * stride_qsc_blk)
    q = q * q_scale
    if USE_BF16:
        q = q.to(tl.bfloat16)
    else:
        q = q.to(tl.float16)

    valid = tl.load(valid_ptr + batch * stride_valid_b + hq * stride_valid_h + q_block * stride_valid_qb)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q_mask = offs_m < seq_len_q
    current_k_block = 0
    offs_n_base = tl.arange(0, BLOCK_N)

    for rank in range(MAX_K_BLOCKS):
        if rank < valid:
            delta = tl.load(
                lut_ptr
                + batch * stride_lut_b
                + hq * stride_lut_h
                + q_block * stride_lut_qb
                + rank * stride_lut_k
            )
            current_k_block += delta
            offs_n = current_k_block * BLOCK_N + offs_n_base
            k_ptrs = (
                k_ptr
                + batch * stride_kb
                + hk * stride_kh
                + offs_n[None, :] * stride_ks
                + offs_d[:, None] * stride_kd
            )
            k = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len_kv, other=0).to(tl.float32)
            k_scale = tl.load(kscale_ptr + batch * stride_ksc_b + hk * stride_ksc_h + current_k_block * stride_ksc_blk)
            k = k * k_scale
            if USE_BF16:
                k = k.to(tl.bfloat16)
            else:
                k = k.to(tl.float16)

            qk = tl.dot(q, k).to(tl.float32) * scale
            qk = tl.where(q_mask[:, None], qk, -float("inf"))
            row_max = tl.max(qk, axis=1)
            new_m = tl.maximum(m_i, row_max)
            p = tl.exp(qk - new_m[:, None])
            p = tl.where(q_mask[:, None], p, 0.0)
            alpha = tl.exp(m_i - new_m)

            v_ptrs = (
                v_ptr
                + batch * stride_vb
                + hk * stride_vh
                + offs_n[:, None] * stride_vs
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len_kv, other=0)
            if USE_BF16:
                v = v.to(tl.bfloat16)
                pv = p.to(tl.bfloat16)
            else:
                v = v.to(tl.float16)
                pv = p.to(tl.float16)
            acc = acc * alpha[:, None] + tl.dot(pv, v).to(tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = new_m

    denom = tl.where(l_i > 0, l_i, 1.0)
    out = acc / denom[:, None]
    o_ptrs = o_ptr + batch * stride_ob + hq * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(o_ptr.type.element_ty), mask=q_mask[:, None])


def triton_sparse_prefill_attention(
    query_hnd: torch.Tensor,
    key_hnd: torch.Tensor,
    value_hnd: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    *,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    scale: float,
    quant_block_size: int,
) -> torch.Tensor:
    if query_hnd.device.type != "xpu":
        raise NotImplementedError("triton_sparse_prefill_attention requires XPU tensors")
    if quant_block_size != 64:
        raise ValueError(f"Only quant_block_size=64 is supported, got {quant_block_size}")

    batch, num_heads_q, seq_len_q, head_dim = query_hnd.shape
    batch_k, num_heads_kv, seq_len_kv, head_dim_k = key_hnd.shape
    batch_v, num_heads_v, seq_len_v, head_dim_v = value_hnd.shape
    if batch_k != batch or batch_v != batch:
        raise ValueError("Batch size mismatch between Q/K/V")
    if num_heads_v != num_heads_kv or seq_len_v != seq_len_kv or head_dim_v != head_dim_k:
        raise ValueError("K/V shape mismatch")
    if head_dim_k != head_dim:
        raise ValueError("Head dim mismatch between Q and K/V")
    if head_dim not in (64, 128):
        raise ValueError(f"Unsupported head_dim={head_dim}; supported: 64, 128")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_kv")

    q_blocks = (seq_len_q + quant_block_size - 1) // quant_block_size
    kv_blocks = (seq_len_kv + quant_block_size - 1) // quant_block_size
    if tuple(lut.shape) != (batch, num_heads_q, q_blocks, kv_blocks):
        raise ValueError(f"Unexpected lut shape {tuple(lut.shape)}")
    if tuple(valid_block_num.shape) != (batch, num_heads_q, q_blocks):
        raise ValueError(f"Unexpected valid_block_num shape {tuple(valid_block_num.shape)}")

    out = torch.empty((batch, num_heads_q, seq_len_q, head_dim), dtype=value_hnd.dtype, device=value_hnd.device)
    grid = (q_blocks, num_heads_q, batch)
    _sage_sparse_prefill_hnd_kernel[grid](
        query_hnd.contiguous(),
        key_hnd.contiguous(),
        value_hnd.contiguous(),
        out,
        qscale.contiguous(),
        kscale.contiguous(),
        lut.contiguous(),
        valid_block_num.contiguous(),
        *query_hnd.stride(),
        *key_hnd.stride(),
        *value_hnd.stride(),
        *out.stride(),
        qscale.stride(0),
        qscale.stride(1),
        qscale.stride(2),
        kscale.stride(0),
        kscale.stride(1),
        kscale.stride(2),
        lut.stride(0),
        lut.stride(1),
        lut.stride(2),
        lut.stride(3),
        valid_block_num.stride(0),
        valid_block_num.stride(1),
        valid_block_num.stride(2),
        seq_len_q,
        seq_len_kv,
        float(scale),
        NUM_KV_GROUPS=num_heads_q // num_heads_kv,
        HEAD_DIM=head_dim,
        BLOCK_M=quant_block_size,
        BLOCK_N=quant_block_size,
        MAX_K_BLOCKS=kv_blocks,
        USE_BF16=value_hnd.dtype == torch.bfloat16,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=2,
    )
    return out
