# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
Optimized fused dequant + matmul Triton kernel.

Differences vs. ``triton_utils/dequant.py::quant_matmul_248`` (the
"dequant-then-cuBLAS" reference):

  * Fully fused: dequantized weights never spill to HBM.
  * No ``g_idx`` indirection - assumes contiguous groups
    ``group_id = k // group_size``. This removes a gather and lets each
    K-tile broadcast a single (1, N) ``scales`` / ``zeros`` row.
  * Optional symmetric path: when ``sym=True`` the integer zero-point
    is a kernel constexpr (e.g. ``2**(bits-1)``) and ``qzeros`` is not
    loaded at all - saves bandwidth and a few ALU ops per element.

Expected layouts (matching AutoGPTQ / GPTQ-Triton):
  qweight : (K // (32 // bits), N) int32, packed along K
  scales  : (G, N)                fp16/bf16, where G = K // group_size
  qzeros  : (G, N // (32 // bits)) int32   (only used when sym=False)
  input   : (M, K)                fp16/bf16
  output  : (M, N)                same dtype as input
"""

import os as _os

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configs. All configs satisfy BLOCK_SIZE_K <= 128 so they are safe
# for the common group_size in {32, 64, 128}. We enforce
# group_size % BLOCK_SIZE_K == 0 in the launcher, so a K-tile never spans two
# groups and we can hoist the (scales, zeros) load to once per K iteration.
# ---------------------------------------------------------------------------
def _make_configs():
    cfgs = []
    # (BM, BN, BK). BK must divide group_size; we additionally include
    # BK=128 configs which are the most "Marlin-like" (deepest mma per
    # iter). They will be pruned at runtime when group_size < 128.
    tile_shapes = [
        # decode / small-M
        (16, 64, 64),
        (16, 128, 64),
        (16, 128, 128),
        (16, 256, 32),
        (16, 256, 64),
        (32, 64, 64),
        (32, 128, 64),
        (32, 128, 128),
        (32, 256, 32),
        (32, 256, 64),
        # mid-M
        (64, 64, 64),
        (64, 64, 128),
        (64, 128, 64),
        (64, 128, 128),
        (64, 256, 32),
        (64, 256, 64),
        # large-M / compute-bound (Marlin's sweet spot)
        (128, 64, 64),
        (128, 64, 128),
        (128, 128, 32),
        (128, 128, 64),
        (128, 128, 128),
        (128, 256, 32),
        (128, 256, 64),
        (256, 64, 64),
        (256, 64, 128),
        (256, 128, 32),
        (256, 128, 64),
    ]
    for bm, bn, bk in tile_shapes:
        tile = bm * bn
        if tile >= 128 * 128:
            stage_warp = [(3, 8), (4, 8), (3, 4)]
        elif tile >= 64 * 128:
            stage_warp = [(3, 4), (4, 4), (3, 8), (4, 8)]
        else:
            stage_warp = [(3, 4), (4, 4), (5, 4), (3, 8)]
        for ns, nw in stage_warp:
            cfgs.append(
                triton.Config(
                    {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8},
                    num_stages=ns,
                    num_warps=nw,
                )
            )
    return cfgs


def _prune_configs_for_group_size(configs, named_args, **kwargs):
    """Drop configs where BLOCK_SIZE_K does not divide group_size."""
    gs = kwargs.get("group_size") or named_args.get("group_size")
    if gs is None:
        return configs
    return [c for c in configs if (c.kwargs["BLOCK_SIZE_K"] <= gs and gs % c.kwargs["BLOCK_SIZE_K"] == 0)]


@triton.autotune(
    configs=_make_configs(),
    key=["M", "N", "K", "bits", "SYM", "SPLIT_K"],
    prune_configs_by={"early_config_prune": _prune_configs_for_group_size, "perf_model": None, "top_k": None},
)
@triton.jit
def _fused_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scratch_ptr,  # (SPLIT_K, M, N) fp32, used when SPLIT_K > 1
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sg,
    stride_sn,
    stride_zg,
    stride_zn,
    bits: tl.constexpr,
    maxq: tl.constexpr,
    group_size: tl.constexpr,
    SYM: tl.constexpr,
    SYM_ZP: tl.constexpr,
    SPLIT_K: tl.constexpr,  # 1, 2, 4, 8: K-dim parallelism for small M
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused dequant+matmul, Marlin-inspired:

    Optimisations vs. v1:
      1. **Group-major nested K loop**: outer loop over groups, inner loop
         over K-tiles within a group. scales/zeros are loaded EXACTLY once
         per group (not every K-tile), no runtime branch.
      2. **Pre-scaled zero point** (FMA-friendly):
            b_fp = b_int * scale - zp_scaled
         where ``zp_scaled = zp * scale`` is precomputed once per group.
      3. **BLOCK_SIZE_K up to group_size**: with gs=128 we can run BK=128.
      4. **SPLIT-K (Triton 3.x style)**: when SPLIT_K > 1, the grid gains a
         pid_k axis and each CTA processes ``num_groups // SPLIT_K`` groups.
         Per-pid_k partials are written to a (SPLIT_K, M, N) fp32 scratch,
         then a separate ``tl.sum(scratch, dim=0)`` reduce produces the
         final output. This pattern is what production
         ``triton_kernels/matmul.py`` uses on Ampere — atomic-free, robust
         across dtypes, and crucial for M=1 where N-tile parallelism alone
         leaves most SMs idle.
    """
    pack: tl.constexpr = 32 // bits

    # ---- swizzled program ids (grouped along M for L2 reuse) -----------
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)  # 0 .. SPLIT_K-1
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_eff = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m_eff)
    pid_n = (pid % num_pid_in_group) // group_size_m_eff

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # ---- K range owned by this CTA (whole groups only) -----------------
    num_groups = K // group_size
    G_PER_SPLIT = (num_groups + SPLIT_K - 1) // SPLIT_K
    g_start = pid_k * G_PER_SPLIT
    g_end = tl.minimum(g_start + G_PER_SPLIT, num_groups)
    k_start = g_start * group_size

    # advance A and B pointers to the correct K-offset for this split
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
    a_mask = offs_am[:, None] < M

    b_ptrs = b_ptr + (((k_start + offs_k[:, None]) // pack) * stride_bk + offs_bn[None, :] * stride_bn)
    shifter = (offs_k % pack) * bits  # (BK,)

    # Alignment hints help Triton emit 128-bit vector loads on bf16 inputs.
    # Safe because BLOCK_SIZE_M/N/K are powers of 2 and >= 16, and the
    # tensors are contiguous (asserted in the launcher).
    tl.multiple_of(a_ptrs, 16)
    tl.multiple_of(b_ptrs, 16)

    if not SYM:
        zeros_shifter = (offs_bn % pack) * bits  # (BN,)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    K_PER_GROUP: tl.constexpr = group_size // BLOCK_SIZE_K

    # --- bf16 / fp16 dtype housekeeping --------------------------------
    fp_dtype: tl.constexpr = a_ptr.dtype.element_ty
    SYM_ZP_F = tl.full((), SYM_ZP, dtype=fp_dtype)

    for g in range(g_start, g_end):
        # scales/zeros are reused across all K_PER_GROUP inner iterations;
        # ``evict_last`` tells the L2 to keep them resident.
        scales = tl.load(
            scales_ptr + g * stride_sg + offs_bn * stride_sn,
            eviction_policy="evict_last",
        )
        if SYM:
            zp_scaled = SYM_ZP_F * scales
        else:
            qz = tl.load(
                zeros_ptr + g * stride_zg + (offs_bn // pack) * stride_zn,
                eviction_policy="evict_last",
            )
            zeros_int = (qz >> zeros_shifter) & maxq
            zp_scaled = zeros_int.to(fp_dtype) * scales

        for ki in tl.static_range(K_PER_GROUP):
            # ``evict_first`` for the streamed input/weight loads — they are
            # consumed once per tile and shouldn't pollute L2.
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first")
            b_packed = tl.load(b_ptrs, eviction_policy="evict_first")
            b_int = (b_packed >> shifter[:, None]) & maxq
            b_fp = b_int.to(fp_dtype) * scales[None, :] - zp_scaled[None, :]

            # 3-arg form (production tutorial style): tl.dot(a, b, acc).
            # Lets Triton fuse the accumulate directly into the mma
            # instruction rather than emitting a separate add.
            accumulator = tl.dot(a, b_fp, accumulator)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // pack) * stride_bk

    # --- Output ---------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if SPLIT_K == 1:
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        # scratch layout (SPLIT_K, M, N), contiguous along N
        scr_off = pid_k * M * N + offs_cm[:, None] * N + offs_cn[None, :]
        s_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(scratch_ptr + scr_off, accumulator, mask=s_mask)


# ---------------------------------------------------------------------------
# Dedicated GEMV kernel for M=1 (decode).
#
# Why a separate kernel:
#   The GEMM path uses ``tl.dot`` which on Ampere requires BLOCK_SIZE_M >= 16.
#   With M=1, 15/16 of the mma work is masked input padding -> wasted issue
#   slots. Worse, the kernel only spawns ~N/BN CTAs (~30 for typical Llama
#   shapes) which leaves >70% of A100's 108 SMs idle. cuBLAS GEMV uses
#   split-K + warp-shuffle reduction; we mirror that here.
#
# Strategy:
#   * Grid = (cdiv(N, BLOCK_N), SPLIT_K). Each CTA owns a (1, BN) output
#     stripe and a 1/SPLIT_K slice of the K dimension.
#   * No tl.dot: each iter computes ``acc += sum_k(x[k] * b_fp[k, :])``
#     via element-wise mul + ``tl.sum``. ``x`` is just (BK,) fp_dtype so
#     no padding waste.
#   * Output: when SPLIT_K > 1, write fp32 partial sums to a (SPLIT_K, N)
#     scratch and reduce in a final torch.sum (avoids bf16 atomics, which
#     are slow / not native on A100).
# ---------------------------------------------------------------------------
def _make_gemv_configs():
    cfgs = []
    for bn in (64, 128, 256):
        for bk in (32, 64, 128):
            for nw in (4, 8):
                for ns in (2, 3, 4):
                    cfgs.append(
                        triton.Config(
                            {"BLOCK_N": bn, "BLOCK_K": bk},
                            num_stages=ns,
                            num_warps=nw,
                        )
                    )
    return cfgs


def _prune_gemv_configs(configs, named_args, **kwargs):
    gs = kwargs.get("group_size") or named_args.get("group_size")
    if gs is None:
        return configs
    return [c for c in configs if c.kwargs["BLOCK_K"] <= gs and gs % c.kwargs["BLOCK_K"] == 0]


@triton.autotune(
    configs=_make_gemv_configs(),
    key=["M", "N", "K", "bits", "SYM", "SPLIT_K", "BLOCK_M"],
    prune_configs_by={"early_config_prune": _prune_gemv_configs, "perf_model": None, "top_k": None},
)
@triton.jit
def _fused_gemv_kernel(
    x_ptr,  # (M, K) fp_dtype
    qw_ptr,  # (K/pack, N) i32
    scales_ptr,  # (G, N) fp_dtype
    qz_ptr,  # (G, N/pack) i32  (unused when SYM)
    out_ptr,  # (M, N) fp_dtype  (only used when SPLIT_K == 1)
    scratch_ptr,  # (SPLIT_K, M, N) fp32 (only used when SPLIT_K > 1)
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_qk,
    stride_qn,
    stride_sg,
    stride_sn,
    stride_zg,
    stride_zn,
    stride_om,
    stride_on,
    bits: tl.constexpr,
    maxq: tl.constexpr,
    group_size: tl.constexpr,
    SYM: tl.constexpr,
    SYM_ZP: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,  # 1, 2, 4, or 8 (no tl.dot path)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Small-batch GEMV-style fused dequant+matmul.

    Used for ``M < 16`` where ``tl.dot`` would waste (16-M)/16 of mma work
    on Ampere (BM=16 is the minimum mma tile). Here we replace the mma
    with elementwise mul + ``tl.sum`` reduction along K, paying ALU cycles
    instead of wasted tensor-core cycles.

    For BLOCK_M >= 16 use the GEMM kernel instead.
    """
    pack: tl.constexpr = 32 // bits
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_mask = offs_n < N
    m_mask = offs_m < M

    num_groups = K // group_size
    G_PER_SPLIT_CEIL = (num_groups + SPLIT_K - 1) // SPLIT_K
    g_start = pid_k * G_PER_SPLIT_CEIL
    g_end = tl.minimum(g_start + G_PER_SPLIT_CEIL, num_groups)

    fp_dtype: tl.constexpr = scales_ptr.dtype.element_ty
    SYM_ZP_F = tl.full((), SYM_ZP, dtype=fp_dtype)

    K_PER_GROUP: tl.constexpr = group_size // BLOCK_K

    # acc shape (BLOCK_M, BLOCK_N), fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if not SYM:
        zeros_shifter = (offs_n % pack) * bits

    for g in range(g_start, g_end):
        scales = tl.load(scales_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0)
        if SYM:
            zp_scaled = SYM_ZP_F * scales
        else:
            qz = tl.load(qz_ptr + g * stride_zg + (offs_n // pack) * stride_zn, mask=n_mask, other=0)
            zeros_int = (qz >> zeros_shifter) & maxq
            zp_scaled = zeros_int.to(fp_dtype) * scales

        for ki in tl.static_range(K_PER_GROUP):
            k_offs = g * group_size + ki * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            # x: (BM, BK) fp_dtype
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            qrow = k_offs // pack
            shifter = (k_offs % pack) * bits
            qw = tl.load(
                qw_ptr + qrow[:, None] * stride_qk + offs_n[None, :] * stride_qn,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
            b_int = (qw >> shifter[:, None]) & maxq

            # Dequant in fp_dtype (bf16/fp16) - the fp16/bf16 ALU has 2x
            # throughput vs fp32 on Ampere, and the mul step is the bulk
            # of the dequant cost. Only the final reduction goes to fp32
            # for stable accumulation across K-tiles.
            b_fp = b_int.to(fp_dtype) * scales[None, :] - zp_scaled[None, :]

            if BLOCK_M == 1:
                # Specialised 2D path: avoids the 3D broadcast intermediate
                # which Triton struggles to fuse fully when M=1.
                # acc[0, n] += sum_k x[0, k] * b_fp[k, n]
                x_row = tl.reshape(x, (BLOCK_K,))  # (BK,)
                prod = (x_row[:, None] * b_fp).to(tl.float32)  # (BK, BN)
                acc_row = tl.sum(prod, axis=0)  # (BN,)
                acc += tl.reshape(acc_row, (1, BLOCK_N))
            else:
                # M ∈ {2, 4, 8}: 3D broadcast then reduce K. Triton lowers
                # this to a sequence of per-row inner products which is
                # still no-tl.dot (avoids BM=16 padding waste of mma path).
                prod = (x[:, :, None] * b_fp[None, :, :]).to(tl.float32)
                acc += tl.sum(prod, axis=1)

    # --- Output ---------------------------------------------------------
    if SPLIT_K == 1:
        out_off = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptr + out_off, acc.to(fp_dtype), mask=m_mask[:, None] & n_mask[None, :])
    else:
        # scratch: (SPLIT_K, M, N) fp32, contiguous M-major within split
        scr_off = pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(scratch_ptr + scr_off, acc, mask=m_mask[:, None] & n_mask[None, :])


def _pick_split_k(K: int, N: int, M: int = 1, block_n_guess: int = 128) -> int:
    """Choose SPLIT_K conservatively.

    Aggressive split-K hurts when:
      * K is small: each split-CTA does too few iterations, so launch +
        scratch-reduce overhead dominates the actual compute.
      * N is large: enough N-tiles already saturate SMs without splitting.

    Empirical (A100, Llama shapes):
      K=4096, N=4096 -> SPLIT_K=2  (32 N-tiles, want ~64 CTAs)
      K=11008,N=4096 -> SPLIT_K=4  (more K-work to amortise reduce)
      K=4096, N=11008-> SPLIT_K=1  (already 86 N-tiles)
    """
    n_tiles = max(1, (N + block_n_guess - 1) // block_n_guess)

    # Each split-CTA should process at least ~2k K-elements to amortise
    # the per-CTA fixed overhead.
    min_k_per_split = 2048
    max_split_by_work = max(1, K // min_k_per_split)

    # Aim for ~1 wave of CTAs across A100's 108 SMs (don't over-split).
    target_ctas = 108 if M == 1 else 64
    desired = max(1, target_ctas // n_tiles)

    sk = min(desired, max_split_by_work)
    for cand in (8, 4, 2, 1):
        if sk >= cand:
            return cand
    return 1


def _next_pow2_block_m(m: int) -> int:
    """Pick BLOCK_M ∈ {1, 2, 4, 8} as the smallest pow2 >= m."""
    if m <= 1:
        return 1
    if m <= 2:
        return 2
    if m <= 4:
        return 4
    return 8  # caller must guarantee m <= 8


SMALL_M_THRESHOLD = 16  # below this, use GEMV-style; >=16 use tl.dot GEMM


def fused_quant_gemv(
    input: torch.Tensor,  # (..., K)
    qweight: torch.Tensor,  # (K/pack, N)
    scales: torch.Tensor,  # (G, N)
    qzeros: torch.Tensor = None,  # (G, N/pack) or None for sym
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    sym_zp: int = None,
    maxq: int = None,
    split_k: int = None,
) -> torch.Tensor:
    """Fused dequant + small-batch matmul (M < SMALL_M_THRESHOLD).

    Internally pads M to the next power of 2 in {1, 2, 4, 8} as ``BLOCK_M``
    and runs a no-tl.dot kernel. Falls back to ``fused_quant_matmul`` when
    M >= SMALL_M_THRESHOLD.
    """
    pack = 32 // bits
    K = qweight.shape[0] * pack
    N = qweight.shape[1]
    if maxq is None:
        maxq = (1 << bits) - 1
    if sym and sym_zp is None:
        sym_zp = 1 << (bits - 1)

    orig_shape = input.shape
    M = 1
    for d in orig_shape[:-1]:
        M *= d

    if M >= SMALL_M_THRESHOLD:
        # Out of the GEMV regime; defer to tl.dot GEMM.
        return fused_quant_matmul(
            input,
            qweight,
            scales,
            qzeros=qzeros,
            bits=bits,
            group_size=group_size,
            sym=sym,
            sym_zp=sym_zp,
            maxq=maxq,
        )

    x = input.reshape(M, K).contiguous()

    if split_k is None:
        split_k = _pick_split_k(K, N, M=M)
    block_m = _next_pow2_block_m(M)

    # qz pointer
    if not sym:
        assert qzeros is not None
        z_ptr = qzeros
        sz_g, sz_n = qzeros.stride(0), qzeros.stride(1)
    else:
        z_ptr = scales  # dummy
        sz_g, sz_n = 0, 0

    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    if split_k == 1:
        scratch = out  # placeholder, not dereferenced
    else:
        scratch = torch.empty((split_k, M, N), device=input.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        split_k,
    )
    _fused_gemv_kernel[grid](
        x,
        qweight,
        scales,
        z_ptr,
        out,
        scratch,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        sz_g,
        sz_n,
        out.stride(0),
        out.stride(1),
        bits=bits,
        maxq=maxq,
        group_size=group_size,
        SYM=sym,
        SYM_ZP=int(sym_zp) if sym else 0,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
    )

    if split_k > 1:
        out = scratch.sum(dim=0).to(input.dtype)

    return out.reshape(*orig_shape[:-1], N)


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fused split-K reduce: (SPLIT_K, M, N) fp32 -> (M, N) fp_dtype
#
# Replaces ``scratch.sum(0).to(c.dtype)`` (which is two launches: torch
# reduction + cast). One Triton launch instead. Trivial workload but at
# decode latency (~50us total) every kernel launch counts (~5us each on
# A100).
# ---------------------------------------------------------------------------
@triton.jit
def _split_k_reduce_kernel(
    scratch_ptr,  # (SPLIT_K, M, N) fp32
    out_ptr,  # (M, N) fp_dtype
    M,
    N,
    stride_om,
    stride_on,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    mn_mask = m_mask[:, None] & n_mask[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # SPLIT_K is constexpr -> compiler fully unrolls
    for s in tl.static_range(SPLIT_K):
        ptr = scratch_ptr + s * M * N + offs_m[:, None] * N + offs_n[None, :]
        acc += tl.load(ptr, mask=mn_mask, other=0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mn_mask)


def _split_k_reduce(scratch: torch.Tensor, out: torch.Tensor, split_k: int) -> None:
    """In-place: out = scratch.sum(dim=0).to(out.dtype)."""
    M, N = out.shape
    # Tile sizes are tiny here; M=1..32, N=4k..32k. Use big BLOCK_N for
    # bandwidth, small BLOCK_M so M=1 case doesn't waste lanes.
    BLOCK_M = 1 if M == 1 else min(16, triton.next_power_of_2(M))
    BLOCK_N = 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _split_k_reduce_kernel[grid](
        scratch,
        out,
        M,
        N,
        out.stride(0),
        out.stride(1),
        SPLIT_K=split_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


def _pick_gemm_split_k(M: int, N: int, K: int, group_size: int, num_sms: int = 108) -> int:
    """Choose SPLIT_K for the GEMM kernel.

    Same intuition as production triton_kernels matmul: aim for ~1 wave of
    CTAs across SMs. With small M (decode), the M-axis only contributes 1
    M-tile; without split-K we'd only spawn ``cdiv(N, BN)`` CTAs (often 30,
    leaving 70% of A100's SMs idle). With split-K we get
    ``SPLIT_K * cdiv(N, BN)`` CTAs, properly filling the device.

    The cost is one extra ``scratch.sum(0)`` reduction over (SPLIT_K, M, N)
    fp32 — small for typical decode shapes.
    """
    # Heuristics:
    #   * M >= 64: GEMM already produces enough CTAs, split-K just adds reduce overhead
    #   * Need at least ~2k K-elements per split CTA to amortise launch + reduce
    if M >= 64:
        return 1
    # use a conservative BN guess (autotune may pick differently but this
    # only affects the heuristic, not correctness)
    bn_guess = 128
    n_tiles = max(1, (N + bn_guess - 1) // bn_guess)
    m_tiles = max(1, (M + 15) // 16)  # min BM=16 for tl.dot
    base_ctas = m_tiles * n_tiles

    desired = max(1, num_sms // base_ctas)
    max_split_by_groups = max(1, K // (4 * group_size))  # at least 4 groups per split
    sk = min(desired, max_split_by_groups, 8)
    for cand in (8, 4, 2, 1):
        if sk >= cand:
            return cand
    return 1


def fused_quant_matmul(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor = None,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    sym_zp: int = None,
    maxq: int = None,
    split_k: int = None,
) -> torch.Tensor:
    """Fused int{2,4,8} dequant + matmul.

    Parameters
    ----------
    input    : (..., K) fp16/bf16
    qweight  : (K // pack, N) int32
    scales   : (G, N) same dtype as input, G = K // group_size
    qzeros   : (G, N // pack) int32, ignored when sym=True
    sym      : if True, use a constant integer zero-point (sym_zp).
    sym_zp   : the integer zero-point used when sym=True.
               Defaults to 2**(bits-1) (i.e. mid-point).
    split_k  : K-axis parallelism. Auto-chosen by ``_pick_gemm_split_k`` when
               None. Set to 1 to disable. Override via env
               ``AR_TRITON_GEMM_SPLIT_K``.
    """
    assert input.is_contiguous() or input.stride(-1) == 1
    assert qweight.is_contiguous()
    assert bits in (2, 4, 8)

    pack_factor = 32 // bits
    K = qweight.shape[0] * pack_factor
    N = qweight.shape[1]

    orig_shape = input.shape
    a = input.reshape(-1, K)
    M = a.shape[0]

    assert scales.shape[0] * group_size == K, f"scales rows {scales.shape[0]} * group_size {group_size} != K {K}"
    assert scales.shape[1] == N
    assert group_size % 16 == 0, "group_size must be multiple of 16"

    if maxq is None:
        maxq = (1 << bits) - 1
    if sym and sym_zp is None:
        sym_zp = 1 << (bits - 1)

    if not sym:
        assert qzeros is not None
        assert qzeros.shape == (scales.shape[0], N // pack_factor)
        z_ptr = qzeros
        sz_g, sz_n = qzeros.stride(0), qzeros.stride(1)
    else:
        z_ptr = scales
        sz_g, sz_n = 0, 0

    # Pick SPLIT_K: env override > arg > heuristic
    if split_k is None:
        env_sk = _os.environ.get("AR_TRITON_GEMM_SPLIT_K")
        if env_sk is not None:
            split_k = int(env_sk)
        else:
            split_k = _pick_gemm_split_k(M, N, K, group_size)

    c = torch.empty((M, N), device=input.device, dtype=input.dtype)

    if split_k > 1:
        # scratch (SPLIT_K, M, N) fp32 — kernel writes partials, then reduce.
        scratch = torch.empty((split_k, M, N), device=input.device, dtype=torch.float32)
    else:
        # 1-element placeholder; kernel only stores to scratch when SPLIT_K>1.
        scratch = torch.empty(1, device=input.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k,
    )

    _fused_matmul_kernel[grid](
        a,
        qweight,
        c,
        scratch,
        scales,
        z_ptr,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        c.stride(0),
        c.stride(1),
        scales.stride(0),
        scales.stride(1),
        sz_g,
        sz_n,
        bits=bits,
        maxq=maxq,
        group_size=group_size,
        SYM=sym,
        SYM_ZP=int(sym_zp) if sym else 0,
        SPLIT_K=split_k,
    )

    if split_k > 1:
        # Fused Triton reduce: 1 launch (sum + cast) instead of torch.sum + cast.
        _split_k_reduce(scratch, c, split_k)

    return c.reshape(*orig_shape[:-1], N)


# ---------------------------------------------------------------------------
# Optimized 2D-tiled dequant-only kernel (for the prefill path).
#
# The reference dequant_kernel_248 in dequant.py is 1D-flattened: each output
# element issues 4 loads (g_idx, scales, qzeros, qweight) to write 1 fp16
# value (2 bytes) -> ~6.5x extra HBM traffic. Optimisations here:
#   * 2D tile (BK, BN): one CTA produces a (BK, BN) tile of the output.
#   * scales/zeros loaded ONCE per (group, BN) and broadcast over BK rows.
#   * No g_idx (contiguous group: g = k // group_size).
#   * SYM constexpr path skips qzeros entirely.
#   * Output dtype = input dtype (fp16/bf16) -> downstream cuBLAS GEMM
#     in bf16 keeps the same fast-math path on A100/H100.
# ---------------------------------------------------------------------------
def _make_dequant_configs():
    cfgs = []
    for bk, bn in [
        (32, 128),
        (32, 256),
        (64, 128),
        (64, 256),
        (128, 64),
        (128, 128),
        (128, 256),
    ]:
        for nw in (4, 8):
            for ns in (2, 3, 4):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_K": bk, "BLOCK_N": bn},
                        num_stages=ns,
                        num_warps=nw,
                    )
                )
    return cfgs


@triton.autotune(configs=_make_dequant_configs(), key=["K", "N", "bits", "SYM"])
@triton.jit
def _fast_dequant_kernel(
    qweight_ptr,
    scales_ptr,
    qzeros_ptr,
    out_ptr,
    K,
    N,
    stride_qk,
    stride_qn,
    stride_sg,
    stride_sn,
    stride_zg,
    stride_zn,
    stride_ok,
    stride_on,
    bits: tl.constexpr,
    maxq: tl.constexpr,
    group_size: tl.constexpr,
    SYM: tl.constexpr,
    SYM_ZP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pack: tl.constexpr = 32 // bits
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_start = pid_k * BLOCK_K
    n_start = pid_n * BLOCK_N

    offs_k = k_start + tl.arange(0, BLOCK_K)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    k_mask = offs_k < K
    n_mask = offs_n < N

    # --- load qweight: BK rows from BK/pack i32 words (compiler/L1 dedupes) -
    qrow = offs_k // pack  # (BK,)
    shifter = (offs_k % pack) * bits  # (BK,)
    qw = tl.load(
        qweight_ptr + qrow[:, None] * stride_qk + offs_n[None, :] * stride_qn,
        mask=k_mask[:, None] & n_mask[None, :],
        other=0,
    )
    w_int = (qw >> shifter[:, None]) & maxq  # (BK, BN) int32

    # --- group is constant across the BK rows (BK <= group_size enforced) ---
    g = k_start // group_size

    scales = tl.load(
        scales_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0
    )  # (BN,) fp_dtype (matches output)

    # Dtype housekeeping (see _fused_matmul_kernel comment): keep math in
    # the output dtype, never let an int-constexpr * bf16 get promoted to
    # fp32 inside the inner loop.
    fp_dtype: tl.constexpr = out_ptr.dtype.element_ty
    SYM_ZP_F = tl.full((), SYM_ZP, dtype=fp_dtype)

    if SYM:
        # FMA-style dequant: w = b_int.to(fp) * s - (SYM_ZP * s)
        zp_scaled = SYM_ZP_F * scales  # (BN,) fp_dtype
        w = w_int.to(fp_dtype) * scales[None, :] - zp_scaled[None, :]
    else:
        zeros_shifter = (offs_n % pack) * bits
        qz = tl.load(
            qzeros_ptr + g * stride_zg + (offs_n // pack) * stride_zn,
            mask=n_mask,
            other=0,
        )
        zeros_int = (qz >> zeros_shifter) & maxq  # (BN,) int32
        zp_scaled = zeros_int.to(fp_dtype) * scales  # (BN,) fp_dtype
        w = w_int.to(fp_dtype) * scales[None, :] - zp_scaled[None, :]

    tl.store(
        out_ptr + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on,
        w,
        mask=k_mask[:, None] & n_mask[None, :],
    )


def fast_dequant248(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor = None,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    sym_zp: int = None,
    maxq: int = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Optimized dequant -> (K, N) tensor of dtype `out_dtype`.

    Drop-in replacement for ``dequant248`` but:
      * no g_idx (assumes contiguous groups),
      * supports sym fast-path (no qzeros load),
      * 2D-tiled, scales/zeros broadcast across BK rows.
    """
    pack = 32 // bits
    K = qweight.shape[0] * pack
    N = qweight.shape[1]
    assert scales.shape[0] * group_size == K
    assert scales.shape[1] == N
    if maxq is None:
        maxq = (1 << bits) - 1
    if sym and sym_zp is None:
        sym_zp = 1 << (bits - 1)

    if not sym:
        assert qzeros is not None
        z_ptr = qzeros
        sz_g, sz_n = qzeros.stride(0), qzeros.stride(1)
    else:
        z_ptr = scales  # dummy, not dereferenced
        sz_g, sz_n = 0, 0

    out = torch.empty((K, N), device=qweight.device, dtype=out_dtype)

    grid = lambda META: (
        triton.cdiv(K, META["BLOCK_K"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _fast_dequant_kernel[grid](
        qweight,
        scales,
        z_ptr,
        out,
        K,
        N,
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        sz_g,
        sz_n,
        out.stride(0),
        out.stride(1),
        bits=bits,
        maxq=maxq,
        group_size=group_size,
        SYM=sym,
        SYM_ZP=int(sym_zp) if sym else 0,
    )
    return out


# ---------------------------------------------------------------------------
# Hybrid dispatcher: fused for decode/small-M, fast-dequant+cuBLAS for prefill.
# ---------------------------------------------------------------------------
# Empirical crossover on A100/bf16 (see bench_triton_matmul.py):
#   * M <= 64:    fused kernel wins by 1.05x-2.3x (memory-bound)
#   * M in 128..256: roughly tied; keep fused for HBM-write savings
#   * M >= 512:   cuBLAS GEMM dominates, fused loses 0.6x-0.9x
# The threshold is a single knob, override via env if needed.
_PREFILL_THRESHOLD = int(_os.environ.get("AR_TRITON_PREFILL_M", "256"))


def auto_quant_matmul(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor = None,
    g_idx: torch.Tensor = None,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    sym_zp: int = None,
    maxq: int = None,
    prefill_threshold: int = None,
) -> torch.Tensor:
    """Auto-select fused-Triton vs dequant+cuBLAS based on batch size.

    The two paths have opposite scaling:
      * fused Triton: best when M is small (decode); avoids materialising
        the dequantised weight in HBM.
      * dequant + cuBLAS: best when M is large (prefill); cuBLAS GEMM
        beats Triton GEMM on A100/H100 by 1.5-2x once compute-bound.

    Set ``AR_TRITON_PREFILL_M`` env var to override the threshold.
    """
    if prefill_threshold is None:
        prefill_threshold = _PREFILL_THRESHOLD

    # M = product of all leading dims of `input`
    M = 1
    for d in input.shape[:-1]:
        M *= d

    if M >= prefill_threshold:
        # prefill path: fast 2D-tiled dequant + cuBLAS GEMM.
        # No need to materialise qzeros for sym, no g_idx required.
        w = fast_dequant248(
            qweight,
            scales,
            qzeros=qzeros if not sym else None,
            bits=bits,
            group_size=group_size,
            sym=sym,
            sym_zp=sym_zp,
            maxq=maxq,
            out_dtype=input.dtype,
        )
        orig_shape = input.shape
        return (input.reshape(-1, w.shape[0]) @ w).reshape(*orig_shape[:-1], w.shape[1])

    # decode / small-M path: fused GEMM kernel with SPLIT_K.
    #
    # We tried a hand-rolled GEMV (no tl.dot, with split-K) but it lost to
    # GEMM-with-BM=16-padding on A100. Reason: tl.dot uses tensor cores
    # (312 TF), while tl.sum-based reduction uses CUDA cores (~20 TF) — a
    # 16x throughput gap that wasted-padding savings can't close. Triton
    # on Ampere doesn't expose m8n8k16 mma, so a Triton-only GEMV is
    # fundamentally slower than cuBLAS GEMV.
    #
    # The fix that actually moves the needle is **SPLIT_K inside the GEMM
    # kernel** (see ``fused_quant_matmul``): for M=1 the kernel without
    # split-K spawns only ``cdiv(N, BN)`` CTAs (~30, leaving 70% of
    # A100's SMs idle), and SPLIT_K=4-8 brings that to one full wave.
    # ``fused_quant_gemv`` is retained for experimentation but unused by
    # the default dispatcher.
    return fused_quant_matmul(
        input,
        qweight,
        scales,
        qzeros=qzeros,
        bits=bits,
        group_size=group_size,
        sym=sym,
        sym_zp=sym_zp,
        maxq=maxq,
    )
