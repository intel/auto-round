# Copyright (c) 2026 Intel Corporation
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
"""Fused INT2 dequant + GEMV kernel for RRQ decode.

Why a dedicated kernel: decode feeds one token per sequence, so each forward is
a matrix-vector product that moves the entire weight and does almost no
arithmetic.  The generic path dequantizes the packed INT2 into a bf16 tensor and
then runs a GEMM, which writes and re-reads 8x more bytes than the packed
weights hold (2 bytes/parameter against 0.25).  At M=1 that materialization is
the whole cost.

This kernel never materializes the weight: each program owns a tile of output
features, walks the input features in blocks, unpacks the INT2 nibbles in
registers and accumulates ``x @ W`` directly.  Measured on a 5120x6144 layer it
takes ~0.02 ms against ~2.0 ms for the dequant-then-GEMM path.

Shapes follow the W2A16 packing used across AutoRound (``bits=2``, 16 values per
int32):

    qweight : (in_features // 16, out_features)  int32
    scales  : (num_groups, out_features)         float16
    qzeros  : (num_groups, out_features // 16)   int32
"""

import torch

try:  # Triton ships with PyTorch on Linux, but is not available everywhere.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - platform dependent
    _HAS_TRITON = False


def _triton_backend_available() -> bool:
    """True when a Triton-capable accelerator (CUDA or XPU) is present."""
    if torch.cuda.is_available():
        return True
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


def triton_gemv_available() -> bool:
    """Whether the fused RRQ decode kernel can run on this machine.

    CUDA (NVIDIA) and XPU (Intel) are both supported: the same Triton kernels
    compile and run on both backends (Triton ships with the Intel XPU PyTorch
    build).  Call sites still fall back per layer when a kernel cannot run.
    """
    return _HAS_TRITON and _triton_backend_available()


if _HAS_TRITON:

    @triton.jit
    def _rrq_gemv_kernel(
        X,  # (K,)        activations, fp16/bf16
        QW,  # (K//epf, N) int32 packed weights
        SC,  # (G, N)      fp16 scales
        QZ,  # (G, N//epf)  int32 packed zero points
        OUT,  # (N,)       fp32 accumulation target
        K,
        N,
        G,
        gs: tl.constexpr,  # group size
        epf: tl.constexpr,  # elements packed per int32 (32 // bits)
        maxq: tl.constexpr,
        bits: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        n_qz_col = n // epf
        n_shift = (n % epf) * bits
        n_stride_qz = N // epf

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < K
            mask2d = kmask[:, None] & nmask[None, :]

            # Unpack the weights for this (BLOCK_K, BLOCK_N) tile.  Rows repeat
            # whenever several input features share an int32; the repeats hit L2.
            qw = tl.load(QW + (k // epf)[:, None] * N + n[None, :], mask=mask2d, other=0)
            w_int = (qw >> ((k % epf) * bits)[:, None]) & maxq

            # Zero points are per (group, output feature), packed along N.
            g = k // gs
            qz = tl.load(QZ + g[:, None] * n_stride_qz + n_qz_col[None, :], mask=mask2d, other=0)
            zeros = ((qz >> n_shift[None, :]) & maxq) + 1

            sc = tl.load(SC + g[:, None] * N + n[None, :], mask=mask2d, other=0)

            w = (w_int.to(tl.float32) - zeros.to(tl.float32)) * sc.to(tl.float32)
            x = tl.load(X + k, mask=kmask, other=0).to(tl.float32)

            acc += tl.sum(x[:, None] * w, axis=0)

        tl.store(OUT + n, acc, mask=nmask)

    @triton.jit
    def _rrq_gemm_kernel(
        X,  # (M, K)        activations
        QW,  # (K//epf, N)   int32 packed weights
        SC,  # (G, N)        fp16 scales
        QZ,  # (G, N//epf)    int32 packed zero points
        OUT,  # (M, N)       fp32 accumulation target
        M,
        K,
        N,
        G,
        gs: tl.constexpr,
        epf: tl.constexpr,
        maxq: tl.constexpr,
        bits: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Batched variant: one program covers BLOCK_M rows and BLOCK_N columns.

        Batched decode is ``(batch, 1, hidden)``; running a GEMV per sequence
        would launch ``batch`` kernels per layer and lose the whole benefit to
        launch and Python overhead.  Sharing one dequantized weight tile across
        BLOCK_M rows amortizes the unpacking, so cost grows sub-linearly in M.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mmask = m < M
        nmask = n < N
        n_qz_col = n // epf
        n_shift = (n % epf) * bits
        n_stride_qz = N // epf

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < K
            mask_kn = kmask[:, None] & nmask[None, :]

            qw = tl.load(QW + (k // epf)[:, None] * N + n[None, :], mask=mask_kn, other=0)
            w_int = (qw >> ((k % epf) * bits)[:, None]) & maxq

            g = k // gs
            qz = tl.load(QZ + g[:, None] * n_stride_qz + n_qz_col[None, :], mask=mask_kn, other=0)
            zeros = ((qz >> n_shift[None, :]) & maxq) + 1

            sc = tl.load(SC + g[:, None] * N + n[None, :], mask=mask_kn, other=0)

            w = (w_int.to(tl.float32) - zeros.to(tl.float32)) * sc.to(tl.float32)

            x = tl.load(X + m[:, None] * K + k[None, :], mask=mmask[:, None] & kmask[None, :], other=0)
            acc += tl.dot(x.to(tl.float32), w, input_precision="ieee")

        tl.store(OUT + m[:, None] * N + n[None, :], acc, mask=mmask[:, None] & nmask[None, :])


def rrq_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int = 2,
    group_size: int = 128,
    maxq: int | None = None,
    block_n: int = 64,
    block_k: int = 256,
) -> torch.Tensor:
    """Fused INT2 dequant + GEMV for a single token.

    Args:
        x: Activations of shape ``(1, in_features)`` or ``(in_features,)``.
        qweight: Packed weights, ``(in_features // (32 // bits), out_features)``.
        scales: Per-group scales, ``(num_groups, out_features)``.
        qzeros: Packed zero points, ``(num_groups, out_features // (32 // bits))``.
        bits: Bits per element (2 for RRQ).
        group_size: Quantization group size.
        maxq: Maximum quantized value, ``2**bits - 1`` by default.
        block_n: Output features per program.
        block_k: Input features per inner step.

    The defaults (``block_n=64``, ``block_k=256``) were tuned on a 5120x6144
    layer: wider tiles raise register pressure and fall off sharply (``bn=512``
    is ~7x slower), while ``bk=512`` overflows the per-group scale window.

    Returns:
        Result of shape ``(out_features,)`` in float32.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available on this platform.")

    in_features = qweight.shape[0] * (32 // bits)
    out_features = qweight.shape[1]
    x_flat = x.reshape(-1)
    maxq = (1 << bits) - 1 if maxq is None else maxq

    out = torch.empty((out_features,), dtype=torch.float32, device=qweight.device)
    grid = (triton.cdiv(out_features, block_n),)
    _rrq_gemv_kernel[grid](
        x_flat,
        qweight,
        scales,
        qzeros,
        out,
        in_features,
        out_features,
        scales.shape[0],
        gs=group_size,
        epf=32 // bits,
        maxq=maxq,
        bits=bits,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def rrq_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int = 2,
    group_size: int = 128,
    maxq: int | None = None,
    block_m: int = 8,
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """Fused INT2 dequant + GEMM for a small batch of tokens.

    Use this instead of :func:`rrq_gemv` when more than one sequence is decoding
    at once: the GEMV would launch one kernel per sequence per layer, and at
    400 layers that launch and Python overhead dominates.  One program here
    covers ``BLOCK_M`` rows, so the unpacked weight tile is shared across them
    and the cost grows sub-linearly in the batch size.

    Args:
        x: Activations of shape ``(M, in_features)`` with ``M <= 32``.
        qweight: Packed weights, ``(in_features // (32 // bits), out_features)``.
        scales: Per-group scales, ``(num_groups, out_features)``.
        qzeros: Packed zero points, ``(num_groups, out_features // (32 // bits))``.
        bits: Bits per element (2 for RRQ).
        group_size: Quantization group size.
        maxq: Maximum quantized value, ``2**bits - 1`` by default.
        block_m: Rows per program.
        block_n: Output features per program.
        block_k: Input features per inner step.

    The defaults keep the ``tl.dot`` staging buffers well inside the 99 KB
    shared-memory limit; larger tiles (16x64x128) need ~208 KB and fail with
    ``OutOfResources``.  ``num_stages=1`` avoids the software pipelining that
    would otherwise double the traffic.

    Returns:
        Result of shape ``(M, out_features)`` in float32.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available on this platform.")

    in_features = qweight.shape[0] * (32 // bits)
    out_features = qweight.shape[1]
    x2d = x.reshape(-1, in_features)
    m = x2d.shape[0]
    maxq = (1 << bits) - 1 if maxq is None else maxq

    out = torch.empty((m, out_features), dtype=torch.float32, device=qweight.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(out_features, block_n))
    _rrq_gemm_kernel[grid](
        x2d,
        qweight,
        scales,
        qzeros,
        out,
        m,
        in_features,
        out_features,
        scales.shape[0],
        gs=group_size,
        epf=32 // bits,
        maxq=maxq,
        bits=bits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=1,
    )
    return out
