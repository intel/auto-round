# Copyright (c) 2024-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-efficient soft cross-entropy for the LFQ (last-block) loss.

The LFQ objective (Eq. 3 of the paper) minimizes the cross-entropy between the
FP model's token distribution ``σ(X·W_FP·W_Head)`` and the quantized model's
distribution ``σ(X·W_q·W_Head)``.  A naive implementation materializes two full
``[batch*seq, vocab]`` float32 tensors (softmax and log_softmax) plus the
autograd graph that produces them -- prohibitive for large-vocab models.

This module reproduces the essence of Unsloth's fused cross-entropy trick, but
for a *soft* target distribution instead of hard integer labels:

* logsumexp reformulation so ``log_softmax`` is never materialized::

      softCE_i = logsumexp(q_i) - Σ_j softmax(fp_i)_j · q_{i,j}        (Σ_j p=1)

* analytic gradient (no autograd graph through softmax)::

      dL/dq_{i,j} = softmax(q_i)_j - softmax(fp_i)_j

* row chunking: only ``chunk_rows × vocab`` float32 elements are live at a time;
  softmax is recomputed in the backward pass (compute-for-memory tradeoff).

Unlike Unsloth's Triton kernel this is pure PyTorch, so it runs on CPU / CUDA /
HPU without a Triton dependency.
"""

from __future__ import annotations

import torch

__all__ = ["soft_cross_entropy", "DEFAULT_SOFT_CE_CHUNK_ELEMS"]

# Target number of (rows × vocab) float32 elements processed per chunk.
# 2**23 ≈ 8.4M elems ≈ 32 MB per temporary fp32 tensor.
DEFAULT_SOFT_CE_CHUNK_ELEMS = 1 << 23


def _rows_per_chunk(vocab_size: int, chunk_elems: int) -> int:
    return max(1, chunk_elems // max(1, vocab_size))


class _SoftCrossEntropy(torch.autograd.Function):
    """Chunked soft cross-entropy with an analytic, memory-light backward."""

    @staticmethod
    def forward(ctx, q_logits, fp_logits, valid_mask, chunk_elems):
        # q_logits / fp_logits: [N, V];  valid_mask: [N] (float, 1.0 keep / 0.0 drop)
        n_rows, vocab_size = q_logits.shape
        rows = _rows_per_chunk(vocab_size, chunk_elems)

        denom = valid_mask.sum().clamp_min(1.0)
        total = q_logits.new_zeros((), dtype=torch.float32)

        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            m = valid_mask[start:end]
            # No host-side ``if m.sum() == 0`` skip here: it would force a
            # device-to-host sync on every chunk (dozens per forward for large
            # vocabs). Invalid rows are already zeroed by multiplying with ``m``,
            # and ``denom`` guards the all-invalid case, so the math stays correct.
            q_c = q_logits[start:end].float()
            lse_q = torch.logsumexp(q_c, dim=-1)
            p_fp = torch.softmax(fp_logits[start:end].float(), dim=-1)
            dot = (p_fp * q_c).sum(dim=-1)
            ce = (lse_q - dot) * m
            total = total + ce.sum()

        loss = total / denom

        ctx.save_for_backward(q_logits, fp_logits, valid_mask, denom)
        ctx.chunk_elems = chunk_elems
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        q_logits, fp_logits, valid_mask, denom = ctx.saved_tensors
        n_rows, vocab_size = q_logits.shape
        rows = _rows_per_chunk(vocab_size, ctx.chunk_elems)

        # scale = dLoss / denom, broadcast per valid row.
        scale = (grad_output / denom).to(torch.float32)
        dq = torch.empty_like(q_logits)

        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            m = valid_mask[start:end].unsqueeze(-1)
            # dL/dq = softmax(q) - softmax(fp), recomputed to avoid storing it.
            grad = torch.softmax(q_logits[start:end].float(), dim=-1)
            grad -= torch.softmax(fp_logits[start:end].float(), dim=-1)
            grad *= m * scale
            dq[start:end] = grad.to(dq.dtype)

        return dq, None, None, None


def soft_cross_entropy(
    q_logits: torch.Tensor,
    fp_logits: torch.Tensor,
    valid_mask: torch.Tensor = None,
    chunk_elems: int = DEFAULT_SOFT_CE_CHUNK_ELEMS,
) -> torch.Tensor:
    """Mean soft cross-entropy ``-Σ_j σ(fp)_j log σ(q)_j`` over valid positions.

    Gradients flow only through ``q_logits``; ``fp_logits`` is treated as a
    detached soft target.

    Args:
        q_logits:   Quantized logits, shape ``[..., vocab]``.
        fp_logits:  FP logits (soft target), same shape as ``q_logits``.
        valid_mask: Optional boolean/0-1 mask over positions (all leading dims
                    except vocab). Positions with 0 are excluded from the loss.
        chunk_elems: Target ``rows × vocab`` float32 elements per chunk.

    Returns:
        Scalar loss tensor.
    """
    vocab_size = q_logits.shape[-1]
    q2d = q_logits.reshape(-1, vocab_size)
    fp2d = fp_logits.reshape(-1, vocab_size).detach()

    if valid_mask is None:
        mask = q2d.new_ones(q2d.shape[0], dtype=torch.float32)
    else:
        mask = valid_mask.reshape(-1).to(device=q2d.device, dtype=torch.float32)

    return _SoftCrossEntropy.apply(q2d, fp2d, mask, chunk_elems)

